import logging

import numpy as np
from pathlib import Path

import torch
import torch_geometric as pyg
from config.loops import GNNLoopConfig, GNNLoopParams
from config.main_config import TrainConfig
from src.loops.loop_registry import register_loop
from src.loops.base_loop import BaseLoop
from src.utils.molecules import LeadCompound, from_lead_compound
from src.utils.training import (
    get_data_loaders,
    train_epoch,
    validate_epoch,
    EarlyStopper,
)

logger = logging.getLogger(__name__)


@register_loop(GNNLoopConfig().name)
class GNNLoop(BaseLoop):
    """
    Your final implementation of the experimental loop.

    The algorithm you implement in the `propose_candidates` method will be repeated
    several times to iteratively improve your candidates.

    The molecules will be sent to the official lab endpoint with a LIMITED NUMBER OF REQUESTS,
    so use this code wisely and care for the synthesizability of your compounds!
    """

    def __init__(
        self,
        loop_params: GNNLoopParams,
        train_cfg: TrainConfig,
        base_dir: Path,
        base_loop: BaseLoop,
        model: torch.nn.Module,
        device: torch.device = torch.device("cuda"),
        target: str = "GSK3β",
    ):
        super().__init__(loop_params, base_dir, target, train_cfg)
        self.loop_params: GNNLoopParams
        self.base_loop = base_loop
        self.model = model
        self.train_metrics = []
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), **self.training_cfg.optimizer
        )
        self.loss_function = torch.nn.MSELoss()
        self.early_stopping = EarlyStopper(**self.training_cfg.early_stopping)

    def _train_model(self, candidates: list[LeadCompound], epochs: int = 10):
        candidates = [c for c in candidates if c.activity != -1.0]

        train_dl, valid_dl = get_data_loaders(
            candidates,
            self.training_cfg.split_ratio,
            self.training_cfg.batch_size,
            self.training_cfg.dataset_path,
        )

        train_activity = [c.y for c in train_dl.dataset]
        top10_activity = sorted(train_activity, reverse=True)[:10]
        metrics = {
            "train_loss": None,
            "valid_loss": None,
            "activity": {
                "mean": np.mean(train_activity),
                "std": np.std(train_activity),
            },
            "top_10_activity": {
                "mean": np.mean(top10_activity),
                "std": np.std(top10_activity),
            },
            "train_size": len(train_dl.dataset),
            "valid_size": len(valid_dl.dataset),
        }

        logger.info(f"Training model for {epochs} epochs.")
        logger.info(
            f"Train size: {metrics['train_size']}, "
            f"Valid size: {metrics['valid_size']}"
        )
        logger.info(
            f"Activity - mean: {metrics['activity']['mean']:.4f}, "
            f"std: {metrics['activity']['std']:.4f}"
        )
        logger.info(
            f"Top 10 activity - mean: {metrics['top_10_activity']['mean']:.4f}, "
            f"std: {metrics['top_10_activity']['std']:.4f}"
        )

        self.early_stopping.reset()
        best_model = self.model.state_dict()
        best_loss = float("inf")

        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model, train_dl, self.optimizer, self.loss_function, self.device
            )
            valid_loss = validate_epoch(
                self.model, valid_dl, self.loss_function, self.device
            )

            logger.info(
                f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}"
            )

            if self.early_stopping.enabled and self.early_stopping.check_stop(
                valid_loss
            ):
                logger.info("Early stopping triggered.")
                break

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.state_dict()
                metrics["train_loss"] = train_loss
                metrics["valid_loss"] = valid_loss

        self.model.load_state_dict(best_model)

        return metrics

    def _predict(self, candidates: list[LeadCompound]) -> list[float]:
        pyg_data = [
            from_lead_compound(compound, self.training_cfg.dataset_path)
            for compound in candidates
        ]

        test_dl = pyg.loader.DataLoader(
            pyg_data,
            batch_size=self.training_cfg.batch_size,
            shuffle=True,
        )

        # evaluation loop
        self.model.eval()
        preds_batches = []
        with torch.no_grad():
            for data in test_dl:
                data = data.to(self.device)
                preds = self.model(data)
                preds_batches.append(preds.cpu().detach().numpy())

        preds = np.concatenate(preds_batches)
        return preds.flatten()

    def _select_top_N(
        self, candidates: list[LeadCompound], n_select: int
    ) -> list[LeadCompound]:
        """Ranks candidates by their predicted activity."""
        # candidates = [c for c in candidates if c.activity != -1]

        if len(candidates) == 0:
            raise ValueError(
                "No previous results to train on (excluded activity = -1). Perhaps your "
                "base loop proposes nonsynthetizable compounds?"
            )
        logger.info(f"Selecting top {n_select} from {len(candidates)} compounds.")

        y_pred = self._predict(candidates)
        sorted_compounds = [
            c
            for _, c in sorted(
                zip(y_pred, candidates), reverse=True, key=lambda a: (a[0])
            )
        ]

        return sorted_compounds[:n_select]

    def propose_candidates(self, n_candidates: int) -> list[LeadCompound]:
        if self.n_iterations < self.loop_params.n_warmup_iterations:
            return self.base_loop.propose_candidates(n_candidates)

        # Load previous results
        previous_results: list[LeadCompound] = self.load()

        # Train the model
        train_metrics = self._train_model(
            previous_results, epochs=self.training_cfg.num_epochs
        )
        self.train_metrics.append(train_metrics)

        # Generate candidates
        logging.info(f"Generating {n_candidates} new candidates.")
        candidates = self.base_loop.propose_candidates(
            n_candidates * self.training_cfg.compounds_multiplier
        )

        # Remove duplicates based on SMILES strings
        smiles_seen = set(previous_results)
        candidates_unique = set()

        for c in candidates:
            if c.smiles not in smiles_seen:
                candidates_unique.add(c)
                smiles_seen.add(c)

        candidates = list(candidates_unique)
        # Select the top N candidates from previous results and newly sampled compounds, based on the model's predictions
        top_candidates = self._select_top_N(candidates, n_candidates)

        return top_candidates
