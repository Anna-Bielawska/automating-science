import logging
from pathlib import Path

import numpy as np
import src.utils
import src.utils.training
import torch
import torch_geometric as pyg
from config.loops import GNNLoopConfig, GNNLoopParams
from config.main_config import TrainConfig
from src.embeddings.embedding_registry import create_embedding
from src.loops.base_loop import BaseLoop
from src.loops.loop_registry import register_loop
from src.utils.molecules import LeadCompound

logger = logging.getLogger(__name__)


@register_loop(GNNLoopConfig().name)
class GNNLoop(BaseLoop):
    """
    Loop that uses a GNN model to propose new candidates.

    Args:
        loop_params (GNNLoopParams): Loop parameters.
        train_cfg (TrainConfig): Training configuration.
        base_dir (Path): Base directory.
        base_loop (BaseLoop): Base loop.
        model (torch.nn.Module): GNN model.
        device (torch.device): Device to run the model on.
        target (str): Target to optimize.
    """

    def __init__(
        self,
        loop_params: GNNLoopParams,
        train_cfg: TrainConfig,
        base_dir: Path,
        base_loop: BaseLoop,
        model: torch.nn.Module,
        device: torch.device,
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
        self.early_stopping = src.utils.training.EarlyStopper(**self.training_cfg.early_stopping)
        self.embedding = create_embedding(self.training_cfg.embedding)

    def _train_model(self, candidates: list[LeadCompound], epochs: int = 10):
        """
        Trains the model using the provided list of candidates for a specified number of epochs.

        Args:
            candidates (list[LeadCompound]): A list of LeadCompound objects representing the candidates.
            epochs (int, optional): The number of training epochs. Defaults to 10.

        Returns:
            dict: A dictionary containing the training metrics including train_loss, valid_loss, activity mean and std,
                  top 10 activity mean and std, train size, and valid size.
        """
        candidates = [c for c in candidates if c.activity != -1.0]

        train_dl, valid_dl = src.utils.training.get_data_loaders(
            candidates,
            self.embedding,
            self.training_cfg.split_ratio,
            self.training_cfg.batch_size,
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
            train_loss = src.utils.training.train_epoch(
                self.model, train_dl, self.optimizer, self.loss_function, self.device
            )
            valid_loss = src.utils.training.validate_epoch(
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
        """
        Predicts the output for a list of candidate lead compounds.

        Args:
            candidates (list[LeadCompound]): A list of LeadCompound objects representing the candidate compounds.

        Returns:
            list[float]: A list of predicted output values for the candidate compounds.
        """
        pyg_data = [
            self.embedding.from_lead_compounds(compound) for compound in candidates
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
        """Selects the top N compounds based on their predicted activity.

        Args:
            candidates (list[LeadCompound]): The list of candidate compounds.
            n_select (int): The number of compounds to select.

        Returns:
            list[LeadCompound]: The top N compounds based on their predicted activity.
        """
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
        """
        Proposes a list of candidate compounds for the next iteration of the loop.

        Args:
            n_candidates (int): The number of candidate compounds to propose.

        Returns:
            list[LeadCompound]: A list of proposed candidate compounds.
        """
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
        logger.debug(
            f"Generating {n_candidates * self.loop_params.compounds_multiplier} new candidates."
        )
        candidates = self.base_loop.propose_candidates(
            n_candidates * self.loop_params.compounds_multiplier
        )

        # Remove duplicates based on SMILES strings
        smiles_seen = set(previous_results)
        candidates_unique = set()

        for c in candidates:
            if c.smiles not in smiles_seen:
                candidates_unique.add(c)
                smiles_seen.add(c)

        candidates = sorted(list(candidates_unique), key=lambda x: x.smiles)

        # Select the top N candidates from previous results and newly sampled compounds, based on the model's predictions
        top_candidates = self._select_top_N(candidates, n_candidates)

        return top_candidates
