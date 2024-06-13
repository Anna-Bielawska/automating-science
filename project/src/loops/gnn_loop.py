import copy
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

import torch
import torch_geometric
from src.loops.loop_registry import register_loop
from src.loops.base_loop import Loop
from src.utils.molecules import LeadCompound, from_lead_compound

logger = logging.getLogger(__name__)


def predict(model, test_loader):
    # evaluation loop
    model.eval()
    preds_batches = []
    with torch.no_grad():
        # for data in tqdm(test_loader):
        for data in test_loader:
            data = data.to("cuda")
            preds = model(data)
            preds_batches.append(preds.cpu().detach().numpy())
    preds = np.concatenate(preds_batches)
    return preds


@register_loop("GNNLoop")
class GNNLoop(Loop):
    """
    Your final implementation of the experimental loop.

    The algorithm you implement in the `propose_candidates` method will be repeated
    several times to iteratively improve your candidates.

    The molecules will be sent to the official lab endpoint with a LIMITED NUMBER OF REQUESTS,
    so use this code wisely and care for the synthesizability of your compounds!
    """

    def __init__(
        self,
        base_dir: Path,
        base_loop: Loop,
        n_warmup_iterations: int = 1,
        user_token=None,
        target="DRD2",
        model=None,
    ):
        self.base_loop = base_loop
        self.n_warmup_iterations = n_warmup_iterations
        self._model = copy.deepcopy(model).to("cuda")
        super().__init__(base_dir, user_token, target)

    def _split(self, X: list[LeadCompound]) -> tuple:
        X_temp, X_test = train_test_split(X, test_size=0.2, random_state=42)
        X_train, X_valid = train_test_split(X_temp, test_size=0.2, random_state=42)
        return X_train, X_valid, X_test

    def _get_dataloader(
        self, data: list[LeadCompound], batch_size: int = 16, shuffle: bool = False
    ) -> torch_geometric.loader.DataLoader:
        data = [from_lead_compound(candidate) for candidate in data]

        loader = torch_geometric.loader.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle
        )

        return loader

    def _train_model(self, candidates: list[LeadCompound], epochs: int = 10):
        candidates = [c for c in candidates if c.activity != -1.0]

        X_train, X_valid, X_test = self._split(candidates)

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_valid)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(
            f"Training set activity mean: {np.mean([c.activity for c in X_train])}"
        )
        logger.info(
            f"Training set activity top_10: {sorted([c.activity for c in X_train], reverse=True)[:10]}"
        )
        logger.info(
            f"Training set average activity top_10: {np.mean(sorted([c.activity for c in X_train], reverse=True)[:10])}"
        )
        logger.info("Proceeding to train GCN")

        train_loader = self._get_dataloader(X_train, batch_size=16, shuffle=True)

        learning_rate = 1e-4
        self._model.train()

        # training loop
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(1, epochs + 1):
            for data in train_loader:
                data = data.to("cuda")
                y = data.y
                self._model.zero_grad()
                preds = self._model(data)
                loss = loss_fn(preds, y.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch}, loss: {loss.item()}")

        return self._model

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
        logger.info(f"Selecting top 10 from {len(candidates)} compounds.")

        test_loader = self._get_dataloader(candidates, batch_size=16, shuffle=False)
        y_pred = predict(self._model, test_loader).flatten()
        # now let's sort the compounds based on their predicted activity
        sorted_compounds = [
            c
            for _, c in sorted(
                zip(y_pred, candidates), reverse=True, key=lambda a: (a[0])
            )
        ]

        return sorted_compounds[:n_select]

    def propose_candidates(self, n_candidates: int) -> list[LeadCompound]:
        if self.n_iterations < self.n_warmup_iterations:
            return self.base_loop.propose_candidates(n_candidates)

        # Load previous results
        previous_results: list[LeadCompound] = self.load()

        # Train the model
        self._train_model(previous_results, epochs=1)

        # Generate candidates
        candidates = self.base_loop.propose_candidates(n_candidates)
        # Gather all candidates
        all_candidates = previous_results + candidates

        # Remove duplicates based on SMILES strings
        smiles_seen = set()
        candidates_unique = []
        for c in all_candidates:
            if c.smiles not in smiles_seen:
                candidates_unique.append(c)
                smiles_seen.add(c.smiles)

        all_candidates = candidates_unique
        # Select the top N candidates from previous results and newly sampled compounds, based on the model's predictions
        top_candidates = self._select_top_N(all_candidates, n_candidates)

        return top_candidates
