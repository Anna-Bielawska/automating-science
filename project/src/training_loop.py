import os
import numpy as np
from sympy import Q
import torch
import torch_geometric
from datetime import datetime

from config.main_config import MainConfig
from src.loops.mutate_loop import MutateLoop
from src.loops.gnn_loop import GNNLoop
import src.models.model_registry
from src.utils.training import (
    train_epoch,
    validate_epoch,
    MAELoss,
)
import logging
from pathlib import Path
from src.utils.dataset import SmallZINC
from src.utils.molecules import LeadCompound, from_lead_compound
from sklearn.model_selection import train_test_split
import src.models

logger = logging.getLogger(__name__)


def run(loop, budget=1000, steps=10):
    os.makedirs(loop.base_dir, exist_ok=True)

    if loop.n_iterations > 0:
        raise ValueError(
            f"Already run. Please remove the folder {loop.base_dir} to run again."
        )

    metrics = []
    all_result: list[LeadCompound] = []
    budget_per_step = budget // steps
    assert budget % steps == 0  # for simplicity
    for step in range(steps):
        logger.info(f"Step {step}")

        candidates = loop.propose_candidates(budget_per_step)
        loop.test_in_lab_and_save(candidates)
        result: list[LeadCompound] = loop.load(iteration_id=step)
        all_result += result
        all_result_sorted = sorted(all_result, key=lambda x: x.activity, reverse=True)

        metrics.append(
            {
                "top10": np.mean([x.activity for x in all_result_sorted[:10]]),
                "top10_synth": np.mean([x.synth_score for x in all_result_sorted[:10]]),
            }
        )

    return metrics


def training_loop(cfg: MainConfig, hydra_output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_space = SmallZINC()
    compounds_sample = [
        LeadCompound(smiles=smile, synth_score=None, activity=None)
        for smile in dataset_space.try_sample(cfg.candidates_sample_size)
    ]
    geo_data_sample = [from_lead_compound(compound) for compound in compounds_sample]

    geo_data_sample_train, geo_data_sample_valid = train_test_split(
        geo_data_sample, test_size=0.2
    )

    train_dl = torch_geometric.loader.DataLoader(
        geo_data_sample_train,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    valid_dl = torch_geometric.loader.DataLoader(
        geo_data_sample_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    model = src.models.model_registry.create_model(cfg.model.name, **cfg.model.params)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    criterion = MAELoss()

    for epoch in range(cfg.num_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        logger.info(f"Train loss: {train_loss}")

        metrics = validate_epoch(model, valid_dl, criterion, {}, device)

        logger.info(f"Validation error: {metrics['validation_loss']}")
        logger.info(f"Epoch {epoch + 1} completed.")

    torch.save(model.state_dict(), hydra_output_dir / f"model_{datetime.now()}.pth")

    base_loop2 = MutateLoop(
        base_dir=hydra_output_dir / "mutate_loop",
        n_warmup_iterations=3,
        target="GSK3β",
        initial_dataset=compounds_sample,
    )

    gcnloop = GNNLoop(
        base_dir=hydra_output_dir / "gcn_loop",
        n_warmup_iterations=2,
        base_loop=base_loop2,
        target="GSK3β",
        model=model,
    )

    gcn_ml_metrics = run(gcnloop, budget=1000, steps=10)
    print(gcn_ml_metrics)
