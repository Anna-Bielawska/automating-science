import torch
import torch_geometric
from src.models.pna import PNA
from torch_geometric.datasets import ZINC
from datetime import datetime

from config.main_config import MainConfig
from src.utils.training import (
    train_epoch,
    validate_epoch,
    calculate_indegree_histogram,
    MAELoss,
)
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def training_loop(cfg: MainConfig, hydra_output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zinc_train = ZINC(root=cfg.dataset_path, subset=True, split="train")
    zinc_valid = ZINC(root=cfg.dataset_path, subset=True, split="val")

    train_dl = torch_geometric.loader.DataLoader(
        zinc_train,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    valid_dl = torch_geometric.loader.DataLoader(
        zinc_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    indegree_histogram = calculate_indegree_histogram(zinc_train)

    model = PNA(indegree_histogram).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    criterion = MAELoss()

    for epoch in range(cfg.num_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        logger.info(f"Train loss: {train_loss}")

        metrics = validate_epoch(model, valid_dl, criterion, {}, device)

        logger.info(f"Validation error: {metrics['validation_loss']}")
        logger.info(f"Epoch {epoch + 1} completed.")

    torch.save(model.state_dict(), hydra_output_dir / f"model_{datetime.now()}.pth")
