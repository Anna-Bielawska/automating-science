import torch
import torch_geometric
from datetime import datetime

from config.main_config import MainConfig
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
