import logging
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric as pyg
from src.embeddings.base_embedding import BaseEmbedding
from src.utils.molecules import LeadCompound
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_data_loaders(
    compounds: list[LeadCompound],
    embedding: BaseEmbedding,
    split_ratio: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Get train and validation data loaders.

    Args:
        compounds (list[LeadCompound]): List of lead compounds.
        embedding (BaseEmbedding): Embedding instance.
        split_ratio (float): Train/Validation split ratio.
        batch_size (int): Batch size.

    Returns:
        tuple[DataLoader, DataLoader]: Train and validation data loaders.
    """
    pyg_data = [embedding.from_lead_compounds(compound) for compound in compounds]
 
    random.shuffle(pyg_data)
    X_train, X_valid = (
        pyg_data[int(len(pyg_data) * split_ratio) :],
        pyg_data[: int(len(pyg_data) * split_ratio)],
    )

    train_dl = pyg.loader.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    valid_dl = pyg.loader.DataLoader(X_valid, batch_size=batch_size, shuffle=False)

    return train_dl, valid_dl


def train_epoch(
    module: nn.Module,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: Callable,
    device: torch.device,
) -> float:
    """Train a module for one epoch.

    Args:
        model (nn.Module): A PyTorch module.
        train_dl (DataLoader): Dataloader for the train data.
        optimizer (optim.Optimizer): Optimizer instance.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): Pytorch device.

    Returns:
        float: Average loss for the epoch.
    """

    module.train()
    train_loss = 0

    total_samples = len(train_dl.dataset)  # type: ignore
    processed_samples = 0
    log_batch_iter = max(1, len(train_dl) // 10)

    for iteration, data in enumerate(train_dl):
        data = data.to(device)
        labels = data.y

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute prediction error
        pred = module(data).squeeze()
        loss = loss_function(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        processed_samples += len(data.x)
        if iteration % log_batch_iter == 0:
            logger.debug(
                f"Processed {processed_samples}/{total_samples} samples, loss: {loss.item():.6f}"
            )

    return train_loss / len(train_dl)


def validate_epoch(
    module: nn.Module,
    valid_dl: DataLoader,
    loss_function: Callable,
    device: torch.DeviceObjType,
) -> float:
    """Validate the model on given data.

    Args:
        module (nn.Module): A PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        device (torch.DeviceObjType): Pytorch device.

    Returns:
        float: Average loss for the epoch.
    """

    module.eval()
    valid_loss = 0

    total_samples = len(valid_dl.dataset)  # type: ignore
    processed_samples = 0
    log_batch_iter = max(1, len(valid_dl) // 10)

    with torch.no_grad():
        for iteration, data in enumerate(valid_dl):
            data = data.to(device)
            labels = data.y

            # Compute prediction error
            pred = module(data).squeeze()
            loss = loss_function(pred, labels)

            valid_loss += loss.item()
            processed_samples += len(data.x)
            if iteration % log_batch_iter == 0:
                logger.debug(
                    f"Processed {processed_samples}/{total_samples} samples, loss: {loss.item():.6f}"
                )

    return valid_loss / len(valid_dl)


class EarlyStopper:
    """Early stopping class.
    Monitors a given metric and stops training if it does not improve after a given patience.
    Patience is the number of epochs to wait for improvement before stopping.
    """

    def __init__(self, enabled: bool, patience: int, min_delta: float):
        """Initialize EarlyStopper.

        Args:
            enabled (bool): Whether to enable early stopping.
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled

        self.counter = 0
        self.best_metric_value = float("inf")

    def check_stop(self, metric_value: float) -> bool:
        """Check if training should be stopped.

        Args:
            metric_value (float): metric on which to check for improvement.
        Returns:
            bool: Boolean indicating whether training should be stopped.
        """
        if self.best_metric_value - metric_value > self.min_delta:
            self.best_metric_value = metric_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopper"""
        self.counter = 0
        self.best_metric_value = float("inf")
