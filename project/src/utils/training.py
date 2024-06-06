import logging
from typing import Callable, Mapping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.utils import degree

logger = logging.getLogger(__name__)


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
        pred = module(data.x, data.edge_index, data.edge_attr, data.batch).squeeze()
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
    metrics_functions: Mapping[str, Callable],
    device: torch.DeviceObjType,
) -> dict[str, float]:
    """Validate the model on given data.
    Args:
        model (nn.Module): PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        metrics_functions (Mapping[str, Callable]): Dictionary with metric_name : callable pairs.
        enable_autocast (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        device (torch.device, optional): Pytorch device.
    Returns:
        dict[str, float]: Dictionary with average loss and metrics for the validation data.
    """

    module.eval()
    metrics = {name: 0.0 for name in metrics_functions.keys()}
    metrics["validation_loss"] = 0.0

    total_samples = len(valid_dl.dataset)  # type: ignore
    processed_samples = 0
    log_batch_iter = max(1, len(valid_dl) // 10)

    with torch.no_grad():
        for iteration, data in enumerate(valid_dl):
            data = data.to(device)
            labels = data.y

            # Compute prediction error
            pred = module(data.x, data.edge_index, data.edge_attr, data.batch).squeeze()
            loss = loss_function(pred, labels)

            metrics["validation_loss"] += loss.item()
            for name, func in metrics_functions.items():
                metrics[name] += func(pred, labels)

            processed_samples += len(data.x)

            if iteration % log_batch_iter == 0:
                logger.debug(
                    f"Processed {processed_samples}/{total_samples} samples, loss: {loss.item():.6f}"
                )

    for name, _ in metrics.items():
        metrics[name] /= len(valid_dl)

    return metrics


def calculate_indegree_histogram(dataset: torch_geometric.data.Dataset) -> torch.Tensor:
    """Calculate the in-degree histogram of the dataset.

    Args:
        dataset (torch_geometric.data.Dataset): The dataset to calculate the in-degree histogram for.

    Returns:
        torch.Tensor: Tensor with count of nodes with each in-degree.
        Shape: [0, 1, ..., max_indegree + 1]
    """

    # Compute the maximum in-degree in the training data.
    degrees = (
        degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        for data in dataset
    )
    max_indegree = max(int(d.max()) for d in degrees)

    logger.info(f"Max in-degree: {max_indegree}")

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_indegree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    logger.info(f"Degree histogram: {deg}")

    return deg


class MAELoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Maximum Absolute Error Loss

        Args:
            pred (torch.Tensor): Prediction tensor shape (batch_size)
            target (torch.Tensor): Label tensor shape (batch_size)

        Returns:
            torch.Tensor: Loss tensor shape (1)
        """
        return torch.mean(torch.abs(pred - target))
