from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from config.models import BaseModelConfig, GraphNeuralNetworkConfig


@dataclass
class OptimizerConfig:
    # must be the same variable names as the pytorch optimizers
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"model": "_"},
        ]
    )

    # how many samples to take from the dataset
    candidates_sample_size: int = 1000

    # dataset
    dataset_path: str = "datasets"

    # dataloaders
    batch_size: int = 32

    # training
    num_epochs: int = 5

    model: BaseModelConfig = MISSING
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=0.001, weight_decay=0.00005)
    )

    _logging_level: str = "INFO"  # DEBUG


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)

# models
config_store.store(
    group="model",
    name=GraphNeuralNetworkConfig().name,
    node=GraphNeuralNetworkConfig,
)