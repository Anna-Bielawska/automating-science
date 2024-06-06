from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


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
        ]
    )

    # dataset
    dataset_path: str = "datasets"

    # dataloaders
    batch_size: int = 32

    # training
    num_epochs: int = 5

    # model
    hidden_dim: int = 64

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=0.001, weight_decay=0.00005)
    )

    _logging_level: str = "INFO"  # DEBUG


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
