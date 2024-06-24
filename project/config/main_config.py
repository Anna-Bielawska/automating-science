from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from config.models import BaseModelConfig, GraphNeuralNetworkConfig, GraphAttentionNetworkConfig
from config.loops import GNNLoopConfig, BaseLoopConfig
from config.embeddings import BaseEmbeddingConfig, BasicEmbeddingConfig


@dataclass
class OptimizerConfig:
    # must be the same variable names as the pytorch optimizers
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 1
    min_delta: float = 1e-4


@dataclass
class PretrainConfig:
    enabled: bool = False
    compounds_budget: int = 1000
    loop_steps: int = 10

    num_epochs: int = 10
    batch_size: int = 32
    split_ratio: float = 0.2

    save_model: bool = False

    embedding: BaseEmbeddingConfig = field(default_factory=BasicEmbeddingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-3, weight_decay=0.00005)
    )


@dataclass
class TrainConfig:
    compounds_budget: int = 1000
    loop_steps: int = 10
    
    num_epochs: int = 1
    batch_size: int = 8
    split_ratio: float = 0.2

    save_model: bool = False

    loop: BaseLoopConfig = MISSING
    embedding: BaseEmbeddingConfig = field(default_factory=BasicEmbeddingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=5e-4, weight_decay=0.00005)
    )


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"model": "_"},
            {"training.loop": "_"},
        ]
    )
    model: BaseModelConfig = MISSING
    pretraining: PretrainConfig = field(default_factory=PretrainConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42

    _repeat: int = 1
    _logging_level: str = "INFO"  # DEBUG
    _device: str = "cuda"  # cpu


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)

# models
config_store.store(
    group="model",
    name=GraphNeuralNetworkConfig().name,
    node=GraphNeuralNetworkConfig,
)

config_store.store(
    group="model",
    name=GraphAttentionNetworkConfig().name,
    node=GraphAttentionNetworkConfig,
)

# loops
config_store.store(
    group="training.loop",
    name=GNNLoopConfig().name,
    node=GNNLoopConfig,
)
