from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class BaseLoopParams:
    """Base class for loop parameters.
    Passed as kwargs to the loop constructor.
    """

    pass


@dataclass
class BaseLoopConfig:
    """Base class for loop configuration."""

    name: str = MISSING
    params: BaseLoopParams = MISSING


@dataclass
class MutateLoopParams(BaseLoopParams):
    """Parameters for the Mutate loop."""

    n_warmup_iterations: int = 3
    mutate_top_k: int = 10


@dataclass
class GNNLoopParams(BaseLoopParams):
    """Parameters for the GNN loop."""

    n_warmup_iterations: int = 3
    compounds_multiplier: int = 3


@dataclass
class GNNLoopConfig(BaseLoopConfig):
    """Configuration for the GNN loop."""

    name: str = "gnn_loop"
    params: GNNLoopParams = field(default_factory=GNNLoopParams)
