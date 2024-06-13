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
class GNNLoopParams(BaseLoopParams):
    """Parameters for the GNN loop."""

    n_warmup_iterations: int = 3


@dataclass
class GNNLoopConfig(BaseLoopConfig):
    """Configuration for the GNN loop."""

    name: str = "gnn_loop"
    params: GNNLoopParams = field(default_factory=GNNLoopParams)
