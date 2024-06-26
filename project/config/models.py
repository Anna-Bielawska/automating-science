from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseModelParams:
    """Base class for model parameters.
    Passed as kwargs to the model constructor.
    """

    pass


@dataclass
class BaseModelConfig:
    """Base class for model configuration."""

    name: str = MISSING
    params: BaseModelParams = MISSING


@dataclass
class GraphNeuralNetworkParams(BaseModelParams):
    """Parameters for the Graph Neural Network model.

    Attributes:
    ----------
    input_dim (int) : Input size for ZINC
    dimensions (list[int]) : Dimensions of the model layers
    dropout_rates (list[float]) : Dropout rates for the model layers
    ----------
    """

    input_dim: int = 34  # Input size for ZINC
    output_dim: int = 1
    dimensions: list[int] = field(default_factory=lambda: [64, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3])
    global_pooling: Optional[str] = "mean"
    concat_global_pooling: Optional[str] = None


@dataclass
class GraphNeuralNetworkConfig(BaseModelConfig):
    """Configuration for the Graph Neural Network model."""

    name: str = "GraphNeuralNetwork"
    params: GraphNeuralNetworkParams = GraphNeuralNetworkParams()


@dataclass
class GraphAttentionNetworkParams(BaseModelParams):
    in_channels: int = 34
    out_channels: int = 1
    dimensions: list[int] = field(default_factory=lambda: [64, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3])
    heads: list[int] = field(default_factory=lambda: [8, 1])  # last attention layer must have 1 head
    global_pooling: Optional[str] = "mean"
    concat_global_pooling: Optional[str] = None


@dataclass
class GraphAttentionNetworkConfig(BaseModelConfig):
    """Configuration for the Graph Attention Network model."""

    name: str = "GraphAttentionNetwork"
    params: GraphAttentionNetworkParams = GraphAttentionNetworkParams()


@dataclass
class GraphIsomorphismNetworkParams(BaseModelParams):
    in_channels: int = 34
    out_channels: int = 1
    trainable_eps: bool = True
    dimensions: list[int] = field(default_factory=lambda: [64, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3])
    global_pooling: Optional[str] = "mean"
    concat_global_pooling: Optional[str] = None


@dataclass
class GraphIsomorphismNetworkConfig(BaseModelConfig):
    """Configuration for the Graph Isomorphism Network model."""

    name: str = "GraphIsomorphismNetwork"
    params: GraphIsomorphismNetworkParams = GraphIsomorphismNetworkParams()


@dataclass
class EdgeConditionedNetworkParams(BaseModelParams):
    in_channels: int = 34
    out_channels: int = 1
    dimensions: list[int] = field(default_factory=lambda: [64, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3])
    global_pooling: Optional[str] = "mean"
    concat_global_pooling: Optional[str] = None


@dataclass
class EdgeConditionedNetworkConfig(BaseModelConfig):
    """Configuration for the Edge Conditioned Network model."""

    name: str = "EdgeConditionedNetwork"
    params: EdgeConditionedNetworkParams = EdgeConditionedNetworkParams()


@dataclass
class GraphResidualNetworkParams(BaseModelParams):
    """Parameters for the Graph Residual Network model.

    Attributes:
    ----------
    input_dim (int) : Input size for ZINC
    dimensions (list[int]) : Dimensions of the model layers
    dropout_rates (list[float]) : Dropout rates for the model layers
    ----------
    """

    input_dim: int = 34
    output_dim: int = 1
    dimensions: list[int] = field(default_factory=lambda: [64, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3])
    global_pooling: Optional[str] = "mean"
    concat_global_pooling: Optional[str] = None


@dataclass
class GraphResidualNetworkConfig(BaseModelConfig):
    """Configuration for the Graph Residual Network model."""

    name: str = "GraphResidualNetwork"
    params: GraphResidualNetworkParams = GraphResidualNetworkParams()