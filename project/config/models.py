from dataclasses import dataclass
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
    dimensions: list[int] = [128, 128]  # Placeholder values
    dropout_rates: list[float] = [0.1, 0.1]  # Placeholder values


@dataclass
class GraphNeuralNetworkConfig(BaseModelConfig):
    """Configuration for the Graph Neural Network model."""

    name: str = "GraphNeuralNetwork"
    params: GraphNeuralNetworkParams = GraphNeuralNetworkParams()
