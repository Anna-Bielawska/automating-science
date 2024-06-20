from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class BaseEmbeddingParams:
    """Base class for embedding parameters.
    Passed as kwargs to the embedding constructor.
    """

    pass


@dataclass
class BaseEmbeddingConfig:
    """Base class for embedding configuration."""

    name: str = MISSING
    params: BaseEmbeddingParams = MISSING


@dataclass
class BasicEmbeddingParams(BaseEmbeddingParams):
    """Parameters for the Basic Embedding model."""

    with_hydrogen: bool = False
    kekulize: bool = False


@dataclass
class BasicEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for the Basic Embedding model."""

    name: str = "basic_embedding"
    params: BasicEmbeddingParams = field(default_factory=BasicEmbeddingParams)
