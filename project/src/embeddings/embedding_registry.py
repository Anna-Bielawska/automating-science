from typing import Callable, Type

from config.embeddings import BaseEmbeddingConfig
from src.embeddings.base_embedding import BaseEmbedding

EMBEDDING_REGISTRY: dict[str, BaseEmbedding] = {}


def register_embedding(embedding_name: str) -> Callable:
    """
    Decorator function to register an embedding class with a given name.

    Args:
        embedding_name (str): The name of the embedding.

    Returns:
        Callable: The wrapper function that registers the embedding class.
    """
    def wrapper(cls: Type[BaseEmbedding]) -> Type[BaseEmbedding]:
        EMBEDDING_REGISTRY[embedding_name] = cls
        return cls

    return wrapper


def create_embedding(embedding_cfg: BaseEmbeddingConfig, **kwargs) -> BaseEmbedding:
    """
    Create an instance of BaseEmbedding based on the provided embedding configuration.

    Args:
        embedding_cfg (BaseEmbeddingConfig): The configuration for the embedding.
        **kwargs: Additional keyword arguments to be passed to the embedding constructor.

    Returns:
        BaseEmbedding: An instance of BaseEmbedding.

    """
    return EMBEDDING_REGISTRY[embedding_cfg.name](params=embedding_cfg.params, **kwargs)
