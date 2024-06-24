from typing import Type, Callable
from src.embeddings.base_embedding import BaseEmbedding

EMBEDDING_REGISTRY: dict[str, BaseEmbedding] = {}


def register_embedding(embedding_name: str) -> Callable:
    def wrapper(cls: Type[BaseEmbedding]) -> Type[BaseEmbedding]:
        EMBEDDING_REGISTRY[embedding_name] = cls
        return cls

    return wrapper


def create_embedding(embedding_name: str, **kwargs) -> BaseEmbedding:
    return EMBEDDING_REGISTRY[embedding_name](**kwargs)
