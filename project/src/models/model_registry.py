import torch

from typing import Type, Callable

MODEL_REGISTRY: dict[str, torch.nn.Module] = {}


def register_model(name: str) -> Callable:
    def wrapper(cls: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
        MODEL_REGISTRY[cls.__name__] = cls
        return cls

    return wrapper


def create_model(model_name: str, **kwargs) -> torch.nn.Module:
    return MODEL_REGISTRY[model_name](**kwargs)
