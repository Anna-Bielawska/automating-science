from typing import Callable, Type

import torch
from config.models import BaseModelConfig

MODEL_REGISTRY: dict[str, torch.nn.Module] = {}


def register_model(name: str) -> Callable:
    """
    Decorator function to register a model class with a given name.

    Args:
        name (str): The name to register the model with.

    Returns:
        Callable: The wrapper function that registers the model class.
    """
    def wrapper(cls: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def create_model(model_cfg: BaseModelConfig, **kwargs) -> torch.nn.Module:
    """
    Create a model based on the given model configuration.

    Args:
        model_cfg (BaseModelConfig): The model configuration object.
        **kwargs: Additional keyword arguments to be passed to the model constructor.

    Returns:
        torch.nn.Module: The created model.

    """
    return MODEL_REGISTRY[model_cfg.name](params=model_cfg.params, **kwargs)
