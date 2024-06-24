from typing import Callable, Type

from config.loops import BaseLoopConfig
from src.loops.base_loop import BaseLoop

LOOP_REGISTRY: dict[str, BaseLoop] = {}


def register_loop(loop_name: str) -> Callable:
    """
    Decorator function to register a loop class with a given loop name.

    Args:
        loop_name (str): The name of the loop.

    Returns:
        Callable: The wrapper function that registers the loop class.

    """
    def wrapper(cls: Type[BaseLoop]) -> Type[BaseLoop]:
        LOOP_REGISTRY[loop_name] = cls
        return cls

    return wrapper


def create_loop(loop_cfg: BaseLoopConfig, **kwargs) -> BaseLoop:
    """
    Create a loop based on the provided loop configuration.

    Args:
        loop_cfg (BaseLoopConfig): The loop configuration object.
        **kwargs: Additional keyword arguments to be passed to the loop constructor.

    Returns:
        BaseLoop: The created loop object.

    """
    return LOOP_REGISTRY[loop_cfg.name](loop_params=loop_cfg.params, **kwargs)
