from typing import Type, Callable
from src.loops.base_loop import BaseLoop

LOOP_REGISTRY: dict[str, BaseLoop] = {}


def register_loop(loop_name: str) -> Callable:
    def wrapper(cls: Type[BaseLoop]) -> Type[BaseLoop]:
        LOOP_REGISTRY[loop_name] = cls
        return cls

    return wrapper


def create_loop(loop_name: str, **kwargs) -> BaseLoop:
    return LOOP_REGISTRY[loop_name](**kwargs)
