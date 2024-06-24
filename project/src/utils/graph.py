from typing import Optional

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


def get_global_pooling(pooling: Optional[str]):
    """
    Returns the global pooling function based on the specified pooling method.

    Args:
        pooling (str): The pooling method to use. Can be one of "mean", "max", or "sum".

    Returns:
        Callable: The global pooling function corresponding to the specified method.

    Raises:
        ValueError: If an unknown pooling method is provided.
    """
    if pooling is None:
        return None
    elif pooling == "mean":
        return global_mean_pool
    elif pooling == "max":
        return global_max_pool
    elif pooling == "sum":
        return global_add_pool
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")