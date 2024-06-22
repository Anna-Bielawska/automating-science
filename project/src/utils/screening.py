"""
Implementation of the simplest virtual screening experiments.
"""

from typing import List, Dict, Literal, Tuple

import numpy as np
from tdc import Oracle
from src.utils.dataset import RdkitCanonicalSmiles
import logging

logger = logging.getLogger(__name__)


def _virtual_screen_TDC_worker(oracle_name, compounds_subset, idx):
    try:
        logger.debug(f"Worker {idx} started.")
        oracle = Oracle(name=oracle_name)
        results = oracle(compounds_subset)
        logger.debug(f"Worker {idx} completed successfully.")
        return results
    except Exception as e:
        logger.error(f"An error occurred in worker {idx}: {e}")
        return [0] * len(compounds_subset)


def virtual_screen_TDC(
    compounds: List[RdkitCanonicalSmiles], oracle_name: str = "GSK3β", n_jobs: int = 1
) -> List[float]:
    """
    Perform virtual screening in the space for compounds achieving high score according to a selected TDC oracle.
    """

    if n_jobs != 1:
        raise NotImplementedError(
            "Currently n_jobs > 1 is not implemented: it is not reliable enough"
        )

    if n_jobs == 1:
        # Single-process execution (original behavior)
        return _virtual_screen_TDC_worker(oracle_name, compounds, 0)


def run_virtual_screening(
    compounds: List[RdkitCanonicalSmiles], experiment: Literal["GSK3β"] = "GSK3β"
) -> Tuple[Dict, List]:
    """Runs virtual screening for a list of spaces."""
    if experiment == "GSK3β":  # choice for workshop
        fnc = virtual_screen_TDC
    else:
        raise NotImplementedError(f"Unknown experiment f{experiment}")

    scores = fnc(compounds)

    sorted_scores = sorted(scores)

    metrics = {}
    for k in [1, 10, 100]:
        metrics[f"top_{k}"] = np.mean(sorted_scores[-k:])

    return metrics, scores
