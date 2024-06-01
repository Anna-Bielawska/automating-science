"""
Implementation of the simplest virtual screening experiments.
"""
import time
import traceback
from typing import List, Dict, Literal, Tuple

import numpy as np
from tdc import Oracle
from src.utils import RdkitCanonicalSmiles

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console

console = Console()
def _get_TDC_oracle(oracle_name):
    return Oracle(name="DRD2")

def _virtual_screen_TDC_worker(oracle_name, compounds_subset, idx):
    try:
        console.log(f"Worker {idx} started.")
        oracle = Oracle(name=oracle_name)
        results = oracle(compounds_subset)
        console.log(f"Worker {idx} completed successfully.")
        return results
    except Exception as e:
        console.log(f"An error occurred in worker {idx}: {e}")
        return [0] * len(compounds_subset)

def virtual_screen_TDC(
    compounds: List[RdkitCanonicalSmiles], oracle_name: str = "DRD2",
    n_jobs: int = 1
) -> List[float]:
    """
    Perform virtual screening in the space for compounds achieving high score according to a selected TDC oracle.
    """

    if n_jobs != 1:
        raise NotImplementedError("Currently n_jobs > 1 is not implemented: it is not reliable enough")

    if n_jobs == 1:
        # Single-process execution (original behavior)
        return _virtual_screen_TDC_worker(oracle_name, compounds, 0)

    try:
        num_compounds = len(compounds)
        chunk_size = max(num_compounds // n_jobs, 1)
        console.log(f"Starting virtual screening with {n_jobs} workers each with chunk {chunk_size}.")

        futures = []

        # min(n_jobs, num_compounds) is important, otherwise i think it spawns too many jobs
        with ProcessPoolExecutor(max_workers=min(n_jobs, num_compounds)) as executor:
            for i in range(0, num_compounds, chunk_size):
                chunk = compounds[i:i + chunk_size]
                console.log(chunk)
                futures.append(executor.submit(_virtual_screen_TDC_worker, oracle_name, chunk, i // chunk_size))

            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        console.log("Virtual screening completed.")
        return results

    except Exception as e:
        console.log(f"An error occurred during multiprocessing: {e}")
        console.log(traceback.format_exc())
        return [0] * num_compounds

def run_virtual_screening(compounds: List[RdkitCanonicalSmiles], experiment: Literal["DRD2"] ="DRD2") -> Tuple[Dict, List]:
    """Runs virtual screening for a list of spaces."""
    if experiment == "GSK3β": # choice for workshop
        fnc = virtual_screen_TDC
    else:
        raise NotImplementedError(f"Unknown experiment f{experiment}")

    scores = fnc(compounds)

    sorted_scores = sorted(scores)

    metrics = {}
    for k in [1, 10, 100]:
        metrics[f"top_{k}"] = np.mean(sorted_scores[-k:])

    return metrics, scores
