"""
Tests for the server.

Before running them, the server database should be restarted.

Run as: python server/tests/test_all.py (don't use py.test as it does not pass env variables easily)
"""
import os
import shutil
from pathlib import Path
from typing import List

import pytest
from requests.exceptions import HTTPError

import numpy as np

from src.al_loop import LeadCompound
from server.app import WORKSHOP_ORACLES
from solutions.task1.random_loop import RandomLoop
from src.server_wrapper import FlaskAppClient

from rich.console import Console

console = Console()

PORT = int(os.environ.get("PORT", "5000"))
BASE_URL = "http://127.0.0.1:" + str(PORT)
BASE_URL = "http://mlinpl23.ngrok.io"
TEST_TOKEN_PREFIX = 'test-' # test-0, test-1, ...

def test_submitting_compounds_to_workshop_oracles():
    """Submits three simple molecules to the server using token test-0, to all workshop oracles."""
    client = FlaskAppClient(base_url=BASE_URL)
    token = TEST_TOKEN_PREFIX + '0'

    # Example for scoring compounds

    for oracle in WORKSHOP_ORACLES:
        compounds = ["CCCCCCCCC", "CCCCCCCC", "CCCCCC=O"]
        response = client.score_compounds_and_update_leaderboard(compounds, oracle, token)
        print(response)
        assert "metrics" in response
        assert "compound_scores" in response
        assert "compound_sas_scores" in response


def _run_random_exploration(protein, token="test-1", steps=10):
    """Simple random exploration of ZINC. Should get above >0.5 score on each oracle."""
    base_dir = Path("tmp")
    shutil.rmtree(base_dir, ignore_errors=True)
    client = FlaskAppClient(base_url=BASE_URL)
    loop = RandomLoop(base_dir=base_dir,
                          user_token=token,
                          target=protein)
    all_result: List[LeadCompound] = []
    budget_per_step = 100
    for step in range(steps):
        console.print(f"[red]Step {step}[/red]")
        candidates = loop.propose_candidates(budget_per_step)
        loop.test_in_lab_and_save(candidates, client=client)
        result: List[LeadCompound] = loop.load(iteration_id=step)
        all_result += result
        all_result_sorted = sorted(all_result, key=lambda x: x.activity, reverse=True)
        metrics = {"top10": np.mean([x.activity for x in all_result_sorted[:10]]),
                        "top10_synth": np.mean([x.synth_score for x in all_result_sorted[:10]])}
        console.log(metrics)

    return metrics


def test_random_exploration_gets_reasonable_score():
    for protein in [ 'GSK3β', 'DRD2_server', 'JNK3']:
        console.log("Testing: " + protein)
        metrics = _run_random_exploration(protein=protein)
        assert metrics['top10'] > 0.1, "Random search should identify reasonable compounds"

def test_leaderboard_ordering_and_user_names():
    _run_random_exploration('DRD2_server', 'test-2', steps=1)
    _run_random_exploration('DRD2_server', 'test-3', steps=1)
    client = FlaskAppClient(base_url=BASE_URL)
    all_results = client.all_results()
    users = [r['user'] for r in all_results]
    print(users)
    assert 'user-2' in users
    assert 'user-3' in users
    all_proteins = ['DRD2', 'JNK3', 'GSK3β']
    sums = [sum([all_results[0]['metrics'][p + "_top_10"]  for p in all_proteins]) for r in all_results]
    assert sums[0] == max(sums), "First result in the leaderboard should be the maximum sum of top10 scores"

def test_call_limits():
    base_dir = Path("tmp")
    token = 'test-10'
    shutil.rmtree(base_dir, ignore_errors=True)
    loop = RandomLoop(base_dir=base_dir,
                      user_token=token,
                      target='DRD2')
    client = FlaskAppClient(base_url=BASE_URL)
    # exhaust limit
    candidates = loop.propose_candidates(1000)
    loop.test_in_lab_and_save(candidates, client=client)
    # run one time more
    candidates = loop.propose_candidates(100)

    with pytest.raises(RuntimeError):
        client.score_compounds_and_update_leaderboard([c.smiles for c in candidates], user_token=token, oracle_name='DRD2')

def test_get_all_scores():
    base_dir = Path("tmp")
    token = 'test-40'
    shutil.rmtree(base_dir, ignore_errors=True)
    loop = RandomLoop(base_dir=base_dir,
                      user_token=token,
                      target='GSK3β_server')
    client = FlaskAppClient(base_url=BASE_URL)
    # exhaust limit
    candidates = loop.propose_candidates(100)
    candidates = loop.test_in_lab_and_save(candidates, client=client)
    # run one time more
    console.log(client.all_scores(token))
    assert len(client.all_scores(token)['compound_sas_scores']['GSK3β']) == len(candidates)
    assert len(client.all_scores(token)['compound_scores']['GSK3β']) == len(candidates)


if __name__ == "__main__":
    test_submitting_compounds_to_workshop_oracles()
    test_random_exploration_gets_reasonable_score()
    test_leaderboard_ordering_and_user_names()
    test_call_limits()
    test_get_all_scores()
    console.log("[green] Tests passed [/green]")