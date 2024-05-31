"""
Script used by load_test.sh to fire a random query to the server

Use as python server/tests/load_test.py 0
"""
import os
import shutil
import time
from pathlib import Path
from src.server_wrapper import FlaskAppClient
from solutions.task1.random_loop import RandomLoop
import sys

PORT = int(os.environ.get("PORT", "8000"))
BASE_URL = "http://127.0.0.1:" + str(PORT)

id = sys.argv[1]

base_dir = Path("tmp" + id)
shutil.rmtree(base_dir, ignore_errors=True)
loop = RandomLoop(base_dir=base_dir,
                  user_token='test-' + id,
                  target='DRD2_server')
client = FlaskAppClient(base_url=BASE_URL)

candidates = loop.propose_candidates(3000)
time_start = time.time()
client.score_compounds_and_update_leaderboard([c.smiles for c in candidates], user_token='test-' + id, oracle_name='DRD2_server')
time_elapsed = time.time() - time_start
with open(base_dir / "time.txt", "w") as f:
    f.write(str(time_elapsed))