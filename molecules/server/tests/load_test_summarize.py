"""
Summarizes load test (server/tests/load_test.sh)

Assumes load test runs id 1.. id 49 and saves results in tmp1 .. tmp49

Run as:
python server/tests/load_test_summarize.py
"""
import numpy as np
import os

files = ['tmp' + str(i) for i in range(1, 50)]

for f in files:
    if not os.path.exists(f + "/time.txt"):
        raise RuntimeError("Failed load test: didn't compute " + f)

print([open(f + "/time.txt").read() for f in files])

times = np.array([eval(open(f + "/time.txt").read()) for f in files])

print("{0:.2f}s +- {1:.2f}s".format(np.mean(times), np.std(times)))