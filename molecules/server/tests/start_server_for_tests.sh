#!/bin/bash
# Script to be run before tests
set -e

ulimit -n 1000000

PORT=${PORT:=5000}

N_JOBS=4 PORT=${PORT} python server/start.py
