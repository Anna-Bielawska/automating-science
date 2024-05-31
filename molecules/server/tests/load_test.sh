#!/bin/bash
# Script to be ru for load testing the server.
set -e

for i in `seq 1 49`; do
  python server/tests/load_test.py $i &
done