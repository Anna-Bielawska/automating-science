#!/bin/bash
# Script to be run before tests

# set -e TODO: uncomment me when hack is fixed.

#rm results.db.tmp || echo 'not found results.db.tmp'
#mv results.db results.db.tmp

PORT=${PORT:=5000}

curl -X POST -H "Content-Type: application/json" \
     http://127.0.0.1:${PORT}/reset