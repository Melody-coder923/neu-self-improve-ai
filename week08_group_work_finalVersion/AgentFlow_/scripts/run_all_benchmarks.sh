#!/bin/bash
# Run all 5 benchmarks sequentially
set -e

if [ -z "$MODAL_PLANNER_URL" ]; then
    echo "ERROR: MODAL_PLANNER_URL not set"
    exit 1
fi

echo "=== Running All Benchmarks ==="
echo "Model URL: $MODAL_PLANNER_URL"

cd test
bash bamboogle/run.sh
bash 2wiki/run.sh
bash hotpotqa/run.sh
bash musique/run.sh
bash gaia/run.sh

echo "=== All benchmarks complete ==="
