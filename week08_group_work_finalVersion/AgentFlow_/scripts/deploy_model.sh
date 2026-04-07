#!/bin/bash
# Deploy a specific model to Modal and run benchmarks
# Usage: bash scripts/deploy_model.sh Qwen/Qwen3.5-0.8B true

MODEL_ID=${1:-"Qwen/Qwen2.5-7B-Instruct"}
IS_QWEN35=${2:-"false"}

echo "Deploying model: $MODEL_ID (is_qwen35=$IS_QWEN35)"

# Update modal_serve.py defaults via env vars
export MODAL_MODEL_ID="$MODEL_ID"
export MODAL_IS_QWEN35="$IS_QWEN35"

modal deploy modal_serve.py

echo "Deployment complete. Set MODAL_PLANNER_URL and run benchmarks."
