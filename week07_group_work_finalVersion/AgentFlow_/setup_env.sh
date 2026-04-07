#!/bin/bash
# Setup script for AgentFlow environment

set -e

echo "=== AgentFlow Environment Setup ==="

# Check Python version
python3 --version | grep -E "3\.11" || echo "WARNING: Python 3.11 recommended"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    pip install uv
fi

# Upgrade core ML libraries to latest versions
pip install --upgrade transformers peft trl accelerate

# Install vLLM (for Qwen2.5 and other models)
# NOTE: vLLM has a weight-prefix mismatch with Qwen3.5 — use SGLang for Qwen3.5 instead
echo "Installing vLLM..."
uv pip install vllm --torch-backend=auto \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install SGLang (for Qwen3.5 — verified compatible)
echo "Installing SGLang..."
uv pip install 'sglang[all]' --find-links \
    https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Install additional dependencies
pip install modal verl datasets sqlparse wandb

echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Copy and configure: cp .env.template agentflow/agentflow/.env"
echo "2. Initialize Modal: modal token new"
echo "3. Verify: python quick_start.py"
