"""
Modal vLLM Deployment for Qwen3.5 Models (Step 3)
===================================================
Deploys Qwen3.5-9B or Qwen3.5-27B on Modal with vLLM.

Usage:
    # Deploy 9B (default)
    MODEL_SIZE=9B modal deploy modal_serve_qwen35.py

    # Deploy 27B
    MODEL_SIZE=27B modal deploy modal_serve_qwen35.py

    # Quick test (run once, not persistent)
    MODEL_SIZE=9B modal run modal_serve_qwen35.py

After deployment, you'll get a URL like:
    https://<workspace>--agentflow-qwen35-9b-serve.modal.run
"""

import os
import modal

# --- Read model size from env (default: 9B) ---
MODEL_SIZE = os.environ.get("MODEL_SIZE", "9B").upper()

# --- Model configurations ---
MODEL_CONFIGS = {
    "9B": {
        "name": "Qwen/Qwen3.5-9B",
        "gpu": modal.gpu.A100(count=1, size="40GB"),
        "max_model_len": 4096,  # Keep short for AgentFlow (saves VRAM)
        "dtype": "bfloat16",
        "tp_size": 1,
    },
    "27B": {
        "name": "Qwen/Qwen3.5-27B",
        "gpu": modal.gpu.A100(count=1, size="80GB"),
        "max_model_len": 4096,
        "dtype": "bfloat16",
        "tp_size": 1,
    },
}

if MODEL_SIZE not in MODEL_CONFIGS:
    raise ValueError(f"MODEL_SIZE must be one of {list(MODEL_CONFIGS.keys())}, got: {MODEL_SIZE}")

config = MODEL_CONFIGS[MODEL_SIZE]
MODEL_NAME = config["name"]

# --- Modal App ---
app = modal.App(f"agentflow-qwen35-{MODEL_SIZE.lower()}")

# --- Docker image with vLLM (need >= 0.17 for Qwen3.5 GDN support) ---
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.7.0",
        "torch>=2.5.0",
        "transformers>=4.50.0",
        "fastapi",
        "uvicorn",
    )
)

# --- Model volume (cache weights across restarts) ---
model_volume = modal.Volume.from_name(
    f"agentflow-qwen35-{MODEL_SIZE.lower()}-cache",
    create_if_missing=True,
)
MODEL_DIR = "/model-cache"


@app.function(
    image=vllm_image,
    gpu=config["gpu"],
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
    allow_concurrent_inputs=16,
    container_idle_timeout=300,  # Keep alive 5 min after last request
)
@modal.web_server(port=8000, startup_timeout=600)  # 10 min for first-time model download
def serve():
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--download-dir", MODEL_DIR,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", str(config["max_model_len"]),
        "--dtype", config["dtype"],
        "--tensor-parallel-size", str(config["tp_size"]),
        "--trust-remote-code",
        "--served-model-name", MODEL_NAME,
        "--language-model-only",   # Text only, skip vision encoder
        "--enforce-eager",         # Safer for new architectures
    ]

    print(f"Starting vLLM with: {' '.join(cmd)}")
    subprocess.Popen(cmd)
