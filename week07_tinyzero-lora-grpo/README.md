# TinyZero Countdown — LoRA + GRPO

Reproducing the TinyZero Countdown task, replacing full fine-tuning with LoRA (PEFT) + GRPO (TRL).

## Project Goal

- Verify whether LoRA + GRPO can enable small models to learn structured reasoning
- Demonstrate parameter efficiency: only ~0.07% of parameters are updated vs full fine-tuning

## Environment

- Python 3.10+
- CUDA GPU (tested on Google Colab A100)
- Key dependencies: `torch>=2.2`, `transformers>=4.46`, `trl==0.29.1`, `peft==0.18.1`

> `trl` and `peft` are pinned to tested versions. Newer releases may have breaking API changes in `GRPOTrainer` and `LoraConfig`.

```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
cd tinyzero-lora-grpo
export PYTHONPATH=$(pwd)
python scripts/prepare_data.py
```

Generates `data/processed/train.parquet` (327,680 rows) and `test.parquet` (1,024 rows), matching TinyZero's exact index-based split.

## Training

All commands assume `PYTHONPATH=$(pwd)` is set from the repo root.

Step 1 — Smoke test with dummy reward (make sure the pipeline runs before real training):

```bash
python scripts/train.py --dummy_reward --max_steps 5 --report_to none --output_dir outputs/smoke
```

Step 2 — SFT warm-up (teach the model the output format):

```bash
python scripts/sft.py --num_examples 120 --epochs 3 --output_dir outputs/sft_adapter
```

Step 3 — GRPO from SFT adapter:

```bash
python scripts/train.py \
    --init_adapter outputs/sft_adapter/final_adapter \
    --max_steps 500 \
    --report_to none \
    --output_dir outputs/grpo_from_sft_v3
```

## Evaluation

All commands assume `PYTHONPATH=$(pwd)` is set from the repo root.

Base model baseline:

```bash
python scripts/eval.py \
    --model_id Qwen/Qwen2.5-1.5B \
    --data data/processed/test.parquet \
    --out outputs/eval_base.json \
    --max_samples 200
```

Trained model:

```bash
python scripts/eval.py \
    --model_id Qwen/Qwen2.5-1.5B \
    --adapter outputs/grpo_from_sft_v3/final_adapter \
    --data data/processed/test.parquet \
    --out outputs/eval_grpo_sft_v3.json \
    --max_samples 200
```

## Outputs

| Path | Description |
|------|-------------|
| `outputs/<run>/checkpoint-*/` | Intermediate checkpoints saved during training |
| `outputs/<run>/final_adapter/` | Final LoRA adapter weights |
| `outputs/<run>/samples.json` | Per-step sample log: prompt, completion, reward breakdown |
| `outputs/eval_base.json` | Evaluation results for base model |
| `outputs/eval_grpo_sft_v3.json` | Evaluation results for trained model |

## Results

| Model | format_rate | valid_expr_rate | solve_rate |
|-------|-------------|-----------------|------------|
| Base model (Qwen2.5-1.5B) | 0.0 | 0.0 | 0.0 |
| SFT + GRPO 500 steps | 0.995 | 0.995 | 0.01 |

LoRA parameter efficiency:
- Total parameters: 1,544,803,840
- Trainable parameters: 1,089,536
- Trainable %: 0.0705%

## Prompt

Based on [TinyZero's countdown.py](https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py) with the same `<think>/<answer>` structure. Two things changed:

1. Dropped the conversation-style framing — this project uses a base model, not instruct.
2. Made the number-usage rule more explicit (`each number exactly as many times as it appears`) so the multiset reward check works cleanly.

## Reward Function (Layered)

| Level | Check | Score |
|-------|-------|-------|
| 1 | Format: `<think>...</think><answer>...</answer>` | +0.5 |
| 2 | Valid arithmetic expression (AST-based, no eval) | +0.5 |
| 3 | Correct number usage (multiset match) | +0.75 |
| 4 | Result equals target | +3.0 |
