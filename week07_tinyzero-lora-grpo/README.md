# TinyZero Countdown with LoRA + GRPO

This repo is my implementation of the TinyZero countdown task using a LoRA adapter (PEFT) instead of full fine-tuning, trained with GRPO (TRL).

## Assignment Checklist

- **Countdown task + LoRA (not full FT):** implemented in `scripts/train.py`
- **Latest PEFT / TRL / Transformers:** installed from `requirements.txt` without version pins; run used `transformers 5.4.0`, `trl 1.0.0`, `peft 0.18.1`
- **Reuse TinyZero prompt:** template follows TinyZero countdown base style in `data/prompts/prompt_template.txt`
- **Repo + runnable instructions:** included below
- **Individual training:** generated artifacts are saved under `outputs/`

References:
- TinyZero: <https://github.com/Jiayi-Pan/TinyZero>
- Countdown prompt source: <https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py>

## Environment

- Python 3.10+
- GPU recommended (Colab A100 preferred for long runs; Mac MPS works for smoke checks)

Install:

```bash
pip install -r requirements.txt
```

Compatibility with the latest TRL / PEFT / Transformers stack was verified via a smoke test (dummy reward training).

Optional environment snapshot:

```bash
pip freeze > environment.txt
```

## Repository Structure

- `scripts/prepare_data.py`: TinyZero-style split + prompt formatting
- `scripts/sft.py`: optional SFT warm-up
- `scripts/train.py`: GRPO + LoRA training
- `scripts/eval.py`: base vs adapter evaluation
- `src/reward.py`: layered reward function
- `data/prompts/prompt_template.txt`: TinyZero-style prompt template

## Run Instructions

Run from this folder:

```bash
cd week07_tinyzero-lora-grpo
export PYTHONPATH="$(pwd)"
```

### 1) Prepare Data

```bash
python scripts/prepare_data.py
```

Outputs:
- `data/processed/train.parquet` (327,680)
- `data/processed/test.parquet` (1,024)

### 2) Smoke Check (pipeline only, not final performance)

```bash
python scripts/train.py --dummy_reward --max_steps 5 --report_to none --output_dir outputs/smoke
```

### 3) SFT Warm-up (used in final reported run)

```bash
python scripts/sft.py --num_examples 120 --epochs 3 --output_dir outputs/sft_adapter
```

### 4) GRPO Training (final reported run)

```bash
python scripts/train.py \
  --init_adapter outputs/sft_adapter/final_adapter \
  --max_steps 500 \
  --report_to none \
  --output_dir outputs/grpo_from_sft_v3
```

### 5) Evaluation

Base:

```bash
python scripts/eval.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --data data/processed/test.parquet \
  --out outputs/eval_base.json \
  --max_samples 200
```

Adapter:

```bash
python scripts/eval.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --adapter outputs/grpo_from_sft_v3/final_adapter \
  --data data/processed/test.parquet \
  --out outputs/eval_grpo_sft_v3.json \
  --max_samples 200
```

## Results

### Final submission metrics (used for report)

`max_samples=200`, greedy decode:

| Model | format_rate | valid_expr_rate | solve_rate |
|---|---:|---:|---:|
| Base model (Qwen2.5-1.5B) | 0.0 | 0.0 | 0.0 |
| SFT + GRPO (500 steps, LoRA) | 0.995 | 0.995 | 0.01 |

SFT-only was observed to achieve high `format_rate` and `valid_expr_rate` but near-zero `solve_rate`, indicating that RL (GRPO) is necessary for reward-driven improvement.

Evidence files:
- `outputs/eval_base.json`
- `outputs/eval_grpo_sft_v3.json`

### Smoke-only metrics (not used as final claim)

Very short local check (`Qwen2.5-0.5B`, `max_steps=5`) can show near-zero gain and is only used to verify the pipeline.

## Parameter Efficiency (LoRA)

- Total parameters: `1,544,803,840`
- Trainable parameters: `1,089,536`
- Trainable ratio: `0.0705%`

## Reward Function (Layered)

| Level | Check | Score |
|---|---|---:|
| 1 | Format (`<think>...</think><answer>...</answer>`) | +0.5 |
| 2 | Valid arithmetic expression (AST-safe) | +0.5 |
| 3 | Number usage check (multiset) | +0.75 |
| 4 | Final value equals target | +3.0 |

## Discussion

From this run, the model clearly learns format and valid-expression behavior much earlier than it learns actual solving.

Even after GRPO, `solve_rate` is still low at this scale/step budget. In practice, this means getting the output shape right is easier than learning real search-style reasoning on countdown.

## Limitations & Future Work

- Limited GRPO steps (500) restrict reasoning emergence
- Small base model (1.5B) may lack sufficient capacity for search
- Sparse reward signal makes optimization difficult

Future directions:
- Increase GRPO training steps
- Evaluate larger base models
- Explore improved reward shaping or intermediate rewards

## Outputs

- `outputs/<run>/checkpoint-*`: intermediate checkpoints
- `outputs/<run>/final_adapter`: final LoRA adapter
- `outputs/<run>/samples.json`: sampled generations and reward breakdowns
- `outputs/eval_base.json`: base model metrics
- `outputs/eval_grpo_sft_v3.json`: final adapter metrics
