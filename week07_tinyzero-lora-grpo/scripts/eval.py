#!/usr/bin/env python3
"""Greedy (single-sample) eval: solve rate, format rate, valid-expression rate.

Use --adapter for a single checkpoint, or --checkpoint_dir to scan all checkpoints
and pick the one with the highest solve_rate on the held-out eval set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.parsing import format_ok_for_reward, parse_countdown_response
from src.reward import compute_countdown_reward


def _find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return all checkpoint-* subdirectories sorted by step number."""
    ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    # Also include final_adapter if present
    final = checkpoint_dir / "final_adapter"
    if final.is_dir():
        ckpts.append(final)
    return ckpts


def _load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def _eval_one_adapter(
    model_id: str,
    adapter: Path | None,
    ds,
    tokenizer,
    max_new_tokens: int,
    n: int,
) -> tuple[dict, list]:
    """Load model (+ optional adapter) and evaluate on ds[:n]. Returns (metrics, examples)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter is not None:
        if not adapter.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))

    model.eval()
    device = next(model.parameters()).device

    fmt_ok = 0
    expr_ok = 0
    solved = 0
    examples = []

    label = adapter.name if adapter is not None else "base"
    for i in tqdm(range(n), desc=f"eval {label}"):
        row = ds[i]
        prompt = row["prompt"]
        nums = row["nums"]
        target = row["target"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

        parsed = parse_countdown_response(completion)
        if format_ok_for_reward(parsed, raw_text=completion):
            fmt_ok += 1
        bd = compute_countdown_reward(completion, list(nums), target)
        if bd.expr_score > 0:
            expr_ok += 1
        if bd.solved:
            solved += 1

        if len(examples) < 5:
            examples.append(
                {
                    "prompt": prompt[:500],
                    "completion": completion[:1500],
                    "solved": bd.solved,
                    "reward_total": bd.total,
                }
            )

    # Explicitly release GPU memory before the next checkpoint load
    del model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    metrics = {
        "n": n,
        "format_rate": fmt_ok / n,
        "valid_expr_rate": expr_ok / n,
        "solve_rate": solved / n,
    }
    return metrics, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter", type=Path, default=None, help="Path to a single PEFT adapter dir (optional)")
    ap.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Scan all checkpoint-* dirs here and pick the one with highest solve_rate",
    )
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "test.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_results.json")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    ds = load_dataset("parquet", data_files=str(args.data), split="train")
    n = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    ds = ds.select(range(n))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once, shared across all checkpoint evaluations
    tokenizer = _load_tokenizer(args.model_id)

    if args.checkpoint_dir is not None:
        # --- Checkpoint selection mode ---
        checkpoints = _find_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* dirs found under {args.checkpoint_dir}")

        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate: {[p.name for p in checkpoints]}")
        all_results = []
        for ckpt in checkpoints:
            metrics, examples = _eval_one_adapter(
                args.model_id, ckpt, ds, tokenizer, args.max_new_tokens, n
            )
            all_results.append({"checkpoint": str(ckpt), "metrics": metrics, "examples": examples})
            print(f"  {ckpt.name}: solve_rate={metrics['solve_rate']:.4f}")

        best = max(all_results, key=lambda r: r["metrics"]["solve_rate"])
        print(f"\nBest checkpoint: {Path(best['checkpoint']).name}  solve_rate={best['metrics']['solve_rate']:.4f}")

        payload = {
            "selection_mode": "best_solve_rate",
            "best_checkpoint": best["checkpoint"],
            "best_metrics": best["metrics"],
            "best_examples": best["examples"],
            "all_checkpoints": [{"checkpoint": r["checkpoint"], "metrics": r["metrics"]} for r in all_results],
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(best["metrics"], indent=2))

    else:
        # --- Single adapter / base model mode ---
        metrics, examples = _eval_one_adapter(
            args.model_id, args.adapter, ds, tokenizer, args.max_new_tokens, n
        )
        payload = {"metrics": metrics, "examples": examples}
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
