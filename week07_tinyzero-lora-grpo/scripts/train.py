#!/usr/bin/env python3
"""GRPO training with LoRA on Countdown (TRL >= 0.29)."""

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
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoTokenizer, TrainerCallback, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from src.model_utils import count_params
from src.reward import countdown_reward, countdown_reward_with_breakdown, dummy_reward

class SampleLogCallback(TrainerCallback):
    def __init__(self, dataset, tokenizer, out_path, log_every=50, n_samples=4, max_new_tokens=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.out_path = out_path
        self.log_every = log_every
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.records = []
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return
        if model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        n = min(self.n_samples, len(self.dataset))
        prompts, nums_list, targets = [], [], []
        for i in range(n):
            row = self.dataset[i]
            prompts.append(row["prompt"])
            nums_list.append(row["nums"])
            targets.append(row["target"])
        completions = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                )
            gen_ids = out[0, inputs.input_ids.shape[1]:]
            completions.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        _, breakdowns = countdown_reward_with_breakdown(
            prompts, completions, nums=nums_list, target=targets
        )
        step_records = []
        for i, (prompt, completion, bd) in enumerate(zip(prompts, completions, breakdowns)):
            step_records.append({
                "step": state.global_step,
                "sample_idx": i,
                "prompt": prompt[:500],
                "completion": completion[:1500],
                "parsed_answer": bd.details.get("value"),
                "reward_breakdown": {
                    "format": bd.format_score,
                    "expr": bd.expr_score,
                    "multiset": bd.multiset_score,
                    "target": bd.target_score,
                    "total": bd.total,
                },
                "solved": bd.solved,
            })
        self.records.extend(step_records)
        self.out_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[SampleLog] step={state.global_step} solve_rate={sum(r['solved'] for r in step_records)}/{n}")
        model.train()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--init_adapter", type=Path, default=None)
    ap.add_argument("--train_data", type=Path, default=ROOT / "data" / "processed" / "train.parquet")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "grpo_run")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--report_to", type=str, default="none")
    ap.add_argument("--dummy_reward", action="store_true", help="Use dummy reward (always 1.0) for pipeline smoke test")
    args = ap.parse_args()

    train_ds = load_dataset("parquet", data_files=str(args.train_data), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none",
    )

    if args.init_adapter:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = args.model_id

    grpo_args = GRPOConfig(
        output_dir=str(args.output_dir), learning_rate=args.learning_rate, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, bf16=True,
        num_generations=4, max_completion_length=512, report_to=[args.report_to] if args.report_to != 'none' else [],
    )

    reward_fn = dummy_reward if args.dummy_reward else countdown_reward
    if args.dummy_reward:
        print("[train] Using dummy_reward (pipeline smoke test)")

    sample_log_path = args.output_dir / "samples.json"
    sample_cb = SampleLogCallback(
        dataset=train_ds,
        tokenizer=tokenizer,
        out_path=sample_log_path,
    )

    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=grpo_args,
        train_dataset=train_ds, processing_class=tokenizer,
        peft_config=peft_config if not args.init_adapter else None,
        callbacks=[sample_cb],
    )

    stats = count_params(trainer.model)
    print(f"Parameters: total={stats.total:,} trainable={stats.trainable:,} ({stats.trainable_pct:.4f}% trainable)")

    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))

if __name__ == "__main__":
    main()
