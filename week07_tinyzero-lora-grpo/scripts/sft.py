#!/usr/bin/env python3
"""SFT warm-up: teach format before GRPO."""
from __future__ import annotations
import argparse, sys
from itertools import permutations, product
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from src.data_utils import build_countdown_dataset, load_prompt_template

def solve(nums, target):
    ops = ["+", "-", "*", "/"]
    n = len(nums)
    tmpls3 = ["({a}{o1}{b}){o2}{c}", "{a}{o1}({b}{o2}{c})"]
    tmpls4 = ["(({a}{o1}{b}){o2}{c}){o3}{d}", "({a}{o1}({b}{o2}{c})){o3}{d}",
              "({a}{o1}{b}){o2}({c}{o3}{d})", "{a}{o1}(({b}{o2}{c}){o3}{d})",
              "{a}{o1}({b}{o2}({c}{o3}{d}))"]
    for perm in permutations(nums):
        p = [str(x) for x in perm]
        if n == 3:
            a,b,c = p
            for s1, s2 in product(ops, repeat=2):
                for tmpl in tmpls3:
                    expr = tmpl.format(a=a,b=b,c=c,o1=s1,o2=s2)
                    try:
                        v = Fraction(eval(expr))
                        if abs(float(v)-target)<1e-5: return expr
                    except: pass
        elif n == 4:
            a,b,c,d = p
            for s1, s2, s3 in product(ops, repeat=3):
                for tmpl in tmpls4:
                    expr = tmpl.format(a=a,b=b,c=c,d=d,o1=s1,o2=s2,o3=s3)
                    try:
                        v = Fraction(eval(expr))
                        if abs(float(v)-target)<1e-5: return expr
                    except: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--num_examples", type=int, default=120)
    ap.add_argument("--output_dir", default="outputs/sft_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    template = load_prompt_template()
    raw = build_countdown_dataset()["train"]

    print("Solving examples...")
    records = []
    for row in raw:
        if len(records) >= args.num_examples:
            break
        nums = list(row["nums"]); target = row["target"]
        expr = solve(nums, target)
        if expr is None: continue
        nums_line = ", ".join(str(n) for n in nums)
        prompt = template.format(nums_line=nums_line, target=target)
        completion = f"I need to use {nums_line} to reach {target}.\nAfter checking: {expr} = {target}.\n</think>\n<answer>{expr}</answer>"
        records.append({"text": prompt + completion})

    print(f"Built {len(records)} SFT examples")
    sft_ds = Dataset.from_list(records)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    targs = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy='no',
        report_to='none',
        dataset_text_field='text',
    )
    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=sft_ds,
        processing_class=tok,
    )
    trainer.train()
    out = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Saved SFT adapter -> {out}")

if __name__ == "__main__":
    main()
