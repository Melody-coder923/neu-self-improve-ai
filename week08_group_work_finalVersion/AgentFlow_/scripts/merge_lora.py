from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def merge(base_id, lora_ckpt, output_dir):
    print(f"Loading base model: {base_id}")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16")
    print(f"Loading LoRA checkpoint: {lora_ckpt}")
    model = PeftModel.from_pretrained(base, lora_ckpt)
    print("Merging and unloading...")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(output_dir)
    print(f"Done: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--lora_ckpt", default="./checkpoints/qwen35-0.8b-flow-grpo-final")
    parser.add_argument("--output_dir", default="./checkpoints/qwen35-0.8b-merged")
    args = parser.parse_args()
    merge(args.base_id, args.lora_ckpt, args.output_dir)
