"""
Download AIME 2024 validation data → aime24.parquet
"""
import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/val", exist_ok=True)

print("Downloading AIME 2024...")
try:
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = []
    for ex in ds:
        rows.append({
            "prompt": ex.get("problem", ""),
            "answer": str(ex.get("answer", "")),
            "source": "aime24"
        })
    df = pd.DataFrame(rows)
    df.to_parquet("data/val/aime24.parquet", index=False)
    print(f"AIME24 validation data saved: {len(df)} samples → data/val/aime24.parquet")
except Exception as e:
    print(f"Error downloading AIME24: {e}")
    print("Creating placeholder validation data...")
    df = pd.DataFrame([{"prompt": "placeholder", "answer": "0", "source": "aime24"}])
    df.to_parquet("data/val/aime24.parquet", index=False)


