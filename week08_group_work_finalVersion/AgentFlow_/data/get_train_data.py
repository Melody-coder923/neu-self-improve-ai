"""
Download and prepare training data: NQ + DeepMath-103K → combined_train.parquet
"""
import os
import itertools
import pandas as pd
from datasets import load_dataset

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)

print("Downloading Natural Questions (NQ) via streaming (avoids full ~300 GB download)...")
nq = load_dataset(
    "google-research-datasets/natural_questions", "default",
    split="train", streaming=True
)
nq_rows = []
nq_skipped = 0
for ex in itertools.islice(nq, 10000):
    q = ex["question"]["text"]
    answers = ex["annotations"]["short_answers"]
    if answers and answers[0]["text"]:
        a = answers[0]["text"][0]
        nq_rows.append({"prompt": q, "answer": a, "source": "nq"})
    else:
        nq_skipped += 1

print(f"NQ samples: {len(nq_rows)} (skipped {nq_skipped} with no short answer)")
nq_df = pd.DataFrame(nq_rows)

print("Downloading DeepMath-103K...")
dm = load_dataset("zwhe99/DeepMath-103K", split="train")
dm_rows = []
for ex in dm:
    dm_rows.append({
        "prompt": ex.get("problem", ex.get("question", "")),
        "answer": ex.get("answer", ""),
        "source": "deepmath"
    })

print(f"DeepMath samples: {len(dm_rows)}")
dm_df = pd.DataFrame(dm_rows)

combined = pd.concat([nq_df, dm_df], ignore_index=True)
combined = combined.dropna(subset=["prompt", "answer"])
combined = combined[combined["prompt"].str.len() > 10]
combined.to_parquet("data/train/combined_train.parquet", index=False)
print(f"Combined training data saved: {len(combined)} samples → data/train/combined_train.parquet")
