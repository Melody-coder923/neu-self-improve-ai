#!/usr/bin/env python3
"""Download Countdown data (TinyZero split) and write processed parquet + sanity-check samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import add_prompt_column, build_countdown_dataset, default_prompt_template_path, load_prompt_template


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--template", type=Path, default=None, help="Override prompt template path")
    p.add_argument("--sanity_print", type=int, default=3, help="Print N samples for manual review")
    args = p.parse_args()

    tpl_path = args.template or default_prompt_template_path()
    template = load_prompt_template(tpl_path)

    ds = build_countdown_dataset()
    train = add_prompt_column(ds["train"], template)
    test = add_prompt_column(ds["test"], template)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.parquet"
    test_path = args.out_dir / "test.parquet"
    train.to_parquet(train_path)
    test.to_parquet(test_path)
    print(f"Wrote {train_path} ({len(train)} rows)")
    print(f"Wrote {test_path} ({len(test)} rows)")

    n = args.sanity_print
    for i in range(min(n, len(train))):
        row = train[i]
        print("\n--- sanity sample", i, "---")
        print(json.dumps({k: row[k] for k in ("nums", "target", "prompt")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
