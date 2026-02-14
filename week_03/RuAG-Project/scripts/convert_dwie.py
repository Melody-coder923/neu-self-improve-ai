import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import load_any_json
from src.sqlite_data import init_db, upsert_documents_normalized


def _split_from_explicit_field(rows):
    train_rows = []
    test_rows = []
    for row in rows:
        if not isinstance(row, dict):
            return None, None
        split = str(row.get("split", "")).strip().lower()
        if split in {"train", "training"}:
            train_rows.append(row)
        elif split in {"test", "dev", "valid", "validation"}:
            test_rows.append(row)
    if train_rows and test_rows:
        return train_rows, test_rows
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Load raw DWIE-like data into sqlite without format conversion."
    )
    parser.add_argument("--input_train", type=str, default=None)
    parser.add_argument("--input_test", type=str, default=None)
    parser.add_argument(
        "--input_all",
        type=str,
        default=None,
        help="Single raw file; script will split train/test by test_ratio.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sqlite_path", type=str, default="data/dwie/dwie.sqlite")
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    if not args.input_all and not (args.input_train and args.input_test):
        raise ValueError("Provide either --input_all OR both --input_train and --input_test.")

    init_db(args.sqlite_path)

    if args.input_all:
        rows = load_any_json(args.input_all)
        if args.max_docs:
            rows = rows[: args.max_docs]
        train_rows, test_rows = _split_from_explicit_field(rows)
        if train_rows is None or test_rows is None:
            random.seed(args.seed)
            random.shuffle(rows)
            n_test = max(1, int(len(rows) * args.test_ratio))
            test_rows = rows[:n_test]
            train_rows = rows[n_test:]
            print(
                "[WARN] No explicit split field found in --input_all rows. "
                "Using random split, which may deviate from paper-reported numbers."
            )
        else:
            print("[OK] Detected explicit split field in --input_all, preserving it.")
    else:
        train_rows = load_any_json(args.input_train)
        test_rows = load_any_json(args.input_test)
        if args.max_docs:
            train_rows = train_rows[: args.max_docs]
            test_rows = test_rows[: max(1, args.max_docs // 5)]

    upsert_documents_normalized(args.sqlite_path, "train", train_rows)
    upsert_documents_normalized(args.sqlite_path, "test", test_rows)

    print(f"[OK] sqlite path: {Path(args.sqlite_path).resolve()}")
    print(f"[OK] train rows loaded: {len(train_rows)}")
    print(f"[OK] test rows loaded : {len(test_rows)}")


if __name__ == "__main__":
    main()
