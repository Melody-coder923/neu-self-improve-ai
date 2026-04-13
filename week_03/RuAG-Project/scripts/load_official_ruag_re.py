import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path when script is called directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sqlite_data import (
    init_db,
    upsert_documents_normalized,
    upsert_filtered_docs,
    upsert_relation_types,
)


# From the official RuAG RE pipeline (documents excluded for GPT protocol reasons).
OFFICIAL_FILTERED_DOCS = [
    "DW_16083654_relations",
    "DW_44141017_relations",
    "DW_17347807_relations",
    "DW_17736433_relations",
    "DW_18751636_relations",
    "DW_19210651_relations",
    "DW_39718698_relations",
]


def _load_folder_rows(folder: Path) -> List[Dict]:
    rows: List[Dict] = []
    for fp in sorted(folder.glob("*.json")):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        if "id" not in obj:
            obj["id"] = fp.stem
        rows.append(obj)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Load official RuAG relation_extraction dataset into sqlite."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/dataset",
        help="Path to official dataset root containing relations_dict.json and entity_relations_pairs/",
    )
    parser.add_argument("--sqlite_path", type=str, default="data/dwie/dwie.sqlite")
    parser.add_argument(
        "--skip_filtered_docs",
        action="store_true",
        help="Do not load the official filtered doc list.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    rel_dict_fp = root / "relations_dict.json"
    train_dir = root / "entity_relations_pairs" / "train"
    test_dir = root / "entity_relations_pairs" / "test"

    if not rel_dict_fp.exists():
        raise FileNotFoundError(f"Missing relations_dict.json: {rel_dict_fp}")
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Missing train/test folders under: {root / 'entity_relations_pairs'}"
        )

    relation_dict = json.loads(rel_dict_fp.read_text(encoding="utf-8"))
    relation_items = [(k, str(v).strip()) for k, v in relation_dict.items()]
    train_rows = _load_folder_rows(train_dir)
    test_rows = _load_folder_rows(test_dir)

    init_db(args.sqlite_path)
    upsert_relation_types(args.sqlite_path, relation_items)
    upsert_documents_normalized(args.sqlite_path, "train", train_rows)
    upsert_documents_normalized(args.sqlite_path, "test", test_rows)
    if args.skip_filtered_docs:
        upsert_filtered_docs(args.sqlite_path, [], reason="")
        filtered_n = 0
    else:
        upsert_filtered_docs(
            args.sqlite_path,
            OFFICIAL_FILTERED_DOCS,
            reason="Official RuAG filtered docs (GPT protocol / empty GT)",
        )
        filtered_n = len(OFFICIAL_FILTERED_DOCS)

    print(f"[OK] sqlite path: {Path(args.sqlite_path).resolve()}")
    print(f"[OK] relation types loaded: {len(relation_items)}")
    print(f"[OK] train rows loaded: {len(train_rows)}")
    print(f"[OK] test rows loaded : {len(test_rows)}")
    print(f"[OK] official filtered docs loaded: {filtered_n}")


if __name__ == "__main__":
    main()
