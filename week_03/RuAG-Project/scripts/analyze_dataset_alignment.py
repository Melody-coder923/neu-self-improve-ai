import argparse
from collections import Counter
import sys

import yaml

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sqlite_data import get_filtered_doc_ids, get_relation_types, load_examples_from_sqlite


def _count_relations(examples):
    c = Counter()
    for ex in examples:
        for _, r, _ in ex.relations:
            c[r] += 1
    return c


def main():
    parser = argparse.ArgumentParser(
        description="Analyze split/schema alignment for paper-like reproducibility."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    sqlite_path = cfg["data"]["sqlite_path"]
    train_split = cfg["data"].get("train_split", "train")
    test_split = cfg["data"].get("test_split", "test")
    exclude_filtered_docs = bool(cfg["data"].get("exclude_filtered_docs", True))
    relation_schema = list(cfg["experiment"].get("relation_schema", []))
    relation_top_k = int(cfg["experiment"].get("relation_top_k", 20))
    use_official_relation_types = bool(cfg["experiment"].get("use_official_relation_types", True))

    train = load_examples_from_sqlite(sqlite_path, split=train_split)
    excluded = get_filtered_doc_ids(sqlite_path) if exclude_filtered_docs else set()
    test = load_examples_from_sqlite(sqlite_path, split=test_split, exclude_doc_ids=excluded)
    train_rel = _count_relations(train)
    test_rel = _count_relations(test)

    if relation_schema:
        active_schema = relation_schema[:relation_top_k] if relation_top_k > 0 else relation_schema
    elif use_official_relation_types:
        official = get_relation_types(sqlite_path)
        active_schema = official[:relation_top_k] if relation_top_k > 0 else official
    else:
        ranked = [r for r, _ in train_rel.most_common()]
        active_schema = ranked[:relation_top_k] if relation_top_k > 0 else ranked
    schema_set = set(active_schema)

    test_total = sum(test_rel.values())
    test_in_schema = sum(v for k, v in test_rel.items() if k in schema_set)
    coverage = (test_in_schema / test_total) if test_total else 0.0

    print(f"train docs: {len(train)}")
    print(f"test docs : {len(test)}")
    print(f"unique train relations: {len(train_rel)}")
    print(f"unique test relations : {len(test_rel)}")
    print(f"active schema size    : {len(active_schema)}")
    print(f"test triple coverage by schema: {coverage:.4f} ({test_in_schema}/{test_total})")

    missing = [(k, v) for k, v in test_rel.items() if k not in schema_set]
    missing.sort(key=lambda x: (-x[1], x[0]))
    if missing:
        print("\nTop missing test relations (outside schema):")
        for k, v in missing[: args.top_n]:
            print(f"- {k}: {v}")
    else:
        print("\nAll test relations are covered by active schema.")

    print("\nTop train relations:")
    for k, v in train_rel.most_common(args.top_n):
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
