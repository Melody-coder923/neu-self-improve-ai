import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from src.baseline_icl import run_icl
from src.baseline_rag import run_rag
from src.baseline_vanilla import run_vanilla
from src.evaluation import precision_recall_f1
from src.llm_client import LLMClient
from src.sqlite_data import (
    get_filtered_doc_ids,
    get_relation_counts,
    get_relation_types,
    get_relation_types_with_desc,
    load_examples_from_sqlite,
    save_run_to_sqlite,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--method", type=str, choices=["vanilla", "icl", "rag"], required=True)
    p.add_argument("--k_shots", type=int, default=None)
    p.add_argument("--top_k", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    sqlite_path = cfg["data"]["sqlite_path"]
    train_split = cfg["data"].get("train_split", "train")
    test_split = cfg["data"].get("test_split", "test")
    exclude_filtered_docs = bool(cfg["data"].get("exclude_filtered_docs", True))
    filtered_doc_ids = get_filtered_doc_ids(sqlite_path) if exclude_filtered_docs else set()

    train_data = load_examples_from_sqlite(sqlite_path, split=train_split)
    test_data = load_examples_from_sqlite(
        sqlite_path,
        split=test_split,
        exclude_doc_ids=filtered_doc_ids,
    )
    if not train_data or not test_data:
        raise RuntimeError(
            "No train/test data loaded from sqlite. "
            "Please load raw data into sqlite first (see README)."
        )

    configured_schema = list(cfg["experiment"].get("relation_schema", []))
    top_k = int(cfg["experiment"].get("relation_top_k", 20))
    use_official_relation_types = bool(cfg["experiment"].get("use_official_relation_types", True))
    if configured_schema:
        relation_schema = configured_schema[:top_k] if top_k > 0 else configured_schema
    elif use_official_relation_types:
        official_relations = get_relation_types(sqlite_path)
        if top_k > 0:
            official_relations = official_relations[:top_k]
        relation_schema = official_relations
    else:
        rel_counts = get_relation_counts(sqlite_path, split=train_split)
        sorted_rels = sorted(
            rel_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
        if top_k > 0:
            sorted_rels = sorted_rels[:top_k]
        relation_schema = [rel for rel, _ in sorted_rels]
    if len(relation_schema) == 0:
        raise RuntimeError("Relation schema is empty. Cannot run extraction.")

    # Load official relation descriptions for prompt enrichment.
    rel_desc_pairs = get_relation_types_with_desc(sqlite_path)
    relation_descriptions = {rel: desc for rel, desc in rel_desc_pairs if desc}
    if not relation_descriptions:
        print("[WARN] No relation descriptions found in sqlite; prompts will use label-only format.")
    else:
        print(f"[INFO] Loaded relation descriptions for {len(relation_descriptions)} relation types.")

    llm = LLMClient(
        provider=cfg["model"].get("provider", "openai"),
        model_name=cfg["model"]["name"],
        temperature=float(cfg["model"]["temperature"]),
        max_tokens=int(cfg["model"]["max_tokens"]),
    )

    if args.method == "vanilla":
        rows = run_vanilla(
            test_data=test_data,
            relation_schema=relation_schema,
            llm=llm,
            prompt_path="prompts/re_vanilla_prompt.txt",
            relation_descriptions=relation_descriptions or None,
        )
    elif args.method == "icl":
        rows = run_icl(
            train_data=train_data,
            test_data=test_data,
            relation_schema=relation_schema,
            llm=llm,
            prompt_path="prompts/re_icl_prompt.txt",
            k_shots=args.k_shots if args.k_shots is not None else int(cfg["experiment"]["k_shots"]),
            relation_descriptions=relation_descriptions or None,
        )
    else:
        rows = run_rag(
            train_data=train_data,
            test_data=test_data,
            relation_schema=relation_schema,
            llm=llm,
            prompt_path="prompts/re_rag_prompt.txt",
            top_k=args.top_k if args.top_k is not None else int(cfg["experiment"]["rag_top_k"]),
            relation_descriptions=relation_descriptions or None,
        )

    # --- Align with official RuAG: gold only includes relations in schema ---
    schema_set = set(relation_schema)
    for row in rows:
        row["gold"] = [
            (h, r, t) for h, r, t in row["gold"] if r in schema_set
        ]

    gold_all = [x["gold"] for x in rows]
    pred_all = [x["pred"] for x in rows]
    metrics = precision_recall_f1(gold_all, pred_all)
    metrics["method"] = args.method

    # ---- All intermediate data stored in sqlite (no intermediate files) ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_run_to_sqlite(
        sqlite_path=sqlite_path,
        run_id=run_id,
        method=args.method,
        rows=rows,
        metrics=metrics,
    )

    print(f"\n[OK] Results saved to sqlite ({sqlite_path})")
    print(f"     run_id={run_id}, method={args.method}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

