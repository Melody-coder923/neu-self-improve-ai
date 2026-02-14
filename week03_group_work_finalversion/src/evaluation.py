"""
evaluation.py - Compute Precision, Recall, F1 for relation extraction.

Reads predictions and ground truth from SQLite database,
computes micro-averaged metrics (matching the paper's evaluation method).
"""

import sqlite3
import os


def evaluate(db_path, method="vanilla"):
    """Evaluate predictions against ground truth."""
    conn = sqlite3.connect(db_path)

    # Get all active test doc_ids
    test_docs = conn.execute("""
        SELECT doc_id FROM documents
        WHERE split = 'test'
        AND doc_id NOT IN (SELECT doc_id FROM filtered_docs)
        ORDER BY doc_id
    """).fetchall()
    test_doc_ids = [row[0] for row in test_docs]

    # Get relation types for per-relation metrics
    relation_types = [row[0] for row in conn.execute("SELECT relation FROM relation_types").fetchall()]

    # Initialize per-relation counters
    relation_metrics = {rel: {"TP": 0, "FP": 0, "FN": 0} for rel in relation_types}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    doc_results = []

    for doc_id in test_doc_ids:
        # Get ground truth
        gt_rows = conn.execute(
            "SELECT entity1, relation, entity2 FROM relations WHERE doc_id = ?",
            (doc_id,)
        ).fetchall()
        ground_truth = set((e1, rel, e2) for e1, rel, e2 in gt_rows)

        # Get predictions
        pred_rows = conn.execute(
            "SELECT entity1, relation, entity2 FROM predictions WHERE doc_id = ? AND method = ?",
            (doc_id, method)
        ).fetchall()
        predictions = set((e1, rel, e2) for e1, rel, e2 in pred_rows)

        # Per-document metrics
        tp = len(predictions & ground_truth)
        fp = len(predictions - ground_truth)
        fn = len(ground_truth - predictions)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        doc_results.append({
            "doc_id": doc_id,
            "predicted": len(predictions),
            "ground_truth": len(ground_truth),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1
        })

        # Per-relation metrics
        for rel in relation_types:
            pred_rel = set(filter(lambda x: x[1] == rel, predictions))
            gt_rel = set(filter(lambda x: x[1] == rel, ground_truth))

            rel_tp = len(pred_rel & gt_rel)
            rel_fp = len(pred_rel - gt_rel)
            rel_fn = len(gt_rel - pred_rel)

            relation_metrics[rel]["TP"] += rel_tp
            relation_metrics[rel]["FP"] += rel_fp
            relation_metrics[rel]["FN"] += rel_fn

    # Overall micro-averaged metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Evaluation Results - Method: {method}")
    print(f"{'='*60}")
    print(f"  Documents evaluated: {len(test_doc_ids)}")
    print(f"  Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"")
    print(f"  Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"  Recall:    {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"  F1:        {overall_f1:.4f} ({overall_f1*100:.2f}%)")
    print(f"{'='*60}")

    # Paper reference values
    print(f"\n  Paper reference (Vanilla GPT-4):")
    print(f"    Precision: 69.61%")
    print(f"    Recall:    35.41%")
    print(f"    F1:        46.94%")

    # Per-relation breakdown
    print(f"\n  Per-relation breakdown:")
    print(f"  {'Relation':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*56}")
    for rel in relation_types:
        m = relation_metrics[rel]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if tp + fp + fn > 0:  # Only show relations that appear
            print(f"  {rel:<20} {tp:>4} {fp:>4} {fn:>4} {p:>6.2%} {r:>6.2%} {f:>6.2%}")

    conn.close()

    return {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "doc_results": doc_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate relation extraction results")
    parser.add_argument("--db", default="../ruag.db", help="Path to SQLite database")
    parser.add_argument("--method", default="vanilla", help="Method to evaluate")
    args = parser.parse_args()

    evaluate(args.db, args.method)
