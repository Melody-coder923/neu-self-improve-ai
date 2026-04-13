"""Export results from sqlite to stdout. No intermediate files — all data stays in DB.

Usage:
    python scripts/export_results.py                    # print comparison table to console
    python scripts/export_results.py --sqlite_path PATH  # custom db path
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def get_latest_metrics(sqlite_path: str):
    """Return the latest run_metrics row per method."""
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT method, precision, recall, f1, tp, fp, fn, run_id
            FROM run_metrics
            ORDER BY run_id DESC
            """
        ).fetchall()

    seen = set()
    latest = []
    for r in rows:
        if r["method"] not in seen:
            seen.add(r["method"])
            latest.append(dict(r))
    # Sort by method name for consistent output.
    latest.sort(key=lambda x: x["method"])
    return latest


def get_predictions(sqlite_path: str, method: str, run_id: Optional[str] = None):
    """Return predictions for a given method (optionally a specific run)."""
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        if run_id:
            rows = conn.execute(
                "SELECT * FROM run_predictions WHERE method = ? AND run_id = ? ORDER BY doc_id",
                (method, run_id),
            ).fetchall()
        else:
            # Get the latest run
            latest = conn.execute(
                "SELECT run_id FROM run_metrics WHERE method = ? ORDER BY run_id DESC LIMIT 1",
                (method,),
            ).fetchone()
            if not latest:
                return []
            rows = conn.execute(
                "SELECT * FROM run_predictions WHERE method = ? AND run_id = ? ORDER BY doc_id",
                (method, latest["run_id"]),
            ).fetchall()
    return [dict(r) for r in rows]


def print_comparison_table(metrics_list):
    """Pretty-print a comparison table to stdout."""
    if not metrics_list:
        print("No results found in sqlite.")
        return

    header = f"{'Method':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}"
    print("=" * len(header))
    print("Baseline Comparison (from sqlite)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m['method']:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Print baseline comparison table from sqlite (no file output)."
    )
    parser.add_argument("--sqlite_path", default="data/dwie/dwie.sqlite")
    args = parser.parse_args()

    if not Path(args.sqlite_path).exists():
        print(f"[ERROR] sqlite not found: {args.sqlite_path}")
        sys.exit(1)

    metrics_list = get_latest_metrics(args.sqlite_path)
    print_comparison_table(metrics_list)


if __name__ == "__main__":
    main()
