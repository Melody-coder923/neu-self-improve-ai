#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for Part 1 baselines:
# 1) Load official RuAG dataset into sqlite (from data/dataset/)
# 2) Run vanilla / icl / rag
#
# Examples:
#   bash run_all.sh                          # full pipeline
#   bash run_all.sh --skip-load              # skip data loading, reuse existing sqlite
#   bash run_all.sh --dataset-root /path/to/dataset  # custom dataset path

SKIP_LOAD=0
DATASET_ROOT="data/dataset"
K_SHOTS="5"
TOP_K="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-load)
      SKIP_LOAD=1
      shift
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --k-shots)
      K_SHOTS="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

SQLITE_PATH="data/dwie/dwie.sqlite"

echo "[1/4] Load official RuAG dataset into sqlite"
if [[ "$SKIP_LOAD" -eq 1 ]]; then
  echo "  - Skip loading (use existing sqlite at $SQLITE_PATH)"
else
  python3 scripts/load_official_ruag_re.py \
    --dataset_root "$DATASET_ROOT" \
    --sqlite_path "$SQLITE_PATH"
fi

echo "[2/4] Run Vanilla baseline"
python3 main.py --method vanilla

echo "[3/4] Run ICL baseline"
python3 main.py --method icl --k_shots "$K_SHOTS"

echo "[4/4] Run RAG baseline"
python3 main.py --method rag --top_k "$TOP_K"

echo
echo "Done. All results saved to sqlite (data/dwie/dwie.sqlite)."
echo "To view results:  python3 scripts/export_results.py"
