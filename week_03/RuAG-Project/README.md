# RuAG Part 1 — Baseline Methods for Document-Level Relation Extraction

**Repository**: https://github.com/Melody-coder923/neu-self-improve-ai

This repository implements Part 1 of the RuAG (Learned-Rule-Augmented Generation) reproduction study, evaluating three inference-only baselines for relation extraction on the DWIE dataset:

- **Vanilla** — direct prompting (no demonstrations, no retrieval)
- **ICL** — in-context learning with *k* labeled demonstrations
- **RAG** — retrieval-augmented generation with similar training documents

## Quick Start (After Clone)

```bash
# 1. Enter project directory
cd RuAG-Project   # or week_03/RuAG-Project depending on your structure

# 2. Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Load data into SQLite (required before running)
python3 scripts/load_official_ruag_re.py \
  --dataset_root data/dataset \
  --sqlite_path data/dwie/dwie.sqlite

# 4. Run Vanilla baseline (configure config.yaml or OPENAI_API_KEY)
python3 main.py --method vanilla

# 5. View results
python3 scripts/export_results.py
```

## Assignment Compliance

- **Data management**: All data is loaded into a **single SQLite database** (`data/dwie/dwie.sqlite`). Data is stored in **normalized tables** (`documents`, `entities`, `relations`) and queried via SQL—no JSON blobs. No intermediate data files are created; all intermediate and final results are stored as tables in the database.
- **Dependencies**: Minimal set — `torch`, `transformers`, `accelerate`, `scikit-learn`, `PyYAML`, `tqdm`. No RL or agent frameworks.
- **Relation schema**: Top-20 official relation types from the RuAG repository (`relations_dict.json`).
- **Prompts**: Fully in English; adapted from the paper's style (reproduction, not direct copy).

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local model mode (default), set `provider: "local"` in `config.yaml`.

If using OpenAI API instead:

```bash
export OPENAI_API_KEY="your_key_here"
```

## 2) Load Data into SQLite

### Official RuAG-aligned mode (recommended)

The dataset is included at `data/dataset/` (from `microsoft/RuAG`):

```bash
python3 scripts/load_official_ruag_re.py \
  --dataset_root data/dataset \
  --sqlite_path data/dwie/dwie.sqlite
```

This will:
- Load the official train/test split (702 train, 100 test documents)
- Load the official 20-relation schema with descriptions from `relations_dict.json`
- Record the filtered document list used by the RuAG RE pipeline

### Alternative: custom raw files

```bash
# From a single raw file (auto-split):
python3 scripts/convert_dwie.py \
  --input_all /path/to/raw_all.json \
  --test_ratio 0.12 \
  --sqlite_path data/dwie/dwie.sqlite

# Or from separate train/test files:
python3 scripts/convert_dwie.py \
  --input_train /path/to/raw_train.json \
  --input_test /path/to/raw_test.json \
  --sqlite_path data/dwie/dwie.sqlite
```

## 3) Preflight Check

```bash
python3 scripts/preflight_check.py
```

Verifies: config keys, provider dependencies, API key (OpenAI mode), SQLite tables, and sample row parseability.

## 4) Run Baselines

### One-command pipeline

```bash
bash run_all.sh                              # full pipeline (load data + run all 3)
bash run_all.sh --skip-load                  # skip data loading, reuse existing sqlite
```

### Individual runs

```bash
python3 main.py --method vanilla
python3 main.py --method icl --k_shots 5
python3 main.py --method rag --top_k 5
```

## 5) Outputs

All results are stored in the **SQLite database** (`data/dwie/dwie.sqlite`):

| Table | Contents |
|-------|----------|
| `documents` | doc_id, split, content (normalized) |
| `entities` | doc_id, name, ord_idx (per-document entity list) |
| `relations` | doc_id, entity1, relation, entity2 (ground truth triples) |
| `relation_types` | The 20-relation schema |
| `filtered_docs` | Documents excluded from evaluation |
| `run_metrics` | Per-run aggregated metrics (P, R, F1, TP, FP, FN) |
| `run_predictions` | Per-document predictions, gold triples, and raw LLM output |

To view results (prints to console; no intermediate files):

```bash
python3 scripts/export_results.py
```

## 6) Reproducibility Check

```bash
python3 scripts/analyze_dataset_alignment.py
```

Reports: train/test document counts, relation schema size, test triple coverage, and any missing relations outside the schema.

## Project Structure

```
RuAG-Project/
├── main.py                     # Entry point for running baselines
├── config.yaml                 # Central configuration
├── requirements.txt            # Python dependencies
├── run_all.sh                  # One-command pipeline script
├── src/
│   ├── baseline_vanilla.py     # Vanilla baseline logic
│   ├── baseline_icl.py         # ICL baseline logic
│   ├── baseline_rag.py         # RAG baseline logic
│   ├── llm_client.py           # LLM loading and generation (local / OpenAI)
│   ├── prompting.py            # Prompt construction
│   ├── inference_utils.py      # Retry logic and triple parsing
│   ├── postprocess.py          # Triple sanitization
│   ├── evaluation.py           # Precision / Recall / F1
│   ├── data_preprocessing.py   # Raw JSON → Example parsing
│   ├── sqlite_data.py          # All SQLite database operations
│   └── utils.py                # Shared utilities
├── prompts/
│   ├── re_vanilla_prompt.txt   # Vanilla prompt template
│   ├── re_icl_prompt.txt       # ICL prompt template
│   └── re_rag_prompt.txt       # RAG prompt template
├── scripts/
│   ├── load_official_ruag_re.py    # Load official RuAG dataset into sqlite
│   ├── convert_dwie.py             # Alternative data loader
│   ├── export_results.py           # Export results from sqlite
│   ├── analyze_dataset_alignment.py # Dataset alignment check
│   └── preflight_check.py          # Environment verification
├── data/
│   ├── dataset/                # Official RuAG dataset files
│   │   ├── relations_dict.json
│   │   └── entity_relations_pairs/{train,test}/
│   └── dwie/
│       └── dwie.sqlite         # Single SQLite database (all data + results)
└── results/
    └── Part1_results_section.md  # Results report
```

## References

- Zhang, Y., et al. (2025). *RuAG: Learned-Rule-Augmented Generation for Large Language Models*. ICLR 2025.
- Zaporojets, K., et al. (2021). *DWIE: An Entity-centric Dataset for Multi-task Document-level Information Extraction*.
- Official RuAG repository: https://github.com/microsoft/RuAG
