# RuAG Relation Extraction - Baseline Implementation

Re-implementation of the baseline approach from the paper:
**RuAG: Learned-Rule-Augmented Generation for Large Language Models** (ICLR 2025)

## Task
Document-level relation extraction on the DWIE dataset using GPT-4 (Vanilla baseline).

## Project Structure
```
ruag-project/
├── ruag.db                    # SQLite database (created by data_loader.py)
├── main.py                    # Main entry point
├── src/
│   ├── data_loader.py         # Load DWIE into SQLite
│   ├── baseline_vanilla.py    # Vanilla LLM baseline
│   └── evaluation.py          # Compute Precision/Recall/F1
├── prompts/
│   └── vanilla_prompt.txt     # Prompt template for Vanilla baseline
├── dataset/                   # DWIE dataset (from official RuAG repo)
│   ├── relations_dict.json    # 20 relation type definitions
│   └── entity_relations_pairs/
│       ├── train/             # 702 training documents
│       └── test/              # 100 test documents (93 active)
└── README.md
```

## Dependencies
```
pip install openai
```
That's it. `sqlite3` is built into Python.

## Setup & Run

### Step 1: Load data into SQLite
```bash
python main.py --step load
```
This reads all JSON files from `dataset/` and creates `ruag.db`.

### Step 2: Run Vanilla baseline
```bash
# Using API key as argument
python main.py --step vanilla --api-key sk-your-key-here

# Or set environment variable
export OPENAI_API_KEY=sk-your-key-here
python main.py --step vanilla
```
This calls GPT-4 for each of 93 test documents and stores predictions in `ruag.db`.

### Step 3: Evaluate results
```bash
python main.py --step evaluate
```
This computes Precision, Recall, F1 by comparing predictions with ground truth.

### Run all steps at once
```bash
python main.py --step all --api-key sk-your-key-here
```

## LLM Configuration (from paper Table A5)
| Parameter | Value |
|-----------|-------|
| Model | gpt-4-0613 |
| Temperature | 0 |
| Top-p | 1 |
| Max tokens | 1000 |
| Frequency penalty | 0 |
| Presence penalty | 0 |

## Expected Results (paper Table 2)
| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| Vanilla (our target) | 46.94% | 69.61% | 35.41% |
| ICL | 50.26% | 74.09% | 38.02% |
| RAG | 52.30% | 78.64% | 39.17% |
| **RuAG (Part 2)** | **60.42%** | **69.44%** | **53.48%** |

## Database Schema
All data is stored in a single `ruag.db` SQLite file:
- `documents`: 802 articles (702 train + 100 test)
- `entities`: all entities per document
- `relations`: ground truth relation triples
- `predictions`: LLM predicted relation triples
- `relation_types`: 20 relation definitions
- `filtered_docs`: 7 documents excluded from evaluation
- `rules`: (Part 2) MCTS-discovered logic rules

## Paper Reference
```
@inproceedings{zhang2025ruag,
    title={RuAG: Learned-rule-augmented Generation for Large Language Models},
    author={Yudi Zhang and Pei Xiao and Lu Wang and others},
    booktitle={ICLR 2025},
    url={https://openreview.net/forum?id=BpIbnXWfhL}
}
```
