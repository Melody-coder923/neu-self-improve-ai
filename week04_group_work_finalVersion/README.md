# RuAG Relation Extraction — RuAG with MCTS-Discovered Logic Rules (Part 2)

***Note:** This is Part 2 (Proposed approach with MCTS). Part 1 (Baseline) was submitted in week03 as `week03_group_work_finalversion`.

---

Re-implementation of the baseline approach from:  
**RuAG: Learned-Rule-Augmented Generation for Large Language Models** (ICLR 2025)

- Paper: [OpenReview](https://openreview.net/forum?id=BpIbnXWfhL)  
- Original repo: [github.com/microsoft/RuAG](https://github.com/microsoft/RuAG)

---

## Course Information

- **Course:** INFO 7375 — Special Topics in AI Engineering and Applications
- **Institution:** Northeastern University
- **Instructor:** Prof. Suhabe Bugrara
- **Assignment:** Week 04 Group Work — RuAG Part 2 (RuAG with MCTS-Discovered Logic Rules)

## Team Members

- Yan Zhao
- Zhenyu Dai
- Chien-Cheng Wang

---

## Overview

This project implements the full **RuAG framework** for document-level relation extraction on the DWIE dataset. Building on the Part 1 Vanilla baseline, Part 2 uses Monte Carlo Tree Search (MCTS) to automatically discover first-order logic rules from training data, translates them into natural language, and injects them into LLM prompts to improve relation extraction performance.

The core idea: instead of relying solely on the LLM's internal knowledge, RuAG mines patterns like "if someone is head_of_state of a country, they are also a citizen_of that country" from training data, then tells the LLM to apply these rules during extraction. This significantly boosts Recall without requiring model fine-tuning.

---

## Results Summary

| Method | Model | Precision | Recall | F1 | RuAG Improvement |
|--------|-------|-----------|--------|-----|-----------------|
| Paper: GPT-3.5 Vanilla | GPT-3.5 | 28.57% | 14.10% | 18.94% | — |
| Paper: GPT-3.5 RuAG | GPT-3.5 | 34.43% | 21.71% | 26.63% | +7.69% |
| Paper: GPT-4 Vanilla | GPT-4 | 69.61% | 35.41% | 46.94% | — |
| Paper: GPT-4 RuAG | GPT-4 | 61.28% | 59.57% | 60.42% | +13.48% |
| **Ours: Vanilla** | **Mistral Small** | **62.63%** | **33.13%** | **43.33%** | **—** |
| **Ours: RuAG** | **Mistral Small** | **54.20%** | **49.65%** | **51.82%** | **+8.49%** |

### Key Observations

**1. RuAG improvement is consistent with the paper**

| Model | RuAG F1 Improvement |
|-------|-------------------|
| Paper: GPT-3.5 | +7.69% |
| **Ours: Mistral Small** | **+8.49%** |
| Paper: GPT-4 | +13.48% |

Our improvement falls between GPT-3.5 and GPT-4, which is expected given Mistral Small 24B sits between them in capability.

**2. Same Precision-Recall tradeoff pattern**

| | Precision Change | Recall Change |
|--|-----------------|---------------|
| Paper: GPT-4 RuAG | -8.33% | +24.16% |
| **Ours: RuAG** | **-8.43%** | **+16.52%** |

Both show the same pattern: rules help the LLM find more correct triples (higher Recall) at the cost of some additional false positives (lower Precision).

**3. Absolute F1 difference from paper**

| | RuAG F1 |
|--|---------|
| Paper: GPT-4 RuAG | 60.42% |
| **Ours: Mistral Small RuAG** | **51.82%** |
| Gap | -8.60% |

This gap is due to:
- Model capability differences (Mistral Small 24B vs GPT-4)
- Minor test set filtering differences (we evaluate 93 docs, paper evaluates ~97)
- Our prompt is adapted from but not identical to the paper's original

---

## Per-Relation Breakdown

| Relation | Van. Prec | Van. Rec | Van. F1 | RuAG Prec | RuAG Rec | RuAG F1 | F1 Change |
|----------|-----------|----------|---------|-----------|----------|---------|-----------|
| citizen_of | 80.56% | 13.81% | 23.58% | 61.42% | 57.62% | 59.46% | **+35.88%** |
| agent_of | 64.29% | 6.87% | 12.41% | 45.76% | 61.83% | 52.60% | **+40.19%** |
| citizen_of-x | 82.14% | 10.27% | 18.25% | 62.29% | 48.66% | 54.64% | **+36.39%** |
| gpe0 | 61.70% | 48.33% | 54.21% | 81.90% | 71.67% | 76.44% | **+22.23%** |
| vs | 35.00% | 19.44% | 25.00% | 44.44% | 33.33% | 38.10% | +13.10% |
| member_of | 55.48% | 43.22% | 48.59% | 57.42% | 60.30% | 58.82% | **+10.23%** |
| based_in0 | 64.89% | 37.95% | 47.89% | 49.81% | 57.14% | 53.22% | +5.33% |
| head_of | 46.51% | 58.82% | 51.95% | 53.42% | 57.35% | 55.32% | +3.37% |
| in0 | 81.37% | 72.78% | 76.83% | 79.75% | 72.22% | 75.80% | -1.03% |
| in0-x | 83.93% | 53.11% | 65.05% | 81.32% | 41.81% | 55.22% | -9.83% |

**Key takeaway**: The relations with the largest F1 improvements (citizen_of +35.88%, agent_of +40.19%, citizen_of-x +36.39%) are exactly the relations targeted by the highest-precision MCTS-discovered rules. For example, the rule "head_of_state → citizen_of" (precision 96.89%) directly helps the LLM find citizen_of triples it would otherwise miss. The Vanilla Recall for citizen_of was only 13.81% — the LLM barely found any on its own — but with rules, Recall jumped to 57.62%.

Relations without corresponding rules (in0, in0-x) show minimal or slightly negative change, which is expected since no rules target them.

---

## Project Structure

```
week04_group_work_finalVersion/
├── main.py                      # Main entry point (all steps)
├── ruag.db                      # SQLite database (data, predictions, rules)
├── writeup.md                   # Write-up for Part 2
├── dataset/
│   ├── relations_dict.json      # 20 relation type definitions
│   └── entity_relations_pairs/
│       ├── train/               # 702 training documents
│       └── test/                # 100 test documents (93 after filtering)
├── prompts/
│   ├── vanilla_prompt.txt       # Part 1: Vanilla LLM prompt
│   └── ruag_prompt.txt          # Part 2: RuAG prompt (with rules)
├── src/
│   ├── data_loader.py           # Load DWIE data into SQLite
│   ├── baseline_vanilla.py      # Part 1: Vanilla LLM inference
│   ├── baseline_ruag.py         # Part 2: RuAG inference (rules in prompt)
│   ├── evaluation.py            # Compute Precision, Recall, F1
│   ├── mcts.py                  # MCTS core algorithm (UCT)
│   ├── rule_node.py             # MCTS node for relation rule search
│   ├── rule_evaluator.py        # Evaluate rule precision on training data
│   └── rule_search.py           # Orchestrate MCTS rule discovery pipeline
└── README.md
```

---

## Dependencies

```bash
pip install openai
```

Only one external package. `sqlite3`, `json`, `re`, `collections`, `math`, `random` are all built into Python.

---

## How to Run

### Step 1: Load Data
```bash
python main.py --step load
```

This reads all 802 JSON files and creates `ruag.db` containing:

| Table | Contents | Rows |
|-------|----------|------|
| `documents` | Article content + train/test split | 802 |
| `entities` | All entities per document | 10,829 |
| `relations` | Ground truth relation triples | 19,493 |
| `relation_types` | 20 relation definitions | 20 |
| `filtered_docs` | 7 docs excluded (violate GPT protocol) | 7 |
| `predictions` | LLM outputs (populated in Steps 2–3) | — |
| `rules` | MCTS-discovered logic rules (populated in Step 3) | — |

### Step 2: Part 1 — Vanilla Baseline
```bash
python main.py --step vanilla --provider mistral --model mistral-small-latest --api-key YOUR_MISTRAL_KEY
python main.py --step evaluate
```

### Step 3: Part 2 — RuAG (Rule-Augmented Generation)
```bash
# MCTS Rule Search (no API key needed, pure computation on training data)
python main.py --step search_rules

# Run RuAG inference (rules injected into prompt)
python main.py --step ruag --provider mistral --model mistral-small-latest --api-key YOUR_MISTRAL_KEY
python main.py --step evaluate_ruag
```

### Supported API Providers

| Provider | Model | Free Tier | Base URL |
|----------|-------|-----------|----------|
| Mistral | mistral-small-latest | Yes (1B tokens/month) | api.mistral.ai |
| Groq | llama-3.1-8b-instant | Yes (500K tokens/day) | api.groq.com |
| Gemini | gemini-2.5-flash | Limited (20 RPD) | generativelanguage.googleapis.com |
| OpenAI | gpt-4 | No (paid) | api.openai.com |

Switching providers requires only changing the `--provider` and `--model` flags. No code changes needed.

---

## Implementation Details

### LLM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0 | Deterministic output (paper Table A5) |
| Top-p | 1 | No nucleus sampling truncation (paper Table A5) |
| Max tokens | 2000 | Sufficient for documents with up to 72 relation triples |

### Prompt Design

**Vanilla Prompt (Part 1)** provides:
1. **Task description** — extract relation triples from a document
2. **20 relation type definitions** with examples (from `relations_dict.json`)
3. **One worked example** demonstrating the output format, including `-x` relation variants and `gpe0`
4. **Entity list and document content** for the specific test article
5. **Explicit instruction** to use exact entity names from the provided list

The Vanilla prompt does **not** include any demonstrations (ICL), retrieved examples (RAG), or logic rules.

**RuAG Prompt (Part 2)** adds on top of the Vanilla prompt:
1. **MCTS-discovered logic rules** translated into natural language (e.g., "If A has relation head_of_state with B, then A and B have relation citizen_of")
2. **Instruction to apply rules** systematically after extracting base triples
3. **Instruction to distinguish** rule-derived triples from directly extracted ones
4. **Chain-of-thought reasoning** ("Let's think step by step") and brief reason for each extraction
5. **Anti-duplicate instruction** to prevent redundant outputs
6. **Extended example** showing both direct extraction and rule-derived triples with explanations

### Response Parsing

The parser extracts `(entity1, relation, entity2)` triples from the LLM output using:
- Regex pattern matching for parenthesized triples
- Format-level cleanup: stripping quotes, backticks, asterisks, and whitespace
- **Strict exact-match validation**: both entities must appear in the document's entity list, and the relation must be one of the 20 defined types

No semantic normalization or fuzzy matching is performed. This is identical to Part 1.

### Filtered Documents

Following the original implementation, 7 test documents are excluded from evaluation because they violate GPT processing protocols or have empty ground truth:

`DW_16083654`, `DW_17347807`, `DW_17736433`, `DW_18751636`, `DW_19210651`, `DW_39718698`, `DW_44141017`

This leaves **93 active test documents**.

### Data Management

All data resides in a single `ruag.db` SQLite database. No intermediate CSV, JSON, or pickle files are created during processing. The database schema supports both Part 1 (baseline) and Part 2 (MCTS rules) workflows.

---

## Method

### Part 1: Vanilla Baseline
The LLM reads each test document with an entity list and 20 relation type definitions, then predicts relation triples. No external knowledge or rules are provided. This establishes the baseline performance of the LLM alone.

### Part 2: RuAG (Three Phases)

**Phase 1 — Logic Rule Search Formulation (Paper Section 3.1):**
- Define each of the 20 relation types as a potential target predicate
- Remove irrelevant body predicates (vs, appears_in, player_of) as identified by the LLM in the paper
- Filter candidate body predicates by entity co-occurrence with target triples to reduce combinatorial search space

**Phase 2 — Logic Rule Search with MCTS (Paper Section 3.2):**
- For each target relation, run MCTS to discover first-order logic rules
- State: partial rule (list of body predicates)
- Action: add one body predicate to the rule
- Reward: rule precision evaluated on all training data
- Terminal condition: rule length >= 2 or precision >= 0.9
- UCT formula with exploration weight C = 0.7
- Hyperparameters follow Paper Table A5 (Relation Extraction column)

**Phase 3 — Learned-Rule-Augmented Generation (Paper Section 3.3):**
- Clean discovered rules: remove duplicates, filter precision > 0.5, remove redundant subsets
- Translate rules into natural language
- Inject translated rules into the LLM prompt alongside relation type definitions
- Run LLM inference on test documents with the rule-augmented prompt
- The LLM extracts triples from the document AND applies rules to derive additional triples

### Example Discovered Rules (from MCTS)

| Body Predicate(s) | Target | Precision | Meaning |
|-------------------|--------|-----------|---------|
| minister_of | agent_of | 99.28% | If someone is minister of a country, they are also an agent of it |
| head_of_state | agent_of | 98.76% | If someone is head of state, they represent that country |
| head_of_state | citizen_of | 96.89% | If someone is head of state, they are a citizen of that country |
| head_of_gov-x | citizen_of-x | 97.98% | If someone governs a country (nominal), they are a citizen (nominal) |
| head_of | member_of | 100% | If someone leads an organization, they are a member of it |
| agency_of | based_in0 | 98.20% | If an org is an agency of a country, it is based in that country |
| minister_of | citizen_of | 90.65% | If someone is minister of a country, they are a citizen of it |

### Key Design Decisions
- **SQLite-only storage**: All data, predictions, and rules stored in a single `ruag.db` database, as required by assignment constraints
- **Minimal dependencies**: Only `openai` package required (no pandas, numpy, tqdm, or RL frameworks)
- **MCTS from scratch**: Core UCT algorithm implemented without external libraries (~150 lines)
- **Multi-provider support**: OpenAI-compatible API interface works with Mistral, Groq, Gemini, and OpenAI

---

## Database Schema (SQLite)

| Table | Description |
|-------|-------------|
| documents | doc_id, content, split (train/test) |
| entities | doc_id, name |
| relations | doc_id, entity1, relation, entity2 (ground truth) |
| relation_types | relation, description (20 evaluated types) |
| predictions | doc_id, entity1, relation, entity2, method (vanilla/ruag) |
| rules | body_predicates, target, precision, description |
| filtered_docs | doc_id (7 documents excluded from evaluation) |

---

## Paper Reference

```bibtex
@inproceedings{zhang2025ruag,
    title={RuAG: Learned-rule-augmented Generation for Large Language Models},
    author={Yudi Zhang and Pei Xiao and Lu Wang and Chaoyun Zhang and Meng Fang 
            and Yali Du and Yevgeniy Puzyrev and Randolph Yao and Si Qin 
            and Qingwei Lin and Mykola Pechenizkiy and Dongmei Zhang 
            and Saravan Rajmohan and Qi Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=BpIbnXWfhL}
}
```
