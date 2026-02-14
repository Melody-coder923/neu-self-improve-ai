# Part 1: Baseline Methods for Document-Level Relation Extraction

## 1. Introduction

This report presents Part 1 of our reproduction study of the RuAG (Learned-Rule-Augmented Generation) framework (Zhang et al., 2025). The goal of Part 1 is to implement and evaluate three inference-only baselines for document-level relation extraction (RE) on the DWIE dataset, establishing reference performance levels before introducing rule-augmented methods in Part 2.

The three baselines are:
- **Vanilla**: Direct prompting — the LLM receives only the document, entity list, and relation schema, with no demonstrations or retrieved context.
- **ICL (In-Context Learning)**: Few-shot prompting — the LLM additionally receives *k* labeled training examples selected via TF-IDF similarity and MMR diversity.
- **RAG (Retrieval-Augmented Generation)**: Similar to ICL, but examples are retrieved based on document-level similarity to provide more relevant context.

## 2. Experimental Setup

### 2.1 Dataset

We use the **DWIE** (Deutsche Welle corpus for Information Extraction) dataset, following the exact train/test split from the official RuAG repository:

| Split | Documents | Notes |
|-------|-----------|-------|
| Train | 702 | Used for ICL example selection and RAG retrieval |
| Test  | 93  | After filtering documents with no in-schema relations (100 → 93) |

The relation schema consists of the **top-20** most frequent relation types from the official RuAG configuration (e.g., `based_in0`, `citizen_of`, `head_of`, `member_of`, etc.), covering **2,180 gold triples** in the test set.

### 2.2 Model

All baselines use **Qwen2.5-7B-Instruct** as the backbone LLM, loaded locally in FP16 precision on a single NVIDIA V100 GPU (32 GB). Generation uses greedy decoding (temperature=0) with a maximum of 900 new tokens per example.

### 2.3 Prompting Strategy

Each prompt includes:
- A task instruction for relation extraction
- The full relation schema with official descriptions (from `relations_dict.json`)
- The entity list extracted from the document
- The document text (truncated to 6,000 tokens if necessary)
- For ICL/RAG: *k*=5 labeled demonstrations or retrieved cases

### 2.4 Evaluation

We report **micro-averaged Precision, Recall, and F1** over all test documents. Matching is **case-sensitive** (aligned with official RuAG evaluation). Gold triples are filtered to include only relations within the 20-relation schema.

## 3. Results

### 3.1 Quantitative Results

| Method  | Precision | Recall | F1     | TP  | FP  | FN   |
|---------|-----------|--------|--------|-----|-----|------|
| Vanilla | 0.3705    | 0.1706 | 0.2337 | 372 | 632 | 1808 |
| ICL     | —         | —      | —      | —   | —   | —    |
| RAG     | —         | —      | —      | —   | —   | —    |

> *Note: ICL and RAG results will be updated upon completion.*

### 3.2 Analysis

**Vanilla Baseline (F1 = 23.37%)**

The Vanilla baseline achieves a precision of 37.05%, indicating that roughly one in three predicted triples is correct. However, recall is considerably lower at 17.06%, meaning the model fails to identify the majority of gold relations. This is expected: without demonstrations or retrieved context, the model must rely entirely on its parametric knowledge and the prompt instructions to both understand the relation definitions and perform extraction — a challenging task for document-level RE where multiple entity pairs and relation types co-occur.

Key observations:
- **372 true positives** out of 2,180 gold triples shows the model can extract some relations correctly from the prompt alone.
- **632 false positives** suggest the model sometimes hallucinates plausible but incorrect triples, a known limitation of LLMs in structured extraction tasks.
- **1,808 false negatives** highlight the fundamental challenge: without examples showing the desired extraction behavior, the model under-extracts significantly.

**Expected trends (ICL and RAG):**

Based on the RuAG paper's findings, we expect:
- ICL to improve over Vanilla by providing the model with concrete examples of correct extractions, helping it learn the output format and extraction granularity.
- RAG to further improve by retrieving contextually similar documents, offering more relevant extraction patterns for each test instance.

## 4. Implementation Details

### 4.1 Data Management

All data is stored in a **single SQLite database** (`dwie.sqlite`). The database uses **normalized tables** (operate over DB via SQL):
- `documents`: doc_id, split, content
- `entities`: doc_id, name, ord_idx (per-document entity list)
- `relations`: doc_id, entity1, relation, entity2 (ground truth triples)
- `relation_types`: the 20-relation schema with official descriptions and ordering
- `filtered_docs`: document IDs excluded from evaluation (no in-schema relations)
- `run_metrics` / `run_predictions`: experiment results and per-document predictions

No intermediate data files are created during the pipeline; all intermediate data is stored as tables in the database, and results are queried on demand via `scripts/export_results.py`.

### 4.2 Dependencies

The project uses a minimal set of dependencies:
- `torch` (≥2.0) — tensor computation and GPU inference
- `transformers` — model loading and tokenization (Qwen2.5-7B-Instruct)
- `accelerate` — efficient multi-device model loading
- `scikit-learn` — TF-IDF vectorization and cosine similarity (for ICL/RAG retrieval)
- `PyYAML` — configuration parsing
- `tqdm` — progress tracking

No RL or agent frameworks are used.

### 4.3 Reproducibility

- All experiments run on Northeastern University's Explorer HPC cluster (NVIDIA V100 32GB GPU).
- Random seed and deterministic generation (temperature=0) ensure reproducible outputs.
- Input prompts are truncated to 6,000 tokens to prevent GPU out-of-memory errors on long documents.
- The code, configuration, and data loading scripts are available in the project repository.

## 5. Conclusion

Part 1 establishes baseline performance for document-level relation extraction on DWIE using three inference-only strategies. The Vanilla baseline (F1 = 23.37%) provides a clear lower bound, demonstrating that direct prompting alone is insufficient for this task. ICL and RAG baselines (results pending) are expected to show progressive improvement, validating the benefit of in-context demonstrations and retrieval augmentation — and motivating the rule-augmented approach introduced in Part 2.

## References

- Zhang, Y., et al. (2025). *RuAG: Learned-Rule-Augmented Generation for Large Language Models*. ICLR 2025.
- Zaporojets, K., et al. (2021). *DWIE: An Entity-centric Dataset for Multi-task Document-level Information Extraction*. 

