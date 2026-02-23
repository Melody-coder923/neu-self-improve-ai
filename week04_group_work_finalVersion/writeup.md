# Replicating RuAG: Learned-Rule-Augmented Generation for Relation Extraction

**Team members:** Yan Zhao, Jason Wang, Zhenyu Dai

**Paper:** RuAG (ICLR 2025) — *Learned-Rule-Augmented Generation for Large Language Models*  
**Task:** Document-level relation extraction on the DWIE dataset  
**Replication scope:** Part 2 — Proposed approach with MCTS-discovered logic rules

---

## 1. Introduction

RuAG improves LLM performance on structured prediction tasks by *mining first-order logic rules from training data* and injecting them into prompts. For relation extraction, instead of relying solely on the LLM’s internal knowledge, RuAG discovers patterns such as “if someone is head_of_state of a country, they are citizen_of that country,” then instructs the LLM to apply these rules during extraction. This boosts Recall without fine-tuning.

We replicate the full RuAG pipeline for the Relationship Extraction task: (1) MCTS-based logic rule search, (2) rule cleaning and translation to natural language, and (3) rule-augmented LLM inference and evaluation.

---

## 2. Method

**Phase 1 — Rule formulation:** For each of the 20 relation types (e.g., citizen_of, agent_of), we define it as a target predicate. Irrelevant body predicates (vs, appears_in, player_of) are removed as in the paper. Body predicate candidates are filtered by entity co-occurrence with target triples to reduce the search space.

**Phase 2 — MCTS rule search:** For each target relation, we run Monte Carlo Tree Search (UCT) to find high-precision logic rules. Each node is a partial rule (list of body predicates). Actions add one body predicate. Reward is rule precision on training data. Terminal conditions: rule length ≥ 2 (max_rule_length) or precision ≥ 0.9. We follow Table A5 (Relationship Extraction): exploration weight C = 0.7, reward = precision, max body predicates = 2.

**Phase 3 — Rule use in generation:** Discovered rules are cleaned (precision > 0.5, redundancy removal), translated to natural language (e.g., “If A has relation head_of_state with B, then A and B have relation citizen_of, with confidence 0.9689”), and injected into the LLM prompt. The LLM extracts triples from the document and applies rules to derive additional triples.

---

## 3. Experimental Setup

**Dataset:** DWIE (802 documents: 702 train, 100 test). Following the original implementation, 7 test documents are excluded (GPT protocol violations or empty ground truth), yielding 93 active test documents.

**Model:** Mistral Small 24B (mistral-small-latest) via API. Our results were obtained using `--provider mistral --model mistral-small-latest` (Mistral API). LLM parameters: temperature = 0, top-p = 1 (Table A5). We set max_tokens = 2000. Table A5 lists 1000; the original implementation sets 2000 in their extract_main.py but comments out the parameter in the API call, so it effectively has no limit. We explicitly use 2000 to avoid truncation while keeping outputs bounded.

**Evaluation:** Micro-averaged Precision, Recall, F1 over the 20 relation types. Predictions are exact matches against ground truth; entities must appear in the document’s entity list.

**Implementation:** SQLite stores all data, predictions, and rules. No RL or agent frameworks. Only the `openai` package is used for API calls. MCTS is implemented from scratch (~150 lines).

---

## 4. Results

| Method  | Precision | Recall | F1   | RuAG Δ |
|---------|-----------|--------|------|--------|
| Vanilla | 62.63%    | 33.13% | 43.33% | —     |
| RuAG    | 54.20%    | 49.65% | 51.82% | +8.49% |

**Replication consistency:** The RuAG F1 improvement (+8.49%) falls between the paper’s GPT-3.5 (+7.69%) and GPT-4 (+13.48%) improvements, consistent with Mistral Small’s intermediate capability.

**Precision–Recall tradeoff:** Like the paper, RuAG increases Recall (+16.52%) while reducing Precision (−8.43%), matching the pattern that rules help find more correct triples but add some false positives.

**Per-relation breakdown:** Largest F1 gains occur for relations with high-precision rules: citizen_of (+35.88%), agent_of (+40.19%), citizen_of-x (+36.39%). For example, the rule head_of_state → citizen_of (96.89% precision) raises Vanilla Recall from 13.81% to 57.62%. Relations without rules (e.g., in0) show little or slightly negative change.

**Example discovered rules:** minister_of → agent_of (99.28%), head_of_state → citizen_of (96.89%), agency_of → based_in0 (98.20%), head_of → member_of (100%).

---

## 5. Conclusion

We successfully replicate the RuAG approach for relation extraction. MCTS discovers high-precision logic rules from training data, and rule-augmented prompts yield a substantial F1 gain with the expected Precision–Recall tradeoff. The absolute F1 gap vs. the paper’s GPT-4 RuAG (51.82% vs. 60.42%) is explained by model differences (Mistral Small vs. GPT-4) and minor test-set filtering. Our implementation follows the paper’s Table A5 parameters and validates the core RuAG contribution.

---

**Reference:** Zhang et al., *RuAG: Learned-Rule-Augmented Generation for Large Language Models*, ICLR 2025.  
OpenReview: https://openreview.net/forum?id=BpIbnXWfhL
