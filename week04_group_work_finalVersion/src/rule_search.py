"""
rule_search.py - Orchestrate MCTS-based logic rule search.

This module implements all three phases of RuAG:
  Phase 1: LLM-based Logic Rule Search Formulation (Section 3.1)
    - Define target predicates (each relation type)
    - Filter impossible body predicates (remove vs, appears_in, player_of)
    - Preprocess: only keep body predicates that share entities with target
  Phase 2: Logic Rule Search with MCTS (Section 3.2)
    - For each target relation, run MCTS to discover rules
    - Collect terminal rules with precision > 0.5
  Phase 3: Clean & translate rules (Section 3.3)
    - Remove duplicates and low-quality rules
    - Translate to natural language
    - Store in SQLite rules table

Hyperparameters (from paper Table A5, Relation Extraction):
    - Total rollouts per target: 500 (EPOCH=100, 5 iterations of choose)
    - Reward metric: Precision
    - Maximum body predicates: 2
    - Terminal condition: Precision > 0.9
    - Exploration weight (C): 0.7
"""

import sqlite3
from collections import defaultdict

from mcts import MCTS
from rule_node import RelationRuleNode
from rule_evaluator import RuleEvaluator


# Phase 1: Relations that LLM identified as irrelevant body predicates
# (Paper Section 4.1: "utilized the LLM to identify and eliminate 15%
#  of the relationships (i.e. appears_in, vs and player_of)")
REMOVAL_PREDICATES = {"vs", "appears_in", "player_of"}

# MCTS hyperparameters (Paper Table A5)
ROLLOUTS_PER_ITERATION = 100   # EPOCH in original code
EXPLORATION_WEIGHT = 0.7
MAX_RULE_LENGTH = 2
PRECISION_THRESHOLD = 0.9
MIN_RULE_PRECISION = 0.5       # only keep rules with precision > 0.5


def preprocess_relations(all_triples, target_relation):
    """Phase 1: Filter body predicate candidates for a given target relation.
    
    Only keep relations whose triples share entity positions with the
    target relation's triples. Specifically, for target triple (X, target, Z),
    keep relation r if any triple (X, r, Y) or (Y, r, Z) exists.
    
    This reduces the search space significantly.
    
    Args:
        all_triples: set of (entity1, relation, entity2) triples
        target_relation: the target predicate
        
    Returns:
        set of relation names that are valid body predicate candidates
    """
    # Collect all entities involved in target triples
    target_triples = [(x, r, z) for x, r, z in all_triples if r == target_relation]

    filtered_relations = set()
    for x, r, y in all_triples:
        for tx, tr, tz in target_triples:
            # Either X or Y appears in the target triple's X or Z position
            if x == tx or y == tz:
                filtered_relations.add(r)

    filtered_relations.discard(target_relation)
    return filtered_relations


def extract_rules_for_target(target_relation, all_relations, potential_relations,
                             evaluator):
    """Run MCTS to discover logic rules for one target relation.
    
    Args:
        target_relation: the target predicate to search rules for
        all_relations: set of all relation types
        potential_relations: filtered candidate body predicates
        evaluator: RuleEvaluator instance
        
    Returns:
        list of (rule_tuple, precision) pairs
    """
    # Create root node (empty rule)
    root = RelationRuleNode(
        rule=[],
        target_relation=target_relation,
        all_relations=all_relations,
        potential_relations=potential_relations,
        evaluator=evaluator,
        max_rule_length=MAX_RULE_LENGTH,
        precision_threshold=PRECISION_THRESHOLD,
    )

    tree = MCTS(exploration_weight=EXPLORATION_WEIGHT)
    board = root

    # Iteratively: do rollouts, then choose best child, repeat
    while True:
        if board.is_terminal():
            break

        for _ in range(ROLLOUTS_PER_ITERATION):
            tree.do_rollout(board)

        board = tree.choose(board)

        if board.is_terminal():
            break

    # Collect all discovered rules and their rewards
    return tree.get_all_rules_last_reward()


def clean_rules(all_rules):
    """Phase 3 step 1: Clean discovered rules.
    
    - Remove rules with precision <= MIN_RULE_PRECISION
    - Remove redundant rules: if rule A's body is a subset of rule B's body
      and A has higher precision, remove B
    
    Args:
        all_rules: list of dicts with 'body', 'target', 'precision'
        
    Returns:
        filtered list of rule dicts
    """
    # Filter by minimum precision
    rules = [r for r in all_rules if r['precision'] > MIN_RULE_PRECISION]

    # Remove redundant rules
    cleaned = []
    for r in rules:
        r_body = set(r['body'])
        is_redundant = False
        for other in rules:
            if other is r:
                continue
            other_body = set(other['body'])
            # If other's body is a subset of r's body and other has higher precision
            if other_body < r_body and other['target'] == r['target']:
                if other['precision'] >= r['precision']:
                    is_redundant = True
                    break
        if not is_redundant:
            cleaned.append(r)

    return cleaned


def translate_rule_to_text(body_predicates, target_predicate, confidence):
    """Phase 3 step 2: Translate a logic rule into natural language.
    
    Follows the exact format from rules2text.py in the original repo:
    "If A has relation r1 with B and B has relation r2 with C,
     then A and C have relation target, with confidence 0.9500"
    
    Args:
        body_predicates: list of body relation names
        target_predicate: target relation name
        confidence: precision value
        
    Returns:
        str: natural language description of the rule
    """
    entity_count = len(body_predicates) + 1
    entities = [chr(65 + i) for i in range(entity_count)]  # A, B, C...

    conditions = []
    for i, predicate in enumerate(body_predicates):
        subject = entities[i]
        object_ = entities[i + 1]
        conditions.append(f"{subject} has relation {predicate} with {object_}")

    condition_str = " and ".join(conditions)
    conclusion = f"{entities[0]} and {entities[-1]} have relation {target_predicate}"

    description = (
        f"If {condition_str}, then {conclusion}, "
        f"with confidence {confidence:.4f}"
    )
    return description


def run_rule_search(db_path):
    """Main entry point: run the complete MCTS rule search pipeline.
    
    1. Load training data from SQLite
    2. For each target relation, run MCTS to discover rules
    3. Clean and translate rules
    4. Store rules in SQLite rules table
    
    Args:
        db_path: path to the SQLite database
    """
    conn = sqlite3.connect(db_path)

    # ── Phase 1: Formulation ──────────────────────────────────────────
    print("Phase 1: Logic Rule Search Formulation")
    print(f"  Removing irrelevant body predicates: {REMOVAL_PREDICATES}")

    # Load all training triples
    rows = conn.execute("""
        SELECT r.entity1, r.relation, r.entity2
        FROM relations r
        JOIN documents d ON r.doc_id = d.doc_id
        WHERE d.split = 'train'
    """).fetchall()

    # Build triples set, filter self-references and removal predicates
    all_triples = set()
    for e1, rel, e2 in rows:
        if e1 != e2 and rel not in REMOVAL_PREDICATES:
            all_triples.add((e1, rel, e2))

    # Get all unique relation types in training data (after removal)
    all_relations = set(triple[1] for triple in all_triples)
    print(f"  Relations after filtering: {len(all_relations)}")

    # Initialize evaluator (loads training data from SQLite)
    evaluator = RuleEvaluator(db_path)

    # ── Phase 2: MCTS Rule Search ─────────────────────────────────────
    print("\nPhase 2: MCTS Rule Search")
    print(f"  Rollouts per iteration: {ROLLOUTS_PER_ITERATION}")
    print(f"  Exploration weight: {EXPLORATION_WEIGHT}")
    print(f"  Max rule length: {MAX_RULE_LENGTH}")
    print(f"  Precision threshold: {PRECISION_THRESHOLD}")
    print()

    all_discovered_rules = []

    for i, target_relation in enumerate(sorted(all_relations)):
        # Preprocess: filter candidates for this target
        potential_relations = preprocess_relations(all_triples, target_relation)
        potential_relations -= REMOVAL_PREDICATES  # ensure removed

        if not potential_relations:
            print(f"  [{i+1}/{len(all_relations)}] {target_relation}: no candidates, skipping")
            continue

        print(f"  [{i+1}/{len(all_relations)}] {target_relation}: "
              f"{len(potential_relations)} candidates...", end=" ", flush=True)

        # Run MCTS
        path_rewards = extract_rules_for_target(
            target_relation, all_relations, potential_relations, evaluator
        )

        # Collect rules with positive precision
        count = 0
        for rule_tuple, precision in path_rewards:
            if precision > 0:
                all_discovered_rules.append({
                    'body': list(rule_tuple),
                    'target': target_relation,
                    'precision': precision,
                })
                count += 1

        print(f"found {count} rules")

    print(f"\nTotal rules discovered: {len(all_discovered_rules)}")

    # ── Phase 3: Clean, translate, store ──────────────────────────────
    print("\nPhase 3: Clean, Translate, and Store Rules")

    # Clean rules
    cleaned_rules = clean_rules(all_discovered_rules)
    print(f"  Rules after cleaning (precision > {MIN_RULE_PRECISION}): {len(cleaned_rules)}")

    # Filter: only keep rules targeting the 20 evaluated relation types
    eval_relations = set(
        row[0] for row in conn.execute("SELECT relation FROM relation_types").fetchall()
    )
    cleaned_rules = [r for r in cleaned_rules if r['target'] in eval_relations]
    print(f"  Rules targeting evaluated relations: {len(cleaned_rules)}")

    # Clear old rules and insert new ones
    conn.execute("DELETE FROM rules")

    for rule in cleaned_rules:
        body_str = " , ".join(rule['body'])
        description = translate_rule_to_text(
            rule['body'], rule['target'], rule['precision']
        )
        conn.execute(
            "INSERT INTO rules (body_predicates, target, precision, description) VALUES (?, ?, ?, ?)",
            (body_str, rule['target'], rule['precision'], description)
        )

    conn.commit()

    # Print summary
    print(f"\n  Stored {len(cleaned_rules)} rules in SQLite:")
    print(f"  {'Body Predicates':<40} {'→ Target':<20} {'Precision':>10}")
    print(f"  {'-'*72}")
    for rule in sorted(cleaned_rules, key=lambda x: -x['precision']):
        body = " + ".join(rule['body'])
        print(f"  {body:<40} → {rule['target']:<20} {rule['precision']:>9.4f}")

    # Print natural language translations
    print(f"\n  Natural language rules (for LLM prompt):")
    for rule in sorted(cleaned_rules, key=lambda x: -x['precision']):
        desc = translate_rule_to_text(rule['body'], rule['target'], rule['precision'])
        print(f"    {desc}")

    conn.close()
    print(f"\nRule search complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCTS rule search")
    parser.add_argument("--db", default="../ruag.db", help="Path to SQLite database")
    args = parser.parse_args()
    run_rule_search(args.db)