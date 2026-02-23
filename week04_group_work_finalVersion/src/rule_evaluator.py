"""
rule_evaluator.py - Evaluate precision of candidate logic rules on training data.

Given a rule like [r1, r2] -> target, this evaluator checks how often
the rule holds true across all training documents.

For single-predicate rules (len=1):
    For each (X, r1, Y) in training data,
    check if (X, target, Y) also exists.

For chain rules (len=2):
    For each (X, r1, Y) and (Y, r2, Z) in training data,
    check if (X, target, Z) also exists.

All data is loaded from the SQLite database (no intermediate files).
"""

import sqlite3
from collections import defaultdict


class RuleEvaluator:
    """Evaluates logic rule precision on training relation triples."""

    def __init__(self, db_path):
        self.db_path = db_path
        # relation -> list of (entity1, entity2) triples across all train docs
        self.relation_to_triples = defaultdict(list)
        self._load_data()

    def _load_data(self):
        """Load all training relation triples from SQLite into memory."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT r.entity1, r.relation, r.entity2
            FROM relations r
            JOIN documents d ON r.doc_id = d.doc_id
            WHERE d.split = 'train'
        """).fetchall()
        conn.close()

        # Build relation -> triples mapping
        # Also filter self-referencing triples (entity1 == entity2)
        triples_set = set()
        for e1, rel, e2 in rows:
            if e1 != e2:
                triples_set.add((e1, rel, e2))

        for e1, rel, e2 in triples_set:
            self.relation_to_triples[rel].append((e1, e2))

    def evaluate_precision(self, rule, target_relation):
        """Evaluate precision of a rule against training data.

        Args:
            rule: list of body predicate relation names, e.g. ['head_of_gov']
                  or ['head_of_gov', 'in0']
            target_relation: the target relation to predict

        Returns:
            precision (float): correct_predictions / total_predictions
                               Returns 0 if no predictions made.
        """
        correct = 0
        total = 0

        target_set = set(self.relation_to_triples[target_relation])

        if len(rule) == 1:
            # Single predicate: (X, r1, Y) => (X, target, Y)
            r1 = rule[0]
            for x, y in self.relation_to_triples[r1]:
                total += 1
                if (x, y) in target_set:
                    correct += 1

        elif len(rule) == 2:
            # Chain rule: (X, r1, Y) + (Y, r2, Z) => (X, target, Z)
            r1, r2 = rule[0], rule[1]
            # Build index for r2: Y -> list of Z
            r2_index = defaultdict(list)
            for y, z in self.relation_to_triples[r2]:
                r2_index[y].append(z)

            for x, y in self.relation_to_triples[r1]:
                if y in r2_index:
                    for z in r2_index[y]:
                        total += 1
                        if (x, z) in target_set:
                            correct += 1

        if total == 0:
            return 0.0

        return correct / total
