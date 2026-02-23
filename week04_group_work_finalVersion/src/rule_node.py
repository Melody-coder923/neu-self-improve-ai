"""
rule_node.py - MCTS node for logic rule search in relation extraction.

Each node represents a partial logic rule:
    State:  a list of body predicates, e.g. ['head_of_gov', 'in0']
    Action: add one more body predicate to the rule
    Reward: precision of the rule on training data

Terminal conditions (from paper Table A5, relation extraction):
    - Rule length reaches max (2 for relation extraction)
    - Precision >= 0.9
    - No more candidate predicates to add
"""

import random
from mcts import Node


class RelationRuleNode(Node):
    """A node in the MCTS search tree for logic rule discovery.
    
    Attributes:
        rule (list): Current body predicates, e.g. ['head_of_gov']
        target_relation (str): The target predicate we're trying to predict
        all_relations (set): All possible relation types
        potential_relations (set): Filtered candidate body predicates
        evaluator (RuleEvaluator): Evaluates rule precision on training data
        max_rule_length (int): Maximum number of body predicates (default: 2)
        precision_threshold (float): Terminal if precision >= this (default: 0.9)
    """

    def __init__(self, rule, target_relation, all_relations, potential_relations,
                 evaluator, max_rule_length=2, precision_threshold=0.9):
        self.rule = rule
        self.target_relation = target_relation
        self.all_relations = all_relations
        self.potential_relations = potential_relations
        self.evaluator = evaluator
        self.max_rule_length = max_rule_length
        self.precision_threshold = precision_threshold
        self._is_terminal = None  # cached

    def find_children(self):
        """Generate all child nodes by adding one candidate predicate."""
        if self.is_terminal():
            return set()

        children = set()
        for rel in self.potential_relations:
            if rel not in self.rule:
                new_rule = self.rule + [rel]
                child = RelationRuleNode(
                    rule=new_rule,
                    target_relation=self.target_relation,
                    all_relations=self.all_relations,
                    potential_relations=self.potential_relations,
                    evaluator=self.evaluator,
                    max_rule_length=self.max_rule_length,
                    precision_threshold=self.precision_threshold,
                )
                children.add(child)
        return children

    def find_random_child(self):
        """Pick a random unexplored predicate and add it to the rule."""
        if self.is_terminal():
            return None
        candidates = [r for r in self.potential_relations if r not in self.rule]
        if not candidates:
            return None
        rel = random.choice(candidates)
        new_rule = self.rule + [rel]
        return RelationRuleNode(
            rule=new_rule,
            target_relation=self.target_relation,
            all_relations=self.all_relations,
            potential_relations=self.potential_relations,
            evaluator=self.evaluator,
            max_rule_length=self.max_rule_length,
            precision_threshold=self.precision_threshold,
        )

    def is_terminal(self):
        """Check if this node is terminal (no further expansion needed).
        
        Terminal conditions:
        1. Rule length >= max_rule_length (2 for relation extraction)
        2. Precision >= precision_threshold (0.9)
        3. No more candidate predicates available
        """
        if self._is_terminal is not None:
            return self._is_terminal

        # Condition 1: reached max rule length
        if len(self.rule) >= self.max_rule_length:
            self._is_terminal = True
            return True

        # Condition 3: no candidates left
        candidates = [r for r in self.potential_relations if r not in self.rule]
        if not candidates:
            self._is_terminal = True
            return True

        # Condition 2: precision already high enough (only check if rule non-empty)
        if len(self.rule) > 0:
            precision = self.evaluator.evaluate_precision(self.rule, self.target_relation)
            if precision >= self.precision_threshold:
                self._is_terminal = True
                return True

        self._is_terminal = False
        return False

    def reward(self):
        """Return the precision of the current rule on training data."""
        if not self.rule:
            return 0.0
        return self.evaluator.evaluate_precision(self.rule, self.target_relation)

    def __hash__(self):
        return hash((tuple(self.rule), self.target_relation))

    def __eq__(self, other):
        return (tuple(self.rule) == tuple(other.rule)
                and self.target_relation == other.target_relation)
