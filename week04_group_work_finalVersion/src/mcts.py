"""
mcts.py - Monte Carlo Tree Search core algorithm.

Re-implemented from scratch following:
  - RuAG paper (ICLR 2025), Section 3.2
  - Week 3 lecture notes: UCT = MCTS + UCB1
  - Algorithm 7 (MCTS), Algorithm 8 (Select), Algorithm 9 (Expand),
    Algorithm 10 (Backpropagation)

No RL or agent framework used. Only standard library (collections, math, abc).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math


class Node(ABC):
    """Abstract base class for MCTS nodes.
    
    Each node represents a state in the search space.
    Subclasses must implement find_children, find_random_child,
    is_terminal, reward, __hash__, and __eq__.
    """

    @abstractmethod
    def find_children(self):
        """Return all possible successor states."""
        return set()

    @abstractmethod
    def find_random_child(self):
        """Return a random successor state (for simulation)."""
        return None

    @abstractmethod
    def is_terminal(self):
        """Return True if this node has no children (leaf/terminal)."""
        return True

    @abstractmethod
    def reward(self):
        """Return the reward for this terminal node (e.g., precision)."""
        return 0

    @abstractmethod
    def __hash__(self):
        return 0

    @abstractmethod
    def __eq__(self, other):
        return True


class MCTS:
    """Monte Carlo Tree Search with UCB1 (Upper Confidence Trees).
    
    UCT formula: UCT_j = Q_j / N_j + C * sqrt(2 * ln(N_parent) / N_j)
    
    Parameters:
        exploration_weight (float): C in UCT formula. Controls exploration
            vs exploitation tradeoff. Paper uses 0.7 for relation extraction.
    """

    def __init__(self, exploration_weight=0.7):
        self.Q = defaultdict(int)       # total reward of each node
        self.N = defaultdict(int)       # total visit count for each node
        self.children = {}              # children of each node
        self.exploration_weight = exploration_weight
        self.rules_last_reward = {}     # terminal node rule -> last reward

    def do_rollout(self, node):
        """One iteration of MCTS: select -> expand -> simulate -> backprop.
        
        This corresponds to Algorithm 7 in the lecture notes.
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def choose(self, node):
        """Choose the best child of node based on average reward.
        
        Used after all rollouts to pick the best action.
        """
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)

    def _select(self, node):
        """Algorithm 8: Select - traverse tree using UCT until unexplored node.
        
        Start at root, use UCT to pick children, until we reach a node
        that is not fully expanded (has unexplored children).
        """
        path = []
        while True:
            path.append(node)
            # If node not yet expanded or has no children, stop
            if node not in self.children or not self.children[node]:
                return path
            # Check if any children are unexplored
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # All children explored: descend using UCT
            node = self._uct_select(node)

    def _expand(self, node):
        """Algorithm 9: Expand - create children for a node."""
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """Simulation (rollout): random walk until terminal, return reward.
        
        At each step, pick a random child until we reach a terminal node,
        then return its reward (precision of the rule on training data).
        """
        while True:
            if node.is_terminal():
                reward = node.reward()
                # Record this terminal rule and its reward
                self.rules_last_reward[tuple(node.rule)] = reward
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        """Algorithm 10: Backpropagation - update Q and N along the path.
        
        For each node in the path from leaf to root:
            N(node) += 1
            Q(node) += reward
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        """Select child using UCT (Upper Confidence Trees) formula.
        
        UCT_j = Q_j/N_j + C * sqrt(ln(N_parent) / N_j)
        
        Balances exploitation (high average reward) with exploration
        (less visited nodes).
        """
        assert all(n in self.children for n in self.children[node])

        log_N_parent = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_parent / self.N[n]
            )

        return max(self.children[node], key=uct)

    def get_all_rules_last_reward(self):
        """Return all terminal rules sorted by reward (descending)."""
        return sorted(self.rules_last_reward.items(), key=lambda x: x[1], reverse=True)
