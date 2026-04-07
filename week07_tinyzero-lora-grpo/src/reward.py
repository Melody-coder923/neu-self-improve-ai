"""Layered Countdown rewards for GRPO (TRL-compatible signature)."""

from __future__ import annotations

import ast
import math
import operator
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .parsing import THINK_CLOSE, THINK_OPEN, format_ok_for_reward, parse_countdown_response

W_FORMAT = 0.5
W_EXPR = 0.5
W_MULTISET = 0.75
W_TARGET = 3.0

RESULT_TOL = 1e-5

_ALLOWED_BINOPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS: dict[type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class RewardBreakdown:
    format_score: float
    expr_score: float
    multiset_score: float
    target_score: float
    total: float
    solved: bool
    details: dict[str, Any]


def _as_number(n: ast.expr) -> Fraction | None:
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return Fraction(v)
        if isinstance(v, float):
            if not math.isfinite(v):
                return None
            return Fraction(v).limit_denominator(10**9)
    return None


def _eval_ast(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            return None
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if left is None or right is None:
            return None
        fn = _ALLOWED_BINOPS[type(node.op)]
        try:
            out = fn(left, right)
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
        if isinstance(out, Fraction):
            return out
        return None
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            return None
        v = _eval_ast(node.operand)
        if v is None:
            return None
        fn = _ALLOWED_UNARYOPS[type(node.op)]
        try:
            out = fn(v)
        except (ValueError, OverflowError):
            return None
        return out if isinstance(out, Fraction) else None
    if isinstance(node, ast.Constant):
        return _as_number(node)
    return None


def _collect_constants(node: ast.AST) -> list[Fraction]:
    out: list[Fraction] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and not isinstance(child.value, bool):
            n = _as_number(child)
            if n is not None:
                out.append(n)
    return out


def _normalize_nums(nums: list) -> list[Fraction]:
    acc: list[Fraction] = []
    for x in nums:
        if isinstance(x, bool):
            raise ValueError("bool")
        if isinstance(x, int):
            acc.append(Fraction(x))
        elif isinstance(x, float):
            acc.append(Fraction(x).limit_denominator(10**9))
        else:
            raise ValueError("bad num type")
    return acc


def _multiset_match(used: list[Fraction], available: list[Fraction]) -> bool:
    def key(f: Fraction) -> tuple:
        return (f.numerator, f.denominator)

    c_used = Counter(key(x) for x in used)
    c_av = Counter(key(x) for x in available)
    return c_used == c_av


def _result_matches_target(value: Fraction, target: float | int) -> bool:
    if isinstance(target, bool):
        return False
    if isinstance(target, int):
        t = Fraction(target)
    else:
        t = Fraction(target).limit_denominator(10**9)
    diff = abs(float(value - t))
    return diff <= RESULT_TOL


def compute_countdown_reward(
    completion: str,
    nums: list,
    target: float | int,
    *,
    think_open: str | None = None,
    think_close: str | None = None,
) -> RewardBreakdown:
    """Compute layered reward for one completion."""
    to = think_open if think_open is not None else THINK_OPEN
    tc = think_close if think_close is not None else THINK_CLOSE

    parsed = parse_countdown_response(completion, think_open=to, think_close=tc)
    details: dict[str, Any] = {}

    fmt_ok = format_ok_for_reward(parsed, raw_text=completion, think_close=tc)
    format_score = W_FORMAT if fmt_ok else 0.0
    details["format_ok"] = fmt_ok

    expr_score = 0.0
    multiset_score = 0.0
    target_score = 0.0
    solved = False

    if not fmt_ok or parsed.answer_inner is None:
        total = format_score
        return RewardBreakdown(
            format_score=format_score,
            expr_score=0.0,
            multiset_score=0.0,
            target_score=0.0,
            total=total,
            solved=False,
            details=details,
        )

    expr_s = parsed.answer_inner.strip()
    try:
        tree = ast.parse(expr_s, mode="eval")
    except SyntaxError:
        details["expr_error"] = "syntax"
        total = format_score
        return RewardBreakdown(
            format_score=format_score,
            expr_score=0.0,
            multiset_score=0.0,
            target_score=0.0,
            total=total,
            solved=False,
            details=details,
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.List, ast.Dict, ast.Lambda)):
            details["expr_error"] = "disallowed_ast"
            total = format_score
            return RewardBreakdown(
                format_score=format_score,
                expr_score=0.0,
                multiset_score=0.0,
                target_score=0.0,
                total=total,
                solved=False,
                details=details,
            )

    value = _eval_ast(tree)
    if value is None:
        details["expr_error"] = "eval"
        total = format_score
        return RewardBreakdown(
            format_score=format_score,
            expr_score=0.0,
            multiset_score=0.0,
            target_score=0.0,
            total=total,
            solved=False,
            details=details,
        )

    expr_score = W_EXPR
    details["value"] = float(value)

    try:
        norm_nums = _normalize_nums(list(nums))
    except ValueError:
        details["nums_error"] = True
        total = format_score + expr_score
        return RewardBreakdown(
            format_score=format_score,
            expr_score=expr_score,
            multiset_score=0.0,
            target_score=0.0,
            total=total,
            solved=False,
            details=details,
        )

    used_constants = _collect_constants(tree)
    if _multiset_match(used_constants, norm_nums):
        multiset_score = W_MULTISET
        details["multiset_ok"] = True
    else:
        details["multiset_ok"] = False
        details["used_constants"] = [float(x) for x in used_constants]
        details["expected"] = [float(x) for x in norm_nums]

    if multiset_score > 0 and _result_matches_target(value, target):
        target_score = W_TARGET
        solved = True
        details["solved"] = True
    else:
        details["solved"] = False

    total = format_score + expr_score + multiset_score + target_score
    return RewardBreakdown(
        format_score=format_score,
        expr_score=expr_score,
        multiset_score=multiset_score,
        target_score=target_score,
        total=total,
        solved=solved,
        details=details,
    )


def dummy_reward(
    prompts: list[str],
    completions: list[str],
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """Always 1.0 - for pipeline smoke test."""
    return [1.0] * len(completions)


def countdown_reward(
    prompts: list[str],
    completions: list[str],
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL GRPO reward: one float per (prompt, completion) row."""
    nums_batch = kwargs.get("nums")
    target_batch = kwargs.get("target")
    if nums_batch is None or target_batch is None:
        raise ValueError("countdown_reward requires `nums` and `target` in kwargs (dataset columns).")
    if len(completions) != len(nums_batch) or len(completions) != len(target_batch):
        raise ValueError("Length mismatch between completions, nums, and target.")

    out: list[float] = []
    for comp, ns, tg in zip(completions, nums_batch, target_batch, strict=True):
        bd = compute_countdown_reward(comp, list(ns), tg)
        out.append(bd.total)
    return out


def countdown_reward_with_breakdown(
    prompts: list[str],
    completions: list[str],
    completion_ids: list | None = None,
    **kwargs: Any,
) -> tuple[list[float], list[RewardBreakdown]]:
    """Same as countdown_reward but returns per-row breakdowns for logging."""
    nums_batch = kwargs.get("nums")
    target_batch = kwargs.get("target")
    if nums_batch is None or target_batch is None:
        raise ValueError("countdown_reward_with_breakdown requires `nums` and `target` in kwargs.")
    rewards: list[float] = []
    breakdowns: list[RewardBreakdown] = []
    for comp, ns, tg in zip(completions, nums_batch, target_batch, strict=True):
        bd = compute_countdown_reward(comp, list(ns), tg)
        rewards.append(bd.total)
        breakdowns.append(bd)
    return rewards, breakdowns
