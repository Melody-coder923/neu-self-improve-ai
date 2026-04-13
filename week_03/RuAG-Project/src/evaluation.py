from typing import Iterable, List, Set, Tuple


Triple = Tuple[str, str, str]


def normalize_triple(triple: Triple) -> Triple:
    """Lowercase + strip normalization (lenient, NOT the official behavior)."""
    h, r, t = triple
    return h.strip().lower(), r.strip().lower(), t.strip().lower()


def exact_triple(triple: Triple) -> Triple:
    """Exact string (official RuAG behaviour — case-sensitive)."""
    return triple[0].strip(), triple[1].strip(), triple[2].strip()


def to_set(triples: Iterable[Triple], exact: bool = True) -> Set[Triple]:
    fn = exact_triple if exact else normalize_triple
    return {fn(x) for x in triples}


def precision_recall_f1(
    gold_all: List[List[Triple]],
    pred_all: List[List[Triple]],
    exact_match: bool = True,
):
    """Compute micro-averaged Precision / Recall / F1.

    When *exact_match* is True (default, aligned with official RuAG),
    comparison is case-sensitive.  Set to False for lenient lowercase
    matching.
    """
    tp = fp = fn = 0
    for gold, pred in zip(gold_all, pred_all):
        gs = to_set(gold, exact=exact_match)
        ps = to_set(pred, exact=exact_match)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

