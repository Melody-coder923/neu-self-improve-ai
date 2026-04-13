from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple


Triple = Tuple[str, str, str]


def _norm_text(x: str) -> str:
    return " ".join(str(x).strip().lower().replace("_", " ").split())


def _best_match(
    value_norm: str,
    key_to_value: Dict[str, str],
    fuzzy_cutoff: float,
) -> Optional[str]:
    if value_norm in key_to_value:
        return key_to_value[value_norm]

    # Substring match can recover truncated entity names.
    candidates = [
        (k, v)
        for k, v in key_to_value.items()
        if value_norm in k or k in value_norm
    ]
    if len(candidates) == 1:
        return candidates[0][1]

    best_key = ""
    best_score = -1.0
    for k in key_to_value.keys():
        score = SequenceMatcher(None, value_norm, k).ratio()
        if score > best_score:
            best_score = score
            best_key = k
    if best_key and best_score >= fuzzy_cutoff:
        return key_to_value[best_key]
    return None


def sanitize_triples(
    triples: Iterable[Triple],
    relation_schema: List[str],
    entities: List[str],
) -> List[Triple]:
    rel_map = {_norm_text(r): r for r in relation_schema}
    ent_map = {_norm_text(e): e for e in entities}
    out: List[Triple] = []
    seen = set()
    for h, r, t in triples:
        hn = _norm_text(h.strip("'\""))
        rn = _norm_text(r.strip("'\""))
        tn = _norm_text(t.strip("'\""))
        if not hn or not rn or not tn:
            continue
        rel = _best_match(rn, rel_map, fuzzy_cutoff=0.9)
        if rel is None:
            continue
        head = _best_match(hn, ent_map, fuzzy_cutoff=0.92)
        tail = _best_match(tn, ent_map, fuzzy_cutoff=0.92)
        if head is None or tail is None:
            continue
        tri = (head, rel, tail)
        key = (_norm_text(tri[0]), _norm_text(tri[1]), _norm_text(tri[2]))
        if key in seen:
            continue
        seen.add(key)
        out.append(tri)
    return out
