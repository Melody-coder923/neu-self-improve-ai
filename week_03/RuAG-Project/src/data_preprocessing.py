import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Triple = Tuple[str, str, str]


@dataclass
class Example:
    doc_id: str
    document: str
    entities: List[str]
    relations: List[Triple]


def _pick_first(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _normalize_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        if raw and all(isinstance(x, list) for x in raw):
            return " ".join(" ".join(str(tok) for tok in sent) for sent in raw).strip()
        return " ".join(str(x) for x in raw).strip()
    return str(raw).strip()


def _extract_entities(raw_doc: Dict[str, Any]) -> List[str]:
    entities: List[str] = []
    raw_entities = _pick_first(raw_doc, ["entities", "vertexSet", "entity_mentions"], default=[])
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if isinstance(item, list):
                for mention in item:
                    if isinstance(mention, dict):
                        name = _pick_first(mention, ["name", "text", "mention", "entity"])
                        if name:
                            entities.append(str(name).strip())
                            break
                continue
            if isinstance(item, dict):
                name = _pick_first(item, ["name", "text", "mention", "entity"])
                if name:
                    entities.append(str(name).strip())
            elif isinstance(item, str):
                entities.append(item.strip())

    # Fallback: if no explicit entity list is provided, derive candidates
    # from relation endpoints (common in DWIE-style simplified exports).
    if not entities:
        rels = _pick_first(raw_doc, ["relations", "labels", "triples"], default=[])
        if isinstance(rels, list):
            for rel in rels:
                if isinstance(rel, dict):
                    h = _pick_first(rel, ["head", "h", "subject", "subj", "entity1", "e1"])
                    t = _pick_first(rel, ["tail", "t", "object", "obj", "entity2", "e2"])
                    if isinstance(h, str) and h.strip():
                        entities.append(h.strip())
                    if isinstance(t, str) and t.strip():
                        entities.append(t.strip())
                elif isinstance(rel, (list, tuple)) and len(rel) == 3:
                    # Many datasets serialize as [head, tail, relation].
                    if isinstance(rel[0], str) and rel[0].strip():
                        entities.append(rel[0].strip())
                    if isinstance(rel[1], str) and rel[1].strip():
                        entities.append(rel[1].strip())

    seen = set()
    uniq = []
    for e in entities:
        if e and e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def _resolve_entity(ref: Any, entities: List[str]) -> str:
    if isinstance(ref, int) and 0 <= ref < len(entities):
        return entities[ref]
    if isinstance(ref, str):
        return ref.strip()
    return ""


_KNOWN_RELATION_LABELS = frozenset({
    "vs", "gpe0", "in0", "based_in0", "citizen_of", "based_in0-x",
    "citizen_of-x", "member_of", "in0-x", "agent_of", "head_of",
    "agency_of", "player_of", "agency_of-x", "head_of_state",
    "head_of_state-x", "appears_in", "head_of_gov", "head_of_gov-x",
    "minister_of",
})


def _looks_like_relation_label(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    s = x.strip()
    if not s:
        return False
    # Fast path: known DWIE relation labels (covers top-20 including "vs").
    if s in _KNOWN_RELATION_LABELS:
        return True
    # DWIE-like labels are typically snake_case with optional "-x"/digits.
    has_relation_shape = ("_" in s) or ("-x" in s) or any(ch.isdigit() for ch in s)
    # Entity mentions are often title-cased / contain spaces; relation labels
    # are usually lowercase tokens.
    return has_relation_shape and s.lower() == s


def _resolve_list_relation(rel: Any, entities: List[str]) -> Triple:
    if not isinstance(rel, (list, tuple)) or len(rel) != 3:
        return "", "", ""
    a_ref, b_ref, c_ref = rel[0], rel[1], rel[2]

    # Candidate interpretations:
    # 1) [head, relation, tail]
    # 2) [head, tail, relation] (common in DWIE exports)
    h1 = _resolve_entity(a_ref, entities)
    t1 = _resolve_entity(c_ref, entities)
    r1 = str(b_ref).strip() if b_ref is not None else ""

    h2 = _resolve_entity(a_ref, entities)
    t2 = _resolve_entity(b_ref, entities)
    r2 = str(c_ref).strip() if c_ref is not None else ""

    # If entity candidates are known, prefer the interpretation where both
    # endpoints map to candidate entities and relation does not.
    valid1 = bool(h1 and t1 and not _resolve_entity(b_ref, entities))
    valid2 = bool(h2 and t2 and not _resolve_entity(c_ref, entities))
    if valid1 and not valid2:
        return h1, r1, t1
    if valid2 and not valid1:
        return h2, r2, t2

    # Heuristic fallback by relation label shape.
    if _looks_like_relation_label(c_ref) and not _looks_like_relation_label(b_ref):
        return h2, r2, t2
    return h1, r1, t1


def _extract_relations(raw_doc: Dict[str, Any], entities: List[str]) -> List[Triple]:
    triples: List[Triple] = []
    rels = _pick_first(raw_doc, ["relations", "labels", "triples"], default=[])
    if isinstance(rels, list):
        for rel in rels:
            h = r = t = ""
            if isinstance(rel, dict):
                h_ref = _pick_first(rel, ["head", "h", "subject", "subj", "entity1", "e1"])
                t_ref = _pick_first(rel, ["tail", "t", "object", "obj", "entity2", "e2"])
                r_ref = _pick_first(rel, ["relation", "r", "predicate", "label", "type"])
                h = _resolve_entity(h_ref, entities)
                t = _resolve_entity(t_ref, entities)
                r = str(r_ref).strip() if r_ref is not None else ""
            elif isinstance(rel, (list, tuple)) and len(rel) == 3:
                h, r, t = _resolve_list_relation(rel, entities)
            if h and r and t:
                triples.append((h, r, t))

    seen = set()
    uniq = []
    for tri in triples:
        if tri not in seen:
            seen.add(tri)
            uniq.append(tri)
    return uniq


def parse_raw_doc(raw_doc: Dict[str, Any], idx: int) -> Example:
    doc_id = str(
        _pick_first(
            raw_doc,
            ["id", "doc_id", "document_id", "__doc_id"],
            default=f"doc_{idx:06d}",
        )
    )
    text = _normalize_text(_pick_first(raw_doc, ["document", "text", "content", "context", "sents"], default=""))
    entities = _extract_entities(raw_doc)
    triples = _extract_relations(raw_doc, entities)
    return Example(doc_id=doc_id, document=text, entities=entities, relations=triples)


def examples_from_raw_records(rows: Sequence[Dict[str, Any]]) -> List[Example]:
    data: List[Example] = []
    for i, row in enumerate(rows, start=1):
        ex = parse_raw_doc(row, idx=i)
        # Keep every document that has text, even if entities/relations are
        # empty (consistent with official RuAG which loads all 702 train docs).
        if ex.document:
            data.append(ex)
    return data


def load_any_json(path: str) -> List[Dict[str, Any]]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if fp.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    data = json.loads(fp.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "documents", "docs", "samples", "instances"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in: {path}")

