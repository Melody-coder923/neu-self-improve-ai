from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.data_preprocessing import Example


Triple = Tuple[str, str, str]


def load_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def format_relations(
    relations: Iterable[str],
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format relation labels for the prompt.

    If *relation_descriptions* is provided (mapping relation name → description),
    each label is accompanied by its official description (aligned with the
    official RuAG ``relationships.txt``).  Otherwise, fall back to a plain
    comma-separated list.
    """
    if relation_descriptions:
        lines = []
        for rel in relations:
            desc = relation_descriptions.get(rel, "")
            if desc:
                lines.append(f"    -'{rel}': {desc}")
            else:
                lines.append(f"    -'{rel}'")
        return "\n".join(lines)
    return ", ".join(relations)


def format_entities(entities: Iterable[str]) -> str:
    return "; ".join(entities) + "."


def format_triples(triples: Iterable[Triple]) -> str:
    return "; ".join([f"({h}, {r}, {t})" for h, r, t in triples]) or "None"


def build_vanilla_prompt(
    template: str,
    relation_schema: List[str],
    example: Example,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    return (
        template.replace("{relationships}", format_relations(relation_schema, relation_descriptions))
        .replace("{entities}", format_entities(example.entities))
        .replace("{document}", example.document)
    )


def build_icl_prompt(
    template: str,
    relation_schema: List[str],
    support_examples: List[Example],
    query_example: Example,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    schema_set = set(relation_schema)
    shots = []
    for idx, ex in enumerate(support_examples, start=1):
        # Only show gold triples whose relation is in the schema (consistent
        # with the official RuAG setup).
        filtered_rels = [
            (h, r, t) for h, r, t in ex.relations if r in schema_set
        ]
        shots.append(
            f"Example {idx}\n"
            f"Entities: {format_entities(ex.entities)}\n"
            f"Document: {ex.document}\n"
            f"Gold Triples: {format_triples(filtered_rels)}"
        )

    return (
        template.replace("{relationships}", format_relations(relation_schema, relation_descriptions))
        .replace("{shots}", "\n\n".join(shots))
        .replace("{entities}", format_entities(query_example.entities))
        .replace("{document}", query_example.document)
    )


def build_rag_prompt(
    template: str,
    relation_schema: List[str],
    retrieved_examples: List[Tuple[Example, float]],
    query_example: Example,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    schema_set = set(relation_schema)
    refs = []
    for i, (ex, score) in enumerate(retrieved_examples, start=1):
        filtered_rels = [
            (h, r, t) for h, r, t in ex.relations if r in schema_set
        ]
        refs.append(
            f"Retrieved Case {i} (score={score:.4f})\n"
            f"Entities: {format_entities(ex.entities)}\n"
            f"Document: {ex.document}\n"
            f"Triples: {format_triples(filtered_rels)}"
        )

    return (
        template.replace("{relationships}", format_relations(relation_schema, relation_descriptions))
        .replace("{retrieved_cases}", "\n\n".join(refs))
        .replace("{entities}", format_entities(query_example.entities))
        .replace("{document}", query_example.document)
    )

