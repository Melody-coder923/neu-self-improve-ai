from typing import List, Tuple

from src.llm_client import LLMClient, parse_triples
from src.postprocess import sanitize_triples


Triple = Tuple[str, str, str]


def run_with_retry(
    llm: LLMClient,
    prompt: str,
    relation_schema: List[str],
    entities: List[str],
) -> Tuple[str, List[Triple]]:
    raw = llm.generate(prompt)
    pred = sanitize_triples(
        parse_triples(raw),
        relation_schema=relation_schema,
        entities=entities,
    )
    if pred:
        return raw, pred

    retry_prompt = (
        f"{prompt}\n\n"
        "Retry format instruction:\n"
        'Return ONLY a JSON array of triples, e.g. [["entity1","relation","entity2"]].\n'
        "If no valid relation exists, return [] only."
    )
    raw_retry = llm.generate(retry_prompt)
    pred_retry = sanitize_triples(
        parse_triples(raw_retry),
        relation_schema=relation_schema,
        entities=entities,
    )
    if pred_retry:
        return raw_retry, pred_retry
    return raw, pred
