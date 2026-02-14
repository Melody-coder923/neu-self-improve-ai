from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.data_preprocessing import Example
from src.inference_utils import run_with_retry
from src.llm_client import LLMClient
from src.prompting import build_rag_prompt, load_template


def _build_index(train_data: List[Example]):
    corpus = [f"{' '.join(x.entities)}\n{x.document}" for x in train_data]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    mat = vectorizer.fit_transform(corpus)
    return vectorizer, mat


def _retrieve_topk(
    query_text: str,
    train_data: List[Example],
    vectorizer: TfidfVectorizer,
    train_mat,
    top_k: int = 3,
) -> List[Tuple[Example, float]]:
    q = vectorizer.transform([query_text])
    sims = cosine_similarity(q, train_mat)[0]
    candidate_idx = np.argsort(-sims)[: min(len(train_data), max(top_k * 8, top_k))]
    selected: List[int] = []
    lambda_div = 0.8
    if len(candidate_idx) == 0:
        return []
    selected.append(int(candidate_idx[0]))
    while len(selected) < min(top_k, len(candidate_idx)):
        best_i = None
        best_score = -1e9
        for i in candidate_idx:
            i = int(i)
            if i in selected:
                continue
            rel_score = float(sims[i])
            pair_sims = cosine_similarity(train_mat[i], train_mat[selected]).flatten()
            div_penalty = float(np.max(pair_sims))
            score = lambda_div * rel_score - (1 - lambda_div) * div_penalty
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
    return [(train_data[i], float(sims[i])) for i in selected]


def run_rag(
    train_data: List[Example],
    test_data: List[Example],
    relation_schema: List[str],
    llm: LLMClient,
    prompt_path: str,
    top_k: int = 3,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    template = load_template(prompt_path)
    retrieval_pool = [x for x in train_data if x.relations]
    if len(retrieval_pool) < max(1, top_k):
        retrieval_pool = train_data
    vectorizer, train_mat = _build_index(retrieval_pool)
    outputs = []

    for ex in tqdm(test_data, desc="RAG"):
        retrieved = _retrieve_topk(
            query_text=f"{' '.join(ex.entities)}\n{ex.document}",
            train_data=retrieval_pool,
            vectorizer=vectorizer,
            train_mat=train_mat,
            top_k=top_k,
        )
        prompt = build_rag_prompt(
            template=template,
            relation_schema=relation_schema,
            retrieved_examples=retrieved,
            query_example=ex,
            relation_descriptions=relation_descriptions,
        )
        raw, pred = run_with_retry(
            llm=llm,
            prompt=prompt,
            relation_schema=relation_schema,
            entities=ex.entities,
        )
        outputs.append(
            {
                "id": ex.doc_id,
                "gold": ex.relations,
                "pred": pred,
                "raw": raw,
            }
        )
    return outputs

