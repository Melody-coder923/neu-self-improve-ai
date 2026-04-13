from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.data_preprocessing import Example
from src.inference_utils import run_with_retry
from src.llm_client import LLMClient
from src.prompting import build_icl_prompt, load_template


def _build_index(train_data: List[Example]):
    corpus = [f"{' '.join(x.entities)}\n{x.document}" for x in train_data]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    mat = vectorizer.fit_transform(corpus)
    return vectorizer, mat


def _retrieve_support(
    query_example: Example,
    train_data: List[Example],
    vectorizer: TfidfVectorizer,
    train_mat,
    k_shots: int,
) -> List[Example]:
    query = f"{' '.join(query_example.entities)}\n{query_example.document}"
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, train_mat)[0]
    # Use MMR to keep both relevance and diversity in demonstrations.
    candidate_idx = np.argsort(-sims)[: min(len(train_data), max(k_shots * 8, k_shots))]
    selected: List[int] = []
    lambda_div = 0.8
    if len(candidate_idx) == 0:
        return []
    selected.append(int(candidate_idx[0]))
    while len(selected) < min(k_shots, len(candidate_idx)):
        best_i = None
        best_score = -1e9
        for i in candidate_idx:
            i = int(i)
            if i in selected:
                continue
            rel_score = float(sims[i])
            if selected:
                pair_sims = cosine_similarity(train_mat[i], train_mat[selected]).flatten()
                div_penalty = float(np.max(pair_sims))
            else:
                div_penalty = 0.0
            score = lambda_div * rel_score - (1 - lambda_div) * div_penalty
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
    return [train_data[i] for i in selected]


def run_icl(
    train_data: List[Example],
    test_data: List[Example],
    relation_schema: List[str],
    llm: LLMClient,
    prompt_path: str,
    k_shots: int = 3,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    template = load_template(prompt_path)
    support_pool = [x for x in train_data if x.relations]
    if len(support_pool) < max(1, k_shots):
        support_pool = train_data
    vectorizer, train_mat = _build_index(support_pool)
    outputs = []

    for ex in tqdm(test_data, desc="ICL"):
        support = _retrieve_support(
            query_example=ex,
            train_data=support_pool,
            vectorizer=vectorizer,
            train_mat=train_mat,
            k_shots=k_shots,
        )
        prompt = build_icl_prompt(
            template=template,
            relation_schema=relation_schema,
            support_examples=support,
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

