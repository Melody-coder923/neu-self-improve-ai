from typing import Dict, List, Optional

from tqdm import tqdm

from src.data_preprocessing import Example
from src.inference_utils import run_with_retry
from src.llm_client import LLMClient
from src.prompting import build_vanilla_prompt, load_template


def run_vanilla(
    test_data: List[Example],
    relation_schema: List[str],
    llm: LLMClient,
    prompt_path: str,
    relation_descriptions: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    template = load_template(prompt_path)
    outputs = []
    for ex in tqdm(test_data, desc="Vanilla"):
        prompt = build_vanilla_prompt(template, relation_schema, ex, relation_descriptions)
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

