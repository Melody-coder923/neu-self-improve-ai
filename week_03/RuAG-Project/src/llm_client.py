import json
import os
import re
from typing import List, Tuple

Triple = Tuple[str, str, str]


class LLMClient:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
        provider: str = "openai",
    ):
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self.tokenizer = None
        self.model = None

        if self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "local":
            # Local mode loads a HuggingFace causal LM once.
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as e:
                raise RuntimeError(
                    "Local provider requires 'torch' and 'transformers'. "
                    "Install them before running local mode."
                ) from e

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'local'.")

    def generate(self, prompt: str) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful relation extraction assistant. "
                            "Return triples only, no extra explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""

        # Local HuggingFace generation path
        import torch

        messages = [
            {
                "role": "system",
                "content": "You are a careful relation extraction assistant. Return triples only.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- Truncate input to avoid OOM on long documents ---
        # Reserve space for generation; stay within GPU memory budget.
        max_input_tokens = 6000  # safe limit for 7B model on 32 GB V100
        model_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.model.device)

        try:
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": self.max_tokens,
                    "do_sample": self.temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "attention_mask": model_inputs.get("attention_mask"),
                }
                if self.temperature > 0:
                    gen_kwargs["temperature"] = max(self.temperature, 1e-5)
                outputs = self.model.generate(
                    model_inputs["input_ids"],
                    **gen_kwargs,
                )

            gen_tokens = outputs[0][model_inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            print("[WARN] CUDA OOM — skipping this example, returning empty.")
            text = ""
        finally:
            # Free KV-cache and intermediate tensors between examples.
            del model_inputs
            torch.cuda.empty_cache()

        return text or ""


TRIPLE_RE = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")


def _json_item_to_triple(x) -> Triple:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return str(x[0]).strip(), str(x[1]).strip(), str(x[2]).strip()
    if isinstance(x, dict):
        h = x.get("head", x.get("h", x.get("subject", x.get("subj"))))
        r = x.get("relation", x.get("r", x.get("predicate", x.get("label"))))
        t = x.get("tail", x.get("t", x.get("object", x.get("obj"))))
        if h is not None and r is not None and t is not None:
            return str(h).strip(), str(r).strip(), str(t).strip()
    return "", "", ""


def _parse_json_payload(text: str) -> List[Triple]:
    triples: List[Triple] = []
    obj = json.loads(text)
    if isinstance(obj, dict):
        for key in ("triples", "relations", "output", "predictions"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
    if isinstance(obj, list):
        for item in obj:
            h, r, t = _json_item_to_triple(item)
            if h and r and t:
                triples.append((h, r, t))
    return triples


def parse_triples(text: str) -> List[Triple]:
    text = text.strip()

    # Prefer JSON parsing first when possible.
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        try:
            triples = _parse_json_payload(text)
            if triples:
                return triples
        except Exception:
            pass

    triples = []
    for h, r, t in TRIPLE_RE.findall(text):
        triples.append((h.strip(), r.strip(), t.strip()))
    return triples

