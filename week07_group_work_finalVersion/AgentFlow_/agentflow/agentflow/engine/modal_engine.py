import os, requests
from .base import BaseLLMEngine

class ModalEngine(BaseLLMEngine):
    def __init__(self, model_id: str = "", temperature: float = 0.0, **kwargs):
        self.url = os.environ["MODAL_PLANNER_URL"]
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> str:
        resp = requests.post(
            self.url,
            json={"messages": messages, "temperature": self.temperature},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate(self, prompt: str, **kwargs) -> str:
        """Satisfy BaseLLMEngine.generate() by wrapping prompt as a user message."""
        return self.chat([{"role": "user", "content": prompt}])
