from abc import ABC, abstractmethod


class BaseLLMEngine(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send a list of chat messages and return the assistant reply as a string."""
        ...

    def generate(self, prompt: str, **kwargs) -> str:
        """Send a plain-text prompt; default implementation wraps it as a user message."""
        return self.chat([{"role": "user", "content": prompt}])
