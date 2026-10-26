"""LLM providers behind one interface: Ollama for local dev, Groq (free tier) for the demo."""

import re
from typing import Protocol

import requests

from .config import Settings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Qwen3-style models may emit a reasoning block before the answer."""
    return _THINK.sub("", text).strip()


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, url: str):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"ollama/{model}"

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return strip_think(resp.json()["message"]["content"])


class GroqLLM:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.model, self.api_key = model, api_key
        self.name = f"groq/{model}"

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        resp = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=120,
        )
        resp.raise_for_status()
        return strip_think(resp.json()["choices"][0]["message"]["content"])


def get_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "groq":
        return GroqLLM(settings.groq_model, settings.groq_api_key)
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.ollama_model, settings.ollama_url)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
