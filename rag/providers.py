"""Embedding and chat-model backends.

`gemini` talks to Google's API. `local` is an offline stand-in so the app,
tests and CI run without an API key: hashed bag-of-words embeddings and an
"LLM" that quotes the best matching passage.
"""

import hashlib
import math
import re
from typing import Protocol

from rag.config import Settings


class Embedder(Protocol):
    default_min_score: float

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class ChatModel(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class MissingAPIKeyError(RuntimeError):
    pass


# ---------- Gemini ----------


class GeminiEmbedder:
    BATCH = 100
    default_min_score = 0.35

    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            resp = self._client.models.embed_content(
                model=self._model,
                contents=texts[i : i + self.BATCH],
                config=types.EmbedContentConfig(task_type=task),
            )
            vectors.extend(list(e.values) for e in resp.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChat:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
        )
        return (resp.text or "").strip()


# ---------- Local / offline ----------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "for",
    "with",
    "what",
    "which",
    "who",
    "how",
    "does",
    "do",
    "it",
    "this",
    "that",
    "be",
    "by",
    "as",
    "at",
    "from",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashEmbedder:
    """Deterministic bag-of-words vectors. No network, no model download."""

    default_min_score = 0.05

    def __init__(self, dims: int = 512):
        self.dims = dims

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ExtractiveChat:
    """Offline stand-in for an LLM: returns the context sentence that shares the
    most words with the question, with its citation. Good enough to demo the flow."""

    def generate(self, system: str, prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[-1]
        q_words = set(_tokens(question))
        best, best_n, best_score = None, 0, 0
        passages = re.findall(r"\[(\d+)\] \([^)]*\)\n(.+?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", prompt, re.S)
        for n, body in passages:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                score = len(q_words & set(_tokens(sentence)))
                if score > best_score:
                    best, best_n, best_score = sentence.strip(" #"), n, score
        if not best:
            return "I cannot find the answer in the uploaded documents."
        return f"{best} [{best_n}]"


def build_providers(settings: Settings) -> tuple[Embedder, ChatModel]:
    if settings.provider == "local":
        return HashEmbedder(), ExtractiveChat()
    if settings.provider == "gemini":
        if not settings.google_api_key:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is not set. Add it to .env, or set PROVIDER=local to run offline."
            )
        return (
            GeminiEmbedder(settings.google_api_key, settings.embed_model),
            GeminiChat(settings.google_api_key, settings.chat_model),
        )
    raise ValueError(f"Unknown provider: {settings.provider!r}")
