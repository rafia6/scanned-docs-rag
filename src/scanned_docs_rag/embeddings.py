"""Embedding models behind a small interface, so tests can swap in a fake."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
