"""Embedding client interface and factory."""

from __future__ import annotations

from typing import Protocol

from arbiter.embeddings.mock import MockEmbeddingsClient
from arbiter.embeddings.openrouter import OpenRouterEmbeddingsClient
from arbiter.embeddings.types import EmbeddingResult


class EmbeddingsClient(Protocol):
    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        ...

    async def aclose(self) -> None:
        ...


def create_embeddings_client(mode: str, model: str, base_url: str | None = None) -> EmbeddingsClient:
    if mode == "openrouter":
        return OpenRouterEmbeddingsClient(model=model, base_url=base_url)
    return MockEmbeddingsClient(model=model)
