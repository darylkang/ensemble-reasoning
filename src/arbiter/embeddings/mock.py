"""Mock embeddings client."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from arbiter.embeddings.types import EmbeddingResult


class MockEmbeddingsClient:
    def __init__(self, model: str, dims: int = 1536) -> None:
        self._model = model
        self._dims = dims

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            seed = int(hashlib.sha256((self._model + "|" + text).encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self._dims)]
            norm = math.sqrt(sum(value * value for value in vec)) or 1.0
            embedding = [value / norm for value in vec]
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self._model,
                    dims=self._dims,
                    raw={"mock": True},
                )
            )
        return results

    async def aclose(self) -> None:
        return None
