"""OpenRouter embeddings client."""

from __future__ import annotations

import base64
from array import array
import os
import time
from typing import Any

import httpx

from arbiter.embeddings.types import EmbeddingResult


class OpenRouterEmbeddingsClient:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        self._model = model
        self._base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for remote embeddings.")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        body = {
            "model": self._model,
            "input": texts,
            "encoding_format": "base64",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        response = await self._client.post(url, json=body, headers=headers)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code >= 300:
            raise RuntimeError(f"OpenRouter embeddings error {response.status_code}: {response.text}")
        payload = response.json()
        data = payload.get("data", [])
        results: list[EmbeddingResult | None] = [None] * len(texts)
        for fallback_index, item in enumerate(data):
            index = item.get("index")
            if index is None:
                index = fallback_index
            raw_embedding = item.get("embedding")
            embedding = _decode_embedding(raw_embedding)
            results[index] = EmbeddingResult(
                text=texts[index],
                embedding=embedding,
                model=payload.get("model", self._model),
                dims=len(embedding),
                raw={"latency_ms": latency_ms, "usage": payload.get("usage"), "raw": item},
            )
        if any(result is None for result in results):
            raise RuntimeError("OpenRouter embeddings response missing entries.")
        return [result for result in results if result is not None]

    async def aclose(self) -> None:
        await self._client.aclose()


def _decode_embedding(raw: Any) -> list[float]:
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        data = base64.b64decode(raw)
        arr = array("f")
        arr.frombytes(data)
        return list(arr)
    raise ValueError("Unsupported embedding format")
