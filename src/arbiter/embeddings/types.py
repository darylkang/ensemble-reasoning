"""Embedding request/response types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    embedding: list[float]
    model: str
    dims: int
    raw: dict[str, Any]
