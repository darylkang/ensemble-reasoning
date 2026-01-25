"""Catalog loader for curated models and personas."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogItem:
    value: str
    name: str
    description: str
    is_default: bool


def load_model_catalog() -> tuple[list[CatalogItem], str | None]:
    return _load_catalog(
        env_var="ARBITER_MODEL_CATALOG_PATH",
        filename="models.json",
        key_field="slug",
    )


def load_persona_catalog() -> tuple[list[CatalogItem], str | None]:
    return _load_catalog(
        env_var="ARBITER_PERSONA_CATALOG_PATH",
        filename="personas.json",
        key_field="id",
    )


def _load_catalog(
    *,
    env_var: str,
    filename: str,
    key_field: str,
) -> tuple[list[CatalogItem], str | None]:
    override = os.getenv(env_var)
    if override:
        try:
            return _load_from_path(Path(override), key_field), None
        except Exception as exc:  # noqa: BLE001
            built_in = _load_builtin(filename, key_field)
            return built_in, f"Catalog override failed ({env_var}). Using built-in catalog."
    return _load_builtin(filename, key_field), None


def _load_from_path(path: Path, key_field: str) -> list[CatalogItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _validate_catalog(data, key_field)


def _load_builtin(filename: str, key_field: str) -> list[CatalogItem]:
    data = resources.files(__name__).joinpath(filename).read_text(encoding="utf-8")
    return _validate_catalog(json.loads(data), key_field)


def _validate_catalog(data: Any, key_field: str) -> list[CatalogItem]:
    if not isinstance(data, list):
        raise ValueError("Catalog must be a list of items.")
    items: list[CatalogItem] = []
    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Catalog item {idx} must be an object.")
        key = raw.get(key_field)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Catalog item {idx} missing '{key_field}'.")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            name = key
        description = raw.get("description")
        if not isinstance(description, str):
            description = ""
        is_default = bool(raw.get("default", False))
        items.append(
            CatalogItem(
                value=key.strip(),
                name=name.strip(),
                description=description.strip(),
                is_default=is_default,
            )
        )
    if not items:
        raise ValueError("Catalog is empty.")
    return items
