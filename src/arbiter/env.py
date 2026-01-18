"""Minimal .env loader for Arbiter CLI."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value == "":
            continue
        if key not in os.environ:
            os.environ[key] = value
    return True
