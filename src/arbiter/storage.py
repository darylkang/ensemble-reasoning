"""Storage helpers for run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def compute_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(f"{data}\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{data}\n")


def create_run_dir(base_dir: Path, timestamp: str, slug: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / f"{timestamp}_{slug}"
    if not candidate.exists():
        candidate.mkdir()
        return candidate

    counter = 1
    while True:
        alternate = base_dir / f"{timestamp}_{slug}_{counter:02d}"
        if not alternate.exists():
            alternate.mkdir()
            return alternate
        counter += 1


def cleanup_run_dir(run_dir: Path) -> None:
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        return
    except OSError:
        return
