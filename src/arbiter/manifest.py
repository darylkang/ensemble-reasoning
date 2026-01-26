"""Manifest schema and runtime metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess


@dataclass(frozen=True)
class GitInfo:
    sha: str
    dirty: bool


@dataclass(frozen=True)
class Manifest:
    run_id: str
    started_at: str
    ended_at: str
    git_sha: str
    git_dirty: bool
    python_version: str
    platform: dict
    config_hash: str
    semantic_config_hash: str
    embedding_model: str
    summarizer_model: str
    summarizer_prompt_version: str
    planned_call_budget: int
    planned_call_budget_scope: str
    planned_total_trials: int
    planned_total_trials_scope: str
    llm_call_count: int
    embedding_call_count: int
    summarizer_call_count: int

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "config_hash": self.config_hash,
            "semantic_config_hash": self.semantic_config_hash,
            "embedding_model": self.embedding_model,
            "summarizer_model": self.summarizer_model,
            "summarizer_prompt_version": self.summarizer_prompt_version,
            "planned_call_budget": self.planned_call_budget,
            "planned_call_budget_scope": self.planned_call_budget_scope,
            "planned_total_trials": self.planned_total_trials,
            "planned_total_trials_scope": self.planned_total_trials_scope,
            "llm_call_count": self.llm_call_count,
            "embedding_call_count": self.embedding_call_count,
            "summarizer_call_count": self.summarizer_call_count,
        }


def get_git_info(cwd: Path) -> GitInfo:
    sha = "unknown"
    dirty = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return GitInfo(sha=sha, dirty=dirty)

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(status.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        dirty = False

    return GitInfo(sha=sha, dirty=dirty)


def platform_info() -> dict:
    return {
        "os": platform.system(),
        "machine": platform.machine(),
    }
