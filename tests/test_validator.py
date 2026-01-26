from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arbiter.config import default_canonical_config


def _run_validate(path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "arbiter.cli", "config", "validate", "--path", str(path)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_validator_exit_codes(tmp_path: Path) -> None:
    valid = default_canonical_config(default_model="openai/gpt-5", llm_mode="mock")
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(valid), encoding="utf-8")
    result_valid = _run_validate(valid_path)
    assert result_valid.returncode == 0

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    result_invalid = _run_validate(invalid_path)
    assert result_invalid.returncode == 1
