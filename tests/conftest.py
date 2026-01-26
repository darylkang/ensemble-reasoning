from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest_asyncio

from arbiter.config import default_canonical_config
from arbiter.engine import execute_trials
from arbiter.runflow import prepare_run, write_manifest


async def _run_mock(tmp_path: Path, *, seed: int, run_name: str) -> tuple:
    config = default_canonical_config(default_model="openai/gpt-5", llm_mode="mock")
    config["question"]["text"] = "Does the sky appear blue on a clear day?"
    config["execution"]["k_max"] = 8
    config["execution"]["workers"] = 2
    config["execution"]["batch_size"] = 2
    config["execution"]["retries"] = 1
    config["execution"]["seed"] = seed
    config["execution"]["parse_failure_policy"] = "continue"
    config["convergence"]["min_trials"] = 4
    config["convergence"]["patience_batches"] = 1
    config["convergence"]["delta_js_threshold"] = 0.5
    config["convergence"]["epsilon_new_threshold"] = 0.5
    config["convergence"]["epsilon_ci_half_width"] = 0.5

    setup = prepare_run(
        input_config=config,
        run_name=run_name,
        output_base_dir=tmp_path,
        default_model="openai/gpt-5",
        api_key_present=False,
        selected_mode="mock",
    )
    result = await execute_trials(
        run_dir=setup.run_dir,
        resolved_config=setup.resolved_config,
        question=setup.question_record,
    )
    write_manifest(setup=setup, ended_at=datetime.now(timezone.utc), execution_result=result)
    return setup, result


@pytest_asyncio.fixture
async def mock_run(tmp_path: Path):
    return await _run_mock(tmp_path, seed=123, run_name="test")


@pytest_asyncio.fixture
async def mock_run_repeatable(tmp_path: Path):
    return await _run_mock(tmp_path, seed=123, run_name="repeatable")
