from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from tests.utils import run_mock


@pytest_asyncio.fixture
async def mock_run(tmp_path: Path):
    return await run_mock(tmp_path, seed=123, run_name="test")


@pytest_asyncio.fixture
async def mock_run_repeatable(tmp_path: Path):
    return await run_mock(tmp_path, seed=123, run_name="repeatable")
