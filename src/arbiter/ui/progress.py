"""Progress and status helpers for arbiter CLI."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from arbiter.ui.console import get_console


def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}", style="label"),
        BarColumn(bar_width=None, style="border", complete_style="accent", finished_style="accent"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    )


def build_execution_progress(worker_count: int, total_trials: int) -> tuple[Progress, int, list[int]]:
    progress = Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}", style="label"),
        BarColumn(bar_width=None, style="border", complete_style="accent", finished_style="accent"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    )
    overall_task_id = progress.add_task("Overall · early stop possible", total=total_trials)
    worker_task_ids = []
    for index in range(worker_count):
        worker_task_ids.append(progress.add_task(f"worker {index + 1} · idle · done 0", total=None))
    return progress, overall_task_id, worker_task_ids


@contextmanager
def status_spinner(message: str) -> Iterator[object]:
    console = get_console()
    with console.status(message, spinner="dots", spinner_style="accent") as status:
        yield status
