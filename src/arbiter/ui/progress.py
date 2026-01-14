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
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    )


@contextmanager
def status_spinner(message: str) -> Iterator[object]:
    console = get_console()
    with console.status(message, spinner="dots", spinner_style="accent") as status:
        yield status
