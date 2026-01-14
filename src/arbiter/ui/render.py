"""Render helpers for arbiter CLI."""

from __future__ import annotations

from typing import Mapping, Sequence

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from arbiter.ui.console import get_console


def render_banner(title: str, subtitle: str) -> None:
    console = get_console()
    title_text = Text(title, style="title")
    subtitle_text = Text(subtitle, style="subtitle")
    panel = Panel(
        Group(Align.center(title_text), Align.center(subtitle_text)),
        border_style="accent",
        padding=(1, 2),
    )
    console.print(panel)


def render_step_header(step_idx: int, step_total: int, title: str) -> None:
    console = get_console()
    text = Text(f"Step {step_idx}/{step_total}  {title}", style="step")
    console.print(Rule(text, style="accent"))


def render_info(text: str) -> None:
    console = get_console()
    console.print(text, style="info", markup=False)


def render_warning(text: str) -> None:
    console = get_console()
    console.print(text, style="warning", markup=False)


def render_success(text: str) -> None:
    console = get_console()
    console.print(text, style="success", markup=False)


def render_error(text: str) -> None:
    console = get_console()
    console.print(text, style="error", markup=False)


def render_summary_table(rows: Mapping[str, str] | Sequence[tuple[str, str]]) -> None:
    console = get_console()
    table = Table(
        show_header=False,
        box=box.MINIMAL,
        pad_edge=False,
    )
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="value")

    items = rows.items() if hasattr(rows, "items") else rows
    for key, value in items:
        table.add_row(str(key), str(value))

    console.print(table)
