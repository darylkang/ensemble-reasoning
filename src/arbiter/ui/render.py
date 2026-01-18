"""Render helpers for arbiter CLI."""

from __future__ import annotations

from typing import Mapping, Sequence
import sys

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arbiter.ui.console import get_console


def render_banner(title: str, subtitle: str) -> None:
    console = get_console()
    panel = Panel(
        Group(Text(title, style="title"), Text(subtitle, style="subtitle")),
        box=box.ROUNDED,
        border_style="grey37",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)
    console.print()


def render_step_header(step_idx: int, step_total: int, title: str, description: str) -> None:
    console = get_console()
    header = Text(f"Step {step_idx}/{step_total} · {title}", style="step")
    content = [header]
    if description:
        content.append(Text(description, style="subtitle"))
    panel = Panel(
        Group(*content),
        box=box.ROUNDED,
        border_style="grey37",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_gap(*, after_prompt: bool) -> None:
    if sys.stdin.isatty():
        print()
        return
    if after_prompt:
        print("\n")
    else:
        print()


def render_info(text: str) -> None:
    console = get_console()
    console.print(text, style="info", markup=False)


def render_warning(text: str) -> None:
    console = get_console()
    console.print(text, style="warning", markup=False)


def render_notice(text: str) -> None:
    console = get_console()
    panel = Panel(
        Text(text, style="warning"),
        box=box.ROUNDED,
        border_style="warning",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_success(text: str) -> None:
    console = get_console()
    console.print(text, style="success", markup=False)


def render_error(text: str) -> None:
    console = get_console()
    panel = Panel(
        Text(text, style="error"),
        box=box.ROUNDED,
        border_style="error",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_summary_table(rows: Mapping[str, str] | Sequence[tuple[str, str]], title: str = "Summary") -> None:
    console = get_console()
    table = Table(
        show_header=False,
        box=None,
        pad_edge=False,
    )
    table.add_column(style="label", no_wrap=True, justify="right")
    table.add_column(style="value")

    items = rows.items() if hasattr(rows, "items") else rows
    for key, value in items:
        label = Text(str(key), style="label")
        value_text = Text(str(value), style="value")
        if str(key).lower() == "run folder":
            value_text.stylize("accent")
        table.add_row(label, value_text)

    panel = Panel(
        table,
        title=Text(title, style="step"),
        title_align="left",
        border_style="grey37",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=True,
    )
    console.print()
    console.print(panel)
