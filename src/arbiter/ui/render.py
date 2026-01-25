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
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)
    console.print()


def render_welcome_panel(
    *,
    title: str,
    subtitle: str,
    status_mode: str,
    status_openrouter: str,
    status_config: str,
    recommendation: str,
    action: str,
) -> None:
    console = get_console()
    status = Text()
    status.append("MODE ", style="dim")
    status.append(status_mode, style="accent")
    status.append("   OPENROUTER ", style="dim")
    status.append(
        status_openrouter,
        style="success" if status_openrouter == "CONNECTED" else "warning",
    )
    status.append("   CONFIG ", style="dim")
    status.append(
        status_config,
        style="success" if status_config == "FOUND" else "warning",
    )

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="label", no_wrap=True, justify="right")
    table.add_column(style="value")
    table.add_row("Status", status)
    table.add_row("Recommended", recommendation)

    panel = Panel(
        Group(
            Text(title, style="title"),
            Text(subtitle, style="subtitle"),
            Text(""),
            table,
            Text(""),
            Text(action, style="dim"),
        ),
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)
    console.print()


def render_mode_select_panel(
    *,
    title: str,
    description: str,
    options: Sequence[tuple[str, str, bool]],
    instructions: str,
    note: str | None = None,
) -> None:
    console = get_console()
    lines = [Text(title, style="step"), Text(description, style="subtitle"), Text("")]
    for index, (label, state, enabled) in enumerate(options, start=1):
        marker = "[x]" if state == "selected" else "[ ]"
        style = "value" if enabled else "disabled"
        lines.append(Text(f"{index:>2} {marker} {label}", style=style))
    lines.append(Text(""))
    lines.append(Text(instructions, style="dim"))
    if note:
        lines.append(Text(note, style="dim"))
    panel = Panel(
        Group(*lines),
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_step_header(step_idx: int, step_total: int, title: str, description: str) -> None:
    console = get_console()
    header = Text(f"Step {step_idx}/{step_total} · {title}", style="step")
    content = [header]
    if description:
        content.append(Text(description, style="subtitle"))
    panel = Panel(
        Group(*content),
        box=box.ROUNDED,
        border_style="border",
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
        border_style="border",
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
        border_style="border",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=True,
    )
    console.print()
    console.print(panel)


def render_validation_panel(title: str, issues: Sequence[str], *, style: str) -> None:
    console = get_console()
    lines = [Text(title, style="step")]
    for issue in issues:
        lines.append(Text(f"- {issue}", style=style))
    panel = Panel(
        Group(*lines),
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_selection_panel(
    *,
    title: str,
    description: str,
    options: Sequence[str],
    selected: Sequence[str],
    instructions: str,
) -> None:
    console = get_console()
    header = Text(title, style="step")
    desc = Text(description, style="subtitle")
    rows = []
    selected_set = set(selected)
    for index, option in enumerate(options, start=1):
        marker = "[x]" if option in selected_set else "[ ]"
        rows.append(Text(f"{index:>2} {marker} {option}", style="value"))
    body = Group(header, desc, Text(""), *rows, Text(""), Text(instructions, style="dim"))
    panel = Panel(
        body,
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_execution_header(summary: Mapping[str, str]) -> None:
    console = get_console()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="label", no_wrap=True, justify="right")
    table.add_column(style="value")
    for key, value in summary.items():
        table.add_row(Text(str(key), style="label"), Text(str(value), style="value"))
    panel = Panel(
        Group(Text("Execution", style="step"), Text(""), table),
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)
    console.print()


def render_batch_checkpoint(row: Mapping[str, str]) -> None:
    console = get_console()
    table = Table(show_header=True, box=box.ROUNDED, pad_edge=False)
    for key in row.keys():
        table.add_column(str(key), style="label", no_wrap=True)
    table.add_row(*[str(value) for value in row.values()])
    panel = Panel(
        table,
        title=Text("Batch checkpoint", style="step"),
        title_align="left",
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print(panel)


def render_receipt_panel(
    *,
    title: str,
    summary: Mapping[str, str],
    top_modes: Sequence[tuple[str, float, str]],
    artifact_path: str,
) -> None:
    console = get_console()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="label", no_wrap=True, justify="right")
    table.add_column(style="value")
    for key, value in summary.items():
        table.add_row(Text(str(key), style="label"), Text(str(value), style="value"))
    mode_lines = []
    if top_modes:
        for mode_id, share, exemplar in top_modes:
            mode_lines.append(Text(f"{mode_id} · {share:.3f} · {exemplar}", style="value"))
    else:
        mode_lines.append(Text("n/a", style="dim"))

    panel = Panel(
        Group(
            Text(title, style="step"),
            Text(""),
            table,
            Text(""),
            Text("Top modes", style="subtitle"),
            *mode_lines,
            Text(""),
            Text(f"Artifacts: {artifact_path}", style="dim"),
        ),
        box=box.ROUNDED,
        border_style="border",
        padding=(0, 2),
        expand=True,
    )
    console.print()
    console.print(panel)
