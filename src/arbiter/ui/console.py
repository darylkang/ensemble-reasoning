"""Shared console for arbiter CLI."""

from __future__ import annotations

from rich.console import Console

from arbiter.ui.theme import THEME

_CONSOLE = Console(theme=THEME, highlight=False)


def get_console() -> Console:
    return _CONSOLE
