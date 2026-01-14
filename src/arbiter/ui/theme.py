"""Rich theme for arbiter CLI."""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "accent": "cyan",
        "title": "bold white",
        "subtitle": "dim",
        "step": "bold white",
        "info": "dim",
        "warning": "yellow",
        "success": "green",
        "error": "bold red",
        "label": "dim",
        "value": "white",
        "path": "cyan",
    }
)
