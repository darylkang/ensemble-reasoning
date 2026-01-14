"""Rich theme for arbiter CLI."""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "accent": "bright_blue",
        "title": "bold bright_blue",
        "subtitle": "dim",
        "step": "bold bright_blue",
        "info": "dim",
        "warning": "red3",
        "success": "green3",
        "error": "bold red3",
        "label": "dim",
        "value": "white",
        "path": "cyan",
    }
)
