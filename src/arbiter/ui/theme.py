"""Rich theme for arbiter CLI."""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "accent": "blue",
        "title": "bold blue",
        "subtitle": "dim",
        "step": "bold blue",
        "info": "dim",
        "warning": "red3",
        "success": "green3",
        "error": "bold red3",
        "label": "dim",
        "value": "white",
        "path": "blue",
    }
)
