"""Rich theme for arbiter CLI."""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "accent": "sky_blue3",
        "title": "bold sky_blue3",
        "subtitle": "dim",
        "step": "bold sky_blue3",
        "info": "dim",
        "warning": "red3",
        "success": "green3",
        "error": "bold red3",
        "label": "dim",
        "value": "white",
        "path": "sky_blue3",
    }
)
