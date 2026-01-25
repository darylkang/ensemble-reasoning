"""Rich theme for arbiter CLI."""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "accent": "gold3",
        "title": "bold wheat1",
        "subtitle": "grey70",
        "dim": "grey50",
        "step": "bold gold3",
        "info": "grey70",
        "disabled": "grey50",
        "warning": "orange3",
        "success": "green3",
        "error": "red3",
        "label": "grey70",
        "value": "white",
        "path": "gold3",
        "border": "grey37",
    }
)
