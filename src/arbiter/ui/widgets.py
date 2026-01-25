"""Textual widgets for Arbiter TUI."""

from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import OptionList


@dataclass(frozen=True)
class SelectOption:
    value: str
    label: str


class MultiSelectList(OptionList):
    """OptionList with checkbox-style multi-select."""

    class Confirmed(Message):
        def __init__(self, sender: "MultiSelectList") -> None:
            super().__init__(sender)

    def __init__(self, options: list[SelectOption], selected: set[str]) -> None:
        self._options = options
        self._selected = set(selected)
        super().__init__()
        self._refresh_options()

    @property
    def selected_values(self) -> set[str]:
        return set(self._selected)

    @property
    def options_values(self) -> list[str]:
        return [option.value for option in self._options]

    def update_options(self, options: list[SelectOption], selected: set[str]) -> None:
        self._options = options
        self._selected = set(selected)
        self._refresh_options()

    def add_custom(self, value: str) -> None:
        if value in self.options_values:
            return
        self._options.append(SelectOption(value=value, label=value))
        self._selected.add(value)
        self._refresh_options()

    def _refresh_options(self) -> None:
        self.clear_options()
        for option in self._options:
            marker = "[x]" if option.value in self._selected else "[ ]"
            self.add_option(f"{marker} {option.label}")

    def _toggle_selected(self) -> None:
        index = self.highlighted
        if index is None:
            return
        if 0 <= index < len(self._options):
            value = self._options[index].value
            if value in self._selected:
                self._selected.remove(value)
            else:
                self._selected.add(value)
            self._refresh_options()
            self.highlighted = index

    async def on_key(self, event) -> None:  # type: ignore[override]
        if event.key == "space":
            self._toggle_selected()
            event.stop()
        elif event.key == "enter":
            await self.post_message(self.Confirmed(self))
            event.stop()
