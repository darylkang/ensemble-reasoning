"""Wizard state for guided and config-driven flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WizardState:
    default_model: str
    api_key_present: bool
    config_path: Path
    config_mode: str | None = None
    input_config: dict[str, Any] = field(default_factory=dict)
    step_order: list[str] = field(default_factory=lambda: ["welcome", "mode_select", "config_mode"])
    save_config: bool = False
    run_name: str = "auto"
    output_base_dir: Path = field(default_factory=lambda: Path("./runs"))
    use_advanced: bool = False
    selected_mode: str = "mock"

    def compute_step_order(self) -> None:
        steps = ["welcome", "mode_select", "config_mode"]
        missing_question = not (self.input_config.get("question", {}) or {}).get("text")
        if self.config_mode == "guided":
            steps.extend(
                [
                    "question",
                    "decode",
                    "personas",
                    "models",
                    "protocol",
                    "advanced_gate",
                ]
            )
            if self.use_advanced:
                steps.append("advanced")
            steps.extend(["review", "run_setup"])
        else:
            if missing_question:
                steps.append("question")
            steps.extend(["review", "run_setup"])
        self.step_order = steps

    def step_index(self, step_id: str) -> tuple[int, int]:
        if step_id not in self.step_order:
            self.step_order.append(step_id)
        index = self.step_order.index(step_id) + 1
        return index, len(self.step_order)
