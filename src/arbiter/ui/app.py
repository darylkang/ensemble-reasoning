"""Textual TUI entrypoint for Arbiter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import os
import asyncio

from textual.app import App

from arbiter.config import default_canonical_config, normalize_canonical_config
from arbiter.env import load_dotenv
from arbiter.runflow import RunSetup, prepare_run, write_manifest, collect_last_checkpoints, collect_top_modes
from arbiter.engine import execute_trials
from arbiter.ui.screens import (
    AdvancedGateScreen,
    AdvancedScreen,
    ConfigScreen,
    DecodeScreen,
    ExecutionScreen,
    ModelsScreen,
    PersonasScreen,
    ProtocolScreen,
    QuestionScreen,
    ReceiptScreen,
    ReviewScreen,
    RunModeScreen,
    WelcomeScreen,
)


@dataclass
class WizardState:
    default_model: str
    api_key_present: bool
    selected_mode: str
    config_path: Path
    input_config: dict = field(default_factory=dict)
    config_mode: str = "guided"
    use_customize: bool = False
    use_advanced: bool = False
    run_name: str = "auto"
    output_base_dir: Path = field(default_factory=lambda: Path("./runs"))


class ArbiterApp(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        default_model = os.getenv("ARBITER_DEFAULT_MODEL", "openai/gpt-5")
        api_key_present = bool(os.getenv("OPENROUTER_API_KEY"))
        self.state = WizardState(
            default_model=default_model,
            api_key_present=api_key_present,
            selected_mode="remote" if api_key_present else "mock",
            config_path=Path("arbiter.config.json"),
        )
        self.run_setup: RunSetup | None = None
        self.execution_result = None
        self.smoke = os.getenv("ARBITER_TUI_SMOKE") == "1"

    async def on_mount(self) -> None:
        if self.smoke:
            await self._run_smoke()
            self.exit()
            return
        await self.push_screen(WelcomeScreen(self.state))

    async def _run_smoke(self) -> None:
        self.state.input_config = default_canonical_config(
            default_model=self.state.default_model,
            llm_mode="mock",
        )
        self.state.input_config.setdefault("question", {})["text"] = "Smoke test question."
        self.state.input_config.setdefault("execution", {})["k_max"] = 8
        self.state.input_config = normalize_canonical_config(
            self.state.input_config,
            default_model=self.state.default_model,
            llm_mode="mock",
        )
        self.state.run_name = "smoke"
        self.state.output_base_dir = Path("./runs")
        self.run_setup = prepare_run(
            input_config=self.state.input_config,
            run_name=self.state.run_name,
            output_base_dir=self.state.output_base_dir,
            default_model=self.state.default_model,
            api_key_present=False,
            selected_mode="mock",
        )
        await execute_trials(
            run_dir=self.run_setup.run_dir,
            resolved_config=self.run_setup.resolved_config,
            question=self.run_setup.question_record,
            on_event=None,
        )
        write_manifest(setup=self.run_setup, ended_at=datetime.now(timezone.utc))

    def show_config(self) -> None:
        self.push_screen(ConfigScreen(self.state))

    def show_question(self) -> None:
        self.push_screen(QuestionScreen(self.state))

    def show_run_mode(self) -> None:
        self.push_screen(RunModeScreen(self.state))

    def show_decode(self) -> None:
        self.push_screen(DecodeScreen(self.state))

    def show_personas(self) -> None:
        self.push_screen(PersonasScreen(self.state))

    def show_models(self) -> None:
        self.push_screen(ModelsScreen(self.state))

    def show_protocol(self) -> None:
        self.push_screen(ProtocolScreen(self.state))

    def show_advanced_gate(self) -> None:
        self.push_screen(AdvancedGateScreen(self.state))

    def show_advanced(self) -> None:
        self.push_screen(AdvancedScreen(self.state))

    def show_review(self) -> None:
        self.push_screen(ReviewScreen(self.state))

    def show_execution(self) -> None:
        self.push_screen(ExecutionScreen(self.state))

    def show_receipt(self, run_dir: Path, execution_result) -> None:
        top_modes = collect_top_modes(run_dir, top_n=3)
        checkpoints = collect_last_checkpoints(run_dir, limit=3)
        self.push_screen(ReceiptScreen(self.state, execution_result, top_modes, checkpoints, run_dir))


def run_app() -> None:
    ArbiterApp().run()
