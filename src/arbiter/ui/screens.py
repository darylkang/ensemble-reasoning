"""Textual screens for Arbiter wizard."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import asyncio
import json

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Label, OptionList, ProgressBar, Static, TextArea
from textual.widgets.option_list import Option

from arbiter.catalog import load_model_catalog, load_persona_catalog
from arbiter.config import default_canonical_config, normalize_canonical_config
from arbiter.engine import execute_trials
from arbiter.runflow import prepare_run, write_manifest
from arbiter.storage import write_json
from arbiter.validation import load_and_validate_config
from arbiter.ui.widgets import MultiSelectList, SelectOption


class BaseScreen(Screen):
    """Base screen with a surface container."""

    def __init__(self, state) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.state = state


def _footer_hint() -> Static:
    return Static("↑↓ navigate · space toggle · enter confirm · q quit", id="key-hint")


class WelcomeScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="welcome-surface", classes="surface")
        surface.border_title = "Welcome"
        with surface:
            yield Label("Ensemble Reasoning · Run Setup", id="welcome-subtitle")
            status = _status_line(self.state)
            yield Static(status, id="welcome-status")
            self.note = Static("", id="welcome-note")
            yield self.note
            self.mode_list = OptionList(id="mode-list")
            yield self.mode_list
            yield Static("Press Enter to begin", id="welcome-hint")
        yield _footer_hint()

    def on_mount(self) -> None:
        self.mode_list.clear_options()
        if not self.state.api_key_present:
            self.state.selected_mode = "mock"
            self.mode_list.add_option(Option("Mock (no network calls)"))
            self.mode_list.add_option(Option("Remote (requires OPENROUTER_API_KEY)", disabled=True))
            self.note.update("To enable Remote, set OPENROUTER_API_KEY in .env (or your environment).")
            self.mode_list.highlighted = 0
        else:
            self.state.selected_mode = "remote"
            self.mode_list.add_option(Option("Mock (no network calls)"))
            self.mode_list.add_option(Option("Remote (OpenRouter)"))
            self.mode_list.highlighted = 1

    def action_continue(self) -> None:
        if self.mode_list.highlighted == 1 and not self.state.api_key_present:
            return
        self.state.selected_mode = "remote" if self.mode_list.highlighted == 1 else "mock"
        self.app.show_config()


class ConfigScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="config-surface", classes="surface")
        surface.border_title = "Configuration Mode"
        with surface:
            self.message = Static("", id="config-message")
            yield self.message
            self.list = OptionList("Load config", "Guided wizard", "Create template & exit", id="config-list")
            yield self.list
            yield Static("Press Enter to continue", id="config-hint")
        yield _footer_hint()

    def on_mount(self) -> None:
        self.list.clear_options()
        if self.state.config_path.exists():
            self.list.add_option("Load config (recommended)")
            self.list.add_option("Guided wizard")
            self.list.add_option("Create template & exit")
            self.list.highlighted = 0
        else:
            self.list.add_option("Load config")
            self.list.add_option("Guided wizard")
            self.list.add_option("Create template & exit")
            self.list.highlighted = 1

    def action_continue(self) -> None:
        choice = self.list.highlighted or 0
        if choice == 0 and self.state.config_path.exists():
            result = load_and_validate_config(
                self.state.config_path,
                default_model=self.state.default_model,
                llm_mode="remote" if self.state.api_key_present else "mock",
            )
            if result.errors:
                lines = [f"- {issue.path}: {issue.message}" for issue in result.errors]
                self.message.update("Config invalid:\n" + "\n".join(lines))
                return
            self.state.input_config = result.config or {}
            self.state.config_mode = "load"
            if not self.state.api_key_present:
                self.state.input_config.setdefault("llm", {})["mode"] = "mock"
            else:
                self.state.input_config.setdefault("llm", {})["mode"] = self.state.selected_mode
            if not (self.state.input_config.get("question", {}) or {}).get("text"):
                self.app.show_question()
            else:
                self.app.show_review()
            return
        if choice == 2:
            example = Path("arbiter.config.example.json")
            if example.exists():
                self.state.config_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                write_json(
                    self.state.config_path,
                    default_canonical_config(
                        default_model=self.state.default_model,
                        llm_mode="remote" if self.state.api_key_present else "mock",
                    ),
                )
            self.app.exit()
            return
        llm_mode = "remote" if self.state.selected_mode == "remote" else "mock"
        self.state.input_config = default_canonical_config(
            default_model=self.state.default_model,
            llm_mode=llm_mode,
        )
        self.state.config_mode = "guided"
        self.app.show_question()


class QuestionScreen(BaseScreen):
    BINDINGS = [("ctrl+enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="question-surface", classes="surface")
        surface.border_title = "Question"
        with surface:
            yield Static("Paste question. Press Ctrl+Enter to continue.", id="question-hint")
            self.text_area = TextArea(id="question-input")
            yield self.text_area
            yield Button("Continue", id="question-continue")
        yield _footer_hint()

    def on_mount(self) -> None:
        question = (self.state.input_config.get("question") or {}).get("text")
        if question:
            self.text_area.text = str(question)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "question-continue":
            self.action_continue()

    def action_continue(self) -> None:
        text = self.text_area.text.strip()
        if not text:
            return
        self.state.input_config.setdefault("question", {})["text"] = text
        if self.state.config_mode == "guided":
            self.app.show_run_mode()
        else:
            self.app.show_review()


class RunModeScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="runmode-surface", classes="surface")
        surface.border_title = "Run Mode"
        with surface:
            yield Static("Quick Run uses recommended defaults.", id="runmode-hint")
            self.list = OptionList("Quick Run (recommended)", "Customize settings", id="runmode-list")
            yield self.list
        yield _footer_hint()

    def action_continue(self) -> None:
        choice = self.list.highlighted or 0
        self.state.use_customize = choice == 1
        if self.state.use_customize:
            self.app.show_decode()
        else:
            self.app.show_review()


class DecodeScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="decode-surface", classes="surface")
        surface.border_title = "Decode Parameters"
        with surface:
            self.policy = OptionList("Fixed temperature", "Range temperature", id="temp-policy")
            yield self.policy
            yield Input(placeholder="Temperature (fixed)", id="temp-fixed")
            yield Input(placeholder="Temperature min", id="temp-min")
            yield Input(placeholder="Temperature max", id="temp-max")
            yield Input(placeholder="Extra decode params (JSON)", id="decode-extra")
            yield Button("Continue", id="decode-continue")
        yield _footer_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "decode-continue":
            self.action_continue()

    def action_continue(self) -> None:
        q = self.state.input_config.setdefault("q", {})
        decode = q.setdefault("decode", {})
        if (self.policy.highlighted or 0) == 0:
            try:
                value = float(self.query_one("#temp-fixed", Input).value or "0.7")
            except ValueError:
                value = 0.7
            decode["temperature"] = {"type": "fixed", "value": value}
        else:
            try:
                temp_min = float(self.query_one("#temp-min", Input).value or "0.2")
                temp_max = float(self.query_one("#temp-max", Input).value or "1.0")
            except ValueError:
                temp_min, temp_max = 0.2, 1.0
            if temp_max < temp_min:
                temp_min, temp_max = temp_max, temp_min
            decode["temperature"] = {"type": "range", "min": temp_min, "max": temp_max}
        extra_raw = self.query_one("#decode-extra", Input).value.strip()
        if extra_raw:
            try:
                extra = json.loads(extra_raw)
            except json.JSONDecodeError:
                extra = {}
            if isinstance(extra, dict):
                decode["extra"] = extra
        decode.setdefault("extra", {})
        self.app.show_personas()


class PersonasScreen(BaseScreen):

    def compose(self) -> ComposeResult:
        surface = Container(id="personas-surface", classes="surface")
        surface.border_title = "Persona Mix"
        with surface:
            yield Static("Select personas to include.", id="personas-hint")
            self.list = MultiSelectList([], set())
            yield self.list
            self.add_input = Input(placeholder="Add persona id", id="persona-add")
            yield self.add_input
            yield Button("Add", id="persona-add-btn")
        yield _footer_hint()

    def on_mount(self) -> None:
        items, warning = load_persona_catalog()
        options = _catalog_options(items)
        if not options:
            options = [SelectOption(value="neutral", label="neutral")]
        if warning:
            self.query_one("#personas-hint", Static).update(warning)
        existing = [item.get("id") for item in (self.state.input_config.get("q", {})
                    .get("personas", {})
                    .get("items", [])) if item.get("id")]
        selected = set(existing) if existing else _default_catalog_selection(items, options)
        self.list.update_options(options, selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "persona-add-btn":
            value = self.add_input.value.strip()
            if value:
                self.list.add_custom(value)
                self.add_input.value = ""

    def on_multi_select_list_confirmed(self, message: MultiSelectList.Confirmed) -> None:
        q = self.state.input_config.setdefault("q", {})
        personas = q.setdefault("personas", {})
        selection = sorted(self.list.selected_values)
        personas["items"] = [{"id": value, "weight": 1.0} for value in selection]
        personas.setdefault("default_behavior", "neutral_if_empty")
        self.app.show_models()


class ModelsScreen(BaseScreen):

    def compose(self) -> ComposeResult:
        surface = Container(id="models-surface", classes="surface")
        surface.border_title = "Model Mix"
        with surface:
            yield Static("Select models to include.", id="models-hint")
            self.list = MultiSelectList([], set())
            yield self.list
            self.add_input = Input(placeholder="Add model slug", id="model-add")
            yield self.add_input
            yield Button("Add", id="model-add-btn")
        yield _footer_hint()

    def on_mount(self) -> None:
        items, warning = load_model_catalog()
        options = _catalog_options(items)
        if not options:
            options = [SelectOption(value=self.state.default_model, label=self.state.default_model)]
        if warning:
            self.query_one("#models-hint", Static).update(warning)
        existing = [item.get("slug") for item in (self.state.input_config.get("q", {})
                    .get("models", {})
                    .get("items", [])) if item.get("slug")]
        selected = set(existing) if existing else _default_catalog_selection(items, options)
        self.list.update_options(options, selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-add-btn":
            value = self.add_input.value.strip()
            if value:
                self.list.add_custom(value)
                self.add_input.value = ""

    def on_multi_select_list_confirmed(self, message: MultiSelectList.Confirmed) -> None:
        selection = sorted(self.list.selected_values)
        if not selection:
            return
        q = self.state.input_config.setdefault("q", {})
        models = q.setdefault("models", {})
        models["items"] = [{"slug": value, "weight": 1.0} for value in selection]
        self.app.show_protocol()


class ProtocolScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="protocol-surface", classes="surface")
        surface.border_title = "Protocol"
        with surface:
            self.message = Static("Select the protocol to apply.", id="protocol-hint")
            yield self.message
            self.list = OptionList("Independent (supported)", "Interaction (coming soon)", id="protocol-list")
            yield self.list
            yield Button("Continue", id="protocol-continue")
        yield _footer_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "protocol-continue":
            choice = self.list.highlighted or 0
            if choice == 1:
                self.message.update("Interaction protocol is not available yet.")
                return
            protocol = self.state.input_config.setdefault("protocol", {})
            protocol["type"] = "independent"
            self.app.show_advanced_gate()


class AdvancedGateScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="advanced-gate-surface", classes="surface")
        surface.border_title = "Advanced Settings"
        with surface:
            yield Static("Use recommended defaults or customize.", id="advanced-gate-hint")
            self.list = OptionList("Use defaults", "Customize advanced", id="advanced-gate-list")
            yield self.list
        yield _footer_hint()

    def action_continue(self) -> None:
        choice = self.list.highlighted or 0
        self.state.use_advanced = choice == 1
        if self.state.use_advanced:
            self.app.show_advanced()
        else:
            self.app.show_review()


class AdvancedScreen(BaseScreen):
    BINDINGS = [("enter", "continue", "Continue")]

    def compose(self) -> ComposeResult:
        surface = Container(id="advanced-surface", classes="surface")
        surface.border_title = "Advanced Settings"
        with surface:
            yield Input(placeholder="K_max (max trials)", id="adv-kmax")
            yield Input(placeholder="Workers (W)", id="adv-workers")
            yield Input(placeholder="Batch size (B)", id="adv-batch")
            yield Input(placeholder="Max retries", id="adv-retries")
            yield Input(placeholder="Seed (optional)", id="adv-seed")
            yield Input(placeholder="Parse failure policy (continue|halt)", id="adv-parse")
            yield Input(placeholder="delta_js", id="adv-delta")
            yield Input(placeholder="epsilon_new", id="adv-epsilon-new")
            yield Input(placeholder="epsilon_ci", id="adv-epsilon-ci")
            yield Input(placeholder="min_trials", id="adv-min")
            yield Input(placeholder="patience_batches", id="adv-patience")
            yield Input(placeholder="clustering method", id="adv-cluster-method")
            yield Input(placeholder="clustering tau", id="adv-cluster-tau")
            yield Input(placeholder="embed_text", id="adv-embed-text")
            yield Static("Embedding model (locked)", id="adv-embed-model")
            yield Static("Summarizer model (locked)", id="adv-summarizer")
            yield Button("Continue", id="advanced-continue")
        yield _footer_hint()

    def on_mount(self) -> None:
        execution = self.state.input_config.get("execution", {})
        convergence = self.state.input_config.get("convergence", {})
        clustering = self.state.input_config.get("clustering", {})
        summarizer = self.state.input_config.get("summarizer", {})

        self.query_one("#adv-kmax", Input).value = str(execution.get("k_max", 1000))
        self.query_one("#adv-workers", Input).value = str(execution.get("workers", 8))
        self.query_one("#adv-batch", Input).value = str(execution.get("batch_size", execution.get("workers", 8)))
        self.query_one("#adv-retries", Input).value = str(execution.get("retries", 2))
        seed_val = execution.get("seed")
        self.query_one("#adv-seed", Input).value = "" if seed_val is None else str(seed_val)
        self.query_one("#adv-parse", Input).value = str(execution.get("parse_failure_policy", "continue"))

        self.query_one("#adv-delta", Input).value = str(convergence.get("delta_js_threshold", 0.02))
        self.query_one("#adv-epsilon-new", Input).value = str(convergence.get("epsilon_new_threshold", 0.01))
        epsilon_ci = convergence.get("epsilon_ci_half_width", 0.05)
        self.query_one("#adv-epsilon-ci", Input).value = "" if epsilon_ci is None else str(epsilon_ci)
        self.query_one("#adv-min", Input).value = str(convergence.get("min_trials", 64))
        self.query_one("#adv-patience", Input).value = str(convergence.get("patience_batches", 2))

        self.query_one("#adv-cluster-method", Input).value = str(clustering.get("method", "leader"))
        self.query_one("#adv-cluster-tau", Input).value = str(clustering.get("tau", 0.85))
        self.query_one("#adv-embed-text", Input).value = str(clustering.get("embed_text", "outcome"))

        embedding_model = clustering.get("embedding_model", "n/a")
        summarizer_model = summarizer.get("model", "n/a")
        self.query_one("#adv-embed-model", Static).update(f"Embedding model (locked): {embedding_model}")
        self.query_one("#adv-summarizer", Static).update(f"Summarizer model (locked): {summarizer_model}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "advanced-continue":
            execution = self.state.input_config.setdefault("execution", {})
            convergence = self.state.input_config.setdefault("convergence", {})
            clustering = self.state.input_config.setdefault("clustering", {})

            execution["k_max"] = _int_or_default(self.query_one("#adv-kmax", Input).value, 1000)
            execution["workers"] = _int_or_default(self.query_one("#adv-workers", Input).value, 8)
            execution["batch_size"] = _int_or_default(self.query_one("#adv-batch", Input).value, execution["workers"])
            execution["retries"] = _int_or_default(self.query_one("#adv-retries", Input).value, 2)
            seed_value = _optional_int(self.query_one("#adv-seed", Input).value)
            if seed_value is not None:
                execution["seed"] = seed_value
            policy_value = self.query_one("#adv-parse", Input).value.strip().lower()
            if policy_value in {"continue", "halt"}:
                execution["parse_failure_policy"] = policy_value

            convergence["delta_js_threshold"] = _float_or_default(self.query_one("#adv-delta", Input).value, 0.02)
            convergence["epsilon_new_threshold"] = _float_or_default(
                self.query_one("#adv-epsilon-new", Input).value, 0.01
            )
            epsilon_raw = self.query_one("#adv-epsilon-ci", Input).value.strip()
            if epsilon_raw:
                try:
                    epsilon_ci = float(epsilon_raw)
                except ValueError:
                    epsilon_ci = 0.05
                convergence["epsilon_ci_half_width"] = epsilon_ci if epsilon_ci > 0 else None
            convergence["min_trials"] = _int_or_default(self.query_one("#adv-min", Input).value, 64)
            convergence["patience_batches"] = _int_or_default(self.query_one("#adv-patience", Input).value, 2)

            clustering["method"] = self.query_one("#adv-cluster-method", Input).value or "leader"
            clustering["tau"] = _float_or_default(self.query_one("#adv-cluster-tau", Input).value, 0.85)
            clustering["embed_text"] = self.query_one("#adv-embed-text", Input).value or "outcome"
            self.app.show_review()


class ReviewScreen(BaseScreen):
    BINDINGS = [("enter", "run", "Run")]

    def compose(self) -> ComposeResult:
        surface = Container(id="review-surface", classes="surface")
        surface.border_title = "Review"
        with surface:
            self.summary = Static("", id="review-summary")
            yield self.summary
            self.write_config = Checkbox("Write arbiter.config.json", value=True, id="review-write")
            yield self.write_config
            self.summarize = Checkbox(
                "Generate cluster summaries after run (remote only)",
                value=False,
                id="review-summarize",
            )
            yield self.summarize
            self.run_name = Input(placeholder="Run name", id="review-run-name")
            yield self.run_name
            self.output_dir = Input(placeholder="Output base directory", id="review-output")
            yield self.output_dir
            yield Button("Run", id="review-run")
        yield _footer_hint()

    def on_mount(self) -> None:
        self.output_dir.value = str(self.state.output_base_dir)
        self.run_name.value = self.state.run_name
        self.summary.update(_review_summary(self.state.input_config))
        summarizer_enabled = bool((self.state.input_config.get("summarizer") or {}).get("enabled", False))
        self.summarize.value = summarizer_enabled
        if self.state.selected_mode != "remote":
            self.summarize.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review-run":
            self.action_run()

    def action_run(self) -> None:
        self.state.run_name = self.run_name.value.strip() or "auto"
        self.state.output_base_dir = Path(self.output_dir.value.strip() or "./runs")
        llm_mode = "remote" if self.state.selected_mode == "remote" else "mock"
        self.state.input_config = normalize_canonical_config(
            self.state.input_config,
            default_model=self.state.default_model,
            llm_mode=llm_mode,
        )
        self.state.input_config.setdefault("summarizer", {})["enabled"] = bool(self.summarize.value)
        if self.write_config.value:
            write_json(self.state.config_path, self.state.input_config)
        self.app.show_execution()


class ExecutionScreen(BaseScreen):
    def compose(self) -> ComposeResult:
        execution_surface = Container(id="execution-surface", classes="surface")
        execution_surface.border_title = "Execution"
        with execution_surface:
            self.header = Static("", id="execution-header")
            yield self.header
            self.progress = ProgressBar(total=1, id="execution-progress")
            yield self.progress
            self.workers = Static("", id="execution-workers")
            yield self.workers
        yield execution_surface

        checkpoint_surface = Container(id="checkpoint-surface", classes="surface")
        checkpoint_surface.border_title = "BATCH CHECKPOINT"
        with checkpoint_surface:
            self.checkpoint = Static("", id="execution-checkpoint")
            yield self.checkpoint
        yield checkpoint_surface
        yield _footer_hint()

    def on_mount(self) -> None:
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.worker_state: dict[int, dict[str, Any]] = {}
        self.latest_checkpoint: dict[str, str] | None = None
        self.checkpoint.update(_checkpoint_text(None, None))
        self.set_interval(0.2, self._drain_events)
        asyncio.create_task(self._start_execution())

    async def _start_execution(self) -> None:
        try:
            setup = prepare_run(
                input_config=self.state.input_config,
                run_name=self.state.run_name,
                output_base_dir=self.state.output_base_dir,
                default_model=self.state.default_model,
                api_key_present=self.state.api_key_present,
                selected_mode=self.state.selected_mode,
            )
        except Exception as exc:  # noqa: BLE001
            self.header.update(f"Failed to prepare run: {exc}")
            return
        self.app.run_setup = setup
        self.base_header = _execution_header(setup)
        self.convergence = setup.resolved_config.semantic.execution.convergence
        self.header.update(self.base_header)
        on_event = lambda event: self.event_queue.put_nowait(event)
        result = await execute_trials(
            run_dir=setup.run_dir,
            resolved_config=setup.resolved_config,
            question=setup.question_record,
            on_event=on_event,
        )
        write_manifest(setup=setup, ended_at=datetime.now(timezone.utc), execution_result=result)
        self.app.execution_result = result
        self.app.show_receipt(setup.run_dir, result)

    def _drain_events(self) -> None:
        while not self.event_queue.empty():
            event = self.event_queue.get_nowait()
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "execution_started":
            call_cap = int(event.get("call_cap", 1))
            worker_count = int(event.get("worker_count", 0))
            self.progress.update(total=call_cap, progress=0)
            self.worker_state = {
                idx: {"status": "IDLE", "done": 0, "model": "—", "atom": "—", "persona": "none"}
                for idx in range(worker_count)
            }
            self._render_workers()
        elif event_type == "progress":
            completed = int(event.get("completed", 0))
            self.progress.update(progress=completed)
        elif event_type == "trial_started":
            worker_id = int(event.get("worker_id", 0))
            self.worker_state[worker_id] = {
                "status": "RUNNING",
                "done": self.worker_state.get(worker_id, {}).get("done", 0),
                "model": event.get("model", "—"),
                "atom": event.get("atom_id", "—"),
                "persona": event.get("persona_id") or "none",
            }
            self._render_workers()
        elif event_type == "trial_finished":
            worker_id = int(event.get("worker_id", 0))
            current = self.worker_state.get(worker_id, {})
            done = int(event.get("completed", current.get("done", 0)))
            self.worker_state[worker_id] = {
                "status": "IDLE",
                "done": done,
                "model": current.get("model", "—"),
                "atom": current.get("atom", "—"),
                "persona": current.get("persona", "none"),
            }
            self._render_workers()
        elif event_type == "batch_checkpoint":
            entry = event.get("entry") or {}
            stop_reason = event.get("stop_reason")
            self.latest_checkpoint = entry
            self.checkpoint.update(_checkpoint_text(entry, getattr(self, "convergence", None), stop_reason))
            entry = event.get("entry") or {}
            clusters = len(entry.get("counts_by_cluster_id") or {})
            top_p = entry.get("top_p")
            if hasattr(self, "base_header"):
                summary = f"Clusters discovered: {clusters}"
                if isinstance(top_p, (int, float)):
                    summary += f" · Top share: {top_p:.3f}"
                self.header.update(self.base_header + "\n" + summary)

    def _render_workers(self) -> None:
        lines = ["WID  STATUS    DONE  MODEL                     ATOM         PERSONA     "]
        for worker_id in sorted(self.worker_state.keys()):
            data = self.worker_state[worker_id]
            line = _format_worker_line(worker_id, data)
            lines.append(line)
        self.workers.update("\n".join(lines))


class ReceiptScreen(BaseScreen):
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, state, execution_result, top_clusters, checkpoints, run_dir: Path) -> None:  # type: ignore[no-untyped-def]
        super().__init__(state)
        self.execution_result = execution_result
        self.top_clusters = top_clusters
        self.checkpoints = checkpoints
        self.run_dir = run_dir

    def compose(self) -> ComposeResult:
        surface = Container(id="receipt-surface", classes="surface")
        surface.border_title = "Receipt"
        with surface:
            yield Static(
                _receipt_text(self.execution_result, self.top_clusters, self.checkpoints, self.run_dir),
                id="receipt",
            )
            yield Button("Exit", id="receipt-exit")
        yield _footer_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "receipt-exit":
            self.app.exit()


def _status_line(state) -> str:  # type: ignore[no-untyped-def]
    mode = "Remote" if state.selected_mode == "remote" else "Mock"
    openrouter = "Connected" if state.api_key_present else "Missing Key"
    config = "Found" if state.config_path.exists() else "Not Found"
    return f"Status  Mode {mode}   OpenRouter {openrouter}   Config {config}"


def _review_summary(input_config: dict[str, Any]) -> str:
    question = input_config.get("question", {})
    q = input_config.get("q", {})
    models = ", ".join(item.get("slug") for item in q.get("models", {}).get("items", []))
    personas = ", ".join(item.get("id") for item in q.get("personas", {}).get("items", []))
    decode = q.get("decode", {})
    temp = decode.get("temperature", {})
    temp_desc = "n/a"
    if isinstance(temp, dict):
        if temp.get("type") == "fixed":
            temp_desc = f"fixed {temp.get('value', 0.7)}"
        elif temp.get("type") == "range":
            temp_desc = f"range {temp.get('min', 0.2)}–{temp.get('max', 1.0)}"
    execution = input_config.get("execution", {})
    convergence = input_config.get("convergence", {})
    clustering = input_config.get("clustering", {})
    summarizer = input_config.get("summarizer", {})
    protocol = input_config.get("protocol", {})
    lines = [
        f"Question: {(question.get('text') or '').strip()[:80]}",
        f"Models: {models or 'none'}",
        f"Personas: {personas or 'none'}",
        f"Temperature: {temp_desc}",
        f"Protocol: {protocol.get('type', 'independent')}",
        f"K_max: {execution.get('k_max', 1000)}",
        f"Workers / batch: {execution.get('workers', 8)} / {execution.get('batch_size', execution.get('workers', 8))}",
        f"Seed: {execution.get('seed', 'auto')}",
        f"Parse failure policy: {execution.get('parse_failure_policy', 'continue')}",
        f"delta_js: {convergence.get('delta_js_threshold', 0.02)}",
        f"epsilon_new: {convergence.get('epsilon_new_threshold', 0.01)}",
        f"epsilon_ci: {convergence.get('epsilon_ci_half_width', 0.05)}",
        f"clustering: {clustering.get('method', 'leader')} (embed_text={clustering.get('embed_text', 'outcome')})",
        f"embedding model: {clustering.get('embedding_model', 'n/a')}",
        f"summarizer: {'enabled' if summarizer.get('enabled') else 'disabled'}",
    ]
    return "\n".join(lines)


def _execution_header(setup) -> str:  # type: ignore[no-untyped-def]
    semantic = setup.resolved_config.semantic
    question = setup.question_record.get("question_text", "")
    models = ", ".join(semantic.models) if semantic.models else "n/a"
    personas = ", ".join(semantic.personas.persona_ids) if semantic.personas.persona_ids else "none"
    mode_label = "remote (OpenRouter)" if semantic.llm.mode == "openrouter" else "mock (no network calls)"
    temp_policy = semantic.temperature_policy
    temp_desc = "fixed"
    if temp_policy.kind == "fixed":
        value = temp_policy.temperatures[0] if temp_policy.temperatures else 0.7
        temp_desc = f"fixed {value}"
    elif temp_policy.kind == "range":
        temp_min = temp_policy.temperatures[0] if temp_policy.temperatures else 0.2
        temp_max = temp_policy.temperatures[1] if len(temp_policy.temperatures) > 1 else temp_min
        temp_desc = f"range {temp_min}–{temp_max}"
    convergence = semantic.execution.convergence
    epsilon_ci = convergence.epsilon_ci_half_width
    epsilon_ci_display = "off" if epsilon_ci is None else epsilon_ci
    lines = [
        f"Run ID: {setup.run_id}",
        f"Started: {setup.started_at.isoformat()}",
        f"Question: {question.strip()[:80]}",
        f"Mode: {mode_label}",
        f"Models: {models}",
        f"Personas: {personas}",
        f"Temperature: {temp_desc}",
        f"Protocol: {semantic.protocol.type}",
        f"Embedding model: {semantic.clustering.embedding_model}",
        f"Cluster tau: {semantic.clustering.tau}",
        f"K_max: {semantic.trial_budget.k_max}",
        f"Workers / batch: {semantic.execution.worker_count} / {semantic.execution.batch_size}",
        f"Convergence: JS<{convergence.delta_js_threshold} · New<{convergence.epsilon_new_threshold} · CI {epsilon_ci_display}",
        f"Seed: {semantic.execution.seed}",
        f"Parse failure policy: {semantic.execution.parse_failure_policy}",
    ]
    return "\n".join(lines)


def _checkpoint_text(
    entry: dict[str, Any] | None,
    convergence: Any | None,
    stop_reason: str | None = None,
) -> str:
    header = "Batch  Trials  Clusters  JS (Δ)        New (Δ)       CI HW (Δ)    Stop"
    if not entry:
        return header
    batch = entry.get("batch_index", "—")
    trials = entry.get("trials_completed_total", "—")
    clusters = len(entry.get("counts_by_cluster_id") or {})
    js = entry.get("js_divergence")
    js_thresh = getattr(convergence, "delta_js_threshold", None)
    new_rate = entry.get("new_cluster_rate")
    new_thresh = getattr(convergence, "epsilon_new_threshold", None)
    ci_hw = entry.get("top_ci_half_width")
    ci_thresh = getattr(convergence, "epsilon_ci_half_width", None)
    js_text = _format_threshold(js, js_thresh)
    new_text = _format_threshold(new_rate, new_thresh)
    ci_text = _format_threshold(ci_hw, ci_thresh, allow_disabled=True)
    stop_text = "yes" if stop_reason else "no"
    return f"{batch:<5}  {trials:<6}  {clusters:<8}  {js_text:<12}  {new_text:<12}  {ci_text:<12}  {stop_text}"


def _format_worker_line(worker_id: int, data: dict[str, Any]) -> str:
    wid = f"W{worker_id + 1:02d}"
    status = str(data.get("status", "IDLE")).ljust(8)
    done = str(data.get("done", 0)).rjust(5)
    model = str(data.get("model", "—"))[:24].ljust(24)
    atom = str(data.get("atom", "—"))[:12].ljust(12)
    persona = str(data.get("persona", "none"))[:12].ljust(12)
    return f"{wid}  {status}  {done}  {model}  {atom}  {persona}"


def _receipt_text(result, top_clusters, checkpoints, run_dir: Path) -> str:  # type: ignore[no-untyped-def]
    aggregates = _load_json(run_dir / "aggregates.json")
    resolved = _load_json(run_dir / "config.resolved.json")
    convergence = (((resolved.get("semantic") or {}).get("execution") or {}).get("convergence") or {})
    execution = ((resolved.get("semantic") or {}).get("execution") or {})
    cluster_count = aggregates.get("discovered_cluster_count")
    entropy = aggregates.get("entropy")
    eff_num = aggregates.get("effective_num_clusters")
    patience = convergence.get("patience_batches")
    seed = execution.get("seed")
    parse_policy = execution.get("parse_failure_policy")
    stop_reason = result.stop_reason
    stop_explainer = stop_reason
    if stop_reason == "converged":
        stop_explainer = f"Converged after {patience} stable batches" if patience else "Converged"
    elif stop_reason == "max_trials_reached":
        stop_explainer = "Reached trial cap"
    elif stop_reason == "parse_failure":
        stop_explainer = "Stopped after parse failures (policy=halt)"
    elif stop_reason == "llm_error":
        stop_explainer = "Stopped after LLM error"
    elif stop_reason == "embedding_error":
        stop_explainer = "Stopped after embedding error"
    lines = [
        f"Stop reason: {stop_reason}",
        f"Why stopped: {stop_explainer}",
        f"Trials executed: {result.stop_at_trials}",
        f"Valid trials: {result.valid_trials}",
        f"Parse failures: {result.parse_error_count}",
        f"Batches: {result.batches_completed}",
    ]
    if cluster_count is not None:
        lines.append(f"Clusters discovered: {cluster_count}")
    if entropy is not None:
        lines.append(f"Entropy: {entropy:.3f}")
    if eff_num is not None:
        lines.append(f"Effective clusters: {eff_num:.2f}")
    if top_clusters:
        lines.append("Top clusters:")
        for cluster_id, share, exemplar in top_clusters:
            lines.append(f"- {cluster_id} · {share:.3f} · {exemplar}")
    if checkpoints:
        lines.append("Last checkpoints:")
        for row in checkpoints:
            lines.append("  ".join(row.values()))
    lines.append("Next steps:")
    if seed is not None:
        lines.append(f"- Rerun with seed: {seed}")
    if parse_policy:
        lines.append(f"- Parse failure policy: {parse_policy}")
    lines.append(f"- Artifacts: {run_dir}")
    lines.append("- Validate config: arbiter config validate --path arbiter.config.json")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _int_or_default(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: str) -> int | None:
    if value is None:
        return None
    trimmed = str(value).strip()
    if not trimmed:
        return None
    try:
        return int(trimmed)
    except ValueError:
        return None


def _format_threshold(value: float | None, threshold: float | None, *, allow_disabled: bool = False) -> str:
    if threshold is None:
        return "—" if allow_disabled else "n/a"
    if value is None:
        return "n/a"
    op = "<" if value <= threshold else ">"
    return f"{value:.3f}{op}{threshold:.3f}"


def _catalog_options(items) -> list[SelectOption]:  # type: ignore[no-untyped-def]
    options: list[SelectOption] = []
    for item in items:
        label = item.value
        if item.name and item.name.lower() != item.value.lower():
            label = f"{label} · {item.name}"
        if item.description:
            label = f"{label} — {item.description}"
        options.append(SelectOption(value=item.value, label=label))
    return options


def _default_catalog_selection(items, options) -> set[str]:  # type: ignore[no-untyped-def]
    defaults = {item.value for item in items if item.is_default}
    if defaults:
        return defaults
    if options:
        return {options[0].value}
    return set()
