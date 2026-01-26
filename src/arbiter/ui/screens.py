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
            self.mode_list = OptionList(
                "Mock (no network calls)",
                "Remote (OpenRouter)",
                id="mode-list",
            )
            yield self.mode_list
            yield Static("Press Enter to begin", id="welcome-hint")
        yield _footer_hint()

    def on_mount(self) -> None:
        if not self.state.api_key_present:
            self.mode_list.clear_options()
            self.mode_list.add_option("Mock (no network calls)")
            self.mode_list.add_option("Remote (requires OPENROUTER_API_KEY)")
        index = 1 if self.state.selected_mode == "remote" else 0
        self.mode_list.highlighted = index

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
            options = ["Load config", "Guided wizard", "Create template & exit"]
            self.list = OptionList(*options, id="config-list")
            yield self.list
            yield Static("Press Enter to continue", id="config-hint")
        yield _footer_hint()

    def on_mount(self) -> None:
        if self.state.config_path.exists():
            self.list.highlighted = 0
        else:
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
            yield Static("Paste question. End with an empty line.", id="question-hint")
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
            yield Static("Choose Quick Run or Customize.", id="runmode-hint")
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
            yield Static("Independent protocol only.", id="protocol-hint")
            yield Button("Continue", id="protocol-continue")
        yield _footer_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "protocol-continue":
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
            yield Input(placeholder="delta_js", id="adv-delta")
            yield Input(placeholder="epsilon_new", id="adv-epsilon-new")
            yield Input(placeholder="epsilon_ci", id="adv-epsilon-ci")
            yield Input(placeholder="min_trials", id="adv-min")
            yield Input(placeholder="patience_batches", id="adv-patience")
            yield Input(placeholder="clustering method", id="adv-cluster-method")
            yield Input(placeholder="clustering tau", id="adv-cluster-tau")
            yield Input(placeholder="embed_text", id="adv-embed-text")
            yield Button("Continue", id="advanced-continue")
        yield _footer_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "advanced-continue":
            execution = self.state.input_config.setdefault("execution", {})
            convergence = self.state.input_config.setdefault("convergence", {})
            clustering = self.state.input_config.setdefault("clustering", {})

            execution["k_max"] = _int_or_default(self.query_one("#adv-kmax", Input).value, 1000)
            execution["workers"] = _int_or_default(self.query_one("#adv-workers", Input).value, 8)
            execution["batch_size"] = _int_or_default(self.query_one("#adv-batch", Input).value, execution["workers"])
            execution["retries"] = _int_or_default(self.query_one("#adv-retries", Input).value, 2)

            convergence["delta_js_threshold"] = _float_or_default(self.query_one("#adv-delta", Input).value, 0.02)
            convergence["epsilon_new_threshold"] = _float_or_default(
                self.query_one("#adv-epsilon-new", Input).value, 0.01
            )
            convergence["epsilon_ci_half_width"] = _float_or_default(
                self.query_one("#adv-epsilon-ci", Input).value, 0.05
            )
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
                "Generate cluster summaries (remote only)",
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
        surface = Container(id="execution-surface", classes="surface")
        surface.border_title = "Execution"
        with surface:
            self.header = Static("", id="execution-header")
            yield self.header
            self.progress = ProgressBar(total=1, id="execution-progress")
            yield self.progress
            self.workers = Static("", id="execution-workers")
            yield self.workers
            self.checkpoint = Static("", id="execution-checkpoint")
            yield self.checkpoint
        yield _footer_hint()

    def on_mount(self) -> None:
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.worker_state: dict[int, dict[str, Any]] = {}
        self.latest_checkpoint: dict[str, str] | None = None
        self.checkpoint.update(_checkpoint_text(None))
        self.set_interval(0.1, self._drain_events)
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
            self.latest_checkpoint = event.get("row")
            self.checkpoint.update(_checkpoint_text(self.latest_checkpoint))
            entry = event.get("entry") or {}
            clusters = len(entry.get("counts_by_cluster_id") or {})
            top_p = entry.get("top_p")
            if hasattr(self, "base_header"):
                summary = f"Clusters discovered: {clusters}"
                if isinstance(top_p, (int, float)):
                    summary += f" · Top share: {top_p:.3f}"
                self.header.update(self.base_header + "\n" + summary)

    def _render_workers(self) -> None:
        lines = ["WID  STATUS   DONE  MODEL                  ATOM         PERSONA"]
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
    mode = state.selected_mode.upper()
    openrouter = "CONNECTED" if state.api_key_present else "MISSING KEY"
    config = "FOUND" if state.config_path.exists() else "NOT FOUND"
    return f"Status  MODE {mode}   OPENROUTER {openrouter}   CONFIG {config}"


def _review_summary(input_config: dict[str, Any]) -> str:
    question = input_config.get("question", {})
    q = input_config.get("q", {})
    models = ", ".join(item.get("slug") for item in q.get("models", {}).get("items", []))
    personas = ", ".join(item.get("id") for item in q.get("personas", {}).get("items", []))
    execution = input_config.get("execution", {})
    convergence = input_config.get("convergence", {})
    clustering = input_config.get("clustering", {})
    summarizer = input_config.get("summarizer", {})
    lines = [
        f"Question: {(question.get('text') or '').strip()[:80]}",
        f"Models: {models or 'none'}",
        f"Personas: {personas or 'none'}",
        f"K_max: {execution.get('k_max', 1000)}",
        f"Workers / batch: {execution.get('workers', 8)} / {execution.get('batch_size', execution.get('workers', 8))}",
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
    lines = [
        f"Run ID: {setup.run_id}",
        f"Started: {setup.started_at.isoformat()}",
        f"Question: {question.strip()[:80]}",
        f"Mode: {mode_label}",
        f"Models: {models}",
        f"Personas: {personas}",
        f"Embedding model: {semantic.clustering.embedding_model}",
        f"Cluster tau: {semantic.clustering.tau}",
        f"K_max: {semantic.trial_budget.k_max}",
        f"Workers / batch: {semantic.execution.worker_count} / {semantic.execution.batch_size}",
    ]
    return "\n".join(lines)


def _checkpoint_text(row: dict[str, str] | None) -> str:
    if not row:
        return "Batch  Trials  Clusters  JS     New    CI HW  Stop"
    return " ".join([str(row.get(key, "")) for key in ["Batch", "Trials", "Clusters", "JS", "New", "CI HW", "Stop"]])


def _format_worker_line(worker_id: int, data: dict[str, Any]) -> str:
    wid = f"W{worker_id + 1:02d}"
    status = str(data.get("status", "IDLE")).ljust(7)
    done = str(data.get("done", 0)).rjust(4)
    model = str(data.get("model", "—"))[:22].ljust(22)
    atom = str(data.get("atom", "—"))[:12].ljust(12)
    persona = str(data.get("persona", "none"))[:10].ljust(10)
    return f"{wid}  {status}  {done}  {model}  {atom}  {persona}"


def _receipt_text(result, top_clusters, checkpoints, run_dir: Path) -> str:  # type: ignore[no-untyped-def]
    aggregates = _load_json(run_dir / "aggregates.json")
    cluster_count = aggregates.get("discovered_cluster_count")
    entropy = aggregates.get("entropy")
    eff_num = aggregates.get("effective_num_clusters")
    lines = [
        f"Stop reason: {result.stop_reason}",
        f"Trials executed: {result.stop_at_trials}",
        f"Valid trials: {result.valid_trials}",
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
    lines.append(f"Artifacts: {run_dir}")
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
