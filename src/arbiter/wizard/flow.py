"""Decision-tree wizard flow."""

from __future__ import annotations

from arbiter.config import normalize_canonical_config
from arbiter.ui.render import render_info
from arbiter.wizard.state import WizardState
from arbiter.wizard.steps import run_step


def run_wizard(state: WizardState) -> WizardState:
    step_id = "welcome"
    while step_id:
        run_step(step_id, state)
        step_id = _next_step(state, step_id)

    state.input_config = normalize_canonical_config(
        state.input_config,
        default_model=state.default_model,
        llm_mode="remote" if state.api_key_present else "mock",
    )
    render_info("Wizard complete. Proceeding to run execution.")
    return state


def _next_step(state: WizardState, step_id: str) -> str | None:
    if step_id == "welcome":
        return "config_mode"
    if step_id == "config_mode":
        if state.config_mode == "guided":
            return "question"
        if state.config_mode == "load":
            question_text = (state.input_config.get("question", {}) or {}).get("text")
            if not question_text:
                return "question"
            return "review"
    if step_id == "question":
        if state.config_mode == "guided":
            return "decode"
        return "review"
    if step_id == "decode":
        return "personas"
    if step_id == "personas":
        return "models"
    if step_id == "models":
        return "protocol"
    if step_id == "protocol":
        return "advanced_gate"
    if step_id == "advanced_gate":
        if state.use_advanced:
            return "advanced"
        return "review"
    if step_id == "advanced":
        return "review"
    if step_id == "review":
        return "run_setup"
    if step_id == "run_setup":
        return None
    return None
