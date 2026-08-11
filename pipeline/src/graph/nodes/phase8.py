"""LangGraph adapter for the existing Phase 8 assembly and reshoot runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase8Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase8``."""

    def __call__(
        self,
        output_dir: Path,
        dry_run: bool,
        transition: str = "crossfade",
        transition_duration: float = 0.5,
        media_profile: str = "1080p",
        target_duration: float | None = None,
        enable_reshoot: bool = True,
        chain_mode: bool = False,
    ) -> dict[str, Any]: ...


def route_after_phase8(state: HonCutState) -> Literal["continue", "end"]:
    """Prevent failed assembly from following the static Phase 9 path."""

    phase_receipt = state.get("phase_results", {}).get("phase8", {})
    if state.get("status") == "failed" or phase_receipt.get("status") == "error":
        return "end"
    return "continue"


def phase8_node(
    state: HonCutState,
    *,
    runner: Phase8Runner,
) -> dict[str, Any] | Command:
    """Run assembly while keeping all visual-QA and reshoot logic in Phase 8."""

    phase_receipt = runner(
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
        transition=state.get("transition", "crossfade"),
        transition_duration=state.get("transition_duration", 0.5),
        media_profile=state.get("media_profile", "1080p"),
        target_duration=state.get("duration"),
        enable_reshoot=state.get("enable_reshoot", True),
        chain_mode=state.get("chain_mode", False),
    )
    update: dict[str, Any] = {
        "phase_results": {
            **state.get("phase_results", {}),
            "phase8": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase8"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 8 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
