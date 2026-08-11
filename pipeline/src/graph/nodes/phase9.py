"""LangGraph adapter for the existing Phase 9 post-production runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase9Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase9``."""

    def __call__(
        self,
        output_dir: Path,
        dry_run: bool,
        media_profile: str = "1080p",
    ) -> dict[str, Any]: ...


def route_after_phase9(state: HonCutState) -> Literal["continue", "end"]:
    """Route successful post-production to delivery QA and failures to END."""

    phase_receipt = state.get("phase_results", {}).get("phase9", {})
    if state.get("status") == "failed" or phase_receipt.get("status") == "error":
        return "end"
    return "continue"


def phase9_node(
    state: HonCutState,
    *,
    runner: Phase9Runner,
) -> dict[str, Any] | Command:
    """Run post-production while leaving media processing in Phase 9."""

    phase_receipt = runner(
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
        media_profile=state.get("media_profile", "1080p"),
    )
    outputs = phase_receipt.get("outputs", [])
    final_video = ""
    if phase_receipt.get("status") == "done" and outputs:
        final_video = next(
            (output for output in outputs if Path(output).name == "polished.mp4"),
            outputs[0],
        )

    update: dict[str, Any] = {
        "final_video": final_video,
        "phase_results": {
            **state.get("phase_results", {}),
            "phase9": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase9"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 9 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
