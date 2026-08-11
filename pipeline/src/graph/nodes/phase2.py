"""LangGraph adapter for the existing Phase 2 storyboard-image runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase2Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase2``."""

    def __call__(
        self,
        storyboard_data: dict[str, Any] | None,
        characters_data: dict[str, Any],
        output_dir: Path,
        dry_run: bool,
    ) -> dict[str, Any]: ...


def phase2_node(
    state: HonCutState,
    *,
    runner: Phase2Runner,
) -> dict[str, Any] | Command:
    """Call Phase 2 and return only its compatibility State patch."""

    phase_receipt = runner(
        storyboard_data=state.get("storyboard"),
        characters_data={"characters": state.get("characters", [])},
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
    )

    storyboard_image = ""
    if phase_receipt.get("status") == "done" and phase_receipt.get("outputs"):
        storyboard_image = phase_receipt["outputs"][0]

    update: dict[str, Any] = {
        "storyboard_image": storyboard_image,
        "phase_results": {
            **state.get("phase_results", {}),
            "phase2": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase2"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 2 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
