"""LangGraph adapter for the existing Phase 3 character-asset runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase3Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase3``."""

    def __call__(
        self,
        output_dir: Path,
        characters_data: dict[str, Any],
        dry_run: bool,
    ) -> dict[str, Any]: ...


def phase3_node(
    state: HonCutState,
    *,
    runner: Phase3Runner,
) -> dict[str, Any] | Command:
    """Call Phase 3 and return only its compatibility State patch."""

    phase_receipt = runner(
        output_dir=Path(state["output_dir"]),
        characters_data={"characters": state.get("characters", [])},
        dry_run=state["dry_run"],
    )
    update: dict[str, Any] = {
        "phase_results": {
            **state.get("phase_results", {}),
            "phase3": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase3"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 3 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
