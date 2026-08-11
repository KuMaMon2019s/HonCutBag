"""LangGraph adapter for the existing Phase 4 orchestration runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase4Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase4``."""

    def __call__(
        self,
        output_dir: Path,
        dry_run: bool,
    ) -> dict[str, Any]: ...


def phase4_node(
    state: HonCutState,
    *,
    runner: Phase4Runner,
) -> dict[str, Any] | Command:
    """Call Phase 4 and return only its compatibility State patch."""

    phase_receipt = runner(
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
    )
    update: dict[str, Any] = {
        "phase_results": {
            **state.get("phase_results", {}),
            "phase4": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase4"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 4 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
