"""LangGraph adapter for the existing Phase 5 QA and supervision gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase5QaRunner(Protocol):
    """Callable contract implemented by ``run_storyboard_qa_gate``."""

    def __call__(self, output_dir: Path) -> dict[str, Any]: ...


class Phase5SupervisionRunner(Protocol):
    """Callable contract implemented by the existing supervision facade."""

    def __call__(
        self,
        storyboard: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]: ...


def phase5_node(
    state: HonCutState,
    *,
    qa_runner: Phase5QaRunner,
    supervision_runner: Phase5SupervisionRunner,
    supervision_blocked_error: type[Exception],
) -> dict[str, Any] | Command:
    """Run the existing QA and supervision gates without owning their logic."""

    output_dir = Path(state["output_dir"])
    phase_receipt = qa_runner(output_dir)
    update: dict[str, Any] = {
        "phase_results": {
            **state.get("phase_results", {}),
            "phase5": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase5"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=phase_receipt.get("error", "Phase 5 blocked Phase 6"),
        )
        return Command(goto=END, update=update)

    try:
        supervision = supervision_runner(
            state.get("storyboard", {}),
            output_dir,
        )
        update["phase_results"]["phase5"] = {
            **phase_receipt,
            "supervision": supervision,
        }
    except supervision_blocked_error as exc:
        update.update(status="failed", error=str(exc))
        return Command(goto=END, update=update)
    return update
