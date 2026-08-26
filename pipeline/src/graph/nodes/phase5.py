"""LangGraph adapter for the existing Phase 5 QA and supervision gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState
from phases.phase5.replanning import (
    MAX_PADDING_SCREENPLAY_REWRITES,
    rewrite_attempt_from_receipt,
    rewrite_request_from_receipt,
)


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
    previous_phase_receipt = state.get("phase_results", {}).get("phase5")
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
        rewrite_request = rewrite_request_from_receipt(phase_receipt)
        previous_attempt = rewrite_attempt_from_receipt(previous_phase_receipt)
        if (
            rewrite_request is not None
            and previous_attempt < MAX_PADDING_SCREENPLAY_REWRITES
        ):
            correction = dict(phase_receipt.get("correction") or {})
            correction.update(
                status="rewrite_scheduled",
                screenplay_rewrite_attempt=previous_attempt + 1,
            )
            scheduled_receipt = {
                **phase_receipt,
                "correction": correction,
            }
            update["phase_results"]["phase5"] = scheduled_receipt
            update.update(
                status="running",
                completed_phases=[],
            )
            return update
        if rewrite_request is not None:
            correction = dict(phase_receipt.get("correction") or {})
            correction.update(
                status="rewrite_exhausted",
                screenplay_rewrite_attempt=previous_attempt,
            )
            phase_receipt = {
                **phase_receipt,
                "correction": correction,
                "error": (
                    "Phase 5 padding budget still blocks Phase 6 after "
                    f"{previous_attempt}/{MAX_PADDING_SCREENPLAY_REWRITES} "
                    "screenplay rewrite"
                ),
            }
            update["phase_results"]["phase5"] = phase_receipt
        update.update(
            status="failed",
            error=phase_receipt.get("error", "Phase 5 blocked Phase 6"),
        )
        return Command(goto=END, update=update)

    if state.get("dry_run"):
        update["phase_results"]["phase5"] = {
            **phase_receipt,
            "supervision": {"status": "skipped", "reason": "dry-run"},
        }
        return update

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
