"""LangGraph adapter for the existing Phase 9.5 delivery QA runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from graph.state import HonCutState


class FinalQaReport(Protocol):
    """Report fields consumed by the delivery-gate adapter."""

    verdict: str
    grade: str
    issues: Sequence[Any]

    def to_dict(self) -> dict[str, Any]: ...


class FinalQaRunner(Protocol):
    """Callable contract implemented by ``run_video_qa``."""

    def __call__(
        self,
        output_dir: Path,
        storyboard_data: dict[str, Any] | None = None,
    ) -> FinalQaReport: ...


def final_qa_node(
    state: HonCutState,
    *,
    runner: FinalQaRunner | None,
) -> dict[str, Any]:
    """Run the fail-closed Phase 9.5 delivery gate."""

    if state.get("dry_run", False):
        return {
            "status": "completed",
            "phase_results": {
                **state.get("phase_results", {}),
                "phase9_5": {"status": "skipped", "reason": "dry-run"},
            },
            "completed_phases": [*state.get("completed_phases", []), "phase9_5"],
            "skip_phase": state.get("skip_phase", []),
        }

    try:
        if runner is None:
            raise ImportError("video_qa not available")
        qa_report = runner(
            Path(state["output_dir"]),
            storyboard_data=state.get("storyboard"),
        )
    except ImportError:
        return {
            "status": "failed",
            "error": "Phase 9.5 delivery QA is unavailable",
            "phase_results": {
                **state.get("phase_results", {}),
                "phase9_5": {
                    "status": "error",
                    "reason": "video_qa not available",
                },
            },
            "completed_phases": state.get("completed_phases", []),
            "skip_phase": state.get("skip_phase", []),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"Phase 9.5 delivery QA failed to run: {exc}",
            "phase_results": {
                **state.get("phase_results", {}),
                "phase9_5": {"status": "error", "error": str(exc)},
            },
            "completed_phases": state.get("completed_phases", []),
            "skip_phase": state.get("skip_phase", []),
        }

    qa_passed = qa_report.verdict == "pass"
    phase_receipt = {
        "status": "done" if qa_passed else "error",
        "verdict": qa_report.verdict,
        "grade": qa_report.grade,
        "issues_count": len(qa_report.issues),
    }
    update: dict[str, Any] = {
        "final_video": state.get("final_video", ""),
        "status": "completed" if qa_passed else "failed",
        "phase_results": {
            **state.get("phase_results", {}),
            "phase9_5": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase9_5"] if qa_passed else []),
        "quality_report": qa_report.to_dict(),
        "skip_phase": state.get("skip_phase", []),
    }
    if not qa_passed:
        update["error"] = (
            f"Phase 9.5 delivery QA requires revision: {qa_report.grade} grade "
            f"({len(qa_report.issues)} issues)"
        )
    return update
