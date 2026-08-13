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
        expected_width: int | None = None,
        expected_height: int | None = None,
        expected_min_duration: float | None = None,
        expected_max_duration: float | None = None,
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
        profile_name = state.get("media_profile", "1080p")
        aliases = {"1080p": "generic_hd", "youtube": "youtube_landscape"}
        width = height = None
        try:
            from utils.media_profiles import get_profile

            profile = get_profile(aliases.get(profile_name, profile_name))
            width, height = profile.width, profile.height
        except (ImportError, ValueError):
            fallbacks = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
            width, height = fallbacks.get(profile_name, (None, None))
        target_duration = state.get("duration")
        qa_report = runner(
            Path(state["output_dir"]),
            storyboard_data=state.get("storyboard"),
            expected_width=width,
            expected_height=height,
            expected_min_duration=(None if target_duration is None else float(target_duration) - 1.0),
            expected_max_duration=(None if target_duration is None else float(target_duration) + 1.0),
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
