"""Phase 1 director plus screenwriter composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from phases.phase1.phase1_director import run_phase1_director
from phases.phase1.phase1_screenwriter import run_phase1_screenwriter
from runtime.phase_timing import _elapsed, _now
from utils.ark_llm import configure_heartbeat_callback
from utils.progress_reporter import ProgressReporter


def run_phase1(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
    *,
    _director_runner=None,
    _screenwriter_runner=None,
) -> dict:
    """Phase 1: director planning followed by the screenwriter engine."""
    started = _now()
    if reporter:
        # LangGraph enters the combined runner directly, so establish visible
        # progress before the director's first network request.  Sequential
        # execution may already have called phase_start; avoid a duplicate.
        if (
            getattr(reporter, "_current_phase", None) != "phase1"
            and hasattr(reporter, "phase_start")
        ):
            reporter.phase_start("phase1", "导演拆解 + 编剧引擎")
        reporter.step("phase1", "导演规划", progress_pct=1)
        reporter.start_heartbeat("phase1")
        configure_heartbeat_callback(
            lambda: reporter.step(
                "phase1",
                "导演规划 LLM 流式响应",
                progress_pct=getattr(reporter, "_progress_pct", 1),
            )
        )
    director_runner = _director_runner or run_phase1_director
    screenwriter_runner = _screenwriter_runner or run_phase1_screenwriter
    try:
        director = director_runner(text, Path(output_dir), dry_run)
        screenwriter = screenwriter_runner(
            text,
            output_dir,
            duration,
            dry_run,
            reporter=reporter,
            shot_duration=shot_duration,
            project_video_spec=project_video_spec,
        )
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()
    combined = dict(screenwriter)
    combined["director"] = director
    combined["duration_s"] = _elapsed(started)
    return combined
