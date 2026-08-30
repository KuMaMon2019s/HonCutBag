"""Phase 1 director plus screenwriter composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION, DEFAULT_SHOT_POLICY
from phases.phase1.phase1_director import run_phase1_director
from phases.phase1.phase1_screenwriter import run_phase1_screenwriter
from runtime.phase_timing import _elapsed, _now
from utils.ark_llm import configure_heartbeat_callback
from utils.progress_reporter import ProgressReporter
from utils.canonical_visual_contracts import SOURCE_DERIVED_POLICY


def run_phase1(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
    screenplay_rewrite_request: dict[str, Any] | None = None,
    shot_policy: str = DEFAULT_SHOT_POLICY,
    max_material_padding_ratio: float = 0.25,
    delivery_overrun_ratio: float = 0.0,
    character_visual_policy: str = SOURCE_DERIVED_POLICY,
    *,
    _director_runner=None,
    _screenwriter_runner=None,
) -> dict:
    """Phase 1 composition; Screenwriter invokes Director after event extraction."""
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
        reporter.step("phase1", "文本解析与事件提取", progress_pct=1)
        reporter.start_heartbeat("phase1")
        configure_heartbeat_callback(
            lambda: reporter.step(
                "phase1",
                "Phase 1 LLM 流式响应",
                progress_pct=getattr(reporter, "_progress_pct", 1),
            )
        )
    director_runner = _director_runner or run_phase1_director
    screenwriter_runner = _screenwriter_runner or run_phase1_screenwriter
    try:
        screenwriter = screenwriter_runner(
            text,
            output_dir,
            duration,
            dry_run,
            reporter=reporter,
            shot_duration=shot_duration,
            project_video_spec=project_video_spec,
            screenplay_rewrite_request=screenplay_rewrite_request,
            shot_policy=shot_policy,
            max_material_padding_ratio=max_material_padding_ratio,
            delivery_overrun_ratio=delivery_overrun_ratio,
            character_visual_policy=character_visual_policy,
            _director_runner=director_runner,
        )
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()
    combined = dict(screenwriter)
    combined["duration_s"] = _elapsed(started)
    return combined
