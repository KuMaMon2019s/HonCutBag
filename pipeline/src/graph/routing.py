"""Pure deterministic routing decisions for the future HonCut workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from schemas.quality import ConsistencyResult, QAResult, ReshootDecision, SupervisionResult

MAX_QUALITY_ATTEMPTS = 2
MAX_RESHOOT_ATTEMPTS = 2

VideoGenerationMode = Literal["txt2vid", "img2vid", "reference"]
QARoute = Literal["generate", "repair", "failed"]
ConsistencyRoute = Literal["assemble", "regenerate", "failed"]
AssemblyRoute = Literal["post_process", "reshoot", "failed"]


def select_video_generation_mode(
    storyboard: Mapping[str, Any], storyboard_image: str
) -> VideoGenerationMode:
    """Preserve the live Phase 5 provider-selection precedence."""

    shots = storyboard.get("shots", [])
    if any(
        isinstance(shot, Mapping) and shot.get("ref_type") == "reference"
        for shot in shots
    ):
        return "reference"
    if storyboard_image:
        return "img2vid"
    return "txt2vid"


def route_phase5(state: Mapping[str, Any]) -> VideoGenerationMode:
    """Select the concrete Phase 6 node from checkpoint-safe graph state."""

    storyboard = state.get("storyboard", {})
    if not isinstance(storyboard, Mapping):
        storyboard = {}
    return select_video_generation_mode(
        storyboard,
        str(state.get("storyboard_image") or ""),
    )


def quality_gate_router(state: Mapping[str, Any]) -> Literal["pass", "block"]:
    """Block structural storyboard failures; Phase 8 owns pixel reshoots."""

    quality = state.get("quality_report", {})
    if not isinstance(quality, Mapping):
        quality = {}
    slideshow_risk = float(quality.get("slideshow_risk", 0.0))
    variation_score = float(quality.get("variation_score", 5.0))
    if slideshow_risk > 0.7 or variation_score < 3.0:
        print(
            "\n  ✗ 故事板质检不通过 "
            f"(slideshow_risk={slideshow_risk}, variation={variation_score})，"
            "阻断视频重拍"
        )
        return "block"
    return "pass"


def route_after_qa(
    qa_result: QAResult,
    supervision: SupervisionResult | None,
    repair_attempts: int,
    *,
    max_attempts: int = MAX_QUALITY_ATTEMPTS,
) -> QARoute:
    """Route storyboard QA without performing repair or generation work."""

    if repair_attempts < 0 or max_attempts < 1:
        raise ValueError("QA attempt counts must be non-negative with a positive maximum")
    blocked = (
        qa_result.passed is False
        or qa_result.verdict in {"block", "revise", "fail"}
        or (supervision is not None and supervision.verdict == "block")
    )
    if not blocked:
        return "generate"
    return "repair" if repair_attempts < max_attempts else "failed"


def route_after_consistency(
    consistency: ConsistencyResult,
    regeneration_attempts: int,
    *,
    max_attempts: int = MAX_QUALITY_ATTEMPTS,
) -> ConsistencyRoute:
    """Route failed-shot regeneration using only structured QA evidence."""

    if regeneration_attempts < 0 or max_attempts < 1:
        raise ValueError("consistency attempt counts must be non-negative with a positive maximum")
    failed = (
        not consistency.passed
        or bool(consistency.failed_shots)
        or (
            consistency.slideshow_risk is not None
            and consistency.slideshow_risk > 0.7
        )
        or (
            consistency.variation_score is not None
            and consistency.variation_score < 3.0
        )
    )
    if not failed:
        return "assemble"
    return "regenerate" if regeneration_attempts < max_attempts else "failed"


def route_after_assembly(
    decision: ReshootDecision,
    *,
    max_attempts: int = MAX_RESHOOT_ATTEMPTS,
) -> AssemblyRoute:
    """Route an explicit Phase 8 decision into post, reshoot, or failure."""

    if max_attempts < 1:
        raise ValueError("reshoot maximum must be positive")
    if not decision.required:
        return "post_process"
    if not decision.shot_ids or decision.attempt >= max_attempts:
        return "failed"
    return "reshoot"
