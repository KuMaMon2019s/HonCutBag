"""Build the backward-compatible Phase 4 continuity plan artifact."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from schemas.continuity import (
    ContinuityAnchors,
    ContinuityPlan,
    ContinuityShot,
    GenerationChunk,
)

DEFAULT_PROVIDER_CHUNK_LIMIT_S = 15.0
DEFAULT_TIMELINE_FPS = 24


def _chunk_frame_budgets(
    target_frames: int,
    limit_frames: int,
    overlap_frames: int,
    *,
    fps: int,
    initial_extension: bool = False,
) -> list[tuple[int, int, int]]:
    """Return requested, replay-overlap, and unique frame budgets per chunk."""
    if target_frames <= limit_frames and not initial_extension:
        return [(target_frames, 0, target_frames)]
    if overlap_frames >= limit_frames:
        raise ValueError("continuation overlap must be shorter than the provider chunk limit")

    usable_extension_frames = limit_frames - overlap_frames
    if initial_extension:
        chunk_count = max(1, math.ceil(target_frames / usable_extension_frames))
        overlap_count = chunk_count
    else:
        chunk_count = 1 + max(
            1,
            math.ceil(max(0, target_frames - limit_frames) / usable_extension_frames),
        )
        overlap_count = chunk_count - 1
    requested_total = target_frames + overlap_count * overlap_frames

    # Preserve whole-second provider requests whenever all inputs permit it.
    if requested_total % fps == 0 and limit_frames % fps == 0:
        total_units = requested_total // fps
        base, remainder = divmod(total_units, chunk_count)
        requested = [(base + (index < remainder)) * fps for index in range(chunk_count)]
    else:
        base, remainder = divmod(requested_total, chunk_count)
        requested = [base + (index < remainder) for index in range(chunk_count)]

    if max(requested) > limit_frames:
        raise ValueError("cannot fit overlap-aware chunk budget within provider limit")
    budgets = []
    for index, requested_frames in enumerate(requested):
        reserved_overlap = overlap_frames if initial_extension or index > 0 else 0
        unique_frames = requested_frames - reserved_overlap
        if unique_frames <= 0:
            raise ValueError("continuation overlap leaves no unique frames in a chunk")
        budgets.append((requested_frames, reserved_overlap, unique_frames))
    return budgets


def _shot_id(shot: Mapping[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip()
    if re.fullmatch(r"[Ss]?\d+", text):
        return f"S{int(text.lstrip('Ss')):02d}"
    return text or f"S{index:02d}"


def _target_duration(shot: Mapping[str, Any]) -> float:
    raw = shot.get("duration")
    if raw is None:
        raw = shot.get("suggested_duration")
    if raw is None:
        raw = 5.0
    return float(raw)


def _boundary_before(
    shot: Mapping[str, Any],
    previous_shot: Mapping[str, Any] | None,
    index: int,
) -> tuple[str, str]:
    from quality.shot_continuity import classify_boundary

    return classify_boundary(previous_shot, shot, index=index)


def _anchors(shot: Mapping[str, Any], scene_contract: Mapping[str, Any]) -> ContinuityAnchors:
    who = shot.get("who") or shot.get("characters") or []
    if not isinstance(who, list):
        who = [who] if who else []
    camera_motion = str(
        shot.get("camera_motion") or shot.get("camera_movement") or shot.get("camera") or ""
    )
    tracking_prompt = str(
        shot.get("continuity_subject")
        or shot.get("tracking_prompt")
        or shot.get("subject_description")
        or ", ".join(str(value) for value in who if value)
        or ""
    ).strip()
    return ContinuityAnchors(
        characters=[str(value) for value in who if value],
        scene=str(shot.get("where") or scene_contract.get("scene_description") or ""),
        screen_direction=str(shot.get("screen_direction") or shot.get("camera_axis") or ""),
        camera_motion=camera_motion,
        style=str(scene_contract.get("style_anchor") or scene_contract.get("style_suffix") or ""),
        tracking_prompt=tracking_prompt[:240],
    )


def build_continuity_plan(
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
    timeline_fps: int = DEFAULT_TIMELINE_FPS,
    continuation_overlap_s: float = 0.0,
) -> ContinuityPlan:
    """Split long editorial shots into a linear sequence of provider-sized chunks."""
    if provider_chunk_limit_s <= 0:
        raise ValueError("provider_chunk_limit_s must be positive")
    if timeline_fps <= 0:
        raise ValueError("timeline_fps must be positive")
    if continuation_overlap_s < 0:
        raise ValueError("continuation_overlap_s must not be negative")

    limit_frames = round(provider_chunk_limit_s * timeline_fps)
    overlap_frames = round(continuation_overlap_s * timeline_fps)

    scene_shots = (scene_consistency or {}).get("shots", {})
    planned_shots: list[ContinuityShot] = []
    cumulative_duration = 0.0
    previous_endpoint_frames = 0
    previous_storyboard_shot: Mapping[str, Any] | None = None
    previous_planned_shot: ContinuityShot | None = None
    group_number = 0
    for index, shot in enumerate(storyboard.get("shots", []), 1):
        shot_id = _shot_id(shot, index)
        boundary_before, continuity_reason = _boundary_before(
            shot,
            previous_storyboard_shot,
            index,
        )
        initial_extension = boundary_before == "continuous" and previous_planned_shot is not None
        if not initial_extension:
            group_number += 1
        continuity_group_id = f"CG{group_number:03d}"
        target_duration = _target_duration(shot)
        cumulative_duration += target_duration
        endpoint_frames = round(cumulative_duration * timeline_fps)
        target_frames = endpoint_frames - previous_endpoint_frames
        previous_endpoint_frames = endpoint_frames
        chunks: list[GenerationChunk] = []
        previous: str | None = None
        budgets = _chunk_frame_budgets(
            target_frames,
            limit_frames,
            overlap_frames,
            fps=timeline_fps,
            initial_extension=initial_extension,
        )
        if initial_extension:
            previous = previous_planned_shot.chunks[-1].chunk_id
        for sequence, (requested_frames, reserved_overlap, unique_frames) in enumerate(budgets, 1):
            chunk_id = f"{shot_id}_C{sequence:02d}"
            chunks.append(
                GenerationChunk(
                    chunk_id=chunk_id,
                    sequence=sequence,
                    target_duration_s=round(requested_frames / timeline_fps, 6),
                    requested_frames=requested_frames,
                    expected_overlap_frames=reserved_overlap,
                    expected_unique_frames=unique_frames,
                    mode="native_extend" if previous is not None else "fresh",
                    depends_on=previous,
                )
            )
            previous = chunk_id

        planned_shots.append(
            ContinuityShot(
                shot_id=shot_id,
                target_duration_s=target_duration,
                target_frames=target_frames,
                boundary_before=boundary_before,
                continuity_group_id=continuity_group_id,
                extends_from_shot_id=(
                    previous_planned_shot.shot_id if initial_extension else None
                ),
                extends_from_chunk_id=(
                    previous_planned_shot.chunks[-1].chunk_id if initial_extension else None
                ),
                continuity_reason=str(
                    shot.get("continuity_reason") or continuity_reason
                ).strip(),
                anchors=_anchors(shot, scene_shots.get(shot_id, {})),
                chunks=chunks,
            )
        )
        previous_storyboard_shot = shot
        previous_planned_shot = planned_shots[-1]
    return ContinuityPlan(
        provider_chunk_limit_s=provider_chunk_limit_s,
        timeline_fps=timeline_fps,
        shots=planned_shots,
    )


def write_continuity_plan(
    output_path: Path,
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
    timeline_fps: int = DEFAULT_TIMELINE_FPS,
    continuation_overlap_s: float = 0.0,
) -> ContinuityPlan:
    """Persist a JSON-safe continuity plan through an atomic replace."""
    plan = build_continuity_plan(
        storyboard,
        scene_consistency,
        provider_chunk_limit_s=provider_chunk_limit_s,
        timeline_fps=timeline_fps,
        continuation_overlap_s=continuation_overlap_s,
    )
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return plan
