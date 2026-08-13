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


def _chunk_durations(target_duration: float, limit: float) -> list[float]:
    """Balance long shots so a split never leaves a tiny unusable tail chunk."""
    if target_duration <= limit:
        return [target_duration]

    chunk_count = math.ceil(target_duration / limit)
    if target_duration.is_integer():
        whole_seconds = int(target_duration)
        base, remainder = divmod(whole_seconds, chunk_count)
        return [float(base + (index < remainder)) for index in range(chunk_count)]

    balanced = round(target_duration / chunk_count, 6)
    durations = [balanced] * (chunk_count - 1)
    durations.append(round(target_duration - sum(durations), 6))
    return durations


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


def _boundary_before(shot: Mapping[str, Any], index: int) -> str:
    if index == 1:
        return "cut"
    explicit = (
        str(shot.get("boundary_before") or shot.get("continuity_boundary") or "").strip().lower()
    )
    return "continuous" if explicit in {"continuous", "continue"} else "cut"


def _anchors(shot: Mapping[str, Any], scene_contract: Mapping[str, Any]) -> ContinuityAnchors:
    who = shot.get("who") or shot.get("characters") or []
    if not isinstance(who, list):
        who = [who] if who else []
    camera_motion = str(
        shot.get("camera_motion") or shot.get("camera_movement") or shot.get("camera") or ""
    )
    return ContinuityAnchors(
        characters=[str(value) for value in who if value],
        scene=str(shot.get("where") or scene_contract.get("scene_description") or ""),
        screen_direction=str(shot.get("screen_direction") or shot.get("camera_axis") or ""),
        camera_motion=camera_motion,
        style=str(scene_contract.get("style_anchor") or scene_contract.get("style_suffix") or ""),
    )


def build_continuity_plan(
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
) -> ContinuityPlan:
    """Split long editorial shots into a linear sequence of provider-sized chunks."""
    if provider_chunk_limit_s <= 0:
        raise ValueError("provider_chunk_limit_s must be positive")

    scene_shots = (scene_consistency or {}).get("shots", {})
    planned_shots: list[ContinuityShot] = []
    for index, shot in enumerate(storyboard.get("shots", []), 1):
        shot_id = _shot_id(shot, index)
        target_duration = _target_duration(shot)
        chunks: list[GenerationChunk] = []
        previous: str | None = None
        for sequence, chunk_duration in enumerate(
            _chunk_durations(target_duration, provider_chunk_limit_s), 1
        ):
            chunk_id = f"{shot_id}_C{sequence:02d}"
            chunks.append(
                GenerationChunk(
                    chunk_id=chunk_id,
                    sequence=sequence,
                    target_duration_s=round(chunk_duration, 6),
                    mode="fresh" if sequence == 1 else "native_extend",
                    depends_on=previous,
                )
            )
            previous = chunk_id

        planned_shots.append(
            ContinuityShot(
                shot_id=shot_id,
                target_duration_s=target_duration,
                boundary_before=_boundary_before(shot, index),
                anchors=_anchors(shot, scene_shots.get(shot_id, {})),
                chunks=chunks,
            )
        )
    return ContinuityPlan(
        provider_chunk_limit_s=provider_chunk_limit_s,
        shots=planned_shots,
    )


def write_continuity_plan(
    output_path: Path,
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
) -> ContinuityPlan:
    """Persist a JSON-safe continuity plan through an atomic replace."""
    plan = build_continuity_plan(
        storyboard,
        scene_consistency,
        provider_chunk_limit_s=provider_chunk_limit_s,
    )
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return plan
