"""Plan the per-shot visual beats that Phase 2 draws and Phase 6 executes."""

from __future__ import annotations

import math
import re
from typing import Any

from utils.video_capabilities import (
    SEEDANCE_2_CAPABILITIES,
    VideoModelCapabilities,
    capabilities_for,
)


MIN_BEAT_SECONDS = int(SEEDANCE_2_CAPABILITIES.min_unique_beat_s)
MAX_BEAT_SECONDS = int(SEEDANCE_2_CAPABILITIES.max_unique_beat_s)
MAX_ACTIONS_PER_BEAT = SEEDANCE_2_CAPABILITIES.max_micro_actions_per_beat


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _compact(value: Any, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _partition(values: list[str], count: int) -> list[list[str]]:
    """Distribute every ordered action across beats without sampling or loss."""
    if not values:
        return [[] for _ in range(count)]
    buckets: list[list[str]] = []
    base, remainder = divmod(len(values), count)
    cursor = 0
    for position in range(count):
        size = base + (1 if position < remainder else 0)
        buckets.append(values[cursor:cursor + size])
        cursor += size
    return buckets


def _duration_budgets(total: float, count: int) -> list[float]:
    """Distribute an editorial duration across provider-friendly whole seconds."""
    rounded_total = round(total)
    if math.isclose(total, rounded_total, abs_tol=1e-6):
        base, remainder = divmod(int(rounded_total), count)
        return [float(base + (position < remainder)) for position in range(count)]
    base = total / count
    values = [round(base, 6) for _ in range(count)]
    values[-1] = round(total - sum(values[:-1]), 6)
    return values


def _beat_count(
    shot: dict[str, Any],
    duration: float,
    actions: list[str],
    capabilities: VideoModelCapabilities,
) -> int:
    """Choose Pxx count from authored semantics, bounded by provider duration."""
    existing = shot.get("storyboard_beats")
    if isinstance(existing, list) and existing:
        return len(existing)
    explicit = int(shot.get("storyboard_beat_count") or 0)
    raw_units = shot.get("source_action_unit_ids") or []
    if isinstance(raw_units, str):
        raw_units = [raw_units]
    action_units = len({str(value) for value in raw_units if str(value).strip()})
    action_count = math.ceil(len(actions) / capabilities.max_micro_actions_per_beat)
    semantic_count = max(
        1,
        explicit,
        action_units,
        action_count,
    )
    duration_count = max(1, math.ceil(duration / capabilities.max_unique_beat_s))
    capacity = max(1, int(duration // capabilities.min_unique_beat_s))
    if semantic_count > capacity:
        raise ValueError(
            f"{_shot_id(shot, 1)} cannot fit {len(actions)} micro-actions into "
            f"{duration:g}s for {capabilities.name}: requires {semantic_count} beats, "
            f"but the duration supports at most {capacity}"
        )
    return min(capacity, max(semantic_count, duration_count))


def _action_for_bucket(
    bucket: list[str],
    *,
    position: int,
    count: int,
    fallback_action: str,
    final_state: str,
) -> str:
    if bucket:
        return " → ".join(bucket)
    if position == count:
        return _compact(
            f"完成本镜动作并稳定到结束状态：{final_state or fallback_action}"
        )
    return _compact(f"继续推进本镜动作：{fallback_action}")


def plan_storyboard_beats(
    storyboard: dict[str, Any],
    capabilities: VideoModelCapabilities | None = None,
) -> dict[str, Any]:
    """Attach one fresh/extend execution ladder to every editorial shot."""
    total = 0
    for index, shot in enumerate(storyboard.get("shots", []), 1):
        if not isinstance(shot, dict):
            continue
        profile = capabilities or capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index)
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
        source_actions = shot.get("micro_actions") or shot.get("generation_actions") or []
        if isinstance(source_actions, str):
            source_actions = [source_actions]
        source_actions = [
            _compact(value)
            for value in source_actions
            if str(value).strip()
        ]
        if not source_actions:
            narrative = str(
                shot.get("action_description")
                or shot.get("what")
                or shot.get("visual")
                or ""
            )
            source_actions = []
            for value in re.split(r"[。！？!?；;\n]+", narrative):
                raw_value = value.strip()
                if raw_value.startswith(("“", "”", '"', "『", "「", "』", "」")):
                    continue
                candidate = _compact(value).strip("“”\"'，,：:、 ")
                if len(candidate) >= 2 and re.search(r"[\w\u3400-\u9fff]", candidate):
                    source_actions.append(candidate)
        fallback_action = _compact(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or "保持当前场景中的自然表演"
        )
        if not source_actions:
            source_actions = [fallback_action]
        count = _beat_count(shot, duration, source_actions, profile)
        action_buckets = _partition(source_actions, count)
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        action_units = list(dict.fromkeys(
            str(value) for value in raw_units if str(value).strip()
        ))
        unit_buckets = _partition(action_units, count)
        durations = _duration_budgets(duration, count)
        start_state = _compact(
            shot.get("start_state")
            or shot.get("prev_shot_context")
            or shot.get("what")
        )
        final_state = _compact(shot.get("end_state") or shot.get("what"))
        existing_beats = [
            beat for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        beats: list[dict[str, Any]] = []
        continues_previous = (
            str(shot.get("boundary_before") or "").strip().lower() == "continuous"
        )
        for position in range(1, count + 1):
            existing_beat = existing_beats[position - 1] if position <= len(existing_beats) else {}
            action = _compact(existing_beat.get("action")) or _action_for_bucket(
                action_buckets[position - 1],
                position=position,
                count=count,
                fallback_action=fallback_action,
                final_state=final_state,
            )
            previous_state = beats[-1]["end_state"] if beats else start_state
            next_state = (
                final_state
                if position == count
                else _compact(f"已完成本格动作：{action}")
            )
            normalized = dict(existing_beat)
            normalized.update({
                "beat_id": f"{sid}_P{position:02d}",
                "position": position,
                "duration_s": durations[position - 1],
                "generation_mode": (
                    "extend" if position > 1 or continues_previous else "fresh"
                ),
                "start_state": _compact(existing_beat.get("start_state")) or previous_state,
                "action": action,
                "micro_actions": action_buckets[position - 1],
                "source_action_unit_ids": unit_buckets[position - 1],
                "end_state": _compact(existing_beat.get("end_state")) or next_state,
                "shot_size": existing_beat.get("shot_size")
                or shot.get("shot_size") or shot.get("shot_type"),
                "camera_movement": existing_beat.get("camera_movement")
                or shot.get("camera_movement") or shot.get("camera_movement_en"),
            })
            beats.append(normalized)
        shot["storyboard_beats"] = beats
        shot["storyboard_beat_count"] = len(beats)
        # The top-level shot prompt is a narrative summary. Paid video prompts
        # are narrowed to one beat later by the continuity provider.
        shot["generation_actions"] = [beat["action"] for beat in beats]
        shot["generation_load"] = {
            **(shot.get("generation_load") or {}),
            "storyboard_beats": len(beats),
            "execution": "fresh_then_extend",
            "capability_profile": profile.name,
        }
        total += len(beats)
    storyboard["storyboard_beat_count"] = total
    storyboard["storyboard_execution"] = "continuity_aware_fresh_then_extend"
    return storyboard
