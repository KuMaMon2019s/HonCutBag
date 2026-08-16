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
SECONDARY_STORYBOARD_VERSION = "honcut.secondary-storyboard.v2"
SECONDARY_EXECUTION = "complexity_aware_three_stage_v2"
MAX_SECONDARY_BEATS = 3


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
    *,
    has_next_shot: bool,
) -> int:
    """Choose one to three Pxx beats from narrative and staging complexity."""
    explicit = int(shot.get("secondary_storyboard_beat_count") or 0)
    raw_units = shot.get("source_action_unit_ids") or []
    if isinstance(raw_units, str):
        raw_units = [raw_units]
    action_units = len({str(value) for value in raw_units if str(value).strip()})
    action_count = math.ceil(len(actions) / capabilities.max_micro_actions_per_beat)
    semantic_minimum = max(
        1,
        explicit,
        action_units,
        action_count,
    )
    duration_count = max(1, math.ceil(duration / capabilities.max_unique_beat_s))
    provider_capacity = max(1, int(duration // capabilities.min_unique_beat_s))
    if semantic_minimum > provider_capacity:
        raise ValueError(
            f"{_shot_id(shot, 1)} cannot fit {len(actions)} micro-actions into "
            f"{duration:g}s for {capabilities.name}: requires {semantic_minimum} beats, "
            f"but the duration supports at most {provider_capacity}"
        )
    capacity = min(MAX_SECONDARY_BEATS, provider_capacity)
    # The third strategy needs the next primary shot's P01 as its last frame.
    if not has_next_shot:
        capacity = min(capacity, 2)
    if semantic_minimum > capacity:
        raise ValueError(
            f"{_shot_id(shot, 1)} requires {semantic_minimum} secondary beats, "
            f"but the three-stage contract supports at most {capacity} here"
        )

    desired = max(semantic_minimum, duration_count)
    if capabilities.name == SEEDANCE_2_CAPABILITIES.name:
        who = shot.get("who") or shot.get("characters") or []
        if isinstance(who, str):
            who = [who]
        camera = str(
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or ""
        ).strip().lower()
        intent = str(shot.get("shot_intent") or "").strip().lower()
        narrative = " ".join(
            str(shot.get(field) or "")
            for field in ("action_description", "what", "visual")
        ).lower()
        complexity_score = 0
        complexity_reasons: list[str] = []
        if len(actions) >= 2:
            complexity_score += 2
            complexity_reasons.append("multiple_ordered_actions")
        if len(actions) >= 3:
            complexity_score += 1
            complexity_reasons.append("three_or_more_actions")
        if len([value for value in who if str(value).strip()]) >= 2:
            complexity_score += 1
            complexity_reasons.append("multi_character_interaction")
        if intent in {"action", "transition", "chase", "fight"}:
            complexity_score += 1
            complexity_reasons.append(f"intent:{intent}")
        if camera not in {"", "static", "fixed", "locked", "unspecified"}:
            complexity_score += 1
            complexity_reasons.append("moving_camera")
        if re.search(
            r"失重|旋转|翻滚|穿过|解除|漂浮|zero.gravity|rotat|roll|disarm|debris",
            narrative,
        ):
            complexity_score += 1
            complexity_reasons.append("complex_spatial_choreography")
        if complexity_score >= 2:
            desired = max(desired, 2)
        if complexity_score >= 4 and has_next_shot:
            desired = max(desired, 3)
        shot["secondary_storyboard_complexity"] = {
            "score": complexity_score,
            "reasons": complexity_reasons,
            "provider_capacity": provider_capacity,
            "selected_count": min(capacity, desired),
        }
    return min(capacity, desired)


def _source_actions(shot: dict[str, Any]) -> list[str]:
    """Recover ordered screenplay actions from the primary shot, not old Pxx output."""
    raw = shot.get("micro_actions") or []
    if isinstance(raw, str):
        raw = [raw]
    values = [str(value).strip() for value in raw if str(value).strip()]
    authored_micro_actions = bool(values)
    if not values:
        narrative = str(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or ""
        )
        values = [
            value.strip()
            for value in re.split(r"(?:\s*(?:→|->)\s*)|[。！？!?；;\n]+", narrative)
            if value.strip()
        ]
    result: list[str] = []
    for value in values:
        raw_value = str(value).strip()
        if raw_value.startswith(("“", "”", '"', "『", "「", "』", "」")):
            continue
        candidate = _compact(value).strip("“”\"'，,：:、 ")
        if not candidate:
            continue
        # Style, runtime and one-take directives constrain the whole film; they
        # are not visible actions and must never consume a secondary beat.
        if re.fullmatch(
            r"(?:科幻)?(?:动作片)?风格(?:，?\s*\d+秒)?(?:，?\s*一镜到底)?",
            candidate,
        ) or re.fullmatch(r"\d+秒(?:，?\s*一镜到底)?", candidate):
            continue
        if (
            (authored_micro_actions or len(candidate) >= 2)
            and re.search(r"[\w\u3400-\u9fff]", candidate)
        ):
            result.append(candidate)
    if result:
        return result
    fallback = _compact(
        shot.get("action_description")
        or shot.get("what")
        or shot.get("visual")
        or "保持当前场景中的自然表演"
    )
    return [fallback]


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
    """Attach a plot-faithful, complexity-aware secondary storyboard ladder."""
    total = 0
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    planned: list[tuple[dict[str, Any], str, list[str], int, VideoModelCapabilities]] = []
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            continue
        profile = capabilities or capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index)
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
        source_actions = _source_actions(shot)
        fallback_action = _compact(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or "保持当前场景中的自然表演"
        )
        count = _beat_count(
            shot,
            duration,
            source_actions,
            profile,
            has_next_shot=index < len(shots),
        )
        planned.append((shot, sid, source_actions, count, profile))

    for index, (shot, sid, source_actions, count, profile) in enumerate(planned, 1):
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
        fallback_action = _compact(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or "保持当前场景中的自然表演"
        )
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
        beats: list[dict[str, Any]] = []
        for position in range(1, count + 1):
            action = _action_for_bucket(
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
            generation_mode = (
                "multi_image"
                if position == 1
                else "tail_video_extend"
                if position == 2
                else "first_last_frame_bridge"
            )
            normalized = {
                "beat_id": f"{sid}_P{position:02d}",
                "position": position,
                "duration_s": durations[position - 1],
                "generation_mode": generation_mode,
                "execution_strategy": generation_mode,
                "planner_version": SECONDARY_STORYBOARD_VERSION,
                "parent_shot_id": sid,
                "plot_fidelity_contract": "primary_shot_source_only_no_invention",
                "start_state": previous_state,
                "action": action,
                "micro_actions": action_buckets[position - 1],
                "source_action_unit_ids": unit_buckets[position - 1],
                "end_state": next_state,
                "shot_size": shot.get("shot_size") or shot.get("shot_type"),
                "camera_movement": shot.get("camera_movement")
                or shot.get("camera_movement_en"),
            }
            beats.append(normalized)
        if len(beats) == 3:
            next_shot, next_sid, *_ = planned[index]
            bridge = beats[-1]
            bridge.update({
                "bridge_target_shot_id": next_sid,
                "bridge_target_beat_id": f"{next_sid}_P01",
                "bridge_target_storyboard_image": f"storyboard_beats/{next_sid}_P01.png",
                "bridge_target_start_state": _compact(
                    next_shot.get("start_state")
                    or next_shot.get("prev_shot_context")
                    or next_shot.get("what")
                ),
                "bridge_contract": (
                    "end on the next primary shot's P01 composition without executing "
                    "the next primary shot's action"
                ),
            })
        shot["storyboard_beats"] = beats
        shot["storyboard_beat_count"] = len(beats)
        # The top-level shot prompt is a narrative summary. Paid video prompts
        # are narrowed to one beat later by the continuity provider.
        shot["generation_actions"] = [beat["action"] for beat in beats]
        shot["generation_load"] = {
            **(shot.get("generation_load") or {}),
            "storyboard_beats": len(beats),
            "execution": SECONDARY_EXECUTION,
            "capability_profile": profile.name,
        }
        total += len(beats)
    storyboard["storyboard_beat_count"] = total
    storyboard["storyboard_execution"] = SECONDARY_EXECUTION
    storyboard["secondary_storyboard_version"] = SECONDARY_STORYBOARD_VERSION
    return storyboard
