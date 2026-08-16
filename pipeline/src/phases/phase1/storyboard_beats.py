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
SECONDARY_STORYBOARD_VERSION = "honcut.secondary-storyboard.v3"
SECONDARY_EXECUTION = "content_capacity_boundary_aware_v3"
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


def _content_beat_requirement(
    shot: dict[str, Any],
    duration: float,
    actions: list[str],
    capabilities: VideoModelCapabilities,
) -> tuple[int, list[str]]:
    """Return one or two story-bearing clips required by provider capacity.

    Camera movement, genre and visual intensity never create an extension by
    themselves.  P02 exists only when P01 cannot carry the complete authored
    duration/action contract within one provider narrative window.
    """
    raw_units = shot.get("source_action_unit_ids") or []
    if isinstance(raw_units, str):
        raw_units = [raw_units]
    action_units = len({str(value) for value in raw_units if str(value).strip()})
    action_count = math.ceil(len(actions) / capabilities.max_micro_actions_per_beat)
    duration_count = max(1, math.ceil(duration / capabilities.max_unique_beat_s))
    required = max(1, action_units, action_count, duration_count)
    reasons: list[str] = []
    if duration_count > 1:
        reasons.append("p01_max_narrative_duration_exceeded")
    if action_count > 1:
        reasons.append("p01_micro_action_capacity_exceeded")
    if action_units > 1:
        reasons.append("p01_action_unit_capacity_exceeded")
    if required > 2:
        raise ValueError(
            f"{_shot_id(shot, 1)} cannot fit {len(actions)} micro-actions into "
            f"one base clip plus one extension for {capabilities.name}: "
            f"requires {required} story-bearing clips"
        )
    return required, reasons


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


def required_content_beat_count(
    shot: dict[str, Any],
    capabilities: VideoModelCapabilities | None = None,
) -> int:
    """Public deterministic capacity check shared by planning and Phase 5 QA."""
    profile = capabilities or capabilities_for(shot)
    duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
    required, _reasons = _content_beat_requirement(
        shot,
        duration,
        _source_actions(shot),
        profile,
    )
    return required


def _bridge_requirement(
    shots: list[dict[str, Any]],
    index: int,
) -> tuple[bool, str]:
    """Create a bridge only across a proven continuous primary-shot boundary."""
    if index + 1 >= len(shots):
        return False, "final primary shot has no following boundary"
    from quality.shot_continuity import classify_boundary

    boundary, reason = classify_boundary(shots[index], shots[index + 1], index=index + 2)
    return boundary == "continuous", reason


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
    """Attach plot-faithful content clips plus boundary-driven bridge clips."""
    total = 0
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    planned: list[
        tuple[
            dict[str, Any],
            str,
            list[str],
            int,
            bool,
            str,
            VideoModelCapabilities,
        ]
    ] = []
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        profile = capabilities or capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index + 1)
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
        source_actions = _source_actions(shot)
        content_count, extension_reasons = _content_beat_requirement(
            shot,
            duration,
            source_actions,
            profile,
        )
        bridge_required, bridge_reason = _bridge_requirement(shots, index)
        total_count = content_count + int(bridge_required)
        provider_capacity = max(1, int(duration // profile.min_unique_beat_s))
        if total_count > MAX_SECONDARY_BEATS:
            raise ValueError(
                f"{sid} requires {content_count} story-bearing clips plus a continuity "
                f"bridge, above the {MAX_SECONDARY_BEATS}-beat contract"
            )
        if total_count > provider_capacity:
            raise ValueError(
                f"{sid} needs {total_count} secondary beats for content and boundary "
                f"continuity, but {duration:g}s supports only {provider_capacity} at "
                f"{profile.min_unique_beat_s:g}s minimum per beat"
            )
        shot["secondary_storyboard_planning"] = {
            "content_beat_count": content_count,
            "extension_required": content_count > 1,
            "extension_reasons": extension_reasons,
            "bridge_required": bridge_required,
            "bridge_reason": bridge_reason,
            "provider_capacity": provider_capacity,
            "selected_count": total_count,
        }
        planned.append(
            (
                shot,
                sid,
                source_actions,
                content_count,
                bridge_required,
                bridge_reason,
                profile,
            )
        )

    for index, (
        shot,
        sid,
        source_actions,
        content_count,
        bridge_required,
        bridge_reason,
        profile,
    ) in enumerate(planned):
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
        fallback_action = _compact(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or "保持当前场景中的自然表演"
        )
        total_count = content_count + int(bridge_required)
        action_buckets = _partition(source_actions, content_count)
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        action_units = list(dict.fromkeys(
            str(value) for value in raw_units if str(value).strip()
        ))
        unit_buckets = _partition(action_units, content_count)
        durations = _duration_budgets(duration, total_count)
        start_state = _compact(
            shot.get("start_state")
            or shot.get("prev_shot_context")
            or shot.get("what")
        )
        final_state = _compact(shot.get("end_state") or shot.get("what"))
        beats: list[dict[str, Any]] = []
        for position in range(1, content_count + 1):
            action = _action_for_bucket(
                action_buckets[position - 1],
                position=position,
                count=content_count,
                fallback_action=fallback_action,
                final_state=final_state,
            )
            previous_state = beats[-1]["end_state"] if beats else start_state
            next_state = (
                final_state
                if position == content_count
                else _compact(f"已完成本格动作：{action}")
            )
            generation_mode = (
                "multi_image"
                if position == 1
                else "tail_video_extend"
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
        if bridge_required:
            next_shot, next_sid, *_ = planned[index + 1]
            next_start_state = _compact(
                next_shot.get("start_state")
                or next_shot.get("prev_shot_context")
                or next_shot.get("what")
            )
            position = len(beats) + 1
            bridge = {
                "beat_id": f"{sid}_P{position:02d}",
                "position": position,
                "duration_s": durations[position - 1],
                "generation_mode": "first_last_frame_bridge",
                "execution_strategy": "first_last_frame_bridge",
                "planner_version": SECONDARY_STORYBOARD_VERSION,
                "parent_shot_id": sid,
                "plot_fidelity_contract": "primary_shot_source_only_no_invention",
                "start_state": final_state,
                "action": _compact(
                    f"保持{sid}结束动作的因果连续，从当前终态平滑过渡到"
                    f"{next_sid}_P01 起始构图；不得执行{next_sid}的新动作"
                ),
                "micro_actions": [],
                "source_action_unit_ids": [],
                "end_state": next_start_state,
                "shot_size": shot.get("shot_size") or shot.get("shot_type"),
                "camera_movement": shot.get("camera_movement")
                or shot.get("camera_movement_en"),
                "bridge_target_shot_id": next_sid,
                "bridge_target_beat_id": f"{next_sid}_P01",
                "bridge_target_storyboard_image": f"storyboard_beats/{next_sid}_P01.png",
                "bridge_target_start_state": next_start_state,
                "bridge_boundary_reason": bridge_reason,
                "bridge_contract": (
                    "end on the next primary shot's P01 composition without executing "
                    "the next primary shot's action"
                ),
            }
            beats.append(bridge)
        shot["storyboard_beats"] = beats
        shot["storyboard_beat_count"] = len(beats)
        # The top-level shot prompt is a narrative summary. Paid video prompts
        # are narrowed to one beat later by the continuity provider.
        shot["generation_actions"] = [
            beat["action"]
            for beat in beats
            if beat["generation_mode"] != "first_last_frame_bridge"
        ]
        shot["generation_load"] = {
            **(shot.get("generation_load") or {}),
            "storyboard_beats": len(beats),
            "content_beats": content_count,
            "bridge_beats": int(bridge_required),
            "execution": SECONDARY_EXECUTION,
            "capability_profile": profile.name,
        }
        total += len(beats)
    storyboard["storyboard_beat_count"] = total
    storyboard["storyboard_execution"] = SECONDARY_EXECUTION
    storyboard["secondary_storyboard_version"] = SECONDARY_STORYBOARD_VERSION
    return storyboard
