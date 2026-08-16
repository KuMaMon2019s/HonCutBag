"""Plan the per-shot visual beats that Phase 2 draws and Phase 6 executes."""

from __future__ import annotations

import math
import re
from typing import Any

from utils.video_capabilities import (
    MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
    VideoModelCapabilities,
    capabilities_for,
)

SECONDARY_STORYBOARD_VERSION = "honcut.secondary-storyboard.v5"
SECONDARY_EXECUTION = "content_capacity_boundary_aware_v5"
SECONDARY_GENERATION_MODES = frozenset({
    "multi_image",
    "tail_video_extend",
    "first_last_frame_bridge",
})
MAX_CONTENT_BEATS = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
MAX_SECONDARY_BEATS = 3
SPOKEN_CHARACTERS_PER_SECOND = 4.0


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _compact(value: Any, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _clean(value: Any) -> str:
    """Normalize authored narrative text without truncating screenplay detail."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def _quantized_units(value: float, capabilities: VideoModelCapabilities) -> int:
    quantum = capabilities.duration_quantum_s
    if quantum <= 0:
        raise ValueError(f"{capabilities.name} duration quantum must be positive")
    units = round(value / quantum)
    if not math.isclose(value, units * quantum, abs_tol=1e-6):
        raise ValueError(
            f"duration {value:g}s cannot be represented by {capabilities.name}'s "
            f"{quantum:g}s duration quantum"
        )
    return units


def _duration_budgets(
    total: float,
    count: int,
    capabilities: VideoModelCapabilities,
    *,
    minimum_durations: list[float] | None = None,
    maximum_durations: list[float] | None = None,
) -> list[float]:
    """Distribute duration without creating values the selected provider cannot execute."""
    if count < 1:
        raise ValueError("duration budget count must be positive")
    quantum = capabilities.duration_quantum_s
    total_units = _quantized_units(total, capabilities)
    minimum_values = minimum_durations or [capabilities.min_unique_beat_s] * count
    maximum_values = maximum_durations or [capabilities.max_unique_beat_s] * count
    if len(minimum_values) != count or len(maximum_values) != count:
        raise ValueError("duration bound count must match duration budget count")
    minimum_units = [math.ceil(value / quantum - 1e-9) for value in minimum_values]
    maximum_units = [math.floor(value / quantum + 1e-9) for value in maximum_values]
    if any(
        minimum > maximum
        for minimum, maximum in zip(minimum_units, maximum_units, strict=True)
    ):
        raise ValueError("duration minimum cannot exceed duration maximum")
    if total_units < sum(minimum_units):
        raise ValueError(
            f"{total:g}s cannot fund {count} {capabilities.name} beats at "
            f"the required provider minima {minimum_values}"
        )
    if total_units > sum(maximum_units):
        raise ValueError(
            f"{total:g}s exceeds {count} {capabilities.name} beats at "
            f"the effective-story maxima {maximum_values}"
        )
    values = list(minimum_units)
    remaining = total_units - sum(values)
    position = 0
    while remaining:
        if values[position] < maximum_units[position]:
            values[position] += 1
            remaining -= 1
        position = (position + 1) % count
    return [round(value * quantum, 6) for value in values]


def _spoken_duration(shot: dict[str, Any]) -> float:
    declared = shot.get("speech_duration_s")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return max(0.0, float(declared))

    def lines(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            preferred = [
                value.get(key)
                for key in ("line", "text", "content")
                if value.get(key)
            ]
            return [str(item) for item in preferred]
        if isinstance(value, (list, tuple)):
            return [item for nested in value for item in lines(nested)]
        return []

    spoken = "".join(lines(shot.get("dialogue")) + lines(shot.get("lines")))
    visible_characters = len(re.sub(r"\s+", "", spoken))
    return visible_characters / SPOKEN_CHARACTERS_PER_SECOND


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
    source_slices = shot.get("source_event_slices") or []
    if isinstance(source_slices, dict):
        source_slices = [source_slices]
    narrative_units = max(
        action_units,
        len([value for value in source_slices if isinstance(value, dict)]),
    )
    unit_count = math.ceil(
        narrative_units / capabilities.max_action_units_per_beat
    ) if narrative_units else 1
    action_count = math.ceil(len(actions) / capabilities.max_micro_actions_per_beat)
    duration_count = max(1, math.ceil(duration / capabilities.max_unique_beat_s))
    spoken_duration = _spoken_duration(shot)
    if spoken_duration > duration + 1e-6:
        raise ValueError(
            f"{_shot_id(shot, 1)} has {spoken_duration:g}s of spoken content but only "
            f"{duration:g}s of story-bearing time after transition reservation"
        )
    dialogue_count = max(
        1,
        math.ceil(spoken_duration / capabilities.max_unique_beat_s),
    )
    required = max(1, unit_count, action_count, duration_count, dialogue_count)
    reasons: list[str] = []
    if duration_count > 1:
        reasons.append("p01_max_narrative_duration_exceeded")
    if action_count > 1:
        reasons.append("p01_micro_action_capacity_exceeded")
    if unit_count > 1:
        reasons.append("p01_action_unit_capacity_exceeded")
    if dialogue_count > 1:
        reasons.append("p01_spoken_content_capacity_exceeded")
    if required > MAX_CONTENT_BEATS:
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
            for value in re.split(
                r"(?:\s*(?:→|->)\s*)|[。！？!?；;，,\n]+|"
                r"(?:随后|然后|接着|继而)",
                narrative,
            )
            if value.strip()
        ]
    result: list[str] = []
    for value in values:
        raw_value = str(value).strip()
        if raw_value.startswith(("“", "”", '"', "『", "「", "』", "」")):
            continue
        candidate = _clean(value).strip("“”\"'，,：:、 ")
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
    fallback = _clean(
        shot.get("action_description")
        or shot.get("what")
        or shot.get("visual")
        or "保持当前场景中的自然表演"
    )
    return [fallback]


def _start_state(shot: dict[str, Any]) -> str:
    actions = _source_actions(shot)
    return _compact(
        shot.get("start_state")
        or shot.get("prev_shot_context")
        or shot.get("what")
        or actions[0]
    )


def _end_state(shot: dict[str, Any]) -> str:
    actions = _source_actions(shot)
    return _compact(
        shot.get("end_state")
        or shot.get("what")
        or actions[-1]
    )


def required_content_beat_count(
    shot: dict[str, Any],
    capabilities: VideoModelCapabilities | None = None,
    *,
    available_duration_s: float | None = None,
) -> int:
    """Public deterministic capacity check shared by planning and Phase 5 QA."""
    profile = capabilities or capabilities_for(shot)
    duration = (
        float(available_duration_s)
        if available_duration_s is not None
        else float(shot.get("duration") or shot.get("suggested_duration") or 5)
    )
    required, _reasons = _content_beat_requirement(
        shot,
        duration,
        _source_actions(shot),
        profile,
    )
    return required


def _bridge_requirement(
    storyboard: dict[str, Any],
    shots: list[dict[str, Any]],
    index: int,
) -> tuple[bool, str]:
    """Create a bridge only across a proven continuous primary-shot boundary."""
    if index + 1 >= len(shots):
        return False, "final primary shot has no following boundary"
    continuity_mode = str(storyboard.get("continuity_mode") or "").strip().lower()
    if continuity_mode in {"one_take", "single_take", "oner"}:
        return (
            True,
            "one-take contract requires a generated moving bridge into the next "
            "primary shot P01 composition",
        )
    from quality.shot_continuity import classify_boundary

    boundary, reason = classify_boundary(shots[index], shots[index + 1], index=index + 2)
    return boundary == "continuous", reason


def secondary_storyboard_requirements(
    storyboard: dict[str, Any],
    index: int,
    capabilities: VideoModelCapabilities | None = None,
) -> dict[str, Any]:
    """Compute the complete, provider-executable Pxx contract without mutating input."""
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    if index < 0 or index >= len(shots):
        raise IndexError(f"secondary storyboard shot index out of range: {index}")
    shot = shots[index]
    profile = capabilities or capabilities_for({**storyboard, **shot})
    sid = _shot_id(shot, index + 1)
    duration = float(shot.get("duration") or shot.get("suggested_duration") or 5)
    _quantized_units(duration, profile)
    bridge_required, bridge_reason = _bridge_requirement(storyboard, shots, index)
    bridge_request_minimum, _bridge_maximum = profile.request_duration_bounds(
        "first_last_frame_bridge"
    )
    bridge_duration = profile.min_unique_beat_s if bridge_required else 0.0
    if bridge_required:
        profile.validate_chunk_durations(
            bridge_request_minimum,
            bridge_duration,
            "first_last_frame_bridge",
            resource_id=f"{sid}_bridge",
        )
    content_duration = round(duration - bridge_duration, 6)
    if content_duration <= 0:
        raise ValueError(
            f"{sid} has no story-bearing duration after reserving its continuity bridge"
        )
    source_actions = _source_actions(shot)
    content_count, extension_reasons = _content_beat_requirement(
        shot,
        content_duration,
        source_actions,
        profile,
    )
    content_durations = _duration_budgets(
        content_duration,
        content_count,
        profile,
        minimum_durations=[profile.min_unique_beat_s] * content_count,
        maximum_durations=[profile.max_unique_beat_s] * content_count,
    )
    modes = ["multi_image"]
    if content_count > 1:
        modes.append("tail_video_extend")
    if bridge_required:
        modes.append("first_last_frame_bridge")
    if len(modes) > MAX_SECONDARY_BEATS:
        raise ValueError(
            f"{sid} requires {len(modes)} secondary beats, above the "
            f"{MAX_SECONDARY_BEATS}-beat contract"
        )
    return {
        "shot_id": sid,
        "profile": profile,
        "duration": duration,
        "source_actions": source_actions,
        "content_duration": content_duration,
        "content_count": content_count,
        "content_durations": content_durations,
        "extension_required": content_count > 1,
        "extension_reasons": extension_reasons,
        "bridge_required": bridge_required,
        "bridge_reason": bridge_reason,
        "bridge_duration": bridge_duration,
        "modes": modes,
        "durations": content_durations + ([bridge_duration] if bridge_required else []),
    }


def secondary_contract_declared(storyboard: dict[str, Any]) -> bool:
    """Return true when an artifact claims or uses the modern Pxx contract."""
    if "secondary_storyboard_version" in storyboard:
        return True
    if str(storyboard.get("storyboard_execution") or "").startswith(
        "content_capacity_boundary_aware_"
    ):
        return True
    return any(
        str(beat.get("generation_mode") or "").strip().lower()
        in SECONDARY_GENERATION_MODES
        for shot in storyboard.get("shots", [])
        if isinstance(shot, dict)
        for beat in (shot.get("storyboard_beats") or [])
        if isinstance(beat, dict)
    )


def secondary_storyboard_contract_errors(
    storyboard: dict[str, Any],
    index: int,
    capabilities: VideoModelCapabilities | None = None,
) -> list[dict[str, Any]]:
    """Return strict v5 contract violations shared by Phase 4 and Phase 5."""
    if not secondary_contract_declared(storyboard):
        return []
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    if index < 0 or index >= len(shots):
        return [{"code": "secondary_storyboard_index_invalid", "message": str(index)}]
    shot = shots[index]
    sid = _shot_id(shot, index + 1)
    errors: list[dict[str, Any]] = []

    def add(code: str, message: str, **details: Any) -> None:
        errors.append({"code": code, "message": message, "details": details})

    version = str(storyboard.get("secondary_storyboard_version") or "")
    if version != SECONDARY_STORYBOARD_VERSION:
        add(
            "secondary_storyboard_version_invalid",
            f"{sid} requires {SECONDARY_STORYBOARD_VERSION}, observed {version or '<missing>'}",
            expected=SECONDARY_STORYBOARD_VERSION,
            observed=version or None,
        )
    execution = str(storyboard.get("storyboard_execution") or "")
    if execution != SECONDARY_EXECUTION:
        add(
            "secondary_storyboard_execution_invalid",
            f"{sid} requires execution contract {SECONDARY_EXECUTION}",
            expected=SECONDARY_EXECUTION,
            observed=execution or None,
        )

    beats = [
        beat for beat in (shot.get("storyboard_beats") or []) if isinstance(beat, dict)
    ]
    if not beats:
        add("secondary_storyboard_beats_missing", f"{sid} has no secondary beats")
        return errors

    requirement: dict[str, Any] | None = None
    try:
        requirement = secondary_storyboard_requirements(storyboard, index, capabilities)
    except (TypeError, ValueError) as exc:
        add(
            "secondary_storyboard_capacity_impossible",
            f"{sid} cannot produce an executable secondary plan: {exc}",
        )

    actual_modes = [
        str(beat.get("generation_mode") or "").strip().lower() for beat in beats
    ]
    expected_modes = requirement["modes"] if requirement else []
    if requirement and actual_modes != expected_modes:
        add(
            "secondary_storyboard_strategy_mismatch",
            f"{sid} secondary strategies do not match content capacity and boundary",
            expected=expected_modes,
            observed=actual_modes,
        )
    for mode in actual_modes:
        if mode not in SECONDARY_GENERATION_MODES:
            add(
                "secondary_storyboard_mode_invalid",
                f"{sid} contains unsupported generation mode {mode or '<missing>'}",
                observed=mode or None,
            )

    expected_durations = requirement["durations"] if requirement else []
    source_action_units = shot.get("source_action_unit_ids") or []
    if isinstance(source_action_units, str):
        source_action_units = [source_action_units]
    source_action_units = list(dict.fromkeys(
        str(value) for value in source_action_units if str(value).strip()
    ))
    observed_actions: list[str] = []
    observed_units: list[str] = []
    content_actions: list[str] = []
    bridge = None
    for position, beat in enumerate(beats, 1):
        beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
        mode = actual_modes[position - 1]
        if beat.get("position") != position:
            add(
                "secondary_storyboard_position_invalid",
                f"{beat_id} position must be {position}",
                expected=position,
                observed=beat.get("position"),
            )
        if str(beat.get("planner_version") or "") != SECONDARY_STORYBOARD_VERSION:
            add(
                "secondary_storyboard_planner_version_invalid",
                f"{beat_id} is not bound to {SECONDARY_STORYBOARD_VERSION}",
                observed=beat.get("planner_version"),
            )
        if str(beat.get("execution_strategy") or "").strip().lower() != mode:
            add(
                "secondary_storyboard_execution_strategy_invalid",
                f"{beat_id} execution strategy must equal generation mode",
                observed=beat.get("execution_strategy"),
                expected=mode,
            )
        if beat.get("duration_semantics") != (
            "effective_story_time_excluding_reference_overlap_and_provider_padding"
        ):
            add(
                "secondary_storyboard_duration_semantics_missing",
                f"{beat_id} must declare effective story-time duration semantics",
            )
        if str(beat.get("parent_shot_id") or "") != sid:
            add(
                "secondary_storyboard_parent_mismatch",
                f"{beat_id} must remain owned by {sid}",
            )
        if beat.get("plot_fidelity_contract") != "primary_shot_source_only_no_invention":
            add(
                "secondary_storyboard_fidelity_missing",
                f"{beat_id} lacks the primary-shot fidelity contract",
            )
        if not str(beat.get("action") or "").strip():
            add("storyboard_beat_action_missing", f"{beat_id} has no executable action")
        if position <= len(expected_durations):
            observed_duration = float(beat.get("duration_s") or 0)
            if not math.isclose(
                observed_duration,
                float(expected_durations[position - 1]),
                abs_tol=1e-6,
            ):
                add(
                    "secondary_storyboard_duration_invalid",
                    f"{beat_id} duration does not match the provider-executable plan",
                    expected=expected_durations[position - 1],
                    observed=observed_duration,
                )
        beat_actions = beat.get("micro_actions") or []
        if isinstance(beat_actions, str):
            beat_actions = [beat_actions]
        beat_actions = [str(value) for value in beat_actions if str(value).strip()]
        beat_units = beat.get("source_action_unit_ids") or []
        if isinstance(beat_units, str):
            beat_units = [beat_units]
        beat_units = [str(value) for value in beat_units if str(value).strip()]
        if mode == "first_last_frame_bridge":
            bridge = beat
            if beat_actions or beat_units:
                add(
                    "secondary_storyboard_bridge_contains_plot",
                    f"{beat_id} bridge must not carry screenplay actions or action units",
                    micro_actions=beat_actions,
                    source_action_unit_ids=beat_units,
                )
        else:
            observed_actions.extend(beat_actions)
            observed_units.extend(beat_units)
            content_actions.append(str(beat.get("action") or ""))

    if requirement:
        if observed_actions != requirement["source_actions"]:
            add(
                "secondary_storyboard_action_order_mismatch",
                f"{sid} content beats must preserve every primary action in order",
                expected=requirement["source_actions"],
                observed=observed_actions,
            )
        if observed_units != source_action_units:
            add(
                "secondary_storyboard_action_unit_coverage_mismatch",
                f"{sid} content beats must preserve every action unit in order",
                expected=source_action_units,
                observed=observed_units,
            )
        if list(shot.get("generation_actions") or []) != content_actions:
            add(
                "secondary_storyboard_generation_actions_mismatch",
                f"{sid} generation_actions must equal its story-bearing beat actions",
                expected=content_actions,
                observed=shot.get("generation_actions") or [],
            )
        if shot.get("storyboard_beat_count") != len(beats):
            add(
                "secondary_storyboard_count_metadata_invalid",
                f"{sid} storyboard_beat_count must equal the authored Pxx count",
                expected=len(beats),
                observed=shot.get("storyboard_beat_count"),
            )
        planning = shot.get("secondary_storyboard_planning") or {}
        expected_planning = {
            "content_beat_count": requirement["content_count"],
            "extension_required": requirement["extension_required"],
            "bridge_required": requirement["bridge_required"],
            "selected_count": len(requirement["modes"]),
        }
        for key, expected in expected_planning.items():
            if planning.get(key) != expected:
                add(
                    "secondary_storyboard_planning_metadata_invalid",
                    f"{sid} planning metadata {key} is stale or incorrect",
                    field=key,
                    expected=expected,
                    observed=planning.get(key),
                )
        if requirement["bridge_required"]:
            next_shot = shots[index + 1]
            next_sid = _shot_id(next_shot, index + 2)
            expected_next_start = _start_state(next_shot)
            expected_current_end = _end_state(shot)
            if bridge is None:
                add("secondary_storyboard_bridge_missing", f"{sid} requires a bridge")
            else:
                expected_target = f"{next_sid}_P01"
                if (
                    bridge.get("bridge_target_shot_id") != next_sid
                    or bridge.get("bridge_target_beat_id") != expected_target
                    or bridge.get("bridge_target_storyboard_image")
                    != f"storyboard_beats/{expected_target}.png"
                    or str(bridge.get("start_state") or "") != expected_current_end
                    or str(bridge.get("end_state") or "") != expected_next_start
                ):
                    add(
                        "secondary_storyboard_bridge_invalid",
                        f"{sid} bridge must connect its exact end state to {expected_target}",
                    )
        elif bridge is not None:
            add(
                "secondary_storyboard_bridge_forbidden",
                f"{sid} must not bridge across a cut or transition boundary",
            )
    return errors


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
    continuity_mode = str(storyboard.get("continuity_mode") or "").strip().lower()
    if continuity_mode in {"one_take", "single_take", "oner"}:
        for index, shot in enumerate(shots):
            if index == 0:
                shot["boundary_before"] = "cut"
                shot["continuity_reason"] = "first shot opens the one-take camera path"
                continue
            authored_boundary = str(shot.get("boundary_before") or "").strip()
            if authored_boundary and authored_boundary != "continuous":
                shot.setdefault("authored_boundary_before", authored_boundary)
            previous = shots[index - 1]
            authored_transition = str(previous.get("transition_to_next") or "").strip()
            if authored_transition and authored_transition != "continuous":
                previous.setdefault("authored_transition_to_next", authored_transition)
            previous["transition_to_next"] = "continuous"
            shot["boundary_before"] = "continuous"
            shot["continuity_reason"] = (
                "one-take contract: preserve the moving camera and action state through "
                "a generated bridge from the preceding primary shot"
            )
    planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, shot in enumerate(shots):
        profile = capabilities or capabilities_for({**storyboard, **shot})
        requirement = secondary_storyboard_requirements(storyboard, index, profile)
        total_count = len(requirement["modes"])
        provider_capacity = max(
            1,
            int(requirement["duration"] // profile.min_unique_beat_s),
        )
        shot["secondary_storyboard_planning"] = {
            "content_beat_count": requirement["content_count"],
            "content_duration_s": requirement["content_duration"],
            "extension_required": requirement["extension_required"],
            "extension_reasons": requirement["extension_reasons"],
            "bridge_required": requirement["bridge_required"],
            "bridge_reason": requirement["bridge_reason"],
            "bridge_duration_s": requirement["bridge_duration"],
            "provider_capacity": provider_capacity,
            "duration_quantum_s": profile.duration_quantum_s,
            "min_effective_story_duration_s": profile.min_unique_beat_s,
            "min_provider_request_duration_s": profile.min_shot_duration_s,
            "min_first_last_frame_request_duration_s": (
                profile.request_duration_bounds("first_last_frame_bridge")[0]
            ),
            "selected_count": total_count,
        }
        planned.append((shot, requirement))

    for index, (shot, requirement) in enumerate(planned):
        sid = requirement["shot_id"]
        source_actions = requirement["source_actions"]
        content_count = requirement["content_count"]
        bridge_required = requirement["bridge_required"]
        bridge_reason = requirement["bridge_reason"]
        profile = requirement["profile"]
        fallback_action = _clean(
            shot.get("action_description")
            or shot.get("what")
            or shot.get("visual")
            or "保持当前场景中的自然表演"
        )
        action_buckets = _partition(source_actions, content_count)
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        action_units = list(dict.fromkeys(
            str(value) for value in raw_units if str(value).strip()
        ))
        unit_buckets = _partition(action_units, content_count)
        durations = requirement["durations"]
        start_state = _start_state(shot)
        final_state = _end_state(shot)
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
                "duration_semantics": (
                    "effective_story_time_excluding_reference_overlap_and_provider_padding"
                ),
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
            next_shot, next_requirement = planned[index + 1]
            next_sid = next_requirement["shot_id"]
            next_start_state = _start_state(next_shot)
            position = len(beats) + 1
            bridge = {
                "beat_id": f"{sid}_P{position:02d}",
                "position": position,
                "duration_s": durations[position - 1],
                "duration_semantics": (
                    "effective_story_time_excluding_reference_overlap_and_provider_padding"
                ),
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
    contract_errors = [
        error
        for index in range(len(shots))
        for error in secondary_storyboard_contract_errors(
            storyboard,
            index,
            capabilities,
        )
    ]
    if contract_errors:
        summary = "; ".join(error["message"] for error in contract_errors[:5])
        raise AssertionError(f"secondary storyboard planner emitted an invalid contract: {summary}")
    return storyboard
