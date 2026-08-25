"""Plan the per-shot visual beats that Phase 2 draws and Phase 6 executes."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from utils.action_units import normalize_action_units
from utils.body_action_contracts import apply_body_action_contract
from utils.material_budget import (
    BRIDGE_TIMELINE_POLICY,
    attach_material_budget,
    material_budget_contract_errors,
)
from utils.video_capabilities import (
    MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
    VideoModelCapabilities,
    capabilities_for,
    max_primary_story_duration,
    min_primary_story_duration,
)

SECONDARY_STORYBOARD_VERSION = "honcut.secondary-storyboard.v13"
SECONDARY_EXECUTION = "content_capacity_post_primary_bridge_v13"
SECONDARY_GENERATION_MODES = frozenset({
    "multi_image",
    "tail_video_extend",
    "first_last_frame_bridge",
})
MAX_CONTENT_BEATS = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
MAX_SECONDARY_BEATS = MAX_CONTENT_BEATS
SPOKEN_CHARACTERS_PER_SECOND = 4.0

# Editorial policy, deliberately separate from provider/model capabilities.
# The provider may accept longer FLF2V clips, but a transition longer than six
# seconds stops behaving like a bridge and should be authored as story content.
HONCUT_BRIDGE_MIN_DURATION_S = 4.0
HONCUT_BRIDGE_MAX_DURATION_S = 6.0


def bridge_planning_duration_bounds(
    capabilities: VideoModelCapabilities,
) -> tuple[float, float]:
    """Intersect HonCut's bridge policy with the selected provider's limits."""
    provider_minimum, provider_maximum = capabilities.effective_duration_bounds(
        "first_last_frame_bridge"
    )
    minimum = max(provider_minimum, HONCUT_BRIDGE_MIN_DURATION_S)
    maximum = min(provider_maximum, HONCUT_BRIDGE_MAX_DURATION_S)
    if minimum > maximum + 1e-6:
        raise ValueError(
            f"{capabilities.name} first/last-frame range "
            f"{provider_minimum:g}-{provider_maximum:g}s has no overlap with "
            f"HonCut's {HONCUT_BRIDGE_MIN_DURATION_S:g}-"
            f"{HONCUT_BRIDGE_MAX_DURATION_S:g}s bridge policy"
        )
    return float(minimum), float(maximum)


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


def _contains_complete_mention(text: str, mention: str) -> bool:
    """Match one authored identity mention without Latin substring collisions."""
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    normalized_mention = unicodedata.normalize("NFKC", mention).casefold().strip()
    if not normalized_mention:
        return False
    if re.search(r"[\u3400-\u9fff]", normalized_mention):
        return normalized_mention in normalized_text
    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_mention)}(?![a-z0-9])",
        normalized_text,
    ) is not None


def _beat_character_ids(
    shot: dict[str, Any],
    *semantic_values: Any,
    source_event_ids: list[int] | None = None,
) -> list[str]:
    """Project shot participants into the canonical cast visible in one Pxx.

    Shot-level cast is a superset across the whole Sxx.  A Pxx may precede a
    character's entrance, so its cast must be derived only from its own
    start/action/end facts and persisted for downstream media owners.
    """
    raw_ids = shot.get("character_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    character_ids = list(dict.fromkeys(
        str(value).strip() for value in raw_ids if str(value).strip()
    ))
    mentions_by_id = {character_id: {character_id} for character_id in character_ids}

    for reference in shot.get("participant_refs") or []:
        if not isinstance(reference, dict):
            continue
        character_id = str(reference.get("character_id") or "").strip()
        mention = str(reference.get("mention") or "").strip()
        if character_id in mentions_by_id and mention:
            mentions_by_id[character_id].add(mention)

    raw_who = shot.get("who") or []
    if isinstance(raw_who, str):
        raw_who = [raw_who]
    who = [str(value).strip() for value in raw_who if str(value).strip()]
    if len(who) == len(character_ids):
        for character_id, mention in zip(character_ids, who, strict=True):
            mentions_by_id[character_id].add(mention)

    semantic_text = "\n".join(
        str(value) for value in semantic_values if str(value or "").strip()
    )
    visible_ids = {
        character_id
        for character_id in character_ids
        if any(
            _contains_complete_mention(semantic_text, mention)
            for mention in mentions_by_id[character_id]
        )
    }
    event_casts = _source_event_cast_index(shot)
    for event_id in source_event_ids or []:
        visible_ids.update(event_casts.get(event_id, []))
    return [
        character_id for character_id in character_ids if character_id in visible_ids
    ]


def _source_event_cast_index(shot: dict[str, Any]) -> dict[int, list[str]]:
    """Validate and index canonical event-level cast lineage for one shot."""
    raw_casts = shot.get("source_event_casts")
    if raw_casts is None:
        return {}
    if not isinstance(raw_casts, list):
        raise ValueError("source_event_casts must be an array")
    shot_ids = list(dict.fromkeys(
        str(value).strip()
        for value in (shot.get("character_ids") or [])
        if str(value).strip()
    ))
    index: dict[int, list[str]] = {}
    for record in raw_casts:
        if not isinstance(record, dict):
            raise ValueError("source_event_casts entries must be objects")
        event_id = record.get("source_event_id")
        raw_ids = record.get("character_ids")
        if not isinstance(event_id, int) or event_id < 1:
            raise ValueError("source_event_casts requires positive source_event_id")
        if event_id in index:
            raise ValueError("source_event_casts contains duplicate source_event_id")
        if not isinstance(raw_ids, list):
            raise ValueError("source_event_casts character_ids must be an array")
        cast_ids = [
            str(value).strip() for value in raw_ids if str(value).strip()
        ]
        if cast_ids != list(dict.fromkeys(cast_ids)):
            raise ValueError("source_event_casts character_ids must be unique")
        if any(character_id not in shot_ids for character_id in cast_ids):
            raise ValueError("source_event_casts contains an ID outside the shot cast")
        index[event_id] = cast_ids
    return index


def _generation_unit_source_event_ids(
    generation_units: list[dict[str, Any]],
) -> list[int]:
    """Return ordered canonical source events represented by one Pxx bucket."""
    return list(dict.fromkeys(
        event_id
        for unit in generation_units
        if isinstance(unit, dict)
        for event_id in [unit.get("source_event_id")]
        if isinstance(event_id, int) and event_id > 0
    ))


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


def _generation_action_units(
    shot: dict[str, Any],
    actions: list[str],
) -> list[dict[str, Any]]:
    """Read persisted normalized units or derive them without mutating the ledger."""

    if "generation_action_units" in shot:
        return [
            dict(unit)
            for unit in (shot.get("generation_action_units") or [])
            if isinstance(unit, dict)
        ]
    return [
        dict(unit)
        for unit in normalize_action_units(
            actions,
            composite_motion=(
                str(shot.get("generation_motion_mode") or "").strip().lower()
                == "composite"
            ),
        )["generation_action_units"]
    ]


def _content_beat_requirement(
    shot: dict[str, Any],
    duration: float,
    actions: list[str],
    capabilities: VideoModelCapabilities,
) -> tuple[int, list[str]]:
    """Return the number of story-bearing clips required by provider capacity.

    Camera movement, genre and visual intensity never create an extension by
    themselves.  P02 exists only when P01 cannot carry the complete authored
    duration/action contract within one provider narrative window.
    """
    generation_action_units = _generation_action_units(shot, actions)
    action_count = (
        math.ceil(
            len(generation_action_units) / capabilities.max_micro_actions_per_beat
        )
        if generation_action_units else 1
    )
    first_minimum, first_maximum = capabilities.effective_duration_bounds("multi_image")
    tail_minimum, tail_maximum = capabilities.effective_duration_bounds(
        "tail_video_extend"
    )
    duration_count = MAX_CONTENT_BEATS + 1
    for candidate in range(1, MAX_CONTENT_BEATS + 1):
        minimum = first_minimum + (candidate - 1) * tail_minimum
        maximum = first_maximum + (candidate - 1) * tail_maximum
        if minimum - 1e-6 <= duration <= maximum + 1e-6:
            duration_count = candidate
            break
    spoken_duration = _spoken_duration(shot)
    if spoken_duration > duration + 1e-6:
        raise ValueError(
            f"{_shot_id(shot, 1)} has {spoken_duration:g}s of spoken content but only "
            f"{duration:g}s of story-bearing time after transition reservation"
        )
    dialogue_count = max(
        1,
        math.ceil(spoken_duration / max(first_maximum, tail_maximum)),
    )
    required = max(1, action_count, duration_count, dialogue_count)
    required_minimum = first_minimum + max(0, required - 1) * tail_minimum
    required_maximum = first_maximum + max(0, required - 1) * tail_maximum
    reasons: list[str] = []
    if duration_count > 1:
        reasons.append("p01_max_narrative_duration_exceeded")
    if action_count > 1:
        reasons.append("p01_generation_action_unit_capacity_exceeded")
    if dialogue_count > 1:
        reasons.append("p01_spoken_content_capacity_exceeded")
    if required > MAX_CONTENT_BEATS:
        raise ValueError(
            f"{_shot_id(shot, 1)} cannot fit {len(actions)} micro-actions into "
            f"one base clip plus bounded extensions for {capabilities.name}: "
            f"{len(generation_action_units)} normalized generation action units "
            f"require {required} story-bearing clips"
        )
    if not required_minimum - 1e-6 <= duration <= required_maximum + 1e-6:
        raise ValueError(
            f"{_shot_id(shot, 1)} needs {required} story-bearing clips for its "
            f"content complexity, but {duration:g}s is outside their executable "
            f"{required_minimum:g}-{required_maximum:g}s range"
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
    primary_minimum = min_primary_story_duration(profile)
    primary_maximum = max_primary_story_duration(profile)
    if not primary_minimum - 1e-6 <= duration <= primary_maximum + 1e-6:
        raise ValueError(
            f"{sid} primary duration {duration:g}s is outside the assembled "
            f"{primary_minimum:g}-{primary_maximum:g}s range"
        )
    bridge_required, bridge_reason = _bridge_requirement(storyboard, shots, index)
    bridge_minimum, _bridge_maximum = bridge_planning_duration_bounds(profile)
    bridge_duration = bridge_minimum if bridge_required else 0.0
    if bridge_required:
        profile.validate_chunk_durations(
            bridge_duration,
            bridge_duration,
            "first_last_frame_bridge",
            resource_id=f"{sid}_bridge",
        )
    # A cross-primary bridge is a separately paid post-generation asset.  It
    # does not expand either primary shot's authored duration; Phase 8 replaces
    # stable edge handles with its visible interval on the edit timeline.
    content_duration = round(duration, 6)
    source_actions = _source_actions(shot)
    generation_action_units = _generation_action_units(shot, source_actions)
    content_count, extension_reasons = _content_beat_requirement(
        shot,
        content_duration,
        source_actions,
        profile,
    )
    first_minimum, first_maximum = profile.effective_duration_bounds("multi_image")
    tail_minimum, tail_maximum = profile.effective_duration_bounds("tail_video_extend")
    content_durations = _duration_budgets(
        content_duration,
        content_count,
        profile,
        minimum_durations=[first_minimum] + [tail_minimum] * (content_count - 1),
        maximum_durations=[first_maximum] + [tail_maximum] * (content_count - 1),
    )
    modes = ["multi_image"] + ["tail_video_extend"] * (content_count - 1)
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
        "generation_action_units": generation_action_units,
        "content_duration": content_duration,
        "content_count": content_count,
        "content_durations": content_durations,
        "extension_required": content_count > 1,
        "extension_reasons": extension_reasons,
        "bridge_required": bridge_required,
        "bridge_reason": bridge_reason,
        "bridge_duration": bridge_duration,
        "modes": modes,
        "durations": content_durations,
    }


def secondary_contract_declared(storyboard: dict[str, Any]) -> bool:
    """Return true when an artifact claims or uses the modern Pxx contract."""
    if "secondary_storyboard_version" in storyboard:
        return True
    if str(storyboard.get("storyboard_execution") or "").startswith(
        ("content_capacity_boundary_aware_", "content_capacity_post_primary_bridge_")
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

    try:
        source_event_casts = _source_event_cast_index(shot)
    except ValueError as exc:
        source_event_casts = {}
        add(
            "secondary_storyboard_source_event_cast_invalid",
            f"{sid} has invalid source event cast lineage: {exc}",
        )
    raw_source_events = shot.get("source_events") or []
    source_events = [
        value for value in raw_source_events if isinstance(value, int) and value > 0
    ] if isinstance(raw_source_events, list) else []
    if storyboard.get("semantic_understanding") and (
        list(source_event_casts) != list(dict.fromkeys(source_events))
    ):
        add(
            "secondary_storyboard_source_event_cast_invalid",
            f"{sid} source event cast lineage must cover canonical source_events",
            expected=list(dict.fromkeys(source_events)),
            observed=list(source_event_casts),
        )

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
    profile = (
        requirement["profile"]
        if requirement is not None
        else capabilities or capabilities_for({**storyboard, **shot})
    )
    source_action_units = shot.get("source_action_unit_ids") or []
    if isinstance(source_action_units, str):
        source_action_units = [source_action_units]
    source_action_units = list(dict.fromkeys(
        str(value) for value in source_action_units if str(value).strip()
    ))
    observed_actions: list[str] = []
    observed_units: list[str] = []
    content_actions: list[str] = []
    bridge_specs = [
        item
        for item in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(item, dict) and str(item.get("source_shot_id") or "") == sid
    ]
    bridge = bridge_specs[0] if len(bridge_specs) == 1 else None
    if len(bridge_specs) > 1:
        add(
            "primary_shot_bridge_duplicate",
            f"{sid} declares more than one bridge for the same outgoing boundary",
        )
    for position, beat in enumerate(beats, 1):
        beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
        mode = actual_modes[position - 1]
        beat_generation_units = [
            unit
            for unit in (beat.get("generation_action_units") or [])
            if isinstance(unit, dict)
        ]
        expected_source_event_ids = _generation_unit_source_event_ids(
            beat_generation_units
        )
        raw_beat_source_event_ids = beat.get("source_event_ids")
        if raw_beat_source_event_ids != expected_source_event_ids:
            add(
                "secondary_storyboard_beat_event_lineage_invalid",
                f"{beat_id} source_event_ids do not match its action units",
                expected=expected_source_event_ids,
                observed=raw_beat_source_event_ids,
            )
        raw_beat_character_ids = beat.get("character_ids")
        if not isinstance(raw_beat_character_ids, list):
            add(
                "secondary_storyboard_beat_cast_invalid",
                f"{beat_id} must declare canonical character_ids as an array",
                observed=raw_beat_character_ids,
            )
        else:
            beat_character_ids = [
                str(value).strip()
                for value in raw_beat_character_ids
                if str(value).strip()
            ]
            expected_beat_character_ids = _beat_character_ids(
                shot,
                beat.get("start_state"),
                beat.get("action"),
                beat.get("end_state"),
                source_event_ids=expected_source_event_ids,
            )
            if (
                beat_character_ids != list(dict.fromkeys(beat_character_ids))
                or beat_character_ids != expected_beat_character_ids
            ):
                add(
                    "secondary_storyboard_beat_cast_invalid",
                    f"{beat_id} cast does not match its own visible facts",
                    expected=expected_beat_character_ids,
                    observed=beat_character_ids,
                )
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
            expected_request_mode = (
                expected_modes[position - 1]
                if position <= len(expected_modes)
                else mode
            )
            expected_request_duration = profile.request_duration_for_effective_story(
                observed_duration,
                expected_request_mode,
            )
            observed_effective = float(
                beat.get("effective_story_duration_s") or 0
            )
            observed_request = float(
                beat.get("provider_request_duration_s") or 0
            )
            observed_padding = float(
                beat.get("provider_minimum_padding_duration_s") or 0
            )
            if not math.isclose(observed_effective, observed_duration, abs_tol=1e-6):
                add(
                    "secondary_storyboard_effective_duration_invalid",
                    f"{beat_id} effective story duration must equal its Pxx story clock",
                    expected=observed_duration,
                    observed=observed_effective,
                )
            if not math.isclose(
                observed_request,
                expected_request_duration,
                abs_tol=1e-6,
            ):
                add(
                    "secondary_storyboard_provider_request_duration_invalid",
                    f"{beat_id} Provider request duration is stale or incorrect",
                    expected=expected_request_duration,
                    observed=observed_request,
                )
            expected_padding = round(
                expected_request_duration - observed_duration,
                6,
            )
            if not math.isclose(observed_padding, expected_padding, abs_tol=1e-6):
                add(
                    "secondary_storyboard_provider_padding_invalid",
                    f"{beat_id} Provider padding does not reconcile with story time",
                    expected=expected_padding,
                    observed=observed_padding,
                )
        beat_actions = beat.get("micro_actions") or []
        if isinstance(beat_actions, str):
            beat_actions = [beat_actions]
        beat_actions = [str(value) for value in beat_actions if str(value).strip()]
        beat_units = beat.get("source_action_unit_ids") or []
        if isinstance(beat_units, str):
            beat_units = [beat_units]
        beat_units = [str(value) for value in beat_units if str(value).strip()]
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
        bridge_policy_bounds = bridge_planning_duration_bounds(requirement["profile"])
        bridge_provider_bounds = requirement["profile"].request_duration_bounds(
            "first_last_frame_bridge"
        )
        expected_planning = {
            "content_beat_count": requirement["content_count"],
            "extension_required": requirement["extension_required"],
            "bridge_required": requirement["bridge_required"],
            "bridge_duration_s": requirement["bridge_duration"],
            "first_last_frame_bridge_duration_range_s": list(bridge_policy_bounds),
            "first_last_frame_bridge_policy_duration_range_s": list(
                bridge_policy_bounds
            ),
            "first_last_frame_bridge_provider_duration_range_s": list(
                bridge_provider_bounds
            ),
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
                if (
                    bridge.get("target_shot_id") != next_sid
                    or bridge.get("execution_strategy")
                    != "first_last_frame_bridge"
                    or bridge.get("generation_phase") != "post_primary_shots"
                    or bridge.get("first_frame_source")
                    != "source_primary_video_tail_frame"
                    or bridge.get("last_frame_source")
                    != "target_primary_video_first_frame"
                    or bridge.get("timeline_insertion_policy")
                    != BRIDGE_TIMELINE_POLICY
                    or str(bridge.get("start_state") or "") != expected_current_end
                    or str(bridge.get("end_state") or "") != expected_next_start
                    or not math.isclose(
                        float(bridge.get("duration_s") or 0),
                        float(requirement["bridge_duration"]),
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        float(bridge.get("source_handle_s") or 0)
                        + float(bridge.get("target_handle_s") or 0),
                        float(requirement["bridge_duration"]),
                        abs_tol=1e-6,
                    )
                ):
                    add(
                        "secondary_storyboard_bridge_invalid",
                        f"{sid} bridge must use the completed source tail and target head",
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
    """Attach plot-faithful Pxx clips and separate post-primary bridges."""
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
        provider_capacity = MAX_CONTENT_BEATS
        multi_story_bounds = profile.effective_duration_bounds("multi_image")
        tail_story_bounds = profile.effective_duration_bounds("tail_video_extend")
        multi_request_bounds = profile.request_duration_bounds("multi_image")
        tail_request_bounds = profile.request_duration_bounds("tail_video_extend")
        bridge_provider_bounds = profile.request_duration_bounds(
            "first_last_frame_bridge"
        )
        bridge_policy_bounds = bridge_planning_duration_bounds(profile)
        shot["secondary_storyboard_planning"] = {
            "content_beat_count": requirement["content_count"],
            "generation_action_unit_count": len(
                requirement["generation_action_units"]
            ),
            "content_duration_s": requirement["content_duration"],
            "extension_required": requirement["extension_required"],
            "extension_reasons": requirement["extension_reasons"],
            "bridge_required": requirement["bridge_required"],
            "bridge_reason": requirement["bridge_reason"],
            "bridge_duration_s": requirement["bridge_duration"],
            "bridge_source_handle_s": (
                requirement["bridge_duration"] / 2
                if requirement["bridge_required"]
                else 0.0
            ),
            "bridge_target_handle_s": (
                requirement["bridge_duration"]
                - requirement["bridge_duration"] / 2
                if requirement["bridge_required"]
                else 0.0
            ),
            "bridge_timeline_policy": (
                BRIDGE_TIMELINE_POLICY if requirement["bridge_required"] else None
            ),
            "provider_capacity": provider_capacity,
            "duration_quantum_s": profile.duration_quantum_s,
            "primary_duration_range_s": [
                min_primary_story_duration(profile),
                max_primary_story_duration(profile),
            ],
            "multi_image_story_duration_range_s": list(multi_story_bounds),
            "tail_video_extend_story_duration_range_s": list(tail_story_bounds),
            "multi_image_provider_request_duration_range_s": list(
                multi_request_bounds
            ),
            "tail_video_extend_provider_request_duration_range_s": list(
                tail_request_bounds
            ),
            "first_last_frame_bridge_duration_range_s": list(bridge_policy_bounds),
            "first_last_frame_bridge_policy_duration_range_s": list(
                bridge_policy_bounds
            ),
            "first_last_frame_bridge_provider_duration_range_s": list(
                bridge_provider_bounds
            ),
            "bridge_generation_phase": "post_primary_shots",
            "selected_count": total_count,
        }
        planned.append((shot, requirement))

    # Reserve symmetric stable edge handles before authoring Pxx prompts.  A
    # generated bridge replaces these handles in Phase 8, so it is additive in
    # the provider-cost ledger without lengthening the edit timeline.
    for shot in shots:
        shot.pop("incoming_bridge_handle_s", None)
        shot.pop("outgoing_bridge_handle_s", None)
    for index, (shot, requirement) in enumerate(planned[:-1]):
        if not requirement["bridge_required"]:
            continue
        duration = float(requirement["bridge_duration"])
        source_handle = round(duration / 2, 6)
        target_handle = round(duration - source_handle, 6)
        shot["outgoing_bridge_handle_s"] = source_handle
        planned[index + 1][0]["incoming_bridge_handle_s"] = target_handle

    primary_bridges: list[dict[str, Any]] = []
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
        generation_action_units = requirement["generation_action_units"]
        generation_unit_buckets = _partition(
            generation_action_units, content_count
        )
        shot["generation_action_units"] = generation_action_units
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        action_units = list(dict.fromkeys(
            str(value) for value in raw_units if str(value).strip()
        ))
        source_unit_buckets = _partition(action_units, content_count)
        durations = requirement["durations"]
        start_state = _start_state(shot)
        final_state = _end_state(shot)
        beats: list[dict[str, Any]] = []
        for position in range(1, content_count + 1):
            beat_generation_units = generation_unit_buckets[position - 1]
            beat_source_event_ids = _generation_unit_source_event_ids(
                beat_generation_units
            )
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
            effective_story_duration = durations[position - 1]
            provider_request_duration = profile.request_duration_for_effective_story(
                effective_story_duration,
                generation_mode,
            )
            normalized = {
                "beat_id": f"{sid}_P{position:02d}",
                "position": position,
                "duration_s": effective_story_duration,
                "effective_story_duration_s": effective_story_duration,
                "provider_request_duration_s": provider_request_duration,
                "provider_minimum_padding_duration_s": round(
                    provider_request_duration - effective_story_duration,
                    6,
                ),
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
                "generation_action_units": beat_generation_units,
                "source_event_ids": beat_source_event_ids,
                "source_action_unit_ids": source_unit_buckets[position - 1],
                "end_state": next_state,
                "character_ids": _beat_character_ids(
                    shot,
                    previous_state,
                    action,
                    next_state,
                    source_event_ids=beat_source_event_ids,
                ),
                "shot_size": shot.get("shot_size") or shot.get("shot_type"),
                "camera_angle": shot.get("camera_angle"),
                "camera_movement": shot.get("camera_movement")
                or shot.get("camera_movement_en"),
                "lens_mm": shot.get("lens_mm"),
                "camera_motion_contract": dict(
                    shot.get("camera_motion_contract") or {}
                ),
                "lighting_key": shot.get("lighting_key"),
                "shot_intent": shot.get("shot_intent"),
                "hero_moment": bool(shot.get("hero_moment")),
                "texture_keywords": list(shot.get("texture_keywords") or []),
                "incoming_bridge_handle_s": (
                    float(shot.get("incoming_bridge_handle_s") or 0)
                    if position == 1
                    else 0.0
                ),
                "outgoing_bridge_handle_s": (
                    float(shot.get("outgoing_bridge_handle_s") or 0)
                    if position == content_count
                    else 0.0
                ),
                "edge_handle_contract": (
                    "hold the authored boundary state with only natural micro-motion; "
                    "do not place unique plot action inside a reserved bridge handle"
                    if (
                        (position == 1 and shot.get("incoming_bridge_handle_s"))
                        or (
                            position == content_count
                            and shot.get("outgoing_bridge_handle_s")
                        )
                    )
                    else ""
                ),
            }
            source_choreography = [
                dict(beat)
                for beat in (shot.get("body_action_choreography") or [])
                if isinstance(beat, dict)
                and (
                    not str(beat.get("micro_action") or "").strip()
                    or str(beat.get("micro_action") or "").strip()
                    in set(action_buckets[position - 1])
                )
            ]
            if source_choreography:
                normalized["body_action_choreography"] = source_choreography
            apply_body_action_contract(normalized)
            beats.append(normalized)
        if bridge_required:
            next_shot, next_requirement = planned[index + 1]
            next_sid = next_requirement["shot_id"]
            next_start_state = _start_state(next_shot)
            source_handle = float(shot.get("outgoing_bridge_handle_s") or 0)
            target_handle = float(next_shot.get("incoming_bridge_handle_s") or 0)
            bridge = {
                "bridge_id": f"{sid}__{next_sid}",
                "source_shot_id": sid,
                "target_shot_id": next_sid,
                "duration_s": requirement["bridge_duration"],
                "generation_duration_s": requirement["bridge_duration"],
                "generation_duration_range_s": list(
                    bridge_planning_duration_bounds(requirement["profile"])
                ),
                "visible_duration_s": requirement["bridge_duration"],
                "source_handle_s": source_handle,
                "target_handle_s": target_handle,
                "timeline_insertion_policy": BRIDGE_TIMELINE_POLICY,
                "execution_strategy": "first_last_frame_bridge",
                "planner_version": SECONDARY_STORYBOARD_VERSION,
                "plot_fidelity_contract": "primary_shot_source_only_no_invention",
                "start_state": final_state,
                "action_prompt": _compact(
                    f"保持{sid}结束动作的因果连续，从当前终态平滑过渡到"
                    f"{next_sid}成片起始状态；不得执行{next_sid}的新动作"
                ),
                "end_state": next_start_state,
                "boundary_kind": "continuous",
                "boundary_reason": bridge_reason,
                "generation_phase": "post_primary_shots",
                "first_frame_source": "source_primary_video_tail_frame",
                "last_frame_source": "target_primary_video_first_frame",
                "bridge_contract": (
                    "generate only after every primary video is complete; use the actual "
                    "source video tail as frame one and actual target video head as the "
                    "last frame"
                ),
            }
            primary_bridges.append(bridge)
        shot["storyboard_beats"] = beats
        shot["storyboard_beat_count"] = len(beats)
        # The top-level shot prompt is a narrative summary. Paid video prompts
        # are narrowed to one beat later by the continuity provider.
        shot["generation_actions"] = [
            beat["action"]
            for beat in beats
        ]
        shot["generation_load"] = {
            **(shot.get("generation_load") or {}),
            "storyboard_beats": len(beats),
            "content_beats": content_count,
            "generation_action_units": len(generation_action_units),
            "bridge_beats": 0,
            "post_primary_bridges": int(bridge_required),
            "execution": SECONDARY_EXECUTION,
            "capability_profile": profile.name,
        }
        total += len(beats)
    storyboard["storyboard_beat_count"] = total
    storyboard["primary_shot_bridges"] = primary_bridges
    storyboard["storyboard_execution"] = SECONDARY_EXECUTION
    storyboard["secondary_storyboard_version"] = SECONDARY_STORYBOARD_VERSION
    attach_material_budget(storyboard)
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
    budget_errors = material_budget_contract_errors(storyboard)
    if budget_errors:
        summary = "; ".join(error["message"] for error in budget_errors[:5])
        raise AssertionError(f"material budget planner emitted an invalid contract: {summary}")
    return storyboard
