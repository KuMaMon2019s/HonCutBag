"""Plan the per-shot visual beats that Phase 2 draws and Phase 6 executes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from schemas.replanning import PADDING_LOSS_ERROR_CODE
from utils.action_units import ACTION_TIMELINE_SCHEMA, normalize_action_units
from utils.body_action_contracts import (
    apply_body_action_contract,
    apply_body_action_kinematics_projection,
)
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

SECONDARY_STORYBOARD_VERSION = "honcut.secondary-storyboard.v17"
SECONDARY_EXECUTION = "canonical_timeline_post_primary_bridge_v17"
SECONDARY_GENERATION_MODES = frozenset({
    "multi_image",
    "tail_video_extend",
    "first_last_frame_bridge",
})
MAX_CONTENT_BEATS = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
MAX_CONTINUITY_CONTENT_BEATS = 4
PRIMARY_SHOT_LAYOUT_SCHEMA = "honcut.primary-shot-layout.v2"
PRIMARY_SHOT_EXECUTION_HANDOFF_SCHEMA = (
    "honcut.primary-shot-execution-handoff.v2"
)
CURRENT_SCREENPLAY_PLAN_SCHEMA = "honcut.screenplay-plan.v7"
CURRENT_EVENT_SCALING_SCHEMA = "honcut.duration-scaled-event-plan.v4"
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
    generation_action_units: list[dict[str, Any]] | None = None,
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

    def mentioned_ids(value: Any) -> set[str]:
        text = str(value or "")
        return {
            character_id
            for character_id in character_ids
            if any(
                _contains_complete_mention(text, mention)
                for mention in mentions_by_id[character_id]
            )
        }

    visible_ids = set().union(*(mentioned_ids(value) for value in semantic_values))
    event_casts = _source_event_cast_index(shot)
    if generation_action_units is None:
        for event_id in source_event_ids or []:
            visible_ids.update(event_casts.get(event_id, []))
    else:
        unit_event_ids = _generation_unit_source_event_ids(generation_action_units)
        expected_event_ids = list(dict.fromkeys(source_event_ids or []))
        unit_event_id_set = set(unit_event_ids)
        if (
            any(event_id not in expected_event_ids for event_id in unit_event_ids)
            or [
                event_id
                for event_id in expected_event_ids
                if event_id in unit_event_id_set
            ] != unit_event_ids
        ):
            raise ValueError(
                "beat cast generation units do not match source_event_ids"
            )
        for unit in generation_action_units:
            event_id = unit.get("source_event_id")
            event_cast = event_casts.get(event_id, [])
            unit_explicit_ids = set().union(*(
                mentioned_ids(action)
                for action in (unit.get("actions") or [])
                if str(action or "").strip()
            ))
            if not event_cast:
                visible_ids.update(unit_explicit_ids)
            elif not unit_explicit_ids or len(event_cast) <= 2:
                # An implicit unit inherits its canonical event cast.  Binary
                # interactions keep both endpoints even when one is expressed
                # through a pronoun or generic role such as "the opponent".
                visible_ids.update(event_cast)
            else:
                # A multi-party source event may be partitioned across Pxx.
                # When one unit explicitly names a strict subset, importing the
                # whole event cast would make characters appear before their
                # own action bucket.
                visible_ids.update(unit_explicit_ids & set(event_cast))
        for event_id in expected_event_ids:
            if event_id not in unit_event_id_set:
                # Static appearances, discoveries and other zero-motion source
                # events consume no GAU capacity but still own visible cast.
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


def _generation_unit_source_event_buckets(
    source_event_ids: list[int],
    generation_unit_buckets: list[list[dict[str, Any]]],
) -> list[list[int]]:
    """Project every ordered source event onto its canonical GAU buckets.

    A source event may intentionally have no motion units.  Such an event is
    attached to the first bucket of the next represented event, or to the last
    bucket of the preceding represented event when it closes the shot.  Events
    represented across multiple buckets remain visible in each owning bucket.
    """
    if not generation_unit_buckets:
        return []
    ordered_events = list(dict.fromkeys(source_event_ids))
    if any(
        not isinstance(event_id, int)
        or isinstance(event_id, bool)
        or event_id < 1
        for event_id in ordered_events
    ):
        raise ValueError("source_events must contain positive integer IDs")
    event_set = set(ordered_events)
    owners: dict[int, list[int]] = {event_id: [] for event_id in ordered_events}
    for bucket_index, bucket in enumerate(generation_unit_buckets):
        for event_id in _generation_unit_source_event_ids(bucket):
            if event_id not in event_set:
                raise ValueError(
                    "generation action unit references an event outside source_events"
                )
            if bucket_index not in owners[event_id]:
                owners[event_id].append(bucket_index)

    represented = [event_id for event_id in ordered_events if owners[event_id]]
    if not represented:
        return _partition(ordered_events, len(generation_unit_buckets))

    for position, event_id in enumerate(ordered_events):
        if owners[event_id]:
            continue
        following = next(
            (
                candidate
                for candidate in ordered_events[position + 1 :]
                if owners[candidate]
            ),
            None,
        )
        if following is not None:
            owners[event_id] = [min(owners[following])]
            continue
        preceding = next(
            (
                candidate
                for candidate in reversed(ordered_events[:position])
                if owners[candidate]
            ),
            None,
        )
        owners[event_id] = [
            max(owners[preceding]) if preceding is not None else 0
        ]

    return [
        [
            event_id
            for event_id in ordered_events
            if bucket_index in owners[event_id]
        ]
        for bucket_index in range(len(generation_unit_buckets))
    ]


def _generation_unit_action_buckets(
    source_actions: list[str],
    generation_unit_buckets: list[list[dict[str, Any]]],
) -> list[list[str]]:
    """Bind every source action to the same buckets as its generation units.

    ``ledger_indexes`` are the canonical link to the primary action ledger.
    Contiguous ranges also retain zero-capacity actions, such as camera motion
    or sustained environment state, without partitioning them independently.
    """
    if not generation_unit_buckets:
        return []
    if not source_actions:
        return [[] for _ in generation_unit_buckets]
    if not any(generation_unit_buckets):
        return _partition(source_actions, len(generation_unit_buckets))

    bucket_indexes: list[list[int]] = []
    all_have_indexes = True
    for bucket in generation_unit_buckets:
        indexes: list[int] = []
        for unit in bucket:
            raw_indexes = unit.get("ledger_indexes")
            if not isinstance(raw_indexes, list) or not raw_indexes:
                all_have_indexes = False
                continue
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(source_actions)
                for index in raw_indexes
            ):
                raise ValueError(
                    "generation action unit contains an invalid ledger index"
                )
            indexes.extend(raw_indexes)
        bucket_indexes.append(indexes)

    if all_have_indexes and all(bucket_indexes):
        for previous, current in zip(bucket_indexes, bucket_indexes[1:], strict=False):
            if max(previous) >= min(current):
                raise ValueError(
                    "generation action unit buckets cross primary action order"
                )
        starts = [0, *(min(indexes) for indexes in bucket_indexes[1:])]
        ends = [*starts[1:], len(source_actions)]
        buckets = [
            source_actions[start:end]
            for start, end in zip(starts, ends, strict=True)
        ]
    else:
        buckets = [
            [
                str(action).strip()
                for unit in bucket
                for action in (unit.get("actions") or [])
                if str(action or "").strip()
            ]
            for bucket in generation_unit_buckets
        ]
        if [action for bucket in buckets for action in bucket] != source_actions:
            raise ValueError(
                "generation action units cannot be reconciled with source actions"
            )

    if [action for bucket in buckets for action in bucket] != source_actions:
        raise ValueError(
            "generation action unit buckets do not cover the primary action ledger"
        )
    return buckets


def _generation_unit_source_action_unit_buckets(
    source_action_unit_ids: list[str],
    source_action_unit_refs: list[dict[str, Any]],
    source_event_buckets: list[list[int]],
    generation_unit_buckets: list[list[dict[str, Any]]],
) -> list[list[str]]:
    """Assign each source action-unit ID to its first owning Pxx bucket."""
    discovered = list(dict.fromkeys(
        str(unit.get("source_action_unit_id") or "").strip()
        for bucket in generation_unit_buckets
        for unit in bucket
        if str(unit.get("source_action_unit_id") or "").strip()
    ))
    if source_action_unit_refs:
        normalized_refs: list[tuple[int, str]] = []
        for reference in source_action_unit_refs:
            if not isinstance(reference, dict):
                raise ValueError("source_action_unit_refs entries must be objects")
            event_id = reference.get("source_event_id")
            action_unit_id = str(reference.get("action_unit_id") or "").strip()
            if (
                not isinstance(event_id, int)
                or isinstance(event_id, bool)
                or event_id < 1
                or not action_unit_id
            ):
                raise ValueError(
                    "source_action_unit_refs require source_event_id and action_unit_id"
                )
            normalized_refs.append((event_id, action_unit_id))
        if (
            [action_unit_id for _event_id, action_unit_id in normalized_refs]
            != source_action_unit_ids
            or len({event_id for event_id, _unit_id in normalized_refs})
            != len(normalized_refs)
            or len({unit_id for _event_id, unit_id in normalized_refs})
            != len(normalized_refs)
        ):
            raise ValueError(
                "source_action_unit_refs do not match source_action_unit_ids"
            )
        ref_by_unit = {
            action_unit_id: event_id
            for event_id, action_unit_id in normalized_refs
        }
        for bucket in generation_unit_buckets:
            for unit in bucket:
                action_unit_id = str(
                    unit.get("source_action_unit_id") or ""
                ).strip()
                if not action_unit_id:
                    continue
                if (
                    action_unit_id not in ref_by_unit
                    or unit.get("source_event_id") != ref_by_unit[action_unit_id]
                ):
                    raise ValueError(
                        "generation action unit conflicts with source_action_unit_refs"
                    )
        buckets = [[] for _ in generation_unit_buckets]
        for event_id, action_unit_id in normalized_refs:
            owner = next(
                (
                    bucket_index
                    for bucket_index, event_ids in enumerate(source_event_buckets)
                    if event_id in event_ids
                ),
                None,
            )
            if owner is None:
                raise ValueError("source action-unit event has no Pxx bucket owner")
            buckets[owner].append(action_unit_id)
        return buckets

    if not discovered:
        return _partition(source_action_unit_ids, len(generation_unit_buckets))
    if discovered != source_action_unit_ids:
        raise ValueError(
            "partial generation action-unit lineage requires source_action_unit_refs"
        )
    remaining = set(source_action_unit_ids)
    buckets: list[list[str]] = []
    for bucket in generation_unit_buckets:
        owned: list[str] = []
        for unit in bucket:
            unit_id = str(unit.get("source_action_unit_id") or "").strip()
            if unit_id and unit_id in remaining:
                owned.append(unit_id)
                remaining.remove(unit_id)
        buckets.append(owned)
    if remaining:
        raise ValueError(
            "source action-unit IDs have no generation bucket owner"
        )
    return buckets


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
    *,
    max_content_beats: int = MAX_CONTENT_BEATS,
) -> tuple[int, list[str]]:
    """Return the number of story-bearing clips required by provider capacity.

    Camera movement, genre and visual intensity never create an extension by
    themselves.  P02 exists only when P01 cannot carry the complete authored
    duration/action contract within one provider narrative window.
    """
    generation_action_units = _generation_action_units(shot, actions)
    action_count = (
        math.ceil(
            len(generation_action_units) / capabilities.temporal_slice_limit
        )
        if generation_action_units else 1
    )
    first_minimum, first_maximum = capabilities.effective_duration_bounds("multi_image")
    tail_minimum, tail_maximum = capabilities.effective_duration_bounds(
        "tail_video_extend"
    )
    duration_count = max_content_beats + 1
    for candidate in range(1, max_content_beats + 1):
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
    if required > max_content_beats:
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
    max_content_beats: int = MAX_CONTENT_BEATS,
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
        max_content_beats=max_content_beats,
    )
    return required


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _primary_layout_error(detail: str) -> ValueError:
    return ValueError(f"primary_shot_layout_handoff_invalid: {detail}")


def _validate_timeline_layout_binding(
    shots: list[dict[str, Any]],
    binding: dict[str, Any],
    layout: dict[str, Any],
) -> str:
    """Validate the one canonical temporal-slice → Sxx/Pxx handoff.

    Phase 1 has already solved the vector layout.  Every downstream owner must
    consume this mapping verbatim; reconstructing buckets from aggregate counts
    would silently change concurrency, effects, and event boundary state.
    """
    if binding.get("schema") != ACTION_TIMELINE_SCHEMA:
        raise _primary_layout_error("unsupported action-timeline binding schema")
    if binding.get("layout_schema") != PRIMARY_SHOT_LAYOUT_SCHEMA:
        raise _primary_layout_error("timeline binding layout schema disagrees")
    if binding.get("duration_plan_schema") != CURRENT_EVENT_SCALING_SCHEMA:
        raise _primary_layout_error("timeline binding duration schema disagrees")
    slice_limit = binding.get("max_temporal_slices_per_content_beat")
    motion_limit = binding.get("max_motion_contributions_per_slice")
    if (
        isinstance(slice_limit, bool)
        or not isinstance(slice_limit, int)
        or slice_limit < 1
        or isinstance(motion_limit, bool)
        or not isinstance(motion_limit, int)
        or motion_limit < 1
        or slice_limit != layout.get("max_temporal_slices_per_content_beat")
        or motion_limit != layout.get("max_motion_contributions_per_slice")
    ):
        raise _primary_layout_error("timeline binding capability matrix disagrees")
    sxx_records = binding.get("sxx")
    assignments = binding.get("assignments")
    zero_attachments = binding.get("zero_story_time_attachments")
    if (
        not isinstance(sxx_records, list)
        or len(sxx_records) != len(shots)
        or not isinstance(assignments, list)
        or not isinstance(zero_attachments, list)
    ):
        raise _primary_layout_error("timeline binding primary-shot ledger is invalid")
    zero_events_by_sxx: dict[str, list[int]] = {}
    seen_zero_events: set[int] = set()
    for attachment in zero_attachments:
        if not isinstance(attachment, dict):
            raise _primary_layout_error(
                "zero-story-time timeline attachment must be an object"
            )
        event_id = attachment.get("source_event_id")
        sxx_id = str(attachment.get("sxx_id") or "").strip()
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id < 1
            or event_id in seen_zero_events
            or not sxx_id
            or attachment.get("consumes_temporal_capacity") is not False
        ):
            raise _primary_layout_error(
                "zero-story-time timeline attachment is invalid"
            )
        seen_zero_events.add(event_id)
        zero_events_by_sxx.setdefault(sxx_id, []).append(event_id)
    assignment_by_id: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise _primary_layout_error("timeline assignment must be an object")
        assignment_id = str(assignment.get("assignment_id") or "").strip()
        event_id = assignment.get("source_event_id")
        slice_order = assignment.get("event_temporal_slice_order")
        motion_load = assignment.get("motion_load")
        if (
            not assignment_id
            or assignment_id in assignment_by_id
            or isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id < 1
            or isinstance(slice_order, bool)
            or not isinstance(slice_order, int)
            or slice_order < 1
            or isinstance(motion_load, bool)
            or not isinstance(motion_load, int)
            or not 1 <= motion_load <= motion_limit
        ):
            raise _primary_layout_error("timeline assignment identity/load is invalid")
        for key in (
            "source_micro_action_indexes",
            "contribution_micro_action_indexes",
            "effect_micro_action_indexes",
            "sustained_micro_action_indexes",
            "performers",
            "targets",
            "state_reads",
            "state_writes",
        ):
            if not isinstance(assignment.get(key), list):
                raise _primary_layout_error(
                    f"timeline assignment {assignment_id} has invalid {key}"
                )
        if not isinstance(assignment.get("start_state"), str) or not isinstance(
            assignment.get("end_state"), str
        ):
            raise _primary_layout_error(
                f"timeline assignment {assignment_id} has invalid boundary state"
            )
        assignment_by_id[assignment_id] = assignment

    observed_assignment_ids: list[str] = []
    for index, (shot, sxx) in enumerate(zip(shots, sxx_records, strict=True)):
        if not isinstance(sxx, dict) or sxx.get("sxx_order") != index + 1:
            raise _primary_layout_error("timeline Sxx order is invalid")
        pxx_records = sxx.get("pxx")
        expected_pxx_count = layout["content_beat_counts"][index]
        expected_capacity = expected_pxx_count * slice_limit
        if (
            not isinstance(pxx_records, list)
            or len(pxx_records) != expected_pxx_count
            or sxx.get("temporal_slice_capacity") != expected_capacity
        ):
            raise _primary_layout_error("timeline Sxx/Pxx capacity is invalid")
        shot_assignment_ids: list[str] = []
        for pxx_order, pxx in enumerate(pxx_records, 1):
            if (
                not isinstance(pxx, dict)
                or pxx.get("pxx_order_within_sxx") != pxx_order
                or pxx.get("temporal_slice_capacity") != slice_limit
                or not isinstance(pxx.get("assignment_ids"), list)
                or len(pxx["assignment_ids"]) > slice_limit
                or not math.isclose(
                    float(pxx.get("effective_story_duration_s") or 0),
                    float(layout["effective_story_durations_s"][index][pxx_order - 1]),
                    abs_tol=1e-6,
                )
            ):
                raise _primary_layout_error("timeline Pxx ledger is invalid")
            for assignment_id in pxx["assignment_ids"]:
                if assignment_id not in assignment_by_id:
                    raise _primary_layout_error(
                        "timeline Pxx references an unknown assignment"
                    )
                assignment = assignment_by_id[assignment_id]
                if (
                    assignment.get("sxx_id") != sxx.get("sxx_id")
                    or assignment.get("pxx_id") != pxx.get("pxx_id")
                ):
                    raise _primary_layout_error(
                        "timeline assignment disagrees with its Sxx/Pxx owner"
                    )
                shot_assignment_ids.append(assignment_id)
            if pxx.get("assigned_pace_weight") != sum(
                int(assignment_by_id[assignment_id].get("pace_weight") or 1)
                for assignment_id in pxx["assignment_ids"]
            ):
                raise _primary_layout_error("timeline Pxx pace ledger is invalid")
        if len(shot_assignment_ids) != sxx.get("assigned_temporal_slice_count"):
            raise _primary_layout_error("timeline Sxx assigned-count is invalid")
        expected_event_counts = shot.get("source_event_generation_unit_counts")
        expected_zero_events = zero_events_by_sxx.get(
            str(sxx.get("sxx_id") or ""),
            [],
        )
        if sxx.get("zero_story_time_source_event_ids", []) != expected_zero_events:
            raise _primary_layout_error(
                "timeline Sxx zero-story-time attachment ledger is invalid"
            )
        observed_event_counts: dict[str, int] = {}
        for assignment_id in shot_assignment_ids:
            event_key = str(assignment_by_id[assignment_id]["source_event_id"])
            observed_event_counts[event_key] = observed_event_counts.get(event_key, 0) + 1
        for event_id in expected_zero_events:
            event_key = str(event_id)
            if event_key in observed_event_counts:
                raise _primary_layout_error(
                    "execution event cannot also be a zero-story-time attachment"
                )
            observed_event_counts[event_key] = 0
        if expected_event_counts != observed_event_counts:
            raise _primary_layout_error(
                "timeline Sxx event allocation disagrees with the production shot"
            )
        generation_units = _generation_action_units(shot, _source_actions(shot))
        if len(generation_units) != len(shot_assignment_ids):
            raise _primary_layout_error(
                "timeline Sxx assignment count disagrees with generation units"
            )
        unit_event_ids = [unit.get("source_event_id") for unit in generation_units]
        assignment_event_ids = [
            assignment_by_id[assignment_id]["source_event_id"]
            for assignment_id in shot_assignment_ids
        ]
        if unit_event_ids != assignment_event_ids:
            raise _primary_layout_error(
                "timeline assignment order disagrees with generation-unit lineage"
            )
        observed_assignment_ids.extend(shot_assignment_ids)
    if observed_assignment_ids != list(assignment_by_id):
        raise _primary_layout_error(
            "timeline assignments are duplicated, missing, or out of canonical order"
        )
    return _canonical_sha256(binding)


def _timeline_generation_unit_buckets(
    shot: dict[str, Any],
    timeline_sxx: dict[str, Any],
    assignment_by_id: dict[str, dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Project exact canonical assignments onto the shot's persisted GAUs."""
    units = _generation_action_units(shot, _source_actions(shot))
    unit_cursor = 0
    unit_buckets: list[list[dict[str, Any]]] = []
    assignment_buckets: list[list[dict[str, Any]]] = []
    for pxx in timeline_sxx["pxx"]:
        pxx_assignments = [
            assignment_by_id[assignment_id]
            for assignment_id in pxx["assignment_ids"]
        ]
        pxx_units = units[unit_cursor:unit_cursor + len(pxx_assignments)]
        unit_cursor += len(pxx_assignments)
        if [unit.get("source_event_id") for unit in pxx_units] != [
            assignment["source_event_id"] for assignment in pxx_assignments
        ]:
            raise _primary_layout_error(
                "canonical Pxx assignments disagree with generation-unit order"
            )
        unit_buckets.append([copy.deepcopy(unit) for unit in pxx_units])
        assignment_buckets.append(
            [copy.deepcopy(assignment) for assignment in pxx_assignments]
        )
    if unit_cursor != len(units):
        raise _primary_layout_error("canonical Pxx assignments left units unbound")
    return unit_buckets, assignment_buckets


def _requires_primary_layout(storyboard: dict[str, Any]) -> bool:
    screenplay = storyboard.get("screenplay_plan")
    return bool(
        storyboard.get("primary_shot_execution")
        or (
            isinstance(screenplay, dict)
            and screenplay.get("schema") == CURRENT_SCREENPLAY_PLAN_SCHEMA
            and str(storyboard.get("shot_policy") or "").strip().lower()
            == "continuity"
        )
    )


def _validate_primary_shot_layout(
    shots: list[dict[str, Any]],
    layout: dict[str, Any],
    capabilities: VideoModelCapabilities | None = None,
    *,
    require_event_allocation: bool = False,
) -> str:
    """Validate one immutable Sxx/Pxx layout against the transformed shots."""
    if layout.get("schema") != PRIMARY_SHOT_LAYOUT_SCHEMA:
        raise _primary_layout_error("unsupported primary-shot layout schema")
    policy = str(layout.get("shot_policy") or "").strip().lower()
    if policy not in {"continuity", "balanced", "cut-driven"}:
        raise _primary_layout_error("unknown shot policy")
    declared_primary_shots = layout.get("primary_shots")
    if (
        isinstance(declared_primary_shots, bool)
        or not isinstance(declared_primary_shots, int)
        or declared_primary_shots != len(shots)
    ):
        raise _primary_layout_error(
            "primary shot count does not match storyboard shots"
        )

    vector_fields = (
        "story_duration_allocations_s",
        "content_beat_counts",
        "effective_story_durations_s",
        "provider_request_durations_s",
        "generation_action_unit_capacities",
    )
    vectors: dict[str, list[Any]] = {}
    for field in vector_fields:
        value = layout.get(field)
        if not isinstance(value, list) or len(value) != len(shots):
            raise _primary_layout_error(
                f"{field} must cover every primary shot"
            )
        vectors[field] = value

    policy_limit = (
        MAX_CONTINUITY_CONTENT_BEATS
        if policy == "continuity"
        else MAX_CONTENT_BEATS
    )
    declared_limit = layout.get("max_content_beats_per_primary_shot")
    if (
        isinstance(declared_limit, bool)
        or not isinstance(declared_limit, int)
        or declared_limit < 1
        or declared_limit > policy_limit
    ):
        raise _primary_layout_error("invalid maximum content-beat limit")

    observed_request_duration = 0.0
    observed_story_duration = 0.0
    observed_action_units = 0
    for index, shot in enumerate(shots):
        sid = _shot_id(shot, index + 1)
        duration = shot.get("duration") or shot.get("suggested_duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise _primary_layout_error(f"{sid} has no numeric duration")
        planned_duration = vectors["story_duration_allocations_s"][index]
        if (
            isinstance(planned_duration, bool)
            or not isinstance(planned_duration, (int, float))
            or not math.isclose(
                float(duration), float(planned_duration), abs_tol=1e-6
            )
        ):
            raise _primary_layout_error(
                f"{sid} duration does not match the canonical layout"
            )

        content_count = vectors["content_beat_counts"][index]
        if (
            isinstance(content_count, bool)
            or not isinstance(content_count, int)
            or not 1 <= content_count <= declared_limit
        ):
            raise _primary_layout_error(
                f"{sid} has an invalid declared Pxx count"
            )
        effective_durations = vectors["effective_story_durations_s"][index]
        request_durations = vectors["provider_request_durations_s"][index]
        if (
            not isinstance(effective_durations, list)
            or not isinstance(request_durations, list)
            or len(effective_durations) != content_count
            or len(request_durations) != content_count
        ):
            raise _primary_layout_error(
                f"{sid} Pxx duration ledger does not match its declared count"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in [*effective_durations, *request_durations]
        ):
            raise _primary_layout_error(f"{sid} Pxx duration ledger is invalid")
        if not math.isclose(
            sum(float(value) for value in effective_durations),
            float(planned_duration),
            abs_tol=1e-6,
        ):
            raise _primary_layout_error(
                f"{sid} Pxx story durations do not sum to the Sxx duration"
            )
        observed_story_duration += float(planned_duration)
        observed_request_duration += sum(
            float(value) for value in request_durations
        )

        profile = capabilities or capabilities_for(shot)
        modes = ["multi_image"] + [
            "tail_video_extend"
        ] * (content_count - 1)
        for position, (effective, requested, mode) in enumerate(
            zip(effective_durations, request_durations, modes, strict=True),
            1,
        ):
            expected_request = profile.request_duration_for_effective_story(
                float(effective), mode
            )
            if not math.isclose(
                float(requested), float(expected_request), abs_tol=1e-6
            ):
                raise _primary_layout_error(
                    f"{sid}_P{position:02d} Provider duration disagrees with "
                    "the capability profile"
                )

        capacity = vectors["generation_action_unit_capacities"][index]
        expected_capacity = content_count * profile.temporal_slice_limit
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity != expected_capacity
        ):
            raise _primary_layout_error(
                f"{sid} action capacity does not match its Pxx count"
            )
        source_actions = _source_actions(shot)
        generation_units = _generation_action_units(shot, source_actions)
        observed_action_units += len(generation_units)
        if len(generation_units) > capacity:
            raise _primary_layout_error(
                f"{sid} action load {len(generation_units)} exceeds capacity {capacity}"
            )
        allocation = shot.get("source_event_generation_unit_counts")
        if require_event_allocation and shot.get("source_events"):
            if not isinstance(allocation, dict):
                raise _primary_layout_error(
                    f"{sid} is missing source-event action allocation"
                )
            expected_event_ids = {
                str(event_id) for event_id in shot["source_events"]
            }
            if set(allocation) != expected_event_ids:
                raise _primary_layout_error(
                    f"{sid} source-event action allocation keys disagree"
                )
            values = list(allocation.values())
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            ) or sum(values) != len(generation_units):
                raise _primary_layout_error(
                    f"{sid} source-event action allocation does not match its load"
                )
        required, _reasons = _content_beat_requirement(
            shot,
            float(duration),
            source_actions,
            profile,
            max_content_beats=policy_limit,
        )
        if required != content_count:
            raise _primary_layout_error(
                f"{sid} requires {required} Pxx but the canonical layout declares "
                f"{content_count}"
            )

    capacities = vectors["generation_action_unit_capacities"]
    total_capacity = sum(capacities)
    production_target = layout.get("production_action_unit_target")
    projected_request = layout.get(
        "projected_content_provider_request_duration_s"
    )
    projected_padding = layout.get(
        "projected_content_provider_padding_duration_s"
    )
    projected_padding_rate = layout.get("projected_padding_loss_rate")
    maximum_padding_rate = layout.get("maximum_padding_loss_rate")
    expected_padding = observed_request_duration - observed_story_duration
    expected_padding_rate = (
        expected_padding / observed_request_duration
        if observed_request_duration
        else 0.0
    )
    objective_decision = layout.get("objective_decision")
    if (
        layout.get("total_generation_action_unit_capacity") != total_capacity
        or layout.get("max_generation_action_units_per_primary_shot")
        != max(capacities, default=0)
        or isinstance(production_target, bool)
        or not isinstance(production_target, int)
        or not 0 <= production_target <= total_capacity
        or (
            require_event_allocation
            and observed_action_units != production_target
        )
        or layout.get("cross_sxx_boundary_count")
        != max(0, declared_primary_shots - 1)
        or isinstance(projected_request, bool)
        or not isinstance(projected_request, (int, float))
        or not math.isclose(
            float(projected_request),
            observed_request_duration,
            abs_tol=1e-6,
        )
        or isinstance(projected_padding, bool)
        or not isinstance(projected_padding, (int, float))
        or not math.isclose(
            float(projected_padding), expected_padding, abs_tol=1e-6
        )
        or isinstance(projected_padding_rate, bool)
        or not isinstance(projected_padding_rate, (int, float))
        or not math.isclose(
            float(projected_padding_rate),
            expected_padding_rate,
            abs_tol=1e-6,
        )
        or isinstance(maximum_padding_rate, bool)
        or not isinstance(maximum_padding_rate, (int, float))
        or not 0 <= float(maximum_padding_rate) <= 1
        or not isinstance(layout.get("capability_profile"), str)
        or not layout["capability_profile"].strip()
        or not isinstance(layout.get("objective_order"), list)
        or not layout["objective_order"]
        or not isinstance(objective_decision, dict)
        or objective_decision.get("selected_primary_shot_count")
        != declared_primary_shots
        or objective_decision.get("selected_production_action_unit_target")
        != production_target
    ):
        raise _primary_layout_error(
            "primary-shot layout aggregate ledger is inconsistent"
        )
    return _canonical_sha256(layout)


def bind_primary_shot_execution_plan(
    storyboard: dict[str, Any],
    screenplay_plan: dict[str, Any],
    screenplay_plan_sha256: str,
    *,
    projected_layout: dict[str, Any],
    capacity_layout: dict[str, Any],
    capabilities: VideoModelCapabilities | None = None,
) -> dict[str, Any]:
    """Bind the canonical screenplay layout before any Pxx inference occurs."""
    if screenplay_plan.get("schema") != CURRENT_SCREENPLAY_PLAN_SCHEMA:
        raise _primary_layout_error("unsupported screenplay plan schema")
    observed_plan_sha256 = _canonical_sha256(screenplay_plan)
    if screenplay_plan_sha256 != observed_plan_sha256:
        raise _primary_layout_error("screenplay plan hash mismatch")
    layout = screenplay_plan.get("primary_shot_layout")
    if not isinstance(layout, dict):
        raise _primary_layout_error("canonical primary-shot layout is missing")
    for name, candidate, required in (
        ("adaptation projection", projected_layout, True),
        ("capacity projection", capacity_layout, True),
        ("storyboard projection", storyboard.get("primary_shot_layout"), False),
    ):
        if candidate is None and not required:
            continue
        if not isinstance(candidate, dict):
            raise _primary_layout_error(f"{name} is missing")
        if candidate != layout:
            raise _primary_layout_error(f"{name} disagrees with SCREENPLAY_PLAN")
    shots = [
        shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)
    ]
    layout_sha256 = _validate_primary_shot_layout(
        shots,
        layout,
        capabilities,
        require_event_allocation=True,
    )
    production_ledger = screenplay_plan.get("production_ledger")
    event_scaling = screenplay_plan.get("event_action_scaling")
    scaling_records = (
        event_scaling.get("events")
        if isinstance(event_scaling, dict)
        else None
    )
    if (
        not isinstance(production_ledger, dict)
        or not isinstance(event_scaling, dict)
        or event_scaling.get("schema") != CURRENT_EVENT_SCALING_SCHEMA
        or not isinstance(scaling_records, list)
    ):
        raise _primary_layout_error(
            "screenplay production event allocation is missing"
        )
    expected_event_units: dict[str, int] = {}
    for record in scaling_records:
        if not isinstance(record, dict):
            raise _primary_layout_error(
                "screenplay production event allocation is invalid"
            )
        status = record.get("production_status")
        if status not in {"kept", "whole_event_omitted"}:
            raise _primary_layout_error(
                "screenplay production event status is invalid"
            )
        if status == "whole_event_omitted":
            continue
        event_id = record.get("source_event_id")
        units = record.get("production_generation_action_units")
        event_key = str(event_id)
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, (int, str))
            or not event_key.strip()
            or event_key in expected_event_units
            or isinstance(units, bool)
            or not isinstance(units, int)
            or units < 0
        ):
            raise _primary_layout_error(
                "screenplay production event allocation is invalid"
            )
        expected_event_units[event_key] = units
    observed_event_units = {event_id: 0 for event_id in expected_event_units}
    for shot in shots:
        for event_id, units in shot[
            "source_event_generation_unit_counts"
        ].items():
            if event_id not in observed_event_units:
                raise _primary_layout_error(
                    "storyboard contains an unplanned source event allocation"
                )
            observed_event_units[event_id] += units
    production_units = production_ledger.get("generation_action_units")
    kept_event_ids = production_ledger.get("kept_source_event_ids")
    if (
        isinstance(production_units, bool)
        or not isinstance(production_units, int)
        or production_units != sum(expected_event_units.values())
        or production_units != layout.get("production_action_unit_target")
        or not isinstance(kept_event_ids, list)
        or {str(event_id) for event_id in kept_event_ids}
        != set(expected_event_units)
        or observed_event_units != expected_event_units
    ):
        raise _primary_layout_error(
            "storyboard source-event allocation disagrees with SCREENPLAY_PLAN"
        )
    timeline_binding = screenplay_plan.get("timeline_layout_binding")
    if not isinstance(timeline_binding, dict):
        raise _primary_layout_error("canonical timeline-layout binding is missing")
    timeline_binding_sha256 = _validate_timeline_layout_binding(
        shots,
        timeline_binding,
        layout,
    )
    storyboard["shot_policy"] = layout["shot_policy"]
    storyboard["primary_shot_layout"] = copy.deepcopy(layout)
    storyboard["primary_shot_layout_sha256"] = layout_sha256
    storyboard["timeline_layout_binding"] = copy.deepcopy(timeline_binding)
    storyboard["timeline_layout_binding_sha256"] = timeline_binding_sha256
    storyboard["primary_shot_execution"] = {
        "schema": PRIMARY_SHOT_EXECUTION_HANDOFF_SCHEMA,
        "screenplay_plan_schema": CURRENT_SCREENPLAY_PLAN_SCHEMA,
        "screenplay_plan_sha256": screenplay_plan_sha256,
        "primary_shot_layout_schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
        "primary_shot_layout_sha256": layout_sha256,
        "action_timeline_schema": ACTION_TIMELINE_SCHEMA,
        "timeline_layout_binding_sha256": timeline_binding_sha256,
    }
    existing_summary = storyboard.get("screenplay_plan")
    if existing_summary is not None and not isinstance(existing_summary, dict):
        raise _primary_layout_error("storyboard screenplay summary is invalid")
    existing_summary = dict(existing_summary or {})
    existing_schema = existing_summary.get("schema")
    if existing_schema is not None and existing_schema != CURRENT_SCREENPLAY_PLAN_SCHEMA:
        raise _primary_layout_error("storyboard screenplay schema disagrees")
    existing_sha256 = existing_summary.get("sha256")
    if existing_sha256 is not None and existing_sha256 != screenplay_plan_sha256:
        raise _primary_layout_error("storyboard screenplay hash disagrees")
    existing_layout_sha256 = existing_summary.get("primary_shot_layout_sha256")
    if (
        existing_layout_sha256 is not None
        and existing_layout_sha256 != layout_sha256
    ):
        raise _primary_layout_error("storyboard layout hash disagrees")
    storyboard["screenplay_plan"] = {
        **existing_summary,
        "schema": CURRENT_SCREENPLAY_PLAN_SCHEMA,
        "sha256": screenplay_plan_sha256,
        "primary_shot_layout_sha256": layout_sha256,
        "timeline_layout_binding_sha256": timeline_binding_sha256,
    }
    return storyboard["primary_shot_layout"]


def _primary_shot_layout_entry(
    storyboard: dict[str, Any],
    index: int,
    capabilities: VideoModelCapabilities | None = None,
) -> dict[str, Any] | None:
    layout = storyboard.get("primary_shot_layout")
    if not isinstance(layout, dict):
        if _requires_primary_layout(storyboard):
            raise _primary_layout_error("canonical primary-shot layout is missing")
        return None
    shots = [
        shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)
    ]
    layout_sha256 = _validate_primary_shot_layout(
        shots,
        layout,
        capabilities,
        require_event_allocation=bool(storyboard.get("primary_shot_execution")),
    )
    execution = storyboard.get("primary_shot_execution")
    if execution is not None:
        timeline_binding = storyboard.get("timeline_layout_binding")
        if not isinstance(timeline_binding, dict):
            raise _primary_layout_error("canonical timeline-layout binding is missing")
        timeline_binding_sha256 = _validate_timeline_layout_binding(
            shots,
            timeline_binding,
            layout,
        )
        screenplay = storyboard.get("screenplay_plan")
        recorded_plan_sha256 = (
            screenplay.get("sha256") if isinstance(screenplay, dict) else None
        )
        execution_plan_sha256 = (
            execution.get("screenplay_plan_sha256")
            if isinstance(execution, dict)
            else None
        )
        if (
            not isinstance(execution, dict)
            or execution.get("schema")
            != PRIMARY_SHOT_EXECUTION_HANDOFF_SCHEMA
            or execution.get("screenplay_plan_schema")
            != CURRENT_SCREENPLAY_PLAN_SCHEMA
            or not isinstance(execution_plan_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", execution_plan_sha256)
            or (
                not isinstance(screenplay, dict)
                or screenplay.get("schema") != CURRENT_SCREENPLAY_PLAN_SCHEMA
                or recorded_plan_sha256 != execution_plan_sha256
                or screenplay.get("primary_shot_layout_sha256")
                != layout_sha256
            )
            or execution.get("primary_shot_layout_schema")
            != PRIMARY_SHOT_LAYOUT_SCHEMA
            or execution.get("primary_shot_layout_sha256") != layout_sha256
            or storyboard.get("primary_shot_layout_sha256") != layout_sha256
            or execution.get("action_timeline_schema") != ACTION_TIMELINE_SCHEMA
            or execution.get("timeline_layout_binding_sha256")
            != timeline_binding_sha256
            or storyboard.get("timeline_layout_binding_sha256")
            != timeline_binding_sha256
            or screenplay.get("timeline_layout_binding_sha256")
            != timeline_binding_sha256
        ):
            raise _primary_layout_error("execution handoff receipt is inconsistent")
    else:
        timeline_binding = None
        timeline_binding_sha256 = None
    timeline_sxx = (
        timeline_binding["sxx"][index]
        if isinstance(timeline_binding, dict)
        else None
    )
    assignment_by_id = (
        {
            assignment["assignment_id"]: assignment
            for assignment in timeline_binding["assignments"]
        }
        if isinstance(timeline_binding, dict)
        else {}
    )
    return {
        "layout_sha256": layout_sha256,
        "content_count": layout["content_beat_counts"][index],
        "action_capacity": layout[
            "generation_action_unit_capacities"
        ][index],
        "effective_story_durations": list(
            layout["effective_story_durations_s"][index]
        ),
        "provider_request_durations": list(
            layout["provider_request_durations_s"][index]
        ),
        "policy_limit": layout["max_content_beats_per_primary_shot"],
        "timeline_binding_sha256": timeline_binding_sha256,
        "timeline_sxx": copy.deepcopy(timeline_sxx),
        "timeline_assignment_by_id": copy.deepcopy(assignment_by_id),
    }


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
    layout_entry = _primary_shot_layout_entry(storyboard, index, profile)
    max_content_beats = (
        layout_entry["policy_limit"]
        if layout_entry is not None
        else MAX_CONTENT_BEATS
    )
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
        max_content_beats=max_content_beats,
    )
    if (
        layout_entry is not None
        and content_count != layout_entry["content_count"]
    ):
        raise _primary_layout_error(
            f"{sid} requires {content_count} Pxx but the canonical layout "
            f"declares {layout_entry['content_count']}"
        )
    first_minimum, first_maximum = profile.effective_duration_bounds("multi_image")
    tail_minimum, tail_maximum = profile.effective_duration_bounds("tail_video_extend")
    content_durations = (
        list(layout_entry["effective_story_durations"])
        if layout_entry is not None
        else _duration_budgets(
            content_duration,
            content_count,
            profile,
            minimum_durations=[first_minimum]
            + [tail_minimum] * (content_count - 1),
            maximum_durations=[first_maximum]
            + [tail_maximum] * (content_count - 1),
        )
    )
    modes = ["multi_image"] + ["tail_video_extend"] * (content_count - 1)
    if len(modes) > max_content_beats:
        raise ValueError(
            f"{sid} requires {len(modes)} secondary beats, above the "
            f"{max_content_beats}-beat contract"
        )
    provider_request_durations = (
        list(layout_entry["provider_request_durations"])
        if layout_entry is not None
        else [
            profile.request_duration_for_effective_story(duration, mode)
            for duration, mode in zip(content_durations, modes, strict=True)
        ]
    )
    if layout_entry is not None and layout_entry["timeline_sxx"] is not None:
        (
            generation_unit_buckets,
            timeline_assignment_buckets,
        ) = _timeline_generation_unit_buckets(
            shot,
            layout_entry["timeline_sxx"],
            layout_entry["timeline_assignment_by_id"],
        )
    else:
        # Historical v15 storyboards have no canonical timeline handoff.  Keep
        # their deterministic legacy bucket shape; new v16 runs never enter it.
        generation_unit_buckets = _partition(
            generation_action_units,
            content_count,
        )
        timeline_assignment_buckets = [[] for _ in range(content_count)]
    return {
        "shot_id": sid,
        "profile": profile,
        "duration": duration,
        "source_actions": source_actions,
        "generation_action_units": generation_action_units,
        "generation_unit_buckets": generation_unit_buckets,
        "timeline_assignment_buckets": timeline_assignment_buckets,
        "timeline_layout_binding_sha256": (
            layout_entry["timeline_binding_sha256"]
            if layout_entry is not None
            else None
        ),
        "content_duration": content_duration,
        "content_count": content_count,
        "provider_capacity": (
            layout_entry["content_count"]
            if layout_entry is not None
            else max_content_beats
        ),
        "max_content_beats": max_content_beats,
        "declared_content_beat_count": (
            layout_entry["content_count"]
            if layout_entry is not None
            else None
        ),
        "declared_generation_action_unit_capacity": (
            layout_entry["action_capacity"]
            if layout_entry is not None
            else None
        ),
        "primary_shot_layout_sha256": (
            layout_entry["layout_sha256"]
            if layout_entry is not None
            else None
        ),
        "content_durations": content_durations,
        "extension_required": content_count > 1,
        "extension_reasons": extension_reasons,
        "bridge_required": bridge_required,
        "bridge_reason": bridge_reason,
        "bridge_duration": bridge_duration,
        "modes": modes,
        "durations": content_durations,
        "provider_request_durations": provider_request_durations,
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
    source_action_unit_refs = shot.get("source_action_unit_refs") or []
    if not isinstance(source_action_unit_refs, list):
        source_action_unit_refs = []
    if (
        storyboard.get("semantic_understanding")
        and source_action_units
        and not source_action_unit_refs
    ):
        add(
            "secondary_storyboard_source_action_unit_ref_invalid",
            f"{sid} canonical action-unit lineage requires source_action_unit_refs",
        )
    expected_generation_unit_buckets: list[list[dict[str, Any]]] = []
    expected_action_buckets: list[list[str]] = []
    expected_source_event_buckets: list[list[int]] = []
    expected_source_action_unit_buckets: list[list[str]] = []
    if requirement:
        expected_generation_unit_buckets = requirement[
            "generation_unit_buckets"
        ]
        try:
            expected_source_event_buckets = _generation_unit_source_event_buckets(
                source_events,
                expected_generation_unit_buckets,
            )
            expected_source_action_unit_buckets = (
                _generation_unit_source_action_unit_buckets(
                    source_action_units,
                    source_action_unit_refs,
                    expected_source_event_buckets,
                    expected_generation_unit_buckets,
                )
            )
            expected_action_buckets = _generation_unit_action_buckets(
                requirement["source_actions"],
                expected_generation_unit_buckets,
            )
        except ValueError as exc:
            add(
                "secondary_storyboard_generation_lineage_invalid",
                f"{sid} cannot reconcile generation-unit lineage: {exc}",
            )
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
        expected_generation_units = (
            expected_generation_unit_buckets[position - 1]
            if position <= len(expected_generation_unit_buckets)
            else []
        )
        if beat_generation_units != expected_generation_units:
            add(
                "secondary_storyboard_beat_generation_lineage_invalid",
                f"{beat_id} generation units do not match its canonical bucket",
                expected=expected_generation_units,
                observed=beat_generation_units,
            )
        expected_timeline_assignments = (
            requirement["timeline_assignment_buckets"][position - 1]
            if requirement
            and position <= len(requirement["timeline_assignment_buckets"])
            else []
        )
        expected_assignment_ids = [
            assignment["assignment_id"]
            for assignment in expected_timeline_assignments
        ]
        if beat.get("timeline_assignment_ids") != expected_assignment_ids:
            add(
                "secondary_storyboard_timeline_assignment_invalid",
                f"{beat_id} does not consume its canonical timeline assignments",
                expected=expected_assignment_ids,
                observed=beat.get("timeline_assignment_ids"),
            )
        expected_source_event_ids = (
            expected_source_event_buckets[position - 1]
            if position <= len(expected_source_event_buckets)
            else []
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
            try:
                expected_beat_character_ids = _beat_character_ids(
                    shot,
                    beat.get("action"),
                    beat.get("end_state"),
                    source_event_ids=expected_source_event_ids,
                    generation_action_units=expected_generation_units,
                )
            except ValueError as exc:
                expected_beat_character_ids = []
                add(
                    "secondary_storyboard_beat_cast_lineage_invalid",
                    f"{beat_id} cannot reconcile cast lineage: {exc}",
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
        if storyboard.get("primary_shot_execution") and beat.get(
            "primary_shot_layout_sha256"
        ) != storyboard.get("primary_shot_layout_sha256"):
            add(
                "secondary_storyboard_layout_lineage_invalid",
                f"{beat_id} is not bound to the canonical primary-shot layout",
                expected=storyboard.get("primary_shot_layout_sha256"),
                observed=beat.get("primary_shot_layout_sha256"),
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
        expected_beat_actions = (
            expected_action_buckets[position - 1]
            if position <= len(expected_action_buckets)
            else []
        )
        if beat_actions != expected_beat_actions:
            add(
                "secondary_storyboard_beat_action_lineage_invalid",
                f"{beat_id} actions do not match its generation-unit bucket",
                expected=expected_beat_actions,
                observed=beat_actions,
            )
        expected_beat_units = (
            expected_source_action_unit_buckets[position - 1]
            if position <= len(expected_source_action_unit_buckets)
            else []
        )
        if beat_units != expected_beat_units:
            add(
                "secondary_storyboard_beat_action_unit_lineage_invalid",
                f"{beat_id} source action-unit IDs do not match its bucket",
                expected=expected_beat_units,
                observed=beat_units,
            )
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
            next_requirement = secondary_storyboard_requirements(
                storyboard,
                index + 1,
                capabilities_for({**storyboard, **next_shot}),
            )
            source_boundary_assignment = next((
                assignment
                for bucket in reversed(requirement["timeline_assignment_buckets"])
                for assignment in reversed(bucket)
            ), None)
            target_boundary_assignment = next((
                assignment
                for bucket in next_requirement["timeline_assignment_buckets"]
                for assignment in bucket
            ), None)
            expected_current_end = (
                str(source_boundary_assignment.get("end_state") or "").strip()
                if source_boundary_assignment is not None
                else ""
            ) or _end_state(shot)
            expected_next_start = (
                str(target_boundary_assignment.get("start_state") or "").strip()
                if target_boundary_assignment is not None
                else ""
            ) or _start_state(next_shot)
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
                    or bridge.get("source_timeline_assignment_id")
                    != (
                        source_boundary_assignment["assignment_id"]
                        if source_boundary_assignment is not None
                        else None
                    )
                    or bridge.get("target_timeline_assignment_id")
                    != (
                        target_boundary_assignment["assignment_id"]
                        if target_boundary_assignment is not None
                        else None
                    )
                    or bridge.get("timeline_layout_binding_sha256")
                    != requirement["timeline_layout_binding_sha256"]
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
    if storyboard.get("primary_shot_layout") is not None:
        _validate_primary_shot_layout(
            shots,
            storyboard["primary_shot_layout"],
            capabilities,
            require_event_allocation=bool(
                storyboard.get("primary_shot_execution")
            ),
        )
    elif _requires_primary_layout(storyboard):
        raise _primary_layout_error("canonical primary-shot layout is missing")
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
        provider_capacity = requirement["provider_capacity"]
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
            "declared_content_beat_count": requirement[
                "declared_content_beat_count"
            ],
            "declared_generation_action_unit_capacity": requirement[
                "declared_generation_action_unit_capacity"
            ],
            "primary_shot_layout_sha256": requirement[
                "primary_shot_layout_sha256"
            ],
            "timeline_layout_binding_sha256": requirement[
                "timeline_layout_binding_sha256"
            ],
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
        generation_action_units = requirement["generation_action_units"]
        generation_unit_buckets = requirement["generation_unit_buckets"]
        timeline_assignment_buckets = requirement[
            "timeline_assignment_buckets"
        ]
        action_buckets = _generation_unit_action_buckets(
            source_actions,
            generation_unit_buckets,
        )
        shot["generation_action_units"] = generation_action_units
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        action_units = list(dict.fromkeys(
            str(value) for value in raw_units if str(value).strip()
        ))
        source_event_buckets = _generation_unit_source_event_buckets(
            [
                event_id
                for event_id in (shot.get("source_events") or [])
                if isinstance(event_id, int)
                and not isinstance(event_id, bool)
                and event_id > 0
            ],
            generation_unit_buckets,
        )
        raw_action_unit_refs = shot.get("source_action_unit_refs") or []
        if not isinstance(raw_action_unit_refs, list):
            raise ValueError("source_action_unit_refs must be an array")
        if (
            storyboard.get("semantic_understanding")
            and action_units
            and not raw_action_unit_refs
        ):
            raise ValueError(
                "canonical action-unit lineage requires source_action_unit_refs"
            )
        source_unit_buckets = _generation_unit_source_action_unit_buckets(
            action_units,
            raw_action_unit_refs,
            source_event_buckets,
            generation_unit_buckets,
        )
        durations = requirement["durations"]
        start_state = _start_state(shot)
        final_state = _end_state(shot)
        beats: list[dict[str, Any]] = []
        for position in range(1, content_count + 1):
            beat_generation_units = generation_unit_buckets[position - 1]
            beat_timeline_assignments = timeline_assignment_buckets[
                position - 1
            ]
            beat_source_event_ids = source_event_buckets[position - 1]
            action = _action_for_bucket(
                action_buckets[position - 1],
                position=position,
                count=content_count,
                fallback_action=fallback_action,
                final_state=final_state,
            )
            canonical_start_state = next((
                str(assignment.get("start_state") or "").strip()
                for assignment in beat_timeline_assignments
                if str(assignment.get("start_state") or "").strip()
            ), "")
            canonical_end_state = next((
                str(assignment.get("end_state") or "").strip()
                for assignment in reversed(beat_timeline_assignments)
                if str(assignment.get("end_state") or "").strip()
            ), "")
            previous_state = canonical_start_state or (
                beats[-1]["end_state"] if beats else start_state
            )
            next_state = canonical_end_state or (
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
            provider_request_duration = requirement[
                "provider_request_durations"
            ][position - 1]
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
                "primary_shot_layout_sha256": requirement[
                    "primary_shot_layout_sha256"
                ],
                "timeline_layout_binding_sha256": requirement[
                    "timeline_layout_binding_sha256"
                ],
                "timeline_assignment_ids": [
                    assignment["assignment_id"]
                    for assignment in beat_timeline_assignments
                ],
                "timeline_assignments": beat_timeline_assignments,
                "motion_contribution_load": sum(
                    int(assignment.get("motion_load") or 0)
                    for assignment in beat_timeline_assignments
                ),
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
                    action,
                    next_state,
                    source_event_ids=beat_source_event_ids,
                    generation_action_units=beat_generation_units,
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
            body_contract = apply_body_action_contract(normalized)
            if body_contract is not None and body_contract.get("valid") is True:
                apply_body_action_kinematics_projection(normalized)
            beats.append(normalized)
        if bridge_required:
            next_shot, next_requirement = planned[index + 1]
            next_sid = next_requirement["shot_id"]
            source_boundary_assignment = next((
                assignment
                for bucket in reversed(timeline_assignment_buckets)
                for assignment in reversed(bucket)
            ), None)
            target_boundary_assignment = next((
                assignment
                for bucket in next_requirement["timeline_assignment_buckets"]
                for assignment in bucket
            ), None)
            bridge_start_state = (
                str(source_boundary_assignment.get("end_state") or "").strip()
                if source_boundary_assignment is not None
                else ""
            ) or final_state
            next_start_state = (
                str(target_boundary_assignment.get("start_state") or "").strip()
                if target_boundary_assignment is not None
                else ""
            ) or _start_state(next_shot)
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
                "start_state": bridge_start_state,
                "action_prompt": _compact(
                    f"保持{sid}结束动作的因果连续，从当前终态平滑过渡到"
                    f"{next_sid}成片起始状态；不得执行{next_sid}的新动作"
                ),
                "end_state": next_start_state,
                "source_timeline_assignment_id": (
                    source_boundary_assignment["assignment_id"]
                    if source_boundary_assignment is not None
                    else None
                ),
                "target_timeline_assignment_id": (
                    target_boundary_assignment["assignment_id"]
                    if target_boundary_assignment is not None
                    else None
                ),
                "timeline_layout_binding_sha256": requirement[
                    "timeline_layout_binding_sha256"
                ],
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
    # Padding efficiency is a Phase 5 policy gate.  Phase 1 must persist the
    # measured ledger so Lifecycle can authorize one bounded screenplay
    # rewrite; every structural/accounting defect remains a local hard error.
    budget_errors = [
        error
        for error in material_budget_contract_errors(storyboard)
        if error.get("code") != PADDING_LOSS_ERROR_CODE
    ]
    if budget_errors:
        summary = "; ".join(error["message"] for error in budget_errors[:5])
        raise AssertionError(f"material budget planner emitted an invalid contract: {summary}")
    return storyboard
