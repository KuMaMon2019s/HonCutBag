"""Story-clock and additive bridge-generation accounting.

Primary Sxx shots and their Pxx children describe the same story clock and
must never be added together. Cross-primary bridges are separate generated
assets: they add provider cost, while the current edit policy replaces equal
boundary handles and therefore does not extend the projected timeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

MATERIAL_BUDGET_SCHEMA = "honcut.material-budget.v2"
BRIDGE_TIMELINE_POLICY = "replace_boundary_handles"
DEFAULT_GENERATED_DURATION_RATIO_REFERENCE = 1.3
DEFAULT_DELIVERY_PACING_SPEED_RANGE = (0.85, 1.25)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _duration_range(bridge: Mapping[str, Any]) -> tuple[float, float]:
    selected = _number(
        bridge.get("generation_duration_s") or bridge.get("duration_s")
    )
    raw = bridge.get("generation_duration_range_s")
    if (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) == 2
    ):
        minimum = _number(raw[0], selected)
        maximum = _number(raw[1], selected)
        if 0 < minimum <= maximum:
            return minimum, maximum
    return selected, selected


def build_material_budget(storyboard: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical two-ledger budget from storyboard assets."""
    shots = [
        shot
        for shot in (storyboard.get("shots") or [])
        if isinstance(shot, Mapping)
    ]
    bridges = [
        bridge
        for bridge in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(bridge, Mapping)
    ]
    primary_duration = sum(
        _number(shot.get("duration") or shot.get("suggested_duration"))
        for shot in shots
    )
    secondary_duration = 0.0
    secondary_declared = False
    for shot in shots:
        beats = [
            beat
            for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, Mapping)
        ]
        if beats:
            secondary_declared = True
            secondary_duration += sum(
                _number(beat.get("duration_s") or beat.get("duration"))
                for beat in beats
            )
    secondary_matches = (
        not secondary_declared
        or math.isclose(primary_duration, secondary_duration, abs_tol=1e-6)
    )
    story_clock_duration = primary_duration
    bridge_generation_duration = sum(
        _number(bridge.get("generation_duration_s") or bridge.get("duration_s"))
        for bridge in bridges
    )
    bridge_ranges = [_duration_range(bridge) for bridge in bridges]
    bridge_generation_minimum = sum(item[0] for item in bridge_ranges)
    bridge_generation_maximum = sum(item[1] for item in bridge_ranges)
    bridge_visible_duration = sum(
        _number(
            bridge.get("visible_duration_s")
            or bridge.get("generation_duration_s")
            or bridge.get("duration_s")
        )
        for bridge in bridges
    )
    replaced_handle_duration = sum(
        _number(bridge.get("source_handle_s"))
        + _number(bridge.get("target_handle_s"))
        for bridge in bridges
        if bridge.get("timeline_insertion_policy") == BRIDGE_TIMELINE_POLICY
    )
    delivery_duration = _number(
        storyboard.get("delivery_target_duration") or storyboard.get("duration")
    )
    ratio_reference = _number(
        storyboard.get("generated_duration_ratio_reference"),
        DEFAULT_GENERATED_DURATION_RATIO_REFERENCE,
    )
    raw_pacing_range = storyboard.get("delivery_pacing_speed_range")
    if (
        isinstance(raw_pacing_range, Sequence)
        and not isinstance(raw_pacing_range, (str, bytes))
        and len(raw_pacing_range) == 2
    ):
        pacing_minimum = _number(
            raw_pacing_range[0], DEFAULT_DELIVERY_PACING_SPEED_RANGE[0]
        )
        pacing_maximum = _number(
            raw_pacing_range[1], DEFAULT_DELIVERY_PACING_SPEED_RANGE[1]
        )
        if not 0 < pacing_minimum <= pacing_maximum:
            pacing_minimum, pacing_maximum = DEFAULT_DELIVERY_PACING_SPEED_RANGE
    else:
        pacing_minimum, pacing_maximum = DEFAULT_DELIVERY_PACING_SPEED_RANGE
    storyboard_limit = delivery_duration if delivery_duration > 0 else None
    total_generated_duration = story_clock_duration + bridge_generation_duration
    total_generated_range = (
        story_clock_duration + bridge_generation_minimum,
        story_clock_duration + bridge_generation_maximum,
    )
    projected_timeline = (
        story_clock_duration - replaced_handle_duration + bridge_visible_duration
    )
    if delivery_duration > 0:
        generated_ratio = total_generated_duration / delivery_duration
        generated_ratio_range = [
            total_generated_range[0] / delivery_duration,
            total_generated_range[1] / delivery_duration,
        ]
    else:
        generated_ratio = None
        generated_ratio_range = None
    return {
        "schema": MATERIAL_BUDGET_SCHEMA,
        "policy": "story_clock_cap_plus_explicit_bridge_overhead",
        "timeline_policy": BRIDGE_TIMELINE_POLICY,
        "delivery_target_duration_s": (
            _rounded(delivery_duration) if delivery_duration > 0 else None
        ),
        "storyboard_duration_limit_s": (
            _rounded(storyboard_limit) if storyboard_limit is not None else None
        ),
        "primary_story_duration_s": _rounded(primary_duration),
        "secondary_story_duration_s": (
            _rounded(secondary_duration) if secondary_declared else None
        ),
        "primary_secondary_duration_match": secondary_matches,
        "story_clock_duration_s": _rounded(story_clock_duration),
        "story_clock_within_delivery_target": (
            storyboard_limit is None
            or story_clock_duration <= storyboard_limit + 1e-6
        ),
        "bridge_count": len(bridges),
        "bridge_generation_duration_s": _rounded(bridge_generation_duration),
        "bridge_generation_duration_range_s": [
            _rounded(bridge_generation_minimum),
            _rounded(bridge_generation_maximum),
        ],
        "bridge_visible_duration_s": _rounded(bridge_visible_duration),
        "bridge_replaced_handle_duration_s": _rounded(replaced_handle_duration),
        "total_generated_duration_s": _rounded(total_generated_duration),
        "total_generated_duration_range_s": [
            _rounded(total_generated_range[0]),
            _rounded(total_generated_range[1]),
        ],
        "total_generated_duration_ratio": (
            _rounded(generated_ratio) if generated_ratio is not None else None
        ),
        "total_generated_duration_ratio_range": (
            [_rounded(value) for value in generated_ratio_range]
            if generated_ratio_range is not None
            else None
        ),
        "generated_duration_ratio_reference": ratio_reference,
        "generated_duration_ratio_is_hard_limit": False,
        "delivery_pacing_speed_range": [
            _rounded(pacing_minimum),
            _rounded(pacing_maximum),
        ],
        "total_generated_covers_delivery_target": (
            storyboard_limit is None
            or total_generated_duration + 1e-6 >= storyboard_limit
        ),
        "projected_pre_edit_timeline_duration_s": _rounded(projected_timeline),
        "bridge_overhead_is_additive": True,
        "primary_secondary_double_count_forbidden": True,
    }


def attach_material_budget(
    storyboard: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Persist the current two-ledger calculation on a storyboard."""
    budget = build_material_budget(storyboard)
    storyboard["material_budget"] = budget
    return budget


def material_budget_contract_errors(
    storyboard: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return actionable accounting errors without trusting a stored ledger."""
    budget = build_material_budget(storyboard)
    errors: list[dict[str, Any]] = []
    if not budget["story_clock_within_delivery_target"]:
        errors.append(
            {
                "code": "storyboard_duration_exceeds_delivery_target",
                "message": (
                    f"story clock {budget['story_clock_duration_s']:g}s exceeds "
                    f"the {budget['storyboard_duration_limit_s']:g}s delivery target"
                ),
                "details": budget,
            }
        )
    if not budget["primary_secondary_duration_match"]:
        errors.append(
            {
                "code": "primary_secondary_duration_mismatch",
                "message": (
                    "secondary Pxx story time must partition, not extend, its "
                    "primary Sxx story clock"
                ),
                "details": budget,
            }
        )
    secondary_version = str(storyboard.get("secondary_storyboard_version") or "")
    if (
        secondary_version.startswith("honcut.secondary-storyboard.")
        and not budget["total_generated_covers_delivery_target"]
    ):
        errors.append(
            {
                "code": "total_generated_duration_below_delivery_target",
                "message": (
                    f"story plus bridge generation totals "
                    f"{budget['total_generated_duration_s']:g}s, below the "
                    f"{budget['delivery_target_duration_s']:g}s delivery target"
                ),
                "details": budget,
            }
        )
    for bridge in storyboard.get("primary_shot_bridges") or []:
        if not isinstance(bridge, Mapping):
            continue
        bridge_id = bridge.get("bridge_id") or "<unknown>"
        if bridge.get("timeline_insertion_policy") != BRIDGE_TIMELINE_POLICY:
            errors.append(
                {
                    "code": "bridge_timeline_policy_invalid",
                    "message": (
                        f"bridge {bridge_id} must replace reserved boundary handles"
                    ),
                }
            )
            continue
        visible = _number(
            bridge.get("visible_duration_s")
            or bridge.get("generation_duration_s")
            or bridge.get("duration_s")
        )
        handles = _number(bridge.get("source_handle_s")) + _number(
            bridge.get("target_handle_s")
        )
        if visible <= 0 or not math.isclose(handles, visible, abs_tol=1e-6):
            errors.append(
                {
                    "code": "bridge_handle_budget_mismatch",
                    "message": (
                        f"bridge {bridge_id} replaces {handles:g}s of handles but "
                        f"contributes {visible:g}s to the timeline"
                    ),
                    "details": {
                        "visible_duration_s": visible,
                        "replacement_handle_duration_s": handles,
                    },
                }
            )
        selected = _number(
            bridge.get("generation_duration_s") or bridge.get("duration_s")
        )
        raw_range = bridge.get("generation_duration_range_s")
        if raw_range is not None and not (
            isinstance(raw_range, Sequence)
            and not isinstance(raw_range, (str, bytes))
            and len(raw_range) == 2
            and 0 < _number(raw_range[0]) <= _number(raw_range[1])
        ):
            errors.append(
                {
                    "code": "bridge_generation_duration_range_invalid",
                    "message": (
                        f"bridge {bridge_id} has an invalid generation duration range"
                    ),
                }
            )
            continue
        minimum, maximum = _duration_range(bridge)
        if not minimum - 1e-6 <= selected <= maximum + 1e-6:
            errors.append(
                {
                    "code": "bridge_generation_duration_outside_range",
                    "message": (
                        f"bridge {bridge_id} duration {selected:g}s is outside its "
                        f"declared {minimum:g}-{maximum:g}s planning range"
                    ),
                }
            )
    stored = storyboard.get("material_budget")
    if secondary_version.startswith("honcut.secondary-storyboard.") and not isinstance(
        stored, Mapping
    ):
        errors.append(
            {
                "code": "material_budget_ledger_missing",
                "message": "secondary storyboard must persist the material budget ledger",
            }
        )
    if isinstance(stored, Mapping):
        if stored.get("schema") != MATERIAL_BUDGET_SCHEMA:
            errors.append(
                {
                    "code": "material_budget_schema_unsupported",
                    "message": (
                        f"stored material budget schema {stored.get('schema')!r} is not "
                        f"the supported {MATERIAL_BUDGET_SCHEMA}"
                    ),
                }
            )
        mismatches = {
            key: {"stored": stored.get(key), "calculated": budget.get(key)}
            for key in sorted(set(stored) | set(budget))
            if stored.get(key) != budget.get(key)
        }
        if mismatches:
            errors.append(
                {
                    "code": "material_budget_ledger_stale",
                    "message": "stored material budget does not match storyboard assets",
                    "details": mismatches,
                }
            )
    return errors
