"""Pre-edit material and bridge-overhead accounting.

Primary Sxx shots are editorial containers whose Pxx clips partition the same
story-time budget.  Cross-primary bridges are separately generated assets, so
they are additive for provider cost but replace reserved boundary handles on
the final edit timeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any

MATERIAL_BUDGET_SCHEMA = "honcut.material-budget.v1"
BRIDGE_TIMELINE_POLICY = "replace_boundary_handles"


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


def build_material_budget(storyboard: Mapping[str, Any]) -> dict[str, Any]:
    """Build the two-ledger budget from authored shots and bridge specs."""
    shots = [shot for shot in (storyboard.get("shots") or []) if isinstance(shot, Mapping)]
    bridges = [
        bridge
        for bridge in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(bridge, Mapping)
    ]
    primary_duration = sum(
        _number(shot.get("duration") or shot.get("suggested_duration")) for shot in shots
    )
    bridge_generation_duration = sum(
        _number(bridge.get("generation_duration_s") or bridge.get("duration_s"))
        for bridge in bridges
    )
    bridge_visible_duration = sum(
        _number(
            bridge.get("visible_duration_s")
            or bridge.get("generation_duration_s")
            or bridge.get("duration_s")
        )
        for bridge in bridges
    )
    replaced_handle_duration = sum(
        _number(bridge.get("source_handle_s")) + _number(bridge.get("target_handle_s"))
        for bridge in bridges
        if bridge.get("timeline_insertion_policy") == BRIDGE_TIMELINE_POLICY
    )
    delivery_duration = _number(
        storyboard.get("delivery_target_duration") or storyboard.get("duration")
    )
    ratio_limit = _number(storyboard.get("pre_edit_duration_ratio_limit"), 1.3)
    primary_limit = delivery_duration * ratio_limit if delivery_duration > 0 else None
    projected_timeline = primary_duration - replaced_handle_duration + bridge_visible_duration
    return {
        "schema": MATERIAL_BUDGET_SCHEMA,
        "policy": "primary_ratio_cap_plus_explicit_bridge_overhead",
        "timeline_policy": BRIDGE_TIMELINE_POLICY,
        "delivery_target_duration_s": (
            _rounded(delivery_duration) if delivery_duration > 0 else None
        ),
        "pre_edit_duration_ratio_limit": ratio_limit,
        "primary_material_duration_s": _rounded(primary_duration),
        "primary_material_limit_s": (
            _rounded(primary_limit) if primary_limit is not None else None
        ),
        "primary_material_within_limit": (
            primary_limit is None or primary_duration <= primary_limit + 1e-6
        ),
        "bridge_count": len(bridges),
        "bridge_generation_duration_s": _rounded(bridge_generation_duration),
        "bridge_visible_duration_s": _rounded(bridge_visible_duration),
        "bridge_replaced_handle_duration_s": _rounded(replaced_handle_duration),
        "total_generated_duration_s": _rounded(primary_duration + bridge_generation_duration),
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
    if not budget["primary_material_within_limit"]:
        errors.append(
            {
                "code": "primary_material_ratio_exceeded",
                "message": (
                    f"primary material {budget['primary_material_duration_s']:g}s exceeds "
                    f"the {budget['pre_edit_duration_ratio_limit']:g}x content limit of "
                    f"{budget['primary_material_limit_s']:g}s"
                ),
                "details": budget,
            }
        )
    for bridge in storyboard.get("primary_shot_bridges") or []:
        if not isinstance(bridge, Mapping):
            continue
        if bridge.get("timeline_insertion_policy") != BRIDGE_TIMELINE_POLICY:
            errors.append(
                {
                    "code": "bridge_timeline_policy_invalid",
                    "message": (
                        f"bridge {bridge.get('bridge_id') or '<unknown>'} must replace "
                        "reserved boundary handles"
                    ),
                }
            )
            continue
        visible = _number(
            bridge.get("visible_duration_s")
            or bridge.get("generation_duration_s")
            or bridge.get("duration_s")
        )
        handles = _number(bridge.get("source_handle_s")) + _number(bridge.get("target_handle_s"))
        if visible <= 0 or not math.isclose(handles, visible, abs_tol=1e-6):
            errors.append(
                {
                    "code": "bridge_handle_budget_mismatch",
                    "message": (
                        f"bridge {bridge.get('bridge_id') or '<unknown>'} replaces "
                        f"{handles:g}s of handles but contributes {visible:g}s to the timeline"
                    ),
                    "details": {
                        "visible_duration_s": visible,
                        "replacement_handle_duration_s": handles,
                    },
                }
            )
    stored = storyboard.get("material_budget")
    if str(storyboard.get("secondary_storyboard_version") or "").endswith(".v8") and not isinstance(
        stored, Mapping
    ):
        errors.append(
            {
                "code": "material_budget_ledger_missing",
                "message": "v8 secondary storyboard must persist the material budget ledger",
            }
        )
    if isinstance(stored, Mapping):
        compared = (
            "primary_material_duration_s",
            "bridge_count",
            "bridge_generation_duration_s",
            "total_generated_duration_s",
            "projected_pre_edit_timeline_duration_s",
        )
        mismatches = {
            key: {"stored": stored.get(key), "calculated": budget.get(key)}
            for key in compared
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
