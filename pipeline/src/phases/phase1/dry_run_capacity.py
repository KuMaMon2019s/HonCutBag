"""Zero-request capacity preflight derived from the actual Phase 1 source."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from phases.phase1.adaptation_engine import (
    DEFAULT_SHOT_POLICY,
    _estimate_action_capacity_plan,
)
from prompt.event_extractor import is_global_production_directive_text
from utils.action_units import (
    classify_micro_action,
    event_uses_composite_motion,
    normalize_event_action_units,
)


PHASE1_DRY_RUN_RECEIPT_SCHEMA = "honcut.phase1-dry-run-receipt.v1"
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\d+\s*[.、．]\s*")
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?；;\n]+")
_META_PREAMBLE_RE = re.compile(
    r"(?:以下|下列).{0,24}(?:编号|条目|各项).{0,32}(?:定义|顺序|清单)|"
    r"(?:the\s+following|each\s+numbered).{0,48}(?:order|list|item)",
    re.IGNORECASE,
)
_SCOPED_COMPOSITE_LIST_RE = re.compile(
    r"(?:每(?:条|项|个编号)|各(?:条|项)).{0,36}(?:并发|同时|同一瞬间).{0,20}复合|"
    r"each\s+numbered\s+item.{0,48}(?:concurrent|simultaneous|composite)",
    re.IGNORECASE,
)


def _source_items(text: str, segments: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool]:
    matches = list(_NUMBERED_ITEM_RE.finditer(text))
    if len(matches) >= 2:
        preamble = text[:matches[0].start()].strip()
        scoped_composite = bool(_SCOPED_COMPOSITE_LIST_RE.search(preamble))
        return [
            text[match.end(): matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip()
            for index, match in enumerate(matches)
        ], scoped_composite
    return [
        str(segment.get("content") or "").strip()
        for segment in segments
        if str(segment.get("content") or "").strip()
    ], False


def _preflight_events(
    text: str,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], int]:
    source_items, scoped_composite = _source_items(text, segments)
    events: list[dict[str, Any]] = []
    ignored_directives: list[str] = []
    for source_item in source_items:
        if _META_PREAMBLE_RE.search(source_item):
            continue
        if is_global_production_directive_text(source_item):
            ignored_directives.append(source_item)
            continue
        local_probe = {
            "what": source_item,
            "source_excerpt": source_item,
        }
        source_is_sustained = classify_micro_action(source_item) == "sustained"
        composite = (
            (scoped_composite and not source_is_sustained)
            or event_uses_composite_motion(local_probe)
        )
        action_texts = [source_item] if composite else [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_RE.split(source_item)
            if sentence.strip()
        ]
        for action_text in action_texts:
            category = classify_micro_action(action_text)
            event_id = len(events) + 1
            events.append(
                {
                    "event_id": event_id,
                    "sequence_id": "DRYRUN_SEQ001",
                    "action_unit_id": f"DRYRUN_AU{event_id:03d}",
                    "event_role": (
                        "character_state" if category == "sustained" else "action_chain"
                    ),
                    "what": action_text,
                    "visual": action_text,
                    "source_excerpt": action_text,
                    "micro_actions": [] if category == "sustained" else [action_text],
                    "generation_motion_mode": "composite" if composite else "atomic",
                    "dry_run_source_derived": True,
                }
            )
    return events, ignored_directives, len(source_items)


def build_dry_run_capacity_preflight(
    text: str,
    segments: Sequence[Mapping[str, Any]],
    *,
    duration: int,
    shot_duration: int,
    shot_policy: str = DEFAULT_SHOT_POLICY,
) -> dict[str, Any]:
    """Estimate source-structure pressure without an LLM or Provider call."""
    events, ignored_directives, source_item_count = _preflight_events(text, segments)
    capacity_plan = _estimate_action_capacity_plan(
        events,
        duration,
        shot_duration,
        shot_policy=shot_policy,
    )
    passed = capacity_plan["action_capacity_status"] == "fits_story_clock"
    receipt = {
        "schema": PHASE1_DRY_RUN_RECEIPT_SCHEMA,
        "status": "passed" if passed else "blocked",
        "method": "source_structure_capacity_estimate",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_segment_count": len(segments),
        "source_item_count": source_item_count,
        "source_derived_event_count": len(events),
        "ignored_global_directive_count": len(ignored_directives),
        "capacity_plan": capacity_plan,
        "remote_requests": 0,
        "limitations": (
            "rule-based source-structure estimate; production event extraction remains "
            "authoritative and may report additional semantic pressure"
        ),
    }
    return {
        "events": events,
        "ignored_global_directives": ignored_directives,
        "receipt": receipt,
    }


def partition_preflight_events_by_layout(
    events: Sequence[Mapping[str, Any]],
    action_capacities: Sequence[int],
) -> list[dict[str, Any]]:
    """Partition dry-run events without exceeding canonical per-Sxx capacity.

    This is test-only structural materialization.  It uses the same normalized
    action-unit classifier as the layout solver, preserves source order, and
    never invents or drops a source event.
    """
    if not action_capacities or any(
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity < 1
        for capacity in action_capacities
    ):
        raise ValueError("dry-run layout action capacities are invalid")

    seen: set[str] = set()
    event_contracts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_event in events:
        event = dict(raw_event)
        event_contracts.append(
            (event, normalize_event_action_units(event, seen=seen))
        )
    total_units = sum(contract["units"] for _, contract in event_contracts)
    total_capacity = sum(action_capacities)
    if total_units > total_capacity:
        raise ValueError(
            "dry-run source action units exceed the canonical layout capacity"
        )

    targets: list[int] = []
    remaining_units = total_units
    remaining_capacity = total_capacity
    for index, capacity in enumerate(action_capacities):
        if index == len(action_capacities) - 1:
            target = remaining_units
        else:
            capacity_after = remaining_capacity - capacity
            minimum_here = max(0, remaining_units - capacity_after)
            proportional = round(
                remaining_units * capacity / max(remaining_capacity, 1)
            )
            target = min(
                capacity,
                max(minimum_here, proportional),
            )
        targets.append(target)
        remaining_units -= target
        remaining_capacity -= capacity
    if remaining_units != 0:
        raise ValueError("dry-run layout action targets are inconsistent")

    groups = [
        {"events": [], "action_contracts": [], "action_unit_count": 0}
        for _ in action_capacities
    ]
    shot_index = 0
    for event, contract in event_contracts:
        event_units = int(contract["units"])
        if (
            shot_index < len(groups) - 1
            and groups[shot_index]["events"]
            and groups[shot_index]["action_unit_count"] >= targets[shot_index]
            and event_units > 0
        ):
            shot_index += 1
        group = groups[shot_index]
        group["events"].append(event)
        group["action_contracts"].append(contract)
        group["action_unit_count"] += event_units

    for index, (group, capacity) in enumerate(
        zip(groups, action_capacities, strict=True),
        1,
    ):
        if group["action_unit_count"] > capacity:
            raise ValueError(
                f"dry-run S{index:02d} exceeds canonical action capacity"
            )
    return groups


def write_dry_run_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(dict(receipt), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
