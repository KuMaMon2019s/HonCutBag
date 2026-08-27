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
from utils.action_units import classify_micro_action, event_uses_composite_motion


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
