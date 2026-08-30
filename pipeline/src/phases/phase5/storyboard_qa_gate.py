"""Phase 5: pre-generation storyboard quality gate.

Every judgment is derived from project artifacts.  The pure gate reports the
shots that need to be redrawn; the Phase 5 correction wrapper may then perform
a bounded, auditable redraw-and-recheck loop before Phase 6.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clients.ark_multimodal_client import ArkMultimodalClient
from phases.phase1.adaptation_engine import (
    DURATION_SCALED_EVENT_PLAN_SCHEMA,
    SCREENPLAY_PLAN_SCHEMA,
    migrate_screenplay_plan,
    terminal_outcome_event_ids,
)
from phases.phase1.storyboard_beats import (
    secondary_contract_declared,
    secondary_storyboard_contract_errors,
    secondary_storyboard_requirements,
)
from runtime.security_boundaries import redact_for_log, redact_text
from runtime.structured_understanding import (
    DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS,
    StructuredUnderstandingExhausted,
    execute_structured_understanding,
)
from utils.action_units import normalize_action_units
from utils.body_action_contracts import body_action_contract_errors
from utils.video_capabilities import capabilities_for
from utils.character_body_contracts import character_visual_description
from utils.camera_motion_contracts import apply_camera_motion_contract
from utils.material_budget import material_budget_contract_errors
from tools.character_reference_board import resolve_character_reference_board

DEFAULT_SIMILARITY_THRESHOLD = 0.65
DEFAULT_MAX_CORRECTION_ATTEMPTS = 2
MAX_CORRECTION_ATTEMPTS = 3
MAX_REVIEW_ADJUDICATIONS = 2
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PHASE5_DRY_RUN_RECEIPT_SCHEMA = "honcut.phase5-dry-run-receipt.v1"
PHASE5_DRY_RUN_RECEIPT_NAME = "phase5_dry_run_receipt.json"
PHASE5_REVIEW_ADJUDICATION_SCHEMA = "honcut.phase5-review-adjudication.v1"
PHASE5_REVIEW_ADJUDICATIONS_SCHEMA = "honcut.phase5-review-adjudications.v1"
PHASE5_REVIEW_ADJUDICATION_REPORT_NAME = (
    "phase5_review_adjudication_report.json"
)
PHASE5_REVIEW_ADJUDICATION_PREVIOUS_REPORT_NAME = (
    "phase5_review_adjudication_previous_report.json"
)
PHASE5_REVIEW_ADJUDICATION_CURRENT_REPORT_NAME = (
    "phase5_review_adjudication_current_report.json"
)
PHASE5_DRY_RUN_SKIPPED_OPERATIONS = (
    "storyboard_pixel_artifact_validation",
    "embedding_review",
    "multimodal_storyboard_review",
    "cinematic_first_frame_review",
    "automatic_image_correction",
    "independent_llm_supervision",
)


class QAObservationGatePaused(RuntimeError):
    """Stop a paid acceptance run after one durable new QA observation."""

    def __init__(
        self,
        *,
        limit: int,
        observation_id: str,
        decision_id: str,
        verdict: str,
        provider_request_count: int,
    ) -> None:
        self.limit = limit
        self.observation_id = observation_id
        self.decision_id = decision_id
        self.verdict = verdict
        self.provider_request_count = provider_request_count
        super().__init__(
            "Phase 5 acceptance QA observation gate reached: "
            f"{observation_id} ({verdict})"
        )

_LIGHT_PERIODS = {
    "night": ("night", "midnight", "moonlight", "夜", "午夜", "月光", "星空"),
    "day": ("daylight", "midday", "noon", "daytime", "日间", "白天", "正午", "午后"),
    "dawn": ("dawn", "sunrise", "morning", "黎明", "清晨", "日出", "晨光"),
    "dusk": ("dusk", "sunset", "evening", "golden hour", "夕阳", "黄昏", "傍晚", "日落", "黄金时段"),
}


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id", shot.get("id", index + 1))
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _periods(text: str) -> set[str]:
    lowered = text.lower()
    return {period for period, terms in _LIGHT_PERIODS.items() if any(term in lowered for term in terms)}


def _issue(layer: str, severity: str, code: str, message: str, shot_ids: list[str] | None = None, **details: Any) -> dict:
    result = {"layer": layer, "severity": severity, "code": code, "message": message, "shot_ids": shot_ids or []}
    if details:
        result["details"] = details
    return result


def run_l1_checks(storyboard: dict, visual_style: str) -> tuple[list[dict], dict[str, dict]]:
    """Check artifact-level lighting, spoken-text fields, and duration."""
    shots = storyboard.get("shots") if isinstance(storyboard.get("shots"), list) else []
    issues: list[dict] = []
    per_shot: dict[str, dict] = {}
    style_periods = _periods(visual_style)
    durations: list[float] = []
    dialogue_fields = ("dialogue", "narration", "voiceover", "voice_over")

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        # Free metadata normalization at the last pre-video QA boundary keeps
        # legacy storyboards inside the same physical camera/lens contract.
        apply_camera_motion_contract(shot)
        sid = _shot_id(shot, index)
        per_shot[sid] = {"issues": [], "characters": []}
        character_assets = [
            str(value)
            for value in (shot.get("associate_assets") or [])
            if str(value).startswith("char:")
        ]
        if shot.get("who") == [] and character_assets:
            item = _issue(
                "L1",
                "severe",
                "no_character_contract_conflict",
                f"{sid} declares who=[] but binds character assets",
                [sid],
                character_assets=character_assets,
            )
            issues.append(item)
            per_shot[sid]["issues"].append(item)
        lighting = " ".join(_text(shot.get(key)) for key in ("lighting_description", "lighting", "prompt", "description", "name"))
        shot_periods = _periods(lighting)
        if style_periods and shot_periods and style_periods.isdisjoint(shot_periods):
            item = _issue("L1", "severe", "lighting_period_mismatch", f"{sid} lighting period conflicts with visual-style.md", [sid], expected=sorted(style_periods), observed=sorted(shot_periods))
            issues.append(item)
            per_shot[sid]["issues"].append(item)

        duration = shot.get("duration", shot.get("duration_seconds"))
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            item = _issue("L1", "moderate", "invalid_duration", f"{sid} has no positive numeric duration", [sid])
            issues.append(item)
            per_shot[sid]["issues"].append(item)
        else:
            durations.append(float(duration))

        present = [
            key
            for key in dialogue_fields
            if key in shot and shot.get(key) is not None
        ]
        if present and all(not _text(shot.get(key)).strip() for key in present):
            item = _issue("L1", "moderate", "empty_spoken_content", f"{sid} declares spoken-content fields but all are empty", [sid])
            issues.append(item)
            per_shot[sid]["issues"].append(item)

    target = storyboard.get("target_duration", storyboard.get("duration"))
    if isinstance(target, (int, float)) and target > 0 and shots:
        actual = sum(durations)
        tolerance = max(1.0, float(target) * 0.05)
        if abs(actual - float(target)) > tolerance:
            issues.append(_issue("L1", "moderate", "duration_budget_mismatch", f"Storyboard duration {actual:g}s differs from target {float(target):g}s", details={"actual_seconds": actual, "target_seconds": float(target), "tolerance_seconds": tolerance}))
    for budget_error in material_budget_contract_errors(storyboard):
        issues.append(
            _issue(
                "L1",
                "severe",
                budget_error["code"],
                budget_error["message"],
                **(budget_error.get("details") or {}),
            )
        )
    return issues, per_shot


def run_generation_capacity_checks(
    storyboard: dict,
    events_data: dict | None = None,
    screenplay_plan: dict | None = None,
) -> list[dict]:
    """Block storyboards that exceed one video clip's narrative capacity."""
    issues: list[dict] = []
    observed_units: set[str] = set()
    uses_strict_secondary_contract = secondary_contract_declared(storyboard)
    ordered_shots = [
        shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)
    ]
    shot_ids = [_shot_id(shot, index) for index, shot in enumerate(ordered_shots)]
    for index, shot in enumerate(ordered_shots):
        if not isinstance(shot, dict):
            continue
        profile = capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index)
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        units = {str(value) for value in raw_units if str(value).strip()}
        observed_units.update(units)
        source_micro_actions = shot.get("micro_actions") or []
        if isinstance(source_micro_actions, str):
            source_micro_actions = [source_micro_actions]
        source_micro_actions = [
            str(value) for value in source_micro_actions if str(value).strip()
        ]
        if "generation_action_units" in shot:
            shot_generation_units = [
                unit
                for unit in (shot.get("generation_action_units") or [])
                if isinstance(unit, dict)
            ]
        else:
            shot_generation_units = normalize_action_units(source_micro_actions)[
                "generation_action_units"
            ]
        generation_actions = shot.get("generation_actions") or []
        if isinstance(generation_actions, str):
            generation_actions = [generation_actions]
        has_generation_action = bool(shot_generation_units or generation_actions)
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 0)
        camera = str(
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or ""
        ).lower()
        storyboard_beats = [
            beat
            for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]

        for contract_error in body_action_contract_errors(shot):
            issues.append(_issue(
                "L1",
                "severe",
                str(contract_error.get("code") or "body_choreography_invalid"),
                f"{sid} {contract_error.get('message') or 'has an invalid body choreography contract'}",
                [sid],
                actions=contract_error.get("actions") or [],
            ))

        if uses_strict_secondary_contract:
            for contract_error in secondary_storyboard_contract_errors(
                storyboard,
                index,
                profile,
            ):
                issues.append(_issue(
                    "L1",
                    "severe",
                    str(contract_error["code"]),
                    str(contract_error["message"]),
                    [sid],
                    **(contract_error.get("details") or {}),
                ))

        if storyboard_beats:
            beat_duration_total = 0.0
            beat_units_seen: list[str] = []
            beat_micro_actions_seen: list[str] = []
            uses_secondary_contract = uses_strict_secondary_contract or all(
                str(beat.get("generation_mode") or "").strip().lower()
                in {
                    "multi_image",
                    "tail_video_extend",
                    "first_last_frame_bridge",
                }
                for beat in storyboard_beats
            )
            expected_modes: list[str] = []
            bridge_required = False
            allowed_content_beats = 3
            if uses_secondary_contract:
                try:
                    requirement = secondary_storyboard_requirements(
                        storyboard,
                        index,
                        profile,
                    )
                except (TypeError, ValueError):
                    requirement = None
                if requirement is not None:
                    # Compare Pxx output with the same canonicalized source
                    # actions used by the Phase 1 secondary-storyboard owner.
                    # Raw authored punctuation (for example a trailing Chinese
                    # colon) is deliberately removed by that owner and must not
                    # be reported here as a plot-order mutation.
                    source_micro_actions = list(requirement["source_actions"])
                    bridge_required = requirement["bridge_required"]
                    expected_modes = list(requirement["modes"])
                    allowed_content_beats = int(
                        requirement["max_content_beats"]
                    )
                actual_modes = [
                    str(beat.get("generation_mode") or "").strip().lower()
                    for beat in storyboard_beats
                ]
                if actual_modes != expected_modes:
                    issues.append(_issue(
                        "L1", "severe", "secondary_storyboard_strategy_mismatch",
                        f"{sid} secondary beats must follow content capacity and the "
                        "next primary-shot boundary",
                        [sid], expected_modes=expected_modes, observed_modes=actual_modes,
                        bridge_required=bridge_required,
                    ))
            if (
                uses_secondary_contract
                and not 1 <= len(storyboard_beats) <= allowed_content_beats
            ):
                issues.append(_issue(
                    "L1", "severe", "secondary_storyboard_count_invalid",
                    f"{sid} must contain one to {allowed_content_beats} "
                    "capacity-selected Pxx beats",
                    [sid], observed_count=len(storyboard_beats),
                    allowed_count=allowed_content_beats,
                ))
            for position, beat in enumerate(storyboard_beats, 1):
                beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
                beat_duration = float(beat.get("duration_s") or 0)
                beat_duration_total += beat_duration
                actual_mode = str(
                    beat.get("generation_mode") or ""
                ).strip().lower()
                if uses_secondary_contract:
                    expected_mode = (
                        expected_modes[position - 1]
                        if position <= len(expected_modes)
                        else "<no-extra-beat>"
                    )
                else:
                    expected_mode = (
                        "extend"
                        if position > 1
                        or (
                            position == 1
                            and str(shot.get("boundary_before") or "")
                            .strip()
                            .lower()
                            == "continuous"
                        )
                        else "fresh"
                    )
                if actual_mode != expected_mode:
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_mode_invalid",
                        f"{beat_id} must use {expected_mode}",
                        [sid], beat_id=beat_id, expected_mode=expected_mode,
                    ))
                if not str(beat.get("action") or "").strip():
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_action_missing",
                        f"{beat_id} has no executable action contract",
                        [sid], beat_id=beat_id,
                    ))
                for contract_error in body_action_contract_errors({**shot, **beat}):
                    issues.append(_issue(
                        "L1",
                        "severe",
                        str(contract_error.get("code") or "body_choreography_invalid"),
                        f"{beat_id} {contract_error.get('message') or 'has an invalid body choreography contract'}",
                        [sid],
                        beat_id=beat_id,
                        actions=contract_error.get("actions") or [],
                    ))
                beat_units = beat.get("source_action_unit_ids") or []
                if isinstance(beat_units, str):
                    beat_units = [beat_units]
                beat_units = [str(value) for value in beat_units if str(value).strip()]
                beat_units_seen.extend(beat_units)
                micro_actions = beat.get("micro_actions") or []
                if isinstance(micro_actions, str):
                    micro_actions = [micro_actions]
                beat_micro_actions_seen.extend(
                    str(value) for value in micro_actions if str(value).strip()
                )
                if "generation_action_units" in beat:
                    beat_generation_units = [
                        unit
                        for unit in (beat.get("generation_action_units") or [])
                        if isinstance(unit, dict)
                    ]
                else:
                    beat_generation_units = normalize_action_units(
                        [str(value) for value in micro_actions if str(value).strip()]
                    )["generation_action_units"]
                if len(beat_generation_units) > profile.temporal_slice_limit:
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_generation_action_overload",
                        f"{beat_id} exceeds {profile.name}'s normalized generation-action "
                        "capacity "
                        f"({profile.temporal_slice_limit})",
                        [sid], beat_id=beat_id,
                        generation_action_units=len(beat_generation_units),
                    ))
                if uses_secondary_contract:
                    if str(beat.get("parent_shot_id") or "") != sid:
                        issues.append(_issue(
                            "L1", "severe", "secondary_storyboard_parent_mismatch",
                            f"{beat_id} must remain owned by primary shot {sid}",
                            [sid], beat_id=beat_id,
                        ))
                    if beat.get("plot_fidelity_contract") != (
                        "primary_shot_source_only_no_invention"
                    ):
                        issues.append(_issue(
                            "L1", "severe", "secondary_storyboard_fidelity_missing",
                            f"{beat_id} has no primary-shot plot fidelity contract",
                            [sid], beat_id=beat_id,
                        ))
                    if actual_mode == "first_last_frame_bridge":
                        expected_target_shot = (
                            shot_ids[index + 1] if index + 1 < len(shot_ids) else None
                        )
                        expected_target_beat = (
                            f"{expected_target_shot}_P01" if expected_target_shot else None
                        )
                        if (
                            not bridge_required
                            or position != len(storyboard_beats)
                            or beat.get("bridge_target_shot_id") != expected_target_shot
                            or beat.get("bridge_target_beat_id") != expected_target_beat
                            or not str(
                                beat.get("bridge_target_storyboard_image") or ""
                            ).strip()
                        ):
                            issues.append(_issue(
                                "L1", "severe", "secondary_storyboard_bridge_invalid",
                                f"{beat_id} must end on {expected_target_beat}",
                                [sid], beat_id=beat_id,
                                expected_target_beat_id=expected_target_beat,
                            ))
                duration_minimum, duration_maximum = (
                    profile.effective_duration_bounds(actual_mode)
                )
                if not duration_minimum <= beat_duration <= duration_maximum:
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_duration_invalid",
                        f"{beat_id} lasts {beat_duration:g}s; expected "
                        f"{duration_minimum:g}-{duration_maximum:g}s for "
                        f"{profile.name} {actual_mode}",
                        [sid], beat_id=beat_id, duration_seconds=beat_duration,
                    ))
                if uses_secondary_contract:
                    try:
                        expected_request_duration = (
                            profile.request_duration_for_effective_story(
                                beat_duration,
                                actual_mode,
                            )
                        )
                    except ValueError as exc:
                        issues.append(_issue(
                            "L1", "severe", "storyboard_beat_story_duration_invalid",
                            f"{beat_id} has no executable Provider request: {exc}",
                            [sid], beat_id=beat_id,
                        ))
                    else:
                        observed_request_duration = float(
                            beat.get("provider_request_duration_s") or 0
                        )
                        if not math.isclose(
                            observed_request_duration,
                            expected_request_duration,
                            abs_tol=1e-6,
                        ):
                            issues.append(_issue(
                                "L1", "severe", "storyboard_beat_provider_request_mismatch",
                                f"{beat_id} Provider request duration must be "
                                f"{expected_request_duration:g}s for {beat_duration:g}s "
                                "of effective story time",
                                [sid], beat_id=beat_id,
                                effective_story_duration_s=beat_duration,
                                expected_provider_request_duration_s=expected_request_duration,
                                observed_provider_request_duration_s=observed_request_duration,
                            ))
            if duration and not math.isclose(
                beat_duration_total,
                duration,
                abs_tol=0.05,
            ):
                issues.append(_issue(
                    "L1", "severe", "storyboard_beat_duration_mismatch",
                    f"{sid} internal beats total {beat_duration_total:g}s, "
                    f"expected {duration:g}s",
                    [sid], beat_duration_seconds=beat_duration_total,
                    shot_duration_seconds=duration,
                ))
            if units and set(beat_units_seen) != units:
                issues.append(_issue(
                    "L1", "severe", "storyboard_beat_action_unit_coverage_mismatch",
                    f"{sid} Pxx action-unit coverage differs from the director shot",
                    [sid], expected_action_unit_ids=sorted(units),
                    observed_action_unit_ids=sorted(set(beat_units_seen)),
                ))
            if (
                uses_secondary_contract
                and source_micro_actions
                and source_micro_actions != beat_micro_actions_seen
            ):
                issues.append(_issue(
                    "L1", "severe", "secondary_storyboard_action_order_mismatch",
                    f"{sid} Pxx actions must preserve every primary-shot action in order",
                    [sid], expected_actions=source_micro_actions,
                    observed_actions=beat_micro_actions_seen,
                ))

        if (
            len(shot_generation_units) > profile.temporal_slice_limit
            and not storyboard_beats
        ):
            issues.append(_issue(
                "L1", "severe", "generation_action_unit_overload",
                f"{sid} contains {len(shot_generation_units)} normalized generation "
                f"action units; {profile.name} supports "
                f"{profile.temporal_slice_limit} per beat",
                [sid], generation_action_units=len(shot_generation_units),
            ))
        if has_generation_action and not generation_actions and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "missing_generation_actions",
                f"{sid} has screenplay action but no bounded generation action contract",
                [sid],
            ))
        action_limit = profile.action_limit(duration)
        if len(generation_actions) > action_limit and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "generation_action_overload",
                f"{sid} asks the video model to perform {len(generation_actions)} actions "
                f"(max {action_limit} for {duration:g}s)",
                [sid], prompted_actions=len(generation_actions), action_limit=action_limit,
            ))
        if has_generation_action and duration > profile.max_unique_beat_s and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "action_shot_too_long",
                f"{sid} action shot lasts {duration:g}s; split within "
                f"{profile.name}'s {profile.min_unique_beat_s:g}-"
                f"{profile.max_unique_beat_s:g}s beat range",
                [sid], duration_seconds=duration,
            ))
        if has_generation_action and camera in {"static", "fixed", "locked", "unspecified", ""} and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "static_action_camera",
                f"{sid} action shot uses a locked/static camera contract",
                [sid], camera_movement=camera or "missing",
            ))
        if (
            not has_generation_action
            and duration > profile.max_unique_beat_s
            and camera in {"static", "fixed", "locked", "unspecified", ""}
            and not storyboard_beats
        ):
            issues.append(_issue(
                "L1", "severe", "static_hold_risk",
                f"{sid} holds a static composition for {duration:g}s",
                [sid], duration_seconds=duration,
            ))

    source_events = [
        event
        for event in (events_data or {}).get("events", [])
        if isinstance(event, dict)
    ]
    production_event_ids: set[int] | None = None
    if screenplay_plan is not None:
        plan_schema = str(screenplay_plan.get("schema") or "").strip()
        if plan_schema == "honcut.screenplay-plan.v1":
            production_event_ids = None
        elif plan_schema != SCREENPLAY_PLAN_SCHEMA:
            issues.append(_issue(
                "L1",
                "severe",
                "screenplay_plan_lineage_invalid",
                f"Unsupported screenplay plan schema: {plan_schema or '<missing>'}",
            ))
            production_event_ids = set()
        else:
            scaling = screenplay_plan.get("event_action_scaling")
            records = scaling.get("events") if isinstance(scaling, dict) else None
            production_ledger = screenplay_plan.get("production_ledger")
            record_ids = [
                record.get("source_event_id")
                for record in records or []
                if isinstance(record, dict)
            ]
            statuses = {
                str(record.get("production_status") or "")
                for record in records or []
                if isinstance(record, dict)
            }
            record_mandatory_ids = {
                record.get("source_event_id")
                for record in records or []
                if isinstance(record, dict) and record.get("mandatory") is True
            }
            record_kept_ids = {
                record.get("source_event_id")
                for record in records or []
                if isinstance(record, dict)
                and record.get("production_status") == "kept"
            }
            base_mandatory_ids = (
                production_ledger.get("base_mandatory_source_event_ids")
                if isinstance(production_ledger, dict)
                else None
            )
            mandatory_ids = (
                production_ledger.get("mandatory_source_event_ids")
                if isinstance(production_ledger, dict)
                else None
            )
            causal_predecessor_ids = (
                production_ledger.get("causal_predecessor_source_event_ids")
                if isinstance(production_ledger, dict)
                else None
            )
            terminal_outcome_ids = (
                production_ledger.get("terminal_outcome_source_event_ids")
                if isinstance(production_ledger, dict)
                else None
            )
            plan_beats = screenplay_plan.get("beats")
            projected_beats = [
                beat
                for beat in plan_beats or []
                if isinstance(beat, dict) and beat.get("director_intent") is not None
            ]
            valid_director_projections = (
                isinstance(plan_beats, list)
                and all(
                    isinstance(beat.get("director_intent"), dict)
                    and beat["director_intent"].get("schema")
                    == "honcut.production-director-intent.v1"
                    and beat["director_intent"].get("source_event_ids")
                    == beat.get("source_refs")
                    and beat["director_intent"].get("sequence_id")
                    == beat.get("sequence_id")
                    for beat in projected_beats
                )
                and (
                    not projected_beats
                    or (
                        isinstance(production_ledger, dict)
                        and production_ledger.get(
                            "production_director_intent_schema"
                        )
                        == "honcut.production-director-intent.v1"
                    )
                )
            )
            valid_mandatory_lineage = (
                isinstance(base_mandatory_ids, list)
                and isinstance(mandatory_ids, list)
                and isinstance(causal_predecessor_ids, list)
                and isinstance(terminal_outcome_ids, list)
                and all(isinstance(value, int) for value in mandatory_ids)
                and all(isinstance(value, int) for value in base_mandatory_ids)
                and all(isinstance(value, int) for value in causal_predecessor_ids)
                and all(isinstance(value, int) for value in terminal_outcome_ids)
                and set(base_mandatory_ids).isdisjoint(causal_predecessor_ids)
                and set(terminal_outcome_ids)
                == terminal_outcome_event_ids(source_events)
                and set(terminal_outcome_ids) <= set(base_mandatory_ids)
                and set(mandatory_ids)
                == set(base_mandatory_ids) | set(causal_predecessor_ids)
                and record_mandatory_ids == set(mandatory_ids)
                and set(mandatory_ids) <= record_kept_ids
            )
            if (
                not isinstance(scaling, dict)
                or scaling.get("schema") != DURATION_SCALED_EVENT_PLAN_SCHEMA
                or not isinstance(records, list)
                or record_ids != list(range(1, len(source_events) + 1))
                or not statuses <= {"kept", "whole_event_omitted"}
                or not statuses
                or not valid_mandatory_lineage
                or not valid_director_projections
            ):
                issues.append(_issue(
                    "L1",
                    "severe",
                    "screenplay_plan_lineage_invalid",
                    "Screenplay production event lineage is incomplete or unsupported",
                ))
                production_event_ids = set()
            else:
                production_event_ids = {
                    int(record["source_event_id"])
                    for record in records
                    if record["production_status"] == "kept"
                }
    expected_units = {
        str(event.get("action_unit_id"))
        for event_id, event in enumerate(source_events, 1)
        if (
            production_event_ids is None
            or event_id in production_event_ids
        )
        and str(event.get("action_unit_id") or "").strip()
    }
    missing_units = sorted(expected_units - observed_units)
    if missing_units:
        issues.append(_issue(
            "L1", "severe", "action_unit_coverage_missing",
            f"Storyboard drops {len(missing_units)} screenplay action unit(s)",
            [], missing_action_unit_ids=missing_units,
        ))
    return issues


def _characters_in_shot(shot: dict, characters: list[dict]) -> list[str]:
    explicit = shot.get("character_ids", shot.get("characters", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    found = {str(value) for value in explicit if value} if isinstance(explicit, list) else set()
    haystack = _text(shot).casefold()
    for character in characters:
        cid = str(character.get("id", ""))
        names = [cid, str(character.get("name", "")), *[str(x) for x in character.get("aliases", []) if x]]
        if cid and any(name and name.casefold() in haystack for name in names):
            found.add(cid)
    return sorted(found)


def find_storyboard_images(output_dir: Path, storyboard: dict) -> dict[str, Path]:
    image_dir = output_dir / "storyboard_images"
    paths = list(image_dir.iterdir()) if image_dir.is_dir() else []
    result: dict[str, Path] = {}
    for index, shot in enumerate(storyboard.get("shots", [])):
        sid = _shot_id(shot, index)
        for path in paths:
            if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.upper() == sid.upper():
                result[sid] = path
                break
    return result


def find_storyboard_beat_images(output_dir: Path, storyboard: dict) -> dict[str, Path]:
    """Resolve every exact Pxx image; callers can detect omissions by count."""
    result: dict[str, Path] = {}
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        sid = _shot_id(shot, shot_index)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
            value = str(beat.get("storyboard_image") or "").strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file() and path.stat().st_size > 1024:
                result[beat_id] = path
    return result


def find_cinematic_first_frame_images(
    output_dir: Path,
    storyboard: dict,
) -> dict[str, Path]:
    """Resolve only Phase 4 frames explicitly declared as cinematic assets."""
    from phases.phase4.cinematic_first_frames import CINEMATIC_FIRST_FRAME_SCHEMA

    result: dict[str, Path] = {}
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            if beat.get("video_first_frame_kind") != CINEMATIC_FIRST_FRAME_SCHEMA:
                continue
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
            value = str(beat.get("video_first_frame") or "").strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file() and path.stat().st_size > 0:
                result[beat_id] = path
    return result


def run_l4_first_frame_review(
    storyboard: dict,
    visual_style: str,
    images: dict[str, Path],
    output_dir: Path,
    client: ArkMultimodalClient | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Fail closed on annotations or style drift in video-bound first frames."""
    declared_ids = [
        str(beat.get("beat_id") or f"{_shot_id(shot, shot_index)}_P{position:02d}")
        for shot_index, shot in enumerate(storyboard.get("shots", []))
        if isinstance(shot, dict)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1)
        if isinstance(beat, dict) and beat.get("video_first_frame_kind")
    ]
    if not declared_ids:
        return [], {
            "status": "skipped",
            "skipped_reason": "legacy storyboard declares no Phase 4 cinematic first frames",
        }
    ordered_ids = [frame_id for frame_id in declared_ids if frame_id in images]
    if not ordered_ids:
        return [], {
            "status": "skipped",
            "skipped_reason": "no declared cinematic first-frame pixels are available",
        }
    records = [
        {
            "input_index": index,
            "kind": "cinematic_video_first_frame",
            "frame_id": frame_id,
            "path": str(images[frame_id]),
            "sha256": _sha256_file(images[frame_id]),
        }
        for index, frame_id in enumerate(ordered_ids, 1)
    ]
    input_manifest = output_dir / "first_frame_qa_inputs.json"
    _write_l3_input_manifest(input_manifest, records)
    try:
        characters_payload = json.loads(
            (output_dir / "CHARACTERS.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        characters_payload = {}
    synthetic_makeup_contract = [
        {
            "character_id": character.get("id"),
            "synthetic_styling": character.get("appearance", {}).get(
                "synthetic_styling"
            ),
        }
        for character in characters_payload.get("characters") or []
        if isinstance(character, dict)
        and isinstance(character.get("appearance"), dict)
        and isinstance(
            character.get("appearance", {}).get("synthetic_styling"), dict
        )
    ]
    if client is None and not os.environ.get("ARK_AGENT_API_KEY"):
        shot_ids = sorted({_parent_shot_id(frame_id) for frame_id in ordered_ids})
        return [
            _issue(
                "L4",
                "severe",
                "first_frame_style_review_unavailable",
                "Cinematic first-frame annotation/style review is unavailable; paid video generation is blocked",
                shot_ids,
                frame_ids=ordered_ids,
            )
        ], {
            "status": "error",
            "input_manifest_path": str(input_manifest),
            "input_count": len(records),
            "skipped_reason": "ARK multimodal API key missing",
        }
    prompt = f"""Review every supplied image as a video-bound cinematic first frame. Each image is attached in ascending input_index order and mapped to an exact frame_id below.

Four fail-closed checks apply independently:
1. ANNOTATION_CONTAMINATION: report any visible action/camera arrow, trajectory or helper line, Sxx/Pxx/Gxx label, letter, number, subtitle, watermark, UI, panel border, split-screen, contact sheet, storyboard grid, handwritten note, or other production annotation.
2. STYLE_MISMATCH: compare the visible palette, materials, rendering medium, stage/environment structure, lighting, and finish against VISUAL STYLE. Report a mismatch when the frame is visibly PREVIS, pencil/charcoal/line-art, generic CGI/photography that contradicts the authored medium, or omits a defining environment such as a stage/curtain explicitly required by the style. Do not report minor composition differences.
3. SUBJECT_DUPLICATION: report an unintended duplicate, translucent copy, double exposure, repeated face/body, or extra instance of the same authored subject. Do not report multiple distinct characters when the scene calls for them.
4. SYNTHETIC_MAKEUP_MISMATCH: when a synthetic makeup contract is supplied below, require the same beautiful, unobscured pearl bio-ceramic porcelain makeup, character-specific narrow temple-to-cheek circuit stripe and luminous iris ring in every face-visible frame. Report untreated human skin, a veil/mask, coarse mechanical face plate, crack, scar, horror distortion, makeup color/pattern drift, or performance-board clone/layout contamination.

Inspect each frame independently. Never copy one observation across multiple IDs. Every blocking issue needs concrete visible evidence, expected, observed, and confidence >= 0.85. Lower-confidence negative findings are diagnostic acceptable deviations. Annotation contamination, subject duplication, and material style mismatch can block only when those evidence rules are met.

Return JSON only: {{"issues":[{{"code":"ANNOTATION_CONTAMINATION|STYLE_MISMATCH|SUBJECT_DUPLICATION|SYNTHETIC_MAKEUP_MISMATCH","severity":"severe|moderate|minor","frame_ids":["S01_P01"],"message":"...","expected":"...","observed":"...","confidence":0.95,"frame_evidence":[{{"frame_id":"S01_P01","observed":"specific visible evidence"}}]}}]}}.
Use only these frame IDs: {json.dumps(ordered_ids, ensure_ascii=False)}.
INPUTS:
{json.dumps(records, ensure_ascii=False)}
VISUAL STYLE:
{visual_style}
SYNTHETIC MAKEUP CONTRACTS:
{json.dumps(synthetic_makeup_contract, ensure_ascii=False, sort_keys=True)}"""
    input_paths = [images[frame_id] for frame_id in ordered_ids]
    try:
        from clients.ark_multimodal_client import review_as
        from quality.visual_qa_policy import (
            POLICY_ID,
            decide_visual_qa,
            policy_sha256,
        )
        from runtime.qa_ledger import QALedger, observation_fingerprint
        from schemas.understanding import FirstFrameUnderstanding
        from utils.canonical_visual_contracts import load_canonical_visual_contract

        review_client = client or ArkMultimodalClient()
        canonical_contract = load_canonical_visual_contract(output_dir)
        canonical_contract_sha256 = str(canonical_contract["contract_sha256"])
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        evaluator_model = str(getattr(review_client, "model", "unknown-vlm"))
        evidence_fingerprint = observation_fingerprint(
            evidence=records,
            canonical_contract_sha256=canonical_contract_sha256,
            evaluator_model=evaluator_model,
            prompt_sha256=prompt_sha256,
            observation_schema="FirstFrameUnderstanding.v1",
        )
        ledger = QALedger(output_dir / "runtime.db")
        stored_observation = ledger.find_observation(evidence_fingerprint)
        if stored_observation is not None:
            typed_review = FirstFrameUnderstanding.model_validate(
                stored_observation.observation
            )
            review_execution = {
                "status": "reused_qa_observation",
                "attempt_count": 0,
                "provider_request_count": 0,
            }
            observation = stored_observation
            observation_reused = True
        else:
            typed_review, review_execution = execute_structured_understanding(
                lambda: review_as(
                    review_client,
                    input_paths,
                    prompt,
                    FirstFrameUnderstanding,
                )
            )
            try:
                run_manifest = json.loads(
                    (output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
                )
                ledger_run_id = str(run_manifest["run_fingerprint"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                ledger_run_id = output_dir.name
            observation, observation_reused = ledger.record_observation(
                run_id=ledger_run_id,
                phase="phase5",
                resource_id="cinematic_first_frames",
                evidence_fingerprint=evidence_fingerprint,
                canonical_contract_sha256=canonical_contract_sha256,
                evaluator_model=evaluator_model,
                prompt_sha256=prompt_sha256,
                observation_schema="FirstFrameUnderstanding.v1",
                observation=typed_review.model_dump(mode="json"),
            )
        parsed = typed_review.model_dump()
        findings = [
            {
                "blocking_category": str(value.get("code") or "").casefold(),
                "confidence": value.get("confidence"),
                "evidence": value.get("frame_evidence")
                or (
                    f"expected={value.get('expected')}; observed={value.get('observed')}"
                    if value.get("expected") and value.get("observed")
                    else ""
                ),
                "finding": value,
            }
            for value in parsed["issues"]
        ]
        policy_decision = decide_visual_qa(
            semantic_score=1.0,
            findings=findings,
        )
        decision, decision_reused = ledger.record_decision(
            observation_id=observation.observation_id,
            phase_owner="phase5.first_frame_qa",
            policy_id=POLICY_ID,
            policy_sha256=policy_sha256(),
            verdict=policy_decision.verdict,
            semantic_score=policy_decision.semantic_score,
            decision=policy_decision.as_dict(),
        )
        valid_ids = set(ordered_ids)
        issues: list[dict[str, Any]] = []
        for value in parsed["issues"]:
            if not isinstance(value, dict):
                continue
            code = str(value.get("code") or "").upper()
            if code not in {
                "ANNOTATION_CONTAMINATION",
                "STYLE_MISMATCH",
                "SUBJECT_DUPLICATION",
                "SYNTHETIC_MAKEUP_MISMATCH",
            }:
                continue
            frame_ids = [
                frame_id
                for frame_id in (value.get("frame_ids") or [])
                if frame_id in valid_ids
            ]
            try:
                confidence = float(value.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0
            expected = str(value.get("expected") or "").strip()
            observed = str(value.get("observed") or "").strip()
            evidence = value.get("frame_evidence") or []
            evidence_ids = {
                str(item.get("frame_id") or "")
                for item in evidence
                if isinstance(item, dict) and str(item.get("observed") or "").strip()
            }
            evidence_valid = bool(
                frame_ids
                and expected
                and observed
                and confidence >= 0.85
                and set(frame_ids).issubset(evidence_ids)
            )
            severity = (
                "severe"
                if evidence_valid
                else "minor"
            )
            issues.append(
                _issue(
                    "L4",
                    severity,
                    f"first_frame_{code.casefold()}",
                    str(value.get("message") or f"First-frame {code} issue"),
                    sorted({_parent_shot_id(frame_id) for frame_id in frame_ids}),
                    frame_ids=frame_ids,
                    expected=expected,
                    observed=observed,
                    confidence=confidence,
                    frame_evidence=evidence,
                    evidence_status="validated" if evidence_valid else "unverified",
                )
            )
        return issues, {
            "status": "completed",
            "input_manifest_path": str(input_manifest),
            "input_count": len(records),
            "raw_issue_count": len(parsed["issues"]),
            "structured_review_execution": review_execution,
            "qa_observation_id": observation.observation_id,
            "qa_observation_reused": observation_reused,
            "qa_decision_id": decision.decision_id,
            "qa_decision_reused": decision_reused,
            "qa_verdict": decision.verdict,
        }
    except Exception as exc:
        shot_ids = sorted({_parent_shot_id(frame_id) for frame_id in ordered_ids})
        layer = {
            "status": "error",
            "input_manifest_path": str(input_manifest),
            "input_count": len(records),
            "skipped_reason": f"multimodal review unavailable: {exc}",
        }
        if isinstance(exc, StructuredUnderstandingExhausted):
            layer["structured_review_execution"] = exc.receipt
        return [
            _issue(
                "L4",
                "severe",
                "first_frame_style_review_unavailable",
                f"Cinematic first-frame review failed: {exc}",
                shot_ids,
                frame_ids=ordered_ids,
            )
        ], layer


def _parent_shot_id(image_id: str) -> str:
    match = re.match(r"^(.*)_P\d+$", image_id, flags=re.IGNORECASE)
    return match.group(1) if match else image_id


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    if norm == 0:
        raise ValueError("embedding vectors must have non-zero norms")
    return dot / norm


def run_l2_checks(storyboard: dict, characters_data: dict, images: dict[str, Path], threshold: float = DEFAULT_SIMILARITY_THRESHOLD, embedder: Callable[[str], list[float] | None] | None = None) -> tuple[list[dict], dict]:
    """Embed whole frames as scene diagnostics, never as character crops.

    A whole-frame embedding cannot isolate a particular person.  Older code
    copied the same frame matrix under every character ID and treated it as
    identity evidence.  L2 now reports one honest scene-level matrix; canonical
    character-reference comparison is delegated to the multimodal L3 review.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("similarity threshold must be between 0 and 1")
    if embedder is None:
        from utils.shot_embedder import embed_image
        embedder = embed_image
    vectors: dict[str, list[float]] = {}
    errors: list[str] = []
    for sid, path in images.items():
        try:
            vector = embedder(str(path))
            if vector:
                vectors[sid] = [float(value) for value in vector]
        except Exception as exc:  # preserve the failure in the report
            errors.append(f"{sid}: {exc}")

    image_ids = [image_id for image_id in images if image_id in vectors]
    matrix: list[list[float]] = []
    for left in image_ids:
        row: list[float] = []
        for right in image_ids:
            try:
                score = cosine_similarity(vectors[left], vectors[right])
            except ValueError as exc:
                errors.append(f"{left}/{right}: {exc}")
                score = 0.0
            row.append(round(score, 6))
        matrix.append(row)
    status = "completed" if vectors else "skipped"
    reason = None if vectors else ("ARK_AGENT_API_KEY missing or embedding service returned no vectors")
    return [], {
        "status": status,
        "skipped_reason": reason,
        "scope": "whole_frame_scene_consistency",
        "character_isolation": False,
        "identity_review_layer": "L3_canonical_references",
        "threshold": threshold,
        "embedded_shots": sorted(vectors),
        "scene_matrix": {"storyboard_ids": image_ids, "matrix": matrix},
        "errors": errors,
    }


def create_storyboard_grid(image_paths: list[Path], output_path: Path, columns: int = 5) -> Path:
    """Create a labelled, artifact-derived contact sheet using Pillow."""
    if not image_paths:
        raise ValueError("at least one storyboard image is required")
    from PIL import Image, ImageDraw, ImageFont
    opened = [Image.open(path).convert("RGB") for path in image_paths]
    width = min(480, max(image.width for image in opened))
    height = max(1, int(max(image.height / image.width for image in opened) * width))
    rows = math.ceil(len(opened) / columns)
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(grid)
    font_size = max(28, width // 10)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # Pillow < 10
            font = ImageFont.load_default()
    for index, (path, source) in enumerate(zip(image_paths, opened, strict=True)):
        source.thumbnail((width, height))
        x = (index % columns) * width + (width - source.width) // 2
        y = (index // columns) * height
        grid.paste(source, (x, y))
        # Put a large high-contrast ID inside each frame. Small captions below
        # a dense contact sheet caused the VLM to associate S15 with S11 and
        # neighbouring final shots with S24 in a real 24-shot run.
        label = path.stem.upper()
        label_x = (index % columns) * width + 10
        label_y = y + 10
        bbox = draw.textbbox((label_x, label_y), label, font=font, stroke_width=1)
        padding = 8
        draw.rounded_rectangle(
            (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ),
            radius=6,
            fill="black",
        )
        draw.text(
            (label_x, label_y),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def create_storyboard_shot_boards(
    storyboard: dict,
    images: dict[str, Path],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Create one high-detail, labelled evidence board per authored shot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    boards: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        storyboard_ids: list[str] = []
        paths: list[Path] = []
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
            if beat_id in images:
                storyboard_ids.append(beat_id)
                paths.append(images[beat_id])
        if not paths and shot_id in images:
            storyboard_ids.append(shot_id)
            paths.append(images[shot_id])
        if not paths:
            continue
        board_path = output_dir / f"storyboard_reference_{shot_id}.jpg"
        create_storyboard_grid(
            paths,
            board_path,
            columns=max(1, min(len(paths), 3)),
        )
        boards.append({
            "shot_id": shot_id,
            "storyboard_ids": storyboard_ids,
            "path": board_path,
            "source_paths": paths,
        })
    return boards


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_l3_input_manifest(path: Path, records: list[dict[str, Any]]) -> Path:
    """Persist the exact ordered pixels sent to the Phase 5 vision model."""
    payload = {
        "schema": "honcut.storyboard-qa-inputs.v1",
        "input_count": len(records),
        "inputs": records,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_l3_batched_input_manifest(
    path: Path,
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> Path:
    """Persist unique evidence plus each exact shot-scoped Provider request."""
    payload = {
        "schema": "honcut.storyboard-qa-inputs.v2",
        "input_count": len(records),
        "provider_input_count": sum(
            int(request.get("input_count") or 0) for request in requests
        ),
        "request_count": len(requests),
        "inputs": records,
        "requests": requests,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _ordered_storyboard_images(
    storyboard: dict, images: dict[str, Path]
) -> list[Path]:
    """Order beat-level images by shot and beat, falling back to shot images."""
    ordered: list[Path] = []
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        beat_paths: list[Path] = []
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
            if beat_id in images:
                beat_paths.append(images[beat_id])
        if beat_paths:
            ordered.extend(beat_paths)
        elif shot_id in images:
            ordered.append(images[shot_id])
    return ordered


def find_character_reference_images(
    output_dir: Path,
    characters_data: dict,
) -> dict[str, list[Path]]:
    """Resolve canonical Phase 3 references in stable identity-first order."""
    output_dir = Path(output_dir)
    result: dict[str, list[Path]] = {}
    for character in characters_data.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id") or "").strip()
        if not character_id:
            continue
        reference_board = resolve_character_reference_board(output_dir, character_id)
        if reference_board is not None:
            result[character_id] = [reference_board]
            continue
        character_dir = output_dir / "characters" / character_id
        card_path = character_dir / "character_card.json"
        declared: dict[str, Any] = {}
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(card.get("reference_images"), dict):
                declared = card["reference_images"]
        except (OSError, json.JSONDecodeError):
            pass
        paths: list[Path] = []
        for view_name in (
            "face_closeup",
            "full_body",
            "closeup",
            "front",
            "side",
            "back",
            "three_quarter",
            "detail",
        ):
            value = declared.get(view_name)
            path = Path(str(value)) if value else character_dir / f"{view_name}.png"
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file() and path.stat().st_size > 0 and path not in paths:
                paths.append(path)
        if paths:
            result[character_id] = paths
    return result


def _calibrate_l3_severity(red_line: str, severity: str, message: str) -> str:
    """Keep L3 blocking for production-breaking mismatches, not pose minutiae."""
    if severity != "severe":
        return severity
    normalized_line = red_line.upper()
    text = message.casefold()
    # Match explicit mismatch assertions, not isolated descriptor words. The
    # former terms "male" and "female" made any sentence beginning with
    # "the female character..." a hard blocker even when no gender mismatch
    # was alleged. Likewise, a VLM's unsupported celebrity resemblance claim
    # is not evidence of a wrong character identity.
    hard_blockers = (
        "wrong identity",
        "different identity",
        "identity mismatch",
        "wrong gender",
        "different gender",
        "gender mismatch",
        "male instead of",
        "female instead of",
        "missing character",
        "wrong character",
        "reversed attacker",
        "wrong location",
        "reset",
        "replay",
        "replays prior",
        "wholly unrelated",
        "unrelated core action",
        "身份错误",
        "身份不一致",
        "性别错误",
        "性别不一致",
        "男性而非",
        "女性而非",
        "角色缺失",
        "人物错误",
        "攻守颠倒",
        "场景错误",
        "重置",
        "重放",
        "重复前格",
        "回到初始",
        "核心动作缺失",
    )
    if normalized_line == "R1" and not any(term in text for term in hard_blockers):
        return "moderate"
    if normalized_line in {"R3", "R4"} and not any(
        term in text for term in hard_blockers
    ):
        return "moderate"
    return severity


_R1_ATTRIBUTE_TERMS = (
    "color",
    "colour",
    "clothing",
    "costume",
    "uniform",
    "outfit",
    "颜色",
    "服装",
    "制服",
    "衣服",
    "穿着",
)


def _normalized_visual_attribute(value: Any) -> str:
    """Normalize concise expected/observed R1 attribute claims."""
    text = str(value or "").casefold()
    aliases = {
        "dark grey": "darkgray",
        "dark gray": "darkgray",
        "深灰色": "深灰",
        "navy blue": "navy",
        "dark blue": "navy",
        "藏蓝色": "藏蓝",
    }
    for source, target in aliases.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


_AFFIRMATIVE_NON_ISSUE = re.compile(
    r"(?:\bno\s+(?:visible\s+)?(?:mismatch|difference|deviation|issue)\b|"
    r"\bno\s+(?:visible\s+)?(?:incomplete|missing|incorrect)\s+"
    r"(?:end[- ]?state|action|sequence)\b|"
    r"\b(?:fully\s+|all\s+)?match(?:es|ing)?\s+(?:the\s+)?canonical\b|"
    r"\b(?:fully\s+|all\s+)?match(?:es|ing)?\s+(?:the\s+)?"
    r"(?:required|authored|expected|contracted)\b|"
    r"\b(?:fully\s+|all\s+)?match(?:es|ing)?\s+(?:the\s+)?"
    r"(?:action|end[- ]?state|sequence)\s+requirements?\b|"
    r"\bconsistent\s+with\s+(?:the\s+)?canonical\b|"
    r"\b(?:is|are|was|were|has\s+been|have\s+been)\s+"
    r"(?:fully|completely)\s+(?:completed|satisfied|fulfilled|achieved)\b|"
    r"(?:没有|无)(?:明显)?(?:不一致|不匹配|不符|偏差|差异|问题)|"
    r"(?:均已|已经|已全部|已完整|已完全)(?:完成|达成|满足)|"
    r"(?:符合|匹配)(?:原始|规范|标准|角色|服装|颜色|设定|合同|契约))",
    re.IGNORECASE,
)
_CONTRASTED_FINDING = re.compile(
    r"(?:\bbut\b|\bhowever\b|\bexcept\b|\byet\b|但|然而|不过|只是|却)",
    re.IGNORECASE,
)
_NEGATED_VISUAL_PROBLEM = re.compile(
    r"(?:\b(?:no|without)\s+(?:visible\s+)?"
    r"(?:mismatch|difference|deviation|issue)\b|"
    r"\b(?:no|without)\s+(?:visible\s+)?"
    r"(?:incomplete|missing|incorrect)\s+"
    r"(?:end[- ]?state|action|sequence)\b|"
    r"(?:没有|无)(?:明显)?(?:不一致|不匹配|不符|偏差|差异|问题))",
    re.IGNORECASE,
)
_NEGATIVE_VISUAL_FINDING = re.compile(
    r"(?:\b(?:does?|do|is|are|was|were|has|have)\s+not\b|"
    r"\bnot\s+(?:fully\s+)?(?:match(?:ing)?|complete(?:d)?|"
    r"consistent|correct)\b|"
    r"\b(?:mismatch(?:es|ed)?|differ(?:s|ed)?|deviat(?:es|ed|ion)|"
    r"wrong|incorrect|missing|absent|incomplete|oversized|undersized|"
    r"violat(?:es|ed|ing|ion)|fail(?:s|ed|ure)?)\b|"
    r"(?:不一致|不匹配|不符|错误|缺失|缺少|未完成|偏离|违反|失败))",
    re.IGNORECASE,
)


def _contains_negative_visual_finding(text: str) -> bool:
    """Return whether an observation still asserts a problem after negation."""
    without_negated_problems = _NEGATED_VISUAL_PROBLEM.sub(" ", text)
    return bool(_NEGATIVE_VISUAL_FINDING.search(without_negated_problems))


def _is_affirmative_non_issue(value: dict[str, Any]) -> bool:
    """Reject provider findings that explicitly say no mismatch exists.

    Structured review models occasionally put a positive observation inside
    the ``issues`` array.  Exact expected/observed equality is conclusive.  An
    explicitly affirmative observation is conclusive only when it contains no
    remaining negative visual claim.  A positive message is filtered only
    when it contains no contrast clause, so ``costume matches, but proportions
    drift`` remains a real finding.
    """
    expected = _normalized_visual_attribute(value.get("expected"))
    observed = _normalized_visual_attribute(value.get("observed"))
    if expected and expected == observed:
        return True
    observed_text = re.sub(
        r"\s+", " ", str(value.get("observed") or "")
    ).strip()
    if (
        observed_text
        and _AFFIRMATIVE_NON_ISSUE.search(observed_text)
        and not _contains_negative_visual_finding(observed_text)
    ):
        return True
    message = re.sub(r"\s+", " ", str(value.get("message") or "")).strip()
    return bool(
        message
        and _AFFIRMATIVE_NON_ISSUE.search(message)
        and not _CONTRASTED_FINDING.search(message)
    )


def _r1_attribute_evidence(
    value: dict[str, Any],
    storyboard_ids: list[str],
    reference_context: int | dict[int, str],
    canonical_contracts: dict[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate evidence for clothing/color continuity claims.

    R1 identity and gender findings remain governed by the severe mismatch
    calibration. Attribute drift is more vulnerable to lighting/style
    hallucinations, so it needs explicit canonical and per-panel evidence
    before it can block paid generation.
    """
    mismatch_type = str(value.get("mismatch_type") or "").casefold()
    message = str(value.get("message") or "").casefold()
    is_attribute_claim = mismatch_type in {
        "clothing_color",
        "clothing",
        "costume",
        "uniform",
        "color",
        "colour",
    } or any(term in message for term in _R1_ATTRIBUTE_TERMS)
    if not is_attribute_claim:
        return True, {"evidence_status": "not_attribute_claim"}

    reference_characters = (
        dict(reference_context)
        if isinstance(reference_context, dict)
        else {}
    )
    reference_count = (
        max(reference_characters, default=0)
        if isinstance(reference_context, dict)
        else int(reference_context)
    )

    expected = str(value.get("expected") or "").strip()
    observed = str(value.get("observed") or "").strip()
    raw_reference_indices = value.get("reference_input_indices") or []
    if not isinstance(raw_reference_indices, list):
        raw_reference_indices = [raw_reference_indices]
    reference_indices = sorted({
        int(index)
        for index in raw_reference_indices
        if str(index).isdigit() and 1 <= int(index) <= reference_count
    })
    raw_panel_evidence = value.get("panel_evidence") or []
    if not isinstance(raw_panel_evidence, list):
        raw_panel_evidence = []
    evidence_ids = {
        str(item.get("shot_id") or "")
        for item in raw_panel_evidence
        if isinstance(item, dict)
        and str(item.get("observed") or "").strip()
    }
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    expected_normalized = _normalized_visual_attribute(expected)
    observed_normalized = _normalized_visual_attribute(observed)
    reasons = []
    if not expected_normalized or not observed_normalized:
        reasons.append("missing_expected_or_observed")
    elif expected_normalized == observed_normalized:
        reasons.append("expected_equals_observed")
    if not reference_indices:
        reasons.append("missing_canonical_reference")
    if not storyboard_ids or not set(storyboard_ids).issubset(evidence_ids):
        reasons.append("missing_per_panel_evidence")
    if not math.isfinite(confidence) or confidence < 0.85 or confidence > 1.0:
        reasons.append("confidence_below_0.85")
    character_evidence = value.get("character_evidence") or []
    if reference_characters:
        if not isinstance(character_evidence, list) or not character_evidence:
            reasons.append("missing_character_evidence")
            character_evidence = []
        covered_storyboard_ids: set[str] = set()
        for evidence in character_evidence:
            if not isinstance(evidence, dict):
                reasons.append("invalid_character_evidence")
                continue
            character_id = str(evidence.get("character_id") or "").strip()
            contract = str((canonical_contracts or {}).get(character_id) or "").strip()
            expected_for_character = str(evidence.get("expected") or "").strip()
            observed_for_character = str(evidence.get("observed") or "").strip()
            raw_indices = evidence.get("reference_input_indices") or []
            if not isinstance(raw_indices, list):
                raw_indices = [raw_indices]
            evidence_indices = {
                int(index) for index in raw_indices if str(index).isdigit()
            }
            if (
                not character_id
                or not evidence_indices
                or any(
                    reference_characters.get(index) != character_id
                    for index in evidence_indices
                )
            ):
                reasons.append("character_reference_mapping_invalid")
            if not contract or (
                _normalized_visual_attribute(expected_for_character)
                != _normalized_visual_attribute(contract)
            ):
                reasons.append("expected_not_exact_canonical_contract")
            if (
                not observed_for_character
                or _normalized_visual_attribute(observed_for_character)
                == _normalized_visual_attribute(expected_for_character)
            ):
                reasons.append("character_expected_equals_observed")
            raw_ids = evidence.get("storyboard_ids") or []
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            covered_storyboard_ids.update(
                str(storyboard_id)
                for storyboard_id in raw_ids
                if str(storyboard_id) in storyboard_ids
            )
        if not set(storyboard_ids).issubset(covered_storyboard_ids):
            reasons.append("character_evidence_missing_storyboard_ids")
    return not reasons, {
        "evidence_status": "validated" if not reasons else "unverified",
        "evidence_reasons": reasons,
        "mismatch_type": mismatch_type or "unspecified_attribute",
        "expected": expected,
        "observed": observed,
        "reference_input_indices": reference_indices,
        "confidence": confidence,
        "character_evidence": character_evidence,
    }


_L3_SHOT_CONTEXT_FIELDS = (
    "source_sequence_ids",
    "where",
    "time",
    "time_window",
    "time_of_day",
    "lighting_description",
    "shot_size",
    "camera_angle",
    "camera_movement",
    "shot_intent",
    "transition_to_next",
)
_L3_LEGACY_SHOT_CONTRACT_FIELDS = (
    *_L3_SHOT_CONTEXT_FIELDS,
    "source_events",
    "what",
    "visual",
    "start_state",
    "end_state",
    "director_intent",
)
_L3_BEAT_CONTRACT_FIELDS = (
    "beat_id",
    "character_ids",
    "action",
    "micro_actions",
    "start_state",
    "end_state",
    "source_action_unit_ids",
    "duration_s",
    "generation_mode",
    "shot_size",
    "camera_angle",
    "camera_movement",
    "lighting_key",
    "shot_intent",
)


def _selected_fields(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Project an artifact onto an explicit semantic prompt contract."""
    return {
        field: value[field]
        for field in fields
        if field in value and value[field] not in (None, "", [], {})
    }


def _l3_storyboard_contract(storyboard: dict[str, Any]) -> dict[str, Any]:
    """Keep visual semantics while excluding prompts, paths, hashes and receipts."""
    shots = []
    for index, shot in enumerate(storyboard.get("shots") or []):
        if not isinstance(shot, dict):
            continue
        beats = [
            beat
            for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        projected = {
            "shot_id": _shot_id(shot, index),
            **_selected_fields(
                shot,
                _L3_SHOT_CONTEXT_FIELDS
                if beats
                else _L3_LEGACY_SHOT_CONTRACT_FIELDS,
            ),
        }
        projected_beats = []
        for beat in beats:
            projected_beat = _selected_fields(beat, _L3_BEAT_CONTRACT_FIELDS)
            if "character_ids" in beat:
                raw_character_ids = beat.get("character_ids")
                if not isinstance(raw_character_ids, list):
                    raise ValueError("L3 beat character_ids must be an array")
                character_ids = [
                    str(value).strip()
                    for value in raw_character_ids
                    if str(value).strip()
                ]
                if character_ids != list(dict.fromkeys(character_ids)):
                    raise ValueError(
                        "L3 beat character_ids must be unique and ordered"
                    )
                # An explicit empty array is a meaningful no-character contract.
                projected_beat["character_ids"] = character_ids
            projected_beats.append(projected_beat)
        projected["storyboard_beats"] = projected_beats
        shots.append(projected)
    return {
        "schema": "honcut.phase5-l3-semantic-projection.v3",
        "shots": shots,
    }


def _l3_review_prompt(
    *,
    reference_inputs: list[dict[str, Any]],
    storyboard_inputs: list[dict[str, Any]],
    overview_input: dict[str, Any],
    canonical_contracts: dict[str, str],
    storyboard: dict[str, Any],
    visual_style: str,
    valid_storyboard_ids: list[str],
) -> str:
    """Build one bounded, de-duplicated semantic review request."""
    reference_index = [
        _selected_fields(
            item,
            ("input_index", "kind", "character_id", "view"),
        )
        for item in reference_inputs
    ]
    storyboard_index = [
        _selected_fields(
            item,
            ("input_index", "kind", "shot_id", "storyboard_ids"),
        )
        for item in storyboard_inputs
    ]
    overview_index = _selected_fields(
        overview_input,
        ("input_index", "kind", "storyboard_ids"),
    )
    compact = lambda value: json.dumps(
        value,
        ensure_ascii=False,
    )
    return f"""[L3 REVIEW CONTRACT]
Review the supplied storyboard evidence against the canonical character references and authored project artifacts. Every image listed below is attached to this exact request in ascending input_index order.

INPUT CONTRACT:
- Canonical character inputs are mapped to character_id and view in CHARACTER REFERENCE INPUTS.
- A canonical input with view=reference_board is one 2x2 board showing four views of the same single character, never four people or clones; compare its face, front, side and back cells as one identity.
- Storyboard inputs are high-detail per-shot boards containing only the exact Pxx images listed in STORYBOARD SHOT INPUTS, with a large black ID badge inside every panel.
- The overview grid is for cross-shot continuity only. Use per-shot boards for fine identity, clothing, prop, and action evidence.
- Never swap expected and observed: canonical character inputs define EXPECTED; storyboard boards define OBSERVED. Never call a character reference a storyboard image.
- Associate observations only with the in-frame Sxx or Sxx_Pxx badge; never infer an ID from a neighbouring cell or row position.
- Each Pxx character_ids array is the exact visible named cast for that panel; an empty array means no named character should appear. Shot-wide cast is intentionally excluded because it is only a superset across the shot. A canonical reference attached for comparison does not imply that character appears in every Pxx.
- When storyboard_beats exist, Pxx action/start/end/character_ids are the only plot-bearing contract. Shot-wide what/visual/source_events/director_intent are intentionally excluded because they summarize multiple Pxx and must not expand one panel's cast or action.

Apply red lines R1-R4: R1 character identity/gender/build/clothing continuity against canonical references; R2 time-of-day and lighting continuity; R3 scene/action continuity; R4 storyboard-to-image semantic fidelity. Do not perform face recognition or infer a public identity from appearance alone. Each Pxx image represents only its authored action and must progress from the previous Pxx without pose reset or premature future action.

The current Sxx narrative-guide image is intentionally annotated production guidance, not a cinematic frame. Under R4, verify it visibly contains the exact Sxx_G01-G09 labels plus red subject/object direction arrows, blue camera-motion arrows where camera motion is declared, and spatial/gaze/action instruction markers. Missing or misindexed guidance is a blocking storyboard-authoring mismatch. Do not report those required annotations as cinematic contamination; the separate L4 review owns the rule that cinematic first frames contain none of them.

Evidence rules:
- Report only visible contradictions. Absence of proof is not proof of mismatch. Do not infer clothing-color drift from red warning light, shadow, monochrome PREVIS rendering, highlights, or low saturation.
- For every R1 clothing/color/uniform claim, set mismatch_type="clothing_color". For each affected character, add one character_evidence object containing the exact character_id, only that character's canonical reference_input_indices, expected copied verbatim from canonical_contracts, an observed visual description, and the exact storyboard_ids checked. Do not paraphrase or invent the expected contract. Also provide confidence from 0 to 1 and separate panel_evidence for every listed ID. Expected and observed must name genuinely different visible attributes. If both are dark gray, both are navy, or the difference is only illumination/style, emit no issue.
- Never copy one panel observation across a range. List multiple IDs only after independently checking each badge; panel_evidence must contain one observation per listed ID.
- For R4, compare the literal actor → action → target → prop ownership → end state. A mutual weapon-disarm action is not satisfied by one actor aiming a weapon. Do not reverse attacker/defender or weapon ownership.
- For the final Pxx, verify the authored result is visibly complete. If the contract says stable/stopped/freeze-frame while another character flies toward or hits a target, ongoing mutual fighting or generic floating is a mismatch.

Reserve severe for production-breaking mismatches: wrong/missing character identity or gender, wrong location/time-of-day, reversed attacker/defender, wholly unrelated core action, or a continuation panel that visibly resets/replays the prior state. Use moderate for material but recoverable semantic mismatches, exact prop/action-state errors, blocking offsets, or intermediate/final-state omissions. Only identify problems; do not propose or perform edits.

Return JSON only: {{"issues":[{{"red_line":"R1|R2|R3|R4","severity":"severe|moderate|minor","mismatch_type":"identity|gender|clothing_color|lighting|action|end_state|other","shot_ids":["S01_P01"],"message":"...","reference_input_indices":[1],"expected":"canonical concise fact","observed":"visibly different concise fact","confidence":0.90,"character_evidence":[{{"character_id":"agent","reference_input_indices":[1,2],"expected":"exact canonical_contract text","observed":"specific visible storyboard attribute","storyboard_ids":["S01_P01"]}}],"panel_evidence":[{{"shot_id":"S01_P01","observed":"specific visible evidence in this panel"}}]}}]}}. Use only these exact IDs: {compact(valid_storyboard_ids)}.

[L3 INPUT INDEX]
CHARACTER REFERENCE INPUTS: {compact(reference_index)}
STORYBOARD SHOT INPUTS: {compact(storyboard_index)}
OVERVIEW INPUT: {compact(overview_index)}

[L3 CANONICAL CONTRACTS]
canonical_contracts: {compact(canonical_contracts)}

[L3 STORYBOARD CONTRACTS]
{compact(_l3_storyboard_contract(storyboard))}

[L3 VISUAL STYLE]
{visual_style}"""


def run_l3_review(
    storyboard: dict,
    characters_data: dict,
    visual_style: str,
    images: dict[str, Path],
    grid_path: Path,
    client: ArkMultimodalClient | None = None,
    *,
    character_reference_images: dict[str, list[Path]] | None = None,
    evidence_dir: Path | None = None,
    input_manifest_path: Path | None = None,
    structured_understanding_max_attempts: int = (
        DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS
    ),
    _acceptance_max_new_observations: int | None = None,
    _acceptance_before_provider_request: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> tuple[list[dict], dict]:
    if (
        _acceptance_max_new_observations is not None
        and (
            isinstance(_acceptance_max_new_observations, bool)
            or not isinstance(_acceptance_max_new_observations, int)
            or _acceptance_max_new_observations < 1
        )
    ):
        raise ValueError(
            "_acceptance_max_new_observations must be a positive integer or None"
        )
    if not images:
        return [], {"status": "skipped", "skipped_reason": "no storyboard images available"}
    ordered = _ordered_storyboard_images(storyboard, images)
    if not ordered:
        return [], {
            "status": "skipped",
            "skipped_reason": "no storyboard images match storyboard IDs",
        }
    create_storyboard_grid(ordered, grid_path)
    reference_inputs: list[Path] = []
    reference_manifest: list[dict[str, Any]] = []
    canonical_contracts: dict[str, str] = {}
    references = character_reference_images or {}
    for character in characters_data.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id") or "").strip()
        if character_id:
            canonical_contracts[character_id] = character_visual_description(character)
    ordered_character_ids = [
        str(character.get("id") or "")
        for character in characters_data.get("characters", [])
        if isinstance(character, dict) and str(character.get("id") or "")
    ]
    ordered_character_ids.extend(
        sorted(set(references) - set(ordered_character_ids))
    )
    for character_id in ordered_character_ids:
        for path in references.get(character_id, []):
            path = Path(path)
            if not path.is_file() or path in reference_inputs:
                continue
            reference_inputs.append(path)
            reference_manifest.append({
                "input_index": len(reference_inputs),
                "kind": "canonical_character_reference",
                "character_id": character_id,
                "view": path.stem,
                "canonical_contract": canonical_contracts.get(character_id, ""),
                "path": str(path),
                "sha256": _sha256_file(path),
            })
    evidence_dir = evidence_dir or grid_path.parent / "phase5_reference_boards"
    shot_boards = create_storyboard_shot_boards(
        storyboard,
        images,
        evidence_dir,
    )
    valid_storyboard_ids = [
        storyboard_id
        for board in shot_boards
        for storyboard_id in board["storyboard_ids"]
    ]
    storyboard_manifest: list[dict[str, Any]] = []
    for board in shot_boards:
        path = Path(board["path"])
        storyboard_manifest.append({
            "input_index": len(reference_manifest) + len(storyboard_manifest) + 1,
            "kind": "storyboard_shot_board",
            "shot_id": board["shot_id"],
            "storyboard_ids": board["storyboard_ids"],
            "path": str(path),
            "sha256": _sha256_file(path),
            "source_images": [
                {"path": str(source), "sha256": _sha256_file(Path(source))}
                for source in board["source_paths"]
            ],
        })
    grid_input_index = len(reference_manifest) + len(storyboard_manifest) + 1
    overview_manifest = {
        "input_index": grid_input_index,
        "kind": "storyboard_overview_grid",
        "storyboard_ids": valid_storyboard_ids,
        "path": str(grid_path),
        "sha256": _sha256_file(grid_path),
    }
    input_records = [*reference_manifest, *storyboard_manifest, overview_manifest]
    shot_values = [
        shot
        for shot in (storyboard.get("shots") or [])
        if isinstance(shot, dict)
    ]
    shot_index_by_id = {
        _shot_id(shot, index): index for index, shot in enumerate(shot_values)
    }
    characters = [
        value
        for value in (characters_data.get("characters") or [])
        if isinstance(value, dict)
    ]
    reference_ids = {
        str(record.get("character_id") or "") for record in reference_manifest
    }
    requests: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for board, board_record in zip(shot_boards, storyboard_manifest):
        shot_id = str(board["shot_id"])
        shot_index = shot_index_by_id.get(shot_id)
        if shot_index is None:
            continue
        shot = shot_values[shot_index]
        explicit = shot.get("character_ids", shot.get("characters", []))
        if isinstance(explicit, str):
            explicit = [explicit]
        explicit_ids = {
            str(value.get("id") if isinstance(value, dict) else value)
            for value in (explicit if isinstance(explicit, list) else [])
            if value
        }
        relevant_character_ids = explicit_ids & reference_ids
        if not relevant_character_ids:
            inferred_ids = set(_characters_in_shot(shot, characters))
            relevant_character_ids = inferred_ids & reference_ids
        if not relevant_character_ids:
            # Legacy storyboards may not carry canonical IDs.  Reviewing every
            # available reference is safer than silently omitting identity QA.
            relevant_character_ids = set(reference_ids)

        local_records: list[dict[str, Any]] = []
        local_paths: list[Path] = []
        for record, path in zip(reference_manifest, reference_inputs):
            if str(record.get("character_id") or "") not in relevant_character_ids:
                continue
            local_paths.append(path)
            local_records.append({
                **record,
                "global_input_index": record["input_index"],
                "input_index": len(local_records) + 1,
            })
        local_paths.append(Path(board["path"]))
        local_board_record = {
            **board_record,
            "global_input_index": board_record["input_index"],
            "input_index": len(local_records) + 1,
        }
        local_records.append(local_board_record)
        narrative_value = str(shot.get("storyboard_board") or "").strip()
        narrative_path = Path(narrative_value) if narrative_value else None
        if narrative_path is not None and not narrative_path.is_absolute():
            narrative_path = grid_path.parent / narrative_path
        local_guide_record = None
        if (
            narrative_path is not None
            and narrative_path.is_file()
            and narrative_path.stat().st_size > 0
        ):
            local_paths.append(narrative_path)
            local_guide_record = {
                "input_index": len(local_records) + 1,
                "kind": "storyboard_narrative_guide",
                "shot_id": shot_id,
                "storyboard_ids": list(board["storyboard_ids"]),
                "path": str(narrative_path),
                "sha256": _sha256_file(narrative_path),
            }
            local_records.append(local_guide_record)
        local_paths.append(grid_path)
        local_overview_record = {
            **overview_manifest,
            "global_input_index": overview_manifest["input_index"],
            "input_index": len(local_records) + 1,
        }
        local_records.append(local_overview_record)

        context_start = max(0, shot_index - 1)
        context_end = min(len(shot_values), shot_index + 2)
        context_storyboard = {"shots": shot_values[context_start:context_end]}
        local_contracts = {
            character_id: canonical_contracts.get(character_id, "")
            for character_id in ordered_character_ids
            if character_id in relevant_character_ids
        }
        prompt = _l3_review_prompt(
            reference_inputs=[
                record
                for record in local_records
                if record["kind"] == "canonical_character_reference"
            ],
            storyboard_inputs=[
                local_board_record,
                *([local_guide_record] if local_guide_record is not None else []),
            ],
            overview_input=local_overview_record,
            canonical_contracts=local_contracts,
            storyboard=context_storyboard,
            visual_style=visual_style,
            valid_storyboard_ids=list(board["storyboard_ids"]),
        )
        request_id = f"l3-{shot_id.lower()}"
        requests.append({
            "request_id": request_id,
            "shot_id": shot_id,
            "context_shot_ids": [
                _shot_id(value, context_start + offset)
                for offset, value in enumerate(context_storyboard["shots"])
            ],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_count": len(local_records),
            "inputs": local_records,
        })
        executions.append({
            "request_id": request_id,
            "shot_id": shot_id,
            "storyboard_ids": list(board["storyboard_ids"]),
            "prompt": prompt,
            "input_paths": local_paths,
            "reference_manifest": [
                record
                for record in local_records
                if record["kind"] == "canonical_character_reference"
            ],
            "canonical_contracts": local_contracts,
        })

    input_manifest_path = _write_l3_batched_input_manifest(
        input_manifest_path or grid_path.parent / "storyboard_qa_inputs.json",
        input_records,
        requests,
    )
    if client is None and not os.environ.get("ARK_AGENT_API_KEY"):
        shot_ids = sorted({_parent_shot_id(value) for value in valid_storyboard_ids})
        return [
            _issue(
                "L3",
                "severe",
                "storyboard_visual_review_unavailable",
                "Canonical storyboard visual review is unavailable; paid video generation is blocked",
                shot_ids,
            )
        ], {
            "status": "error",
            "grid_path": str(grid_path),
            "input_manifest_path": str(input_manifest_path),
            "input_count": len(input_records),
            "provider_input_count": sum(
                int(request["input_count"]) for request in requests
            ),
            "request_count": len(requests),
            "provider_request_count": 0,
            "skipped_reason": "ARK multimodal API key missing",
        }

    from clients.ark_multimodal_client import review_as
    from schemas.understanding import StoryboardVisualUnderstanding

    review_client = client or ArkMultimodalClient()
    from quality.visual_qa_policy import (
        POLICY_ID,
        decide_visual_qa,
        policy_sha256,
    )
    from runtime.qa_ledger import QALedger, observation_fingerprint

    ledger_root = grid_path.parent
    ledger = QALedger(ledger_root / "runtime.db")
    run_manifest_path = ledger_root / "RUN_MANIFEST.json"
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        ledger_run_id = str(run_manifest["run_fingerprint"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        ledger_run_id = ledger_root.name
    contract_path = ledger_root / "CANONICAL_VISUAL_CONTRACT.json"
    try:
        canonical_contract_sha256 = str(
            json.loads(contract_path.read_text(encoding="utf-8"))["contract_sha256"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        canonical_contract_sha256 = hashlib.sha256(
            json.dumps(
                canonical_contracts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    evaluator_model = str(getattr(review_client, "model", "unknown-vlm"))
    issues: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    raw_issue_count = 0
    filtered_non_issue_count = 0
    failed_batches = 0
    provider_request_count = 0
    new_observation_count = 0
    for execution in executions:
        try:
            prompt_sha256 = hashlib.sha256(
                execution["prompt"].encode("utf-8")
            ).hexdigest()
            evidence = [
                {
                    "input_index": index,
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for index, path in enumerate(execution["input_paths"], 1)
            ]
            evidence_fingerprint = observation_fingerprint(
                evidence=evidence,
                canonical_contract_sha256=canonical_contract_sha256,
                evaluator_model=evaluator_model,
                prompt_sha256=prompt_sha256,
                observation_schema="StoryboardVisualUnderstanding.v1",
            )
            stored_observation = ledger.find_observation(evidence_fingerprint)
            if stored_observation is not None:
                typed_review = StoryboardVisualUnderstanding.model_validate(
                    stored_observation.observation
                )
                review_execution = {
                    "status": "reused_qa_observation",
                    "attempt_count": 0,
                    "provider_request_count": 0,
                }
                observation = stored_observation
                observation_reused = True
            else:
                if _acceptance_before_provider_request is not None:
                    _acceptance_before_provider_request({
                        "provider_family": "multimodal_observation",
                        "phase": "phase5",
                        "resource_id": execution["shot_id"],
                        "model": evaluator_model,
                        "observation_schema": (
                            "StoryboardVisualUnderstanding.v1"
                        ),
                        "evidence_fingerprint": evidence_fingerprint,
                        "prompt_sha256": prompt_sha256,
                        "inputs": evidence,
                    })
                typed_review, review_execution = execute_structured_understanding(
                    lambda execution=execution: review_as(
                        review_client,
                        execution["input_paths"],
                        execution["prompt"],
                        StoryboardVisualUnderstanding,
                    ),
                    max_attempts=structured_understanding_max_attempts,
                )
                provider_request_count += int(
                    review_execution.get("provider_request_count")
                    or review_execution.get("attempt_count")
                    or 1
                )
                observation, observation_reused = ledger.record_observation(
                    run_id=ledger_run_id,
                    phase="phase5",
                    resource_id=execution["shot_id"],
                    evidence_fingerprint=evidence_fingerprint,
                    canonical_contract_sha256=canonical_contract_sha256,
                    evaluator_model=evaluator_model,
                    prompt_sha256=prompt_sha256,
                    observation_schema="StoryboardVisualUnderstanding.v1",
                    observation=typed_review.model_dump(mode="json"),
                )
                new_observation_count += 1
            parsed = typed_review.model_dump(mode="json")
            policy_findings = [
                {
                    "blocking_category": {
                        "R1": "character_identity",
                        "R2": "visual_continuity",
                        "R3": "story_action",
                        "R4": "end_state",
                    }.get(str(value.get("red_line") or "").upper(), ""),
                    "confidence": value.get("confidence"),
                    "evidence": value.get("panel_evidence")
                    or (
                        f"expected={value.get('expected')}; "
                        f"observed={value.get('observed')}"
                        if value.get("expected") and value.get("observed")
                        else ""
                    ),
                    "finding": value,
                }
                for value in parsed["issues"]
            ]
            policy_decision = decide_visual_qa(
                semantic_score=1.0,
                findings=policy_findings,
            )
            decision, decision_reused = ledger.record_decision(
                observation_id=observation.observation_id,
                phase_owner="phase5.storyboard_qa",
                policy_id=POLICY_ID,
                policy_sha256=policy_sha256(),
                verdict=policy_decision.verdict,
                semantic_score=policy_decision.semantic_score,
                decision=policy_decision.as_dict(),
            )
            if (
                _acceptance_max_new_observations is not None
                and not observation_reused
                and new_observation_count
                >= _acceptance_max_new_observations
            ):
                raise QAObservationGatePaused(
                    limit=_acceptance_max_new_observations,
                    observation_id=observation.observation_id,
                    decision_id=decision.decision_id,
                    verdict=decision.verdict,
                    provider_request_count=provider_request_count,
                )
            raw_issue_count += len(parsed["issues"])
            batch = {
                "request_id": execution["request_id"],
                "shot_id": execution["shot_id"],
                "status": "completed",
                "input_count": len(execution["input_paths"]),
                "raw_issue_count": len(parsed["issues"]),
                "structured_review_execution": review_execution,
                "qa_observation_id": observation.observation_id,
                "qa_observation_reused": observation_reused,
                "qa_decision_id": decision.decision_id,
                "qa_decision_reused": decision_reused,
                "qa_verdict": decision.verdict,
            }
            batches.append(batch)
            valid_ids = set(execution["storyboard_ids"])
            for value in parsed["issues"]:
                if not isinstance(value, dict):
                    continue
                if _is_affirmative_non_issue(value):
                    filtered_non_issue_count += 1
                    continue
                red_line = str(value.get("red_line", "semantic_review"))
                message = str(value.get("message", "Multimodal review issue"))
                requested_severity = (
                    value.get("severity")
                    if value.get("severity") in {"severe", "moderate", "minor"}
                    else "moderate"
                )
                severity = _calibrate_l3_severity(
                    red_line, requested_severity, message
                )
                storyboard_ids = [
                    sid for sid in value.get("shot_ids", []) if sid in valid_ids
                ]
                evidence_valid, evidence_details = (
                    _r1_attribute_evidence(
                        value,
                        storyboard_ids,
                        {
                            int(item["input_index"]): str(item["character_id"])
                            for item in execution["reference_manifest"]
                        },
                        execution["canonical_contracts"],
                    )
                    if red_line.upper() == "R1"
                    else (True, {"evidence_status": "not_required"})
                )
                if not evidence_valid:
                    severity = "minor"
                shot_ids = sorted({
                    _parent_shot_id(storyboard_id)
                    for storyboard_id in storyboard_ids
                })
                correction_evidence = {
                    "storyboard_ids": storyboard_ids,
                    "mismatch_type": str(value.get("mismatch_type") or "other"),
                    "expected": str(value.get("expected") or "").strip(),
                    "observed": str(value.get("observed") or "").strip(),
                    "confidence": value.get("confidence"),
                    "panel_evidence": value.get("panel_evidence") or [],
                }
                correction_evidence.update(evidence_details)
                issues.append(_issue(
                    "L3", severity, red_line, message, shot_ids,
                    **correction_evidence,
                ))
        except QAObservationGatePaused:
            raise
        except Exception as exc:
            failed_batches += 1
            batch = {
                "request_id": execution["request_id"],
                "shot_id": execution["shot_id"],
                "status": "error",
                "input_count": len(execution["input_paths"]),
                "raw_issue_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, StructuredUnderstandingExhausted):
                batch["structured_review_execution"] = exc.receipt
            batches.append(batch)
            issues.append(_issue(
                "L3",
                "severe",
                "storyboard_visual_review_unavailable",
                f"Canonical storyboard visual review failed: {exc}",
                [execution["shot_id"]],
            ))

    layer = {
        "status": "error" if failed_batches else "completed",
        "grid_path": str(grid_path),
        "input_manifest_path": str(input_manifest_path),
        "input_count": len(input_records),
        "provider_input_count": sum(len(value["input_paths"]) for value in executions),
        "request_count": len(executions),
        "provider_request_count": provider_request_count,
        "raw_issue_count": raw_issue_count,
        "filtered_non_issue_count": filtered_non_issue_count,
        "accepted_issue_count": len(issues),
        "structured_review_batches": batches,
    }
    if len(batches) == 1 and "structured_review_execution" in batches[0]:
        layer["structured_review_execution"] = batches[0][
            "structured_review_execution"
        ]
    if failed_batches:
        layer["skipped_reason"] = (
            f"multimodal review unavailable for {failed_batches} "
            "shot-scoped batch(es)"
        )
    return issues, layer


def is_blocking_issue(issue: dict) -> bool:
    """Return whether an issue must stop paid video generation."""
    layer = str(issue.get("layer") or "")
    code = str(issue.get("code") or "")
    if code.endswith("_review_unavailable"):
        return True
    if layer not in {"L3", "L4"}:
        return issue.get("severity") == "severe"
    details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
    try:
        confidence = float(details.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    concrete_evidence = bool(
        str(details.get("expected") or "").strip()
        and str(details.get("observed") or "").strip()
        and (details.get("panel_evidence") or details.get("frame_evidence"))
    )
    blocking_category = bool(
        (layer == "L3" and code.upper() in {"R1", "R2", "R3", "R4"})
        or (layer == "L4" and code.startswith("first_frame_"))
    )
    from quality.visual_qa_policy import NEGATIVE_BLOCK_THRESHOLD

    return bool(
        blocking_category
        and concrete_evidence
        and confidence >= NEGATIVE_BLOCK_THRESHOLD
        and details.get("evidence_status") != "unverified"
    )


def blocking_issues(issues: list[dict]) -> list[dict]:
    """Return deterministic errors and evidenced high-confidence negatives."""
    return [issue for issue in issues if is_blocking_issue(issue)]


def grade_issues(issues: list[dict]) -> str:
    blocking = blocking_issues(issues)
    severe = sum(issue.get("severity") == "severe" for issue in blocking)
    moderate = sum(issue.get("severity") == "moderate" for issue in issues)
    if severe >= 3 or len(blocking) >= 3:
        return "D"
    if blocking:
        return "C"
    if moderate > 2:
        return "B"
    return "A"


def run_storyboard_qa_gate(
    output_dir: Path,
    similarity_threshold: float | None = None,
    embedder: Callable[[str], list[float] | None] | None = None,
    multimodal_client: ArkMultimodalClient | None = None,
    *,
    structured_understanding_max_attempts: int = (
        DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS
    ),
    _acceptance_max_new_observations: int | None = None,
    _acceptance_before_provider_request: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> dict:
    """Run all QA layers, persist the report, and return a phase result."""
    output_dir = Path(output_dir)
    report_path = output_dir / "storyboard_qa_report.json"
    try:
        storyboard = json.loads((output_dir / "STORYBOARD.json").read_text(encoding="utf-8"))
        characters_path = output_dir / "CHARACTERS.json"
        characters = json.loads(characters_path.read_text(encoding="utf-8")) if characters_path.is_file() else {"characters": []}
        from utils.canonical_visual_contracts import (
            load_canonical_visual_contract,
        )

        visual_contract = load_canonical_visual_contract(
            output_dir,
            characters_data=characters,
        )
        if (
            storyboard.get("canonical_visual_contract_sha256")
            != visual_contract["contract_sha256"]
        ):
            raise ValueError("storyboard canonical visual contract hash mismatch")
        style_path = output_dir / "visual-style.md"
        visual_style = style_path.read_text(encoding="utf-8") if style_path.is_file() else ""
        events_path = output_dir / "phase1_events.json"
        events_data = (
            json.loads(events_path.read_text(encoding="utf-8"))
            if events_path.is_file()
            else None
        )
        screenplay_plan_path = output_dir / "SCREENPLAY_PLAN.json"
        screenplay_plan = (
            migrate_screenplay_plan(
                json.loads(screenplay_plan_path.read_text(encoding="utf-8"))
            )
            if screenplay_plan_path.is_file()
            else None
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result = {"status": "error", "grade": "D", "gate_passed": False, "error": f"required artifact unreadable: {exc}", "issues": [_issue("L1", "severe", "artifact_unreadable", str(exc))], "failed_shot_ids": []}
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    beat_images = find_storyboard_beat_images(output_dir, storyboard)
    cinematic_images = find_cinematic_first_frame_images(output_dir, storyboard)
    expected_beats = [
        str(beat.get("beat_id") or f"{_shot_id(shot, shot_index)}_P{position:02d}")
        for shot_index, shot in enumerate(storyboard.get("shots", []))
        if isinstance(shot, dict)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1)
        if isinstance(beat, dict)
    ]
    images = beat_images or find_storyboard_images(output_dir, storyboard)
    l1_issues, per_shot = run_l1_checks(storyboard, visual_style)
    # These checks inspect only storyboard metadata, so they belong before the
    # paid video boundary. Running them in Phase 7 used to discover an
    # unfixable storyboard defect only after Phase 6 had spent quota.
    from quality.slideshow_risk import score_slideshow_risk
    from quality.variation_checker import check_scene_variation

    scenes = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    variation = check_scene_variation(scenes)
    slideshow = score_slideshow_risk(scenes)
    variation_quality = round(5.0 - float(variation.get("score", 5.0)), 2)
    slideshow_risk = round(float(slideshow.get("average", 5.0)) / 5.0, 3)
    (output_dir / "variation_report.json").write_text(
        json.dumps(variation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "slideshow_risk_report.json").write_text(
        json.dumps(slideshow, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    structural_issues = []
    if variation_quality < 3.0:
        structural_issues.append(
            _issue(
                "L1",
                "severe",
                "scene_variation_insufficient",
                f"Storyboard variation quality {variation_quality:g}/5 requires revision",
                details={"violations": variation.get("violations", [])},
            )
        )
    if slideshow_risk > 0.7:
        structural_issues.append(
            _issue(
                "L1",
                "severe",
                "slideshow_risk_high",
                f"Storyboard slideshow risk {slideshow_risk:.3f} exceeds 0.7",
                details={"dimensions": slideshow.get("dimensions", {})},
            )
        )
    threshold = similarity_threshold if similarity_threshold is not None else float(os.environ.get("HONCUT_STORYBOARD_QA_SIMILARITY", DEFAULT_SIMILARITY_THRESHOLD))
    l2_issues, l2 = run_l2_checks(storyboard, characters, images, threshold, embedder)
    character_reference_images = find_character_reference_images(
        output_dir,
        characters,
    )
    l3_issues, l3 = run_l3_review(
        storyboard,
        characters,
        visual_style,
        images,
        output_dir / "storyboard_qa_grid.jpg",
        multimodal_client,
        character_reference_images=character_reference_images,
        structured_understanding_max_attempts=(
            structured_understanding_max_attempts
        ),
        _acceptance_max_new_observations=(
            _acceptance_max_new_observations
        ),
        _acceptance_before_provider_request=(
            _acceptance_before_provider_request
        ),
    )
    l4_issues, l4 = run_l4_first_frame_review(
        storyboard,
        visual_style,
        cinematic_images,
        output_dir,
        multimodal_client,
    )
    capacity_issues = run_generation_capacity_checks(
        storyboard,
        events_data,
        screenplay_plan,
    )
    artifact_issues = [
        _issue(
            "L1", "severe", "storyboard_beat_image_missing",
            f"{beat_id} has no valid Phase 2 storyboard image",
            [_parent_shot_id(beat_id)], beat_id=beat_id,
        )
        for beat_id in expected_beats
        if beat_id not in beat_images
    ]
    from phases.phase2.shot_storyboards import validate_shot_storyboard_artifacts

    narrative_guide_errors = [
        error
        for error in validate_shot_storyboard_artifacts(output_dir, storyboard)
        if any(
            marker in error.casefold()
            for marker in ("narrative guide", "nine-grid", "gxx", "3x3")
        )
    ]
    narrative_guide_issues = [
        _issue(
            "L3",
            "severe",
            "storyboard_narrative_guide_invalid",
            error,
            sorted(
                {
                    match.group(1)
                    for match in [re.match(r"^(S[^ _]+)", error)]
                    if match
                }
            ),
        )
        for error in narrative_guide_errors
    ]
    declared_cinematic_ids = [
        str(beat.get("beat_id") or f"{_shot_id(shot, shot_index)}_P{position:02d}")
        for shot_index, shot in enumerate(storyboard.get("shots", []))
        if isinstance(shot, dict)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1)
        if isinstance(beat, dict) and beat.get("video_first_frame_kind")
    ]
    cinematic_artifact_issues = [
        _issue(
            "L4",
            "severe",
            "cinematic_first_frame_missing",
            f"{frame_id} has no valid Phase 4 cinematic first frame",
            [_parent_shot_id(frame_id)],
            frame_id=frame_id,
        )
        for frame_id in declared_cinematic_ids
        if frame_id not in cinematic_images
    ]
    if declared_cinematic_ids:
        from phases.phase4.cinematic_first_frames import (
            validate_cinematic_first_frame_artifacts,
        )

        for error in validate_cinematic_first_frame_artifacts(output_dir, storyboard):
            matched = re.match(r"^(S[^ ]+)", error)
            resource_id = matched.group(1) if matched else ""
            cinematic_artifact_issues.append(
                _issue(
                    "L4",
                    "severe",
                    "cinematic_first_frame_provenance_invalid",
                    error,
                    [_parent_shot_id(resource_id)] if resource_id else [],
                    resource_id=resource_id or None,
                )
            )
    issues = (
        l1_issues
        + structural_issues
        + artifact_issues
        + narrative_guide_issues
        + cinematic_artifact_issues
        + capacity_issues
        + l2_issues
        + l3_issues
        + l4_issues
    )
    for index, shot in enumerate(storyboard.get("shots", [])):
        sid = _shot_id(shot, index)
        detail = per_shot.setdefault(sid, {"issues": []})
        detail["characters"] = _characters_in_shot(shot, characters.get("characters", []))
        shot_beat_images = {
            image_id: str(path)
            for image_id, path in beat_images.items()
            if _parent_shot_id(image_id) == sid
        }
        detail["image_path"] = str(images[sid]) if sid in images else None
        detail["storyboard_beat_images"] = shot_beat_images
        detail["cinematic_first_frames"] = {
            image_id: str(path)
            for image_id, path in cinematic_images.items()
            if _parent_shot_id(image_id) == sid
        }
        detail["issues"] = [issue for issue in issues if sid in issue.get("shot_ids", [])]
    grade = grade_issues(issues)
    failed_shots = sorted({
        sid
        for issue in blocking_issues(issues)
        for sid in issue.get("shot_ids", [])
    })
    report = {"status": "done" if grade in {"A", "B"} else "error", "grade": grade, "gate_passed": grade in {"A", "B"}, "issues": issues, "issue_counts": {severity: sum(item.get("severity") == severity for item in issues) for severity in ("severe", "moderate", "minor")}, "failed_shot_ids": failed_shots, "shots": per_shot, "variation_score": variation_quality, "slideshow_risk": slideshow_risk, "layers": {"L1": {"status": "completed"}, "L2": l2, "L3": l3, "L4": l4}, "outputs": ["storyboard_qa_report.json", "variation_report.json", "slideshow_risk_report.json", *( ["storyboard_qa_grid.jpg"] if grid_path_exists(output_dir) else []), *( ["storyboard_qa_inputs.json"] if (output_dir / "storyboard_qa_inputs.json").is_file() else []), *( ["first_frame_qa_inputs.json"] if (output_dir / "first_frame_qa_inputs.json").is_file() else [])]}
    if not report["gate_passed"]:
        report["error"] = f"Storyboard QA grade {grade} blocks Phase 6; redraw only failed_shot_ids"
        if any(issue.get("layer") == "L4" for issue in blocking_issues(issues)):
            report["recommended_restart_phase"] = "phase4"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _storyboard_adjudication_scope(
    storyboard: dict[str, Any],
    storyboard_ids: list[str],
) -> dict[str, Any]:
    """Project canonical story data onto only the disputed Pxx evidence."""
    targets = set(storyboard_ids)
    scoped_shots: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(storyboard.get("shots") or []):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        scoped_beats = []
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
            if beat_id in targets:
                scoped_beats.append(copy.deepcopy(beat))
        if not scoped_beats:
            continue
        scoped_shot = copy.deepcopy(shot)
        scoped_shot["storyboard_beats"] = scoped_beats
        scoped_shots.append(scoped_shot)
    return {"shots": scoped_shots}


def _adjudication_unavailable_report(
    storyboard_ids: list[str],
    error: BaseException | str,
) -> dict[str, Any]:
    safe_error = redact_text(str(error))
    shot_ids = sorted({_parent_shot_id(value) for value in storyboard_ids})
    issue = _issue(
        "L3",
        "severe",
        "storyboard_visual_review_unavailable",
        f"Narrow storyboard adjudication is unavailable: {safe_error}",
        shot_ids,
    )
    return {
        "status": "error",
        "grade": "D",
        "gate_passed": False,
        "issues": [issue],
        "failed_shot_ids": shot_ids,
        "layers": {
            "L3": {
                "status": "error",
                "request_count": 0,
                "provider_request_count": 0,
                "error": safe_error,
            }
        },
    }


def _run_storyboard_adjudication_review(
    output_dir: Path,
    storyboard_ids: list[str],
    *,
    multimodal_client: ArkMultimodalClient | None = None,
    structured_understanding_max_attempts: int = (
        DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS
    ),
) -> dict[str, Any]:
    """Run one strict L3-only review against the exact disputed Pxx pixels."""
    output_dir = Path(output_dir)
    targets = sorted(dict.fromkeys(str(value) for value in storyboard_ids if value))
    try:
        storyboard = json.loads(
            (output_dir / "STORYBOARD.json").read_text(encoding="utf-8")
        )
        characters_path = output_dir / "CHARACTERS.json"
        characters = (
            json.loads(characters_path.read_text(encoding="utf-8"))
            if characters_path.is_file()
            else {"characters": []}
        )
        style_path = output_dir / "visual-style.md"
        visual_style = (
            style_path.read_text(encoding="utf-8") if style_path.is_file() else ""
        )
        scoped_storyboard = _storyboard_adjudication_scope(storyboard, targets)
        images = {
            storyboard_id: path
            for storyboard_id, path in find_storyboard_beat_images(
                output_dir, storyboard
            ).items()
            if storyboard_id in set(targets)
        }
        scoped_ids = {
            str(beat.get("beat_id") or "")
            for shot in scoped_storyboard.get("shots") or []
            for beat in shot.get("storyboard_beats") or []
            if isinstance(beat, dict)
        }
        if not targets or scoped_ids != set(targets) or set(images) != set(targets):
            raise ValueError(
                "disputed storyboard IDs do not resolve to exact canonical Pxx pixels"
            )
        character_reference_images = find_character_reference_images(
            output_dir,
            characters,
        )
        issues, layer = run_l3_review(
            scoped_storyboard,
            characters,
            visual_style,
            images,
            output_dir / "phase5_review_adjudication_grid.jpg",
            multimodal_client,
            character_reference_images=character_reference_images,
            evidence_dir=output_dir / "phase5_review_adjudication_evidence",
            input_manifest_path=(
                output_dir / "phase5_review_adjudication_inputs.json"
            ),
            structured_understanding_max_attempts=(
                structured_understanding_max_attempts
            ),
        )
        grade = grade_issues(issues)
        gate_passed = grade in {"A", "B"} and layer.get("status") == "completed"
        report = {
            "status": "done" if gate_passed else "error",
            "grade": grade,
            "gate_passed": gate_passed,
            "scope": "unchanged_panel_adjudication_l3_only",
            "storyboard_ids": targets,
            "issues": issues,
            "failed_shot_ids": sorted({
                shot_id
                for issue in blocking_issues(issues)
                for shot_id in issue.get("shot_ids") or []
            }),
            "layers": {"L3": layer},
            "outputs": [
                "phase5_review_adjudication_confirmation.json",
                "phase5_review_adjudication_inputs.json",
                "phase5_review_adjudication_grid.jpg",
            ],
        }
        if not gate_passed:
            report["error"] = (
                "Narrow Phase 5 L3 adjudication did not produce a passing verdict"
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = _adjudication_unavailable_report(targets, exc)
    _atomic_json(
        output_dir / "phase5_review_adjudication_confirmation.json",
        report,
    )
    return report


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _phase5_input_artifacts(output_dir: Path) -> list[dict[str, str]]:
    """Hash only run-local Phase 5 inputs; never embed prompts or credentials."""
    artifacts = []
    for name in (
        "STORYBOARD.json",
        "CHARACTERS.json",
        "visual-style.md",
        "phase1_events.json",
    ):
        path = output_dir / name
        if path.is_file():
            artifacts.append({"path": name, "sha256": _sha256_file(path)})
    return artifacts


def _dry_run_layer_receipt(layer: str) -> dict[str, str]:
    return {
        "status": "skipped",
        "skipped_reason": (
            f"{layer} requires production image or model evidence and is disabled "
            "during dry-run"
        ),
    }


def _run_storyboard_qa_dry_run(output_dir: Path) -> dict[str, Any]:
    """Run deterministic metadata checks without touching image/model owners."""
    output_dir = Path(output_dir)
    input_artifacts = _phase5_input_artifacts(output_dir)
    try:
        storyboard = json.loads(
            (output_dir / "STORYBOARD.json").read_text(encoding="utf-8")
        )
        characters_path = output_dir / "CHARACTERS.json"
        characters = (
            json.loads(characters_path.read_text(encoding="utf-8"))
            if characters_path.is_file()
            else {"characters": []}
        )
        style_path = output_dir / "visual-style.md"
        visual_style = (
            style_path.read_text(encoding="utf-8") if style_path.is_file() else ""
        )
        events_path = output_dir / "phase1_events.json"
        events_data = (
            json.loads(events_path.read_text(encoding="utf-8"))
            if events_path.is_file()
            else None
        )
        screenplay_plan_path = output_dir / "SCREENPLAY_PLAN.json"
        screenplay_plan = (
            migrate_screenplay_plan(
                json.loads(screenplay_plan_path.read_text(encoding="utf-8"))
            )
            if screenplay_plan_path.is_file()
            else None
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issue = _issue(
            "L1",
            "severe",
            "artifact_unreadable",
            f"required artifact unreadable: {exc}",
        )
        variation = {"status": "skipped", "reason": "required artifact unreadable"}
        slideshow = {"status": "skipped", "reason": "required artifact unreadable"}
        issues = [issue]
        per_shot: dict[str, dict[str, Any]] = {}
        variation_quality = 0.0
        slideshow_risk = 1.0
    else:
        l1_issues, per_shot = run_l1_checks(storyboard, visual_style)
        capacity_issues = run_generation_capacity_checks(
            storyboard,
            events_data,
            screenplay_plan,
        )

        from quality.slideshow_risk import score_slideshow_risk
        from quality.variation_checker import check_scene_variation

        scenes = [
            shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)
        ]
        variation = check_scene_variation(scenes)
        slideshow = score_slideshow_risk(scenes)
        variation_quality = round(5.0 - float(variation.get("score", 5.0)), 2)
        slideshow_risk = round(float(slideshow.get("average", 5.0)) / 5.0, 3)
        structural_issues = []
        if variation_quality < 3.0:
            structural_issues.append(
                _issue(
                    "L1",
                    "severe",
                    "scene_variation_insufficient",
                    "Storyboard variation quality "
                    f"{variation_quality:g}/5 requires revision",
                    details={"violations": variation.get("violations", [])},
                )
            )
        if slideshow_risk > 0.7:
            structural_issues.append(
                _issue(
                    "L1",
                    "severe",
                    "slideshow_risk_high",
                    f"Storyboard slideshow risk {slideshow_risk:.3f} exceeds 0.7",
                    details={"dimensions": slideshow.get("dimensions", {})},
                )
            )
        issues = l1_issues + structural_issues + capacity_issues
        character_list = characters.get("characters", [])
        for index, shot in enumerate(storyboard.get("shots", [])):
            if not isinstance(shot, dict):
                continue
            shot_id = _shot_id(shot, index)
            detail = per_shot.setdefault(shot_id, {})
            detail["characters"] = _characters_in_shot(shot, character_list)
            detail["image_path"] = None
            detail["storyboard_beat_images"] = {}
            detail["cinematic_first_frames"] = {}
            detail["issues"] = [
                issue for issue in issues if shot_id in issue.get("shot_ids", [])
            ]

    _atomic_json(output_dir / "variation_report.json", variation)
    _atomic_json(output_dir / "slideshow_risk_report.json", slideshow)
    grade = grade_issues(issues)
    gate_passed = grade in {"A", "B"}
    failed_shots = sorted(
        {
            shot_id
            for issue in blocking_issues(issues)
            for shot_id in issue.get("shot_ids", [])
        }
    )
    issue_counts = {
        severity: sum(item.get("severity") == severity for item in issues)
        for severity in ("severe", "moderate", "minor")
    }
    outputs = [
        "storyboard_qa_report.json",
        "variation_report.json",
        "slideshow_risk_report.json",
        PHASE5_DRY_RUN_RECEIPT_NAME,
    ]
    layers = {
        "L1": {
            "status": "completed",
            "checks": [
                "metadata",
                "generation_capacity",
                "scene_variation",
                "slideshow_risk",
            ],
        },
        "L2": _dry_run_layer_receipt("L2 embedding review"),
        "L3": _dry_run_layer_receipt("L3 multimodal review"),
        "L4": _dry_run_layer_receipt("L4 cinematic first-frame review"),
    }
    receipt = {
        "schema": PHASE5_DRY_RUN_RECEIPT_SCHEMA,
        "status": "completed" if gate_passed else "blocked",
        "dry_run": True,
        "evidence_scope": "dry_run_structural_only",
        "grade_scope": "structural_metadata_only",
        "input_artifacts": input_artifacts,
        "grade": grade,
        "gate_passed": gate_passed,
        "production_gate_passed": False,
        "issues": issues,
        "issue_counts": issue_counts,
        "failed_shot_ids": failed_shots,
        "variation_score": variation_quality,
        "slideshow_risk": slideshow_risk,
        "layers": layers,
        "skipped_operations": list(PHASE5_DRY_RUN_SKIPPED_OPERATIONS),
        "outputs": outputs,
    }
    _atomic_json(output_dir / PHASE5_DRY_RUN_RECEIPT_NAME, receipt)
    report = {
        "schema": "honcut.storyboard-qa-report.v1",
        "status": "done" if gate_passed else "error",
        "grade": grade,
        "gate_passed": gate_passed,
        "production_gate_passed": False,
        "dry_run": True,
        "evidence_scope": "dry_run_structural_only",
        "grade_scope": "structural_metadata_only",
        "dry_run_receipt": PHASE5_DRY_RUN_RECEIPT_NAME,
        "input_artifacts": input_artifacts,
        "issues": issues,
        "issue_counts": issue_counts,
        "failed_shot_ids": failed_shots,
        "shots": per_shot,
        "variation_score": variation_quality,
        "slideshow_risk": slideshow_risk,
        "layers": layers,
        "skipped_operations": list(PHASE5_DRY_RUN_SKIPPED_OPERATIONS),
        "outputs": outputs,
    }
    if not gate_passed:
        report["error"] = (
            f"Storyboard dry-run structural QA grade {grade} blocks Phase 6"
        )
    _atomic_json(output_dir / "storyboard_qa_report.json", report)
    return report


def _resolved_correction_attempts(value: int | None) -> int:
    raw: Any = (
        os.environ.get(
            "HONCUT_PHASE5_MAX_CORRECTIONS",
            str(DEFAULT_MAX_CORRECTION_ATTEMPTS),
        )
        if value is None
        else value
    )
    if isinstance(raw, bool):
        raise ValueError("Phase 5 correction attempts must be an integer")
    try:
        attempts = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 5 correction attempts must be an integer") from exc
    if attempts < 0 or attempts > MAX_CORRECTION_ATTEMPTS:
        raise ValueError(
            f"Phase 5 correction attempts must be between 0 and {MAX_CORRECTION_ATTEMPTS}"
        )
    return attempts


def _correctable_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only blocking visual issues that a storyboard redraw can fix."""
    result: list[dict[str, Any]] = []
    for issue in blocking_issues(report.get("issues") or []):
        if not issue.get("shot_ids"):
            continue
        layer = str(issue.get("layer") or "").upper()
        code = str(issue.get("code") or "").upper()
        if (
            layer == "L3"
            and code in {"R1", "R2", "R3", "R4"}
            and _correctable_storyboard_ids([issue])
        ):
            result.append(issue)
    return result


def _correctable_cinematic_frame_ids(
    issues: list[dict[str, Any]],
) -> list[str]:
    """Return exact evidence-backed first frames safe for local regeneration."""
    frame_ids: list[str] = []
    for issue in issues:
        if str(issue.get("layer") or "").upper() != "L4":
            continue
        if str(issue.get("code") or "").casefold() not in {
            "first_frame_annotation_contamination",
            "first_frame_subject_duplication",
        }:
            continue
        details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
        if details.get("evidence_status") != "validated":
            continue
        raw_ids = details.get("frame_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        evidence = details.get("frame_evidence") or []
        if not isinstance(evidence, list):
            continue
        evidence_by_id = {
            str(item.get("frame_id") or "").strip(): str(item.get("observed") or "").strip()
            for item in evidence
            if isinstance(item, dict)
            and str(item.get("frame_id") or "").strip()
            and str(item.get("observed") or "").strip()
        }
        candidate_ids = [
            str(value or "").strip()
            for value in raw_ids
            if re.fullmatch(r"S[^\s]+_P\d+", str(value or "").strip(), re.IGNORECASE)
        ]
        if not candidate_ids or not set(candidate_ids).issubset(evidence_by_id):
            continue
        if any(
            _is_affirmative_non_issue({"message": evidence_by_id[frame_id]})
            for frame_id in candidate_ids
        ):
            continue
        for frame_id in candidate_ids:
            if frame_id not in frame_ids:
                frame_ids.append(frame_id)
    return sorted(frame_ids)


def _correctable_cinematic_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only local L4 contamination with complete per-frame evidence."""
    return [
        issue
        for issue in blocking_issues(report.get("issues") or [])
        if _correctable_cinematic_frame_ids([issue])
    ]


def _correctable_storyboard_ids(
    issues: list[dict[str, Any]],
) -> list[str]:
    """Return evidence-backed Pxx IDs without affirmative observations.

    The visual reviewer may attach one shot-level finding to multiple Pxx
    panels.  A panel is redraw authority only when it has its own visible
    evidence and that evidence describes a mismatch rather than an explicit
    match.  This prevents a failing sibling panel from authorizing mutation of
    an already-correct asset.
    """
    storyboard_ids: list[str] = []
    for issue in issues:
        details = (
            issue.get("details")
            if isinstance(issue.get("details"), dict)
            else {}
        )
        raw_ids = details.get("storyboard_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        raw_evidence = details.get("panel_evidence") or []
        if not isinstance(raw_evidence, list):
            raw_evidence = []
        observations_by_id: dict[str, list[str]] = {}
        for evidence in raw_evidence:
            if not isinstance(evidence, dict):
                continue
            storyboard_id = str(evidence.get("shot_id") or "").strip()
            observed = str(evidence.get("observed") or "").strip()
            if storyboard_id and observed:
                observations_by_id.setdefault(storyboard_id, []).append(observed)
        for raw_id in raw_ids:
            storyboard_id = str(raw_id or "").strip()
            if "_P" not in storyboard_id:
                continue
            observations = observations_by_id.get(storyboard_id, [])
            if not observations:
                continue
            if all(
                _is_affirmative_non_issue({"message": observed})
                for observed in observations
            ):
                continue
            if storyboard_id not in storyboard_ids:
                storyboard_ids.append(storyboard_id)
    return sorted(storyboard_ids)


def _blocking_storyboard_issue_map(
    report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Index evidence-backed blocking findings by their canonical Pxx ID."""
    indexed: dict[str, list[dict[str, Any]]] = {}
    for issue in _correctable_issues(report):
        for storyboard_id in _correctable_storyboard_ids([issue]):
            indexed.setdefault(storyboard_id, []).append(issue)
    return indexed


def _storyboard_panel_hashes(output_dir: Path) -> dict[str, str]:
    """Hash the exact Pxx pixels reviewed by L3 without trusting sidecars."""
    try:
        storyboard = json.loads(
            (output_dir / "STORYBOARD.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return _storyboard_panel_hashes_for_storyboard(output_dir, storyboard)


def _storyboard_panel_hashes_for_storyboard(
    output_dir: Path,
    storyboard: dict[str, Any],
) -> dict[str, str]:
    """Hash Pxx evidence from an explicit root, including immutable archives."""
    return {
        storyboard_id: _sha256_file(path)
        for storyboard_id, path in find_storyboard_beat_images(
            output_dir, storyboard
        ).items()
    }


def _issue_for_storyboard_id(
    issue: dict[str, Any],
    storyboard_id: str,
) -> dict[str, Any] | None:
    """Narrow a multi-panel finding to one adjudicated Pxx."""
    if storyboard_id not in _correctable_storyboard_ids([issue]):
        return None
    narrowed = copy.deepcopy(issue)
    details = (
        narrowed.get("details")
        if isinstance(narrowed.get("details"), dict)
        else {}
    )
    details["storyboard_ids"] = [storyboard_id]
    details["panel_evidence"] = [
        value
        for value in (details.get("panel_evidence") or [])
        if isinstance(value, dict)
        and str(value.get("shot_id") or "").strip() == storyboard_id
    ]
    narrowed["details"] = details
    narrowed["shot_ids"] = [_parent_shot_id(storyboard_id)]
    return narrowed


def _recompute_qa_verdict(
    report: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild derived gate fields after evidence-level adjudication."""
    reconciled = copy.deepcopy(report)
    reconciled["issues"] = issues
    grade = grade_issues(issues)
    gate_passed = grade in {"A", "B"}
    reconciled["grade"] = grade
    reconciled["gate_passed"] = gate_passed
    reconciled["status"] = "done" if gate_passed else "error"
    reconciled["issue_counts"] = {
        severity: sum(issue.get("severity") == severity for issue in issues)
        for severity in ("severe", "moderate", "minor")
    }
    reconciled["failed_shot_ids"] = sorted({
        shot_id
        for issue in blocking_issues(issues)
        for shot_id in issue.get("shot_ids") or []
    })
    shots = reconciled.get("shots")
    if isinstance(shots, dict):
        for shot_id, detail in shots.items():
            if isinstance(detail, dict):
                detail["issues"] = [
                    issue
                    for issue in issues
                    if shot_id in (issue.get("shot_ids") or [])
                ]
    if gate_passed:
        reconciled.pop("error", None)
        reconciled.pop("recommended_restart_phase", None)
    else:
        reconciled["error"] = (
            f"Storyboard QA grade {grade} blocks Phase 6; "
            "redraw only failed_shot_ids"
        )
    return reconciled


class _ReviewAdjudicationBlocked(RuntimeError):
    """Carry the persisted fail-closed receipt to the Phase 5 correction owner."""

    def __init__(self, message: str, receipt: dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


def _adjudication_snapshot(
    output_dir: Path,
    name: str,
    report: dict[str, Any],
) -> dict[str, str]:
    path = output_dir / name
    _atomic_json(path, report)
    return {"path": name, "sha256": _sha256_file(path)}


def _adjudication_confirmation_audit(
    output_dir: Path,
    confirmation: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Keep request/error receipts while excluding prompts and raw responses."""
    layers = confirmation.get("layers")
    layer = layers.get("L3") if isinstance(layers, dict) else {}
    layer = layer if isinstance(layer, dict) else {}
    audit: dict[str, Any] = {
        key: copy.deepcopy(layer[key])
        for key in (
            "status",
            "request_count",
            "provider_request_count",
            "input_count",
            "provider_input_count",
            "raw_issue_count",
            "filtered_non_issue_count",
            "accepted_issue_count",
            "skipped_reason",
        )
        if key in layer
    }
    errors: list[str] = []
    for key in ("error", "skipped_reason"):
        value = layer.get(key)
        if value:
            errors.append(str(value))
    batches = []
    for batch in layer.get("structured_review_batches") or []:
        if not isinstance(batch, dict):
            continue
        selected = {
            key: copy.deepcopy(batch[key])
            for key in (
                "request_id",
                "shot_id",
                "status",
                "input_count",
                "raw_issue_count",
                "error",
                "structured_review_execution",
            )
            if key in batch
        }
        if selected.get("error"):
            errors.append(str(selected["error"]))
        batches.append(redact_for_log(selected))
    if batches:
        audit["structured_review_batches"] = batches
    manifest_value = layer.get("input_manifest_path")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        try:
            manifest_path.resolve().relative_to(output_dir.resolve())
        except (OSError, ValueError):
            manifest_path = Path()
        if manifest_path.is_file():
            audit["input_manifest"] = {
                "path": manifest_path.name,
                "sha256": _sha256_file(manifest_path),
            }
    if not errors and confirmation.get("error"):
        errors.append(str(confirmation["error"]))
    safe_error = redact_text("; ".join(errors) or "visual confirmation unavailable")
    return redact_for_log(audit), safe_error


def _write_review_adjudication_report(
    output_dir: Path,
    receipts: list[dict[str, Any]],
) -> None:
    status = str(receipts[-1].get("status") or "error") if receipts else "error"
    _atomic_json(
        output_dir / PHASE5_REVIEW_ADJUDICATION_REPORT_NAME,
        {
            "schema": PHASE5_REVIEW_ADJUDICATIONS_SCHEMA,
            "status": status,
            "adjudications": receipts,
        },
    )


def _pending_adjudication_receipt(
    output_dir: Path,
    previous: dict[str, Any],
    current: dict[str, Any],
    disputed: list[str],
    current_hashes: dict[str, str],
    *,
    status: str,
    safe_error: str,
    confirmation: dict[str, Any] | None,
    after_correction_attempt: int | None,
    legacy_reconstructed: bool = False,
) -> dict[str, Any]:
    previous_map = _blocking_storyboard_issue_map(previous)
    current_map = _blocking_storyboard_issue_map(current)
    confirmation_map = (
        _blocking_storyboard_issue_map(confirmation)
        if isinstance(confirmation, dict)
        else {}
    )
    confirmation_audit: dict[str, Any] = {}
    if isinstance(confirmation, dict):
        confirmation_audit, audited_error = _adjudication_confirmation_audit(
            output_dir, confirmation
        )
        safe_error = audited_error or safe_error
    receipt: dict[str, Any] = {
        "schema": PHASE5_REVIEW_ADJUDICATION_SCHEMA,
        "status": status,
        "reason": "unchanged_panel_verdict_flip",
        "storyboard_ids": disputed,
        "asset_sha256": {
            storyboard_id: current_hashes[storyboard_id]
            for storyboard_id in disputed
            if storyboard_id in current_hashes
        },
        "input_artifacts": _phase5_input_artifacts(output_dir),
        "votes": {
            storyboard_id: {
                "previous_blocked": storyboard_id in previous_map,
                "current_blocked": storyboard_id in current_map,
                "confirmation_blocked": (
                    storyboard_id in confirmation_map
                    if confirmation is not None
                    and status not in {"blocked_unavailable", "blocked_evidence_changed"}
                    else None
                ),
            }
            for storyboard_id in disputed
        },
        "decisions": {storyboard_id: "unresolved" for storyboard_id in disputed},
        "safe_error": redact_text(safe_error),
        "confirmation": confirmation_audit,
        "report_snapshots": {
            "previous": _adjudication_snapshot(
                output_dir,
                PHASE5_REVIEW_ADJUDICATION_PREVIOUS_REPORT_NAME,
                previous,
            ),
            "current": _adjudication_snapshot(
                output_dir,
                PHASE5_REVIEW_ADJUDICATION_CURRENT_REPORT_NAME,
                current,
            ),
        },
    }
    if after_correction_attempt is not None:
        receipt["after_correction_attempt"] = after_correction_attempt
    if legacy_reconstructed:
        receipt["legacy_reconstructed"] = True
    return receipt


def _adjudicate_unchanged_panel_flips(
    output_dir: Path,
    previous: dict[str, Any],
    previous_hashes: dict[str, str],
    current: dict[str, Any],
    current_hashes: dict[str, str],
    confirmation_runner: Callable[[Path, list[str]], dict[str, Any]],
    *,
    after_correction_attempt: int | None = None,
    prior_receipts: list[dict[str, Any]] | None = None,
    confirmation_allowed: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Use one bounded tie-break review when unchanged pixels flip verdict.

    A previous vote, current vote, and confirmation vote form a per-Pxx
    two-out-of-three decision.  Changed pixels are never adjudicated against
    their pre-redraw verdict because they are new evidence.
    """
    previous_map = _blocking_storyboard_issue_map(previous)
    current_map = _blocking_storyboard_issue_map(current)
    disputed = sorted(
        storyboard_id
        for storyboard_id in set(previous_map) ^ set(current_map)
        if previous_hashes.get(storyboard_id)
        and previous_hashes.get(storyboard_id) == current_hashes.get(storyboard_id)
    )
    if not disputed:
        return current, None

    if not confirmation_allowed:
        receipt = _pending_adjudication_receipt(
            output_dir,
            previous,
            current,
            disputed,
            current_hashes,
            status="blocked_exhausted",
            safe_error="Phase 5 review adjudication budget is exhausted",
            confirmation=None,
            after_correction_attempt=after_correction_attempt,
        )
        _write_review_adjudication_report(
            output_dir, [*(prior_receipts or []), receipt]
        )
        raise _ReviewAdjudicationBlocked(
            "Phase 5 review adjudication budget is exhausted",
            receipt,
        )

    try:
        confirmation = confirmation_runner(output_dir, disputed)
    except Exception as exc:
        confirmation = _adjudication_unavailable_report(
            disputed,
            f"{type(exc).__name__}: {redact_text(str(exc))}",
        )
    if not isinstance(confirmation, dict):
        confirmation = _adjudication_unavailable_report(
            disputed,
            "confirmation runner returned a non-object result",
        )
    confirmation_hashes = _storyboard_panel_hashes(output_dir)
    if any(
        confirmation_hashes.get(storyboard_id) != current_hashes.get(storyboard_id)
        for storyboard_id in disputed
    ):
        receipt = _pending_adjudication_receipt(
            output_dir,
            previous,
            current,
            disputed,
            current_hashes,
            status="blocked_evidence_changed",
            safe_error="Pxx pixels changed during verdict adjudication",
            confirmation=confirmation,
            after_correction_attempt=after_correction_attempt,
        )
        _write_review_adjudication_report(
            output_dir, [*(prior_receipts or []), receipt]
        )
        raise _ReviewAdjudicationBlocked(
            "Phase 5 QA mutated Pxx pixels during verdict adjudication",
            receipt,
        )
    confirmation_map = _blocking_storyboard_issue_map(confirmation)
    confirmation_layer = (confirmation.get("layers") or {}).get("L3", {})
    unavailable = any(
        str(issue.get("code") or "") == "storyboard_visual_review_unavailable"
        for issue in confirmation.get("issues") or []
    ) or str(confirmation_layer.get("status") or "") != "completed"
    if unavailable:
        _, safe_error = _adjudication_confirmation_audit(
            output_dir, confirmation
        )
        receipt = _pending_adjudication_receipt(
            output_dir,
            previous,
            current,
            disputed,
            current_hashes,
            status="blocked_unavailable",
            safe_error=safe_error,
            confirmation=confirmation,
            after_correction_attempt=after_correction_attempt,
        )
        _write_review_adjudication_report(
            output_dir, [*(prior_receipts or []), receipt]
        )
        raise _ReviewAdjudicationBlocked(
            "Phase 5 tie-break visual review is unavailable; refusing to "
            f"adjudicate unchanged pixels: {receipt['safe_error']}",
            receipt,
        )

    decisions: dict[str, bool] = {}
    votes: dict[str, dict[str, bool]] = {}
    for storyboard_id in disputed:
        panel_votes = {
            "previous_blocked": storyboard_id in previous_map,
            "current_blocked": storyboard_id in current_map,
            "confirmation_blocked": storyboard_id in confirmation_map,
        }
        votes[storyboard_id] = panel_votes
        decisions[storyboard_id] = sum(panel_votes.values()) >= 2

    pass_ids = {
        storyboard_id
        for storyboard_id, blocked in decisions.items()
        if not blocked
    }
    issues: list[dict[str, Any]] = []
    for issue in current.get("issues") or []:
        issue_ids = set(_correctable_storyboard_ids([issue]))
        removed = issue_ids & pass_ids
        if not removed:
            issues.append(copy.deepcopy(issue))
            continue
        retained = sorted(issue_ids - pass_ids)
        if not retained:
            continue
        for storyboard_id in retained:
            narrowed = _issue_for_storyboard_id(issue, storyboard_id)
            if narrowed is not None:
                issues.append(narrowed)

    current_ids = set(_blocking_storyboard_issue_map({"issues": issues}))
    for storyboard_id, blocked in decisions.items():
        if not blocked or storyboard_id in current_ids:
            continue
        source_issue = next(
            iter(
                previous_map.get(storyboard_id)
                or confirmation_map.get(storyboard_id)
                or []
            ),
            None,
        )
        if source_issue is not None:
            narrowed = _issue_for_storyboard_id(source_issue, storyboard_id)
            if narrowed is not None:
                issues.append(narrowed)

    reconciled = _recompute_qa_verdict(current, issues)
    receipt = {
        "schema": PHASE5_REVIEW_ADJUDICATION_SCHEMA,
        "status": "completed",
        "reason": "unchanged_panel_verdict_flip",
        "storyboard_ids": disputed,
        "asset_sha256": {
            storyboard_id: current_hashes[storyboard_id]
            for storyboard_id in disputed
        },
        "votes": votes,
        "decisions": {
            storyboard_id: "blocked" if blocked else "passed"
            for storyboard_id, blocked in decisions.items()
        },
        "confirmation_layers": copy.deepcopy(confirmation.get("layers") or {}),
    }
    if after_correction_attempt is not None:
        receipt["after_correction_attempt"] = after_correction_attempt
    _write_review_adjudication_report(
        output_dir, [*(prior_receipts or []), receipt]
    )
    return reconciled, receipt


def _global_uncorrectable_issues(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return blockers that cannot be isolated to a storyboard redraw."""
    correctable_ids = {
        id(issue)
        for issue in _correctable_issues(report) + _correctable_cinematic_issues(report)
    }
    return [
        issue
        for issue in blocking_issues(report.get("issues") or [])
        # Only an evidence-complete L3 R1-R4 finding belongs to the Phase 2
        # redraw loop.  A shot-scoped L1 contract defect is still an upstream
        # planning defect; treating its Sxx as redraw authority wastes quota
        # without repairing canonical metadata. Evidence-complete L4 annotation
        # contamination owns exact cinematic frames; other L4 defects restart Phase 4.
        if id(issue) not in correctable_ids
    ]


def _archive_correction_inputs(
    output_dir: Path,
    shot_ids: list[str],
    attempt: int,
) -> dict[str, Any]:
    """Copy the exact pre-redraw evidence into an immutable attempt folder."""
    correction_root = output_dir / "phase5_corrections"
    attempt_dir = correction_root / f"attempt_{attempt:02d}"
    revision = 1
    while attempt_dir.exists():
        revision += 1
        attempt_dir = correction_root / f"attempt_{attempt:02d}_r{revision:02d}"
    before_dir = attempt_dir / "before"
    before_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    root_files = ("storyboard_qa_report.json", "SHOT_STORYBOARDS.json")
    for name in root_files:
        source = output_dir / name
        if source.is_file():
            target = before_dir / name
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(output_dir)))
    patterns = []
    for shot_id in shot_ids:
        patterns.extend(
            (
                f"storyboard_beats/{shot_id}_P*",
                f"shot_storyboards/{shot_id}*",
                f"storyboard_images/{shot_id}.*",
                f"storyboard_bridges/{shot_id}__*",
                f"storyboard_bridges/*__{shot_id}*",
            )
        )
    for pattern in patterns:
        for source in sorted(output_dir.glob(pattern)):
            if not source.is_file():
                continue
            target = before_dir / source.relative_to(output_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(output_dir)))
    return {
        "attempt": attempt,
        "archive_dir": str(attempt_dir.relative_to(output_dir)),
        "copied": copied,
    }


def _cinematic_dependency_frame_ids(
    storyboard: dict[str, Any],
    target_frame_ids: list[str],
) -> list[str]:
    """Include later frames that consume a corrected prior cinematic frame."""
    targets = set(target_frame_ids)
    declared: set[str] = set()
    dependency_ids: set[str] = set()
    for shot_index, shot in enumerate(storyboard.get("shots") or [], 1):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        beat_ids = [
            str(beat.get("beat_id") or f"{shot_id}_P{beat_index:02d}")
            for beat_index, beat in enumerate(shot.get("storyboard_beats") or [], 1)
            if isinstance(beat, dict) and beat.get("video_first_frame")
        ]
        declared.update(beat_ids)
        first_target = next(
            (index for index, beat_id in enumerate(beat_ids) if beat_id in targets),
            None,
        )
        if first_target is not None:
            dependency_ids.update(beat_ids[first_target:])
    missing = targets - declared
    if missing:
        raise RuntimeError(
            "Phase 5 cinematic correction references unknown frame IDs: "
            + ", ".join(sorted(missing))
        )
    return sorted(dependency_ids)


def _archive_cinematic_correction_inputs(
    output_dir: Path,
    frame_ids: list[str],
    dependency_frame_ids: list[str],
    attempt: int,
) -> dict[str, Any]:
    """Archive exact Phase 4 pixels and receipts before local regeneration."""
    correction_root = output_dir / "phase5_cinematic_corrections"
    attempt_dir = correction_root / f"attempt_{attempt:02d}"
    revision = 1
    while attempt_dir.exists():
        revision += 1
        attempt_dir = correction_root / f"attempt_{attempt:02d}_r{revision:02d}"
    before_dir = attempt_dir / "before"
    before_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    sources = [
        output_dir / name
        for name in (
            "storyboard_qa_report.json",
            "STORYBOARD.json",
            "CINEMATIC_FIRST_FRAMES.json",
        )
    ]
    for frame_id in dependency_frame_ids:
        sources.extend(sorted((output_dir / "video_first_frames").glob(f"{frame_id}.*")))
    affected_shots = sorted({_parent_shot_id(frame_id) for frame_id in frame_ids})
    for shot_id in affected_shots:
        sources.extend(sorted((output_dir / "storyboard_images").glob(f"{shot_id}.*")))
    for source in dict.fromkeys(sources):
        if not source.is_file():
            continue
        target = before_dir / source.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(output_dir)))
    return {
        "attempt": attempt,
        "archive_dir": str(attempt_dir.relative_to(output_dir)),
        "frame_ids": frame_ids,
        "dependency_frame_ids": dependency_frame_ids,
        "copied": copied,
    }


def _redraw_failed_cinematic_frames(
    output_dir: Path,
    frame_ids: list[str],
    issues: list[dict[str, Any]],
    attempt: int,
    *,
    image_client: Any = None,
) -> dict[str, Any]:
    """Regenerate rejected Phase 4 frames through the canonical Phase 4 owner."""
    storyboard_path = output_dir / "STORYBOARD.json"
    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    targets = sorted(set(frame_ids))
    if not targets:
        raise RuntimeError("Phase 5 cinematic correction has no frame targets")
    dependency_frame_ids = _cinematic_dependency_frame_ids(storyboard, targets)
    archive = _archive_cinematic_correction_inputs(
        output_dir, targets, dependency_frame_ids, attempt
    )

    def optional_json(name: str, fallback: dict[str, Any]) -> dict[str, Any]:
        path = output_dir / name
        if not path.is_file():
            return fallback
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else fallback

    characters = optional_json("CHARACTERS.json", {"characters": []})
    scene_consistency = optional_json("SCENE_CONSISTENCY.json", {"shots": {}})
    previous_manifest = optional_json("CINEMATIC_FIRST_FRAMES.json", {})
    visual_style_path = next(
        (
            candidate
            for candidate in (
                output_dir / "visual-style.md",
                output_dir / "visual_style_spec.md",
            )
            if candidate.is_file()
        ),
        None,
    )
    from phases.phase4.cinematic_first_frames import (
        generate_cinematic_first_frames,
        validate_cinematic_first_frame_artifacts,
    )
    from utils.clip_style_classifier import ClipStyleClassifier

    manifest = generate_cinematic_first_frames(
        output_dir,
        storyboard,
        characters.get("characters") or [],
        scene_consistency,
        client=image_client,
        size=str(previous_manifest.get("size_requested") or "2K"),
        visual_style_path=visual_style_path,
        aspect_ratio=(
            str(previous_manifest.get("aspect_ratio") or "").strip()
            or str(storyboard.get("aspect_ratio") or "16:9")
        ),
        style_classifier=ClipStyleClassifier(),
    )
    artifact_errors = validate_cinematic_first_frame_artifacts(output_dir, storyboard)
    if artifact_errors:
        raise RuntimeError(
            "Phase 5 correction produced invalid cinematic first frames: "
            + "; ".join(artifact_errors[:8])
        )
    rejected_ids = set(manifest.get("phase5_rejected_frame_ids") or [])
    if not set(targets).issubset(rejected_ids):
        raise RuntimeError("Phase 4 did not consume every local Phase 5 frame rejection")
    regenerated_ids = {
        str(frame.get("beat_id") or "")
        for frame in manifest.get("frames") or []
        if isinstance(frame, dict) and frame.get("cache_hit") is not True
    }
    missing_regeneration = set(targets) - regenerated_ids
    unexpected_regeneration = regenerated_ids - set(dependency_frame_ids)
    if missing_regeneration or unexpected_regeneration:
        raise RuntimeError(
            "Phase 5 cinematic correction violated its target boundary: "
            f"missing={sorted(missing_regeneration)}, "
            f"unexpected={sorted(unexpected_regeneration)}"
        )
    _atomic_json(storyboard_path, storyboard)
    receipt = {
        "attempt": attempt,
        "status": "redrawn",
        "correction_family": "cinematic_first_frame",
        "frame_ids": targets,
        "dependency_frame_ids": dependency_frame_ids,
        "regenerated_frame_ids": sorted(regenerated_ids),
        "issue_codes": sorted({str(issue.get("code") or "") for issue in issues}),
        "archive": archive,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    attempt_dir = output_dir / archive["archive_dir"]
    _atomic_json(attempt_dir / "redraw_receipt.json", receipt)
    return receipt


def _restore_cinematic_aliases_after_previs_redraw(
    output_dir: Path,
    storyboard: dict[str, Any],
    shot_ids: list[str],
    archive: dict[str, Any],
) -> list[str]:
    """Prevent a Phase 5 PREVIS redraw from downgrading Phase 4 aliases."""
    from phases.phase4.cinematic_first_frames import CINEMATIC_FIRST_FRAME_SCHEMA

    requested = set(shot_ids)
    archive_dir = output_dir / str(archive.get("archive_dir") or "") / "before"
    restored: list[str] = []
    for index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, index)
        if shot_id not in requested:
            continue
        first_beat = next(
            (
                beat
                for beat in (shot.get("storyboard_beats") or [])
                if isinstance(beat, dict)
            ),
            None,
        )
        if not first_beat or first_beat.get(
            "video_first_frame_kind"
        ) != CINEMATIC_FIRST_FRAME_SCHEMA:
            continue
        source_value = str(first_beat.get("video_first_frame") or "").strip()
        source = Path(source_value)
        if not source.is_absolute():
            source = output_dir / source
        archived_image = archive_dir / "storyboard_images" / f"{shot_id}.png"
        archived_receipt = archive_dir / "storyboard_images" / f"{shot_id}.json"
        if not source.is_file() or not archived_image.is_file() or not archived_receipt.is_file():
            raise RuntimeError(
                f"{shot_id} cinematic alias archive is incomplete after PREVIS redraw"
            )
        try:
            receipt = json.loads(archived_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{shot_id} archived cinematic alias receipt is invalid: {exc}"
            ) from exc
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        archived_sha256 = hashlib.sha256(archived_image.read_bytes()).hexdigest()
        if (
            archived_sha256 != source_sha256
            or receipt.get("kind") != CINEMATIC_FIRST_FRAME_SCHEMA
            or str(receipt.get("canonical_source") or "") != source_value
        ):
            raise RuntimeError(
                f"{shot_id} archived storyboard alias is not its cinematic P01"
            )
        alias_dir = output_dir / "storyboard_images"
        alias_dir.mkdir(parents=True, exist_ok=True)
        for archived_path in (archived_image, archived_receipt):
            target = alias_dir / archived_path.name
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(archived_path, temporary)
            temporary.replace(target)
        if hashlib.sha256((alias_dir / f"{shot_id}.png").read_bytes()).hexdigest() != source_sha256:
            raise RuntimeError(
                f"{shot_id} cinematic alias restoration hash mismatch"
            )
        restored.append(shot_id)
    return restored


def _redraw_failed_storyboards(
    output_dir: Path,
    shot_ids: list[str],
    issues: list[dict[str, Any]],
    attempt: int,
    *,
    image_client: Any = None,
) -> dict[str, Any]:
    """Redraw only failed shots while retaining canonical visual references."""
    storyboard_path = output_dir / "STORYBOARD.json"
    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    characters_path = output_dir / "CHARACTERS.json"
    characters_data = (
        json.loads(characters_path.read_text(encoding="utf-8"))
        if characters_path.is_file()
        else {"characters": []}
    )
    valid_shot_ids = {
        _shot_id(shot, index)
        for index, shot in enumerate(storyboard.get("shots", []))
        if isinstance(shot, dict)
    }
    targets = sorted(set(shot_ids) & valid_shot_ids)
    if not targets:
        raise RuntimeError("Phase 5 correction has no valid failed shot IDs")
    context_by_shot = {
        shot_id: [
            issue
            for issue in issues
            if shot_id in (issue.get("shot_ids") or [])
        ]
        for shot_id in targets
    }
    target_storyboard_ids = [
        storyboard_id
        for storyboard_id in _correctable_storyboard_ids(issues)
        if _parent_shot_id(storyboard_id) in targets
    ]
    if not target_storyboard_ids:
        raise RuntimeError(
            "Phase 5 correction has no evidence-backed Pxx targets"
        )
    archive = _archive_correction_inputs(output_dir, targets, attempt)
    previous_manifest_path = output_dir / "SHOT_STORYBOARDS.json"
    try:
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        previous_manifest = {}

    from phases.phase2.shot_storyboards import (
        SHOT_STORYBOARD_SIZE,
        generate_shot_storyboards,
        validate_shot_storyboard_artifacts,
    )

    director_reference = storyboard.get("director_storyboard") or {}
    contract = generate_shot_storyboards(
        output_dir,
        storyboard,
        characters_data.get("characters", []),
        client=image_client,
        size=str(previous_manifest.get("size_requested") or SHOT_STORYBOARD_SIZE),
        director_storyboard_path=(
            director_reference.get("image")
            if isinstance(director_reference, dict)
            else None
        ),
        aspect_ratio=(
            str(previous_manifest.get("aspect_ratio") or "").strip() or None
        ),
        correction_context_by_shot=context_by_shot,
        correction_attempt=attempt,
        target_shot_ids=set(targets),
        target_beat_ids=set(target_storyboard_ids),
    )
    artifact_errors = validate_shot_storyboard_artifacts(output_dir, storyboard)
    if artifact_errors:
        raise RuntimeError(
            "Phase 5 correction produced invalid storyboard artifacts: "
            + "; ".join(artifact_errors[:8])
        )
    restored_cinematic_aliases = _restore_cinematic_aliases_after_previs_redraw(
        output_dir,
        storyboard,
        targets,
        archive,
    )
    _atomic_json(storyboard_path, storyboard)
    receipt = {
        "attempt": attempt,
        "status": "redrawn",
        "shot_ids": targets,
        "storyboard_ids": target_storyboard_ids,
        "issue_codes": sorted({str(issue.get("code") or "") for issue in issues}),
        "archive": archive,
        "restored_cinematic_aliases": restored_cinematic_aliases,
        "total_boards": contract.get("total_boards"),
        "total_panels": contract.get("total_panels"),
        "regenerated_panel_count": contract.get("regenerated_panel_count", 0),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    attempt_dir = output_dir / archive["archive_dir"]
    _atomic_json(attempt_dir / "redraw_receipt.json", receipt)
    return receipt


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _load_adjudication_snapshot(
    output_dir: Path,
    reference: dict[str, Any],
    expected_name: str,
) -> dict[str, Any]:
    if str(reference.get("path") or "") != expected_name:
        raise ValueError(f"adjudication snapshot must be {expected_name}")
    path = output_dir / expected_name
    if not path.is_file() or _sha256_file(path) != reference.get("sha256"):
        raise ValueError(f"adjudication snapshot {expected_name} is missing or changed")
    return _read_json_object(path)


def _reconstruct_legacy_pending_adjudication(
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Recover the pre-receipt failure shape without rerunning image generation."""
    correction_path = output_dir / "phase5_correction_report.json"
    current_path = output_dir / "storyboard_qa_report.json"
    storyboard_path = output_dir / "STORYBOARD.json"
    if not (
        correction_path.is_file()
        and current_path.is_file()
        and storyboard_path.is_file()
    ):
        return None
    correction = _read_json_object(correction_path)
    history = correction.get("history")
    if not isinstance(history, list) or not history:
        return None
    last = history[-1]
    if not isinstance(last, dict) or last.get("status") != "adjudication_error":
        return None
    error = str(last.get("error") or "")
    if "tie-break visual review is unavailable" not in error:
        return None
    archive_dir = str(
        (((last.get("redraw") or {}).get("archive") or {}).get("archive_dir"))
        or ""
    )
    previous_path = output_dir / archive_dir / "before" / "storyboard_qa_report.json"
    try:
        previous_path.resolve().relative_to(output_dir.resolve())
    except (OSError, ValueError):
        raise ValueError("legacy adjudication archive escapes the run directory")
    if not previous_path.is_file():
        raise ValueError("legacy adjudication archive is incomplete")
    previous = _read_json_object(previous_path)
    current = _read_json_object(current_path)
    storyboard = _read_json_object(storyboard_path)
    previous_hashes = _storyboard_panel_hashes_for_storyboard(
        previous_path.parent,
        storyboard,
    )
    current_hashes = _storyboard_panel_hashes_for_storyboard(
        output_dir,
        storyboard,
    )
    previous_map = _blocking_storyboard_issue_map(previous)
    current_map = _blocking_storyboard_issue_map(current)
    disputed = sorted(
        storyboard_id
        for storyboard_id in set(previous_map) ^ set(current_map)
        if previous_hashes.get(storyboard_id)
        and previous_hashes.get(storyboard_id) == current_hashes.get(storyboard_id)
    )
    if not disputed:
        raise ValueError(
            "legacy adjudication failure has no unchanged disputed Pxx evidence"
        )
    receipt = _pending_adjudication_receipt(
        output_dir,
        previous,
        current,
        disputed,
        current_hashes,
        status="blocked_unavailable",
        safe_error=(
            f"{error}; original confirmation transport details were not retained"
        ),
        confirmation=None,
        after_correction_attempt=int(last.get("attempt") or 0) or None,
        legacy_reconstructed=True,
    )
    _write_review_adjudication_report(output_dir, [receipt])
    return receipt, [receipt]


def _load_pending_adjudication(
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    report_path = output_dir / PHASE5_REVIEW_ADJUDICATION_REPORT_NAME
    if not report_path.is_file():
        return _reconstruct_legacy_pending_adjudication(output_dir)
    report = _read_json_object(report_path)
    if report.get("schema") != PHASE5_REVIEW_ADJUDICATIONS_SCHEMA:
        raise ValueError("unknown Phase 5 review adjudication report schema")
    receipts = report.get("adjudications")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("Phase 5 review adjudication report has no receipts")
    if report.get("status") == "completed":
        return None
    pending = receipts[-1]
    if not isinstance(pending, dict):
        raise ValueError("Phase 5 review adjudication receipt must be an object")
    if pending.get("schema") != PHASE5_REVIEW_ADJUDICATION_SCHEMA:
        raise ValueError("unknown Phase 5 review adjudication receipt schema")
    return pending, copy.deepcopy(receipts)


def _load_completed_adjudication_correction(
    output_dir: Path,
) -> dict[str, Any] | None:
    """Load a completed tie-break without resetting its correction budget."""
    adjudication_path = output_dir / PHASE5_REVIEW_ADJUDICATION_REPORT_NAME
    if not adjudication_path.is_file():
        return None
    report = _read_json_object(adjudication_path)
    if report.get("schema") != PHASE5_REVIEW_ADJUDICATIONS_SCHEMA:
        raise ValueError("unknown Phase 5 review adjudication report schema")
    if report.get("status") != "completed":
        return None
    receipts = report.get("adjudications")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("completed Phase 5 adjudication report has no receipts")
    completed = receipts[-1]
    if not isinstance(completed, dict):
        raise ValueError("completed Phase 5 adjudication receipt must be an object")
    if completed.get("schema") != PHASE5_REVIEW_ADJUDICATION_SCHEMA:
        raise ValueError("unknown Phase 5 review adjudication receipt schema")
    if completed.get("status") != "completed":
        raise ValueError("Phase 5 adjudication report status disagrees with its receipt")

    qa_path = output_dir / "storyboard_qa_report.json"
    correction_path = output_dir / "phase5_correction_report.json"
    if not qa_path.is_file() or not correction_path.is_file():
        raise ValueError("completed Phase 5 adjudication has incomplete correction state")
    result = _read_json_object(qa_path)
    correction = _read_json_object(correction_path)
    if result.get("correction") != correction:
        raise ValueError("Phase 5 correction report disagrees with the QA report")
    if result.get("review_adjudications") != receipts:
        raise ValueError("Phase 5 QA report lost adjudication history")
    if correction.get("review_adjudications") != receipts:
        raise ValueError("Phase 5 correction report lost adjudication history")

    history = correction.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("completed Phase 5 adjudication has no correction history")
    attempts_used = correction.get("attempts_used")
    if isinstance(attempts_used, bool) or not isinstance(attempts_used, int):
        raise ValueError("Phase 5 correction attempt count is invalid")
    if attempts_used != len(history):
        raise ValueError("Phase 5 correction attempt count disagrees with history")
    if any(
        not isinstance(entry, dict) or entry.get("attempt") != index
        for index, entry in enumerate(history, 1)
    ):
        raise ValueError("Phase 5 correction attempt history is not contiguous")
    after_attempt = completed.get("after_correction_attempt")
    if (
        isinstance(after_attempt, bool)
        or not isinstance(after_attempt, int)
        or after_attempt < 1
        or after_attempt > attempts_used
    ):
        raise ValueError("completed Phase 5 adjudication attempt lineage changed")
    if history[after_attempt - 1].get("review_adjudication") != completed:
        raise ValueError("completed Phase 5 adjudication is not bound to correction history")

    decisions = completed.get("decisions")
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("completed Phase 5 adjudication has no decisions")
    if any(value not in {"passed", "blocked"} for value in decisions.values()):
        raise ValueError("completed Phase 5 adjudication has an unknown decision")
    continuable = after_attempt == attempts_used
    if continuable:
        asset_hashes = completed.get("asset_sha256")
        if (
            not isinstance(asset_hashes, dict)
            or sorted(asset_hashes) != sorted(decisions)
        ):
            raise ValueError("completed Phase 5 adjudication asset scope changed")
        current_hashes = _storyboard_panel_hashes(output_dir)
        if any(
            current_hashes.get(storyboard_id) != asset_hashes.get(storyboard_id)
            for storyboard_id in decisions
        ):
            raise ValueError("completed Phase 5 adjudication pixels changed")

        lineage_receipt = next(
            (
                receipt
                for receipt in reversed(receipts)
                if isinstance(receipt, dict)
                and isinstance(receipt.get("input_artifacts"), list)
            ),
            None,
        )
        if (
            lineage_receipt is None
            or lineage_receipt.get("input_artifacts")
            != _phase5_input_artifacts(output_dir)
        ):
            raise ValueError("completed Phase 5 adjudication input lineage changed")

        issue_map = _blocking_storyboard_issue_map(result)
        for storyboard_id, decision in decisions.items():
            if decision == "blocked" and storyboard_id not in issue_map:
                raise ValueError("blocked adjudication decision lost its QA evidence")
            if decision == "passed" and storyboard_id in issue_map:
                raise ValueError("passed adjudication decision still has blocking QA evidence")
    return {
        "result": copy.deepcopy(result),
        "correction": copy.deepcopy(correction),
        "receipts": copy.deepcopy(receipts),
        "continuable": continuable,
    }


def _adjudication_resume_result(
    output_dir: Path,
    base_result: dict[str, Any],
    correction: dict[str, Any],
    receipts: list[dict[str, Any]],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(base_result)
    result["correction"] = correction
    result["review_adjudications"] = copy.deepcopy(receipts)
    outputs = list(result.get("outputs") or [])
    for name in (
        PHASE5_REVIEW_ADJUDICATION_REPORT_NAME,
        "phase5_correction_report.json",
    ):
        if name not in outputs:
            outputs.append(name)
    result["outputs"] = outputs
    if error:
        result.update(status="error", gate_passed=False, error=error)
    _atomic_json(output_dir / "phase5_correction_report.json", correction)
    _atomic_json(output_dir / "storyboard_qa_report.json", result)
    return result


def _resume_pending_review_adjudication(
    output_dir: Path,
    confirmation_runner: Callable[[Path, list[str]], dict[str, Any]],
) -> dict[str, Any] | None:
    """Resume only an exact failed confirmation; never rerun QA or redraw."""
    try:
        loaded = _load_pending_adjudication(output_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "status": "error",
            "grade": "D",
            "gate_passed": False,
            "issues": [],
            "failed_shot_ids": [],
            "error": f"Phase 5 adjudication resume refused: {redact_text(str(exc))}",
        }
        _atomic_json(output_dir / "storyboard_qa_report.json", result)
        return result
    if loaded is None:
        return None
    pending, prior_receipts = loaded
    try:
        snapshots = pending.get("report_snapshots")
        if not isinstance(snapshots, dict):
            raise ValueError("pending adjudication has no report snapshots")
        previous = _load_adjudication_snapshot(
            output_dir,
            snapshots.get("previous") or {},
            PHASE5_REVIEW_ADJUDICATION_PREVIOUS_REPORT_NAME,
        )
        current = _load_adjudication_snapshot(
            output_dir,
            snapshots.get("current") or {},
            PHASE5_REVIEW_ADJUDICATION_CURRENT_REPORT_NAME,
        )
        base_result = _read_json_object(output_dir / "storyboard_qa_report.json")
        expected_inputs = pending.get("input_artifacts")
        if expected_inputs != _phase5_input_artifacts(output_dir):
            raise ValueError("Phase 5 input artifact lineage changed")
        disputed = pending.get("storyboard_ids")
        if not isinstance(disputed, list) or not disputed:
            raise ValueError("pending adjudication has no disputed storyboard IDs")
        disputed = [str(value) for value in disputed]
        current_hashes = _storyboard_panel_hashes(output_dir)
        asset_hashes = pending.get("asset_sha256")
        if not isinstance(asset_hashes, dict) or any(
            current_hashes.get(storyboard_id) != asset_hashes.get(storyboard_id)
            for storyboard_id in disputed
        ):
            changed = copy.deepcopy(pending)
            changed.update(
                status="blocked_evidence_changed",
                safe_error="Pxx pixels changed before explicit adjudication resume",
                observed_asset_sha256={
                    storyboard_id: current_hashes.get(storyboard_id)
                    for storyboard_id in disputed
                },
            )
            receipts = [*prior_receipts, changed]
            _write_review_adjudication_report(output_dir, receipts)
            correction = copy.deepcopy(base_result.get("correction") or {})
            correction.update(
                final_gate_passed=False,
                review_adjudications=receipts,
            )
            return _adjudication_resume_result(
                output_dir,
                base_result,
                correction,
                receipts,
                error=(
                    "Phase 5 adjudication resume refused because disputed Pxx "
                    "pixels changed"
                ),
            )
        expected_disputed = sorted(
            storyboard_id
            for storyboard_id in (
                set(_blocking_storyboard_issue_map(previous))
                ^ set(_blocking_storyboard_issue_map(current))
            )
            if storyboard_id in asset_hashes
        )
        if expected_disputed != sorted(disputed):
            raise ValueError("pending adjudication vote lineage changed")
        if pending.get("status") != "blocked_unavailable":
            raise ValueError(
                f"pending adjudication status {pending.get('status')} is not resumable"
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        base_result = locals().get("base_result") or {
            "status": "error",
            "grade": "D",
            "gate_passed": False,
            "issues": [],
        }
        correction = copy.deepcopy(base_result.get("correction") or {})
        correction.update(
            final_gate_passed=False,
            review_adjudications=prior_receipts,
        )
        return _adjudication_resume_result(
            output_dir,
            base_result,
            correction,
            prior_receipts,
            error=f"Phase 5 adjudication resume refused: {redact_text(str(exc))}",
        )

    attempt = pending.get("after_correction_attempt")
    try:
        reconciled, completed = _adjudicate_unchanged_panel_flips(
            output_dir,
            previous,
            {storyboard_id: asset_hashes[storyboard_id] for storyboard_id in disputed},
            current,
            current_hashes,
            confirmation_runner,
            after_correction_attempt=(int(attempt) if attempt is not None else None),
            prior_receipts=prior_receipts,
        )
    except _ReviewAdjudicationBlocked as exc:
        receipts = [*prior_receipts, exc.receipt]
        correction = copy.deepcopy(base_result.get("correction") or {})
        correction.update(
            final_gate_passed=False,
            review_adjudications=receipts,
        )
        history = correction.get("history")
        if isinstance(history, list) and history:
            history[-1].update(
                status="adjudication_error",
                error=str(exc),
                review_adjudication=exc.receipt,
            )
        return _adjudication_resume_result(
            output_dir,
            base_result,
            correction,
            receipts,
            error=f"Phase 5 automatic correction failed: {exc}",
        )

    if completed is None:
        return None
    receipts = [*prior_receipts, completed]
    correction = copy.deepcopy(base_result.get("correction") or {})
    history = correction.get("history")
    if isinstance(history, list) and history:
        history[-1].update(
            status="passed" if reconciled.get("gate_passed") is True else "rejected",
            after_grade=reconciled.get("grade"),
            review_adjudication=completed,
        )
        history[-1].pop("error", None)
    correction.update(
        final_gate_passed=reconciled.get("gate_passed") is True,
        review_adjudications=receipts,
    )
    return _adjudication_resume_result(
        output_dir,
        reconciled,
        correction,
        receipts,
    )


def run_storyboard_qa_with_correction(
    output_dir: Path,
    *,
    max_correction_attempts: int | None = None,
    qa_runner: Callable[[Path], dict[str, Any]] | None = None,
    adjudication_runner: Callable[
        [Path, list[str]], dict[str, Any]
    ] | None = None,
    redraw_runner: Callable[[Path, list[str], list[dict[str, Any]], int], dict[str, Any]] | None = None,
    cinematic_redraw_runner: Callable[[Path, list[str], list[dict[str, Any]], int], dict[str, Any]] | None = None,
    image_client: Any = None,
    dry_run: bool = False,
    resume_pending_adjudication: bool = False,
) -> dict[str, Any]:
    """Run Phase 5 with one bounded correction family and recheck loop."""
    output_dir = Path(output_dir)
    if dry_run:
        return _run_storyboard_qa_dry_run(output_dir)
    attempts_allowed = _resolved_correction_attempts(max_correction_attempts)
    qa = qa_runner or run_storyboard_qa_gate
    confirmation = adjudication_runner or _run_storyboard_adjudication_review
    completed_correction: dict[str, Any] | None = None
    if resume_pending_adjudication:
        resumed = _resume_pending_review_adjudication(output_dir, confirmation)
        if resumed is not None:
            return resumed
        try:
            completed_correction = _load_completed_adjudication_correction(
                output_dir
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            try:
                result = _read_json_object(
                    output_dir / "storyboard_qa_report.json"
                )
            except (OSError, json.JSONDecodeError, ValueError):
                result = {
                    "grade": "D",
                    "issues": [],
                    "failed_shot_ids": [],
                }
            result.update(
                status="error",
                gate_passed=False,
                error=(
                    "Phase 5 completed adjudication continuation refused: "
                    f"{redact_text(str(exc))}"
                ),
            )
            _atomic_json(output_dir / "storyboard_qa_report.json", result)
            return result
    if completed_correction is None:
        result = qa(output_dir)
        history: list[dict[str, Any]] = []
        adjudications: list[dict[str, Any]] = []
        first_attempt = 1
        attempt_limit = attempts_allowed
        correction_family = (
            "cinematic_first_frame"
            if _correctable_cinematic_issues(result)
            else "storyboard_previs"
        )
    else:
        result = completed_correction["result"]
        persisted_correction = completed_correction["correction"]
        if completed_correction["continuable"] is not True:
            return result
        history = copy.deepcopy(persisted_correction["history"])
        adjudications = completed_correction["receipts"]
        persisted_limit = persisted_correction.get("max_attempts")
        if isinstance(persisted_limit, bool) or not isinstance(persisted_limit, int):
            result.update(
                status="error",
                gate_passed=False,
                error="Phase 5 completed adjudication correction limit is invalid",
            )
            _atomic_json(output_dir / "storyboard_qa_report.json", result)
            return result
        attempt_limit = min(attempts_allowed, persisted_limit)
        first_attempt = len(history) + 1
        correction_family = str(
            persisted_correction.get("correction_family") or ""
        )
        if correction_family != "storyboard_previs":
            return result
        if result.get("gate_passed") is True or first_attempt > attempt_limit:
            return result
    previous_result = copy.deepcopy(result)
    previous_hashes = _storyboard_panel_hashes(output_dir)

    global_issues = _global_uncorrectable_issues(result)
    if result.get("gate_passed") is not True and global_issues:
        if completed_correction is not None:
            result.update(
                status="error",
                gate_passed=False,
                error=(
                    "Phase 5 completed adjudication cannot change correction "
                    "family; restart from the owner recorded by the QA issues"
                ),
            )
            _atomic_json(output_dir / "storyboard_qa_report.json", result)
            return result
        from phases.phase5.replanning import (
            build_padding_screenplay_rewrite_request,
        )

        issue_codes = sorted({
            str(issue.get("code") or "unknown") for issue in global_issues
        })
        restart_phase = (
            "phase4"
            if any(str(issue.get("layer") or "").upper() == "L4" for issue in global_issues)
            else "phase1"
            if any(
                str(issue.get("layer") or "").upper() == "L1"
                for issue in global_issues
            )
            else "phase2"
        )
        correction = {
            "enabled": attempts_allowed > 0,
            "max_attempts": attempts_allowed,
            "attempts_used": 0,
            "status": "requires_replanning",
            "global_issue_codes": issue_codes,
            "recommended_restart_phase": restart_phase,
            "history": [],
            "final_gate_passed": False,
        }
        rewrite_request = build_padding_screenplay_rewrite_request(global_issues)
        if restart_phase == "phase1" and rewrite_request is not None:
            correction["replanning_policy"] = (
                "rewrite_screenplay_once_then_fail_closed"
            )
            correction["screenplay_rewrite_request"] = rewrite_request
        result = {
            **result,
            "status": "error",
            "gate_passed": False,
            "correction": correction,
            "error": (
                "Storyboard QA has global blocking issues that cannot be "
                "corrected by redrawing isolated shots; restart from "
                f"{restart_phase}: {', '.join(issue_codes)}"
            ),
        }
        outputs = list(result.get("outputs") or [])
        if "phase5_correction_report.json" not in outputs:
            outputs.append("phase5_correction_report.json")
        result["outputs"] = outputs
        _atomic_json(output_dir / "phase5_correction_report.json", correction)
        _atomic_json(output_dir / "storyboard_qa_report.json", result)
        return result

    for attempt in range(first_attempt, attempt_limit + 1):
        if result.get("gate_passed") is True or result.get("status") != "error":
            break
        if _global_uncorrectable_issues(result):
            break
        if correction_family == "cinematic_first_frame":
            issues = _correctable_cinematic_issues(result)
            target_frame_ids = _correctable_cinematic_frame_ids(issues)
            target_storyboard_ids: list[str] = []
            target_ids = sorted({_parent_shot_id(value) for value in target_frame_ids})
        else:
            issues = _correctable_issues(result)
            target_storyboard_ids = _correctable_storyboard_ids(issues)
            target_frame_ids = []
            target_ids = sorted({
                _parent_shot_id(value) for value in target_storyboard_ids
            })
        if not issues or not target_ids:
            break
        before_grade = result.get("grade")
        try:
            if correction_family == "cinematic_first_frame":
                if cinematic_redraw_runner is not None:
                    redraw_receipt = cinematic_redraw_runner(
                        output_dir, target_frame_ids, issues, attempt
                    )
                else:
                    redraw_receipt = _redraw_failed_cinematic_frames(
                        output_dir,
                        target_frame_ids,
                        issues,
                        attempt,
                        image_client=image_client,
                    )
            elif redraw_runner is not None:
                redraw_receipt = redraw_runner(
                    output_dir, target_ids, issues, attempt
                )
            else:
                redraw_receipt = _redraw_failed_storyboards(
                    output_dir,
                    target_ids,
                    issues,
                    attempt,
                    image_client=image_client,
                )
        except Exception as exc:
            history.append({
                "attempt": attempt,
                "status": "redraw_error",
                "correction_family": correction_family,
                "shot_ids": target_ids,
                **({"frame_ids": target_frame_ids} if target_frame_ids else {}),
                "before_grade": before_grade,
                "error": str(exc),
            })
            result = {
                **result,
                "status": "error",
                "gate_passed": False,
                "error": f"Phase 5 automatic correction failed: {exc}",
            }
            break
        current_hashes = _storyboard_panel_hashes(output_dir)
        result = qa(output_dir)
        adjudication = None
        try:
            result, adjudication = _adjudicate_unchanged_panel_flips(
                output_dir,
                previous_result,
                previous_hashes,
                result,
                current_hashes,
                confirmation,
                after_correction_attempt=attempt,
                prior_receipts=adjudications,
                confirmation_allowed=(
                    len(adjudications) < MAX_REVIEW_ADJUDICATIONS
                ),
            )
        except Exception as exc:
            failed_adjudication = (
                exc.receipt
                if isinstance(exc, _ReviewAdjudicationBlocked)
                else None
            )
            if failed_adjudication is not None:
                adjudications.append(failed_adjudication)
            history.append({
                "attempt": attempt,
                "status": "adjudication_error",
                "correction_family": correction_family,
                "shot_ids": target_ids,
                "storyboard_ids": target_storyboard_ids,
                "before_grade": before_grade,
                "after_grade": result.get("grade"),
                "redraw": redraw_receipt,
                "error": str(exc),
                **(
                    {"review_adjudication": failed_adjudication}
                    if failed_adjudication is not None
                    else {}
                ),
            })
            result = {
                **result,
                "status": "error",
                "gate_passed": False,
                "error": f"Phase 5 automatic correction failed: {exc}",
            }
            if failed_adjudication is not None:
                outputs = list(result.get("outputs") or [])
                if PHASE5_REVIEW_ADJUDICATION_REPORT_NAME not in outputs:
                    outputs.append(PHASE5_REVIEW_ADJUDICATION_REPORT_NAME)
                result["outputs"] = outputs
            break
        if adjudication is not None:
            adjudications.append(adjudication)
            result["review_adjudications"] = copy.deepcopy(adjudications)
            outputs = list(result.get("outputs") or [])
            if PHASE5_REVIEW_ADJUDICATION_REPORT_NAME not in outputs:
                outputs.append(PHASE5_REVIEW_ADJUDICATION_REPORT_NAME)
            result["outputs"] = outputs
            _atomic_json(output_dir / "storyboard_qa_report.json", result)
        history.append({
            "attempt": attempt,
            "status": "passed" if result.get("gate_passed") is True else "rejected",
            "correction_family": correction_family,
            "shot_ids": target_ids,
            "storyboard_ids": target_storyboard_ids,
            **({"frame_ids": target_frame_ids} if target_frame_ids else {}),
            "before_grade": before_grade,
            "after_grade": result.get("grade"),
            "redraw": redraw_receipt,
            **({"review_adjudication": adjudication} if adjudication else {}),
        })
        previous_result = copy.deepcopy(result)
        previous_hashes = current_hashes

    if history:
        correction = {
            "enabled": attempt_limit > 0,
            "max_attempts": attempt_limit,
            "attempts_used": len(history),
            "correction_family": correction_family,
            "history": history,
            "review_adjudications": adjudications,
            "final_gate_passed": result.get("gate_passed") is True,
        }
        result["correction"] = correction
        outputs = list(result.get("outputs") or [])
        if "phase5_correction_report.json" not in outputs:
            outputs.append("phase5_correction_report.json")
        result["outputs"] = outputs
        automatic_failure = str(result.get("error") or "").startswith(
            "Phase 5 automatic correction failed:"
        )
        if result.get("gate_passed") is not True and not automatic_failure:
            if len(history) < attempt_limit:
                result["error"] = (
                    "Storyboard QA still blocks Phase 6 after completing the "
                    f"{correction_family} correction family; no second correction "
                    "family was started"
                )
            else:
                result["error"] = (
                    "Storyboard QA still blocks Phase 6 after "
                    f"{len(history)}/{attempt_limit} automatic correction attempt(s)"
                )
        _atomic_json(output_dir / "phase5_correction_report.json", correction)
        _atomic_json(output_dir / "storyboard_qa_report.json", result)
    return result


def grid_path_exists(output_dir: Path) -> bool:
    return (output_dir / "storyboard_qa_grid.jpg").is_file()
