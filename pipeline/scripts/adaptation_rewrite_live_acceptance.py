#!/usr/bin/env python3
"""One-request live acceptance for Phase 1 source-indexed Adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import canonical_visual_ledger_36s_acceptance as full_chain_acceptance
from phases.phase1 import adaptation_engine
from runtime.provider_attempt_policy import provider_attempt_scope
from utils.config import get_api_key, validate_config


INPUT_SCHEMA = "honcut.adaptation-rewrite-live-input.v1"
REGRESSION_SCHEMA = "honcut.adaptation-rewrite-regression.v1"
RECEIPT_SCHEMA = "honcut.adaptation-rewrite-live-acceptance.v1"
RECEIPT_NAME = "adaptation_rewrite_live_acceptance.json"
LIVE_DIRECTORY = Path("live_acceptance") / "adaptation_rewrite"
REQUEST_RECEIPT_NAME = "adaptation_rewrite_request.json"
EVIDENCE_NAME = "adaptation_rewrite_result.json"
_SAFE_TRANSPORT_FIELDS = frozenset({
    "error_type",
    "max_tokens",
    "messages_sha256",
    "model",
    "provider_family",
    "response_format_sha256",
    "response_sha256",
    "stream",
    "submission_outcome",
    "transport_status",
})


def _read_object(path: Path) -> dict[str, Any]:
    return full_chain_acceptance._read_object(path)


def _canonical_sha256(value: Any) -> str:
    return full_chain_acceptance._canonical_sha256(value)


def _safe_transport_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key in _SAFE_TRANSPORT_FIELDS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }


def _load_input(path: Path) -> dict[str, Any]:
    document = _read_object(path)
    if document.get("schema") != INPUT_SCHEMA:
        raise RuntimeError("unsupported Adaptation live input schema")
    contract = {
        key: document.get(key)
        for key in (
            "schema",
            "source_events",
            "director_plan",
            "target_duration_s",
            "shot_duration_s",
            "shot_policy",
            "max_material_padding_ratio",
            "delivery_overrun_ratio",
        )
    }
    if document.get("contract_sha256") != _canonical_sha256(contract):
        raise RuntimeError("Adaptation live input hash mismatch")
    events = contract["source_events"]
    if not isinstance(events, list) or not events or any(
        not isinstance(event, dict) for event in events
    ):
        raise RuntimeError("Adaptation live source events are invalid")
    if not isinstance(contract["director_plan"], dict):
        raise RuntimeError("Adaptation live director plan is invalid")
    return contract


def _build_rewrite_contract(
    live_input: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    events = live_input["source_events"]
    director_plan = live_input["director_plan"]
    target_duration = int(live_input["target_duration_s"])
    shot_duration = int(live_input["shot_duration_s"])
    shot_policy = adaptation_engine._validate_shot_policy(
        str(live_input["shot_policy"])
    )
    max_padding = float(live_input["max_material_padding_ratio"])
    delivery_overrun = float(live_input["delivery_overrun_ratio"])
    capability_profile = adaptation_engine.get_video_capabilities()
    capacity_plan = adaptation_engine._estimate_action_capacity_plan(
        events,
        target_duration,
        shot_duration,
        shot_policy=shot_policy,
        max_material_padding_ratio=max_padding,
        delivery_overrun_ratio=delivery_overrun,
    )
    layout = capacity_plan["primary_shot_layout"]
    material_duration = int(capacity_plan["material_duration"])
    primary_shots = int(capacity_plan["primary_shots"])
    effective_shot_duration = max(
        adaptation_engine.min_primary_story_duration(capability_profile),
        min(
            adaptation_engine.max_primary_story_duration(capability_profile),
            round(material_duration / primary_shots),
        ),
    )
    production_events, duration_plan = (
        adaptation_engine._build_duration_scaled_event_plan(
            events,
            target_duration=material_duration,
            beat_count=primary_shots,
            effective_shot_duration=effective_shot_duration,
            capabilities=capability_profile,
            max_generation_units_per_beat=layout[
                "max_generation_action_units_per_primary_shot"
            ],
            maximum_total_generation_units=layout[
                "production_action_unit_target"
            ],
            generation_unit_capacities_per_beat=list(
                layout["generation_action_unit_capacities"]
            ),
        )
    )
    rewrite_ids = [
        record["source_event_id"]
        for record in duration_plan["events"]
        if record.get("scaling") == "rewrite"
    ]
    if not rewrite_ids:
        raise RuntimeError("Adaptation live input does not require a rewrite")
    if any(
        record.get("scaling") == "representative"
        for record in duration_plan["events"]
    ):
        raise RuntimeError("Adaptation live input has mixed scaling generations")
    projection = {
        "schema": INPUT_SCHEMA,
        "source_events_sha256": _canonical_sha256(events),
        "director_plan_sha256": _canonical_sha256(director_plan),
        "production_events_sha256": _canonical_sha256(production_events),
        "duration_plan_sha256": _canonical_sha256(duration_plan),
        "rewrite_source_event_ids": rewrite_ids,
        "target_duration_s": target_duration,
        "material_duration_s": material_duration,
        "primary_shots": primary_shots,
        "effective_shot_duration_s": effective_shot_duration,
    }
    return events, production_events, duration_plan, projection


def _load_regression(path: Path, git_commit: str) -> dict[str, Any]:
    receipt = _read_object(path)
    if (
        receipt.get("schema") != REGRESSION_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != git_commit
        or receipt.get("provider_request_count") != 0
    ):
        raise RuntimeError("Adaptation regression receipt is not current")
    return receipt


class AdaptationRequestReceipt:
    """Persist the only allowed Ark request before the transport sends it."""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME
        self.token: str | None = None

    def before(self, payload: dict[str, Any]) -> str:
        if self.path.exists() or self.token is not None:
            raise RuntimeError(
                "Adaptation live request already exists; resubmission forbidden"
            )
        portable = _safe_transport_metadata(payload)
        if portable.get("provider_family") != "ark_text":
            raise RuntimeError("Adaptation live request is not Ark text")
        token = _canonical_sha256(portable)
        full_chain_acceptance._atomic_write_json(self.path, {
            "schema": RECEIPT_SCHEMA,
            "status": "submission_uncertain",
            "request_fingerprint": token,
            "provider_request_count": 1,
            "safe_payload": portable,
            "automatic_resubmission_forbidden": True,
        })
        self.token = token
        return token

    def after(self, token: str, outcome: dict[str, Any]) -> None:
        if token != self.token or not self.path.is_file():
            raise RuntimeError("Adaptation live completion token is invalid")
        receipt = _read_object(self.path)
        receipt.update({
            "status": "provider_completed",
            "safe_outcome": _safe_transport_metadata(outcome),
        })
        full_chain_acceptance._atomic_write_json(self.path, receipt)

    def failed(self, token: str, outcome: dict[str, Any]) -> None:
        if token != self.token or not self.path.is_file():
            raise RuntimeError("Adaptation live failure token is invalid")
        receipt = _read_object(self.path)
        known_rejected = outcome.get("submission_outcome") == "known_rejected"
        receipt.update({
            "status": (
                "provider_failed" if known_rejected else "submission_uncertain"
            ),
            "safe_failure": _safe_transport_metadata(outcome),
            "automatic_resubmission_forbidden": True,
        })
        full_chain_acceptance._atomic_write_json(self.path, receipt)

    def completed(self) -> dict[str, Any]:
        receipt = _read_object(self.path)
        if receipt.get("status") != "provider_completed":
            raise RuntimeError("Adaptation live request did not settle")
        return receipt


def build_preflight(
    workspace: Path,
    input_path: Path,
    regression_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    input_path = full_chain_acceptance._inside(input_path, workspace)
    regression_path = full_chain_acceptance._inside(
        regression_path,
        workspace,
    )
    live_input = _load_input(input_path)
    _events, _production_events, _duration_plan, projection = (
        _build_rewrite_contract(live_input)
    )
    source_identity = full_chain_acceptance._repo_source_identity()
    regression = _load_regression(
        regression_path,
        source_identity["git_commit"],
    )
    formal_config = validate_config(["ARK_AGENT"])
    checks = {
        "worktree_clean": source_identity["worktree_clean"],
        "regression_receipt_current": regression["status"] == "passed",
        "ark_agent_configured": bool(
            formal_config.get("valid") and get_api_key("ARK_AGENT_API_KEY")
        ),
        "no_previous_request": not (
            workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME
        ).exists(),
    }
    missing = sorted(key for key, passed in checks.items() if not passed)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "preflight_passed" if not missing else "preflight_blocked",
        "call_chain_verdict": "not_submitted",
        "business_verdict": "not_evaluated",
        "provider_request_count": 0,
        "exact_request_limit": 1,
        "source": {
            **source_identity,
            "input_path": input_path.relative_to(workspace).as_posix(),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "regression_path": regression_path.relative_to(workspace).as_posix(),
            "regression_sha256": hashlib.sha256(
                regression_path.read_bytes()
            ).hexdigest(),
        },
        "projection": projection,
        "checks": checks,
        "missing_configuration": missing,
        "next_stage": "paid_single_rewrite" if not missing else "stop_zero_request",
    }


def execute_single_rewrite(
    workspace: Path,
    input_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if preflight.get("status") != "preflight_passed":
        raise RuntimeError("Adaptation paid gate refused before preflight passed")
    workspace = workspace.resolve()
    input_path = full_chain_acceptance._inside(input_path, workspace)
    live_input = _load_input(input_path)
    events, production_events, duration_plan, projection = (
        _build_rewrite_contract(live_input)
    )
    if projection != preflight.get("projection"):
        raise RuntimeError("Adaptation live projection changed after preflight")
    request_receipt = AdaptationRequestReceipt(workspace)
    with provider_attempt_scope(
        max_retries=0,
        before_provider_request=request_receipt.before,
        after_provider_request=request_receipt.after,
        failed_provider_request=request_receipt.failed,
    ):
        selected_events, selected_plan = (
            adaptation_engine._apply_director_action_selection(
                events,
                production_events,
                duration_plan,
                live_input["director_plan"],
            )
        )
    transport = request_receipt.completed()
    reconciliation = selected_plan.get(
        "source_indexed_rewrite_reconciliation"
    )
    expected_ids = projection["rewrite_source_event_ids"]
    if (
        not isinstance(reconciliation, dict)
        or reconciliation.get("schema")
        != adaptation_engine.SOURCE_INDEXED_REWRITE_RECONCILIATION_SCHEMA
        or reconciliation.get("expected_source_event_ids") != expected_ids
        or reconciliation.get("reconciled_source_event_ids") != expected_ids
        or reconciliation.get("source_fact_loss_count") != 0
        or reconciliation.get("provider_request_count") != 0
    ):
        raise RuntimeError("Adaptation live reconciliation verdict failed")
    evidence_path = workspace / LIVE_DIRECTORY / EVIDENCE_NAME
    full_chain_acceptance._atomic_write_json(evidence_path, {
        "schema": RECEIPT_SCHEMA,
        "source_events_sha256": projection["source_events_sha256"],
        "production_events_before_sha256": projection[
            "production_events_sha256"
        ],
        "selected_events_sha256": _canonical_sha256(selected_events),
        "rewrite_source_event_ids": expected_ids,
        "original_source_event_ids": reconciliation[
            "original_source_event_ids"
        ],
        "reconciled_source_event_ids": reconciliation[
            "reconciled_source_event_ids"
        ],
        "duplicate_count": reconciliation["duplicate_count"],
        "reconciliation_policy_sha256": reconciliation["policy_sha256"],
        "source_fact_loss_count": 0,
        "provider_request_count": 1,
    })
    return {
        **preflight,
        "status": "passed",
        "call_chain_verdict": "passed",
        "business_verdict": "passed",
        "provider_request_count": 1,
        "request_receipt_sha256": hashlib.sha256(
            request_receipt.path.read_bytes()
        ).hexdigest(),
        "evidence_path": evidence_path.relative_to(workspace).as_posix(),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "transport_status": transport["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    receipt_path = workspace / RECEIPT_NAME
    existing = _read_object(receipt_path) if receipt_path.is_file() else None
    if args.submit and existing and existing.get("status") in {
        "passed",
        "live_acceptance_failed",
        "submission_uncertain",
    }:
        raise RuntimeError(
            "Adaptation paid gate is terminal; resubmission is forbidden"
        )
    preflight = build_preflight(
        workspace,
        args.input,
        args.regression,
    )
    full_chain_acceptance._atomic_write_json(receipt_path, preflight)
    if not args.submit:
        return 0 if preflight["status"] == "preflight_passed" else 2
    try:
        receipt = execute_single_rewrite(
            workspace,
            args.input,
            preflight,
        )
    except BaseException as error:
        failed = {
            **preflight,
            "status": "live_acceptance_failed",
            "call_chain_verdict": "failed",
            "business_verdict": "not_evaluated",
            "safe_error_type": type(error).__name__,
            "provider_request_count": int(
                (workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME).is_file()
            ),
        }
        full_chain_acceptance._atomic_write_json(receipt_path, failed)
        raise
    full_chain_acceptance._atomic_write_json(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
