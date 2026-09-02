#!/usr/bin/env python3
"""Build a zero-submit paid-admission receipt for the Phase 3 QA fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REGRESSION_SCHEMA = "honcut.phase3-visual-qa-regression.v1"
REPLAY_SCHEMA = "honcut.prop-detail-replay-acceptance.v1"
INVENTORY_SCHEMA = "honcut.phase1-5-visual-gate-inventory.v1"
ADMISSION_SCHEMA = "honcut.phase3-visual-qa-paid-admission.v1"
REQUIRED_CHECKS = (
    "targeted_pytest",
    "full_test",
    "lint",
    "diff_check",
    "offline_phase1_9",
    "recovery_matrix",
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid admission evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"admission evidence must be an object: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _first_failure(
    *,
    candidate_commit: str,
    regression: dict[str, Any],
    replay: dict[str, Any],
    inventory: dict[str, Any],
) -> str | None:
    if regression.get("schema") != REGRESSION_SCHEMA:
        return "regression_schema_mismatch"
    if regression.get("candidate_commit") != candidate_commit:
        return "regression_candidate_mismatch"
    if regression.get("status") != "passed":
        return "regression_not_passed"
    if regression.get("provider_request_count") != 0:
        return "regression_provider_request_count_nonzero"
    checks = regression.get("checks")
    if not isinstance(checks, dict):
        return "regression_checks_missing"
    for name in REQUIRED_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict) or check.get("status") != "passed":
            return f"regression_check_failed:{name}"
    recovery = checks["recovery_matrix"]
    if (
        recovery.get("rounds") != 10
        or recovery.get("provider_request_count") != 0
        or recovery.get("stable") is not True
    ):
        return "recovery_matrix_not_stable"
    if replay.get("schema") != REPLAY_SCHEMA:
        return "replay_schema_mismatch"
    if replay.get("candidate_commit") != candidate_commit:
        return "replay_candidate_mismatch"
    if replay.get("status") not in {"passed", "acceptable_deviation"}:
        return "replay_not_accepted"
    if (
        replay.get("provider_request_count") != 0
        or replay.get("provider_submission_count") != 0
    ):
        return "replay_provider_count_nonzero"
    if replay.get("historical_run_mutated") is not False:
        return "historical_run_mutation_not_disproved"
    source = replay.get("source")
    if not isinstance(source, dict) or source.get("historical_status") != "live_acceptance_failed":
        return "historical_run_status_changed"
    if inventory.get("schema") != INVENTORY_SCHEMA:
        return "inventory_schema_mismatch"
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        return "inventory_empty"
    if any(not isinstance(entry, dict) or not entry.get("classification") for entry in entries):
        return "inventory_unclassified_entry"
    return None


def write_paid_admission(
    *,
    candidate_commit: str,
    regression_receipt_path: Path,
    replay_receipt_path: Path,
    inventory_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write a hash-bound admission decision without making network requests."""
    regression = _read_object(regression_receipt_path)
    replay = _read_object(replay_receipt_path)
    inventory = _read_object(inventory_path)
    evidence = {
        "regression": {
            "path": regression_receipt_path.name,
            "sha256": _file_sha256(regression_receipt_path),
        },
        "replay": {
            "path": replay_receipt_path.name,
            "sha256": _file_sha256(replay_receipt_path),
        },
        "visual_gate_inventory": {
            "path": inventory_path.name,
            "sha256": _file_sha256(inventory_path),
        },
    }
    failure = _first_failure(
        candidate_commit=candidate_commit,
        regression=regression,
        replay=replay,
        inventory=inventory,
    )
    stable: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA,
        "status": "paid_admission_blocked" if failure else "pending_live_acceptance",
        "admission": "blocked" if failure else "paid_admission",
        "candidate_commit": candidate_commit,
        "evidence": evidence,
        "first_failure_signature": failure,
        "provider_request_count": 0,
        "provider_submission_count": 0,
        "no_submit_preflight": {
            "scope": "phase3_prop_detail_single_live_gate",
            "provider_family_limits": {
                "ark_text": 0,
                "seedream": 0,
                "multimodal_vlm": 1,
                "seedance": 0,
                "tos": 0,
            },
            "submit_authorized": False,
        },
        "full_chain_36s": {
            "status": "separate_authorization_required",
            "stage0_required": True,
        },
        "automatic_retry": False,
        "automatic_redraw": False,
        "automatic_rereview": False,
        "automatic_reshoot": False,
        "automatic_task_expansion": False,
        "automatic_budget_expansion": False,
    }
    stable["integrity_sha256"] = _canonical_sha256(stable)
    if output_path.exists():
        existing = _read_object(output_path)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        if comparable != stable:
            raise RuntimeError("existing paid-admission receipt disagrees with evidence")
        return existing
    receipt = {**stable, "created_at": datetime.now(UTC).isoformat()}
    _atomic_json(output_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the Phase 3 visual-QA no-submit paid-admission receipt"
    )
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_paid_admission(
        candidate_commit=args.candidate_commit,
        regression_receipt_path=args.regression_receipt,
        replay_receipt_path=args.replay_receipt,
        inventory_path=args.inventory,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["admission"] == "paid_admission" else 1


if __name__ == "__main__":
    raise SystemExit(main())
