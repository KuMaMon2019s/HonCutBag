"""Acceptance tests for Phase 3 visual-QA paid admission."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.scripts.phase3_visual_qa_paid_admission import (
    ADMISSION_SCHEMA,
    INVENTORY_SCHEMA,
    REGRESSION_SCHEMA,
    REPLAY_SCHEMA,
    write_paid_admission,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(tmp_path: Path, *, candidate: str) -> tuple[Path, Path, Path]:
    regression = tmp_path / "regression.json"
    replay = tmp_path / "replay.json"
    inventory = tmp_path / "inventory.json"
    checks: dict[str, dict[str, object]] = {
        name: {"status": "passed"}
        for name in (
            "targeted_pytest",
            "full_test",
            "lint",
            "diff_check",
            "offline_phase1_9",
        )
    }
    checks["recovery_matrix"] = {
        "status": "passed",
        "rounds": 10,
        "provider_request_count": 0,
        "stable": True,
    }
    _write(regression, {
        "schema": REGRESSION_SCHEMA,
        "status": "passed",
        "candidate_commit": candidate,
        "provider_request_count": 0,
        "checks": checks,
    })
    _write(replay, {
        "schema": REPLAY_SCHEMA,
        "status": "passed",
        "candidate_commit": candidate,
        "provider_request_count": 0,
        "provider_submission_count": 0,
        "historical_run_mutated": False,
        "source": {"historical_status": "live_acceptance_failed"},
    })
    _write(inventory, {
        "schema": INVENTORY_SCHEMA,
        "entries": [{"classification": "ledgered_policy"}],
    })
    return regression, replay, inventory


def test_paid_admission_requires_every_zero_request_gate(tmp_path: Path):
    candidate = "a" * 40
    regression, replay, inventory = _evidence(tmp_path, candidate=candidate)
    output = tmp_path / "admission.json"

    receipt = write_paid_admission(
        candidate_commit=candidate,
        regression_receipt_path=regression,
        replay_receipt_path=replay,
        inventory_path=inventory,
        output_path=output,
    )

    assert receipt["schema"] == ADMISSION_SCHEMA
    assert receipt["status"] == "pending_live_acceptance"
    assert receipt["admission"] == "paid_admission"
    assert receipt["provider_request_count"] == 0
    assert receipt["no_submit_preflight"]["submit_authorized"] is False
    assert receipt["no_submit_preflight"]["provider_family_limits"] == {
        "ark_text": 0,
        "seedream": 0,
        "multimodal_vlm": 1,
        "seedance": 0,
        "tos": 0,
    }
    assert receipt["full_chain_36s"]["status"] == "separate_authorization_required"
    assert receipt["automatic_retry"] is False
    assert receipt["automatic_redraw"] is False
    assert receipt["automatic_reshoot"] is False


def test_failed_gate_writes_blocked_receipt_without_expansion(tmp_path: Path):
    candidate = "b" * 40
    regression, replay, inventory = _evidence(tmp_path, candidate=candidate)
    value = json.loads(regression.read_text(encoding="utf-8"))
    value["checks"]["recovery_matrix"]["stable"] = False
    _write(regression, value)

    receipt = write_paid_admission(
        candidate_commit=candidate,
        regression_receipt_path=regression,
        replay_receipt_path=replay,
        inventory_path=inventory,
        output_path=tmp_path / "blocked.json",
    )

    assert receipt["status"] == "paid_admission_blocked"
    assert receipt["first_failure_signature"] == "recovery_matrix_not_stable"
    assert receipt["provider_request_count"] == 0
    assert receipt["automatic_budget_expansion"] is False


def test_candidate_change_invalidates_paid_admission(tmp_path: Path):
    candidate = "c" * 40
    regression, replay, inventory = _evidence(tmp_path, candidate=candidate)

    receipt = write_paid_admission(
        candidate_commit="d" * 40,
        regression_receipt_path=regression,
        replay_receipt_path=replay,
        inventory_path=inventory,
        output_path=tmp_path / "candidate-changed.json",
    )

    assert receipt["status"] == "paid_admission_blocked"
    assert receipt["first_failure_signature"] == "regression_candidate_mismatch"
    assert receipt["provider_submission_count"] == 0
