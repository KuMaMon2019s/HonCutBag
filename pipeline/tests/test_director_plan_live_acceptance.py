from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import director_plan_live_acceptance as acceptance
from utils.provider_request_guard import (
    provider_request_completed,
    provider_request_started,
)


def _event() -> dict:
    return {
        "sequence_id": "SEQ001",
        "event_role": "scene_setup",
        "who": ["CHAR001"],
        "what": "A fictional character enters a station.",
        "micro_actions": ["looks toward a train"],
    }


def _sequence(*, alternate: bool = False) -> dict:
    return {
        "sequence_id": "SEQ001",
        "scene_goal": (
            "alternate proposal" if alternate else "establish the arrival"
        ),
        "emotion_arc": "calm to alert",
        "visual_focus": "the character and the arriving train",
        "spatial_intent": "character foreground, train background",
        "transition_intent": "follow the character toward the door",
    }


def _write_input(workspace: Path) -> tuple[Path, Path]:
    input_path = workspace / "input" / "director_live_input.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": acceptance.INPUT_SCHEMA,
        "source_events": [_event()],
    }
    input_path.write_text(json.dumps({
        **contract,
        "contract_sha256": acceptance._canonical_sha256(contract),
    }), encoding="utf-8")
    regression_path = workspace / "director_regression.json"
    regression_path.write_text(json.dumps({
        "schema": acceptance.REGRESSION_SCHEMA,
        "status": "passed",
        "git_commit": "test-commit",
        "provider_request_count": 0,
    }), encoding="utf-8")
    return input_path, regression_path


def _enable_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acceptance.full_chain_acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "test-commit", "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _requirements: {"valid": True},
    )
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "test-key")


def test_preflight_freezes_owner_request_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, regression_path = _write_input(tmp_path)
    _enable_preflight(monkeypatch)

    receipt = acceptance.build_preflight(
        tmp_path,
        input_path,
        regression_path,
    )

    assert receipt["status"] == "preflight_passed"
    assert receipt["provider_request_count"] == 0
    assert receipt["exact_request_limit"] == 1
    assert receipt["projection"]["expected_sequence_ids"] == ["SEQ001"]
    assert receipt["projection"]["reconciliation_policy_sha256"] == (
        acceptance.director_planner.DIRECTOR_PLAN_RECONCILIATION_POLICY_SHA256
    )


def test_single_director_uses_owner_once_and_persists_only_hash_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, regression_path = _write_input(tmp_path)
    _enable_preflight(monkeypatch)
    preflight = acceptance.build_preflight(
        tmp_path,
        input_path,
        regression_path,
    )
    calls = 0

    def fake_stream(**kwargs):
        nonlocal calls
        calls += 1
        assert acceptance._canonical_sha256(kwargs["messages"]) == (
            preflight["projection"]["messages_sha256"]
        )
        assert acceptance._canonical_sha256(kwargs["response_format"]) == (
            preflight["projection"]["response_format_sha256"]
        )
        token = provider_request_started({
            "provider_family": "ark_text",
            "model": "fixture-model",
            "messages_sha256": "c" * 64,
        })
        provider_request_completed(token, {
            "transport_status": "response_completed",
            "response_sha256": "d" * 64,
        })
        return json.dumps({
            "schema": acceptance.director_planner.DIRECTOR_PLAN_SCHEMA,
            "sequences": [_sequence(), _sequence(alternate=True)],
        })

    monkeypatch.setattr(
        acceptance.director_planner,
        "create_ark_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        acceptance.director_planner,
        "get_api_key",
        lambda _name: "test-key",
    )
    monkeypatch.setattr(
        acceptance.director_planner,
        "call_llm_stream",
        fake_stream,
    )

    receipt = acceptance.execute_single_director(
        tmp_path,
        input_path,
        preflight,
    )

    assert calls == 1
    assert receipt["status"] == "passed"
    assert receipt["provider_request_count"] == 1
    evidence_text = (
        tmp_path / acceptance.LIVE_DIRECTORY / acceptance.EVIDENCE_NAME
    ).read_text(encoding="utf-8")
    assert "alternate proposal" not in evidence_text
    assert "establish the arrival" not in evidence_text
    evidence = json.loads(evidence_text)
    assert evidence["duplicate_count"] == 1
    assert evidence["reconciled_sequence_ids"] == ["SEQ001"]


def test_tampered_live_input_fails_before_provider(tmp_path: Path) -> None:
    input_path, _regression_path = _write_input(tmp_path)
    document = json.loads(input_path.read_text(encoding="utf-8"))
    document["source_events"][0]["sequence_id"] = "SEQ009"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        acceptance._load_input(input_path)
