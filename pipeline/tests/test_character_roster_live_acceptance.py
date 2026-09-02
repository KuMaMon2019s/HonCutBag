from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import character_roster_live_acceptance as acceptance
from phases.phase1.character_roster import (
    compile_character_roster,
    reconcile_character_observations,
)
from utils.provider_request_guard import (
    provider_request_completed,
    provider_request_started,
)


def _write_inputs(workspace: Path) -> tuple[Path, Path]:
    events = [
        {
            "id": 1,
            "sequence_id": "SEQ001",
            "continuity_before": "cut",
            "who": ["年轻男性"],
            "source_excerpt": "年轻男性停在入口。",
            "what": "年轻男性停在入口",
        },
        {
            "id": 2,
            "sequence_id": "SEQ001",
            "continuity_before": "continuous",
            "who": ["守卫1", "守卫2", "守卫3"],
            "source_excerpt": "三名守卫同时出现。",
            "what": "三名守卫同时出现",
        },
        *[
            {
                "id": ordinal + 2,
                "sequence_id": "SEQ001",
                "continuity_before": "continuous",
                "who": [
                    "男子" if ordinal == 3 else "年轻男性",
                    f"第{label}名守卫",
                ],
                "source_excerpt": (
                    f"第{label}名守卫靠近，男子保持警戒。"
                    if ordinal >= 2
                    else f"第{label}名守卫靠近年轻男性。"
                ),
                "what": f"第{label}名守卫保持警戒",
            }
            for ordinal, label in enumerate(("一", "二", "三"), 1)
        ],
    ]
    unsigned = {"schema": acceptance.INPUT_SCHEMA, "events": events}
    events_path = workspace / "events_contract.json"
    events_path.write_text(
        json.dumps({
            **unsigned,
            "events_sha256": acceptance._canonical_sha256(unsigned),
        }),
        encoding="utf-8",
    )
    expectations_path = workspace / "acceptance_expectations.json"
    expectations_path.write_text(
        json.dumps({
            "schema": "honcut.full-chain-acceptance-expectations.v2",
            "expected_duration_s": 36,
            "expected_character_entities": 2,
            "expected_character_instances": 4,
            "entity_expectations": [
                {
                    "expectation_id": "lead",
                    "source_mentions_any": ["年轻男性", "男子"],
                    "same_instance_mentions": ["年轻男性", "男子"],
                    "instance_count": 1,
                    "visual_facts": {},
                },
                {
                    "expectation_id": "guards",
                    "source_mentions_any": ["三名守卫"],
                    "instance_count": 3,
                    "visual_facts": {},
                },
            ],
            "required_events": ["停在入口", "三名守卫同时出现"],
            "visual_facts": {},
        }),
        encoding="utf-8",
    )
    return events_path, expectations_path


def _enable_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acceptance.full_chain_acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance.full_chain_acceptance,
        "_regression_evidence",
        lambda _workspace, _commit: {
            "status": "passed",
            "path": "regression_acceptance.json",
            "sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _required: {"valid": True, "missing": []},
    )
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "configured")


def test_preflight_validates_source_driven_two_entity_four_instance_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path, expectations_path = _write_inputs(tmp_path)
    _enable_preflight(monkeypatch)

    receipt = acceptance.build_preflight(
        tmp_path,
        events_path,
        expectations_path,
    )

    assert receipt["status"] == "preflight_passed"
    assert receipt["provider_request_count"] == 0
    assert receipt["exact_request_limit"] == 1
    assert set(receipt["entity_matches"]) == {"lead", "guards"}


def test_preflight_rejects_tampered_external_event_contract(
    tmp_path: Path,
) -> None:
    events_path, expectations_path = _write_inputs(tmp_path)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["events"][0]["what"] = "changed"
    events_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        acceptance.build_preflight(tmp_path, events_path, expectations_path)


def test_request_receipt_is_one_shot_and_drops_sensitive_fields(
    tmp_path: Path,
) -> None:
    guard = acceptance.CharacterRosterRequestReceipt(tmp_path)
    token = guard.before({
        "provider_family": "ark_text",
        "model": "fixture-model",
        "messages_sha256": "a" * 64,
        "prompt": "must-not-persist",
        "url": "https://example.invalid/private",
        "secret": "must-not-persist-either",
    })
    guard.after(token, {
        "transport_status": "response_completed",
        "response_sha256": "b" * 64,
        "response": "must-not-persist",
    })

    receipt_text = guard.path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "provider_completed"
    assert receipt["safe_payload"] == {
        "provider_family": "ark_text",
        "model": "fixture-model",
        "messages_sha256": "a" * 64,
    }
    assert "must-not-persist" not in receipt_text
    assert "example.invalid" not in receipt_text
    with pytest.raises(RuntimeError, match="resubmission forbidden"):
        guard.before({"provider_family": "ark_text"})


def test_single_observation_accepts_deterministic_missing_entity_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path, expectations_path = _write_inputs(tmp_path)
    events, events_sha256 = acceptance._load_events_contract(events_path)
    roster = compile_character_roster(events)
    lead = next(entity for entity in roster["entities"] if entity["instance_count"] == 1)
    characters, diagnostics = reconcile_character_observations(
        [{
            "id": lead["entity_id"],
            "name": lead["display_name"],
            "aliases": [],
            "role": "protagonist",
            "appearance": {
                "gender": "unknown",
                "age_range": "adult",
                "height": "average",
                "build": "average",
                "hair": "short hair",
                "face": "fictional face",
                "clothing": "plain clothing",
                "interaction_props": [],
                "identity_props": [],
                "distinguishing": "",
                "summary": "fictional traveler",
                "variants": [],
            },
            "personality": {"traits": [], "speech_style": "", "motivation": ""},
            "style": "cinematic",
            "negative": "",
            "size": "2K",
            "first_appearance": 1,
            "appearance_count": 1,
            "relationships": [],
        }],
        roster,
        semantic_qa_enabled=False,
    )
    calls = 0

    def fake_discover(_events, **_kwargs):
        nonlocal calls
        calls += 1
        token = provider_request_started({
            "provider_family": "ark_text",
            "model": "fixture-model",
            "messages_sha256": "c" * 64,
        })
        provider_request_completed(token, {
            "transport_status": "response_completed",
            "response_sha256": "d" * 64,
        })
        return {
            "characters": characters,
            "character_roster": roster,
            "character_roster_sha256": roster["roster_sha256"],
            "semantic_diagnostics": diagnostics,
        }

    monkeypatch.setattr(acceptance, "discover_characters", fake_discover)
    receipt = acceptance.execute_single_observation(
        tmp_path,
        events_path,
        expectations_path,
        {
            "status": "preflight_passed",
            "source": {"events_sha256": events_sha256},
            "provider_request_count": 0,
        },
    )

    assert calls == 1
    assert receipt["status"] == "passed"
    assert receipt["provider_request_count"] == 1
    evidence = json.loads(
        (tmp_path / acceptance.LIVE_DIRECTORY / "CHARACTER_ROSTER_RESULT.json")
        .read_text(encoding="utf-8")
    )
    assert evidence["character_entities"] == 2
    assert evidence["character_instances"] == 4
    assert evidence["identity_reconciliation_count"] == 4
    assert "model_entity_missing" in evidence["semantic_diagnostic_codes"]
