from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import adaptation_rewrite_live_acceptance as acceptance
from utils.provider_request_guard import (
    provider_request_completed,
    provider_request_started,
)


def _write_input(workspace: Path) -> tuple[Path, Path]:
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "turning_point",
            "dramatic_turn": True,
            "what": f"来源事件{event_id}",
            "start_state": "承受压力",
            "end_state": "完成动作",
            "micro_actions": [
                f"事件{event_id}来源动作{action_index}"
                for action_index in range(1, 8)
            ],
        }
        for event_id in range(1, 12)
    ]
    director_plan = {
        "schema": "honcut.director-plan.v1",
        "sequences": [{
            "sequence_id": "SEQ001",
            "scene_goal": "保持完整因果",
            "emotion_arc": "压力逐步释放",
            "visual_focus": "动作落点",
            "spatial_intent": "连续空间",
            "transition_intent": "动作承接",
        }],
    }
    contract = {
        "schema": acceptance.INPUT_SCHEMA,
        "source_events": events,
        "director_plan": director_plan,
        "target_duration_s": 36,
        "shot_duration_s": 12,
        "shot_policy": "continuity",
        "max_material_padding_ratio": 0.25,
        "delivery_overrun_ratio": 0,
    }
    input_path = workspace / "adaptation_rewrite_input.json"
    input_path.write_text(
        json.dumps({
            **contract,
            "contract_sha256": acceptance._canonical_sha256(contract),
        }),
        encoding="utf-8",
    )
    regression_path = workspace / "adaptation_rewrite_regression.json"
    regression_path.write_text(
        json.dumps({
            "schema": acceptance.REGRESSION_SCHEMA,
            "status": "passed",
            "git_commit": "a" * 40,
            "provider_request_count": 0,
        }),
        encoding="utf-8",
    )
    return input_path, regression_path


def _enable_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acceptance.full_chain_acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _required: {"valid": True, "missing": []},
    )
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "configured")


def test_preflight_freezes_one_rewrite_request_without_provider(
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
    assert receipt["projection"]["rewrite_source_event_ids"] == list(
        range(1, 12)
    )


def test_single_rewrite_uses_owner_once_and_persists_only_hash_evidence(
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
    live_input = acceptance._load_input(input_path)
    _events, production_events, _duration_plan, _projection = (
        acceptance._build_rewrite_contract(live_input)
    )
    calls = 0

    def fake_stream(**_kwargs):
        nonlocal calls
        calls += 1
        token = provider_request_started({
            "provider_family": "ark_text",
            "model": "fixture-model",
            "messages_sha256": "c" * 64,
        })
        response = {
            "schema": "honcut.source-indexed-screenplay-rewrite.v1",
            "events": [],
        }
        for event_id, event in enumerate(production_events, start=1):
            groups = event["production_action_rewrite"]["groups"]
            response["events"].append({
                "source_event_id": event_id,
                "production_actions": [
                    {
                        "production_action_index": group[
                            "production_action_index"
                        ],
                        "source_micro_action_indexes": group[
                            "source_micro_action_indexes"
                        ],
                        "rewritten_micro_action": "；".join(
                            group["source_actions"]
                        ),
                    }
                    for group in groups
                ],
                "narrative_purpose": "保留完整因果",
                "emotional_beat": "保持动作压力",
                "director_alignment": "对齐动作落点",
            })
        provider_request_completed(token, {
            "transport_status": "response_completed",
            "response_sha256": "d" * 64,
        })
        return json.dumps(response, ensure_ascii=False)

    monkeypatch.setattr(
        acceptance.adaptation_engine,
        "create_ark_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        acceptance.adaptation_engine,
        "call_llm_stream",
        fake_stream,
    )

    receipt = acceptance.execute_single_rewrite(
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
    assert "rewritten_micro_action" not in evidence_text
    assert "来源事件" not in evidence_text


def test_tampered_live_input_fails_before_provider(tmp_path: Path) -> None:
    input_path, _regression_path = _write_input(tmp_path)
    document = json.loads(input_path.read_text(encoding="utf-8"))
    document["target_duration_s"] = 35
    input_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        acceptance._load_input(input_path)
