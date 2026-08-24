"""Structured text/vision contracts shared across production understanding stages."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import clients.ark_multimodal_client as ark_multimodal_client  # noqa: E402
from clients.ark_multimodal_client import ArkMultimodalClient, review_as  # noqa: E402
from phases.phase1 import character_discoverer  # noqa: E402
from phases.phase8.frame_analysis import decide_shot_action  # noqa: E402
from schemas.understanding import (  # noqa: E402
    CharacterUnderstandingBatch,
    ShotSemanticReview,
    StoryOrderUnderstanding,
    native_chat_json_schema_format,
    parse_structured_output,
)
from utils.semantic_contracts import bind_story_semantics  # noqa: E402


def test_structured_parser_rejects_unknown_fields_and_invalid_enums():
    raw = json.dumps({
        "verdict": "maybe",
        "issues": [],
        "confidence": 0.8,
        "free_form_escape_hatch": "must not cross the contract boundary",
    })

    with pytest.raises(ValidationError):
        parse_structured_output(raw, ShotSemanticReview)


def test_multimodal_client_sends_native_json_schema_and_returns_typed_model(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=json.dumps({
                    "suggested_order": ["S01", "S02"],
                    "narrative_consistent": True,
                    "issues": [],
                }),
                output=[],
            )

    client = ArkMultimodalClient(
        api_key="test",
        client=SimpleNamespace(responses=Responses()),
        media_url_resolver=lambda _path: "https://tos.example/frame.png",
    )

    result = client.review_structured(
        [image],
        "Review the ordered frames.",
        StoryOrderUnderstanding,
    )

    assert isinstance(result, StoryOrderUnderstanding)
    assert result.suggested_order == ["S01", "S02"]
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    assert captured["text"]["format"]["schema"]["additionalProperties"] is False


def test_multimodal_client_ignores_ambient_socks_proxy(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:65535")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:65535")
    monkeypatch.setattr(ark_multimodal_client, "OpenAI", FakeOpenAI)

    ArkMultimodalClient(api_key="test-key")

    http_client = captured["http_client"]
    try:
        assert http_client._trust_env is False
    finally:
        http_client.close()


def test_test_reviewer_adapter_still_enforces_the_business_schema(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")

    class LegacyTestReviewer:
        def review(self, _paths, _prompt):
            return '{"verdict":"pass","issues":[],"confidence":0.9}'

    result = review_as(
        LegacyTestReviewer(),
        [image],
        "Review.",
        ShotSemanticReview,
    )

    assert isinstance(result, ShotSemanticReview)


def test_character_understanding_schema_is_native_strict_and_enveloped():
    response_format = native_chat_json_schema_format(CharacterUnderstandingBatch)

    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["characters"]
    with pytest.raises(ValidationError):
        parse_structured_output("[]", CharacterUnderstandingBatch)


def test_unavailable_visual_understanding_cannot_keep_a_paid_shot():
    decision = decide_shot_action(
        8,
        [],
        [],
        [],
        {"verdict": "unavailable", "issues": [], "error": "schema mismatch"},
    )

    assert decision["action"] == "reshoot"


def test_text_semantic_ledger_binds_mentions_to_stable_character_ids():
    events = [
        {
            "event_id": 1,
            "action_unit_id": "AU001",
            "who": ["操作员", "第三名入侵者"],
            "micro_actions": ["操作员格挡第三名入侵者"],
        },
        {
            "event_id": 2,
            "action_unit_id": "AU002",
            "who": ["操作员", "第三名入侵者"],
            "micro_actions": ["操作员完成控制"],
        },
    ]
    characters = [
        {"id": "operator", "name": "操作员", "aliases": []},
        {
            "id": "intruder_3",
            "name": "第三名入侵者",
            "aliases": ["机械拳套入侵者"],
        },
    ]

    ledger = bind_story_semantics(events, characters)

    assert ledger["schema"] == "honcut.semantic-understanding.v1"
    assert events[0]["character_ids"] == ["operator", "intruder_3"]
    assert events[1]["character_ids"] == ["operator", "intruder_3"]
    assert events[0]["participant_refs"][1]["ref_id"] == (
        events[1]["participant_refs"][1]["ref_id"]
    )
    assert characters[1]["source_identity_ref_ids"] == [
        events[0]["participant_refs"][1]["ref_id"]
    ]


def test_text_semantic_ledger_fails_closed_on_unbound_participant():
    events = [{"event_id": 1, "who": ["未绑定巡检员"], "micro_actions": []}]
    characters = [{"id": "operator", "name": "操作员", "aliases": []}]

    with pytest.raises(ValueError, match="unbound participant"):
        bind_story_semantics(events, characters)


def test_explicit_identity_declaration_promotes_generic_label_without_whitelist():
    declared = character_discoverer._collect_character_stats([{
        "id": 1,
        "who": ["观察者"],
        "source_excerpt": "代号“观察者”的男子抬起芯片。",
    }])
    generic = character_discoverer._collect_character_stats([{
        "id": 1,
        "who": ["观察者"],
        "source_excerpt": "观察者注意到远处的灯光。",
    }])

    assert list(character_discoverer._filter_non_human_characters(declared)) == [
        "观察者"
    ]
    assert "明示该称呼为稳定身份" in character_discoverer._build_character_context(
        declared
    )
    assert character_discoverer._filter_non_human_characters(generic) == {}
