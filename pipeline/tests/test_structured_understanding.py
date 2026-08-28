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
from runtime.structured_understanding import (  # noqa: E402
    StructuredUnderstandingExhausted,
    execute_structured_understanding,
)
from schemas.understanding import (  # noqa: E402
    CharacterUnderstandingBatch,
    ShotSemanticReview,
    StoryOrderUnderstanding,
    native_chat_json_schema_format,
    parse_structured_output,
)
from utils.semantic_contracts import bind_story_semantics  # noqa: E402


def _ark_response(
    *texts: str,
    response_status: str = "completed",
    error=None,
    incomplete_details=None,
    item_type: str = "message",
    role: str = "assistant",
    message_status: str = "completed",
    block_type: str = "output_text",
    extra_output=(),
    output_text: str | None = None,
):
    message = SimpleNamespace(
        type=item_type,
        role=role,
        status=message_status,
        content=[SimpleNamespace(type=block_type, text=text) for text in texts],
    )
    return SimpleNamespace(
        status=response_status,
        error=error,
        incomplete_details=incomplete_details,
        output=[message, *extra_output],
        output_text="".join(texts) if output_text is None else output_text,
    )


def test_structured_parser_rejects_unknown_fields_and_invalid_enums():
    raw = json.dumps({
        "verdict": "maybe",
        "issues": [],
        "confidence": 0.8,
        "free_form_escape_hatch": "must not cross the contract boundary",
    })

    with pytest.raises(ValidationError):
        parse_structured_output(raw, ShotSemanticReview)


@pytest.mark.parametrize(
    ("raw", "expected_issues"),
    [
        ('{"verdict":"pass" "issues":[],"confidence":0.8}', []),
        ('{"verdict":"pass","issues":[],"confidence":0.8', []),
        ('{"verdict":"pass","confidence":0.8,"issues":["minor"}', ["minor"]),
        ('```json\n{"verdict":"pass" "issues":[],"confidence":0.8}\n```', []),
    ],
)
def test_structured_parser_repairs_single_document_json_syntax(
    raw,
    expected_issues,
):
    parsed = parse_structured_output(raw, ShotSemanticReview)

    assert parsed.verdict == "pass"
    assert parsed.issues == expected_issues
    assert parsed.confidence == 0.8


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict":"pass","issues":[],"confidence":0.8} trailing prose',
        (
            '{"verdict":"pass","issues":[],"confidence":0.8}'
            '{"verdict":"fail","issues":[],"confidence":0.1}'
        ),
        '{"verdict":"pass","issues":[',
        '{"verdict":"pass","confidence":0.8,"issues":[}',
        '{"verdict":"pass","issues":[],"confidence":tru',
    ],
)
def test_structured_parser_rejects_unsafe_document_salvage(raw):
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        parse_structured_output(raw, ShotSemanticReview)


def test_multimodal_client_sends_native_json_schema_and_returns_typed_model(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _ark_response(json.dumps({
                "suggested_order": ["S01", "S02"],
                "narrative_consistent": True,
                "issues": [],
            }))

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


def test_multimodal_client_ignores_the_sdk_output_text_aggregate(tmp_path):
    canonical = '{"verdict":"pass","issues":[],"confidence":0.9}'

    class Responses:
        @staticmethod
        def create(**_kwargs):
            return _ark_response(
                canonical,
                output_text=canonical + " trailing SDK aggregate",
            )

    image = tmp_path / "frame.png"
    image.write_bytes(b"image")
    client = ArkMultimodalClient(
        api_key="test",
        client=SimpleNamespace(responses=Responses()),
        media_url_resolver=lambda _path: "https://tos.example/frame.png",
    )

    result = client.review_structured(
        [image],
        "Review.",
        ShotSemanticReview,
    )

    assert result.verdict == "pass"


def test_multimodal_client_rejects_multiple_output_text_blocks_without_raw_text():
    private_text = "PRIVATE_SECOND_MODEL_BLOCK"
    response = _ark_response(
        '{"verdict":"pass","issues":[],"confidence":0.9}',
        private_text,
    )

    with pytest.raises(json.JSONDecodeError, match="exactly one output_text") as raised:
        ark_multimodal_client._extract_single_completed_output_text(response)

    assert private_text not in str(raised.value)
    assert private_text not in raised.value.doc
    assert "output_text_utf8_lengths" in str(raised.value)
    assert "output_text_sha256" in str(raised.value)


def test_multimodal_client_accepts_the_documented_dict_envelope_shape():
    text = '{"verdict":"pass","issues":[],"confidence":0.9}'
    response = {
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }],
    }

    assert (
        ark_multimodal_client._extract_single_completed_output_text(response)
        == text
    )


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            _ark_response(
                response_status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            "response status must be completed",
        ),
        (
            _ark_response(
                error={"message": "PRIVATE_PROVIDER_ERROR"},
            ),
            "must not contain error",
        ),
        (
            _ark_response(
                incomplete_details={"reason": "PRIVATE_INCOMPLETE_DETAIL"},
            ),
            "must not contain incomplete details",
        ),
        (
            _ark_response(item_type="reasoning"),
            "exactly one assistant message",
        ),
        (
            _ark_response(item_type="PRIVATE_PROVIDER_ITEM_TYPE"),
            "exactly one assistant message",
        ),
        (
            _ark_response(
                "first",
                extra_output=(_ark_response("second").output[0],),
            ),
            "exactly one assistant message",
        ),
        (
            _ark_response("PRIVATE", role="user"),
            "message role must be assistant",
        ),
        (
            _ark_response("PRIVATE", message_status="incomplete"),
            "message status must be completed",
        ),
        (
            _ark_response("PRIVATE", block_type="refusal"),
            "exactly one output_text",
        ),
        (
            _ark_response("   "),
            "output_text must be non-empty",
        ),
    ],
)
def test_multimodal_client_rejects_noncanonical_responses_without_raw_text(
    response,
    reason,
):
    with pytest.raises(json.JSONDecodeError, match=reason) as raised:
        ark_multimodal_client._extract_single_completed_output_text(response)

    serialized_error = str(raised.value)
    assert "PRIVATE" not in serialized_error
    assert "PRIVATE" not in raised.value.doc
    assert "output_count" in serialized_error


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


def test_runtime_replays_one_rejected_structured_value_without_salvaging_json():
    calls = 0

    def review_operation():
        nonlocal calls
        calls += 1
        raw = (
            '{"verdict":"pass","issues":'
            if calls == 1
            else '{"verdict":"pass","issues":[],"confidence":0.9}'
        )
        return parse_structured_output(raw, ShotSemanticReview)

    result, receipt = execute_structured_understanding(review_operation)

    assert result.verdict == "pass"
    assert calls == 2
    assert [attempt["status"] for attempt in receipt["attempts"]] == [
        "schema_rejected",
        "succeeded",
    ]


def test_runtime_owns_one_replay_for_a_rejected_ark_response_envelope():
    calls = 0

    def review_operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            response = _ark_response(
                "PRIVATE_REJECTED_MODEL_TEXT",
                extra_output=(_ark_response(item_type="reasoning").output[0],),
            )
        else:
            response = _ark_response(
                '{"verdict":"pass","issues":[],"confidence":0.9}'
            )
        raw = ark_multimodal_client._extract_single_completed_output_text(response)
        return parse_structured_output(raw, ShotSemanticReview)

    result, receipt = execute_structured_understanding(review_operation)

    assert result.verdict == "pass"
    assert calls == 2
    assert [attempt["status"] for attempt in receipt["attempts"]] == [
        "schema_rejected",
        "succeeded",
    ]
    assert "PRIVATE_REJECTED_MODEL_TEXT" not in json.dumps(receipt)


def test_runtime_does_not_retry_non_schema_understanding_failure():
    calls = 0

    def review_operation():
        nonlocal calls
        calls += 1
        raise RuntimeError("authentication rejected")

    with pytest.raises(RuntimeError, match="authentication rejected"):
        execute_structured_understanding(review_operation)

    assert calls == 1


def test_runtime_schema_failure_receipt_never_persists_rejected_payload():
    def review_operation():
        return parse_structured_output(
            json.dumps({
                "verdict": "invalid-secret-payload",
                "issues": ["PRIVATE_MODEL_TEXT"],
                "confidence": 0.9,
            }),
            ShotSemanticReview,
        )

    with pytest.raises(StructuredUnderstandingExhausted) as raised:
        execute_structured_understanding(review_operation, max_attempts=1)

    serialized = json.dumps(raised.value.receipt, ensure_ascii=False)
    assert "PRIVATE_MODEL_TEXT" not in serialized
    assert "invalid-secret-payload" not in serialized


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

    assert ledger["schema"] == "honcut.semantic-understanding.v2"
    assert events[0]["character_ids"] == ["operator", "intruder_3"]
    assert events[1]["character_ids"] == ["operator", "intruder_3"]
    assert events[0]["participant_refs"][1]["ref_id"] == (
        events[1]["participant_refs"][1]["ref_id"]
    )
    assert characters[1]["source_identity_ref_ids"] == [
        events[0]["participant_refs"][1]["ref_id"]
    ]
    assert ledger["source_mentions"] == [
        {
            "ref_id": events[0]["participant_refs"][0]["ref_id"],
            "text": "操作员",
            "language": "zh",
            "character_id": "operator",
        },
        {
            "ref_id": events[0]["participant_refs"][1]["ref_id"],
            "text": "第三名入侵者",
            "language": "zh",
            "character_id": "intruder_3",
        },
    ]
    assert ledger["entities"][0]["machine_semantics"] == {
        "entity_type": "character",
        "gender": "unknown",
        "role": "unknown",
    }


def test_text_semantic_ledger_keeps_dialogue_verbatim_and_uses_controlled_enums():
    original_line = {
        "dialogue_id": "D001",
        "speaker": "男子",
        "line": "别动，这是原始对白。",
        "confidence": 1.0,
        "evidence": "source",
    }
    events = [{
        "event_id": 1,
        "who": ["男性"],
        "lines": [dict(original_line)],
    }]
    characters = [{
        "id": "lead_01",
        "name": "男子",
        "aliases": ["男性"],
        "role": "protagonist",
        "appearance": {"gender": "male"},
    }]

    ledger = bind_story_semantics(events, characters)

    assert events[0]["lines"] == [original_line]
    assert ledger["source_mentions"] == [{
        "ref_id": events[0]["participant_refs"][0]["ref_id"],
        "text": "男性",
        "language": "zh",
        "character_id": "lead_01",
    }]
    assert ledger["entities"][0]["machine_semantics"] == {
        "entity_type": "character",
        "gender": "male",
        "role": "protagonist",
    }


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
