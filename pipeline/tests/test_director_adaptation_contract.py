"""Director intent must be a first-class input to cinematic adaptation."""

from __future__ import annotations

import json

import pytest

from phases.phase1 import adaptation_engine, director_planner
from phases.phase1.phase1_pipeline import run_phase1


def _events() -> list[dict]:
    return [
        {
            "sequence_id": "SEQ001",
            "event_role": "scene_setup",
            "who": ["林夏"],
            "where": "空旷控制室",
            "what": "林夏独自检查异常信号",
            "emotion": "平静转为不安",
            "micro_actions": ["抬眼看向屏幕"],
            "generation_action_unit_count": 1,
        },
        {
            "sequence_id": "SEQ002",
            "event_role": "turning_point",
            "who": ["林夏"],
            "where": "空旷控制室",
            "what": "屏幕映出与林夏相同的面孔",
            "emotion": "疑惑转为恐惧",
            "micro_actions": ["后退半步"],
            "generation_action_unit_count": 1,
        },
    ]


def _director_plan(*, second_goal: str = "揭示主角可能并非真正的人类") -> dict:
    return {
        "schema": "honcut.director-plan.v1",
        "sequences": [
            {
                "sequence_id": "SEQ001",
                "scene_goal": "建立人物孤独感",
                "emotion_arc": "平静 → 不安",
                "visual_focus": "人物与巨大空间的比例",
                "spatial_intent": "主角置于画面边缘，环境形成压迫",
                "transition_intent": "以主角视线切向屏幕",
            },
            {
                "sequence_id": "SEQ002",
                "scene_goal": second_goal,
                "emotion_arc": "疑惑 → 恐惧",
                "visual_focus": "主角面部与屏幕中相同面孔的对照",
                "spatial_intent": "主角前景，屏幕位于右后方形成纵深",
                "transition_intent": "以屏幕冷光收束",
            },
        ],
    }


def test_combined_phase1_delegates_event_aware_director_to_screenwriter(tmp_path):
    calls: list[str] = []
    events = _events()

    def director_runner(received_events, output_dir, dry_run):
        calls.append("director")
        assert received_events == events
        assert output_dir == tmp_path
        assert dry_run is False
        return {"status": "done", "plan": _director_plan()}

    def screenwriter_runner(
        _text,
        output_dir,
        _duration,
        _dry_run,
        *,
        _director_runner,
        **_kwargs,
    ):
        calls.append("events")
        director = _director_runner(events, output_dir, False)
        calls.append("adaptation")
        return {"status": "done", "director": director}

    result = run_phase1(
        "synthetic screenplay",
        tmp_path,
        30,
        False,
        _director_runner=director_runner,
        _screenwriter_runner=screenwriter_runner,
    )

    assert calls == ["events", "director", "adaptation"]
    assert result["director"]["plan"]["schema"] == "honcut.director-plan.v1"


def test_director_planner_consumes_sequences_and_rejects_missing_coverage(
    monkeypatch,
    tmp_path,
):
    observed: dict = {}
    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")
    monkeypatch.setattr(director_planner, "create_ark_client", lambda **_kwargs: object())

    def fake_stream(*, messages, **kwargs):
        observed["messages"] = messages
        observed["response_format"] = kwargs["response_format"]
        return json.dumps(_director_plan(), ensure_ascii=False)

    monkeypatch.setattr(director_planner, "call_llm_stream", fake_stream)

    result = director_planner.plan_director(_events(), tmp_path)

    prompt = observed["messages"][1]["content"]
    assert "SEQ001" in prompt and "SEQ002" in prompt
    assert "不要重新分场" in prompt
    assert observed["response_format"]["type"] == "json_schema"
    schema = observed["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert result["plan"] == _director_plan()

    incomplete = _director_plan()
    incomplete["sequences"] = incomplete["sequences"][:1]
    with pytest.raises(ValueError, match="SEQ002"):
        director_planner.validate_director_plan(incomplete, _events())

    with_extra_shot_field = _director_plan()
    with_extra_shot_field["sequences"][0]["camera_angle"] = "low"
    with pytest.raises(ValueError, match="extra"):
        director_planner.validate_director_plan(
            with_extra_shot_field,
            _events(),
        )


def test_director_planner_retries_one_invalid_complete_response(
    monkeypatch,
    tmp_path,
):
    responses = [
        json.dumps(_director_plan(), ensure_ascii=False) + " trailing prose",
        json.dumps(_director_plan(), ensure_ascii=False),
    ]
    prompts: list[str] = []
    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")
    monkeypatch.setattr(
        director_planner,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def fake_stream(*, messages, **_kwargs):
        prompts.append(messages[1]["content"])
        return responses.pop(0)

    monkeypatch.setattr(director_planner, "call_llm_stream", fake_stream)

    result = director_planner.plan_director(_events(), tmp_path)

    assert result["plan"] == _director_plan()
    assert len(prompts) == 2
    assert "上次输出未通过导演计划业务校验" in prompts[1]


def test_director_intent_changes_layered_checkpoint_identity():
    common = {
        "events": _events(),
        "characters_summary": "林夏",
        "target_duration": 30,
        "shot_duration": 15,
        "expected_beats": 2,
    }

    first = adaptation_engine._layered_input_fingerprint(
        **common,
        director_plan=_director_plan(),
    )
    changed = adaptation_engine._layered_input_fingerprint(
        **common,
        director_plan=_director_plan(second_goal="揭示屏幕正在复制主角意识"),
    )

    assert first != changed


def test_duration_scaling_uses_strict_director_aligned_source_index_selection(
    monkeypatch,
):
    events = [{
        "sequence_id": "SEQ001",
        "event_role": "turning_point",
        "dramatic_turn": True,
        "what": "角色识破威胁并完成反制",
        "start_state": "角色处于受压状态",
        "end_state": "角色重新掌握主动",
        "micro_actions": [
            f"依次完成来源动作{index}" for index in range(1, 8)
        ],
    }]
    director_plan = {
        "schema": "honcut.director-plan.v1",
        "sequences": [{
            "sequence_id": "SEQ001",
            "scene_goal": "让角色从受压转为主动",
            "emotion_arc": "警觉 → 决断",
            "visual_focus": "识破威胁的反应与最终控制动作",
            "spatial_intent": "角色由后撤位置重新贴近对手",
            "transition_intent": "以控制完成后的稳定状态收束",
        }],
    }
    production_events, scaling_plan = (
        adaptation_engine._build_duration_scaled_event_plan(
            events,
            target_duration=6,
            beat_count=1,
            effective_shot_duration=6,
        )
    )
    assert scaling_plan["intra_event_scaling_applied"] is True
    observed: dict = {}
    monkeypatch.setattr(
        adaptation_engine,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def fake_stream(*, messages, response_format, **_kwargs):
        observed["prompt"] = messages[1]["content"]
        observed["response_format"] = response_format
        return json.dumps({
            "schema": "honcut.duration-scaled-action-selection.v1",
            "events": [{
                "source_event_id": 1,
                "selected_source_generation_unit_indexes": [1, 3, 5, 7],
                "narrative_purpose": "保留由受压到主动的因果端点",
                "emotional_beat": "警觉转为决断",
                "director_alignment": "对齐反应、控制动作与稳定收束",
            }],
        }, ensure_ascii=False)

    monkeypatch.setattr(adaptation_engine, "call_llm_stream", fake_stream)

    selected_events, selected_plan = (
        adaptation_engine._apply_director_action_selection(
            events,
            production_events,
            scaling_plan,
            director_plan,
        )
    )

    assert observed["response_format"]["type"] == "json_schema"
    assert observed["response_format"]["json_schema"]["strict"] is True
    assert "让角色从受压转为主动" in observed["prompt"]
    assert selected_events[0]["micro_actions"] == [
        events[0]["micro_actions"][index] for index in (0, 2, 4, 6)
    ]
    assert selected_plan["semantic_selection_status"] == "director_aligned"
    assert selected_plan["events"][0]["narrative_purpose"] == (
        "保留由受压到主动的因果端点"
    )
