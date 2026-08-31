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
    reconciliation = json.loads(
        (tmp_path / "director_plan_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    assert reconciliation["schema"] == (
        director_planner.DIRECTOR_PLAN_RECONCILIATION_SCHEMA
    )
    assert reconciliation["duplicate_count"] == 0
    assert reconciliation["source_sequence_loss_count"] == 0

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


def test_director_planner_reconciles_adjacent_duplicate_sequence_observations(
    monkeypatch,
    tmp_path,
):
    duplicate = _director_plan()
    alternate = {
        **duplicate["sequences"][0],
        "scene_goal": "另一个不具来源权威的导演提案",
        "emotion_arc": "紧张 → 更紧张",
    }
    duplicate["sequences"].insert(1, alternate)
    calls = 0
    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")
    monkeypatch.setattr(
        director_planner,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def fake_stream(**_kwargs):
        nonlocal calls
        calls += 1
        return json.dumps(duplicate, ensure_ascii=False)

    monkeypatch.setattr(director_planner, "call_llm_stream", fake_stream)

    result = director_planner.plan_director(_events(), tmp_path)

    assert calls == 1
    assert result["plan"] == _director_plan()
    receipt = json.loads(
        (tmp_path / "director_plan_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["expected_sequence_ids"] == ["SEQ001", "SEQ002"]
    assert receipt["original_sequence_ids"] == [
        "SEQ001",
        "SEQ001",
        "SEQ002",
    ]
    assert receipt["reconciled_sequence_ids"] == ["SEQ001", "SEQ002"]
    assert receipt["duplicate_count"] == 1
    assert receipt["duplicates"][0]["retained_position"] == 1
    assert receipt["duplicates"][0]["dropped_position"] == 2
    assert receipt["source_sequence_loss_count"] == 0
    receipt_text = json.dumps(receipt, ensure_ascii=False)
    assert "另一个不具来源权威的导演提案" not in receipt_text
    assert "建立人物孤独感" not in receipt_text


@pytest.mark.parametrize(
    ("sequence_ids", "message"),
    [
        (["SEQ001", "SEQ002", "SEQ001"], "non-adjacent duplicate"),
        (["SEQ001", "SEQ003", "SEQ002"], "unexpected sequence_id"),
        (["SEQ002", "SEQ001"], "coverage/order mismatch"),
        (["SEQ001"], "coverage/order mismatch"),
    ],
)
def test_director_planner_reconciliation_rejects_authority_changes(
    sequence_ids,
    message,
):
    templates = {
        item["sequence_id"]: item for item in _director_plan()["sequences"]
    }
    unknown = {
        **templates["SEQ002"],
        "sequence_id": "SEQ003",
    }
    plan = {
        "schema": director_planner.DIRECTOR_PLAN_SCHEMA,
        "sequences": [
            dict(templates.get(sequence_id, unknown))
            for sequence_id in sequence_ids
        ],
    }

    with pytest.raises(ValueError, match=message):
        director_planner.validate_director_plan(plan, _events())


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


def test_duration_scaling_uses_strict_source_indexed_screenplay_rewrite(
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
        groups = production_events[0]["production_action_rewrite"]["groups"]
        return json.dumps({
            "schema": "honcut.source-indexed-screenplay-rewrite.v1",
            "events": [{
                "source_event_id": 1,
                "production_actions": [
                    {
                        "production_action_index": group[
                            "production_action_index"
                        ],
                        "source_micro_action_indexes": group[
                            "source_micro_action_indexes"
                        ],
                        "rewritten_micro_action": (
                            "连续保全：" + "；".join(group["source_actions"])
                        ),
                    }
                    for group in groups
                ],
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
    assert all(
        action.startswith("连续保全：")
        for action in selected_events[0]["micro_actions"]
    )
    rewrite = selected_events[0]["production_action_rewrite"]
    assert {
        index
        for group in rewrite["groups"]
        for index in group["source_micro_action_indexes"]
    } == set(range(1, 8))
    assert rewrite["omitted_source_micro_action_indexes"] == []
    assert selected_plan["semantic_selection_status"] == (
        "source_indexed_rewrite"
    )
    assert selected_plan["events"][0]["narrative_purpose"] == (
        "保留由受压到主动的因果端点"
    )


def test_duration_scaling_excludes_zero_action_static_event_from_rewrite(
    monkeypatch,
):
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "character_state",
            "what": "角色停在入口前观察环境",
            "micro_actions": [],
        },
        {
            "sequence_id": "SEQ001",
            "event_role": "turning_point",
            "dramatic_turn": True,
            "what": "角色识破威胁并完成反制",
            "micro_actions": [
                f"依次完成来源动作{index}" for index in range(1, 8)
            ],
        },
    ]
    director_plan = {
        "schema": "honcut.director-plan.v1",
        "sequences": [{
            "sequence_id": "SEQ001",
            "scene_goal": "保留观察到反制的完整因果",
            "emotion_arc": "警觉 → 决断",
            "visual_focus": "观察后的连续反制",
            "spatial_intent": "保持连续空间",
            "transition_intent": "以反制结果收束",
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

    assert [
        record["scaling"] for record in scaling_plan["events"]
    ] == ["full", "rewrite"]
    assert production_events[0]["production_action_rewrite"]["groups"] == []

    calls = 0
    monkeypatch.setattr(
        adaptation_engine,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def fake_stream(*, messages, **_kwargs):
        nonlocal calls
        calls += 1
        assert "角色停在入口前观察环境" not in messages[1]["content"]
        groups = production_events[1]["production_action_rewrite"]["groups"]
        return json.dumps({
            "schema": "honcut.source-indexed-screenplay-rewrite.v1",
            "events": [{
                "source_event_id": 2,
                "production_actions": [
                    {
                        "production_action_index": group[
                            "production_action_index"
                        ],
                        "source_micro_action_indexes": group[
                            "source_micro_action_indexes"
                        ],
                        "rewritten_micro_action": (
                            "连续保全：" + "；".join(group["source_actions"])
                        ),
                    }
                    for group in groups
                ],
                "narrative_purpose": "保留观察后的反制因果",
                "emotional_beat": "警觉转为决断",
                "director_alignment": "对齐连续反制与收束",
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

    assert calls == 1
    assert selected_events[0]["micro_actions"] == []
    assert selected_plan["semantic_selection_status"] == (
        "source_indexed_rewrite"
    )


def test_source_indexed_rewrite_collapses_adjacent_lineage_equivalent_duplicate(
    monkeypatch,
):
    events = [{
        "sequence_id": "SEQ001",
        "event_role": "turning_point",
        "dramatic_turn": True,
        "what": "来源事件1",
        "start_state": "受压",
        "end_state": "稳定",
        "micro_actions": [f"事件1动作{index}" for index in range(1, 8)],
    }]
    director_plan = {
        "schema": "honcut.director-plan.v1",
        "sequences": [{
            "sequence_id": "SEQ001",
            "scene_goal": "保留来源因果",
            "emotion_arc": "受压 → 稳定",
            "visual_focus": "动作落点",
            "spatial_intent": "连续空间",
            "transition_intent": "动作承接",
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
    monkeypatch.setattr(
        adaptation_engine,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def observation(event_id, *, suffix):
        groups = production_events[event_id - 1][
            "production_action_rewrite"
        ]["groups"]
        return {
            "source_event_id": event_id,
            "production_actions": [
                {
                    "production_action_index": group[
                        "production_action_index"
                    ],
                    "source_micro_action_indexes": group[
                        "source_micro_action_indexes"
                    ],
                    "rewritten_micro_action": (
                        f"{suffix}：" + "；".join(group["source_actions"])
                    ),
                }
                for group in groups
            ],
            "narrative_purpose": f"目的{suffix}",
            "emotional_beat": f"情绪{suffix}",
            "director_alignment": f"导演对齐{suffix}",
        }

    monkeypatch.setattr(
        adaptation_engine,
        "call_llm_stream",
        lambda **_kwargs: json.dumps(
            {
                "schema": "honcut.source-indexed-screenplay-rewrite.v1",
                "events": [
                    observation(1, suffix="保留版本"),
                    observation(1, suffix="重复版本"),
                ],
            },
            ensure_ascii=False,
        ),
    )

    selected_events, selected_plan = (
        adaptation_engine._apply_director_action_selection(
            events,
            production_events,
            scaling_plan,
            director_plan,
        )
    )

    assert selected_events[0]["micro_actions"][0].startswith("保留版本：")
    reconciliation = selected_plan[
        "source_indexed_rewrite_reconciliation"
    ]
    assert reconciliation["schema"] == (
        "honcut.source-indexed-screenplay-rewrite-reconciliation.v1"
    )
    assert reconciliation["original_source_event_ids"] == [1, 1]
    assert reconciliation["reconciled_source_event_ids"] == [1]
    assert reconciliation["duplicate_count"] == 1
    assert reconciliation["source_fact_loss_count"] == 0
    assert reconciliation["provider_request_count"] == 0
    assert reconciliation["duplicates"][0]["retained_position"] == 1
    assert reconciliation["duplicates"][0]["dropped_position"] == 2
    assert "rewritten_micro_action" not in json.dumps(
        reconciliation,
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "returned_order",
    [
        [1, 2, 1],
        [1, 1],
        [1, 3, 2],
    ],
)
def test_source_indexed_rewrite_rejects_non_equivalent_event_coverage(
    returned_order,
):
    expected_groups = {
        event_id: [{
            "production_action_index": 1,
            "source_micro_action_indexes": [1, 2],
        }]
        for event_id in (1, 2)
    }

    def observation(event_id):
        return {
            "source_event_id": event_id,
            "production_actions": [{
                "production_action_index": 1,
                "source_micro_action_indexes": [1, 2],
                "rewritten_micro_action": "保持全部来源动作",
            }],
            "narrative_purpose": "保持目的",
            "emotional_beat": "保持情绪",
            "director_alignment": "保持导演意图",
        }

    with pytest.raises(ValueError, match="coverage/order mismatch"):
        adaptation_engine._reconcile_source_indexed_rewrite_observations(
            [observation(event_id) for event_id in returned_order],
            [1, 2],
            expected_groups,
        )


def test_source_indexed_rewrite_rejects_adjacent_duplicate_lineage_change():
    expected_groups = {
        1: [{
            "production_action_index": 1,
            "source_micro_action_indexes": [1, 2],
        }]
    }

    def observation(source_indexes):
        return {
            "source_event_id": 1,
            "production_actions": [{
                "production_action_index": 1,
                "source_micro_action_indexes": source_indexes,
                "rewritten_micro_action": "保持全部来源动作",
            }],
            "narrative_purpose": "保持目的",
            "emotional_beat": "保持情绪",
            "director_alignment": "保持导演意图",
        }

    with pytest.raises(ValueError, match="changed lineage"):
        adaptation_engine._reconcile_source_indexed_rewrite_observations(
            [observation([1, 2]), observation([1])],
            [1],
            expected_groups,
        )
