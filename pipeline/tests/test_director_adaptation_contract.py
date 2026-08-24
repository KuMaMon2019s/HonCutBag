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

    def fake_stream(*, messages, **_kwargs):
        observed["messages"] = messages
        return json.dumps(_director_plan(), ensure_ascii=False)

    monkeypatch.setattr(director_planner, "call_llm_stream", fake_stream)

    result = director_planner.plan_director(_events(), tmp_path)

    prompt = observed["messages"][1]["content"]
    assert "SEQ001" in prompt and "SEQ002" in prompt
    assert "不要重新分场" in prompt
    assert result["plan"] == _director_plan()

    incomplete = _director_plan()
    incomplete["sequences"] = incomplete["sequences"][:1]
    with pytest.raises(ValueError, match="SEQ002"):
        director_planner.validate_director_plan(incomplete, _events())


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

