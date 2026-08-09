import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import adaptation_engine as engine


def _events(count):
    return [{"summary": f"事件{i}", "dialogue": f"台词{i}"} for i in range(1, count + 1)]


def _beat(i):
    return {
        "beat_order": i,
        "source_events": [i],
        "action": "keep",
        "reason": "保留因果",
        "who": ["凛"],
        "where": "庭院",
        "what": f"发生事件{i}",
        "suggested_duration": 12,
        "_source_event_details": [{"event_id": i, "summary": f"事件{i}"}],
    }


def _shot(order, visual=None):
    return {
        "shot_order": order,
        "source_events": [order],
        "action": "keep",
        "reason": "保留",
        "who": ["凛"],
        "where": "庭院",
        "what": f"动作{order}",
        "emotion": "平静",
        "visual": visual or f"凛 — 黑发, 目光坚定 — 动作{order}",
        "suggested_duration": 12,
        "transition_to_next": "cut",
        "associate_assets": ["char:lin", "scene:庭院"],
        "shot_size": "medium",
        "camera_movement": "static",
        "lighting_key": "natural",
        "shot_intent": "action",
        "dialogue": None,
        "gen_strategy": "phantom",
    }


def _batch_response(first, count=3, final_visual=None):
    shots = [_shot(first + i) for i in range(count)]
    if final_visual:
        shots[-1]["visual"] = final_visual
    return json.dumps({"strategy": "批次", "shots": shots}, ensure_ascii=False)


def test_beat_skeleton_parsing_and_coverage(monkeypatch):
    events = _events(3)
    payload = {
        "strategy": "压缩主线",
        "beats": [
            dict(_beat(1), source_events=[1, 2], action="merge"),
            dict(_beat(2), source_events=[3]),
        ],
    }
    for beat in payload["beats"]:
        beat.pop("_source_event_details", None)
    monkeypatch.setattr(engine, "estimate_shot_count", lambda *_: 2)
    monkeypatch.setattr(engine, "_call_llm", lambda prompt, max_tokens=0: json.dumps(payload, ensure_ascii=False))

    result = engine._build_beat_skeleton(events, "- 凛", 24, 12)

    assert result["strategy"] == "压缩主线"
    assert {event_id for beat in result["beats"] for event_id in beat["source_events"]} == {1, 2, 3}
    assert result["beats"][0]["_source_event_details"][1]["event_id"] == 2


def test_expand_three_batches_has_global_contiguous_order(monkeypatch):
    responses = iter([_batch_response(91), _batch_response(41), _batch_response(7)])
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", lambda *args, **kwargs: next(responses))

    shots = engine._expand_beats_to_shots([_beat(i) for i in range(1, 10)], "- 凛", 108)

    assert [shot["shot_order"] for shot in shots] == list(range(1, 10))


def test_second_batch_prompt_contains_last_shot_relay(monkeypatch):
    prompts = []
    tail = "前缀会被裁掉" + "接力片段" * 30

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return _batch_response(1 if len(prompts) == 1 else 4, final_visual=tail if len(prompts) == 1 else None)

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", fake_call)
    engine._expand_beats_to_shots([_beat(i) for i in range(1, 7)], "- 凛", 72)

    assert "上一批最后一镜接力上下文" in prompts[1]
    assert "who=[\"凛\"]" in prompts[1]
    assert "where=庭院" in prompts[1]
    assert tail[-100:] in prompts[1]
    assert tail[:-100] not in prompts[1]


def test_skeleton_rejects_uncovered_source_event():
    payload = {"strategy": "遗漏", "beats": [{k: v for k, v in _beat(1).items() if not k.startswith("_")}]}
    with pytest.raises(ValueError, match="未覆盖"):
        engine._parse_beat_skeleton(json.dumps(payload, ensure_ascii=False), 1, 2)


def test_single_mode_uses_legacy_path(monkeypatch):
    calls = {"single": 0, "skeleton": 0}

    def single(prompt, **kwargs):
        calls["single"] += 1
        return _batch_response(1, count=1)

    monkeypatch.setenv("HONCUT_ADAPT_MODE", "single")
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", single)
    monkeypatch.setattr(engine, "_build_beat_skeleton", lambda *args: calls.__setitem__("skeleton", 1))

    result = engine.adapt_events(_events(11), target_duration=12)

    assert calls == {"single": 1, "skeleton": 0}
    assert result["strategy"] == "批次"


def test_batch_parse_retry_only_retries_failing_batch(monkeypatch):
    prompts = []
    calls = [
        _batch_response(1),
        "{broken",
        _batch_response(4),
        _batch_response(7),
    ]

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return calls[len(prompts) - 1]

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", fake_call)
    monkeypatch.setattr(engine.time, "sleep", lambda *_: None)

    shots = engine._expand_beats_to_shots([_beat(i) for i in range(1, 10)], "- 凛", 108)

    assert len(shots) == 9
    assert len(prompts) == 4
    assert prompts[1] == prompts[2]
    assert prompts[0] != prompts[1]


def test_small_script_automatically_uses_single(monkeypatch):
    monkeypatch.delenv("HONCUT_ADAPT_MODE", raising=False)
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", lambda *args, **kwargs: _batch_response(1, count=1))
    monkeypatch.setattr(engine, "_build_beat_skeleton", lambda *args: pytest.fail("small script used layered mode"))

    assert engine.adapt_events(_events(10), target_duration=12)["estimated_shots"] == 1
