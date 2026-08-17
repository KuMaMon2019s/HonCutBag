import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import phases.phase1.adaptation_engine as engine


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
        "beat_order": order,
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


def _batch_response(first, count=3, final_visual=None, beat_first=None):
    shots = [_shot(first + i) for i in range(count)]
    if beat_first is not None:
        for i, shot in enumerate(shots):
            shot["beat_order"] = beat_first + i
            shot["source_events"] = [beat_first + i]
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
    call = lambda prompt, max_tokens=0: json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", call)

    result = engine._build_beat_skeleton(events, "- 凛", 24, 12)

    assert result["strategy"] == "压缩主线"
    assert {event_id for beat in result["beats"] for event_id in beat["source_events"]} == {1, 2, 3}
    assert result["beats"][0]["_source_event_details"][1]["event_id"] == 2


def test_expand_three_batches_has_global_contiguous_order(monkeypatch):
    responses = iter([
        _batch_response(91, beat_first=1),
        _batch_response(41, beat_first=4),
        _batch_response(7, beat_first=7),
    ])
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", lambda *args, **kwargs: next(responses))

    shots = engine._expand_beats_to_shots([_beat(i) for i in range(1, 10)], "- 凛", 108, 12)

    assert [shot["shot_order"] for shot in shots] == list(range(1, 10))
    assert all("beat_order" not in shot for shot in shots)


def test_second_batch_prompt_contains_last_shot_relay(monkeypatch):
    prompts = []
    tail = "前缀会被裁掉" + "接力片段" * 30

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return _batch_response(1 if len(prompts) == 1 else 4, final_visual=tail if len(prompts) == 1 else None)

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", fake_call)
    engine._expand_beats_to_shots([_beat(i) for i in range(1, 7)], "- 凛", 72, 12)

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

    result = engine.adapt_events(_events(11), target_duration=15)

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

    shots = engine._expand_beats_to_shots([_beat(i) for i in range(1, 10)], "- 凛", 108, 12)

    assert len(shots) == 9
    assert len(prompts) == 4
    assert prompts[1] == prompts[2]
    assert prompts[0] != prompts[1]


def test_small_script_automatically_uses_single(monkeypatch):
    monkeypatch.delenv("HONCUT_ADAPT_MODE", raising=False)
    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", lambda *args, **kwargs: _batch_response(1, count=1))
    monkeypatch.setattr(engine, "_build_beat_skeleton", lambda *args: pytest.fail("small script used layered mode"))

    assert engine.adapt_events(_events(10), target_duration=15)["estimated_shots"] == 1


def test_beat_order_mismatch_retries_batch(monkeypatch):
    wrong = json.loads(_batch_response(1))
    wrong["shots"][0]["beat_order"] = 2
    responses = iter([json.dumps(wrong, ensure_ascii=False), _batch_response(1)])
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", fake_call)
    monkeypatch.setattr(engine.time, "sleep", lambda *_: None)

    shots = engine._expand_beats_to_shots([_beat(i) for i in range(1, 4)], "- 凛", 36, 12)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert all("beat_order" not in shot for shot in shots)


def test_batch_prompt_uses_local_duration_budget_and_formats_safely():
    beats = [dict(_beat(1), suggested_duration=7), dict(_beat(2), suggested_duration=9)]

    prompt = engine._batch_prompt(beats, "- 凛", 999, 8, 0, None)

    assert "每镜约8秒" in prompt
    assert "本批目标时长：16秒" in prompt
    assert "接近 16 秒" in prompt
    assert "接近 target_duration" not in prompt
    assert "接近 999 秒" not in prompt
    assert '"beat_order":1' in prompt


@pytest.mark.parametrize("visual", ["", None, 123])
def test_batch_prompt_empty_relay_visual_uses_fallback(visual):
    prompt = engine._batch_prompt(
        [_beat(1)], "- 凛", 12, 12, 0,
        {"who": ["凛"], "where": "庭院", "visual": visual},
    )

    assert "上一镜无可用 visual，仅按 who/where 承接" in prompt
    assert "visual末尾=" not in prompt


def test_stage1_uses_timeout_retry_wrapper(monkeypatch):
    payload = {"strategy": "骨架", "beats": [{k: v for k, v in _beat(1).items() if not k.startswith("_")}]}
    calls = []
    monkeypatch.setattr(engine, "estimate_shot_count", lambda *_: 1)

    def fake_call(prompt, max_tokens):
        calls.append(max_tokens)
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", fake_call)
    monkeypatch.setattr(engine, "_call_llm", lambda *args, **kwargs: pytest.fail("Stage 1 bypassed retry wrapper"))

    engine._build_beat_skeleton(_events(1), "- 凛", 12, 12)

    assert calls == [8000]


def test_action_dense_script_fails_when_full_detail_cannot_fit_target_runtime():
    action_events = [
            {
                "event_role": "action_chain",
                "action_unit_id": f"AU{i:03d}",
                "micro_actions": [f"动作{i}-{j}" for j in range(8)],
            }
            for i in range(1, 13)
    ]
    events = [
        {"event_role": "scene_setup"},
        *action_events[:4],
        {"event_role": "dialogue"},
        *action_events[4:8],
        {"event_role": "dialogue"},
        *action_events[8:],
        {"event_role": "scene_setup"},
    ]

    with pytest.raises(ValueError, match="cannot carry the authored action detail"):
        engine.estimate_action_aware_shot_count(events, 60, 10)


def test_event_semantics_bounds_generation_actions_but_keeps_full_ledger():
    events = [{
        "action_unit_id": "AU001",
        "sequence_id": "SEQ001",
        "event_role": "action_chain",
        "micro_actions": [f"动作{i}" for i in range(8)],
        "start_state": "二人相隔数米对峙",
        "end_state": "护栏断裂",
    }]
    shots = [{"source_events": [1]}]

    engine._inherit_event_semantics(shots, events)

    assert shots[0]["micro_actions"] == [f"动作{i}" for i in range(8)]
    assert shots[0]["generation_actions"] == ["动作0", "动作2", "动作5", "动作7"]
    assert shots[0]["action_description"] == "动作0 → 动作2 → 动作5 → 动作7"
    assert shots[0]["generation_load"]["compression"] == "representative"
    assert shots[0]["start_state"] == "二人相隔数米对峙"
    assert shots[0]["end_state"] == "护栏断裂"


def test_short_action_clip_uses_single_visible_action_budget():
    actions = [f"动作{i}" for i in range(8)]

    selected = engine.select_generation_actions(actions, duration_seconds=4)

    assert selected == ["动作0"]
    assert engine.generation_action_limit(4) == 1
    assert engine.generation_action_limit(6) == 2


def test_duration_normalization_closes_exact_target_with_integer_seconds():
    shots = [{"suggested_duration": 15} for _ in range(4)]

    engine.normalize_shot_durations(shots, 60)

    assert sum(shot["suggested_duration"] for shot in shots) == 60
    assert [shot["suggested_duration"] for shot in shots] == [15] * 4


def test_duration_normalization_gives_dense_action_more_provider_capacity():
    shots = [
        {"micro_actions": ["翻开旧书"]},
        {"micro_actions": ["抬头", "发现", "触摸", "迟疑"]},
    ]

    engine.normalize_shot_durations(shots, 31)

    assert sum(shot["suggested_duration"] for shot in shots) == 31
    assert shots[1]["suggested_duration"] > shots[0]["suggested_duration"]
    assert shots[1]["suggested_duration"] >= 12
    assert shots[1]["duration_allocation"]["complexity_weight"] == 2


def test_beat_capacity_allows_continuous_action_units_but_rejects_cross_sequence_merge():
    events = [
        {"action_unit_id": "AU001", "sequence_id": "SEQ001"},
        {"action_unit_id": "AU002", "sequence_id": "SEQ001"},
    ]
    beats = [{"beat_order": 1, "action": "merge", "source_events": [1, 2]}]

    engine._validate_beat_action_capacity(beats, events)

    events[1]["sequence_id"] = "SEQ002"
    with pytest.raises(ValueError, match="unrelated sequences"):
        engine._validate_beat_action_capacity(beats, events)


def test_layered_mode_persists_skeleton_and_each_batch(monkeypatch, tmp_path):
    events = _events(11)
    beats = [_beat(i) for i in range(1, 4)]
    skeleton = {"strategy": "preserve causal spine", "beats": beats}
    writes = []

    monkeypatch.setattr(engine, "estimate_shot_count", lambda *_: 3)
    monkeypatch.setattr(engine, "_build_beat_skeleton", lambda *args: skeleton)
    monkeypatch.setattr(
        engine,
        "_call_llm_with_timeout_retry",
        lambda *args, **kwargs: _batch_response(1),
    )
    original_write = engine._atomic_write_json

    def recording_write(path, value):
        original_write(path, value)
        writes.append((path.name, json.loads(path.read_text(encoding="utf-8"))))

    monkeypatch.setattr(engine, "_atomic_write_json", recording_write)
    result = engine.adapt_events(events, target_duration=45, output_dir=tmp_path)

    persisted_skeleton = json.loads((tmp_path / "beat_skeleton.json").read_text(encoding="utf-8"))
    partial = json.loads((tmp_path / "shots_partial.json").read_text(encoding="utf-8"))
    assert persisted_skeleton["strategy"] == "preserve causal spine"
    assert len(persisted_skeleton["beats"]) == 3
    assert partial["completed_batches"] == [1]
    assert [shot["shot_order"] for shot in partial["shots"]] == [1, 2, 3]
    assert [shot["shot_order"] for shot in result["shots"]] == [1, 2, 3]
    assert [name for name, _ in writes] == ["beat_skeleton.json", "shots_partial.json"]
    assert not list(tmp_path.glob("*.tmp"))


def test_layered_resume_skips_cached_skeleton_and_batches(monkeypatch, tmp_path):
    events = _events(11)
    fingerprint = engine._layered_input_fingerprint(
        events, "（无角色信息）", 90, 15, 6
    )
    public_beats = []
    for i in range(1, 7):
        beat = _beat(i)
        beat.pop("_source_event_details")
        public_beats.append(beat)
    public_beats[-1]["source_events"] = list(range(6, 12))
    (tmp_path / "beat_skeleton.json").write_text(
        json.dumps({
            "_checkpoint": {
                "schema": engine.LAYERED_CHECKPOINT_SCHEMA,
                "input_fingerprint": fingerprint,
            },
            "strategy": "cached strategy",
            "beats": public_beats,
        }),
        encoding="utf-8",
    )
    cached_shots = [_shot(i) for i in range(1, 4)]
    for shot in cached_shots:
        shot.pop("beat_order")
    (tmp_path / "shots_partial.json").write_text(
        json.dumps({
            "_checkpoint": {
                "schema": engine.LAYERED_CHECKPOINT_SCHEMA,
                "input_fingerprint": fingerprint,
            },
            "completed_batches": [1],
            "shots": cached_shots,
        }),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(engine, "estimate_shot_count", lambda *_: 6)
    monkeypatch.setattr(
        engine, "_build_beat_skeleton", lambda *args: pytest.fail("cached skeleton was rebuilt")
    )

    def expand_only_missing(prompt, **kwargs):
        calls.append(prompt)
        response = json.loads(_batch_response(4, beat_first=4))
        response["shots"][-1]["source_events"] = list(range(6, 12))
        return json.dumps(response)

    monkeypatch.setattr(engine, "_call_llm_with_timeout_retry", expand_only_missing)
    result = engine.adapt_events(events, target_duration=90, output_dir=tmp_path)

    assert len(calls) == 1
    assert "第一个 shot_order 必须为 4" in calls[0]
    assert result["strategy"] == "cached strategy"
    assert [shot["shot_order"] for shot in result["shots"]] == list(range(1, 7))
    assert json.loads((tmp_path / "shots_partial.json").read_text())["completed_batches"] == [1, 2]
