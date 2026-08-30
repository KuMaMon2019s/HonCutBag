import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import phases.phase1.director_planner as director_planner
import phases.phase1.storyboard_generator as storyboard_generator
from phases import pipeline_core
from phases.phase1.phase1_pipeline import run_phase1
from prompt import event_extractor, text_parser
from runtime.provider_attempt_policy import provider_attempt_scope
from utils import ark_llm
from utils.config import DEFAULT_TEXT_MODEL
from utils.progress_reporter import ProgressReporter


def _deterministic_character_fixture(events):
    from phases.phase1.character_roster import (
        compile_character_roster,
        reconcile_character_observations,
    )

    roster = compile_character_roster(events)
    characters, diagnostics = reconcile_character_observations(
        [], roster, semantic_qa_enabled=False
    )
    return {
        "characters": characters,
        "total_characters": len(characters),
        "total_character_instances": sum(
            character["instance_count"] for character in characters
        ),
        "character_roster": roster,
        "character_roster_sha256": roster["roster_sha256"],
        "semantic_qa_enabled": False,
        "semantic_diagnostics": diagnostics,
    }


def _chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


def _canonical_adapted_result(
    adaptation_engine,
    *,
    actions,
    target_duration,
    shot_duration,
):
    relations = [
        {
            "micro_action_index": index,
            "performers": [],
            "targets": [],
            "action_kind": "state_change",
            "temporal_relation": "root" if index == 1 else "after",
            "reference_action_indexes": [] if index == 1 else [index - 1],
            "pace": "normal",
            "state_reads": [],
            "state_writes": [action],
        }
        for index, action in enumerate(actions, 1)
    ]
    event = {
        "event_role": "action_chain",
        "sequence_id": "SEQ001",
        "micro_actions": list(actions),
        "action_temporal_relations": relations,
    }
    events = [event]
    source_capacity_plan = adaptation_engine._estimate_action_capacity_plan(
        events,
        target_duration,
        shot_duration,
        shot_policy="continuity",
    )
    layout = source_capacity_plan["primary_shot_layout"]
    effective_shot_duration = round(
        target_duration / layout["primary_shots"]
    )
    production_events, duration_plan = (
        adaptation_engine._build_duration_scaled_event_plan(
            events,
            target_duration=target_duration,
            beat_count=layout["primary_shots"],
            effective_shot_duration=effective_shot_duration,
            capabilities=adaptation_engine.get_video_capabilities(),
            max_generation_units_per_beat=layout[
                "max_generation_action_units_per_primary_shot"
            ],
            maximum_total_generation_units=layout[
                "production_action_unit_target"
            ],
            generation_unit_capacities_per_beat=list(
                layout["generation_action_unit_capacities"]
            ),
        )
    )
    source_timeline = adaptation_engine.build_action_timeline(
        events,
        max_motion_contributions_per_slice=(
            adaptation_engine.get_video_capabilities().motion_contribution_limit
        ),
    )
    production_timeline = adaptation_engine.build_action_timeline(
        production_events,
        max_motion_contributions_per_slice=(
            adaptation_engine.get_video_capabilities().motion_contribution_limit
        ),
    )
    timeline_binding = adaptation_engine._bind_action_timeline_to_primary_layout(
        production_events,
        duration_plan,
        layout,
        adaptation_engine.get_video_capabilities(),
    )
    contracts = adaptation_engine._canonical_beat_contracts(
        production_events,
        duration_plan,
        timeline_binding,
    )
    production_actions = production_events[0]["micro_actions"]
    action_cursor = 0
    shots = []
    for contract in contracts:
        unit_count = contract["execution_subslice_count"]
        shot_actions = production_actions[action_cursor:action_cursor + unit_count]
        action_cursor += unit_count
        shots.append({
            **contract,
            "id": contract["sxx_id"],
            "duration": contract["suggested_duration"],
            "source_sequence_ids": [contract["sequence_id"]],
            "source_event_casts": [{
                "source_event_id": 1,
                "character_ids": [],
            }],
            "micro_actions": shot_actions,
            "generation_actions": shot_actions,
            "generation_motion_mode": "atomic",
            "generation_action_units": [
                {
                    "unit_id": (
                        f"{contract['sxx_id']}_GAU{unit_index:03d}"
                    ),
                    "source_event_id": 1,
                    "actions": [action],
                }
                for unit_index, action in enumerate(shot_actions, 1)
            ],
            "action": "keep",
            "what": " → ".join(shot_actions),
        })
    screenplay_plan, capacity_plan = adaptation_engine._build_screenplay_plan(
        events,
        shots,
        source_capacity_plan,
        target_duration=target_duration,
        production_events=production_events,
        duration_scaled_event_plan=duration_plan,
        primary_shot_layout=layout,
        source_action_timeline=source_timeline,
        production_action_timeline=production_timeline,
        timeline_layout_binding=timeline_binding,
    )
    screenplay_plan_sha256 = hashlib.sha256(
        json.dumps(
            screenplay_plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "shots": shots,
        "material_duration": target_duration,
        "capacity_plan": capacity_plan,
        "primary_shot_layout": layout,
        "screenplay_plan": screenplay_plan,
        "screenplay_plan_sha256": screenplay_plan_sha256,
    }


def _phase1_stub_adapted_result(adaptation_engine):
    return _canonical_adapted_result(
        adaptation_engine,
        actions=["站立"],
        target_duration=15,
        shot_duration=15,
    )


def _phase1_long_shot_adapted_result(adaptation_engine):
    return _canonical_adapted_result(
        adaptation_engine,
        actions=[f"有序动作{index}" for index in range(45)],
        target_duration=36,
        shot_duration=6,
    )


def test_wall_timeout_closes_blocked_stream():
    class BlockedStream:
        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            self.closed.wait(1)
            if False:
                yield None

        def close(self):
            self.closed.set()

    stream = BlockedStream()
    completions = SimpleNamespace(create=lambda **_kwargs: stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(ark_llm.LLMWallTimeout):
        ark_llm.call_llm_stream(
            [{"role": "user", "content": "synthetic"}],
            wall_timeout=0.03,
            idle_timeout=1,
            _client=client,
        )
    assert stream.closed.is_set()


def test_idle_timeout_closes_stalled_stream_before_wall_timeout():
    class BlockedStream:
        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            self.closed.wait(1)
            if False:
                yield None

        def close(self):
            self.closed.set()

    stream = BlockedStream()
    completions = SimpleNamespace(create=lambda **_kwargs: stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(ark_llm.LLMIdleTimeout):
        ark_llm.call_llm_stream(
            [{"role": "user", "content": "synthetic"}],
            wall_timeout=1,
            idle_timeout=0.03,
            _client=client,
        )
    assert stream.closed.is_set()


def test_active_stream_refreshes_idle_timeout_and_throttles_heartbeat():
    callbacks = []
    observed = {}

    class ActiveStream:
        def __iter__(self):
            for content in ("a", "b", "c"):
                # Keep the gap far below idle_timeout (100x margin) so the
                # test never becomes a scheduler-timing race.
                time.sleep(0.005)
                yield _chunk(content)

        def close(self):
            pass

    def create_stream(**kwargs):
        observed.update(kwargs)
        return ActiveStream()

    completions = SimpleNamespace(create=create_stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    content = ark_llm.call_llm_stream(
        [{"role": "user", "content": "synthetic"}],
        wall_timeout=1,
        idle_timeout=0.5,
        heartbeat_callback=lambda: callbacks.append(time.monotonic()),
        heartbeat_interval=1,
        _client=client,
    )

    assert content == "abc"
    assert len(callbacks) == 1
    assert observed["model"] == DEFAULT_TEXT_MODEL == "doubao-seed-evolving"


def test_incomplete_chunked_stream_is_classified_as_retryable():
    class BrokenStream:
        def __iter__(self):
            yield _chunk("partial")
            raise ark_llm.httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )

        def close(self):
            pass

    completions = SimpleNamespace(create=lambda **_kwargs: BrokenStream())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(ark_llm.LLMStreamError, match="peer closed"):
        ark_llm.call_llm_stream(
            [{"role": "user", "content": "synthetic"}],
            wall_timeout=1,
            idle_timeout=1,
            _client=client,
        )


def test_event_extractor_concurrent_results_remain_ordered(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()

    def extract(segment):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02 * (4 - segment["id"]))
        with lock:
            active -= 1
        return [{
            "who": [], "where": "", "what": str(segment["id"]),
            "emotion": "", "visual": "", "time": "", "action_type": "transition",
        }]

    monkeypatch.setattr(event_extractor, "_extract_events_from_segment", extract)
    result = event_extractor.extract_events([
        {"id": index, "content": "synthetic"} for index in range(1, 4)
    ])

    assert peak == 3
    assert [event["segment_id"] for event in result["events"]] == [1, 2, 3]
    assert result["covered_segment_ids"] == [1, 2, 3]


def test_event_extractor_retries_stream_interruption(monkeypatch):
    calls = 0

    def call(_prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ark_llm.LLMStreamError("incomplete chunked read")
        return json.dumps({"events": [{
            "who": ["凛"],
            "where": "高架",
            "what": "凛出现",
            "emotion": "紧张",
            "visual": "凛站在高架上",
            "time": "夜晚",
            "action_type": "reveal",
        }]}, ensure_ascii=False)

    monkeypatch.setattr(event_extractor, "_call_llm", call)
    monkeypatch.setattr(event_extractor.time, "sleep", lambda _seconds: None)

    events = event_extractor._extract_events_from_segment({"id": 1, "content": "凛出现"})

    assert calls == 2
    assert events[0]["who"] == ["凛"]


def test_event_extractor_fails_closed_after_stream_retries(monkeypatch):
    monkeypatch.setattr(
        event_extractor,
        "_call_llm",
        lambda _prompt: (_ for _ in ()).throw(ark_llm.LLMStreamError("broken stream")),
    )
    monkeypatch.setattr(event_extractor.time, "sleep", lambda _seconds: None)

    with pytest.raises(event_extractor.EventExtractionError, match="segment 7"):
        event_extractor.extract_events([{"id": 7, "content": "不可丢失"}])


def test_medium_screenplay_lines_are_coalesced_into_bounded_segments():
    text = "\n".join(f"凛执行第{index:03d}个动作并观察烬的反应。" for index in range(131))

    parsed = text_parser.parse_text(text)

    assert parsed["input_type"] == "medium"
    assert 1 < len(parsed["segments"]) < 20
    assert max(segment["char_count"] for segment in parsed["segments"]) <= text_parser.SEGMENT_MAX_CHARS


def test_environment_objects_do_not_become_character_assets():
    from phases.phase1.character_discoverer import _is_human_character

    for name in ("断裂的霓虹牌", "积水", "破碎路面", "机械手掌", "钢梁"):
        assert not _is_human_character(name)


def test_named_human_role_phrase_is_not_dropped_only_for_length():
    from phases.phase1.character_discoverer import _filter_descriptive_phrases

    stats = {
        "年轻东方古装仙女": {"events": [1, 2], "contexts": []},
        "金色夕阳下的无边云海": {"events": [1], "contexts": []},
    }

    filtered = _filter_descriptive_phrases(stats)

    assert "年轻东方古装仙女" in filtered
    assert "金色夕阳下的无边云海" not in filtered


def test_character_context_includes_visual_appearance_constraints():
    from phases.phase1.character_discoverer import _collect_character_stats

    stats = _collect_character_stats([{
        "id": 2,
        "who": ["年轻东方古装仙女"],
        "where": "白玉栏杆旁",
        "what": "仙女俯瞰云海",
        "visual": "她身穿淡粉白色多层轻纱仙裙，玉簪点缀银色流苏",
        "emotion": "清冷温柔",
    }])

    context = stats["年轻东方古装仙女"]["contexts"][0]
    assert "视觉硬约束" in context
    assert "淡粉白色多层轻纱仙裙" in context


def test_progress_reporter_emits_heartbeat(tmp_path):
    reporter = ProgressReporter(str(tmp_path))
    reporter.start_heartbeat("phase1", interval_s=0.01)
    time.sleep(0.035)
    reporter.stop_heartbeat()

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    heartbeats = [event for event in events if event["event"] == "heartbeat"]
    assert heartbeats
    assert heartbeats[0]["phase"] == "phase1"
    assert "elapsed_s" in heartbeats[0]


def test_director_planner_uses_shared_streaming_client(monkeypatch, tmp_path):
    client = object()
    observed = {}
    events = [{
        "sequence_id": "SEQ001",
        "what": "仙宫从云海中显现",
        "emotion": "敬畏",
    }]
    plan = {
        "schema": director_planner.DIRECTOR_PLAN_SCHEMA,
        "sequences": [{
            "sequence_id": "SEQ001",
            "scene_goal": "建立仙宫尺度",
            "emotion_arc": "平静→敬畏",
            "visual_focus": "仙宫与云海的尺度关系",
            "spatial_intent": "仙宫居中，云海形成纵深",
            "transition_intent": "沿云层运动进入下一段",
        }],
    }

    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")

    def fake_client(**kwargs):
        observed["client_args"] = kwargs
        return client

    monkeypatch.setattr(director_planner, "create_ark_client", fake_client)

    def fake_stream(*, messages, **kwargs):
        observed["messages"] = messages
        observed["stream_args"] = kwargs
        return json.dumps(plan, ensure_ascii=False)

    monkeypatch.setattr(director_planner, "call_llm_stream", fake_stream)

    result = director_planner.plan_director(events, tmp_path)

    assert result["status"] == "done"
    assert observed["client_args"] == {"read_timeout": director_planner.LLM_IDLE_TIMEOUT}
    assert observed["stream_args"]["_client"] is client
    assert observed["stream_args"]["wall_timeout"] == director_planner.LLM_WALL_TIMEOUT
    assert observed["stream_args"]["idle_timeout"] == director_planner.LLM_IDLE_TIMEOUT
    assert observed["stream_args"]["model"] == DEFAULT_TEXT_MODEL
    assert observed["stream_args"]["response_format"]["type"] == "json_schema"
    assert observed["stream_args"]["response_format"]["json_schema"]["strict"] is True
    assert observed["messages"][0]["role"] == "system"
    assert json.loads((tmp_path / "director_plan.json").read_text()) == plan


def test_director_planner_propagates_provider_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")
    monkeypatch.setattr(director_planner, "create_ark_client", lambda **_kwargs: object())

    def provider_timeout(**_kwargs):
        raise TimeoutError("provider wall timeout")

    monkeypatch.setattr(director_planner, "call_llm_stream", provider_timeout)

    with pytest.raises(RuntimeError, match="director planning failed: provider wall timeout"):
        director_planner.plan_director(
            [{"sequence_id": "SEQ001", "what": "云海中的仙宫"}],
            tmp_path,
        )

    assert not (tmp_path / "director_plan.json").exists()


def test_director_planner_live_scope_disables_schema_resubmission(
    monkeypatch,
    tmp_path,
):
    calls = 0
    monkeypatch.setattr(director_planner, "get_api_key", lambda _name: "test-key")
    monkeypatch.setattr(
        director_planner,
        "create_ark_client",
        lambda **_kwargs: object(),
    )

    def invalid_schema(**_kwargs):
        nonlocal calls
        calls += 1
        return "not-json"

    monkeypatch.setattr(director_planner, "call_llm_stream", invalid_schema)
    with provider_attempt_scope(max_retries=0):
        with pytest.raises(RuntimeError, match="director planning failed"):
            director_planner.plan_director(
                [{"sequence_id": "SEQ001", "what": "fictional scene"}],
                tmp_path,
            )

    assert calls == 1


def test_storyboard_long_stream_timeout_budget_is_consistent(monkeypatch):
    client = object()
    observed = {}

    def fake_client(**kwargs):
        observed["client_args"] = kwargs
        return client

    def fake_stream(*, messages, **kwargs):
        observed["messages"] = messages
        observed["stream_args"] = kwargs
        return json.dumps({"prompt": "cinematic cloud sea", "caption": "云海"})

    monkeypatch.setattr(storyboard_generator, "create_ark_client", fake_client)
    monkeypatch.setattr(storyboard_generator, "call_llm_stream", fake_stream)

    result = storyboard_generator._call_llm("synthetic storyboard prompt")

    assert json.loads(result)["caption"] == "云海"
    assert observed["client_args"] == {
        "read_timeout": storyboard_generator.LLM_IDLE_TIMEOUT
    }
    assert observed["stream_args"]["_client"] is client
    assert (
        observed["stream_args"]["wall_timeout"]
        == storyboard_generator.LLM_TIMEOUT
    )
    assert (
        observed["stream_args"]["idle_timeout"]
        == storyboard_generator.LLM_IDLE_TIMEOUT
    )
    assert storyboard_generator.LLM_TIMEOUT >= 360
    assert (
        storyboard_generator.SHOT_WALL_CLOCK_S
        >= storyboard_generator.LLM_TIMEOUT * 2 + 30
    )


def test_style_summary_uses_long_stream_and_idle_timeouts(monkeypatch):
    observed = {}

    def fake_stream(*, messages, **kwargs):
        observed["messages"] = messages
        observed["stream_args"] = kwargs
        return "东方仙侠电影写实风格"

    monkeypatch.setattr(pipeline_core, "call_llm_stream", fake_stream)
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")

    result = pipeline_core._summarize_visual_style_with_llm("云海仙宫")

    assert result == "东方仙侠电影写实风格"
    assert (
        observed["stream_args"]["wall_timeout"]
        == pipeline_core.STYLE_SUMMARY_WALL_TIMEOUT
    )
    assert (
        observed["stream_args"]["read_timeout"]
        == pipeline_core.STYLE_SUMMARY_IDLE_TIMEOUT
    )
    assert (
        observed["stream_args"]["idle_timeout"]
        == pipeline_core.STYLE_SUMMARY_IDLE_TIMEOUT
    )


def test_storyboard_template_supports_current_shot_schema():
    template = "```\n角色：{{CHARACTER_REFERENCE}}\n共{{PANEL_COUNT}}格\n{{STORYBOARD_CONTENT}}\n```"
    storyboard = {"shots": [{
        "visual": "巨大蓝色星球从云层后显现",
        "what": "仙女抬头望向天空",
        "camera_movement": "orbit",
    }]}
    characters = {"characters": [{
        "name": "年轻东方古装仙女",
        "appearance": {"summary": "身穿淡粉白色多层轻纱仙裙"},
    }]}

    prompt = pipeline_core.fill_storyboard_template(template, storyboard, characters)

    assert "面板 1: 巨大蓝色星球从云层后显现" in prompt
    assert "动作: 仙女抬头望向天空" in prompt
    assert "镜头: orbit" in prompt
    assert "淡粉白色多层轻纱仙裙" in prompt


def test_storyboard_template_rejects_empty_panels():
    template = "```\n{{STORYBOARD_CONTENT}}\n```"

    with pytest.raises(ValueError, match="shot 1 has no visual content"):
        pipeline_core.fill_storyboard_template(template, {"shots": [{}]}, {})


def test_video_reference_selects_single_shot_image_not_overview(tmp_path):
    overview = tmp_path / "storyboard.png"
    overview.write_bytes(b"x" * 4096)

    assert pipeline_core._shot_storyboard_reference(tmp_path, 6) is None

    shot_dir = tmp_path / "storyboard_images"
    shot_dir.mkdir()
    shot_image = shot_dir / "S06.png"
    shot_image.write_bytes(b"y" * 4096)

    assert pipeline_core._shot_storyboard_reference(tmp_path, 6) == shot_image
    assert pipeline_core._shot_storyboard_reference(tmp_path, "S06") == shot_image


def test_scenery_keyframe_prompt_does_not_inject_action_movie_concepts():
    prompt = pipeline_core._storyboard_keyframe_description({
        "who": [],
        "what": "镜头穿透厚重云墙",
        "visual": "金色云海与悬浮天宫逐渐显现",
    })

    assert "Environment-only" in prompt
    assert "zero people" in prompt
    for forbidden in ("fight", "weapon", "attacker", "defender", "blade"):
        assert forbidden not in prompt.lower()


def test_character_keyframe_prompt_has_no_unrequested_weapon_vocabulary():
    prompt = pipeline_core._storyboard_keyframe_description({
        "who": ["仙女"],
        "subject_description": "淡粉白色轻纱仙裙",
        "what": "仙女抚琴",
        "visual": "仙女端坐古琴前",
    })

    assert "Character identity lock" in prompt
    for forbidden in ("fight", "weapon", "attacker", "defender", "blade"):
        assert forbidden not in prompt.lower()


def test_scenery_keyframe_does_not_absorb_project_plot_objects():
    shot = {
        "id": 1,
        "who": [],
        "what": "镜头掠过云层表面",
        "visual": "金色夕阳下无边云海剧烈翻涌",
    }
    prompt = pipeline_core._storyboard_keyframe_description(shot)
    project_style = "东方神话风格，悬浮天宫，仙女与古琴"

    assert "天宫" not in prompt
    assert "仙女" not in prompt
    assert "古琴" not in prompt
    assert project_style not in prompt


def test_shot_image_batch_does_not_append_global_style_summary(monkeypatch, tmp_path):
    observed = {}

    def fake_batch(scenes, style_context):
        observed["scenes"] = scenes
        observed["style_context"] = style_context
        return [{"scene_id": 1, "prompt": scenes[0]["description"]}]

    monkeypatch.setattr(pipeline_core, "build_batch_prompts", fake_batch)
    result = pipeline_core._generate_shot_images(tmp_path, {"shots": [{
        "id": 1,
        "who": ["仙女"],
        "what": "仙女抚琴",
        "visual": "仙女端坐古琴前",
    }]})

    assert result == 0
    assert observed["style_context"] is None


def test_character_prompt_locks_explicit_costume_facts():
    import phases.phase1.character_discoverer as character_discoverer

    assert "必须原样保留" in character_discoverer.SYSTEM_PROMPT
    assert "不能把淡粉改成月白" in character_discoverer.USER_PROMPT_TEMPLATE


def test_combined_phase1_starts_progress_before_director(monkeypatch, tmp_path):
    calls = []

    class Reporter:
        _current_phase = None
        _progress_pct = 0

        def phase_start(self, phase, name):
            self._current_phase = phase
            calls.append(("phase_start", phase, name))

        def step(self, phase, message, progress_pct=None):
            self._progress_pct = progress_pct or self._progress_pct
            calls.append(("step", phase, message))

        def start_heartbeat(self, phase):
            calls.append(("heartbeat_start", phase))

        def stop_heartbeat(self):
            calls.append(("heartbeat_stop",))

    def fake_director(*_args, **_kwargs):
        calls.append(("director",))
        return {"status": "done"}

    monkeypatch.setattr(pipeline_core, "run_phase1_director", fake_director)

    def fake_screenwriter(
        _text,
        output_dir,
        _duration,
        dry_run,
        *,
        _director_runner,
        **_kwargs,
    ):
        calls.append(("event_extractor",))
        director = _director_runner(
            [{"sequence_id": "SEQ001", "what": "synthetic"}],
            output_dir,
            dry_run,
        )
        return {"status": "done", "director": director}

    monkeypatch.setattr(
        pipeline_core,
        "run_phase1_screenwriter",
        fake_screenwriter,
    )

    result = pipeline_core.run_phase1(
        "synthetic", tmp_path, 30, False, reporter=Reporter()
    )

    assert result["status"] == "done"
    assert calls.index(("heartbeat_start", "phase1")) < calls.index(("event_extractor",))
    assert calls.index(("event_extractor",)) < calls.index(("director",))
    assert calls[-1] == ("heartbeat_stop",)


def test_combined_phase1_fails_closed_before_adaptation_when_director_fails(tmp_path):
    screenwriter_called = False
    adaptation_called = False

    def failed_director(*_args, **_kwargs):
        return {"status": "error", "error": "director timeout"}

    def screenwriter_with_failed_director(
        _text,
        output_dir,
        _duration,
        dry_run,
        *,
        _director_runner,
        **_kwargs,
    ):
        nonlocal adaptation_called, screenwriter_called
        from phases.phase1.phase1_screenwriter import _plan_director_for_events

        screenwriter_called = True
        _plan_director_for_events(
            [{"sequence_id": "SEQ001", "what": "synthetic"}],
            output_dir,
            dry_run,
            _director_runner,
        )
        adaptation_called = True
        return {"status": "done"}

    with pytest.raises(RuntimeError, match="director planning returned error"):
        run_phase1(
            "synthetic",
            tmp_path,
            30,
            False,
            _director_runner=failed_director,
            _screenwriter_runner=screenwriter_with_failed_director,
        )

    assert screenwriter_called is True
    assert adaptation_called is False


def test_storyboard_default_path_runs_three_shots_concurrently(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()

    def generate(_shot, index, _total, *_args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"id": index, "name": str(index), "prompt": "synthetic"}

    monkeypatch.delenv("HONCUT_SHOT_QUEUE", raising=False)
    monkeypatch.setattr(storyboard_generator, "_generate_single_shot", generate)
    result = storyboard_generator.generate_storyboard([
        {"suggested_duration": 1} for _ in range(3)
    ])

    assert peak == 3
    assert [shot["id"] for shot in result["shots"]] == [1, 2, 3]


def test_phase1_checkpoints_are_written_and_reused(monkeypatch, tmp_path):
    import phases.phase1.character_discoverer as character_discoverer
    import phases.phase1.adaptation_engine as adaptation_engine
    import prompt.text_parser as text_parser

    calls = {"events": 0, "characters": 0}
    events_payload = {
        "events": [{"id": 1, "sequence_id": "SEQ001", "who": []}]
    }
    characters_payload = _deterministic_character_fixture(events_payload["events"])

    def director_with_body_repair(events, *_args, **_kwargs):
        events[0]["body_action_director_repairs"] = [
            {
                "micro_action_index": 1,
                "fields": ["footwork"],
            }
        ]
        return {
            "status": "done",
            "plan": {"schema": "honcut.director-plan.v1", "sequences": []},
        }

    monkeypatch.setattr(text_parser, "parse_text", lambda _text: {"segments": [{"id": 1}]})
    monkeypatch.setattr(
        event_extractor,
        "extract_events",
        lambda _segments, **_kwargs: (
            calls.__setitem__("events", calls["events"] + 1) or events_payload
        ),
    )
    monkeypatch.setattr(character_discoverer, "discover_characters", lambda _events, **_kwargs: (calls.__setitem__("characters", calls["characters"] + 1) or characters_payload))
    monkeypatch.setattr(
        adaptation_engine,
        "adapt_events",
        lambda *_args, **_kwargs: _phase1_stub_adapted_result(
            adaptation_engine
        ),
    )
    monkeypatch.setattr(
        storyboard_generator,
        "generate_storyboard",
        lambda shots, *_args, **_kwargs: {
            "shots": [{**shots[0], "id": "S01", "duration": 15}]
        },
    )
    monkeypatch.setattr(pipeline_core, "_integrate_storyboard_prompts", lambda value, _characters: value)
    monkeypatch.setattr(pipeline_core, "annotate_shot_pacing", lambda _shots: None)
    monkeypatch.setattr(pipeline_core, "_summarize_visual_style_with_llm", lambda _text: None)
    monkeypatch.setattr(
        pipeline_core,
        "_attach_director_storyboard",
        lambda *_args, **_kwargs: {"panels": []},
    )
    monkeypatch.setattr(pipeline_core, "run_quality_check", lambda *_args: SimpleNamespace(passed=True, grade="A"))
    monkeypatch.setattr("quality.quality_gate.run_storyboard_review", lambda **_kwargs: {"grade": "A"})
    monkeypatch.setattr(
        pipeline_core,
        "run_phase1_director",
        director_with_body_repair,
    )

    first = pipeline_core.run_phase1_screenwriter("synthetic input", tmp_path, 15, False)
    second = pipeline_core.run_phase1_screenwriter("synthetic input", tmp_path, 15, False)

    assert first["status"] == second["status"] == "done"
    assert calls == {"events": 1, "characters": 1}
    stored_events = json.loads((tmp_path / "phase1_events.json").read_text())
    stored_characters = json.loads((tmp_path / "phase1_characters.json").read_text())
    assert stored_events["events"] == events_payload["events"]
    assert stored_events["director_body_repair_count"] == 1
    assert stored_characters["characters"] == characters_payload["characters"]
    assert stored_events["_checkpoint"]["schema_version"] == pipeline_core.PHASE1_CHECKPOINT_SCHEMA_VERSION
    assert stored_characters["_checkpoint"]["schema_version"] == pipeline_core.PHASE1_CHECKPOINT_SCHEMA_VERSION


def test_phase1_production_route_preserves_canonical_four_plus_three_layout(
    monkeypatch,
    tmp_path,
):
    import phases.phase1.adaptation_engine as adaptation_engine
    import phases.phase1.character_discoverer as character_discoverer

    monkeypatch.setattr(
        text_parser,
        "parse_text",
        lambda _text: {"segments": [{"id": 1, "content": "连续动作"}]},
    )
    monkeypatch.setattr(
        event_extractor,
        "extract_events",
        lambda _segments, **_kwargs: {
            "events": [{"id": 1, "sequence_id": "SEQ001", "who": []}]
        },
    )
    monkeypatch.setattr(
        character_discoverer,
        "discover_characters",
        lambda _events, **_kwargs: _deterministic_character_fixture(_events),
    )
    monkeypatch.setattr(
        adaptation_engine,
        "adapt_events",
        lambda *_args, **_kwargs: _phase1_long_shot_adapted_result(
            adaptation_engine
        ),
    )
    monkeypatch.setattr(
        storyboard_generator,
        "generate_storyboard",
        lambda shots, *_args, **_kwargs: {
            "shots": [
                {**shot, "duration": shot["suggested_duration"]}
                for shot in shots
            ]
        },
    )
    monkeypatch.setattr(
        pipeline_core,
        "_integrate_storyboard_prompts",
        lambda value, _characters: value,
    )
    monkeypatch.setattr(
        pipeline_core,
        "annotate_shot_pacing",
        lambda _shots: None,
    )
    monkeypatch.setattr(
        pipeline_core,
        "_summarize_visual_style_with_llm",
        lambda _text: None,
    )
    monkeypatch.setattr(
        pipeline_core,
        "_attach_director_storyboard",
        lambda *_args, **_kwargs: {"panels": []},
    )
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda *_args: SimpleNamespace(passed=True, grade="A"),
    )
    monkeypatch.setattr(
        "quality.quality_gate.run_storyboard_review",
        lambda **_kwargs: {"grade": "A"},
    )
    monkeypatch.setattr(
        pipeline_core,
        "run_phase1_director",
        lambda *_args, **_kwargs: {
            "status": "done",
            "plan": {"schema": "honcut.director-plan.v1", "sequences": []},
        },
    )

    result = pipeline_core.run_phase1_screenwriter(
        "连续动作",
        tmp_path,
        36,
        False,
        shot_duration=6,
        shot_policy="continuity",
    )

    assert result["status"] == "done"
    storyboard = result["_storyboard"]
    assert [
        shot["storyboard_beat_count"] for shot in storyboard["shots"]
    ] == [4, 3]
    assert [
        [beat["provider_request_duration_s"] for beat in shot["storyboard_beats"]]
        for shot in storyboard["shots"]
    ] == [[8, 6, 6, 6], [8, 6, 6]]
    assert storyboard["primary_shot_execution"]["primary_shot_layout_sha256"] == (
        storyboard["screenplay_plan"]["primary_shot_layout_sha256"]
    )


def test_phase1_legacy_checkpoint_is_regenerated(monkeypatch, tmp_path):
    import phases.phase1.character_discoverer as character_discoverer
    import phases.phase1.adaptation_engine as adaptation_engine
    import prompt.text_parser as text_parser

    (tmp_path / "phase1_events.json").write_text(
        json.dumps({"events": [{"id": 99, "who": ["积水"]}]}),
        encoding="utf-8",
    )
    calls = {"events": 0}
    monkeypatch.setattr(text_parser, "parse_text", lambda _text: {"segments": [{"id": 1, "content": "凛出现"}]})
    monkeypatch.setattr(
        event_extractor,
        "extract_events",
        lambda _segments, **_kwargs: (
            calls.__setitem__("events", calls["events"] + 1)
            or {
                "events": [
                    {"id": 1, "sequence_id": "SEQ001", "who": ["凛"]}
                ]
            }
        ),
    )
    monkeypatch.setattr(
        character_discoverer,
        "discover_characters",
        lambda _events, **_kwargs: _deterministic_character_fixture(_events),
    )
    monkeypatch.setattr(
        adaptation_engine,
        "adapt_events",
        lambda *_args, **_kwargs: _phase1_stub_adapted_result(
            adaptation_engine
        ),
    )
    monkeypatch.setattr(
        storyboard_generator,
        "generate_storyboard",
        lambda shots, *_args, **_kwargs: {
            "shots": [{**shots[0], "id": "S01", "duration": 15}]
        },
    )
    monkeypatch.setattr(pipeline_core, "_integrate_storyboard_prompts", lambda value, _characters: value)
    monkeypatch.setattr(pipeline_core, "annotate_shot_pacing", lambda _shots: None)
    monkeypatch.setattr(pipeline_core, "_summarize_visual_style_with_llm", lambda _text: None)
    monkeypatch.setattr(
        pipeline_core,
        "_attach_director_storyboard",
        lambda *_args, **_kwargs: {"panels": []},
    )
    monkeypatch.setattr(pipeline_core, "run_quality_check", lambda *_args: SimpleNamespace(passed=True, grade="A"))
    monkeypatch.setattr("quality.quality_gate.run_storyboard_review", lambda **_kwargs: {"grade": "A"})
    monkeypatch.setattr(
        pipeline_core,
        "run_phase1_director",
        lambda *_args, **_kwargs: {
            "status": "done",
            "plan": {"schema": "honcut.director-plan.v1", "sequences": []},
        },
    )

    result = pipeline_core.run_phase1_screenwriter("凛出现", tmp_path, 15, False)

    assert result["status"] == "done"
    assert calls["events"] == 1
    assert json.loads((tmp_path / "phase1_events.json").read_text())["events"][0]["id"] == 1


def test_phase1_reporter_receives_steps_and_stops_heartbeat(monkeypatch, tmp_path):
    steps = []

    class Reporter:
        def start_heartbeat(self, phase):
            steps.append(("start", phase))

        def step(self, phase, message, progress_pct=None):
            steps.append((phase, message, progress_pct))

        def stop_heartbeat(self):
            steps.append(("stop", None))

    result = pipeline_core.run_phase1_screenwriter(
        "synthetic input", tmp_path, 10, True, reporter=Reporter()
    )

    assert result["status"] == "done"
    assert any(item[0] == "phase1" for item in steps)
    assert steps[-1] == ("stop", None)


def test_flf2v_similarity_gate_allows_changed_staging_but_rejects_copy(tmp_path):
    from PIL import Image

    first = tmp_path / "first.png"
    changed = tmp_path / "changed.png"
    copied = tmp_path / "copied.png"
    Image.new("RGB", (128, 72), (100, 100, 100)).save(first)
    # The grayscale-MSE similarity is about 0.933: visibly changed luminance,
    # but above the former 0.93 cutoff observed to reject real action progress.
    Image.new("RGB", (128, 72), (166, 166, 166)).save(changed)
    Image.new("RGB", (128, 72), (100, 100, 100)).save(copied)

    changed_report = pipeline_core._validate_end_frame(first, changed)
    copied_report = pipeline_core._validate_end_frame(first, copied)

    assert changed_report["passed"] is True
    assert 0.93 < changed_report["similarity"] < 0.97
    assert copied_report["passed"] is False
    assert "too similar" in copied_report["reason"]
