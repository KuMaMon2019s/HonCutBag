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

from phases import pipeline_core
import phases.phase1.storyboard_generator as storyboard_generator
from prompt import event_extractor, text_parser
from utils import ark_llm
from utils.progress_reporter import ProgressReporter


def _chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
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

    class ActiveStream:
        def __iter__(self):
            for content in ("a", "b", "c"):
                time.sleep(0.02)
                yield _chunk(content)

        def close(self):
            pass

    completions = SimpleNamespace(create=lambda **_kwargs: ActiveStream())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    content = ark_llm.call_llm_stream(
        [{"role": "user", "content": "synthetic"}],
        wall_timeout=1,
        idle_timeout=0.03,
        heartbeat_callback=lambda: callbacks.append(time.monotonic()),
        heartbeat_interval=1,
        _client=client,
    )

    assert content == "abc"
    assert len(callbacks) == 1


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

    def call(_prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ark_llm.LLMStreamError("incomplete chunked read")
        return json.dumps([{
            "who": ["凛"],
            "where": "高架",
            "what": "凛出现",
            "emotion": "紧张",
            "visual": "凛站在高架上",
            "time": "夜晚",
            "action_type": "reveal",
        }], ensure_ascii=False)

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
    events_payload = {"events": [{"id": 1, "who": []}]}
    characters_payload = {"characters": []}

    monkeypatch.setattr(text_parser, "parse_text", lambda _text: {"segments": [{"id": 1}]})
    monkeypatch.setattr(event_extractor, "extract_events", lambda _segments: (calls.__setitem__("events", calls["events"] + 1) or events_payload))
    monkeypatch.setattr(character_discoverer, "discover_characters", lambda _events: (calls.__setitem__("characters", calls["characters"] + 1) or characters_payload))
    monkeypatch.setattr(adaptation_engine, "adapt_events", lambda *_args, **_kwargs: {"shots": [{}]})
    monkeypatch.setattr(storyboard_generator, "generate_storyboard", lambda *_args, **_kwargs: {"shots": []})
    monkeypatch.setattr(pipeline_core, "_integrate_storyboard_prompts", lambda value, _characters: value)
    monkeypatch.setattr(pipeline_core, "annotate_shot_pacing", lambda _shots: None)
    monkeypatch.setattr(pipeline_core, "_summarize_visual_style_with_llm", lambda _text: None)
    monkeypatch.setattr(pipeline_core, "run_quality_check", lambda *_args: SimpleNamespace(passed=True, grade="A"))
    monkeypatch.setattr("quality.quality_gate.run_storyboard_review", lambda **_kwargs: {"grade": "A"})

    first = pipeline_core.run_phase1_screenwriter("synthetic input", tmp_path, 10, False)
    second = pipeline_core.run_phase1_screenwriter("synthetic input", tmp_path, 10, False)

    assert first["status"] == second["status"] == "done"
    assert calls == {"events": 1, "characters": 1}
    stored_events = json.loads((tmp_path / "phase1_events.json").read_text())
    stored_characters = json.loads((tmp_path / "phase1_characters.json").read_text())
    assert stored_events["events"] == events_payload["events"]
    assert stored_characters["characters"] == characters_payload["characters"]
    assert stored_events["_checkpoint"]["schema_version"] == 2
    assert stored_characters["_checkpoint"]["schema_version"] == 2


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
        lambda _segments: (
            calls.__setitem__("events", calls["events"] + 1)
            or {"events": [{"id": 1, "who": ["凛"]}]}
        ),
    )
    monkeypatch.setattr(
        character_discoverer,
        "discover_characters",
        lambda _events: {"characters": [], "total_characters": 0},
    )
    monkeypatch.setattr(adaptation_engine, "adapt_events", lambda *_args, **_kwargs: {"shots": [{}]})
    monkeypatch.setattr(storyboard_generator, "generate_storyboard", lambda *_args, **_kwargs: {"shots": []})
    monkeypatch.setattr(pipeline_core, "_integrate_storyboard_prompts", lambda value, _characters: value)
    monkeypatch.setattr(pipeline_core, "annotate_shot_pacing", lambda _shots: None)
    monkeypatch.setattr(pipeline_core, "_summarize_visual_style_with_llm", lambda _text: None)
    monkeypatch.setattr(pipeline_core, "run_quality_check", lambda *_args: SimpleNamespace(passed=True, grade="A"))
    monkeypatch.setattr("quality.quality_gate.run_storyboard_review", lambda **_kwargs: {"grade": "A"})

    result = pipeline_core.run_phase1_screenwriter("凛出现", tmp_path, 10, False)

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
