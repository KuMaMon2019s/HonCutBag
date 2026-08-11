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

from phases import pipeline_core, storyboard_generator
from prompt import event_extractor
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
            _client=client,
        )
    assert stream.closed.is_set()


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
    import phases.character_discoverer as character_discoverer
    import phases.adaptation_engine as adaptation_engine
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

    first = pipeline_core.run_phase2("synthetic input", tmp_path, 10, False)
    second = pipeline_core.run_phase2("synthetic input", tmp_path, 10, False)

    assert first["status"] == second["status"] == "done"
    assert calls == {"events": 1, "characters": 1}
    assert json.loads((tmp_path / "phase1_events.json").read_text()) == events_payload
    assert json.loads((tmp_path / "phase1_characters.json").read_text()) == characters_payload


def test_phase1_reporter_receives_steps_and_stops_heartbeat(monkeypatch, tmp_path):
    steps = []

    class Reporter:
        def start_heartbeat(self, phase):
            steps.append(("start", phase))

        def step(self, phase, message, progress_pct=None):
            steps.append((phase, message, progress_pct))

        def stop_heartbeat(self):
            steps.append(("stop", None))

    result = pipeline_core.run_phase2("synthetic input", tmp_path, 10, True, reporter=Reporter())

    assert result["status"] == "done"
    assert any(item[0] == "phase1" for item in steps)
    assert steps[-1] == ("stop", None)
