import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import storyboard_generator
from utils import shot_queue


def _shot():
    return {
        "what": "通用镜头",
        "visual": "一个通用测试场景",
        "where": "测试地点",
        "suggested_duration": 3,
    }


def _queue_payload(index):
    return shot_queue.make_payload(
        {"what": f"shot-{index}", "who": []}, index, 3, characters=[],
        visual_style_text="style", scene_style_map={}, previous_shot=None,
        visual_style_path=None,
    )


def test_shot_queue_payload_recovery_and_sorting(tmp_path):
    payload = _queue_payload(2)
    assert shot_queue.deserialize_payload(shot_queue.serialize_payload(payload)) == payload
    checkpoint = tmp_path / "shots_partial.json"
    shot_queue.write_completed(checkpoint, "run-a", {2: {"id": 2}})
    completed = shot_queue.load_completed(checkpoint, "run-a")
    assert [item["index"] for item in shot_queue.missing_payloads(
        [_queue_payload(1), _queue_payload(2), _queue_payload(3)], completed
    )] == [1, 3]
    assert shot_queue.sorted_results({2: {"id": 2}, 1: {"id": 1}}) == [
        {"id": 1}, {"id": 2}
    ]


def test_shot_queue_enqueue_worker_collect_without_redis(tmp_path, monkeypatch):
    import pipeline.src.phases.storyboard_generator as canonical_generator

    monkeypatch.setattr(canonical_generator, "_generate_single_shot", lambda **payload: {
        "id": payload["index"], "name": payload["shot"]["what"]
    })

    class FakeJob:
        def __init__(self, payload): self.payload = payload
        async def result(self, timeout=None):
            return await shot_queue.generate_shot_job({"job_try": 1}, self.payload)

    class FakeRedis:
        async def enqueue_job(self, _function, payload, **kwargs): return FakeJob(payload)
        async def aclose(self): pass

    async def pool_factory(_settings): return FakeRedis()
    result = asyncio.run(shot_queue.enqueue_and_collect(
        [_queue_payload(3), _queue_payload(1), _queue_payload(2)],
        run_tag="integration", partial_path=tmp_path / "shots_partial.json",
        pool_factory=pool_factory,
    ))
    assert [item["id"] for item in result] == [1, 2, 3]


def test_call_llm_uses_streaming_and_joins_chunks(monkeypatch):
    calls = []
    stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"prompt":"A '))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='shot","caption":"测试"}'))]),
    ]

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return stream

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(storyboard_generator, "_get_client", lambda: client)

    result = storyboard_generator._call_llm("测试 prompt", "测试风格")

    assert json.loads(result)["prompt"] == "A shot"
    assert calls[0]["stream"] is True
    assert calls[0]["max_tokens"] == 16000
    assert "测试风格" in calls[0]["messages"][0]["content"]


def test_shot_wall_clock_timeout_uses_fallback(monkeypatch, capsys):
    clock = iter([0, 0, 200, 200, 400, 400])
    monkeypatch.setattr(storyboard_generator.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(storyboard_generator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        storyboard_generator,
        "_call_llm",
        lambda *_args: (_ for _ in ()).throw(TimeoutError("mock timeout")),
    )

    result = storyboard_generator.generate_storyboard([_shot()], visual_style_text="测试风格")

    assert result["shots"][0]["caption"] == "通用镜头"
    assert "总时限 480s 到，使用降级方案" in capsys.readouterr().err


def test_shot_progress_heartbeat_is_printed(monkeypatch, capsys):
    monkeypatch.setattr(
        storyboard_generator,
        "_call_llm",
        lambda *_args: '{"prompt":"A cinematic shot","caption":"测试镜头"}',
    )

    storyboard_generator.generate_storyboard([_shot()], visual_style_text="测试风格")

    stdout = capsys.readouterr().out
    assert "Shot 1/1 开始生成..." in stdout
    assert "Shot 1/1 ✅ 完成" in stdout
