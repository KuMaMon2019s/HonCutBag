import json
import sys
from pathlib import Path
from types import SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import storyboard_generator


def _shot():
    return {
        "what": "通用镜头",
        "visual": "一个通用测试场景",
        "where": "测试地点",
        "suggested_duration": 3,
    }


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
