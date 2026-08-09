import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from clients import local_video_client, seedance_client
from phases import pipeline_core
from tools import asset_packager
import phase_orchestrator

_runner_spec = importlib.util.spec_from_file_location(
    "pipeline_runner_cli", SRC / "pipeline_runner.py"
)
pipeline_runner_cli = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(pipeline_runner_cli)


def test_phase_orchestrator_failure_prints_stdout_and_stderr_tails(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "input": str(tmp_path / "story.txt"),
        "duration": 45,
        "output_dir": str(tmp_path),
    }))
    (tmp_path / "pipeline_report.json").write_text(json.dumps({"phases": {}}))
    monkeypatch.setattr(
        phase_orchestrator,
        "run_phase",
        lambda phase, config: {
            "phase": phase,
            "exit_code": 1,
            "stdout": "A" * 500 + "stdout-cause",
            "stderr": "B" * 500 + "stderr-cause",
            "timestamp": "2026-08-10T00:00:00",
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["phase_orchestrator.py", "--config", str(config_path), "--resume-from", "phase4"]
    )

    with pytest.raises(SystemExit, match="1"):
        phase_orchestrator.main()

    output = capsys.readouterr().out
    assert "Stdout tail:" in output and "stdout-cause" in output
    assert "Stderr tail:" in output and "stderr-cause" in output


def test_pipeline_runner_prints_selected_phase_error(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_runner_cli._core,
        "run_pipeline",
        lambda **kwargs: {
            "status": "partial",
            "phases": {"4": {"status": "error", "error": "orchestrator timed out"}},
        },
    )
    monkeypatch.setattr(sys, "argv", ["pipeline_runner.py", "--input", "story.txt", "--phase", "phase4"])

    with pytest.raises(SystemExit, match="1"):
        pipeline_runner_cli.main()

    assert "Phase phase4 failed: orchestrator timed out" in capsys.readouterr().out


def test_phase4_timeout_prints_subprocess_output_tails(monkeypatch, tmp_path, capsys):
    (tmp_path / "STORYBOARD.json").write_text(json.dumps({"shots": []}))
    monkeypatch.setattr(
        pipeline_core.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pipeline_core.subprocess.TimeoutExpired(
                args[0], 120, output=b"inner stdout cause", stderr=b"inner stderr cause"
            )
        ),
    )

    result = pipeline_core.run_phase4(tmp_path, dry_run=False)

    output = capsys.readouterr().out
    assert result["status"] == "error"
    assert result["error"] == "orchestrator timed out"
    assert "inner stdout cause" in output
    assert "inner stderr cause" in output


def test_submit_content_sends_top_level_agent_plan_payload(monkeypatch):
    content = [
        {"type": "text", "text": "move slowly"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/frame.jpg"},
            "role": "first_frame",
        },
    ]
    posted = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "task-direct-1"}

    def fake_post(url, **kwargs):
        posted.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(seedance_client.requests, "post", fake_post)

    task_id = seedance_client.submit_content(
        content,
        api_key="test-key",
        model="doubao-seedance-2.0-mini",
        duration=12,
        seed=42,
        generate_audio="enabled",
    )

    assert task_id == "task-direct-1"
    assert posted["url"] == seedance_client.SUBMIT_ENDPOINT
    assert posted["json"] == {
        "model": "doubao-seedance-2.0-mini",
        "content": content,
        "generate_audio": "enabled",
        "ratio": "16:9",
        "duration": 12,
        "watermark": False,
        "seed": 42,
    }
    assert posted["json"]["content"] is content
    assert "parameters" not in posted["json"]


def _write_shot(output_dir):
    shot_dir = output_dir / "shots" / "S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "quiet landscape", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    return shot_dir


def _mock_common_direct(monkeypatch, shot_dir):
    monkeypatch.setattr(pipeline_core, "get_api_key", lambda service: "test-key")
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    monkeypatch.setattr(seedance_client, "poll", lambda task_id, api_key: "https://video.test/out.mp4")

    def fake_download(url, output_path):
        Path(output_path).write_bytes(b"v" * 11000)
        return output_path

    monkeypatch.setattr(seedance_client, "download", fake_download)
    monkeypatch.setattr(
        pipeline_core.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12.0\n", returncode=0),
    )


@pytest.mark.parametrize("provider", [None, "seedance", "ark"])
def test_direct_providers_bypass_bridge(tmp_path, monkeypatch, provider):
    shot_dir = _write_shot(tmp_path)
    _mock_common_direct(monkeypatch, shot_dir)
    if provider is None:
        monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("VIDEO_PROVIDER", provider)
    direct_calls = []
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda content, **kwargs: direct_calls.append((content, kwargs)) or "task-1",
    )
    monkeypatch.setattr(
        local_video_client,
        "is_available",
        lambda timeout: pytest.fail("Bridge availability must not be checked"),
    )

    result = pipeline_core._run_phase5_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(direct_calls) == 1
    assert direct_calls[0][1]["duration"] == 12
    assert (shot_dir / "output.mp4").exists()


@pytest.mark.parametrize("provider", ["local", "wan", "bridge"])
def test_explicit_bridge_providers_use_local_client(tmp_path, monkeypatch, provider):
    _write_shot(tmp_path)
    monkeypatch.setenv("VIDEO_PROVIDER", provider)
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    bridge_calls = []

    def fake_generate(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"v" * 11000)
        bridge_calls.append(kwargs)
        return {
            "output_path": kwargs["output_path"],
            "last_frame_path": None,
            "actual_model": kwargs["model"],
        }

    monkeypatch.setattr(local_video_client, "generate_video", fake_generate)
    monkeypatch.setattr(local_video_client, "generate_video_with_fallback", fake_generate)
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda *args, **kwargs: pytest.fail("Direct ARK must not be called"),
    )
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": "shot"}],
    )

    result = pipeline_core._run_phase5_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(bridge_calls) == 1


def test_direct_ark_quota_error_uses_existing_retry_loop(tmp_path, monkeypatch):
    shot_dir = _write_shot(tmp_path)
    _mock_common_direct(monkeypatch, shot_dir)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    attempts = []

    def flaky_submit(content, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("429 QuotaExceeded")
        return "task-after-retry"

    monkeypatch.setattr(seedance_client, "submit_content", flaky_submit)
    monkeypatch.setattr(pipeline_core.time, "sleep", lambda seconds: None)

    result = pipeline_core._run_phase5_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(attempts) == 2
