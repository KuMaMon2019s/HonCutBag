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
from utils import shot_embedder
import phase_orchestrator

_runner_spec = importlib.util.spec_from_file_location(
    "pipeline_runner_cli", SRC / "pipeline_runner.py"
)
pipeline_runner_cli = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(pipeline_runner_cli)


def test_phase_ids_are_contiguous_in_execution_order():
    expected = [f"phase{number}" for number in range(1, 10)]
    assert phase_orchestrator.PHASES == expected
    assert list(pipeline_runner_cli.PHASES) == expected
    assert phase_orchestrator.PHASE_NUMBERS == {
        phase: str(index) for index, phase in enumerate(expected, start=1)
    }


def test_progress_file_is_written_with_new_phase_ids(tmp_path):
    progress = tmp_path / "phase_progress.json"
    phase_orchestrator._write_progress(
        progress,
        {
            "results": [{"phase": "phase1", "exit_code": 0}],
            "current_phase": "phase2",
            "status": "running",
            "phases": phase_orchestrator.PHASES,
        },
    )
    written = json.loads(progress.read_text(encoding="utf-8"))
    assert written["phases"] == [f"phase{number}" for number in range(1, 10)]
    assert written["current_phase"] == "phase2"


def test_embed_image_sends_tos_url_as_string_input(monkeypatch, tmp_path):
    image_path = tmp_path / "arbitrary-frame.png"
    image_bytes = b"\x89PNG\r\n\x1a\nproject-specific-image-data"
    image_path.write_bytes(image_bytes)
    signed_url = "https://example-bucket.tos.example/arbitrary/object.png?signature=test"
    request = {}

    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-api-key")

    def fake_upload(data, content_type):
        assert data == image_bytes
        assert content_type == "image/png"
        return signed_url

    def fake_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"data": [{"embedding": [0.25, 0.75]}]},
        )

    monkeypatch.setattr(shot_embedder.tos_uploader, "upload_image", fake_upload)
    monkeypatch.setattr(shot_embedder.requests, "post", fake_post)

    assert shot_embedder.embed_image(str(image_path)) == [0.25, 0.75]
    assert request["url"] == f"{shot_embedder.ARK_BASE_URL}/embeddings"
    assert request["json"] == {
        "model": shot_embedder.EMBEDDING_MODEL,
        "input": [signed_url],
    }
    assert isinstance(request["json"]["input"][0], str)


def test_embed_image_does_not_call_api_when_tos_upload_fails(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg-data")

    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-api-key")
    monkeypatch.setattr(shot_embedder.tos_uploader, "upload_image", lambda *_: None)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("embedding API must not receive an invalid local image reference")

    monkeypatch.setattr(shot_embedder.requests, "post", unexpected_post)

    assert shot_embedder.embed_image(str(image_path)) is None


def test_detect_shot_characters_resolves_display_names(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "lin", "name": "凛"}]}),
        encoding="utf-8",
    )
    shot_meta = {"associate_assets": ["char:凛"]}

    assert asset_packager._detect_shot_characters(tmp_path, shot_meta) == ["lin"]


def test_detect_shot_characters_passthrough_valid_ids(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "lin", "name": "凛"}]}),
        encoding="utf-8",
    )
    shot_meta = {"associate_assets": ["char:lin"]}

    assert asset_packager._detect_shot_characters(tmp_path, shot_meta) == ["lin"]


def test_detect_shot_characters_resolves_explicit_display_names(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({
            "characters": [
                {"id": "lin", "name": "凛"},
                {"id": "jin", "name": "烬"},
            ]
        }),
        encoding="utf-8",
    )
    shot_meta = {"_char_ids": ["凛", "烬"]}

    assert sorted(asset_packager._detect_shot_characters(tmp_path, shot_meta)) == [
        "jin",
        "lin",
    ]


def test_detect_shot_characters_passes_through_explicit_valid_id(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "lin", "name": "凛"}]}),
        encoding="utf-8",
    )
    shot_meta = {"_char_ids": ["lin"]}

    assert asset_packager._detect_shot_characters(tmp_path, shot_meta) == ["lin"]


def test_detect_shot_characters_passes_through_unknown_explicit_name(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "lin", "name": "凛"}]}),
        encoding="utf-8",
    )
    shot_meta = {"_char_ids": ["幽灵"]}

    assert asset_packager._detect_shot_characters(tmp_path, shot_meta) == ["幽灵"]


def test_collect_character_references_resolves_explicit_display_names(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({
            "characters": [
                {"id": "lin", "name": "凛"},
                {"id": "jin", "name": "烬"},
            ]
        }),
        encoding="utf-8",
    )
    for char_id in ("lin", "jin"):
        char_dir = tmp_path / "characters" / char_id
        char_dir.mkdir(parents=True)
        (char_dir / "face_closeup.png").write_bytes(b"f" * 1025)
        (char_dir / "full_body.png").write_bytes(b"b" * 1025)

    references = asset_packager.collect_character_reference_assets(
        tmp_path, {"_char_ids": ["凛", "烬"]}
    )

    assert len(references) >= 2
    assert {reference["char_id"] for reference in references} == {"lin", "jin"}


def _stub_tos_upload(monkeypatch):
    monkeypatch.setattr(
        "clients.tos_uploader.upload_image",
        lambda image_data, content_type: f"https://tos.test/{len(image_data)}.png",
    )


def test_phantom_content_uses_reference_images_only(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    storyboard_dir.mkdir()
    (storyboard_dir / "S01.png").write_bytes(b"s" * 1025)
    char_dir = tmp_path / "characters" / "lin"
    char_dir.mkdir(parents=True)
    (char_dir / "face_closeup.png").write_bytes(b"f" * 1025)
    (char_dir / "full_body.png").write_bytes(b"b" * 1025)

    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S01",
        {"prompt": "phantom shot", "gen_strategy": "phantom", "_char_ids": ["lin"]},
    )

    roles = [item.get("role", item["type"]) for item in content]
    assert "first_frame" not in roles
    assert "last_frame" not in roles
    assert roles.count("reference_image") >= 2
    assert roles[0] == "text"


def test_flf2v_content_keeps_first_and_last_frames(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    storyboard_dir.mkdir()
    (storyboard_dir / "S01.png").write_bytes(b"s" * 1025)
    (storyboard_dir / "S01_end.png").write_bytes(b"e" * 1025)

    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S01",
        {"prompt": "frame shot", "gen_strategy": "flf2v"},
    )

    assert [item.get("role", item["type"]) for item in content] == [
        "text",
        "first_frame",
        "last_frame",
    ]


def test_flf2v_injects_text_identity_lock_and_rejects_drifted_relay(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    storyboard_dir.mkdir()
    (storyboard_dir / "S03.png").write_bytes(b"s" * 1025)
    (storyboard_dir / "S03_end.png").write_bytes(b"e" * 1025)
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{
            "id": "rin",
            "name": "凛",
            "appearance": {
                "hair": "银灰色短发",
                "face": "小巧鹅蛋脸",
                "clothing": "黑色短外套",
                "build": "纤细少女体型",
            },
        }]}),
        encoding="utf-8",
    )

    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S03",
        {"prompt": "凛转身看向镜头。", "gen_strategy": "flf2v", "_char_ids": ["rin"]},
    )

    prompt = next(item["text"] for item in content if item["type"] == "text")
    roles = [item.get("role") for item in content if item["type"] == "image_url"]
    assert "[identity-lock: text-only; no reference media]" in prompt
    assert "hair: 银灰色短发" in prompt
    assert "face: 小巧鹅蛋脸" in prompt
    assert "clothing: 黑色短外套" in prompt
    assert "body build: 纤细少女体型" in prompt
    assert roles == ["first_frame", "last_frame"]
    assert "reference_image" not in roles

    relayed = pipeline_core._apply_chain_relay(content, "drifted-tail", "S03")
    assert relayed is content
    assert relayed[1]["role"] == "first_frame"


def test_chain_relay_skips_reference_only_content():
    content = [
        {"type": "text", "text": "phantom shot"},
        {
            "type": "image_url",
            "image_url": {"url": "https://tos.test/reference.png"},
            "role": "reference_image",
        },
    ]

    relayed = pipeline_core._apply_chain_relay(content, "relay-data", "S01")

    assert relayed is content
    assert not any(item.get("role") == "first_frame" for item in relayed)


def test_phase_orchestrator_writes_full_streamed_log(monkeypatch, tmp_path):
    full_output = "x" * 2500 + "\n"
    monkeypatch.setattr(
        phase_orchestrator.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=full_output, stderr=""
        ),
    )
    result = phase_orchestrator.run_phase("phase5", {
        "input": str(tmp_path / "story.txt"),
        "duration": 45,
        "shot_duration": 12,
        "output_dir": str(tmp_path),
    })

    log_path = Path(result["log_path"])
    assert log_path.exists()
    assert full_output in log_path.read_text(encoding="utf-8")
    assert log_path.stat().st_size > 2000
    assert result["stdout"] == full_output[-2000:]


def test_phase_orchestrator_failure_prints_stdout_and_stderr_tails(
    monkeypatch, tmp_path, capsys
):
    attempted_phases = []
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "input": str(tmp_path / "story.txt"),
        "duration": 45,
        "output_dir": str(tmp_path),
    }))
    (tmp_path / "pipeline_report.json").write_text(json.dumps({"phases": {}}))
    def fail_phase(phase, config):
        attempted_phases.append(phase)
        return {
            "phase": phase,
            "exit_code": 1,
            "stdout": "A" * 500 + "stdout-cause",
            "stderr": "B" * 500 + "stderr-cause",
            "timestamp": "2026-08-10T00:00:00",
        }

    monkeypatch.setattr(phase_orchestrator, "run_phase", fail_phase)
    monkeypatch.setattr(
        sys, "argv", ["phase_orchestrator.py", "--config", str(config_path), "--resume-from", "phase4"]
    )

    with pytest.raises(SystemExit, match="1"):
        phase_orchestrator.main()

    output = capsys.readouterr().out
    assert attempted_phases == ["phase4"]
    assert "Stdout tail:" in output and "stdout-cause" in output
    assert "Stderr tail:" in output and "stderr-cause" in output


def test_pipeline_runner_prints_selected_phase_error(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_runner_cli._core,
        "run_pipeline",
        lambda **kwargs: {
            "status": "partial",
            "phases": {"phase4": {"status": "error", "error": "orchestrator timed out"}},
        },
    )
    monkeypatch.setattr(pipeline_runner_cli, "_record_run_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["pipeline_runner.py", "--input", "story.txt", "--phase", "phase4"])

    with pytest.raises(SystemExit, match="1"):
        pipeline_runner_cli.main()

    assert "Phase phase4 failed: orchestrator timed out" in capsys.readouterr().out


def test_pipeline_runner_reconnects_successful_report_checkpoints(monkeypatch, tmp_path):
    pipeline_runner_cli._record_report_checkpoints(
        {
            "phases": {
                "2": {"status": "done", "outputs": ["artifact.json"]},
                "3": {"status": "error", "error": "ignored"},
                "4": {"status": "skipped", "reason": "selected range"},
            }
        },
        tmp_path,
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["completed"] == ["phase2", "phase4"]
    assert checkpoint["results"]["phase2"]["status"] == "done"
    assert (tmp_path / "checkpoint.db").exists()
    sqlite_state = pipeline_core.load_state_from_sqlite(tmp_path, thread_id="pipeline_run")
    assert sqlite_state["completed"] == ["phase2", "phase4"]


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

    result = pipeline_core._run_phase6_fallback(tmp_path)

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

    result = pipeline_core._run_phase6_fallback(tmp_path)

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

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(attempts) == 2
