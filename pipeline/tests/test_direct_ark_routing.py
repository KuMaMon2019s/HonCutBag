# ruff: noqa: E402

import hashlib
import importlib.util
import json
import multiprocessing
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase_orchestrator

from clients import (
    ark_multimodal_client,
    local_video_client,
    seedance_client,
    tos_uploader,
)
from clients.seedream_client import SeedreamClient
from phases import pipeline_core
from phases.phase4 import phase4_orchestrator
from phases.phase4.shot_setup import normalize_shots
from phases.phase5.storyboard_qa_gate import _calibrate_l3_severity
from graph.nodes.phase6 import phase6_txt2vid_node
from phases.phase8.edit_decisions import _build_timeline, build_edit_decisions
from phases.phase8.reshoot_transaction import ReshootTransaction, durable_attempt_count
from phases.phase8.story_order_reviewer import _shot_id, storyboard_shot_ids
from runtime.bridge_execution import execute_bridge_video_task
from runtime.capacity import (
    CapacityTable,
    CapacityWaitTimeoutError,
    CrossProcessSlotTable,
    SlotTable,
)
from runtime.generation_tasks import GenerationTaskStore
from runtime.execution_errors import ProviderJobFailedError, ProviderPreparationError
from runtime.seedance_execution import (
    ProviderEndpointChangedError,
    SubmissionUncertainError,
    _provider_rejected_submission,
    execute_seedance_video_task,
)
from tools import asset_packager
from utils import shot_embedder
from utils.config import Models, SEEDANCE_MODEL
from vendor.video_tools.tools.video.remotion_caption_burn import RemotionCaptionBurn

_runner_spec = importlib.util.spec_from_file_location(
    "pipeline_runner_cli", SRC / "pipeline_runner.py"
)
pipeline_runner_cli = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(pipeline_runner_cli)


@pytest.fixture(autouse=True)
def _isolate_provider_capacity_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "provider-capacity.db"))
    # Most routing tests use byte strings instead of encoded MP4 fixtures. The
    # strict ffprobe contract has dedicated tests below; keep unrelated routing
    # tests focused on provider behavior.
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)


def test_phase_ids_are_contiguous_in_execution_order():
    expected = [f"phase{number}" for number in range(1, 10)] + ["phase9_5"]
    assert phase_orchestrator.PHASES == expected
    assert list(pipeline_runner_cli.PHASES) == expected
    assert phase_orchestrator.PHASE_NUMBERS == {
        **{f"phase{number}": str(number) for number in range(1, 10)},
        "phase9_5": "9.5",
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
    assert written["phases"] == [f"phase{number}" for number in range(1, 10)] + ["phase9_5"]
    assert written["current_phase"] == "phase2"


def test_pipeline_runner_can_select_phase9_5_independently():
    parser = pipeline_runner_cli._build_parser()
    args = parser.parse_args(["--text", "story", "--phase", "phase9_5"])

    assert pipeline_runner_cli._phase_skip_list(args, parser) == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ]


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


def test_structured_who_overrides_stale_associate_character_alias(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{
            "id": "xian_nv",
            "name": "年轻东方古装仙女",
            "aliases": ["仙女"],
        }]}),
        encoding="utf-8",
    )
    shot_meta = {
        "who": ["年轻东方古装仙女"],
        "_char_ids": ["young_eastern_fairy"],
        "associate_assets": ["char:young_eastern_fairy"],
    }

    assert asset_packager._detect_shot_characters(tmp_path, shot_meta) == ["xian_nv"]


def test_explicit_empty_who_blocks_stale_character_assets(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "lin", "name": "凛"}]}),
        encoding="utf-8",
    )

    assert asset_packager._detect_shot_characters(
        tmp_path,
        {"who": [], "_char_ids": ["lin"], "associate_assets": ["char:lin"]},
    ) == []


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
    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_BUCKET", "honcut-fixtures")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    monkeypatch.setattr(
        "clients.tos_uploader.upload_image",
        lambda image_data, content_type: tos_uploader.get_signed_url(
            f"fixture/{len(image_data)}.png"
        ),
    )


def _signed_tos_url(monkeypatch, object_key):
    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_BUCKET", "honcut-fixtures")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    return tos_uploader.get_signed_url(object_key)


def _write_cinematic_frame(path: Path, marker: bytes = b"s") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(marker * 1025)
    path.with_suffix(".json").write_text(
        json.dumps({
            "kind": "honcut.cinematic-first-frame.v1",
            "status": "done",
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "previs_reference_images": [],
        }),
        encoding="utf-8",
    )


def test_phantom_with_cinematic_frame_uses_strict_first_frame_only(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    _write_cinematic_frame(storyboard_dir / "S01.png")
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
    assert roles == ["text", "first_frame"]
    assert "last_frame" not in roles
    assert "reference_image" not in roles


def test_flf2v_content_keeps_first_and_last_frames(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    _write_cinematic_frame(storyboard_dir / "S01.png")
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


def test_video_image_packaging_fails_closed_when_any_tos_upload_fails(
    tmp_path,
    monkeypatch,
):
    storyboard_dir = tmp_path / "storyboard_images"
    _write_cinematic_frame(storyboard_dir / "S01.png")
    (storyboard_dir / "S01_end.png").write_bytes(b"e" * 1025)
    upload_count = 0
    first_url = _signed_tos_url(monkeypatch, "fixture/first.png")

    def fail_second_upload(_image_data, _content_type):
        nonlocal upload_count
        upload_count += 1
        return first_url if upload_count == 1 else None

    monkeypatch.setattr(tos_uploader, "upload_image", fail_second_upload)

    with pytest.raises(RuntimeError, match="TOS upload failed.*last_frame"):
        asset_packager.build_content_for_shot(
            tmp_path,
            "S01",
            {"prompt": "frame shot", "gen_strategy": "flf2v"},
        )

    assert upload_count == 2


def test_flf2v_injects_text_identity_lock_and_rejects_drifted_relay(tmp_path, monkeypatch):
    _stub_tos_upload(monkeypatch)
    storyboard_dir = tmp_path / "storyboard_images"
    _write_cinematic_frame(storyboard_dir / "S03.png")
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


def test_chain_relay_uploads_the_image_to_tos(monkeypatch):
    content = [{"type": "text", "text": "continue the shot"}]
    observed = []
    relay_url = _signed_tos_url(monkeypatch, "fixture/relay.jpg")
    monkeypatch.setattr(
        tos_uploader,
        "base64_to_signed_url",
        lambda value: observed.append(value) or relay_url,
    )

    relayed = pipeline_core._apply_chain_relay(content, "relay-data", "S01")

    assert observed == ["relay-data"]
    assert relayed[1]["image_url"]["url"] == relay_url
    assert not relayed[1]["image_url"]["url"].startswith("data:")


def test_chain_relay_fails_closed_when_tos_upload_fails(monkeypatch):
    monkeypatch.setattr(tos_uploader, "base64_to_signed_url", lambda _value: None)

    with pytest.raises(RuntimeError, match="TOS upload failed.*chain relay"):
        pipeline_core._apply_chain_relay(
            [{"type": "text", "text": "continue the shot"}],
            "relay-data",
            "S01",
        )


def test_phase_orchestrator_writes_full_streamed_log(monkeypatch, tmp_path):
    full_output = "x" * 2500 + "\n"

    def fake_stream(cmd, log_path, cwd, env, monitor=None):
        # Mirror the real contract: the streamed log carries the full
        # output while only a tail is returned in the result payload.
        Path(log_path).write_text(full_output, encoding="utf-8")
        return {"returncode": 0, "stdout": full_output, "stderr": ""}

    monkeypatch.setattr(phase_orchestrator, "_stream_subprocess", fake_stream)
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


def test_pipeline_runner_rejects_partial_top_level_status(monkeypatch):
    monkeypatch.setattr(
        pipeline_runner_cli._core,
        "run_pipeline",
        lambda **kwargs: {"status": "partial", "phases": {}},
    )
    monkeypatch.setattr(
        pipeline_runner_cli, "_record_run_memory", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(sys, "argv", ["pipeline_runner.py", "--text", "story"])

    with pytest.raises(SystemExit, match="1"):
        pipeline_runner_cli.main()


@pytest.mark.parametrize(
    ("enable_reshoot", "expected_flag"),
    [(True, "--enable-reshoot"), (False, "--disable-reshoot")],
)
def test_orchestrator_forwards_reshoot_and_transition_configuration(
    tmp_path, monkeypatch, enable_reshoot, expected_flag
):
    captured = {}

    def fake_stream(command, *args, **kwargs):
        captured["command"] = command
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(phase_orchestrator, "_stream_subprocess", fake_stream)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = {
        "input": str(tmp_path / "story.txt"),
        "duration": 30,
        "shot_duration": 5,
        "output_dir": str(output_dir),
        "media_profile": "720p",
        "transition_duration": 0.75,
        "enable_reshoot": enable_reshoot,
        "no_real_person": True,
    }

    phase_orchestrator.run_phase("phase8", config)

    command = captured["command"]
    assert command[command.index("--transition-duration") + 1] == "0.75"
    assert expected_flag in command
    assert "--no-real-person" in command
    opposite = "--disable-reshoot" if enable_reshoot else "--enable-reshoot"
    assert opposite not in command


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
    assert checkpoint["completed"] == ["phase2"]
    assert checkpoint["results"]["phase2"]["status"] == "done"
    assert (tmp_path / "checkpoint.db").exists()
    sqlite_state = pipeline_core.load_state_from_sqlite(tmp_path, thread_id="pipeline_run")
    assert sqlite_state["completed"] == ["phase2"]


def _forbid_phase4_subprocess(*_args, **_kwargs):
    pytest.fail("Phase 4 invoked a subprocess")


def test_phase4_native_setup_does_not_invoke_subprocess(monkeypatch, tmp_path):
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"id": 1, "visual": "first shot"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_core.subprocess, "run", _forbid_phase4_subprocess)

    result = pipeline_core.run_phase4(tmp_path, dry_run=True)

    assert result["status"] == "done"
    assert (tmp_path / "shots" / "S01" / "SHOT_META.json").is_file()
    review = result["constraint_review"]
    assert review["mode"] == "deterministic_code"
    assert review["human_review_required"] is False
    assert review["model_review_used"] is False
    assert {item["id"] for item in review["checks"]} == {
        "storyboard_artifacts_complete",
        "scene_and_continuity_contracts_written",
        "cinematic_first_frames_previs_isolated",
        "native_shot_metadata_only",
        "shot_meta_ids_exact",
    }


def test_phase4_native_setup_completes_partial_expected_directories(
    monkeypatch, tmp_path
):
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(
            {
                "shots": [
                    {"id": 1, "visual": "first shot"},
                    {"id": 2, "visual": "second shot"},
                ]
            }
        ),
        encoding="utf-8",
    )
    partial = tmp_path / "shots" / "S01"
    partial.mkdir(parents=True)
    (partial / "SHOT_META.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline_core.subprocess, "run", _forbid_phase4_subprocess)

    result = pipeline_core.run_phase4(tmp_path, dry_run=True)

    assert result["status"] == "done"
    for shot_id in ("S01", "S02"):
        metadata = json.loads(
            (tmp_path / "shots" / shot_id / "SHOT_META.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["shot_id"] == shot_id
        assert metadata["status"] == "pending"


def test_phase4_native_setup_rejects_unexpected_stale_shot_directory(
    monkeypatch, tmp_path
):
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"id": 1, "visual": "first shot"}]}),
        encoding="utf-8",
    )
    stale = tmp_path / "shots" / "S99"
    stale.mkdir(parents=True)
    (stale / "SHOT_META.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline_core.subprocess, "run", _forbid_phase4_subprocess)

    result = pipeline_core.run_phase4(tmp_path, dry_run=True)

    assert result["status"] == "error"
    assert result["error"] == "Phase 4 shot directory invariant failed"
    assert result["unexpected_shot_ids"] == ["S99"]


def test_phase4_materializes_canonical_storyboard_without_legacy_adapter(
    monkeypatch, tmp_path
):
    storyboard_path = tmp_path / "STORYBOARD.json"
    storyboard_path.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "S01",
                        "shot_intent": "approach the platform",
                        "visual": "a blue-lit tunnel",
                    },
                    {
                        "id": "2",
                        "name": "authored name",
                        "prompt": "authored prompt",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    canonical_before = storyboard_path.read_bytes()
    monkeypatch.setattr(pipeline_core.subprocess, "run", _forbid_phase4_subprocess)

    result = pipeline_core.run_phase4(tmp_path, dry_run=True)

    assert result["status"] == "done"
    assert "phase4_legacy_storyboard.json" not in result["outputs"]
    assert not (tmp_path / "phase4_legacy_storyboard.json").exists()
    first = json.loads(
        (tmp_path / "shots" / "S01" / "SHOT_META.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (tmp_path / "shots" / "S02" / "SHOT_META.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["shot_id"] == "S01"
    assert first["name"] == "approach the platform"
    assert first["prompt"] == "a blue-lit tunnel"
    assert second["shot_id"] == "S02"
    assert second["name"] == "authored name"
    assert second["prompt"] == "authored prompt"
    assert storyboard_path.read_bytes() == canonical_before


@pytest.mark.parametrize(
    ("shot", "message"),
    [
        ({"id": "shot_ref", "visual": "tunnel"}, "non-numeric shot ID"),
        ({"id": 1, "name": "opening"}, "no prompt-compatible"),
    ],
)
def test_phase4_native_normalization_rejects_invalid_shots(shot, message):
    with pytest.raises(ValueError, match=message):
        normalize_shots({"shots": [shot]}, storyboard_dir=None)



def test_submit_content_sends_top_level_agent_plan_payload(monkeypatch):
    frame_url = _signed_tos_url(monkeypatch, "fixture/frame.jpg")
    content = [
        {"type": "text", "text": "move slowly"},
        {
            "type": "image_url",
            "image_url": {"url": frame_url},
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
        resolution="480p",
        seed=42,
        generate_audio=True,
    )

    assert task_id == "task-direct-1"
    assert posted["url"] == seedance_client.SUBMIT_ENDPOINT
    assert posted["json"] == {
        "model": "doubao-seedance-2.0-mini",
        "content": content,
        "generate_audio": True,
        "ratio": "16:9",
        "duration": 12,
        "resolution": "480p",
        "watermark": False,
        "seed": 42,
    }
    assert posted["json"]["content"] is content
    assert "parameters" not in posted["json"]


def test_seedance_direct_first_frame_is_uploaded_to_tos_before_submit(monkeypatch):
    posted = {}
    first_frame_url = _signed_tos_url(monkeypatch, "fixture/first-frame.png")

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"id": "task-tos-first-frame"}

    monkeypatch.setattr(
        tos_uploader,
        "base64_to_signed_url",
        lambda value: first_frame_url if value == "first-frame-b64" else None,
    )
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda url, **kwargs: posted.update(url=url, **kwargs) or Response(),
    )

    task_id = seedance_client._submit_direct(
        "continue the visible action",
        "test-key",
        model="doubao-seedance-2.0-fast",
        first_frame_base64="first-frame-b64",
    )

    assert task_id == "task-tos-first-frame"
    image_item = next(
        item for item in posted["json"]["content"] if item["type"] == "image_url"
    )
    assert image_item["image_url"]["url"] == first_frame_url
    assert image_item["role"] == "first_frame"


@pytest.mark.parametrize(
    ("kwargs", "uploader_name", "message"),
    [
        (
            {"first_frame_base64": "first-frame"},
            "base64_to_signed_url",
            "first-frame image",
        ),
        (
            {"reference_image_base64": "reference-image"},
            "base64_to_signed_url",
            "reference image",
        ),
        (
            {"reference_video_base64": "reference-video"},
            "base64_video_to_signed_url",
            "reference video",
        ),
    ],
)
def test_seedance_direct_media_upload_failure_never_reaches_provider(
    monkeypatch,
    kwargs,
    uploader_name,
    message,
):
    monkeypatch.setattr(tos_uploader, uploader_name, lambda _value: None)
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "provider submission must not run after TOS failure"
        ),
    )

    with pytest.raises(RuntimeError, match=f"TOS upload failed.*{message}"):
        seedance_client._submit_direct(
            "test",
            "test-key",
            model="doubao-seedance-2.0-fast",
            **kwargs,
        )


def test_seedance_submit_content_rejects_inline_media_before_provider(monkeypatch):
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail(
            "inline media must be rejected before provider submission"
        ),
    )

    with pytest.raises(RuntimeError, match="configured TOS origin"):
        seedance_client.submit_content(
            [
                {"type": "text", "text": "continue"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                    "role": "first_frame",
                },
            ],
            api_key="test-key",
            model="doubao-seedance-2.0-fast",
            duration=4,
        )


def test_submit_content_rejects_non_boolean_generate_audio_before_submission(monkeypatch):
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("invalid payload must not be submitted"),
    )

    with pytest.raises(ValueError, match="generate_audio must be a boolean"):
        seedance_client.submit_content(
            [{"type": "text", "text": "a cinematic shot"}],
            api_key="test-key",
            model="doubao-seedance-2.0-fast",
            duration=5,
            resolution="480p",
            generate_audio="true",
        )


def test_agent_plan_default_model_id_is_the_exact_seedance_fast_id():
    assert Models.ARK_VIDEO == "doubao-seedance-2.0-fast"
    assert SEEDANCE_MODEL == "doubao-seedance-2.0-fast"


def test_seedance_fast_rejects_unsupported_media_profile_before_submission():
    with pytest.raises(ValueError, match="expected 480p or 720p"):
        seedance_client.resolution_for_media_profile(
            "1080p",
            "doubao-seedance-2.0-fast",
        )


def test_phase6_graph_node_forwards_media_profile_to_phase_owner(tmp_path):
    calls = []
    state = {
        "output_dir": str(tmp_path),
        "dry_run": True,
        "chain_mode": False,
        "media_profile": "480p",
        "storyboard": {"shots": []},
        "phase_results": {},
        "completed_phases": [],
        "retry_count": 0,
        "skip_phase": [],
    }

    phase6_txt2vid_node(
        state,
        runner=lambda **kwargs: calls.append(kwargs) or {"status": "done"},
    )

    assert calls[0]["media_profile"] == "480p"


def test_submit_content_rejects_frame_control_mixed_with_reference_media(monkeypatch):
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("invalid content must not reach Seedance"),
    )

    with pytest.raises(ValueError, match="cannot mix first/last frame control"):
        seedance_client.submit_content(
            [
                {"type": "text", "text": "move slowly"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/frame.jpg"},
                    "role": "first_frame",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/board.jpg"},
                    "role": "reference_image",
                },
            ],
            api_key="test-key",
            model="doubao-seedance-2.0-mini",
            duration=8,
        )


def test_seedance_download_is_atomic_on_interrupted_stream(tmp_path, monkeypatch):
    destination = tmp_path / "output.mp4"
    destination.write_bytes(b"previous-complete-video")

    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield b"partial"
            raise ConnectionError("stream interrupted")

    monkeypatch.setattr(
        seedance_client.requests,
        "get",
        lambda *args, **kwargs: InterruptedResponse(),
    )

    with pytest.raises(ConnectionError, match="interrupted"):
        seedance_client.download("https://video.test/output.mp4", str(destination))

    assert destination.read_bytes() == b"previous-complete-video"
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob(".*.part")) == []


def _write_shot(output_dir):
    shot_dir = output_dir / "shots" / "S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "quiet landscape", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    return shot_dir


def _mock_common_direct(monkeypatch, shot_dir):
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "direct")
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
    assert direct_calls[0][1]["model"] == "doubao-seedance-2.0-fast"
    assert direct_calls[0][1]["resolution"] == "480p"
    assert (shot_dir / "output.mp4").exists()


def test_phase6_does_not_reuse_large_unledgered_output(tmp_path, monkeypatch):
    shot_dir = _write_shot(tmp_path)
    (shot_dir / "output.mp4").write_bytes(b"partial" * 4_000)
    _mock_common_direct(monkeypatch, shot_dir)
    submissions = []
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda content, **kwargs: submissions.append(kwargs) or "new-provider-job",
    )

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(submissions) == 1
    assert (shot_dir / "output.mp4").read_bytes() == b"v" * 11000


def test_phase6_current_failure_is_not_hidden_by_an_old_clip(
    tmp_path, monkeypatch
):
    failed_shot = _write_shot(tmp_path)
    failed_meta = failed_shot / "SHOT_META.json"
    failed_meta.write_text(
        json.dumps({"prompt": "fail this shot", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    old_clip = b"old-video" * 2_000
    (failed_shot / "output.mp4").write_bytes(old_clip)
    successful_shot = tmp_path / "shots/S02"
    successful_shot.mkdir(parents=True)
    (successful_shot / "SHOT_META.json").write_text(
        json.dumps({"prompt": "good shot", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    _mock_common_direct(monkeypatch, failed_shot)

    def submit(content, **_kwargs):
        text = " ".join(str(item.get("text", "")) for item in content)
        if "fail this shot" in text:
            raise RuntimeError("Seedance API 400: InvalidParameter test rejection")
        return "good-provider-job"

    monkeypatch.setattr(seedance_client, "submit_content", submit)

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "error"
    assert result["missing_shots"] == ["S01"]
    assert result["outputs"] == ["shots/S02/output.mp4"]
    assert (failed_shot / "output.mp4").read_bytes() == old_clip


@pytest.mark.parametrize("provider", ["local", "wan", "bridge"])
def test_explicit_bridge_providers_use_local_client(tmp_path, monkeypatch, provider):
    _write_shot(tmp_path)
    monkeypatch.setenv("VIDEO_PROVIDER", provider)
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    bridge_calls = []

    def fake_generate(**kwargs):
        kwargs["on_submit_start"]()
        kwargs["on_submitted"]("bridge-job-1")
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


def test_seedance_provider_honors_bridge_generation_mode(tmp_path, monkeypatch):
    _write_shot(tmp_path)
    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "bridge")
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    calls = []

    def fake_generate(**kwargs):
        kwargs["on_submit_start"]()
        kwargs["on_submitted"]("bridge-seedance-1")
        Path(kwargs["output_path"]).write_bytes(b"v" * 11000)
        calls.append(kwargs)
        return {
            "output_path": kwargs["output_path"],
            "last_frame_path": None,
            "actual_model": kwargs["model"],
        }

    monkeypatch.setattr(local_video_client, "generate_video", fake_generate)
    monkeypatch.setattr(local_video_client, "generate_video_with_fallback", fake_generate)
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": "shot"}],
    )
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda *args, **kwargs: pytest.fail("Bridge mode must not call direct ARK"),
    )

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert calls[0]["model"] == "seedance"


def test_direct_seedance_receives_storyboard_aspect_ratio(tmp_path, monkeypatch):
    shot_dir = _write_shot(tmp_path)
    meta_path = shot_dir / "SHOT_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["aspect_ratio"] = "9:16"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    _mock_common_direct(monkeypatch, shot_dir)
    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    calls = []
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda content, **kwargs: calls.append(kwargs) or "task-portrait-1",
    )

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert calls[0]["ratio"] == "9:16"
    persisted = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted["ratio"] == "9:16"
    assert (persisted["width"], persisted["height"]) == (720, 1280)


def test_local_client_resume_skips_submission(tmp_path, monkeypatch):
    output_path = tmp_path / "shots" / "S01" / "output.mp4"
    output_path.parent.mkdir(parents=True)
    submissions = []
    polled = []
    submission_starts = []
    callbacks = []

    monkeypatch.setattr(
        local_video_client,
        "submit",
        lambda **kwargs: submissions.append(kwargs) or "unexpected-job",
    )
    monkeypatch.setattr(
        local_video_client,
        "poll",
        lambda task_id: polled.append(task_id) or {"status": "completed"},
    )

    def download(task_id, destination, **kwargs):
        Path(destination).write_bytes(b"v" * 11000)
        return destination

    monkeypatch.setattr(local_video_client, "download", download)

    generated = local_video_client.generate_video(
        prompt="quiet landscape",
        output_path=str(output_path),
        duration=4,
        model="wan22",
        resume_task_id="bridge-job-1",
        on_submit_start=lambda: submission_starts.append("started"),
        on_submitted=callbacks.append,
    )

    assert generated == str(output_path)
    assert submissions == []
    assert submission_starts == []
    assert callbacks == []
    assert polled == ["bridge-job-1"]


def test_local_client_reports_job_id_before_polling(tmp_path, monkeypatch):
    output_path = tmp_path / "shots" / "S01" / "output.mp4"
    output_path.parent.mkdir(parents=True)
    events = []

    def submit(**kwargs):
        events.append("submitted")
        return "bridge-job-1"

    def remember(task_id):
        events.append(f"persisted:{task_id}")

    def poll(task_id):
        assert events == ["starting", "submitted", "persisted:bridge-job-1"]
        raise TimeoutError("stop after persistence proof")

    monkeypatch.setattr(local_video_client, "submit", submit)
    monkeypatch.setattr(local_video_client, "poll", poll)

    with pytest.raises(TimeoutError, match="persistence proof"):
        local_video_client.generate_video(
            prompt="quiet landscape",
            output_path=str(output_path),
            duration=4,
            model="wan22",
            on_submit_start=lambda: events.append("starting"),
            on_submitted=remember,
        )

    assert events == ["starting", "submitted", "persisted:bridge-job-1"]


def test_bridge_runtime_resumes_persisted_job_without_resubmitting(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "shots" / "S01" / "output.mp4"
    output_path.parent.mkdir(parents=True)
    resume_ids = []

    def interrupted_generate(*, resume_task_id, on_submit_start, on_submitted):
        resume_ids.append(resume_task_id)
        on_submit_start()
        on_submitted("bridge-job-1")
        raise TimeoutError("Bridge polling interrupted")

    with pytest.raises(TimeoutError, match="polling interrupted"):
        execute_bridge_video_task(
            store,
            run_id="run-1",
            resource_id="S01",
            payload={"shot_id": "S01", "model": "wan22"},
            provider_endpoint="http://bridge.test",
            output_path=output_path,
            generate=interrupted_generate,
        )

    def resumed_generate(*, resume_task_id, on_submit_start, on_submitted):
        resume_ids.append(resume_task_id)
        Path(output_path).write_bytes(b"v" * 11000)
        return {
            "output_path": str(output_path),
            "last_frame_path": None,
            "actual_model": "wan22",
        }

    execution = execute_bridge_video_task(
        store,
        run_id="run-1",
        resource_id="S01",
        payload={"shot_id": "S01", "model": "wan22"},
        provider_endpoint="http://bridge.test",
        output_path=output_path,
        generate=resumed_generate,
    )

    assert execution.resumed is True
    assert execution.provider_job_id == "bridge-job-1"
    assert resume_ids == [None, "bridge-job-1"]
    assert store.get(execution.task_id).status == "succeeded"


def test_bridge_succeeded_ledger_recovers_missing_output_by_job_id(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "shots" / "S01" / "chunks" / "S01_C01.mp4"
    output_path.parent.mkdir(parents=True)
    payload = {"shot_id": "S01", "input_fingerprint": "same-input"}

    def initial_generate(*, resume_task_id, on_submit_start, on_submitted):
        assert resume_task_id is None
        on_submit_start()
        on_submitted("bridge-job-1")
        output_path.write_bytes(b"video")
        return str(output_path)

    first = execute_bridge_video_task(
        store,
        run_id="run-1",
        resource_id="S01_C01",
        payload=payload,
        provider_endpoint="http://bridge.test",
        output_path=output_path,
        generate=initial_generate,
    )
    output_path.unlink()
    resume_ids = []

    def recover_generate(*, resume_task_id, on_submit_start, on_submitted):
        resume_ids.append(resume_task_id)
        output_path.write_bytes(b"video")
        return str(output_path)

    recovered = execute_bridge_video_task(
        store,
        run_id="run-1",
        resource_id="S01_C01",
        payload=payload,
        provider_endpoint="http://bridge.test",
        output_path=output_path,
        generate=recover_generate,
    )

    assert recovered.task_id == first.task_id
    assert recovered.resumed is True
    assert resume_ids == ["bridge-job-1"]


def test_bridge_fallback_submit_timeout_becomes_uncertain(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "output.mp4"

    def first_attempt(*, resume_task_id, on_submit_start, on_submitted):
        on_submit_start()
        on_submitted("bridge-seedance-job")
        raise TimeoutError("Seedance polling timed out")

    arguments = {
        "run_id": "run-1",
        "resource_id": "S01",
        "payload": {"shot_id": "S01", "model": "seedance"},
        "provider_endpoint": "http://bridge.test",
        "output_path": output_path,
    }
    with pytest.raises(TimeoutError, match="polling timed out"):
        execute_bridge_video_task(store, generate=first_attempt, **arguments)

    def uncertain_fallback(*, resume_task_id, on_submit_start, on_submitted):
        assert resume_task_id == "bridge-seedance-job"
        on_submit_start()
        raise TimeoutError("Wan fallback submit timed out")

    with pytest.raises(TimeoutError, match="fallback submit timed out"):
        execute_bridge_video_task(store, generate=uncertain_fallback, **arguments)

    active = store.find_active(
        run_id="run-1",
        task_type="video.generate",
        resource_id="S01",
        provider_id="bridge",
    )
    assert active is not None
    assert active.status == "submission_uncertain"
    assert active.provider_job_id == "bridge-seedance-job"


def test_generation_tasks_are_deduped_per_provider(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    common = {
        "run_id": "run-1",
        "task_type": "video.generate",
        "media_type": "video",
        "resource_id": "S01",
        "payload": {"shot_id": "S01"},
    }

    bridge = store.enqueue(**common, provider_id="bridge")
    seedance = store.enqueue(**common, provider_id="seedance")

    assert bridge.deduped is False
    assert seedance.deduped is False
    assert bridge.task.task_id != seedance.task.task_id


def test_reshoot_transaction_restores_clip_and_metadata_after_failure(tmp_path):
    shot = tmp_path / "shots" / "S01"
    shot.mkdir(parents=True)
    (shot / "output.mp4").write_bytes(b"old-video")
    (shot / "SHOT_META.json").write_text(
        '{"gen_strategy":"flf2v"}', encoding="utf-8"
    )

    transaction = ReshootTransaction.begin(
        tmp_path, kind="visual_quality", shot_ids=["S01"]
    )
    transaction.remove_sources()
    (shot / "output.mp4").write_bytes(b"partial-new-video")
    (shot / "SHOT_META.json").write_text(
        '{"gen_strategy":"phantom"}', encoding="utf-8"
    )
    transaction.rollback("provider quota exceeded")

    assert (shot / "output.mp4").read_bytes() == b"old-video"
    assert json.loads((shot / "SHOT_META.json").read_text())["gen_strategy"] == "flf2v"
    assert durable_attempt_count(tmp_path) == 1


def test_edit_timeline_accounts_for_trim_speed_and_overlap(tmp_path, monkeypatch):
    shots = tmp_path / "shots"
    for shot_id in ("S01", "S02"):
        directory = shots / shot_id
        directory.mkdir(parents=True)
        (directory / "output.mp4").write_bytes(b"fixture")

    monkeypatch.setattr(
        "phases.phase8.edit_decisions.probe_video",
        lambda _path: {
            "duration": 10.0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "has_audio": True,
            "has_video": True,
        },
    )
    monkeypatch.setattr(
        "phases.phase8.edit_decisions.detect_black_frames",
        lambda _path: {"trim_start": 1.0, "trim_end": 1.0},
    )
    decisions = build_edit_decisions(
        str(shots),
        transition_decisions=[{"decision": "dissolve"}],
        transition_duration=0.5,
        target_duration=None,
    )
    decisions["cuts"][0]["speed"] = 0.8

    timeline = _build_timeline(decisions["cuts"], decisions["transitions"])

    assert timeline[0]["output_duration_s"] == pytest.approx(10.0)
    assert timeline[1]["output_start_s"] == pytest.approx(9.5)


def test_seedance_http_400_is_definite_rejection_not_uncertain_submission():
    assert _provider_rejected_submission(
        RuntimeError("Seedance API 400: InvalidParameter content is not valid")
    ) is True
    assert _provider_rejected_submission(
        TimeoutError("connection closed before submission response")
    ) is False
    assert _provider_rejected_submission(
        ProviderPreparationError("tail anchor extraction failed")
    ) is True


def test_task_store_releases_only_uncertain_task_without_provider_job(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    task = store.enqueue(
        run_id="run-1",
        task_type="video.generate",
        media_type="video",
        resource_id="S14",
        payload={"shot_id": "S14"},
    ).task
    store.claim(task.task_id)
    store.mark_submission_uncertain(task.task_id, "legacy misclassification")

    resolved = store.resolve_unsubmitted_uncertain_as_failed(
        task.task_id, "confirmed HTTP 400 rejection"
    )

    assert resolved.status == "failed"
    assert resolved.provider_job_id is None


def test_phase6_bridge_resume_reuses_persisted_job(tmp_path, monkeypatch):
    _write_shot(tmp_path)
    monkeypatch.setenv("VIDEO_PROVIDER", "local")
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    monkeypatch.setattr(local_video_client, "get_api_url", lambda: "http://bridge.test")
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": "shot"}],
    )
    resume_ids = []

    def generate(**kwargs):
        resume_ids.append(kwargs["resume_task_id"])
        if kwargs["resume_task_id"] is None:
            kwargs["on_submit_start"]()
            kwargs["on_submitted"]("bridge-job-1")
            raise TimeoutError("worker stopped while Bridge was polling")
        Path(kwargs["output_path"]).write_bytes(b"v" * 11000)
        return {
            "output_path": kwargs["output_path"],
            "last_frame_path": None,
            "actual_model": "wan22",
        }

    monkeypatch.setattr(local_video_client, "generate_video", generate)

    first = pipeline_core._run_phase6_fallback(tmp_path)
    second = pipeline_core._run_phase6_fallback(tmp_path)

    assert first["status"] == "error"
    assert second["status"] == "done"
    assert resume_ids == [None, "bridge-job-1"]
    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        persisted = connection.execute(
            "SELECT status, provider_id, provider_job_id FROM generation_tasks"
        ).fetchone()
    assert persisted == ("succeeded", "bridge", "bridge-job-1")


def test_direct_ark_quota_error_uses_existing_retry_loop(tmp_path, monkeypatch):
    shot_dir = _write_shot(tmp_path)
    _mock_common_direct(monkeypatch, shot_dir)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    attempts = []

    def flaky_submit(content, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError(
                "Seedance API 429 QuotaExceeded; request id contains 4017c4cb"
            )
        return "task-after-retry"

    monkeypatch.setattr(seedance_client, "submit_content", flaky_submit)
    monkeypatch.setattr(pipeline_core.time, "sleep", lambda seconds: None)

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(attempts) == 2


def test_direct_ark_quota_retries_do_not_consume_privacy_retry_budget(
    tmp_path, monkeypatch
):
    shot_dir = _write_shot(tmp_path)
    _mock_common_direct(monkeypatch, shot_dir)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    attempts = []
    content = [
        {"type": "text", "text": "shot"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/rejected.png?sig=1"},
            "role": "reference_image",
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/safe.png?sig=1"},
            "role": "reference_image",
        },
    ]
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **_kwargs: [dict(item) for item in content],
    )

    def mixed_failures(submitted_content, **_kwargs):
        attempts.append(submitted_content)
        if len(attempts) <= 3:
            raise RuntimeError("429 QuotaExceeded")
        if len(attempts) == 4:
            raise RuntimeError("PrivacyInformation content[1]")
        return "task-after-independent-retries"

    monkeypatch.setattr(seedance_client, "submit_content", mixed_failures)
    monkeypatch.setattr(pipeline_core.time, "sleep", lambda _seconds: None)

    result = pipeline_core._run_phase6_fallback(tmp_path)

    assert result["status"] == "done"
    assert len(attempts) == 5
    assert [item.get("image_url", {}).get("url") for item in attempts[-1]] == [
        None,
        "https://example.test/safe.png?sig=1",
    ]


def _enqueue_runtime_video(store, resource_id="S01"):
    return store.enqueue(
        run_id="run-1",
        task_type="video.generate",
        media_type="video",
        resource_id=resource_id,
        payload={"shot_id": resource_id},
        provider_id="seedance",
    )


def test_generation_runtime_active_dedupe_and_atomic_claim(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")

    first = _enqueue_runtime_video(store)
    duplicate = _enqueue_runtime_video(store)

    assert duplicate.deduped is True
    assert duplicate.task.task_id == first.task.task_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: store.claim(first.task.task_id), range(2)))

    winners = [task for task in claims if task is not None]
    assert len(winners) == 1
    assert winners[0].status == "running"
    assert winners[0].attempt_count == 1


def test_seedance_poll_failure_resumes_without_resubmitting(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "shots" / "S01" / "output.mp4"
    output_path.parent.mkdir(parents=True)
    submissions = []

    def submit():
        submissions.append("submitted")
        return "provider-job-1"

    with pytest.raises(TimeoutError, match="poll interrupted"):
        execute_seedance_video_task(
            store,
            run_id="run-1",
            resource_id="S01",
            payload={"shot_id": "S01"},
            provider_endpoint="https://seedance.test",
            output_path=output_path,
            submit=submit,
            poll=lambda provider_job_id: (_ for _ in ()).throw(
                TimeoutError("poll interrupted")
            ),
            download=lambda url, path: path,
        )

    active = _enqueue_runtime_video(store).task
    assert active.status == "running"
    assert active.provider_job_id == "provider-job-1"

    def download(url, path):
        Path(path).write_bytes(b"video")
        return path

    execution = execute_seedance_video_task(
        store,
        run_id="run-1",
        resource_id="S01",
        payload={"shot_id": "S01"},
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: pytest.fail("resume must not submit a second paid task"),
        poll=lambda provider_job_id: "https://video.test/output.mp4",
        download=download,
    )

    assert execution.resumed is True
    assert execution.provider_job_id == "provider-job-1"
    assert submissions == ["submitted"]
    assert store.get(execution.task_id).status == "succeeded"


def test_seedance_generated_output_download_does_not_upload_to_tos(
    tmp_path,
    monkeypatch,
):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "shots" / "S01" / "output.mp4"
    downloaded = []
    monkeypatch.setattr(
        tos_uploader,
        "upload_media_file",
        lambda *_args, **_kwargs: pytest.fail(
            "a generated output is not TOS input material until it is reused"
        ),
    )

    def download(url, path):
        downloaded.append((url, path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"video")
        return path

    execution = execute_seedance_video_task(
        store,
        run_id="run-output-download",
        resource_id="S01",
        payload={"shot_id": "S01"},
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: "provider-job-output",
        poll=lambda _provider_job_id: "https://provider.test/output.mp4",
        download=download,
    )

    assert downloaded == [
        ("https://provider.test/output.mp4", str(output_path)),
    ]
    assert output_path.read_bytes() == b"video"
    assert execution.output_path == str(output_path)


@pytest.mark.parametrize("status_code", [403, 429, 500])
def test_seedance_poll_http_error_never_resubmits_paid_job(tmp_path, status_code):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "output.mp4"
    submissions = []
    arguments = {
        "run_id": "run-1",
        "resource_id": "S01",
        "payload": {"shot_id": "S01", "input_fingerprint": "same"},
        "provider_endpoint": "https://seedance.test",
        "output_path": output_path,
        "submit": lambda: submissions.append("submit") or "provider-job-1",
        "download": lambda _url, path: Path(path).write_bytes(b"video") or path,
    }

    with pytest.raises(RuntimeError, match=str(status_code)):
        execute_seedance_video_task(
            store,
            poll=lambda _job_id: (_ for _ in ()).throw(
                RuntimeError(f"Seedance get task API {status_code}: transient")
            ),
            **arguments,
        )

    active = store.find_active(
        run_id="run-1",
        task_type="video.generate",
        resource_id="S01",
        provider_id="seedance",
    )
    assert active is not None
    assert active.status == "running"
    assert active.provider_job_id == "provider-job-1"

    recovered = execute_seedance_video_task(
        store,
        poll=lambda _job_id: "https://video.test/output.mp4",
        **arguments,
    )
    assert recovered.resumed is True
    assert submissions == ["submit"]


def test_generation_store_refuses_nonterminal_failure_after_provider_job(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    task = _enqueue_runtime_video(store).task
    store.claim(task.task_id)
    store.persist_provider_job(
        task.task_id,
        provider_job_id="provider-job-1",
        provider_endpoint="https://seedance.test",
    )

    with pytest.raises(RuntimeError, match="refusing to fail"):
        store.mark_failed(task.task_id, "temporary network failure")

    assert store.get(task.task_id).status == "running"


def test_seedance_invalid_succeeded_cache_redownloads_without_resubmit(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "output.mp4"
    payload = {"input_fingerprint": "same"}
    first = execute_seedance_video_task(
        store,
        run_id="run-1",
        resource_id="S01",
        payload=payload,
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: "provider-job-1",
        poll=lambda _job_id: "https://video.test/output.mp4",
        download=lambda _url, path: Path(path).write_bytes(b"valid") or path,
        validate_output=lambda path: path.read_bytes() == b"valid",
    )
    output_path.write_bytes(b"x" * 20_000)
    downloads = []

    recovered = execute_seedance_video_task(
        store,
        run_id="run-1",
        resource_id="S01",
        payload=payload,
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: pytest.fail("cache recovery must not resubmit"),
        poll=lambda _job_id: pytest.fail("succeeded job must not be repolled"),
        download=lambda url, path: downloads.append(url)
        or Path(path).write_bytes(b"valid")
        or path,
        validate_output=lambda path: path.read_bytes() == b"valid",
    )

    assert recovered.task_id == first.task_id
    assert downloads == ["https://video.test/output.mp4"]


def test_seedance_provider_rejection_during_poll_is_terminal(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output = tmp_path / "clip.mp4"

    with pytest.raises(RuntimeError, match="InvalidParameter"):
        execute_seedance_video_task(
            store,
            run_id="run-1",
            resource_id="S01_C01",
            payload={"duration": 5},
            provider_endpoint="https://ark.example.test/api/v3",
            output_path=output,
            submit=lambda: "provider-job-1",
            poll=lambda _job_id: (_ for _ in ()).throw(
                ProviderJobFailedError("Task failed: InvalidParameter")
            ),
            download=lambda _url, _path: pytest.fail("failed task must not download"),
        )

    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        status, provider_job_id = connection.execute(
            "SELECT status, provider_job_id FROM generation_tasks"
        ).fetchone()
    assert status == "failed"
    assert provider_job_id == "provider-job-1"


def test_seedance_succeeded_ledger_recovers_after_lineage_crash_without_resubmitting(
    tmp_path,
):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    output_path = tmp_path / "shots" / "S01" / "chunks" / "S01_C01.mp4"
    output_path.parent.mkdir(parents=True)
    payload = {"shot_id": "S01", "input_fingerprint": "same-input"}
    submissions = []

    first = execute_seedance_video_task(
        store,
        run_id="run-1",
        resource_id="S01_C01",
        payload=payload,
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: submissions.append("submit") or "provider-job-1",
        poll=lambda provider_job_id: "https://video.test/output.mp4",
        download=lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    output_path.unlink()
    downloads = []

    recovered = execute_seedance_video_task(
        store,
        run_id="run-1",
        resource_id="S01_C01",
        payload=payload,
        provider_endpoint="https://seedance.test",
        output_path=output_path,
        submit=lambda: pytest.fail("successful task must not be resubmitted"),
        poll=lambda provider_job_id: pytest.fail("successful task must not be repolled"),
        download=lambda url, path: downloads.append(url)
        or Path(path).write_bytes(b"video")
        or path,
    )

    assert recovered.task_id == first.task_id
    assert recovered.resumed is True
    assert submissions == ["submit"]
    assert downloads == ["https://video.test/output.mp4"]


def test_generation_runtime_rejects_changed_payload_for_an_active_resource(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    _enqueue_runtime_video(store)

    with pytest.raises(RuntimeError, match="payload changed"):
        store.enqueue(
            run_id="run-1",
            task_type="video.generate",
            media_type="video",
            resource_id="S01",
            payload={"shot_id": "S01", "input_fingerprint": "changed"},
            provider_id="seedance",
        )


def test_uncertain_seedance_submission_blocks_automatic_resubmit(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    calls = []

    def uncertain_submit():
        calls.append("submit")
        raise TimeoutError("request timed out before a job id was returned")

    arguments = {
        "run_id": "run-1",
        "resource_id": "S01",
        "payload": {"shot_id": "S01"},
        "provider_endpoint": "https://seedance.test",
        "output_path": tmp_path / "output.mp4",
        "poll": lambda provider_job_id: "unused",
        "download": lambda url, path: path,
    }
    with pytest.raises(TimeoutError):
        execute_seedance_video_task(store, submit=uncertain_submit, **arguments)

    with pytest.raises(SubmissionUncertainError, match="refusing to resubmit"):
        execute_seedance_video_task(store, submit=uncertain_submit, **arguments)

    assert calls == ["submit"]
    assert _enqueue_runtime_video(store).task.status == "submission_uncertain"


def test_seedance_local_preparation_failure_is_safe_to_retry(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    arguments = {
        "run_id": "run-1",
        "resource_id": "S01",
        "payload": {"shot_id": "S01"},
        "provider_endpoint": "https://seedance.test",
        "output_path": tmp_path / "output.mp4",
        "poll": lambda provider_job_id: "https://video.test/output.mp4",
        "download": lambda url, path: Path(path).write_bytes(b"video") or path,
    }

    with pytest.raises(ProviderPreparationError, match="tail anchor"):
        execute_seedance_video_task(
            store,
            submit=lambda: (_ for _ in ()).throw(
                ProviderPreparationError("tail anchor extraction failed")
            ),
            **arguments,
        )

    with sqlite3.connect(store.database_path) as connection:
        failed_status, failed_provider_job_id = connection.execute(
            "SELECT status, provider_job_id FROM generation_tasks ORDER BY queued_at LIMIT 1"
        ).fetchone()
    assert failed_status == "failed"
    assert failed_provider_job_id is None

    recovered = execute_seedance_video_task(
        store,
        submit=lambda: "provider-job-1",
        **arguments,
    )

    assert recovered.provider_job_id == "provider-job-1"
    assert store.get(recovered.task_id).status == "succeeded"


def test_seedance_resume_refuses_a_changed_provider_endpoint(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")

    with pytest.raises(TimeoutError):
        execute_seedance_video_task(
            store,
            run_id="run-1",
            resource_id="S01",
            payload={"shot_id": "S01"},
            provider_endpoint="https://seedance-a.test",
            output_path=tmp_path / "output.mp4",
            submit=lambda: "provider-job-1",
            poll=lambda provider_job_id: (_ for _ in ()).throw(TimeoutError()),
            download=lambda url, path: path,
        )

    with pytest.raises(ProviderEndpointChangedError, match="endpoint changed"):
        execute_seedance_video_task(
            store,
            run_id="run-1",
            resource_id="S01",
            payload={"shot_id": "S01"},
            provider_endpoint="https://seedance-b.test",
            output_path=tmp_path / "output.mp4",
            submit=lambda: pytest.fail("resume must not submit"),
            poll=lambda provider_job_id: pytest.fail("wrong endpoint must not poll"),
            download=lambda url, path: path,
        )


def test_phase6_resumes_persisted_seedance_job_after_poll_interruption(
    tmp_path, monkeypatch
):
    shot_dir = _write_shot(tmp_path)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "direct")
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 1)
    monkeypatch.setattr(pipeline_core, "get_api_key", lambda service: "test-key")
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    submissions = []
    poll_calls = []

    def submit(content, **kwargs):
        submissions.append(kwargs)
        return "provider-job-1"

    def poll(provider_job_id, api_key):
        poll_calls.append(provider_job_id)
        if len(poll_calls) == 1:
            raise TimeoutError("worker stopped while polling")
        return "https://video.test/output.mp4"

    def download(url, output_path):
        Path(output_path).write_bytes(b"v" * 11000)
        return output_path

    monkeypatch.setattr(seedance_client, "submit_content", submit)
    monkeypatch.setattr(seedance_client, "poll", poll)
    monkeypatch.setattr(seedance_client, "download", download)
    monkeypatch.setattr(
        pipeline_core.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12.0\n", returncode=0),
    )

    first = pipeline_core._run_phase6_fallback(tmp_path)
    second = pipeline_core._run_phase6_fallback(tmp_path)

    assert first["status"] == "error"
    assert second["status"] == "done"
    assert len(submissions) == 1
    assert poll_calls == ["provider-job-1", "provider-job-1"]
    assert json.loads((shot_dir / "SHOT_META.json").read_text())["task_id"] == (
        "provider-job-1"
    )

    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        persisted = connection.execute(
            "SELECT status, provider_job_id FROM generation_tasks"
        ).fetchone()
    assert persisted == ("succeeded", "provider-job-1")


def test_capacity_table_uses_provider_lane_and_media_default(monkeypatch):
    monkeypatch.setenv("HONCUT_SEEDANCE_VIDEO_CONCURRENCY", "3")
    capacities = CapacityTable.for_seedance_video(fallback=1)

    assert capacities.get("seedance", "video") == 3
    assert capacities.get("seedance", "image") == 0
    assert capacities.get("unknown-provider", "video") == 1


def test_slot_table_caps_concurrency_and_releases_after_failure():
    slots = SlotTable()
    lock = threading.Lock()
    active = 0
    peak = 0

    def occupy(index):
        nonlocal active, peak
        with slots.reserve("seedance", "video", f"task-{index}", capacity=2):
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                if index == 2:
                    raise RuntimeError("provider execution failed")
            finally:
                with lock:
                    active -= 1

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(occupy, index) for index in range(5)]
        failures = []
        for future in futures:
            try:
                future.result()
            except RuntimeError as error:
                failures.append(str(error))

    assert failures == ["provider execution failed"]
    assert peak == 2
    assert slots.occupied("seedance", "video") == 0


def _hold_shared_slot(
    database_path: str,
    task_id: str,
    attempting,
    acquired,
    release,
) -> None:
    slots = CrossProcessSlotTable(
        Path(database_path),
        lease_ttl=2,
        heartbeat_interval=0.1,
        poll_interval=0.01,
    )
    attempting.put(task_id)
    with slots.reserve(
        "seedance",
        "video",
        task_id,
        capacity=1,
        wait_timeout=4,
    ):
        acquired.put(task_id)
        if not release.wait(4):
            raise TimeoutError(f"test release signal not received for {task_id}")


def test_cross_process_slot_blocks_until_other_process_releases(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process capacity proof requires a POSIX fork runtime")
    context = multiprocessing.get_context("fork")
    database_path = str(tmp_path / "capacity.db")
    attempting = context.Queue()
    acquired = context.Queue()
    release_first = context.Event()
    release_second = context.Event()
    first = context.Process(
        target=_hold_shared_slot,
        args=(database_path, "first", attempting, acquired, release_first),
    )
    second = context.Process(
        target=_hold_shared_slot,
        args=(database_path, "second", attempting, acquired, release_second),
    )

    first.start()
    try:
        assert attempting.get(timeout=5) == "first"
        assert acquired.get(timeout=5) == "first"
        second.start()
        assert attempting.get(timeout=5) == "second"
        with pytest.raises(Empty):
            acquired.get(timeout=0.25)

        release_first.set()
        assert acquired.get(timeout=5) == "second"
        release_second.set()
    finally:
        release_first.set()
        release_second.set()
        first.join(timeout=5)
        if second.pid is not None:
            second.join(timeout=5)
        if first.is_alive():
            first.terminate()
            first.join(timeout=2)
        if second.pid is not None and second.is_alive():
            second.terminate()
            second.join(timeout=2)

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_expired_capacity_lease_can_be_reclaimed(tmp_path):
    database_path = tmp_path / "capacity.db"
    first = CrossProcessSlotTable(
        database_path,
        lease_ttl=0.05,
        heartbeat_interval=0.01,
        poll_interval=0.005,
    )
    abandoned = first.acquire("seedance", "video", "abandoned", capacity=1)

    time.sleep(0.08)
    second = CrossProcessSlotTable(
        database_path,
        lease_ttl=0.2,
        heartbeat_interval=0.05,
        poll_interval=0.005,
    )
    reclaimed = second.acquire(
        "seedance",
        "video",
        "replacement",
        capacity=1,
        wait_timeout=0.2,
    )

    assert reclaimed.lease_id != abandoned.lease_id
    assert reclaimed.slot_index == 0
    second.release(reclaimed)
    assert second.occupied("seedance", "video") == 0


def test_heartbeat_keeps_a_long_running_capacity_lease_alive(tmp_path):
    # Timing scaled up ~5x from the original millisecond values: the contract
    # (heartbeat refreshes the lease) is unchanged, but the margins are now
    # immune to scheduler jitter on busy machines.
    database_path = tmp_path / "capacity.db"
    first = CrossProcessSlotTable(
        database_path,
        lease_ttl=0.4,
        heartbeat_interval=0.05,
        poll_interval=0.02,
    )
    second = CrossProcessSlotTable(
        database_path,
        lease_ttl=0.4,
        heartbeat_interval=0.05,
        poll_interval=0.02,
    )

    with first.reserve("seedance", "video", "long-running", capacity=1):
        time.sleep(0.7)
        with pytest.raises(CapacityWaitTimeoutError):
            second.acquire(
                "seedance",
                "video",
                "blocked",
                capacity=1,
                wait_timeout=0.2,
            )

    assert first.occupied("seedance", "video") == 0


def test_phase6_honors_seedance_provider_capacity(tmp_path, monkeypatch):
    for index in range(1, 5):
        shot_dir = tmp_path / "shots" / f"S{index:02d}"
        shot_dir.mkdir(parents=True)
        (shot_dir / "SHOT_META.json").write_text(
            json.dumps(
                {"prompt": f"quiet landscape {index}", "gen_strategy": "i2v"}
            ),
            encoding="utf-8",
        )

    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "direct")
    monkeypatch.setenv("HONCUT_SEEDANCE_VIDEO_CONCURRENCY", "2")
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 5)
    monkeypatch.setattr(pipeline_core, "get_api_key", lambda service: "test-key")
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    lock = threading.Lock()
    active = 0
    peak = 0
    submission_count = 0

    def submit(content, **kwargs):
        nonlocal active, peak, submission_count
        with lock:
            active += 1
            peak = max(peak, active)
            submission_count += 1
            provider_job_id = f"provider-job-{submission_count}"
        try:
            time.sleep(0.03)
            return provider_job_id
        finally:
            with lock:
                active -= 1

    def download(url, output_path):
        Path(output_path).write_bytes(b"v" * 11000)
        return output_path

    monkeypatch.setattr(seedance_client, "submit_content", submit)
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda provider_job_id, api_key: f"https://video.test/{provider_job_id}.mp4",
    )
    monkeypatch.setattr(seedance_client, "download", download)
    monkeypatch.setattr(
        pipeline_core.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12.0\n", returncode=0),
    )

    phase_outcome = pipeline_core._run_phase6_fallback(tmp_path)

    assert phase_outcome["status"] == "done"
    assert len(phase_outcome["outputs"]) == 4
    assert submission_count == 4
    assert peak == 2
    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        succeeded = connection.execute(
            "SELECT COUNT(*) FROM generation_tasks WHERE status = 'succeeded'"
        ).fetchone()[0]
    assert succeeded == 4


def test_multimodal_review_uses_responses_contract_and_bounded_output(tmp_path):
    observed = {}

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text='{"ok": true}')],
                    )
                ]
            )

    class FakeClient:
        responses = FakeResponses()

    image = tmp_path / "grid.jpg"
    image.write_bytes(b"image-bytes")
    client = ark_multimodal_client.ArkMultimodalClient(
        client=FakeClient(),
        media_url_resolver=lambda _path: "https://tos.test/grid.jpg?signed=1",
    )

    assert client.review([image], "review") == '{"ok": true}'
    assert observed["model"] == "doubao-seed-2-0-lite-260428"
    assert observed["max_output_tokens"] == 4096
    assert observed["extra_body"] == {"thinking": {"type": "disabled"}}
    assert observed["text"] == {"format": {"type": "json_object"}}
    assert observed["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "review"},
                {"type": "input_text", "text": "Input image 1: grid"},
                {
                    "type": "input_image",
                    "image_url": "https://tos.test/grid.jpg?signed=1",
                    "detail": "high",
                },
            ],
        }
    ]


def test_multimodal_request_preserves_image_video_document_audio_order(tmp_path):
    observed = {}

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(output_text='{"ok": true}', output=[])

    class FakeClient:
        responses = FakeResponses()

    paths = [
        tmp_path / "frame.png",
        tmp_path / "motion.mp4",
        tmp_path / "notes.pdf",
        tmp_path / "dialogue.mp3",
    ]
    for path in paths:
        path.write_bytes(b"fixture")

    client = ark_multimodal_client.ArkMultimodalClient(
        client=FakeClient(),
        media_url_resolver=lambda path: f"https://tos.test/{path.name}?signed=1",
    )

    assert client.review_media(paths, "audit") == '{"ok": true}'
    content = observed["input"][0]["content"]
    media_items = [item for item in content if item["type"] != "input_text"]
    assert media_items == [
        {
            "type": "input_image",
            "image_url": "https://tos.test/frame.png?signed=1",
            "detail": "high",
        },
        {
            "type": "input_video",
            "video_url": "https://tos.test/motion.mp4?signed=1",
            "fps": 1.0,
        },
        {
            "type": "input_file",
            "file_url": "https://tos.test/notes.pdf?signed=1",
        },
        {
            "type": "input_audio",
            "audio_url": "https://tos.test/dialogue.mp3?signed=1",
        },
    ]


def test_multimodal_client_uses_standard_responses_url_with_agent_key(monkeypatch):
    observed = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setenv("ARK_AGENT_API_KEY", "agent-plan-key")
    monkeypatch.setenv("ARK_API_KEY", "coding-plan-key")
    monkeypatch.setenv(
        "HONCUT_STORYBOARD_REVIEW_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/plan/v3",
    )
    monkeypatch.setattr(ark_multimodal_client, "OpenAI", FakeOpenAI)

    ark_multimodal_client.ArkMultimodalClient()

    assert observed["api_key"] == "agent-plan-key"
    assert observed["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert observed["max_retries"] == 0


def test_seedream_accepts_multiple_character_references(tmp_path, monkeypatch):
    first = tmp_path / "lin.png"
    second = tmp_path / "jin.jpg"
    first.write_bytes(b"lin")
    second.write_bytes(b"jin")
    observed = {}
    client = SeedreamClient(api_key="test")

    def fake_call(payload, output_path, timeout=180):
        observed.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_call_and_save", fake_call)

    assert client.image_to_image(
        "two locked characters",
        [str(first), str(second)],
        output_path=str(tmp_path / "out.png"),
    ) == "ok"
    assert isinstance(observed["image"], list)
    assert observed["image"][0].startswith("data:image/png;base64,")
    assert observed["image"][1].startswith("data:image/jpeg;base64,")


def test_storyboard_keyframe_prompt_prioritizes_identity_and_action():
    prompt = pipeline_core._storyboard_keyframe_description(
        {
            "subject_description": "凛—银白长发女性；烬—黑色短发男性",
            "action_description": "烬抓住凛的手腕，将她甩向汽车残骸",
            "visual": "雨夜高架桥，二人近身缠斗",
        }
    )

    assert "银白长发女性" in prompt
    assert "Exact action contract" in prompt
    assert "将她甩向汽车残骸" in prompt
    assert "decisive final pose" in prompt
    assert "No exposed midriff" in prompt


def test_storyboard_keyframe_prompt_keeps_dialogue_out_of_image():
    prompt = pipeline_core._storyboard_keyframe_description(
        {
            "who": ["凛", "烬"],
            "action_description": "刀锋撞上机械臂。\n“凛，停下。”“放手！”",
            "visual": "暴雨中的废弃高架",
        }
    )

    assert "刀锋撞上机械臂" in prompt
    assert "凛，停下" not in prompt
    assert "放手" not in prompt
    assert "no speech bubbles" in prompt


def test_end_frame_prompt_uses_last_micro_action_and_full_cast():
    prompt = pipeline_core.build_end_frame_prompt(
        {
            "who": ["凛", "烬"],
            "micro_actions": ["凛冲出", "烬扣住刀背，二人陷入角力僵持"],
            "subject_description": "凛银白长发；烬黑色机械左臂",
            "prompt": "暴雨中的废弃高架",
        }
    )

    assert "Exact final micro-action: 烬扣住刀背，二人陷入角力僵持" in prompt
    assert "exactly 2 principal character(s): 凛, 烬" in prompt
    assert "凛银白长发；烬黑色机械左臂" in prompt
    assert "do not omit" in prompt
    assert "no speech bubbles" in prompt


def test_end_frame_prompt_does_not_invent_alliance_staging():
    prompt = pipeline_core.build_end_frame_prompt(
        {
            "who": ["凛", "烬"],
            "what": "凛与烬并肩持刃共同迎敌",
            "micro_actions": ["两柄黑刃同时抬起，指向前方"],
            "prompt": "机械军阵正在远处雨幕中逼近",
        }
    )

    assert "两柄黑刃同时抬起，指向前方" in prompt
    assert "stand shoulder-to-shoulder" not in prompt
    assert "rear three-quarter camera behind the allies" not in prompt
    assert "blades stay parallel and must not cross" not in prompt


def test_storyboard_keyframe_explicit_empty_cast_is_environment_only():
    prompt = pipeline_core._storyboard_keyframe_description(
        {"who": [], "action_description": "云层翻涌", "visual": "金色云海"}
    )

    assert "Environment-only cinematic keyframe" in prompt
    assert "zero people" in prompt


def test_l3_severity_blocks_identity_but_not_static_pose_nuance():
    assert _calibrate_l3_severity(
        "R1", "severe", "wrong character identity and gender"
    ) == "severe"
    assert _calibrate_l3_severity(
        "R1", "severe", "black jacket instead of distressed black denim"
    ) == "moderate"
    assert _calibrate_l3_severity(
        "R4", "severe", "blade angle is diagonal rather than horizontal"
    ) == "moderate"
    assert _calibrate_l3_severity(
        "R4", "severe", "reversed attacker and defender"
    ) == "severe"
    assert _calibrate_l3_severity(
        "R1", "severe", "the female character resembles a celebrity"
    ) == "moderate"
    assert _calibrate_l3_severity(
        "R1", "severe", "gender mismatch: female instead of male"
    ) == "severe"


def test_story_order_accepts_phase1_numeric_ids():
    assert _shot_id(1) == "S01"
    assert _shot_id("2") == "S02"
    assert _shot_id("s03") == "S03"
    assert _shot_id(True) is None
    assert storyboard_shot_ids({"shots": [{"id": 1}, {"id": 2}]}) == [
        "S01",
        "S02",
    ]


def test_ffmpeg_subtitle_fallback_honors_readable_style(tmp_path, monkeypatch):
    burner = RemotionCaptionBurn()
    observed = {}

    def capture(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(burner, "run_command", capture)
    burner._render_with_subtitles(
        "input.mp4",
        "output.mp4",
        tmp_path / "captions.srt",
        48,
        "#FFFFFF",
        "#000000",
        3,
        60,
    )

    video_filter = observed["command"][observed["command"].index("-vf") + 1]
    assert "FontSize=48" in video_filter
    assert "PrimaryColour=&H00FFFFFF" in video_filter
    assert "OutlineColour=&H00000000" in video_filter
    assert "Outline=3" in video_filter
    assert "MarginV=60" in video_filter


def test_ass_subtitles_keep_dialogue_cues_separate_and_fade():
    content = RemotionCaptionBurn._ass_subtitle_content(
        [
            {"word": "第一条", "startMs": 1000, "endMs": 2000, "cueId": 0},
            {"word": "第二条", "startMs": 3000, "endMs": 4000, "cueId": 1},
        ],
        fade_in_ms=180,
        fade_out_ms=220,
    )

    dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 2
    assert "{\\fad(180,220)}第一条" in dialogue_lines[0]
    assert "{\\fad(180,220)}第二条" in dialogue_lines[1]
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content


def test_ffmpeg_fallback_keeps_all_words_in_one_dialogue_cue():
    pages = RemotionCaptionBurn._caption_pages([
        {"word": word, "startMs": index * 100, "endMs": (index + 1) * 100, "cueId": 0}
        for index, word in enumerate("我从来没想伤你")
    ])

    assert pages == [("我从来没想伤你", 0.0, 0.7)]


def test_asr_speech_is_suppressed_on_unmarked_shots_in_scripted_scene():
    merged = pipeline_core._merge_shot_transcripts(
        [
            {"shot_id": "S01", "dialogue": {"line": "留下"}},
            {"shot_id": "S02"},
        ],
        [1000, 1000],
        [
            {"text": "留下", "segments": []},
            {"text": "你骗了我", "segments": [
                {"word": "你骗了我", "start_ms": 100, "end_ms": 700},
            ]},
        ],
    )

    assert merged["shots"][1]["source"] == "none"
    assert merged["shots"][1]["text"] == ""
    assert [item["text"] for item in merged["caption_segments"]] == ["留下"]


def test_explicit_script_line_takes_priority_over_mismatched_asr():
    merged = pipeline_core._merge_shot_transcripts(
        [{"shot_id": "S01", "dialogue": {"line": "剧本台词"}}],
        [1000],
        [{"text": "实际说出的话", "segments": [
            {"word": "实际说出的话", "start_ms": 100, "end_ms": 800},
        ]}],
    )

    assert merged["shots"][0]["source"] == "dialogue_script"
    assert merged["shots"][0]["text"] == "剧本台词"


def test_final_mix_asr_uses_utterance_boundaries_for_caption_cues():
    captions = pipeline_core._caption_segments_from_final_asr({
        "utterances": [
            {
                "text": "第一句。",
                "start_ms": 1000,
                "end_ms": 1800,
                "words": [{"word": "第一句", "start_ms": 1000, "end_ms": 1800}],
            },
            {
                "text": "第二句！",
                "start_ms": 2500,
                "end_ms": 3300,
                "words": [{"word": "第二句", "start_ms": 2500, "end_ms": 3300}],
            },
        ],
    })

    assert [item["text"] for item in captions] == ["第一句", "第二句"]
    assert [(item["start"], item["end"]) for item in captions] == [
        (1.0, 1.8),
        (2.5, 3.3),
    ]


def test_final_mix_asr_splits_one_utterance_at_audible_pause():
    captions = pipeline_core._caption_segments_from_final_asr({
        "utterances": [{
            "text": "第一句第二句",
            "words": [
                {"word": "第一句", "start_ms": 1000, "end_ms": 1800},
                {"word": "第二句", "start_ms": 2100, "end_ms": 2800},
            ],
        }],
    })

    assert [item["text"] for item in captions] == ["第一句", "第二句"]


def test_final_mix_asr_rejects_transient_noise_as_ultrashort_syllables():
    captions = pipeline_core._caption_segments_from_final_asr({
        "utterances": [{
            "text": "还不够K五少令",
            "words": [
                {"word": "还", "start_ms": 1000, "end_ms": 1320},
                {"word": "不", "start_ms": 1320, "end_ms": 1520},
                {"word": "够", "start_ms": 1520, "end_ms": 1720},
                {"word": "K", "start_ms": 2000, "end_ms": 2040},
                {"word": "五", "start_ms": 2160, "end_ms": 2200},
                {"word": "少", "start_ms": 2320, "end_ms": 2400},
                {"word": "令", "start_ms": 2560, "end_ms": 2600},
            ],
        }],
    })

    assert [item["text"] for item in captions] == ["还不够"]
