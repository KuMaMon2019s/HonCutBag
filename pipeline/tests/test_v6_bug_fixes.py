import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clients import local_video_client
from clients import tos_uploader
from phases.character_discoverer import (
    ENTITY_SUFFIXES,
    _filter_descriptive_phrases,
    _post_filter_characters,
)
from phases.character_discoverer import _add_reference_contract
from phases.character_factory import build_combined_sheet_prompt, build_model_reference_prompts
from phases.scene_consistency import generate_scene_consistency
from phases import adaptation_engine, storyboard_generator
from phases import pipeline_core
from phases.adaptation_engine import estimate_shot_count
from phases.storyboard_generator import _build_characters_map, _build_shot_prompt
from phases.video_generator import BASE_NEGATIVE_PROMPT, build_video_prompt
from prompt.eight_layer_summary import build_subject_summary
from tools import asset_packager
import cron_monitor
import phase_orchestrator


def test_default_shot_count_uses_twelve_second_target():
    assert estimate_shot_count(45) == 4


def test_orchestrator_passes_configured_shot_duration(monkeypatch, tmp_path):
    config = {
        "input": "story.txt",
        "duration": 45,
        "shot_duration": 10,
        "output_dir": str(tmp_path),
    }
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(phase_orchestrator.subprocess, "run", fake_run)

    assert phase_orchestrator._normalize_shot_duration(config) == 10
    phase_orchestrator.run_phase("phase2", config)

    command = commands[0]
    assert command[command.index("--shot-duration") + 1] == "10"


def test_orchestrator_clamps_shot_duration_with_warning(capsys):
    config = {"shot_duration": 20}

    assert phase_orchestrator._normalize_shot_duration(config) == 15
    assert config["shot_duration"] == 15
    assert "clamped to 15s" in capsys.readouterr().err


def test_orchestrator_passes_chain_mode(monkeypatch, tmp_path):
    config = {
        "input": "story.txt",
        "duration": 45,
        "shot_duration": 12,
        "chain_mode": True,
        "output_dir": str(tmp_path),
    }
    commands = []
    monkeypatch.setattr(
        phase_orchestrator.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    phase_orchestrator.run_phase("phase5", config)

    assert "--chain-mode" in commands[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_seedance_submit_last_frame_flag_is_opt_in(monkeypatch, enabled):
    posted = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"task_id": "seedance-1"}

    session = SimpleNamespace(
        post=lambda *args, **kwargs: posted.append(kwargs["json"]) or Response()
    )
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)

    local_video_client.submit(
        prompt="shot",
        content=[{"type": "text", "text": "shot"}],
        model="seedance",
        return_last_frame=enabled,
    )

    if enabled:
        assert posted[0]["return_last_frame"] is True
    else:
        assert "return_last_frame" not in posted[0]


def test_generate_saves_base64_last_frame_and_missing_value_warns(
    tmp_path, monkeypatch, capsys
):
    poll_results = iter([
        {"status": "completed", "progress": 100, "last_frame_url": "ZnJhbWU="},
        {"status": "completed", "progress": 100, "last_frame_url": None},
    ])
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: next(poll_results))

    def fake_download(task_id, output_path, verification_out, **kwargs):
        Path(output_path).write_bytes(b"v" * 2048)

    monkeypatch.setattr(local_video_client, "download", fake_download)
    (tmp_path / "S01").mkdir()
    (tmp_path / "S02").mkdir()

    first = local_video_client.generate_video(
        prompt="one",
        output_path=str(tmp_path / "S01" / "output.mp4"),
        model="seedance",
        return_last_frame=True,
    )
    second = local_video_client.generate_video(
        prompt="two",
        output_path=str(tmp_path / "S02" / "output.mp4"),
        model="seedance",
        return_last_frame=True,
    )

    assert Path(first["last_frame_path"]).read_bytes() == b"frame"
    assert second["last_frame_path"] is None
    assert "[chain] S02: Bridge 未返回尾帧" in capsys.readouterr().out


def test_phase5_chain_is_serial_and_injects_previous_last_frame(
    tmp_path, monkeypatch
):
    shots = tmp_path / "shots"
    for shot_id in ("S01", "S02", "S03"):
        shot_dir = shots / shot_id
        shot_dir.mkdir(parents=True)
        (shot_dir / "SHOT_META.json").write_text(
            json.dumps({"prompt": f"prompt-{shot_id}", "gen_strategy": "i2v"})
        )

    calls = []

    def fake_generate(**kwargs):
        shot_dir = Path(kwargs["output_path"]).parent
        calls.append((shot_dir.name, kwargs.get("content")))
        Path(kwargs["output_path"]).write_bytes(b"v" * 11000)
        last_frame = shot_dir / "last_frame.jpg"
        last_frame.write_bytes(f"frame-{shot_dir.name}".encode())
        return {
            "output_path": kwargs["output_path"],
            "last_frame_path": str(last_frame),
            "actual_model": "seedance",
        }

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    monkeypatch.setattr(local_video_client, "generate_video_with_fallback", fake_generate)
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_id"]}],
    )
    monkeypatch.setattr(asset_packager, "package_shot_assets", lambda **kwargs: (None, []))
    monkeypatch.setattr("utils.config.VIDEO_GEN_CONCURRENCY", 3)

    result = pipeline_core._run_phase5_fallback(tmp_path, chain_mode=True)

    assert result["status"] == "done"
    assert [shot_id for shot_id, _ in calls] == ["S01", "S02", "S03"]
    for index, expected in ((1, b"frame-S01"), (2, b"frame-S02")):
        image_items = [item for item in calls[index][1] if item.get("role") == "first_frame"]
        encoded = image_items[0]["image_url"]["url"].split(",", 1)[1]
        import base64
        assert base64.b64decode(encoded) == expected
    first_meta = json.loads((shots / "S01" / "SHOT_META.json").read_text())
    second_meta = json.loads((shots / "S02" / "SHOT_META.json").read_text())
    assert first_meta["chain_active"] is False
    assert second_meta["chain_active"] is True
    assert second_meta["chain_source"] == "S01"


class _DownloadResponse:
    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"downloaded-video"


def test_download_keeps_file_when_ffprobe_metadata_is_unreadable(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "output.mp4"
    session = SimpleNamespace(get=lambda *args, **kwargs: _DownloadResponse())
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    monkeypatch.setattr(
        local_video_client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid data"
        ),
    )

    local_video_client.download(
        "task-123",
        str(output_path),
        expected_duration=10.0,
        expected_width=1280,
        expected_height=720,
    )

    assert output_path.read_bytes() == b"downloaded-video"
    assert "WARNING: metadata unavailable" in capsys.readouterr().err


def test_seedance_download_keeps_valid_video_when_dimensions_differ(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "output.mp4"
    session = SimpleNamespace(get=lambda *args, **kwargs: _DownloadResponse())
    monkeypatch.setattr(
        _DownloadResponse,
        "iter_content",
        lambda self, chunk_size: iter([b"v" * 2048]),
    )
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    monkeypatch.setattr(
        local_video_client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"format":{"duration":"5.1"},"streams":['
                   '{"codec_type":"video","width":1920,"height":1080}]}',
            stderr="",
        ),
    )

    local_video_client.download(
        "seedance-task",
        str(output_path),
        expected_duration=7.0,
        expected_width=1280,
        expected_height=720,
        model="seedance",
    )

    assert output_path.exists()
    assert output_path.stat().st_size == 2048


def test_seedance_download_deletes_file_without_video_stream(tmp_path, monkeypatch):
    output_path = tmp_path / "output.mp4"
    session = SimpleNamespace(get=lambda *args, **kwargs: _DownloadResponse())
    monkeypatch.setattr(
        _DownloadResponse,
        "iter_content",
        lambda self, chunk_size: iter([b"x" * 2048]),
    )
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    monkeypatch.setattr(
        local_video_client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"format":{},"streams":[{"codec_type":"audio"}]}',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="Download verification failed"):
        local_video_client.download(
            "seedance-task",
            str(output_path),
            expected_duration=7.0,
            expected_width=1280,
            expected_height=720,
            model="seedance",
        )

    assert not output_path.exists()


def test_seedance_timeout_falls_back_to_wan22_and_records_actual_model(
    tmp_path, monkeypatch, capsys
):
    shot_dir = tmp_path / "S03"
    shot_dir.mkdir()
    output_path = shot_dir / "output.mp4"
    meta_path = shot_dir / "SHOT_META.json"
    meta_path.write_text(json.dumps({"duration": 12, "prompt": "identity-lock"}))

    submissions = []
    polls = iter([TimeoutError("stalled"), {"status": "completed", "progress": 100}])

    def fake_submit(**kwargs):
        submissions.append(kwargs)
        return f"task-{kwargs['model']}"

    def fake_poll(task_id):
        result = next(polls)
        if isinstance(result, Exception):
            raise result
        return result

    def fake_download(task_id, path, verification_out, **kwargs):
        Path(path).write_bytes(b"video")
        verification_out.update(duration=97 / 24, num_frames=97)

    monkeypatch.setattr(local_video_client, "submit", fake_submit)
    monkeypatch.setattr(local_video_client, "poll", fake_poll)
    monkeypatch.setattr(local_video_client, "download", fake_download)

    local_video_client.generate_video_with_fallback(
        prompt="identity-lock and positive negative-guardrails",
        output_path=str(output_path),
        duration=12,
        content=[{"type": "text", "text": "same assets"}],
    )

    assert [call["model"] for call in submissions] == ["seedance", "wan22"]
    assert submissions[0]["timeout"] == 60
    assert submissions[0]["prompt"] == submissions[1]["prompt"]
    assert submissions[0]["content"] == submissions[1]["content"]
    assert submissions[1]["num_frames"] == 145
    assert json.loads(meta_path.read_text())["actual_model"] == "wan22"
    metadata = json.loads(meta_path.read_text())
    assert metadata["requested_duration"] == 12
    assert metadata["actual_duration"] == pytest.approx(97 / 24)
    assert (
        "[fallback] S03: Seedance timeout → Wan2.2 "
        "(duration 12s → 6s, Wan2.2 max ~6s)"
    ) in capsys.readouterr().out


def test_phantom_uses_new_character_reference_assets(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "hero"
    char_dir.mkdir(parents=True)
    (char_dir / "face_closeup.png").write_bytes(b"f" * 2048)
    (char_dir / "full_body.png").write_bytes(b"b" * 2048)
    monkeypatch.setattr(
        tos_uploader,
        "upload_image",
        lambda data, content_type: f"https://assets.test/{data[:1].decode()}.png",
    )

    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S04",
        {"prompt": "hero speaks", "gen_strategy": "phantom", "_char_ids": ["hero"]},
    )

    references = [item for item in content if item.get("role") == "reference_image"]
    assert len(references) == 2
    assert [item["image_url"]["url"] for item in references] == [
        "https://assets.test/f.png",
        "https://assets.test/b.png",
    ]

    (char_dir / "face_closeup.png").unlink()
    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S04",
        {"prompt": "hero speaks", "gen_strategy": "phantom", "_char_ids": ["hero"]},
    )
    references = [item for item in content if item.get("role") == "reference_image"]
    assert references[0]["image_url"]["url"] == "https://assets.test/b.png"

    (char_dir / "full_body.png").unlink()
    (char_dir / "variant_中文造型.png").write_bytes(b"v" * 2048)
    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S04",
        {"prompt": "hero speaks", "gen_strategy": "phantom", "_char_ids": ["hero"]},
    )
    references = [item for item in content if item.get("role") == "reference_image"]
    assert references[0]["image_url"]["url"] == "https://assets.test/v.png"


def test_tos_upload_uses_jpeg_metadata_after_large_png_compression(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tos_uploader,
        "_get_tos_config",
        lambda: {
            "ak": "ak",
            "sk": "sk",
            "bucket": "bucket",
            "endpoint": "tos.example.test",
            "region": "cn-test",
        },
    )
    monkeypatch.setattr(
        tos_uploader,
        "compress_image_bytes",
        lambda data: b"\xff\xd8\xff" + b"j" * 2048,
    )

    def fake_put(url, data, headers, timeout):
        captured.update(url=url, data=data, headers=headers, timeout=timeout)
        return SimpleNamespace(status_code=200, text="")

    monkeypatch.setattr(tos_uploader.requests, "put", fake_put)

    url = tos_uploader.upload_image(b"p" * (5_400_000), "image/png")

    assert url is not None
    assert ".jpg?" in url
    assert captured["headers"]["Content-Type"] == "image/jpeg"


def test_compound_robot_name_passes_character_filter():
    name = "白色金属AI巡检机器人"
    stats = {name: {"count": 3, "events": ["E01", "E02", "E03"]}}

    filtered = _filter_descriptive_phrases(stats)

    assert name in filtered


def test_scifi_entity_suffixes_and_compound_mercenary_survive_filter():
    for suffix in ("佣兵", "机械体", "合成人", "复制体"):
        assert suffix in ENTITY_SUFFIXES

    name = "黑色重型机械合成人佣兵"
    assert name in _filter_descriptive_phrases({name: {"count": 1, "events": [1]}})


def test_overlong_entity_description_is_filtered_even_with_known_suffix():
    name = "与零七号外观完全相同胸部能源核心呈红色的机械复制体"
    assert name not in _filter_descriptive_phrases({name: {"count": 1, "events": [1]}})


def test_post_filter_merges_aliases_and_removes_generic_protagonist():
    characters = [
        {"id": "zero_seven", "name": "07号", "aliases": ["佣兵"], "role": "protagonist"},
        {"id": "mercenary", "name": "佣兵", "aliases": [], "role": "protagonist"},
        {"id": "generic", "name": "主角", "aliases": [], "role": "protagonist"},
        {"id": "technician", "name": "机械技师", "aliases": [], "role": "supporting"},
        {"id": "silver_technician", "name": "银白色机械技师", "aliases": [], "role": "supporting"},
    ]

    filtered = _post_filter_characters(characters)

    assert [char["name"] for char in filtered] == ["07号", "银白色机械技师"]
    assert {"佣兵", "主角"} <= set(filtered[0]["aliases"])
    assert "机械技师" in filtered[1]["aliases"]


def test_adaptation_prompt_restricts_who_to_canonical_character_names():
    assert "who 只能逐字引用上方角色列表中的主名" in adaptation_engine.USER_PROMPT_TEMPLATE
    assert "群体/群众/背景元素" in adaptation_engine.USER_PROMPT_TEMPLATE


def test_storyboard_character_map_uses_new_face_reference_contract():
    mapping = _build_characters_map(
        [{"id": "hero", "name": "主角", "aliases": ["英雄"]}]
    )

    assert mapping["主角"] == "characters/hero/face_closeup.png"
    assert mapping["英雄"] == "characters/hero/face_closeup.png"


def test_storyboard_prompt_uses_eight_layers_without_fast_motion():
    shot = {
        "id": 1,
        "who": ["林夏"],
        "shot_size": "close_up",
        "what": "转身看向窗外",
        "emotion": "紧张",
        "where": "夜间办公室",
        "duration": 6,
    }
    characters = [{
        "id": "lin_xia", "name": "林夏", "aliases": [],
        "appearance": {"hair": "黑色短发", "face": "圆脸杏眼", "clothing": "白衬衫"},
    }]

    prompt = _build_shot_prompt(shot, characters)

    for layer in ("元素参考声明", "镜头1", "动作：", "运镜：", "场景与光影：", "音效：", "全局收尾："):
        assert layer in prompt
    assert "fast" not in prompt.lower()
    assert "快速" not in prompt


def test_character_reference_contract_and_model_routes():
    character = {
        "name": "林夏",
        "appearance": {"hair": "黑色短发", "face": "圆脸", "clothing": "白衬衫"},
    }
    _add_reference_contract(character)
    assert character["face_reference"] == "face_closeup.png"
    assert character["body_reference"] == "full_body.png"
    assert "<主体1>" in character["prompt_definition"]
    seedance_prompts = build_model_reference_prompts("角色", target_model="seedance")
    assert set(seedance_prompts) == {"face_closeup", "full_body"}
    for prompt in seedance_prompts.values():
        assert "fictional" in prompt.lower()
        assert "virtual avatar" in prompt.lower()
    assert set(build_model_reference_prompts("角色", target_model="kling")) == {"front", "side", "three_quarter", "detail"}
    assert build_combined_sheet_prompt("角色").startswith(
        "【宏观描述】所有角色均为 AI 生成的虚拟形象，非真实人物。"
    )


def test_scene_contract_and_video_prompt_model_routing():
    storyboard = {"shots": [{
        "id": 1, "who": ["林夏"], "where": "夜间工厂", "shot_size": "medium",
        "what": "左手攥紧衣角", "camera_movement": "push_in", "duration": 8,
    }]}
    characters = {"characters": [{
        "id": "lin_xia", "name": "林夏", "aliases": [],
        "appearance": {"hair": "黑色短发", "face": "圆脸", "clothing": "白衬衫"},
        "prompt_definition": "将图片1中的[白衬衫、圆脸、黑色短发]定义为<主体1>",
        "negative_guardrails": "更换服装, 颜色偏移",
    }]}
    contract = generate_scene_consistency(storyboard, characters)
    scene = contract["shots"]["S01"]
    assert "5600K" in scene["lighting_description"]
    assert scene["spatial_layout"]["subject"]
    assert len(scene["negative_prompt"].split(", ")) == 7

    seedance = build_video_prompt(storyboard["shots"][0], characters, contract, "seedance")
    kling = build_video_prompt(storyboard["shots"][0], characters, contract, "kling")
    assert isinstance(seedance, str) and "identity-lock" in seedance and BASE_NEGATIVE_PROMPT in seedance
    assert "虚拟形象声明：片中角色均为 AI 生成的虚构角色，非真实人物" in seedance
    assert isinstance(kling, dict) and BASE_NEGATIVE_PROMPT in kling["negative_prompt"]
    assert kling["prompt"].startswith("虚拟形象声明：片中角色均为 AI 生成的虚构角色，非真实人物。")
    assert "推入(push in)" in kling["prompt"]


def test_subject_summary_limit_never_truncates_layer_eight_guardrails():
    long_text = "人物与环境细节" * 40
    summary = build_subject_summary([
        ("景别与主体：", long_text),
        ("动作：", long_text),
        ("运镜：", "缓慢推进(dolly in)"),
        ("场景与光影：", long_text),
    ])
    assert 40 <= len(summary) <= 100
    for label in ("景别与主体：", "动作：", "运镜：", "场景与光影："):
        assert label in summary

    storyboard = {"shots": [{
        "id": 1, "who": ["林夏"], "where": long_text,
        "subject_description": long_text, "what": long_text,
        "camera_movement": "push_in", "duration": 8,
    }]}
    contract = generate_scene_consistency(storyboard)
    complete_prompt = build_video_prompt(
        storyboard["shots"][0], {"characters": []}, contract, "seedance"
    )
    bounded = complete_prompt.split("主体总结：", 1)[1].split("。", 1)[0]
    assert 40 <= len(bounded) <= 100
    assert "虚拟形象声明：片中角色均为 AI 生成的虚构角色，非真实人物" in complete_prompt
    assert complete_prompt.endswith(f"约束条件：{BASE_NEGATIVE_PROMPT}")
    for guardrail in BASE_NEGATIVE_PROMPT.split(", "):
        assert guardrail in complete_prompt

    storyboard_shot = dict(storyboard["shots"][0])
    _build_shot_prompt(storyboard_shot)
    blueprint = storyboard_shot["eight_layer_prompt"]
    storyboard_summary = blueprint.split("主体总结：", 1)[1].split("\n", 1)[0]
    assert 40 <= len(storyboard_summary) <= 100
    assert f"约束词：{storyboard_generator.QUALITY_GUARDRAILS}" in blueprint
    assert blueprint.index("主体总结：") < blueprint.index("约束词：")


def _phase_result(phase, exit_code=0):
    return {"phase": phase, "exit_code": exit_code, "timestamp": "2026-08-08T08:00:00"}


def test_resume_merges_progress_and_pipeline_report(tmp_path):
    progress = tmp_path / "phase_progress.json"
    progress.write_text(json.dumps({"results": [_phase_result("phase2")]}), encoding="utf-8")
    (tmp_path / "pipeline_report.json").write_text(
        json.dumps({"phases": {"2.5": {"status": "completed"}, "phase3": {"status": "success"}}}),
        encoding="utf-8",
    )
    results = phase_orchestrator._resume_results(tmp_path, progress, "phase4")
    assert [result["phase"] for result in results] == ["phase2", "phase2_5", "phase3"]


def test_monitor_total_and_report_use_canonical_phase_list(tmp_path):
    (tmp_path / "phase_progress.json").write_text(
        json.dumps({
            "status": "running",
            "current_phase": "phase4",
            "phases": phase_orchestrator.PHASES,
            "results": [_phase_result("phase2"), _phase_result("phase2_5"), _phase_result("phase3")],
        }),
        encoding="utf-8",
    )
    status = cron_monitor.get_phase_status(str(tmp_path))
    report = cron_monitor.format_report(status)
    assert status["total_phases"] == 8
    assert "**Progress**: 3/8 phases" in report
    assert "✅ phase2" in report
    assert "⏳ phase4 (running)" in report


def test_check_process_rejects_invalid_pid():
    assert cron_monitor.check_process("0")["running"] is False
    assert cron_monitor.check_process("not-a-pid")["running"] is False


def test_explicit_lighting_adds_missing_atmosphere_dimension():
    contract = generate_scene_consistency({"shots": [{
        "id": 1,
        "where": "办公室",
        "lighting_description": "主光从左上方照射，色温4200K",
    }]})
    lighting = contract["shots"]["S01"]["lighting_description"]
    assert "左上方" in lighting
    assert "4200K" in lighting
    assert "气氛" in lighting
