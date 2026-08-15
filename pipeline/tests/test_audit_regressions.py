"""Regression coverage for the 2026-08-15 P0-P2 production audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import detached_pipeline_launch
import pipeline_runner as pipeline_runner_cli
from phases import pipeline_core
from phases.phase1 import adaptation_engine, director_planner
from phases.phase1.director_storyboard import materialize_director_panels
from phases.phase1.storyboard_beats import plan_storyboard_beats
from phases.phase1.storyboard_generator import _build_shot_prompt_legacy
from phases.phase2.shot_storyboards import (
    build_shot_storyboard_prompt,
    generate_shot_storyboards,
)
from phases.phase6.video_generator import build_video_prompt
from phases.pipeline_core import _write_project_visual_style
from prompt import event_extractor
from runtime.run_manifest import prepare_run_manifest
from utils.video_capabilities import get_video_capabilities
from utils.video_geometry import resolve_video_geometry
from utils.ark_llm import create_ark_client


def test_seedance_limits_are_provider_capabilities_not_global_director_rules():
    seedance = get_video_capabilities(provider="seedance")
    generic = get_video_capabilities(provider="kling")

    assert seedance.name == "seedance-2.x"
    assert seedance.action_limit(5) == 1
    assert generic.name == "generic-video"
    assert generic.action_limit(5) == 2

    seedance_board = {
        "video_provider": "seedance",
        "shots": [{"id": 1, "duration": 8, "micro_actions": ["a", "b", "c", "d"]}],
    }
    generic_board = {
        "video_provider": "kling",
        "shots": [{"id": 1, "duration": 8, "micro_actions": ["a", "b", "c", "d"]}],
    }
    plan_storyboard_beats(seedance_board)
    plan_storyboard_beats(generic_board)

    assert seedance_board["shots"][0]["storyboard_beat_count"] == 2
    assert generic_board["shots"][0]["storyboard_beat_count"] == 1
    assert generic_board["shots"][0]["generation_load"]["capability_profile"] == (
        "generic-video"
    )


def test_event_extractor_selects_a_generic_or_action_contract(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_call(prompt: str, system_prompt: str) -> str:
        calls.append((prompt, system_prompt))
        return "[]"

    monkeypatch.setattr(event_extractor, "_call_llm", fake_call)
    event_extractor._extract_events_from_segment(
        {"content": "她收到一封信。", "format_hint": "general_prose"}
    )
    event_extractor._extract_events_from_segment(
        {"content": "她跃过矮墙。", "format_hint": "prose_action_screenplay"}
    )

    assert "【通用叙事规则】" in calls[0][0]
    assert "动作影视编剧" not in calls[0][1]
    assert "【动作型叙事规则】" in calls[1][0]
    assert "动作影视编剧" in calls[1][1]
    contaminated_examples = ("护栏", "柱体", "武器停", "共同迎敌")
    assert not any(word in calls[0][0] for word in contaminated_examples)


def test_video_prompt_keeps_requested_ratio_and_fast_action_semantics():
    prompt = build_video_prompt(
        {
            "id": 1,
            "aspect_ratio": "9:16",
            "duration": 5,
            "where": "抽象的黑色舞台",
            "action_description": "角色快速冲过光带，fast lateral movement",
            "who": [],
        },
        {"characters": []},
        {},
        "seedance",
    )

    assert "4K, 9:16, 5秒" in prompt
    assert "快速冲过光带" in prompt
    assert "fast lateral" in prompt
    assert "smooth lateral" not in prompt
    assert "4K, 16:9" not in prompt


def test_shared_video_geometry_supports_portrait_and_explicit_dimensions():
    assert resolve_video_geometry({"aspect_ratio": "9:16"}) == ("9:16", 720, 1280)
    assert resolve_video_geometry({"width": 1080, "height": 1920}) == (
        "9:16",
        1080,
        1920,
    )


def test_legacy_storyboard_prompt_respects_project_visual_style(tmp_path):
    style_path = _write_project_visual_style(
        tmp_path,
        "二维赛璐璐动画，手绘轮廓，冷青色夜景",
    )
    prompt = _build_shot_prompt_legacy(
        {"id": 1, "who": [], "where": "车站", "what": "信使收起雨伞"},
        visual_style_path=str(style_path),
    )

    assert "二维赛璐璐" in prompt
    assert "Photorealistic" not in prompt
    assert "no cartoon" not in prompt


def test_detached_launcher_is_portable_and_reads_the_daemon_pid(tmp_path):
    config = tmp_path / "pipeline.json"
    config.write_text("{}", encoding="utf-8")
    command = detached_pipeline_launch.build_launch_command(
        config,
        project_root=tmp_path,
        python_executable="/opt/honcut/python",
    )

    assert command == [
        "/opt/honcut/python",
        "-u",
        str(tmp_path / "pipeline/scripts/phase_orchestrator.py"),
        "--config",
        str(config),
    ]
    assert "conda" not in command

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"43210")
    finally:
        os.close(write_fd)
    try:
        assert detached_pipeline_launch._read_daemon_pid(read_fd) == 43210
    finally:
        os.close(read_fd)


def test_repository_tests_do_not_reference_a_user_home_fixture():
    forbidden = "/" + "Users/soda/"
    offenders = []
    for path in (ROOT / "pipeline/tests").glob("test_*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_ark_llm_client_does_not_require_ambient_socks_proxy(monkeypatch):
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:65535")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:65535")

    client = create_ark_client()
    try:
        assert str(client.base_url).startswith("https://ark.cn-beijing.volces.com/")
    finally:
        client.close()


def test_new_run_refuses_a_different_run_in_the_same_workspace(tmp_path):
    spec = pipeline_core._project_video_spec("1080p")
    config = {
        "duration": 30,
        "shot_duration": 5,
        "video_provider": "seedance",
        "video_model": "doubao-seedance-2.0-mini",
        "project_video_spec": spec,
    }
    prepare_run_manifest(
        tmp_path,
        source_text="first script",
        resolved_config=config,
        repo_root=ROOT,
        resume=False,
    )

    with pytest.raises(RuntimeError, match="belongs to a different immutable run"):
        prepare_run_manifest(
            tmp_path,
            source_text="second script",
            resolved_config=config,
            repo_root=ROOT,
            resume=False,
        )


def test_layered_checkpoints_are_bound_to_the_full_semantic_input(tmp_path):
    old_events = [{"what": "old event", "sequence_id": "Q1"}]
    new_events = [{"what": "new event", "sequence_id": "Q1"}]
    old_fingerprint = adaptation_engine._layered_input_fingerprint(
        old_events, "CHARACTER_A", 12, 12, 1
    )
    (tmp_path / "beat_skeleton.json").write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "schema": adaptation_engine.LAYERED_CHECKPOINT_SCHEMA,
                    "input_fingerprint": old_fingerprint,
                },
                "strategy": "old",
                "beats": [
                    {
                        "beat_order": 1,
                        "source_events": [1],
                        "action": "keep",
                        "reason": "old",
                        "who": [],
                        "where": "old place",
                        "what": "old event",
                        "suggested_duration": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    current_fingerprint = adaptation_engine._layered_input_fingerprint(
        new_events, "CHARACTER_A", 12, 12, 1
    )

    skeleton, shots = adaptation_engine._load_layered_checkpoints(
        tmp_path,
        new_events,
        1,
        current_fingerprint,
    )

    assert skeleton is None
    assert shots == []


def test_explicit_empty_who_is_a_hard_no_character_storyboard_contract():
    prompt, _beats = build_shot_storyboard_prompt(
        {
            "id": "S01",
            "who": [],
            "where": "empty salt flat",
            "storyboard_beats": [
                {
                    "beat_id": "S01_P01",
                    "duration_s": 5,
                    "action": "cloud shadow crosses the ground",
                }
            ],
        },
        "S01",
        [
            {
                "id": "CHAR_A",
                "name": "Character A",
                "appearance": {"summary": "red coat and black hair"},
            }
        ],
    )

    assert "禁止生成任何角色" in prompt
    assert "Character A" not in prompt
    assert "red coat" not in prompt


def test_generation_route_ignores_action_like_nouns_without_authored_motion():
    assert adaptation_engine.determine_gen_strategy(
        {
            "who": ["CHAR_A"],
            "visual": "CHAR_A studies a blade-shaped museum label",
            "what": "the blade remains under glass",
            "generation_actions": [],
            "action_description": "",
        }
    ) == "phantom"
    assert adaptation_engine.determine_gen_strategy(
        {
            "who": ["CHAR_A"],
            "generation_actions": ["CHAR_A 拔刀并转身"],
        }
    ) == "flf2v"


def test_character_references_refresh_the_canonical_pxx_chain(tmp_path):
    from PIL import Image

    director = tmp_path / "director_storyboard.png"
    character_reference = tmp_path / "characters/CHAR_A/face_closeup.png"
    character_reference.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), "white").save(director)
    Image.new("RGB", (64, 64), "red").save(character_reference)
    director_panels, extraction = materialize_director_panels(
        director,
        [{"position": 1, "shot_id": "S01"}],
        1,
        1,
        tmp_path,
    )
    (tmp_path / "director_storyboard.json").write_text(
        json.dumps({
            "status": "done",
            "panels": director_panels,
            "panel_extraction": extraction,
        }),
        encoding="utf-8",
    )
    calls = []

    class FakeClient:
        model = "fake"

        def image_to_image(self, prompt, ref_image, output_path, size):
            calls.append(ref_image)
            Image.new("RGB", (64, 64), "blue").save(output_path)
            return "https://image.invalid/result.png"

        def text_to_image(self, **_kwargs):
            raise AssertionError("character shot must use image-to-image")

    storyboard = {
        "shots": [
            {
                "id": "S01",
                "who": ["CHAR_A"],
                "where": "LOCATION_A",
                "storyboard_beats": [
                    {
                        "beat_id": "S01_P01",
                        "duration_s": 5,
                        "action": "turns toward the window",
                    }
                ],
            }
        ]
    }
    generate_shot_storyboards(
        tmp_path,
        storyboard,
        [{"id": "CHAR_A", "name": "Character A"}],
        client=FakeClient(),
        director_storyboard_path=director,
    )

    assert len(calls) == 1
    assert str(character_reference) in calls[0]
    assert (tmp_path / "storyboard_images/S01.png").read_bytes() == (
        tmp_path / "storyboard_beats/S01_P01.png"
    ).read_bytes()


def test_semantic_media_ratios_and_cli_resume_defaults(tmp_path):
    assert pipeline_core._project_video_spec("cinematic")["aspect_ratio"] == "21:9"
    assert pipeline_core._project_video_spec("480p")["aspect_ratio"] == "16:9"
    (tmp_path / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "duration": 37,
                    "shot_duration": 5,
                    "chain_mode": True,
                    "dry_run": False,
                    "transition": "cut",
                    "transition_duration": 0.0,
                    "media_profile": "cinematic",
                    "enable_reshoot": False,
                }
            }
        ),
        encoding="utf-8",
    )
    parser = pipeline_runner_cli._build_parser()
    args = parser.parse_args(["--resume", "--output-dir", str(tmp_path)])

    assert args.text is None and args.input is None
    assert pipeline_runner_cli._resolved_run_arguments(args)["duration"] == 37
    assert pipeline_runner_cli._resolved_run_arguments(args)["media_profile"] == (
        "cinematic"
    )


def test_director_failure_removes_a_stale_plan(tmp_path, monkeypatch):
    stale = tmp_path / "director_plan.json"
    stale.write_text('{"scenes":[{"scene_id":"OLD"}]}', encoding="utf-8")
    monkeypatch.setattr(
        director_planner,
        "plan_director",
        lambda *_args, **_kwargs: {"status": "error", "error": "provider down"},
    )

    result = pipeline_core.run_phase1_director("new script", tmp_path, False)

    assert result["status"] == "error"
    assert not stale.exists()


def test_manifest_records_effective_route_and_phase6_requires_phase5(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setenv("VIDEO_PROVIDER_SEEDANCE", "direct")
    monkeypatch.delenv("VIDEO_GENERATION_MODE", raising=False)
    all_phases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5]
    initial = pipeline_core.run_pipeline(
        text="a clean-room test script",
        output_dir=str(tmp_path),
        dry_run=True,
        skip_phase=all_phases,
    )
    manifest = json.loads((tmp_path / "RUN_MANIFEST.json").read_text())

    assert initial["status"] == "completed"
    assert manifest["resolved_config"]["video_generation_mode"] == "direct"

    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"id": "S01", "duration": 5}]}),
        encoding="utf-8",
    )
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "test", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_core,
        "run_phase6",
        lambda *_args, **_kwargs: pytest.fail("Phase 6 must not bypass Phase 5"),
    )
    selected = pipeline_core.run_pipeline(
        text="a clean-room test script",
        output_dir=str(tmp_path),
        dry_run=True,
        skip_phase=[1, 2, 3, 4, 5, 7, 8, 9, 9.5],
    )

    assert selected["status"] == "failed"
    assert "no passing Phase 5 checkpoint" in selected["error"]


def test_phase7_hands_pixel_quality_to_phase8(tmp_path):
    (tmp_path / "storyboard_qa_report.json").write_text(
        json.dumps(
            {
                "gate_passed": True,
                "variation_score": 4.25,
                "slideshow_risk": 0.2,
            }
        ),
        encoding="utf-8",
    )

    result = pipeline_core.run_phase7(tmp_path, False, storyboard_data={"shots": []})

    assert result["status"] == "done"
    assert result["video_quality_owner"] == "phase8"
    assert result["variation_score"] == 4.25
    assert not (tmp_path / "consistency_report.json").exists()


def test_phase8_vlm_compares_video_frames_with_character_reference(
    tmp_path, monkeypatch
):
    from PIL import Image
    from phases.phase8 import frame_analysis
    from clients import ark_multimodal_client

    character_reference = tmp_path / "characters/CHAR_A/face_closeup.png"
    character_reference.parent.mkdir(parents=True)
    frame = tmp_path / "shots/S01/frames/frame_000.jpg"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "red").save(character_reference)
    Image.new("RGB", (32, 32), "blue").save(frame)
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [{"id": "CHAR_A", "name": "Character A"}]}),
        encoding="utf-8",
    )
    captured = {}

    class FakeClient:
        def review(self, paths, prompt):
            captured["paths"] = paths
            captured["prompt"] = prompt
            return '{"verdict":"pass","issues":[],"confidence":1.0}'

    monkeypatch.setattr(ark_multimodal_client, "ArkMultimodalClient", FakeClient)
    reviewer = frame_analysis._automatic_semantic_reviewer(tmp_path)

    result = reviewer(
        [frame],
        {"shot_id": "S01", "who": ["CHAR_A"], "associate_assets": ["char:CHAR_A"]},
    )

    assert result["verdict"] == "pass"
    assert captured["paths"] == [character_reference, frame]
    assert "canonical character references" in captured["prompt"]
