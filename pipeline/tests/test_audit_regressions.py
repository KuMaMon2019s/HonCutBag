"""Regression coverage for the 2026-08-15 P0-P2 production audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
SCRIPTS = ROOT / "pipeline" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import detached_pipeline_launch
import phase_orchestrator
import pipeline_runner as pipeline_runner_cli
from phases import pipeline_core
from phases.phase1 import (
    adaptation_engine,
    character_discoverer,
    director_planner,
    storyboard_generator,
)
from phases.phase1.director_storyboard import materialize_director_panels
from phases.phase1.storyboard_beats import plan_storyboard_beats
from phases.phase1.storyboard_generator import _build_shot_prompt_legacy
from phases.phase2.shot_storyboards import (
    _build_panel_prompt,
    _character_reference_paths,
    build_shot_storyboard_prompt,
    generate_shot_storyboards,
)
from phases.phase3 import character_factory
from phases.phase4.continuity_plan import build_continuity_plan
from phases.phase5 import storyboard_qa_gate
from phases.phase6.video_generator import build_video_prompt
from phases.pipeline_core import _write_project_visual_style
from prompt import event_extractor
from quality.quality_gate import run_quality_check
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


def test_detached_launcher_forwards_resume_phase(tmp_path):
    config = tmp_path / "pipeline.json"
    config.write_text("{}", encoding="utf-8")

    command = detached_pipeline_launch.build_launch_command(
        config,
        project_root=tmp_path,
        python_executable="/opt/honcut/python",
        resume_from="phase5",
    )

    assert command[-2:] == ["--resume-from", "phase5"]


def test_phase_orchestrator_marks_resumed_children(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_stream(cmd, _log_path, _cwd, _env, monitor=None):
        captured["cmd"] = cmd
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(phase_orchestrator, "_stream_subprocess", fake_stream)
    result = phase_orchestrator.run_phase(
        "phase5",
        {
            "input": str(tmp_path / "input.txt"),
            "duration": 60,
            "shot_duration": 10,
            "output_dir": str(tmp_path),
            "media_profile": "720p",
            "transition_duration": 0.5,
            "enable_reshoot": True,
            "_resume": True,
        },
    )

    assert result["exit_code"] == 0
    assert "--resume" in captured["cmd"]


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


def test_phase5_l3_orders_beat_images_without_shot_image_keys(tmp_path):
    from PIL import Image

    from phases.phase5 import storyboard_qa_gate

    storyboard = {
        "shots": [
            {
                "id": 1,
                "storyboard_beats": [
                    {"beat_id": "S01_P01"},
                    {"beat_id": "S01_P02"},
                ],
            }
        ]
    }
    images = {}
    for beat_id in ("S01_P01", "S01_P02"):
        path = tmp_path / f"{beat_id}.png"
        Image.new("RGB", (160, 90), "white").save(path)
        images[beat_id] = path

    class ReviewClient:
        def __init__(self):
            self.calls = []

        def review(self, image_paths, prompt):
            self.calls.append((image_paths, prompt))
            return '{"issues": []}'

    client = ReviewClient()
    grid_path = tmp_path / "storyboard_qa_grid.jpg"
    issues, layer = storyboard_qa_gate.run_l3_review(
        storyboard,
        {"characters": []},
        "cinematic",
        images,
        grid_path,
        client,
    )

    assert issues == []
    assert layer["status"] == "completed"
    assert grid_path.is_file()
    assert client.calls[0][0][-1] == grid_path
    assert len(client.calls[0][0]) == 2
    assert client.calls[0][0][0].name == "storyboard_reference_S01.jpg"
    assert (tmp_path / "storyboard_qa_inputs.json").is_file()
    assert '"S01_P01", "S01_P02"' in client.calls[0][1]


def test_phase5_l3_skips_unmatched_image_ids_instead_of_crashing(tmp_path):
    from PIL import Image

    from phases.phase5 import storyboard_qa_gate

    path = tmp_path / "S99_P01.png"
    Image.new("RGB", (160, 90), "white").save(path)

    class ReviewClient:
        def review(self, _image_paths, _prompt):
            raise AssertionError("unmatched images must not reach the review client")

    issues, layer = storyboard_qa_gate.run_l3_review(
        {"shots": [{"id": 1}]},
        {"characters": []},
        "cinematic",
        {"S99_P01": path},
        tmp_path / "storyboard_qa_grid.jpg",
        ReviewClient(),
    )

    assert issues == []
    assert layer == {
        "status": "skipped",
        "skipped_reason": "no storyboard images match storyboard IDs",
    }


def test_phase1_keeps_qualified_profession_characters():
    stats = {
        "Agent": {"events": [1], "contexts": []},
        "敌方保安": {"events": [1], "contexts": []},
    }

    filtered = character_discoverer._filter_descriptive_phrases(stats)

    assert set(filtered) == {"Agent", "敌方保安"}


def test_phase1_partitions_reused_event_actions_without_replay():
    events = [
        {
            "sequence_id": "SEQ001",
            "action_unit_id": "AU001",
            "micro_actions": ["进入走廊", "警灯亮起", "重力失效", "身体浮起"],
            "start_state": "正常重力",
            "end_state": "完全失重",
        }
    ]
    shots = [
        {"shot_order": 1, "source_events": [1]},
        {"shot_order": 2, "source_events": [1]},
    ]

    adaptation_engine._inherit_event_semantics(shots, events)

    assert shots[0]["micro_actions"] == ["进入走廊", "警灯亮起"]
    assert shots[1]["micro_actions"] == ["重力失效", "身体浮起"]
    assert shots[1]["start_state"] == shots[0]["end_state"]
    assert shots[0]["micro_actions"] + shots[1]["micro_actions"] == events[0][
        "micro_actions"
    ]


def test_phase1_canonical_storyboard_preserves_event_partition_audit(monkeypatch):
    monkeypatch.setattr(
        storyboard_generator,
        "_call_llm",
        lambda *_args, **_kwargs: json.dumps(
            {"prompt": "cinematic corridor", "caption": "继续推进"}
        ),
    )
    shot = {
        "suggested_duration": 5,
        "who": [],
        "where": "运输船走廊",
        "what": "镜头继续推进",
        "visual": "空走廊",
        "source_events": [2],
        "source_event_slices": [
            {
                "event_id": 2,
                "occurrence": 2,
                "occurrence_count": 2,
                "micro_actions": ["穿过舱门"],
            }
        ],
    }

    canonical = storyboard_generator._generate_single_shot(shot, 1, 1)

    assert canonical["source_events"] == [2]
    assert canonical["source_event_slices"] == shot["source_event_slices"]


def test_phase1_beat_planner_never_emits_over_capacity_beats():
    storyboard = {
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 10,
                "micro_actions": [f"action-{index}" for index in range(7)],
            }
        ],
    }

    with pytest.raises(ValueError, match="cannot fit 7 micro-actions"):
        plan_storyboard_beats(storyboard)


def test_phase1_marks_continuous_shot_p01_as_extend():
    storyboard = {
        "video_provider": "seedance",
        "shots": [
            {"id": "S01", "duration": 5, "micro_actions": ["进入"], "boundary_before": "cut"},
            {
                "id": "S02",
                "duration": 5,
                "micro_actions": ["继续前进"],
                "boundary_before": "continuous",
            },
        ],
    }

    plan_storyboard_beats(storyboard)

    assert storyboard["shots"][0]["storyboard_beats"][0]["generation_mode"] == "fresh"
    assert storyboard["shots"][1]["storyboard_beats"][0]["generation_mode"] == "extend"


def test_phase1_detects_explicit_one_take_direction():
    assert pipeline_core._continuity_mode_from_text(
        "科幻动作片，60秒，一镜到底。"
    ) == "one_take"
    assert pipeline_core._continuity_mode_from_text(
        "科幻动作片，采用常规覆盖镜头。"
    ) is None


def test_phase2_uses_face_and_body_references_for_each_character(tmp_path):
    from PIL import Image

    char_dir = tmp_path / "characters/agent"
    char_dir.mkdir(parents=True)
    for name in ("face_closeup.png", "full_body.png", "side.png", "back.png"):
        Image.new("RGB", (64, 64), "white").save(char_dir / name)

    references = _character_reference_paths(
        tmp_path,
        [{"id": "agent", "name": "特工"}],
        ["特工"],
    )

    assert [path.name for path in references] == ["face_closeup.png", "full_body.png"]


def test_phase2_defers_character_storyboards_until_phase3(monkeypatch, tmp_path):
    from PIL import Image

    director = tmp_path / "director_storyboard.png"
    Image.new("RGB", (128, 128), "white").save(director)
    storyboard = {
        "director_storyboard": {"image": director.name, "status": "done"},
        "shots": [
            {
                "id": "S01",
                "who": ["特工"],
                "storyboard_beats": [{"beat_id": "S01_P01", "duration_s": 5}],
            }
        ],
    }
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda *_args, **_kwargs: type("Report", (), {"passed": True, "grade": "A"})(),
    )

    result = pipeline_core.run_phase2(
        storyboard,
        {"characters": [{"id": "agent", "name": "特工"}]},
        tmp_path,
        False,
    )

    assert result["status"] == "done"
    assert result["provider"] == "deferred_to_phase3"
    assert result["storyboard_panels_generated"] == 0


def test_phase3_seedance_contract_and_gate_require_four_views(tmp_path):
    from PIL import Image

    def write_reference(path):
        Image.effect_noise((512, 512), 96).convert("RGB").save(path)

    prompts = character_factory.build_model_reference_prompts(
        "深灰色战术服特工", target_model="seedance"
    )
    assert list(prompts) == ["face_closeup", "full_body", "side", "back"]

    char_dir = tmp_path / "characters/agent"
    char_dir.mkdir(parents=True)
    card = {
        "reference_images": {
            name: f"characters/agent/{name}.png" for name in prompts
        }
    }
    (char_dir / "character_card.json").write_text(
        json.dumps(card), encoding="utf-8"
    )
    write_reference(char_dir / "face_closeup.png")

    incomplete = run_quality_check("phase3", tmp_path)
    assert incomplete.passed is False

    for name in ("full_body", "side", "back"):
        write_reference(char_dir / f"{name}.png")
    complete = run_quality_check("phase3", tmp_path)
    assert complete.passed is True


def test_phase4_preserves_authored_continuous_boundary_with_legacy_fresh_p01():
    storyboard = {
        "shots": [
            {
                "id": "S01",
                "duration": 5,
                "boundary_before": "cut",
                "storyboard_beats": [
                    {
                        "beat_id": "S01_P01",
                        "duration_s": 5,
                        "generation_mode": "fresh",
                    }
                ],
            },
            {
                "id": "S02",
                "duration": 5,
                "boundary_before": "continuous",
                "storyboard_beats": [
                    {
                        "beat_id": "S02_P01",
                        "duration_s": 5,
                        "generation_mode": "fresh",
                    }
                ],
            },
        ]
    }

    plan = build_continuity_plan(storyboard)

    assert plan.shots[1].boundary_before == "continuous"
    assert plan.shots[1].chunks[0].mode == "native_extend"
    assert plan.shots[1].chunks[0].depends_on == "S01_C01"


def test_phase5_null_dialogue_is_not_empty_spoken_content():
    issues, _ = storyboard_qa_gate.run_l1_checks(
        {"shots": [{"id": "S01", "duration": 5, "dialogue": None}]},
        "",
    )

    assert "empty_spoken_content" not in {issue["code"] for issue in issues}


def test_phase5_reset_or_replay_remains_a_severe_l3_issue():
    assert storyboard_qa_gate._calibrate_l3_severity(
        "R3",
        "severe",
        "S02_P01 resets and replays the previous action state",
    ) == "severe"
    assert storyboard_qa_gate._calibrate_l3_severity(
        "R3",
        "severe",
        "S02_P01 动作重置并重复前格状态",
    ) == "severe"


def test_phase5_systemic_moderate_continuity_issue_blocks_generation():
    issues = [
        storyboard_qa_gate._issue(
            "L3",
            "moderate",
            "R3",
            "all continuous shots reset",
            ["S01", "S02", "S03"],
        )
    ]

    assert storyboard_qa_gate.grade_issues(issues) == "C"
    assert storyboard_qa_gate.is_blocking_issue(issues[0]) is True


def test_phase5_separate_moderate_findings_become_systemic_across_shots():
    issues = [
        storyboard_qa_gate._issue(
            "L3", "moderate", "R1", f"identity drift in {shot_id}", [shot_id]
        )
        for shot_id in ("S01", "S02", "S03")
    ]

    assert storyboard_qa_gate.grade_issues(issues) == "D"
    assert storyboard_qa_gate.blocking_issues(issues) == issues


def test_phase5_r1_color_claim_requires_distinct_canonical_panel_evidence():
    valid, details = storyboard_qa_gate._r1_attribute_evidence(
        {
            "mismatch_type": "clothing_color",
            "expected": "深灰色战术服",
            "observed": "深灰色战术服",
            "reference_input_indices": [1],
            "confidence": 0.96,
            "panel_evidence": [
                {"shot_id": "S02_P01", "observed": "深灰色战术服"},
                {"shot_id": "S03_P01", "observed": "深灰色战术服"},
            ],
        },
        ["S02_P01", "S03_P01"],
        2,
    )

    assert valid is False
    assert "expected_equals_observed" in details["evidence_reasons"]

    valid, details = storyboard_qa_gate._r1_attribute_evidence(
        {
            "mismatch_type": "clothing_color",
            "expected": "深灰色战术服",
            "observed": "亮白色太空服",
            "reference_input_indices": [1],
            "confidence": 0.93,
            "panel_evidence": [
                {"shot_id": "S02_P01", "observed": "亮白色太空服"},
                {"shot_id": "S03_P01", "observed": "亮白色太空服"},
            ],
        },
        ["S02_P01", "S03_P01"],
        2,
    )

    assert valid is True
    assert details["evidence_status"] == "validated"

    valid, details = storyboard_qa_gate._r1_attribute_evidence(
        {
            "mismatch_type": "clothing_color",
            "expected": "深黑色战术服",
            "observed": "深色战术服",
            "reference_input_indices": [1],
            "confidence": 0.99,
            "panel_evidence": [
                {"shot_id": "S02_P01", "observed": "深色战术服"},
            ],
            "character_evidence": [{
                "character_id": "agent",
                "reference_input_indices": [1],
                "expected": "深黑色战术服",
                "observed": "深色战术服",
                "storyboard_ids": ["S02_P01"],
            }],
        },
        ["S02_P01"],
        {1: "agent"},
        {"agent": "深灰色战术作战服配黑色作战靴"},
    )

    assert valid is False
    assert "expected_not_exact_canonical_contract" in details["evidence_reasons"]


def test_phase2_panel_prompt_enforces_disarm_and_final_state_contracts():
    prompt = _build_panel_prompt(
        {"who": ["agent", "guard"], "where": "旋转运输船走廊"},
        {
            "beat_id": "S06_P02",
            "generation_mode": "extend",
            "start_state": "双方仍在失重翻滚",
            "action": "Agent 解除保安武器并将保安扔向观察窗",
            "end_state": "Agent 恢复稳定，保安飞向观察窗，画面定格",
        },
        2,
        2,
        [
            {"id": "agent", "appearance": {"summary": "深灰色战术服"}},
            {"id": "guard", "appearance": {"summary": "藏蓝色保安制服"}},
        ],
    )

    assert "项目角色参考与下方角色合同始终优先于上一格" in prompt
    assert "双方同时接触并控制同一武器" in prompt
    assert "动作→对象→道具→结束状态" in prompt
    assert "不得仍停留在搏斗、准备、过渡或前一动作中" in prompt
    assert "不得用运动线否定静止/定格" in prompt


def test_phase5_l3_supplies_canonical_character_images(tmp_path):
    from PIL import Image

    beat_path = tmp_path / "S01_P01.png"
    face_path = tmp_path / "face_closeup.png"
    body_path = tmp_path / "full_body.png"
    for path in (beat_path, face_path, body_path):
        Image.new("RGB", (160, 90), "white").save(path)

    class ReviewClient:
        def __init__(self):
            self.paths = []
            self.prompt = ""

        def review(self, image_paths, prompt):
            self.paths = list(image_paths)
            self.prompt = prompt
            return '{"issues": []}'

    client = ReviewClient()
    storyboard_qa_gate.run_l3_review(
        {
            "shots": [
                {
                    "id": "S01",
                    "storyboard_beats": [{"beat_id": "S01_P01"}],
                }
            ]
        },
        {"characters": [{"id": "agent", "name": "特工"}]},
        "cinematic",
        {"S01_P01": beat_path},
        tmp_path / "grid.jpg",
        client,
        character_reference_images={"agent": [face_path, body_path]},
    )

    assert client.paths[:2] == [face_path, body_path]
    assert client.paths[-2].name == "storyboard_reference_S01.jpg"
    assert client.paths[-1] == tmp_path / "grid.jpg"
    manifest = json.loads((tmp_path / "storyboard_qa_inputs.json").read_text())
    assert [item["kind"] for item in manifest["inputs"]] == [
        "canonical_character_reference",
        "canonical_character_reference",
        "storyboard_shot_board",
        "storyboard_overview_grid",
    ]
    assert manifest["inputs"][0]["path"] == str(face_path)
    assert manifest["inputs"][2]["source_images"][0]["path"] == str(beat_path)
    assert 'mismatch_type="clothing_color"' in client.prompt
    assert "reference_input_indices" in client.prompt
    assert "canonical_contract" in client.prompt
    assert "character_evidence" in client.prompt
    assert "panel_evidence" in client.prompt
    assert "A mutual weapon-disarm action is not satisfied" in client.prompt
    assert "stable/stopped/freeze-frame" in client.prompt


def test_phase5_unverified_systemic_clothing_claim_is_non_blocking(tmp_path):
    from PIL import Image

    images = {}
    for beat_id in ("S02_P01", "S03_P01"):
        path = tmp_path / f"{beat_id}.png"
        Image.new("RGB", (160, 90), "gray").save(path)
        images[beat_id] = path
    reference = tmp_path / "agent_full_body.png"
    Image.new("RGB", (90, 160), "gray").save(reference)

    class ContradictoryReviewClient:
        def review(self, _image_paths, _prompt):
            return json.dumps({
                "issues": [{
                    "red_line": "R1",
                    "severity": "moderate",
                    "mismatch_type": "clothing_color",
                    "shot_ids": ["S02_P01", "S03_P01"],
                    "message": "服装颜色不符",
                    "reference_input_indices": [1],
                    "expected": "深灰色战术服",
                    "observed": "深灰色战术服",
                    "confidence": 0.98,
                    "panel_evidence": [
                        {"shot_id": "S02_P01", "observed": "深灰色战术服"},
                        {"shot_id": "S03_P01", "observed": "深灰色战术服"},
                    ],
                }],
            }, ensure_ascii=False)

    issues, status = storyboard_qa_gate.run_l3_review(
        {
            "shots": [
                {"id": "S02", "storyboard_beats": [{"beat_id": "S02_P01"}]},
                {"id": "S03", "storyboard_beats": [{"beat_id": "S03_P01"}]},
            ],
        },
        {"characters": [{"id": "agent"}]},
        "cinematic",
        images,
        tmp_path / "grid.jpg",
        ContradictoryReviewClient(),
        character_reference_images={"agent": [reference]},
    )

    assert status["status"] == "completed"
    assert issues[0]["severity"] == "minor"
    assert issues[0]["details"]["evidence_status"] == "unverified"
    assert "expected_equals_observed" in issues[0]["details"]["evidence_reasons"]
    assert storyboard_qa_gate.grade_issues(issues) == "A"
