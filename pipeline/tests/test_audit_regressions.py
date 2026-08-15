"""Regression coverage for the 2026-08-15 P0-P2 production audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import detached_pipeline_launch
from phases.phase1.storyboard_beats import plan_storyboard_beats
from phases.phase1.storyboard_generator import _build_shot_prompt_legacy
from phases.phase6.video_generator import build_video_prompt
from phases.pipeline_core import _write_project_visual_style
from prompt import event_extractor
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
