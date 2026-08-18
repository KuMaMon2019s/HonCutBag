"""Regression coverage for the 2026-08-15 P0-P2 production audit."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
SCRIPTS = ROOT / "pipeline" / "scripts"
PHASE5_VARIATION_FIXTURE = (
    ROOT / "pipeline" / "tests" / "fixtures" / "flashmob_60s_phase5_storyboard.json"
)
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
from phases.phase1.director_storyboard import (
    _character_lines,
    build_director_storyboard_prompt,
    materialize_director_panels,
)
from phases.phase1.storyboard_beats import (
    bridge_planning_duration_bounds,
    plan_storyboard_beats,
    secondary_storyboard_contract_errors,
)
from phases.phase1.storyboard_generator import _build_shot_prompt_legacy
from phases.phase2.shot_storyboards import (
    _build_panel_prompt,
    _character_contract,
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
from quality import video_qa
from quality.character_reference_qa import (
    build_character_reference_qa_receipt,
    parse_character_reference_qa,
)
from quality.quality_gate import run_quality_check
from quality.shot_continuity import annotate_boundaries, classify_boundary
from quality.variation_checker import check_scene_variation
from runtime.run_manifest import prepare_run_manifest
from utils.video_capabilities import get_video_capabilities
from utils.video_geometry import resolve_video_geometry
from utils.ark_llm import create_ark_client
from utils.character_body_contracts import (
    ADULT_LEAD_DISCOVERY_INSTRUCTIONS,
    apply_adult_lead_body_contracts,
    character_reference_identity_description,
    character_visual_description,
)


def test_seedance_limits_are_provider_capabilities_not_global_director_rules():
    seedance = get_video_capabilities(provider="seedance")
    generic = get_video_capabilities(provider="kling")

    assert seedance.name == "seedance-2.x"
    assert seedance.action_limit(5) == 1
    assert generic.name == "generic-video"
    assert generic.action_limit(5) == 2

    seedance_board = {
        "video_provider": "seedance",
        "shots": [{"id": 1, "duration": 20, "micro_actions": ["a", "b", "c", "d"]}],
    }
    generic_board = {
        "video_provider": "kling",
        "shots": [{"id": 1, "duration": 20, "micro_actions": ["a", "b", "c", "d"]}],
    }
    plan_storyboard_beats(seedance_board)
    plan_storyboard_beats(generic_board)

    assert seedance_board["shots"][0]["storyboard_beat_count"] == 2
    assert generic_board["shots"][0]["storyboard_beat_count"] == 1
    assert generic_board["shots"][0]["generation_load"]["capability_profile"] == (
        "generic-video"
    )


def test_flf2v_provider_capability_is_separate_from_honcut_bridge_policy():
    seedance = get_video_capabilities(provider="seedance")

    assert seedance.request_duration_bounds("first_last_frame_bridge") == (4, 15)
    assert bridge_planning_duration_bounds(seedance) == (4, 6)
    assert seedance.validate_chunk_durations(
        15,
        15,
        "first_last_frame_bridge",
        resource_id="provider_probe_upper_bound",
    ) == (15, 15)
    with pytest.raises(ValueError, match="outside seedance"):
        seedance.validate_chunk_durations(
            16,
            16,
            "first_last_frame_bridge",
            resource_id="provider_probe_above_upper_bound",
        )


@pytest.mark.parametrize(
    ("duration", "expected_durations"),
    [
        (15, [15]),
        (16, [9, 7]),
        (25, [15, 10]),
        (26, [10, 8, 8]),
        (30, [12, 9, 9]),
    ],
)
def test_secondary_v6_enforces_strategy_ranges_and_primary_assembly_budget(
    duration, expected_durations
):
    storyboard = {
        "video_provider": "seedance",
        "shots": [{"id": "S01", "duration": duration, "micro_actions": ["前进"]}],
    }

    plan_storyboard_beats(storyboard)
    beats = storyboard["shots"][0]["storyboard_beats"]

    assert [beat["duration_s"] for beat in beats] == expected_durations
    assert 8 <= beats[0]["duration_s"] <= 15
    assert all(6 <= beat["duration_s"] <= 10 for beat in beats[1:])
    assert sum(beat["duration_s"] for beat in beats) == duration


def test_secondary_v6_keeps_bridge_time_outside_primary_content_capacity():
    storyboard = {
        "video_provider": "seedance",
        "delivery_target_duration": 30,
        "pre_edit_duration_ratio_limit": 1.3,
        "shots": [
            {
                "id": "S01",
                "duration": 15,
                "where": "旋转走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent抓住扶手"],
                "end_state": "Agent抓住扶手稳定身体",
            },
            {
                "id": "S02",
                "duration": 15,
                "where": "旋转走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent穿过舱门"],
                "boundary_before": "continuous",
                "start_state": "Agent抓住扶手稳定身体",
            },
        ],
    }

    plan_storyboard_beats(storyboard)

    first = storyboard["shots"][0]
    assert [beat["generation_mode"] for beat in first["storyboard_beats"]] == [
        "multi_image",
    ]
    assert [beat["duration_s"] for beat in first["storyboard_beats"]] == [15]
    assert first["secondary_storyboard_planning"]["content_duration_s"] == 15
    assert first["secondary_storyboard_planning"]["bridge_duration_s"] == 4
    assert first["secondary_storyboard_planning"][
        "first_last_frame_bridge_duration_range_s"
    ] == [4.0, 6.0]
    assert first["secondary_storyboard_planning"][
        "first_last_frame_bridge_policy_duration_range_s"
    ] == [4.0, 6.0]
    assert first["secondary_storyboard_planning"][
        "first_last_frame_bridge_provider_duration_range_s"
    ] == [4, 15]
    assert storyboard["primary_shot_bridges"][0]["generation_phase"] == (
        "post_primary_shots"
    )
    bridge = storyboard["primary_shot_bridges"][0]
    assert bridge["generation_duration_s"] == 4
    assert bridge["visible_duration_s"] == 4
    assert bridge["source_handle_s"] == 2.0
    assert bridge["target_handle_s"] == 2.0
    assert bridge["timeline_insertion_policy"] == "replace_boundary_handles"
    assert first["storyboard_beats"][-1]["outgoing_bridge_handle_s"] == 2.0
    assert storyboard["shots"][1]["storyboard_beats"][0][
        "incoming_bridge_handle_s"
    ] == 2.0
    assert storyboard["material_budget"] == {
        "schema": "honcut.material-budget.v1",
        "policy": "primary_ratio_cap_plus_explicit_bridge_overhead",
        "timeline_policy": "replace_boundary_handles",
        "delivery_target_duration_s": 30.0,
        "pre_edit_duration_ratio_limit": 1.3,
        "primary_material_duration_s": 30.0,
        "primary_material_limit_s": 39.0,
        "primary_material_within_limit": True,
        "bridge_count": 1,
        "bridge_generation_duration_s": 4.0,
        "bridge_visible_duration_s": 4.0,
        "bridge_replaced_handle_duration_s": 4.0,
        "total_generated_duration_s": 34.0,
        "projected_pre_edit_timeline_duration_s": 30.0,
        "bridge_overhead_is_additive": True,
        "primary_secondary_double_count_forbidden": True,
    }

    first["secondary_storyboard_planning"][
        "first_last_frame_bridge_provider_duration_range_s"
    ] = [4.0, 6.0]
    errors = secondary_storyboard_contract_errors(storyboard, 0)
    assert any(
        issue["code"] == "secondary_storyboard_planning_metadata_invalid"
        and issue["details"]["field"]
        == "first_last_frame_bridge_provider_duration_range_s"
        for issue in errors
    )


def test_secondary_v4_rejects_false_continuity_and_fractional_seedance_duration():
    boundary, reason = classify_boundary(
        {"where": "走廊", "who": ["Agent"]},
        {
            "where": "机库",
            "who": ["敌人"],
            "boundary_before": "continuous",
        },
        index=2,
    )
    assert boundary == "cut"
    assert "location" in reason

    with pytest.raises(ValueError, match="duration quantum"):
        plan_storyboard_beats({
            "video_provider": "seedance",
            "shots": [{"id": "S01", "duration": 10.5, "micro_actions": ["挥拳"]}],
        })


def test_generic_action_unit_capacity_does_not_inherit_seedance_split():
    storyboard = {
        "video_provider": "kling",
        "shots": [{
            "id": "S01",
            "duration": 8,
            "micro_actions": ["挥拳"],
            "source_action_unit_ids": ["U1", "U2"],
        }],
    }

    plan_storyboard_beats(storyboard)

    assert [
        beat["generation_mode"] for beat in storyboard["shots"][0]["storyboard_beats"]
    ] == ["multi_image"]


def test_secondary_v6_cannot_downgrade_or_use_storyboard_proxy_for_bridge():
    storyboard = {
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 15,
                "where": "走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent稳定姿态"],
                "end_state": "Agent扶住门框",
            },
            {
                "id": "S02",
                "duration": 15,
                "where": "走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent进入下一舱"],
                "boundary_before": "continuous",
                "start_state": "Agent扶住门框",
            },
        ],
    }
    plan_storyboard_beats(storyboard)

    downgraded = json.loads(json.dumps(storyboard))
    downgraded["shots"][0]["storyboard_beats"][0]["generation_mode"] = "fresh"
    downgrade_codes = {
        issue["code"]
        for issue in storyboard_qa_gate.run_generation_capacity_checks(downgraded)
    }
    assert "secondary_storyboard_mode_invalid" in downgrade_codes
    with pytest.raises(ValueError, match="invalid secondary storyboard contract"):
        build_continuity_plan(downgraded)

    invented = json.loads(json.dumps(storyboard))
    bridge = invented["primary_shot_bridges"][0]
    bridge["last_frame_source"] = "target_storyboard_image"
    invented_codes = {
        issue["code"]
        for issue in storyboard_qa_gate.run_generation_capacity_checks(invented)
    }
    assert "secondary_storyboard_bridge_invalid" in invented_codes
    with pytest.raises(ValueError, match="invalid secondary storyboard contract"):
        build_continuity_plan(invented)


def test_action_aware_adaptation_splits_paid_probe_before_secondary_planning():
    events = [
        {
            "sequence_id": "SEQ001",
            "continuity_before": "cut",
            "micro_actions": ["挥拳", "肘击", "膝击", "缴械", "抓门框", "稳定"],
        },
        {
            "sequence_id": "SEQ001",
            "continuity_before": "continuous",
            "micro_actions": ["穿门", "避让工具箱", "推向观察窗", "稳定"],
        },
    ]
    assert adaptation_engine.estimate_action_aware_shot_count(events, 45, 12) == 3

    shots = [
        {
            "id": "S01",
            "where": "旋转走廊",
            "who": ["Agent"],
            "micro_actions": events[0]["micro_actions"][:3],
            "boundary_before": "cut",
        },
        {
            "id": "S02",
            "where": "旋转走廊",
            "who": ["Agent"],
            "micro_actions": events[0]["micro_actions"][3:],
            "boundary_before": "continuous",
        },
        {
            "id": "S03",
            "where": "旋转走廊",
            "who": ["Agent"],
            "micro_actions": events[1]["micro_actions"],
            "boundary_before": "continuous",
        },
    ]
    annotate_boundaries(shots)
    adaptation_engine.normalize_shot_durations(shots, 45)
    assert [shot["suggested_duration"] for shot in shots] == [15, 15, 15]

    storyboard = {"video_provider": "seedance", "shots": shots}
    plan_storyboard_beats(storyboard)
    assert [
        [beat["generation_mode"] for beat in shot["storyboard_beats"]]
        for shot in shots
    ] == [
        ["multi_image", "tail_video_extend"],
        ["multi_image", "tail_video_extend"],
        ["multi_image", "tail_video_extend"],
    ]
    assert [bridge["bridge_id"] for bridge in storyboard["primary_shot_bridges"]] == [
        "S01__S02", "S02__S03",
    ]


def test_capacity_repair_deterministically_fixes_paid_model_event_mapping():
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "scene_setup",
            "who": ["Agent"],
            "where": "旋转走廊",
            "what": "建立失重走廊",
            "visual": "红色警示灯闪烁，Agent进入走廊",
            "micro_actions": [],
        },
        {
            "sequence_id": "SEQ001",
            "action_unit_id": "AU001",
            "who": ["Agent", "敌方保安"],
            "where": "旋转走廊",
            "what": "Agent完成连续格斗并抓住门框",
            "visual": "Agent依次挥拳、肘击、膝击、缴械并抓住门框",
            "micro_actions": ["挥拳", "肘击", "膝击", "缴械", "抓住门框"],
        },
        {
            "sequence_id": "SEQ001",
            "action_unit_id": "AU002",
            "who": ["Agent", "敌方保安"],
            "where": "旋转走廊",
            "what": "Agent穿门并把保安推向观察窗",
            "visual": "Agent穿门、避让、推敌并稳定",
            "micro_actions": ["穿门", "避让", "推向观察窗", "稳定"],
        },
    ]
    model_beats = [
        {"beat_order": 1, "source_events": [1], "action": "keep", "reason": "", "who": ["Agent"], "where": "旋转走廊", "what": "建立", "visual": "建立", "suggested_duration": 8},
        {"beat_order": 2, "source_events": [2], "action": "keep", "reason": "", "who": ["Agent"], "where": "旋转走廊", "what": "格斗", "visual": "格斗", "suggested_duration": 8},
        {"beat_order": 3, "source_events": [3], "action": "keep", "reason": "", "who": ["Agent"], "where": "旋转走廊", "what": "穿门", "visual": "穿门", "suggested_duration": 8},
    ]

    repaired = adaptation_engine._repair_beat_action_capacity(model_beats, events)

    assert [beat["source_events"] for beat in repaired] == [[1], [2], [3]]
    assert "capacity_repair" not in repaired[0]
    adaptation_engine._validate_beat_action_capacity(repaired, events)


def test_source_event_identity_overrides_llm_character_synonyms():
    events = [{
        "who": ["Agent", "敌方保安"],
        "where": "旋转走廊",
        "what": "Agent抓住门框",
        "micro_actions": ["Agent抓住门框"],
        "sequence_id": "SEQ001",
    }]
    shots = [{
        "source_events": [1],
        "who": ["特工", "敌方保安"],
        "where": "旋转走廊",
        "what": "特工抓住门框",
    }]

    adaptation_engine._inherit_event_semantics(shots, events)

    assert shots[0]["who"] == ["Agent", "敌方保安"]


def test_source_event_identity_resolves_qualified_alias_to_character_asset():
    events = [{
        "who": ["身穿深灰色战术服的Agent", "敌方保安"],
        "where": "旋转走廊",
        "what": "Agent与敌方保安搏斗",
        "micro_actions": ["Agent挥拳"],
        "sequence_id": "SEQ001",
    }]
    shots = [{
        "source_events": [1],
        "who": ["Agent", "敌方保安"],
        "where": "旋转走廊",
        "what": "双方搏斗",
    }]
    characters = [
        {"id": "agent", "name": "特工", "aliases": ["Agent", "他"]},
        {"id": "security_guard", "name": "敌方保安", "aliases": ["保安"]},
    ]

    adaptation_engine._inherit_event_semantics(shots, events, characters)

    assert shots[0]["who"] == ["特工", "敌方保安"]
    assert shots[0]["source_character_mentions"] == [
        "身穿深灰色战术服的Agent", "敌方保安",
    ]


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


def test_event_extractor_allows_healthy_long_streams_but_keeps_idle_guard(
    monkeypatch,
):
    captured: dict[str, object] = {}
    client = object()

    monkeypatch.setattr(event_extractor, "_get_client", lambda: client)

    def fake_call_llm_stream(**kwargs):
        captured.update(kwargs)
        return "[]"

    monkeypatch.setattr(event_extractor, "call_llm_stream", fake_call_llm_stream)

    assert event_extractor._call_llm("提取事件") == "[]"
    assert captured["_client"] is client
    assert captured["wall_timeout"] == event_extractor.LLM_TIMEOUT == 900
    assert captured["idle_timeout"] == event_extractor.LLM_IDLE_TIMEOUT == 75
    assert event_extractor.LLM_TIMEOUT > event_extractor.LLM_IDLE_TIMEOUT


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
    assert "[storyboard-motion-notation]" in prompt
    assert "主体箭头控制主体的运动方向、路径和速度趋势" in prompt
    assert "最终视频的任何一帧都不得出现或残留箭头" in prompt


def test_video_prompt_preserves_full_subject_action_ledger_and_rejects_background_only_motion():
    actions = ["演员屈膝下沉并抬起双臂", "演员转移重心后跨步旋转"]

    prompt = build_video_prompt(
        {
            "id": 2,
            "duration": 8,
            "where": "排练厅",
            "who": ["CHAR_A"],
            "generation_actions": actions,
            "action_description": "演员完成动作",
        },
        {"characters": [{"id": "CHAR_A", "name": "演员"}]},
        {},
        "seedance",
    )

    assert f"主体动作逐项硬合同：{' → '.join(actions)}" in prompt
    assert "躯干、四肢、关节、重心及相关道具必须按动作连续产生可见变化" in prompt
    assert "不得只让背景人群、车辆、光影、粒子、布料、头发或摄影机产生运动" in prompt
    assert "action_description" not in prompt


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


def test_phase1_production_prompts_are_symbolic_and_identity_is_structured():
    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "pipeline/src/phases/phase1").glob("*.py")
    )
    forbidden_story_terms = (
        "数百名机械居民",
        "企业执法机械体群",
        "林夏",
        "便利店门口",
        "char:lin_xia",
        "char:shen_yu",
    )

    assert not any(term in production_source for term in forbidden_story_terms)
    assert "3-6 个视觉特征" not in adaptation_engine.USER_PROMPT_TEMPLATE
    assert "3–6 个视觉特征" not in adaptation_engine.USER_PROMPT_TEMPLATE
    assert "身份只通过 who 与 associate_assets 结构化绑定" in (
        adaptation_engine.USER_PROMPT_TEMPLATE
    )
    assert "who=[] 时 visual 和 associate_assets 都不得引入任何角色" in (
        adaptation_engine.USER_PROMPT_TEMPLATE
    )


def test_ci_fails_when_codecov_upload_fails():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "fail_ci_if_error: true" in workflow
    assert "fail_ci_if_error: false" not in workflow


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


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("visual-style.md", b"old style"),
        ("music.mp3", b"old music"),
        ("audio/old.wav", b"old audio"),
        ("audio_layer/continuous_bgm.m4a", b"old mix"),
    ],
)
def test_new_run_refuses_stale_style_and_audio_assets(
    tmp_path, relative_path, content
):
    stale = tmp_path / relative_path
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(content)
    spec = pipeline_core._project_video_spec("1080p")

    with pytest.raises(RuntimeError, match="unowned pipeline artifacts"):
        prepare_run_manifest(
            tmp_path,
            source_text="new script",
            resolved_config={
                "duration": 30,
                "shot_duration": 5,
                "video_provider": "seedance",
                "video_model": "doubao-seedance-2.0-mini",
                "project_video_spec": spec,
            },
            repo_root=ROOT,
            resume=False,
        )


def test_phase6_stale_output_requires_current_succeeded_ledger_receipt(tmp_path):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"old output from another execution")

    assert pipeline_core._phase6_output_failure(
        "S01", output, None, None, validate_video=lambda _path: True
    ) == "no successful current-input generation receipt"

    receipt = {"input_fingerprint": "current"}
    task = SimpleNamespace(
        status="succeeded",
        resource_id="S01",
        payload={"input_fingerprint": "current"},
        outcome={"output_sha256": pipeline_core._file_sha256(output)},
    )
    assert pipeline_core._phase6_output_failure(
        "S01", output, receipt, task, validate_video=lambda _path: True
    ) is None

    output.write_bytes(b"stale file replaced the ledger output")
    assert pipeline_core._phase6_output_failure(
        "S01", output, receipt, task, validate_video=lambda _path: True
    ) == "output.mp4 hash does not match generation ledger"


def test_phase6_current_ledger_receipt_accepts_a_real_minimal_video(tmp_path):
    output = tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "color=c=blue:s=64x64:d=0.4:r=12",
            "-pix_fmt", "yuv420p", str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = {"input_fingerprint": "current"}
    task = SimpleNamespace(
        status="succeeded",
        resource_id="S01",
        payload={"input_fingerprint": "current"},
        outcome={"output_sha256": pipeline_core._file_sha256(output)},
    )

    assert pipeline_core._phase6_output_failure(
        "S01", output, receipt, task
    ) is None


def test_final_vlm_reviews_every_shot_without_a_global_twelve_frame_cap():
    frames = [
        video_qa.FrameSample(
            path=f"/tmp/{shot_id}_{suffix}.jpg",
            timestamp=float(index),
            label=f"{shot_id}_{suffix}",
        )
        for index, shot_id in enumerate(
            f"S{number:02d}" for number in range(1, 21)
        )
        for suffix in ("first", "mid", "last")
    ]
    storyboard = {
        "shots": [
            {
                "shot_id": f"S{number:02d}",
                "who": ["CHAR_A"] if number == 3 else [],
                "shot_intent": "action" if number == 4 else "establishing",
            }
            for number in range(1, 21)
        ]
    }
    calls = []

    class FakeClient:
        def review(self, paths, prompt):
            calls.append((paths, prompt))
            return '{"verdict":"pass","issues":[],"confidence":0.99}'

    result = video_qa._vlm_semantic_check(FakeClient(), frames, storyboard)

    assert result["status"] == "completed"
    assert result["review_batches"] >= 2
    assert len(result["sampled_frames"]) > 12
    assert set(result["covered_shots"]) == {
        f"S{number:02d}" for number in range(1, 21)
    }
    assert all(
        f"S{number:02d}_mid" in result["sampled_frames"]
        for number in range(1, 21)
    )
    assert {"S03_first", "S03_mid", "S03_last"}.issubset(
        result["sampled_frames"]
    )
    assert {"S04_first", "S04_mid", "S04_last"}.issubset(
        result["sampled_frames"]
    )
    assert all(len(paths) <= 12 for paths, _prompt in calls)


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
                    "no_real_person": True,
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
    assert pipeline_runner_cli._resolved_run_arguments(args)["no_real_person"] is True


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
        no_real_person=True,
        skip_phase=all_phases,
    )
    manifest = json.loads((tmp_path / "RUN_MANIFEST.json").read_text())

    assert initial["status"] == "completed"
    assert manifest["resolved_config"]["video_generation_mode"] == "direct"
    assert manifest["resolved_config"]["no_real_person"] is True

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
        no_real_person=True,
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


def test_phase1_normalizes_visual_description_around_explicit_agent_name():
    stats = {
        "身穿深灰色战术服的Agent": {
            "events": [1, 2],
            "contexts": ["事件1", "事件2"],
            "dialogue_count": 0,
        },
        "敌方保安": {
            "events": [2],
            "contexts": ["事件2"],
            "dialogue_count": 0,
        },
    }

    filtered = character_discoverer._filter_descriptive_phrases(stats)

    assert set(filtered) == {"Agent", "敌方保安"}
    assert filtered["Agent"]["events"] == [1, 2]
    assert filtered["Agent"]["source_aliases"] == ["身穿深灰色战术服的Agent"]


def test_phase1_normalizes_qualified_identities_without_script_language_bias():
    stats = {
        "头戴护目镜的Mira Chen": {"events": [1], "contexts": []},
        "身披红色斗篷的机械师": {"events": [2], "contexts": []},
        "受伤的林岚": {"events": [3], "contexts": []},
        "未来战士": {"events": [4], "contexts": []},
    }

    filtered = character_discoverer._filter_descriptive_phrases(stats)

    assert set(filtered) == {"Mira Chen", "机械师", "林岚", "未来战士"}
    assert filtered["Mira Chen"]["source_aliases"] == ["头戴护目镜的Mira Chen"]
    assert filtered["机械师"]["source_aliases"] == ["身披红色斗篷的机械师"]
    assert filtered["林岚"]["source_aliases"] == ["受伤的林岚"]


def test_phase1_does_not_promote_qualified_objects_or_merge_relational_roles():
    stats = {
        "穿着漂亮的红裙": {"events": [1], "contexts": []},
        "金色夕阳下的无边云海": {"events": [1], "contexts": []},
        "小明的父亲": {"events": [2], "contexts": []},
    }

    filtered = character_discoverer._filter_descriptive_phrases(stats)

    assert set(filtered) == {"小明的父亲"}


def test_phase1_character_filter_does_not_confuse_role_vocabulary_with_objects():
    for name in ("持枪的枪手", "背剑的剑客", "职业车手", "宇航员", "林海"):
        assert character_discoverer._is_human_character(name)

    for name in ("两把金属刀具", "断裂的车门", "无边云海", "冷空气"):
        assert not character_discoverer._is_human_character(name)


def test_phase1_source_identity_evidence_aggregates_aliases_and_events():
    characters = [{
        "id": "mira_chen",
        "name": "米拉",
        "aliases": ["Mira Chen"],
        "role": "protagonist",
    }]
    stats = {
        "Mira Chen": {
            "events": [1, 2],
            "contexts": [],
            "source_aliases": ["头戴护目镜的Mira Chen"],
        },
        "米拉": {"events": [3], "contexts": []},
    }

    character_discoverer._attach_source_identity_evidence(characters, stats)

    assert characters[0]["first_appearance"] == 1
    assert characters[0]["appearance_count"] == 3
    assert characters[0]["aliases"] == [
        "Mira Chen",
        "头戴护目镜的Mira Chen",
    ]


def test_character_identity_resolution_is_token_safe_and_unambiguous():
    from utils.character_identity import resolve_character_name

    characters = [
        {"id": "ann", "name": "安", "aliases": ["Ann"]},
        {"id": "mira", "name": "米拉", "aliases": ["Captain Mira"]},
    ]

    assert resolve_character_name("受伤的 Ann", characters) == "安"
    assert resolve_character_name("injured Captain Mira", characters) == "米拉"
    assert resolve_character_name("Joanne", characters) is None
    assert resolve_character_name("Ann007", characters) is None


def test_event_extractor_contract_keeps_transient_descriptors_out_of_who():
    assert "稳定身份标签" in event_extractor.USER_PROMPT_TEMPLATE
    assert "服装、年龄、伤势、动作、站位和地点修饰不得进入 who" in (
        event_extractor.USER_PROMPT_TEMPLATE
    )
    assert "同一人物必须沿用" in event_extractor.GENERAL_PROSE_CONTRACT
    assert "同一人物必须沿用" in event_extractor.ACTION_SCREENPLAY_CONTRACT


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
                "duration": 20,
                "micro_actions": [f"action-{index}" for index in range(7)],
            }
        ],
    }

    with pytest.raises(ValueError, match="cannot fit 7 micro-actions"):
        plan_storyboard_beats(storyboard)


def test_phase1_starts_each_primary_shot_p01_with_multi_image_generation():
    storyboard = {
        "video_provider": "seedance",
        "shots": [
            {"id": "S01", "duration": 15, "micro_actions": ["进入"], "boundary_before": "cut"},
            {
                "id": "S02",
                "duration": 15,
                "micro_actions": ["继续前进"],
                "boundary_before": "continuous",
            },
        ],
    }

    plan_storyboard_beats(storyboard)

    assert storyboard["shots"][0]["storyboard_beats"][0]["generation_mode"] == (
        "multi_image"
    )
    assert storyboard["shots"][1]["storyboard_beats"][0]["generation_mode"] == (
        "multi_image"
    )


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
    missing_semantic_qa = run_quality_check("phase3", tmp_path)
    assert missing_semantic_qa.passed is False
    assert any(
        issue.rule == "character_reference_qa_passed"
        for issue in missing_semantic_qa.issues
    )

    view_paths = {name: char_dir / f"{name}.png" for name in prompts}
    receipt = build_character_reference_qa_receipt(
        char_id="agent",
        view_paths=view_paths,
        attempts=[{"attempt": 1, "passed": True, "failed_views": []}],
    )
    qa_path = char_dir / "character_reference_qa.json"
    qa_path.write_text(json.dumps(receipt), encoding="utf-8")
    card["reference_qa_report"] = "characters/agent/character_reference_qa.json"
    (char_dir / "character_card.json").write_text(
        json.dumps(card), encoding="utf-8"
    )

    complete = run_quality_check("phase3", tmp_path)
    assert complete.passed is True

    Image.effect_noise((512, 512), 24).convert("RGB").save(char_dir / "back.png")
    stale_receipt = run_quality_check("phase3", tmp_path)
    assert stale_receipt.passed is False
    assert any(
        issue.rule == "character_reference_qa_passed"
        for issue in stale_receipt.issues
    )


def test_phase3_reference_generation_uses_face_only_as_identity_anchor(
    monkeypatch, tmp_path
):
    from PIL import Image

    calls = []

    class ImageClient:
        def __init__(self, model):
            self.model = model

        @staticmethod
        def _write(path):
            Image.effect_noise((512, 512), 96).convert("RGB").save(path)

        def text_to_image(self, *, prompt, output_path, size):
            calls.append(("text", prompt, None, Path(output_path).name, size))
            self._write(output_path)

        def image_to_image(self, *, prompt, ref_image, output_path, size):
            calls.append(("image", prompt, ref_image, Path(output_path).name, size))
            self._write(output_path)

    class Reviewer:
        def __init__(self):
            self.paths = []
            self.prompt = ""
            self.calls = 0

        def review(self, paths, prompt):
            self.calls += 1
            self.paths = list(paths)
            self.prompt = prompt
            common = {
                "passed": True,
                "view_match": True,
                "framing_match": True,
                "neutral_pose": True,
                "plain_background": True,
                "single_character": True,
                "face_visible": True,
                "both_eyes_visible": True,
                "issues": [],
            }
            back = {
                **common,
                "face_visible": self.calls == 1,
                "both_eyes_visible": False,
                "passed": self.calls > 1,
                "view_match": self.calls > 1,
                "issues": (
                    ["front-facing face is visible; back view is absent"]
                    if self.calls == 1
                    else []
                ),
            }
            return json.dumps({
                "views": {
                    "face_closeup": common,
                    "full_body": common,
                    "side": {**common, "both_eyes_visible": False},
                    "back": back,
                },
                "cross_view": {
                    "passed": True,
                    "identity_consistent": True,
                    "outfit_consistent": True,
                    "body_proportions_consistent": True,
                    "issues": [],
                },
                "failed_views": ["back"] if self.calls == 1 else [],
                "summary": (
                    "back is front-facing" if self.calls == 1 else "all contracts pass"
                ),
            })

    monkeypatch.setattr(character_factory, "SeedreamClient", ImageClient)
    reviewer = Reviewer()
    result = character_factory.generate_character(
        char_id="agent",
        name="Agent",
        description="adult woman; black ponytail; white shirt; gray trousers",
        output_dir=str(tmp_path),
        style="7-Eleven street dance, moving handheld camera, cheering crowd",
        review_client=reviewer,
        view_qa_max_retries=1,
    )

    assert [path.name for path in reviewer.paths] == [
        "face_closeup.png",
        "full_body.png",
        "side.png",
        "back.png",
    ]
    assert calls[0][0] == "text"
    assert reviewer.calls == 2
    assert len(calls) == 5
    assert all(call[0] == "image" for call in calls[1:])
    assert all(str(call[2]).endswith("face_closeup.png") for call in calls[1:])
    assert all("7-Eleven" not in call[1] for call in calls)
    assert all("cheering crowd" not in call[1] for call in calls)
    assert "strict 90-degree left side" in calls[2][1].lower()
    assert "face, eyes, nose, mouth" in calls[3][1].lower()
    assert "previous back failed blocking view qa" in calls[4][1].lower()
    assert "front-facing face is visible" in calls[4][1].lower()
    assert (tmp_path / "characters/agent/reference_qa_attempts/attempt_01/back.png").is_file()
    assert Path(result["card"]).is_file()
    assert run_quality_check("phase3", tmp_path).passed is True


def _adult_lead_character(name, gender, age_range, role="protagonist"):
    return {
        "id": name.lower(),
        "name": name,
        "aliases": [],
        "role": role,
        "appearance": {
            "gender": gender,
            "age_range": age_range,
            "hair": "black hair",
            "face": "oval face",
            "clothing": "dark jacket",
            "summary": f"{name}, black hair, dark jacket",
        },
    }


def test_adult_lead_body_contracts_are_exact_and_scoped():
    male = _adult_lead_character("LIN", "male", "25-30")
    female = _adult_lead_character("SU", "female", "22-28")
    second_male = _adult_lead_character("ZHOU", "male", "30-35")
    supporting = _adult_lead_character("ASSISTANT", "female", "25-30", "supporting")
    child = _adult_lead_character("CHILD", "female", "12-15")

    apply_adult_lead_body_contracts([male, female, second_male, supporting, child])

    male_contract = male["appearance"]["body_contract"]
    assert male["appearance"]["height"] == "182cm"
    assert male_contract["profile"] == "adult_male_lead"
    assert male_contract["height_cm"] == 182
    assert male_contract["head_to_body_ratio"] == 7.8
    assert male_contract["build"] == "lean athletic"
    assert male_contract["shoulders"] == "moderately broad shoulders"
    assert male_contract["leg_proportion"] == "slightly long legs"
    assert male_contract["body_fat"] == "low-to-normal body fat"
    assert male_contract["posture"] == "upright, confident"
    assert male_contract["schema_version"] == 2
    human_contract = male_contract["human_proportion_constraints"]
    assert human_contract["head_to_body_ratio_range"] == [7.6, 8.0]
    assert human_contract["max_head_width_to_shoulder_width"] == 0.43
    assert human_contract["extremity_scale"] == "hands and feet proportionate to height"

    female_contract = female["appearance"]["body_contract"]
    assert female["appearance"]["height"] == "166cm"
    assert female_contract["head_to_body_ratio"] == 7.6
    assert female_contract["build"] == "slender balanced"
    assert female_contract["shoulders_and_hips"] == "natural proportional shoulders and hips"
    assert female_contract["waistline"] == "naturally defined waist"
    assert female_contract["body_fat"] == "healthy slim"
    assert female_contract["forbidden"] == [
        "oversized head",
        "extremely tiny waist",
        "exaggerated curves",
    ]
    assert "posture" not in female_contract
    assert "body_contract" not in second_male["appearance"]
    assert "body_contract" not in supporting["appearance"]
    assert "body_contract" not in child["appearance"]


def test_phase3_identity_description_excludes_story_action_and_location():
    character = _adult_lead_character("SU", "female", "22-28")
    character["appearance"].update({
        "hair": "black high ponytail",
        "face": "oval face and almond eyes",
        "clothing": "white crop top, black jacket, gray trousers",
        "summary": "street dancer performs a powerful routine outside a convenience store",
        "distinguishing": "keeps dancing and making eye contact with the moving camera",
    })
    apply_adult_lead_body_contracts([character])

    description = character_reference_identity_description(character)

    assert "black high ponytail" in description
    assert "white crop top" in description
    assert "height exactly 166 cm" in description
    assert "dancer" not in description
    assert "dancing" not in description
    assert "convenience store" not in description
    assert "moving camera" not in description


def test_character_reference_qa_recomputes_wrong_view_verdict():
    passing_view = {
        "passed": True,
        "view_match": True,
        "framing_match": True,
        "neutral_pose": True,
        "plain_background": True,
        "single_character": True,
        "face_visible": True,
        "both_eyes_visible": False,
        "issues": [],
    }
    payload = {
        "views": {
            "face_closeup": {**passing_view, "both_eyes_visible": True},
            "full_body": {**passing_view, "both_eyes_visible": True},
            "side": passing_view,
            # A model-level passed=true cannot override visible face evidence.
            "back": {**passing_view, "face_visible": True},
        },
        "cross_view": {
            "passed": True,
            "identity_consistent": True,
            "outfit_consistent": True,
            "body_proportions_consistent": True,
            "issues": [],
        },
        "failed_views": [],
        "summary": "back is actually front-facing",
    }

    review = parse_character_reference_qa(json.dumps(payload))

    assert review["passed"] is False
    assert review["failed_views"] == ["back"]
    assert review["views"]["back"]["passed"] is False


def test_character_discovery_body_contract_is_prompted_and_normalized(monkeypatch):
    response = json.dumps(
        [_adult_lead_character("LIN", "male", "25-30")],
        ensure_ascii=False,
    )
    captured = {}

    def fake_call(prompt):
        captured["prompt"] = prompt
        return response

    monkeypatch.setattr(character_discoverer, "_call_llm", fake_call)
    result = character_discoverer.discover_characters([{
        "id": 1,
        "who": ["LIN"],
        "what": "LIN walks into frame",
        "visual": "full-body view",
    }])

    discovered = result["characters"][0]
    assert discovered["appearance"]["body_contract"]["height_cm"] == 182
    assert "head_to_body_ratio=7.8" in captured["prompt"]
    assert "head_to_body_ratio=7.6" in captured["prompt"]
    assert "头宽不得超过肩宽的 43%" in captured["prompt"]
    assert ADULT_LEAD_DISCOVERY_INSTRUCTIONS in character_discoverer.SYSTEM_PROMPT
    assert character_discoverer.CHARACTER_CONTEXT_SCHEMA_VERSION == 6
    assert "bodybuilder physique" in discovered["negative_guardrails"]
    assert "Body-proportion lock" in discovered["prompt_definition"]


def test_body_contract_reaches_character_storyboard_and_video_prompts():
    male = _adult_lead_character("LIN", "male", "25-30")
    apply_adult_lead_body_contracts([male])
    description = character_visual_description(male)

    assert description.startswith("Body-proportion lock:")
    assert "height exactly 182 cm" in description
    assert "exactly 7.8 heads tall" in description
    assert "moderately broad shoulders" in description
    assert "adult head-to-body ratio stays within 7.6–8.0" in description
    assert "head width never exceeds 43% of shoulder width" in description
    assert "hands and feet proportionate to height" in description
    assert "same head size and body proportions" in description
    assert "Do not depict: oversized head" in description
    assert "childlike body proportions" in description
    assert "has priority over any conflicting body wording" in description

    reference_prompts = character_factory.build_model_reference_prompts(description)
    assert all("height exactly 182 cm" in prompt for prompt in reference_prompts.values())
    assert all("bodybuilder physique" in prompt for prompt in reference_prompts.values())
    director_contract = _character_lines([male])[0]
    assert "height exactly 182 cm" in director_contract
    assert "bodybuilder physique" in director_contract
    storyboard_contract = _character_contract([male], ["LIN"])
    assert "exactly 7.8 heads tall" in storyboard_contract
    assert "bodybuilder physique" in storyboard_contract

    video_prompt = build_video_prompt(
        {
            "id": "S01",
            "who": ["LIN"],
            "shot_size": "full",
            "what": "LIN strides across the room",
            "duration": 5,
        },
        [male],
        {"shots": {}},
        "seedance-2.0",
    )
    assert isinstance(video_prompt, str)
    assert "角色身体比例逐镜硬合同" in video_prompt
    assert "height exactly 182 cm" in video_prompt
    assert "bodybuilder physique" in video_prompt
    assert "运镜物理硬合同" in video_prompt
    assert "lens: 50mm equivalent" in video_prompt
    assert "50–85mm equivalent cinematic lens" in video_prompt
    assert "wide-angle distortion" in video_prompt


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


def test_phase5_verified_single_shot_end_state_omission_triggers_correction():
    issue = storyboard_qa_gate._issue(
        "L3",
        "moderate",
        "R4",
        "最终动作未完成",
        ["S03"],
        storyboard_ids=["S03_P02"],
        mismatch_type="end_state",
        expected="敌方保安抵达透明观察窗",
        observed="敌方保安仅漂浮在观察窗前方",
        confidence=0.85,
        panel_evidence=[{
            "shot_id": "S03_P02",
            "observed": "保安停在观察窗前，未完成最终接触",
        }],
        evidence_status="not_required",
    )

    assert storyboard_qa_gate.is_blocking_issue(issue) is True
    assert storyboard_qa_gate.grade_issues([issue]) == "C"
    assert storyboard_qa_gate._correctable_issues({"issues": [issue]}) == [issue]


def test_phase5_low_confidence_single_shot_end_state_claim_stays_non_blocking():
    issue = storyboard_qa_gate._issue(
        "L3",
        "moderate",
        "R4",
        "疑似终态偏差",
        ["S03"],
        mismatch_type="end_state",
        expected="抵达观察窗",
        observed="可能仍在窗前",
        confidence=0.6,
        panel_evidence=[{"shot_id": "S03_P02", "observed": "画面遮挡"}],
        evidence_status="not_required",
    )

    assert storyboard_qa_gate.is_blocking_issue(issue) is False
    assert storyboard_qa_gate.grade_issues([issue]) == "A"


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
    assert "解除武器的完成态" in prompt
    assert "敌方不得继续握持武器" in prompt
    assert "所有列出动作完成后的终态快照" in prompt
    assert "每个角色只出现一次" in prompt
    assert "动作→对象→道具→结束状态" in prompt
    assert "不得仍停留在搏斗、争夺、准备或前一动作中" in prompt
    assert "不得用运动线否定静止" in prompt


def test_phase2_bridge_panel_is_not_treated_as_the_last_story_beat():
    content_prompt = _build_panel_prompt(
        {"who": ["Agent", "敌方保安"], "where": "旋转走廊"},
        {
            "beat_id": "S02_P01",
            "generation_mode": "multi_image",
            "action": "Agent解除敌方保安的武器 → Agent抓住门框稳定身体",
            "end_state": "敌方已被缴械，Agent抓住门框稳定身体",
        },
        1,
        2,
        [],
        is_last_content_beat=True,
    )
    bridge_prompt = _build_panel_prompt(
        {"who": ["Agent", "敌方保安"], "where": "旋转走廊"},
        {
            "beat_id": "S02_P02",
            "generation_mode": "first_last_frame_bridge",
            "action": "保持当前终态并接到 S03_P01",
            "end_state": "敌方已被缴械，Agent抓住门框稳定身体",
        },
        2,
        2,
        [],
        is_last_content_beat=False,
    )

    assert "最后一个承载剧情的故事格" in content_prompt
    assert "不得仍停留在搏斗、争夺" in content_prompt
    assert "桥接预览格，不是新的剧情动作格" in bridge_prompt
    assert "绝不能把不同参考图复制成多个实体" in bridge_prompt


def test_phase2_panel_prompt_turns_phase5_evidence_into_negative_constraints():
    prompt = _build_panel_prompt(
        {"who": ["agent", "guard"], "where": "透明观察窗前"},
        {
            "beat_id": "S06_P02",
            "generation_mode": "extend",
            "start_state": "保安飞向观察窗",
            "action": "Agent 恢复稳定姿态",
            "end_state": "观察窗保持完整，Agent 定格",
        },
        2,
        2,
        [
            {"id": "agent", "appearance": {"summary": "深灰色战术服"}},
            {"id": "guard", "appearance": {"summary": "藏蓝色保安制服"}},
        ],
        correction_contract=(
            "这是第 1 轮自动纠偏，只修复 S06。\n"
            "- 纠偏项 1（R4）：必须满足=观察窗保持完整；"
            "已观察到且禁止复现=保安撞破观察窗飞入太空。"
        ),
    )

    assert "Phase 5 定向纠偏合同" in prompt
    assert "观察窗保持完整" in prompt
    assert "保安撞破观察窗飞入太空" in prompt
    assert "禁止复现的负面约束，不是要继续画入画面的剧情" in prompt
    assert "不得通过增加破坏、伤亡、道具或画外事件来规避问题" in prompt


def test_storyboard_prompts_use_generic_role_and_prop_fidelity_contracts():
    storyboard = json.loads(PHASE5_VARIATION_FIXTURE.read_text(encoding="utf-8"))
    shot = storyboard["shots"][0]
    director_prompt, _panels, _layout = build_director_storyboard_prompt(
        storyboard
    )
    prompt = _build_panel_prompt(
        shot,
        shot["storyboard_beats"][0],
        1,
        1,
        [],
    )

    assert "每个角色只执行本格明确分配给自己的动作" in prompt
    assert "严格保留本格声明的道具类型、持有者和使用方式" in prompt
    assert "角色职责与道具合同" in director_prompt
    assert "每个具名角色只执行逐格内容合同明确分配给自己的动作" in director_prompt
    assert "摄影物理合同" in director_prompt
    assert "摄影禁止项" in director_prompt
    assert "random camera movement" in director_prompt
    assert "摄影师是持机记录者，不是舞者" not in prompt
    assert "Groove" not in prompt
    assert "手机HDR高光" not in director_prompt


def test_phase2_panel_prompt_requires_a_kinetic_subject_pose_not_a_neutral_end_pose():
    prompt = _build_panel_prompt(
        {"who": ["表演者"], "where": "排练空间"},
        {
            "beat_id": "S01_P01",
            "generation_mode": "multi_image",
            "start_state": "表演者准备移动",
            "action": "表演者屈膝、摆臂并向前跨步",
            "end_state": "表演者到达前方标记点并继续动作",
        },
        1,
        1,
        [],
    )

    assert "不得把“终态”误画成人物中性站立" in prompt
    assert "关节弯曲、肢体伸展、重心偏移" in prompt
    assert "主动作角色必须是画面的主要运动来源" in prompt
    assert "背景与运镜只能辅助，不能替代主体动作" in prompt


def test_generic_role_contract_does_not_rewrite_dslr_photojournalist():
    shot = {
        "id": "S01",
        "who": ["战地摄影师"],
        "where": "临时避难所",
        "what": "战地摄影师使用 DSLR 记录撤离行动",
        "visual": "摄影师趴在掩体后用长焦相机拍摄",
        "source_excerpt": "摄影师不得暴露位置",
        "storyboard_beats": [{
            "beat_id": "S01_P01",
            "action": "摄影师保持隐蔽并使用 DSLR 拍摄",
            "start_state": "摄影师伏低身体",
            "end_state": "摄影师仍握持 DSLR",
        }],
    }
    storyboard = {"shots": [shot]}
    director_prompt, _panels, _layout = build_director_storyboard_prompt(storyboard)
    panel_prompt = _build_panel_prompt(
        shot,
        shot["storyboard_beats"][0],
        1,
        1,
        [],
    )

    assert "DSLR" in director_prompt
    assert "DSLR" in panel_prompt
    for forbidden in ("iPhone/手机", "不是舞者", "Groove", "领舞姿态"):
        assert forbidden not in director_prompt
        assert forbidden not in panel_prompt


def test_generic_role_contract_preserves_explicitly_authored_participation():
    shot = {
        "id": "S01",
        "who": ["记录员"],
        "where": "排练厅",
        "what": "记录员放下记录板后按导演指令加入队形",
        "visual": "记录员完成记录工作，然后加入集体动作",
        "storyboard_beats": [{
            "beat_id": "S01_P01",
            "action": "记录员放下记录板并加入队形",
            "start_state": "记录员手持记录板",
            "end_state": "记录员位于队形末端",
        }],
    }
    prompt = _build_panel_prompt(
        shot,
        shot["storyboard_beats"][0],
        1,
        1,
        [],
    )

    assert "记录员放下记录板并加入队形" in prompt
    assert "让角色无故放下道具" in prompt
    assert "禁止参与" not in prompt


def test_phase1_rejects_real_17_06_default_shot_language_before_paid_work():
    storyboard = json.loads(PHASE5_VARIATION_FIXTURE.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match=r"variation quality 1\.4/5"):
        adaptation_engine._validate_shot_language_variation(storyboard["shots"])


def test_phase5_global_variation_requires_replanning_without_mutation(tmp_path):
    storyboard = json.loads(PHASE5_VARIATION_FIXTURE.read_text(encoding="utf-8"))
    storyboard_path = tmp_path / "STORYBOARD.json"
    storyboard_path.write_text(
        json.dumps(storyboard, ensure_ascii=False),
        encoding="utf-8",
    )
    before = storyboard_path.read_bytes()
    issue = storyboard_qa_gate._issue(
        "L1",
        "severe",
        "scene_variation_insufficient",
        "Storyboard variation quality 1.4/5 requires revision",
        [],
    )
    qa_calls = []

    def qa(_output_dir):
        qa_calls.append(_output_dir)
        return {
            "status": "error",
            "grade": "C",
            "gate_passed": False,
            "issues": [issue],
            "failed_shot_ids": [],
        }

    result = storyboard_qa_gate.run_storyboard_qa_with_correction(
        tmp_path,
        max_correction_attempts=2,
        qa_runner=qa,
        redraw_runner=lambda *_args: pytest.fail("global issue must not redraw images"),
    )

    assert result["gate_passed"] is False
    assert result["correction"]["status"] == "requires_replanning"
    assert result["correction"]["recommended_restart_phase"] == "phase1"
    assert result["correction"]["attempts_used"] == 0
    assert result["correction"]["global_issue_codes"] == [
        "scene_variation_insufficient"
    ]
    assert storyboard_path.read_bytes() == before
    assert qa_calls == [tmp_path]
    assert not (tmp_path / "phase5_corrections").exists()


def test_phase5_global_issue_prevents_wasted_visual_redraw(tmp_path):
    storyboard = json.loads(PHASE5_VARIATION_FIXTURE.read_text(encoding="utf-8"))
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(storyboard, ensure_ascii=False),
        encoding="utf-8",
    )
    variation_issue = storyboard_qa_gate._issue(
        "L1",
        "severe",
        "scene_variation_insufficient",
        "Storyboard variation quality 1.4/5 requires revision",
        [],
    )
    camera_issue = storyboard_qa_gate._issue(
        "L3",
        "severe",
        "R4",
        "摄影师参与舞蹈",
        ["S02"],
        expected="摄影师持 iPhone 拍摄",
        observed="摄影师模仿女主舞蹈",
    )
    qa_calls = []
    redraw_calls = []

    def qa(_output_dir):
        qa_calls.append(_output_dir)
        return {
            "status": "error",
            "grade": "D",
            "gate_passed": False,
            "issues": [variation_issue, camera_issue],
            "failed_shot_ids": ["S02"],
        }

    def redraw(_output_dir, shot_ids, issues, attempt):
        redraw_calls.append((shot_ids, issues, attempt))
        return {"status": "redrawn", "shot_ids": shot_ids, "attempt": attempt}

    result = storyboard_qa_gate.run_storyboard_qa_with_correction(
        tmp_path,
        max_correction_attempts=2,
        qa_runner=qa,
        redraw_runner=redraw,
    )

    assert result["gate_passed"] is False
    assert redraw_calls == []
    assert qa_calls == [tmp_path]
    assert result["correction"]["status"] == "requires_replanning"
    assert result["correction"]["attempts_used"] == 0
    assert result["issues"] == [variation_issue, camera_issue]


@pytest.mark.parametrize("shot_count", [3, 5, 8])
def test_variation_check_is_pure_for_different_storyboard_shapes(shot_count):
    scenes = [
        {
            "id": f"S{index:02d}",
            "what": "博物馆导览继续",
            "visual": "刻意保持统一广角构图和三脚架机位",
            "shot_size": "wide",
            "camera_movement": "static",
            "lighting_key": "natural",
            "shot_intent": "atmosphere",
            "texture_keywords": ["石材墙面", "玻璃反射"],
        }
        for index in range(1, shot_count + 1)
    ]
    before = json.loads(json.dumps(scenes, ensure_ascii=False))

    report = check_scene_variation(scenes)

    assert scenes == before
    assert report["verdict"] in {"strong", "acceptable", "revise", "fail"}


def test_production_sources_do_not_contain_flashmob_specific_fixups():
    sources = [
        ROOT / "pipeline" / "src" / "phases" / "phase1" / "adaptation_engine.py",
        ROOT / "pipeline" / "src" / "phases" / "phase1" / "director_storyboard.py",
        ROOT / "pipeline" / "src" / "phases" / "phase2" / "shot_storyboards.py",
        ROOT / "pipeline" / "src" / "quality" / "variation_checker.py",
    ]
    production_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sources
    )

    for leaked_fixup in (
        "摄影师是持机记录者，不是舞者",
        "手机HDR高光",
        "路面纹理",
        "Groove",
        "领舞姿态",
        "_SHOT_SIZE_PALETTE",
    ):
        assert leaked_fixup not in production_text


def test_phase5_automatically_redraws_failed_shots_then_rechecks(tmp_path):
    blocking_issue = storyboard_qa_gate._issue(
        "L3",
        "moderate",
        "R4",
        "最终动作状态偏离合同",
        ["S05", "S06"],
        storyboard_ids=["S05_P02", "S06_P02"],
        expected="观察窗完整且 Agent 稳定定格",
        observed="观察窗破裂且人物飞入太空",
    )
    qa_results = iter([
        {
            "status": "error",
            "grade": "C",
            "gate_passed": False,
            "issues": [blocking_issue],
            "failed_shot_ids": ["S05", "S06"],
            "outputs": ["storyboard_qa_report.json"],
        },
        {
            "status": "done",
            "grade": "A",
            "gate_passed": True,
            "issues": [],
            "failed_shot_ids": [],
            "outputs": ["storyboard_qa_report.json"],
        },
    ])
    redraw_calls = []

    def redraw(output_dir, shot_ids, issues, attempt):
        redraw_calls.append((output_dir, shot_ids, issues, attempt))
        return {"status": "redrawn", "shot_ids": shot_ids, "attempt": attempt}

    result = storyboard_qa_gate.run_storyboard_qa_with_correction(
        tmp_path,
        max_correction_attempts=2,
        qa_runner=lambda _output_dir: next(qa_results),
        redraw_runner=redraw,
    )

    assert result["status"] == "done"
    assert result["gate_passed"] is True
    assert redraw_calls[0][0] == tmp_path
    assert redraw_calls[0][1] == ["S05", "S06"]
    assert redraw_calls[0][3] == 1
    assert result["correction"]["attempts_used"] == 1
    assert result["correction"]["final_gate_passed"] is True
    assert "phase5_correction_report.json" in result["outputs"]
    persisted = json.loads(
        (tmp_path / "storyboard_qa_report.json").read_text(encoding="utf-8")
    )
    assert persisted["correction"]["history"][0]["status"] == "passed"


def test_phase5_correction_is_bounded_and_fails_closed(tmp_path):
    blocking_issue = storyboard_qa_gate._issue(
        "L3", "severe", "R4", "动作完全错误", ["S03"],
        expected="Agent 解除武器", observed="保安持枪射击",
    )
    qa_calls = 0
    redraw_attempts = []

    def qa(_output_dir):
        nonlocal qa_calls
        qa_calls += 1
        return {
            "status": "error",
            "grade": "C",
            "gate_passed": False,
            "issues": [blocking_issue],
            "failed_shot_ids": ["S03"],
        }

    def redraw(_output_dir, _shot_ids, _issues, attempt):
        redraw_attempts.append(attempt)
        return {"status": "redrawn", "attempt": attempt}

    result = storyboard_qa_gate.run_storyboard_qa_with_correction(
        tmp_path,
        max_correction_attempts=2,
        qa_runner=qa,
        redraw_runner=redraw,
    )

    assert qa_calls == 3
    assert redraw_attempts == [1, 2]
    assert result["status"] == "error"
    assert result["gate_passed"] is False
    assert "after 2/2 automatic correction attempt" in result["error"]
    assert result["correction"]["attempts_used"] == 2


def test_phase5_real_redraw_reuses_generator_and_archives_failed_shot(tmp_path):
    from PIL import Image

    storyboard = {
        "shots": [
            {
                "id": "S01",
                "where": "走廊入口",
                "storyboard_beats": [{
                    "beat_id": "S01_P01",
                    "duration_s": 5,
                    "generation_mode": "fresh",
                    "action": "Agent 进入走廊",
                    "end_state": "Agent 看向前方",
                }],
            },
            {
                "id": "S02",
                "where": "完整透明观察窗前",
                "storyboard_beats": [{
                    "beat_id": "S02_P01",
                    "duration_s": 5,
                    "generation_mode": "fresh",
                    "action": "Agent 将保安扔向观察窗",
                    "end_state": "观察窗保持完整，Agent 稳定定格",
                }],
            },
        ]
    }
    (tmp_path / "CHARACTERS.json").write_text(
        '{"characters": []}', encoding="utf-8"
    )
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(storyboard, ensure_ascii=False), encoding="utf-8"
    )
    generated_prompts = []

    class FakeClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            generated_prompts.append(prompt)
            Image.effect_noise((320, 180), 80).convert("RGB").save(output_path)
            return "https://image.invalid/panel.png"

        def image_to_image(self, **_kwargs):
            pytest.fail("reset-boundary shots without characters use text-to-image")

    client = FakeClient()
    generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=client,
        director_storyboard_path=tmp_path / "missing.png",
    )
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(storyboard, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "storyboard_qa_report.json").write_text(
        '{"status":"error","grade":"C"}', encoding="utf-8"
    )
    issue = storyboard_qa_gate._issue(
        "L3",
        "severe",
        "R4",
        "观察窗状态错误",
        ["S02"],
        storyboard_ids=["S02_P01"],
        expected="观察窗保持完整，Agent 稳定定格",
        observed="保安撞破观察窗飞入太空",
    )

    receipt = storyboard_qa_gate._redraw_failed_storyboards(
        tmp_path,
        ["S02"],
        [issue],
        1,
        image_client=client,
    )

    assert len(generated_prompts) == 3
    assert "Phase 5 定向纠偏合同" in generated_prompts[-1]
    assert "观察窗保持完整，Agent 稳定定格" in generated_prompts[-1]
    assert "保安撞破观察窗飞入太空" in generated_prompts[-1]
    archive = tmp_path / receipt["archive"]["archive_dir"] / "before"
    assert (archive / "storyboard_beats/S02_P01.png").is_file()
    assert (archive / "storyboard_qa_report.json").is_file()
    manifest = json.loads(
        (tmp_path / "SHOT_STORYBOARDS.json").read_text(encoding="utf-8")
    )
    assert manifest["correction"] == {"attempt": 1, "shot_ids": ["S02"]}


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
