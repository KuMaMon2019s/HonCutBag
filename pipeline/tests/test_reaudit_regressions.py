"""Regression coverage for the 84e45a0 production-path re-audit."""

from __future__ import annotations

import hashlib
import json

import pytest

from phases import pipeline_core
from phases.phase1.phase1_screenwriter import run_phase1_screenwriter
from runtime.run_manifest import prepare_run_manifest


def _resolved_config(spec: dict) -> dict:
    return {
        "duration": 30,
        "shot_duration": 5,
        "video_provider": "seedance",
        "video_model": "doubao-seedance-2.0-mini",
        "project_video_spec": spec,
    }


def test_run_manifest_refuses_changed_script_or_config_on_resume(tmp_path):
    spec = pipeline_core._project_video_spec("tiktok")
    initial = prepare_run_manifest(
        tmp_path,
        source_text="first script",
        resolved_config=_resolved_config(spec),
        repo_root=pipeline_core.SCRIPT_DIR.parent.parent,
        resume=False,
    )

    resumed = prepare_run_manifest(
        tmp_path,
        source_text="first script",
        resolved_config=_resolved_config(spec),
        repo_root=pipeline_core.SCRIPT_DIR.parent.parent,
        resume=True,
    )
    assert resumed["run_fingerprint"] == initial["run_fingerprint"]

    with pytest.raises(RuntimeError, match="immutable run identity changed"):
        prepare_run_manifest(
            tmp_path,
            source_text="second script",
            resolved_config=_resolved_config(spec),
            repo_root=pipeline_core.SCRIPT_DIR.parent.parent,
            resume=True,
        )

    changed = {**_resolved_config(spec), "video_model": "different-model"}
    with pytest.raises(RuntimeError, match="immutable run identity changed"):
        prepare_run_manifest(
            tmp_path,
            source_text="first script",
            resolved_config=changed,
            repo_root=pipeline_core.SCRIPT_DIR.parent.parent,
            resume=True,
        )


def test_resume_without_manifest_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="RUN_MANIFEST.json is missing"):
        prepare_run_manifest(
            tmp_path,
            source_text="script",
            resolved_config=_resolved_config(pipeline_core._project_video_spec("1080p")),
            repo_root=pipeline_core.SCRIPT_DIR.parent.parent,
            resume=True,
        )


def test_tiktok_project_spec_reaches_phase1_storyboard(tmp_path):
    spec = pipeline_core._project_video_spec("tiktok")

    result = pipeline_core.run_phase1_screenwriter(
        "一个原创人物穿过雨中的空旷街道。",
        tmp_path,
        duration=15,
        dry_run=True,
        project_video_spec=spec,
    )

    assert result["status"] == "done"
    storyboard = json.loads((tmp_path / "STORYBOARD.json").read_text(encoding="utf-8"))
    assert (storyboard["aspect_ratio"], storyboard["width"], storyboard["height"]) == (
        "9:16",
        1080,
        1920,
    )
    assert all(shot["aspect_ratio"] == "9:16" for shot in storyboard["shots"])


def test_phase1_dry_run_blocks_real_source_capacity_instead_of_fixed_mock(tmp_path):
    source = "\n\n".join(
        f"{index}. 操作员先完成步骤{index}，随后确认不可逆结果{index}。"
        for index in range(1, 97)
    )

    result = pipeline_core.run_phase1_screenwriter(
        source,
        tmp_path,
        duration=32,
        dry_run=True,
    )

    assert result["status"] == "error"
    assert "dry-run source capacity preflight failed" in result["error"]
    assert not (tmp_path / "STORYBOARD.json").exists()
    receipt = json.loads(
        (tmp_path / "phase1_dry_run_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert receipt["capacity_plan"]["action_capacity_status"] == (
        "screenplay_compression_required"
    )
    assert receipt["remote_requests"] == 0


def test_phase1_dry_run_uses_scoped_composite_source_in_structural_fixture(tmp_path):
    source = (
        "以下编号定义叙事顺序；每条均为同一瞬间完成的并发复合动作。\n\n"
        + "\n\n".join(
            f"{index}. 操作员同时移动组件{index}并锁定组件{index}。"
            for index in range(1, 9)
        )
        + "\n\n9. 持续状态：操作员保持待命。"
        + "\n\n全程保持同一摄影机风格，禁止水印和字幕。"
    )

    result = pipeline_core.run_phase1_screenwriter(
        source,
        tmp_path,
        duration=32,
        dry_run=True,
    )

    assert result["status"] == "done"
    receipt = json.loads(
        (tmp_path / "phase1_dry_run_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["capacity_plan"]["generation_action_units"] == 8
    assert receipt["ignored_global_directive_count"] == 1
    storyboard = json.loads(
        (tmp_path / "STORYBOARD.json").read_text(encoding="utf-8")
    )
    assert "组件1" in json.dumps(storyboard, ensure_ascii=False)
    assert "神秘地图" not in json.dumps(storyboard, ensure_ascii=False)
    assert all(event["dry_run_source_derived"] for event in storyboard["events"])


def test_phase1_dry_run_indexes_actions_after_a_sustained_event(tmp_path):
    items = [
        "操作员打开第一道安全门。",
        "操作员检查控制面板。",
        "持续状态：操作员保持警戒。",
        "操作员按下启动按钮。",
        "操作员锁定第一组组件。",
        "操作员移动第二组组件。",
        "操作员确认压力读数。",
        "操作员关闭旁路阀门。",
        "操作员接通备用电源。",
        "操作员校准导航模块。",
        "操作员完成系统复位。",
        "持续状态：操作员保持待命。",
    ]
    source = "\n\n".join(
        f"{index}. {item}" for index, item in enumerate(items, 1)
    )

    result = run_phase1_screenwriter(
        source,
        tmp_path,
        duration=36,
        dry_run=True,
    )

    assert result["status"] == "done"
    receipt = json.loads(
        (tmp_path / "phase1_dry_run_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_derived_event_count"] == 12
    assert receipt["capacity_plan"]["generation_action_units"] == 10
    assert receipt["remote_requests"] == 0

    storyboard = json.loads(
        (tmp_path / "STORYBOARD.json").read_text(encoding="utf-8")
    )
    first_shot = storyboard["shots"][0]
    assert first_shot["source_events"] == [1, 2, 3, 4]
    assert len(first_shot["micro_actions"]) == 3
    assert [
        unit["ledger_indexes"]
        for unit in first_shot["generation_action_units"]
    ] == [[0], [1], [2]]


def test_production_source_has_no_audited_story_specific_branches():
    source = (pipeline_core.SCRIPT_DIR / "phases" / "pipeline_core.py").read_text(
        encoding="utf-8"
    ).casefold()

    for forbidden in (
        "lin xia",
        "shen yu",
        '"xia"',
        '"yu"',
        "s05 \"raises her hand",
        "rear three-quarter camera behind the allies",
        "blades stay parallel and must not cross",
    ):
        assert forbidden not in source
