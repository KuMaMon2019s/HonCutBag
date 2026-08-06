#!/usr/bin/env python3
"""M6 artifact_chain 单元测试：产物链 + checkpoint + 恢复。"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.artifact_chain import (
    ARTIFACT_CHAIN,
    PHASE_SEQUENCE,
    save_checkpoint,
    can_resume_from,
    verify_artifacts,
)


def test_artifact_chain_contains_all_phases():
    """ARTIFACT_CHAIN 包含 phase1 到 phase8 所有键。"""
    expected = {"phase1", "phase2", "phase2_5", "phase3", "phase4",
                "phase5", "phase6", "phase7", "phase8"}
    assert expected.issubset(set(ARTIFACT_CHAIN.keys())), (
        f"ARTIFACT_CHAIN 缺少键：{expected - set(ARTIFACT_CHAIN.keys())}"
    )
    # 每个 phase 都有 produces 和 requires
    for phase, info in ARTIFACT_CHAIN.items():
        assert "produces" in info, f"{phase} 缺少 'produces'"
        assert "requires" in info, f"{phase} 缺少 'requires'"


def test_phase_sequence_order():
    """PHASE_SEQUENCE 顺序正确，且包含所有 phase。"""
    assert PHASE_SEQUENCE[0] == "phase1"
    assert PHASE_SEQUENCE[-1] == "phase8"
    # phase2_5 必须在 phase2 之后、phase3 之前
    idx_2 = PHASE_SEQUENCE.index("phase2")
    idx_2_5 = PHASE_SEQUENCE.index("phase2_5")
    idx_3 = PHASE_SEQUENCE.index("phase3")
    assert idx_2 < idx_2_5 < idx_3, "phase2 < phase2_5 < phase3 顺序错误"
    # 所有 phase 都在序列中
    assert set(PHASE_SEQUENCE) == set(ARTIFACT_CHAIN.keys())


def test_save_checkpoint_writes_file(tmp_path):
    """save_checkpoint() 写入 checkpoint_phaseN.json 文件。"""
    output_dir = tmp_path / "output"
    artifacts = {"director_plan.json": "ok"}

    path = save_checkpoint("phase1", output_dir, artifacts)

    assert path.exists(), f"checkpoint 文件未生成：{path}"
    assert path.name == "checkpoint_phase1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phase"] == "phase1"
    assert data["status"] == "done"
    assert data["artifacts"] == artifacts


def test_can_resume_from_dependencies_met(tmp_path):
    """can_resume_from() 在前置依赖满足时返回 True。"""
    output_dir = tmp_path
    # phase1 没有 requires，始终可恢复
    assert can_resume_from("phase1", output_dir) is True

    # phase2 需要 director_plan.json
    (output_dir / "director_plan.json").write_text("{}")
    assert can_resume_from("phase2", output_dir) is True


def test_can_resume_from_dependencies_missing(tmp_path):
    """can_resume_from() 在前置依赖缺失时返回 False。"""
    output_dir = tmp_path
    # phase2 需要 director_plan.json，但目录为空
    assert can_resume_from("phase2", output_dir) is False

    # phase4 需要 storyboard.json
    assert can_resume_from("phase4", output_dir) is False


def test_verify_artifacts_detects_presence(tmp_path):
    """verify_artifacts() 检测产物存在性。"""
    output_dir = tmp_path

    # phase1 产出 director_plan.json，不存在时 exists=False
    result = verify_artifacts("phase1", output_dir)
    assert result["exists"] is False
    assert "director_plan.json" in result["missing"]

    # 创建后 exists=True
    (output_dir / "director_plan.json").write_text("{}")
    result = verify_artifacts("phase1", output_dir)
    assert result["exists"] is True
    assert "director_plan.json" in result["found"]
    assert result["missing"] == []
