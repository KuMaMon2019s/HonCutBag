#!/usr/bin/env python3
"""
artifact_chain.py — M6: HonCut 产物链 + Checkpoint
每阶段产出结构化 JSON 产物，支持从任意阶段恢复。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# HonCut 产物链定义
ARTIFACT_CHAIN = {
    "phase1":   {"produces": "director_plan.json + events.json + CHARACTERS.json + STORYBOARD.json", "requires": []},
    "phase2":   {"produces": "SHOT_STORYBOARDS.json + storyboard_beats/ + shot_storyboards/ + storyboard_images/", "requires": ["STORYBOARD.json"]},
    "phase3":   {"produces": "characters/",                           "requires": ["CHARACTERS.json"]},
    "phase4":   {"produces": "shots/",                                "requires": ["STORYBOARD.json"]},
    "phase5":   {"produces": "storyboard_qa_report.json",             "requires": ["STORYBOARD.json", "shots/"]},
    "phase6":   {"produces": "shots/*/output.mp4",                   "requires": ["shots/", "storyboard_qa_report.json"]},
    "phase7":   {"produces": "quality_report.json",                   "requires": ["shots/"]},
    "phase8":   {"produces": "edit_decisions.json + raw_assembly.mp4", "requires": ["shots/"]},
    "phase9":   {"produces": "polished.mp4 + render_report.json",     "requires": ["raw_assembly.mp4"]},
    "phase9_5": {"produces": "video_qa_report.json",                  "requires": ["polished.mp4"]},
}

# Phase 执行顺序
PHASE_SEQUENCE = [
    "phase1", "phase2", "phase3", "phase4", "phase5",
    "phase6", "phase7", "phase8", "phase9", "phase9_5",
]


def phase_numbers_before(phase: str) -> list[float]:
    """Return CLI phase numbers that precede ``phase`` in canonical order."""
    if phase not in PHASE_SEQUENCE:
        raise ValueError(f"unknown phase: {phase}")
    numbers: list[float] = []
    for name in PHASE_SEQUENCE[:PHASE_SEQUENCE.index(phase)]:
        numbers.append(9.5 if name == "phase9_5" else float(name.removeprefix("phase")))
    return numbers


def save_checkpoint(phase: str, output_dir: Path, artifacts: dict = None) -> Path:
    """每阶段完成后写 checkpoint。
    
    Args:
        phase: 阶段名（如 "phase1"）
        output_dir: 输出目录
        artifacts: 该阶段产出的文件列表/信息
    
    Returns:
        checkpoint 文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "artifacts": artifacts or {},
        "produces": ARTIFACT_CHAIN.get(phase, {}).get("produces", ""),
        "status": "done",
    }
    
    checkpoint_path = output_dir / f"checkpoint_{phase}.json"

    # --- P2-5a: HonCut 历史归档 ---
    try:
        if checkpoint_path.exists():
            history_dir = output_dir / "history"
            history_dir.mkdir(exist_ok=True)
            import shutil
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(str(checkpoint_path), str(history_dir / f"checkpoint_{phase}_{ts}.json"))
    except Exception as e:
        print(f"  ⚠ [P2-5a] checkpoint 历史归档失败（降级跳过）: {e}")

    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    return checkpoint_path


def can_resume_from(phase: str, output_dir: Path) -> bool:
    """检查是否可以从指定阶段恢复（前置依赖是否满足）。
    
    Args:
        phase: 目标阶段名
        output_dir: 输出目录
    
    Returns:
        True 如果所有前置依赖文件存在
    """
    output_dir = Path(output_dir)
    required = ARTIFACT_CHAIN.get(phase, {}).get("requires", [])
    
    for artifact in required:
        # 处理 glob 模式（如 "shots/*/output.mp4"）
        if "*" in artifact:
            import glob
            matches = list(output_dir.glob(artifact))
            if not matches:
                return False
        elif artifact.endswith("/"):
            # 目录
            if not (output_dir / artifact.rstrip("/")).exists():
                return False
        else:
            # 处理 "A + B" 格式（多产物）
            for single in artifact.split(" + "):
                single = single.strip()
                if single and not (output_dir / single).exists():
                    return False
    if phase in {"phase4", "phase5", "phase6", "phase7", "phase8", "phase9", "phase9_5"}:
        storyboard_path = output_dir / "STORYBOARD.json"
        if storyboard_path.is_file():
            try:
                storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
                authored_beats = [
                    beat
                    for shot in storyboard.get("shots", [])
                    if isinstance(shot, dict)
                    for beat in (shot.get("storyboard_beats") or [])
                    if isinstance(beat, dict)
                ]
                if authored_beats:
                    manifest = output_dir / "SHOT_STORYBOARDS.json"
                    if not manifest.is_file():
                        return False
                    document = json.loads(manifest.read_text(encoding="utf-8"))
                    if document.get("status") != "done":
                        return False
                    for beat in authored_beats:
                        value = str(beat.get("storyboard_image") or "").strip()
                        if not value:
                            return False
                        image_path = Path(value)
                        if not image_path.is_absolute():
                            image_path = output_dir / image_path
                        if not image_path.is_file() or image_path.stat().st_size <= 1024:
                            return False
            except (OSError, ValueError, json.JSONDecodeError):
                return False
    return True


def get_resumable_phase(output_dir: Path) -> Optional[str]:
    """找到可以恢复的最早未完成阶段。
    
    Returns:
        阶段名，如果全部完成则返回 None
    """
    output_dir = Path(output_dir)
    
    for phase in PHASE_SEQUENCE:
        checkpoint = output_dir / f"checkpoint_{phase}.json"
        if not checkpoint.exists():
            # 检查是否可以从此阶段恢复
            if can_resume_from(phase, output_dir):
                return phase
            else:
                # 前置不满足，从头开始
                return PHASE_SEQUENCE[0]
    
    return None  # 全部完成


def verify_artifacts(phase: str, output_dir: Path) -> dict:
    """验证指定阶段的产物是否存在。
    
    Returns:
        {"phase": str, "exists": bool, "missing": [...], "found": [...]}
    """
    output_dir = Path(output_dir)
    produces = ARTIFACT_CHAIN.get(phase, {}).get("produces", "")
    
    found = []
    missing = []
    
    for artifact in produces.split(" + "):
        artifact = artifact.strip()
        if not artifact:
            continue
        if "*" in artifact:
            import glob
            matches = list(output_dir.glob(artifact))
            if matches:
                found.append(artifact)
            else:
                missing.append(artifact)
        elif artifact.endswith("/"):
            if (output_dir / artifact.rstrip("/")).exists():
                found.append(artifact)
            else:
                missing.append(artifact)
        else:
            if (output_dir / artifact).exists():
                found.append(artifact)
            else:
                missing.append(artifact)
    
    return {
        "phase": phase,
        "exists": len(missing) == 0,
        "found": found,
        "missing": missing,
    }
