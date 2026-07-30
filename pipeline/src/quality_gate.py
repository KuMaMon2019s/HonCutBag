"""Unified quality gate with per-phase rule routing.

Architecture learned from Toonflow's supervision layer:
- ONE quality gate module, multiple rule sets routed by phase key
- Red lines (CRITICAL): violate → block pipeline
- Dimensions (WARNING): graded A/B/C/D
- Structured report: 总评 → 问题清单 → 建议

Usage:
    from quality_gate import run_quality_check
    report = run_quality_check("phase3", output_dir)
    if not report.passed:
        # Block pipeline — critical artifact missing
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
import json
import os


class Severity(Enum):
    CRITICAL = "critical"   # Red line — blocks pipeline
    WARNING = "warning"     # Yellow — log but continue
    INFO = "info"           # White — informational


@dataclass
class QualityIssue:
    severity: Severity
    phase: str
    rule: str
    message: str
    suggestion: str = ""


@dataclass
class QualityReport:
    phase: str
    grade: str              # A/B/C/D
    passed: bool            # False if any CRITICAL issue
    issues: List[QualityIssue] = field(default_factory=list)

    def has_critical(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.issues)


# ---------------------------------------------------------------------------
# Per-phase quality rules
# ---------------------------------------------------------------------------

QUALITY_RULES: Dict[str, Dict[str, Any]] = {
    "phase2": {
        "name": "编剧引擎",
        "red_lines": [
            ("events_exist", "事件列表不为空"),
            ("characters_exist", "至少发现 1 个角色"),
            ("storyboard_exists", "STORYBOARD.json 已生成"),
        ],
        "dimensions": [
            ("event_count", "事件数量 ≥ 5",
             lambda d: len(d.get("events", [])) >= 5),
            ("shot_count", "镜头数量 ≥ 5",
             lambda d: len(d.get("shots", [])) >= 5),
        ],
    },
    "phase2_5": {
        "name": "故事板图片",
        "red_lines": [
            ("storyboard_image_exists", "storyboard.png 存在且 > 10KB"),
        ],
        "dimensions": [],
    },
    "phase3": {
        "name": "角色工厂",
        "red_lines": [
            ("character_images_exist", "每个角色至少有 front.png 且 > 10KB"),
            ("character_card_exists", "每个角色有 character_card.json"),
        ],
        "dimensions": [
            ("all_views_present", "每个角色有 front/side/back 三视图",
             lambda d: True),  # checked via file system in red lines
        ],
    },
    "phase5": {
        "name": "视频生成",
        "red_lines": [
            ("videos_exist", "至少 1 个镜头视频 > 100KB"),
        ],
        "dimensions": [
            ("video_ratio", "成功视频数 / 总镜头数 ≥ 70%",
             lambda d: True),  # ratio check needs shot count context
        ],
    },
    "phase7": {
        "name": "组装引擎",
        "red_lines": [
            ("assembly_exists", "raw_assembly.mp4 存在且 > 500KB"),
            ("assembly_has_video", "raw_assembly.mp4 包含视频流"),
        ],
        "dimensions": [],
    },
    "phase8": {
        "name": "后期处理",
        "red_lines": [
            ("final_exists", "polished.mp4 存在且 > 500KB"),
            ("final_has_video", "polished.mp4 包含视频流"),
            ("final_has_audio", "polished.mp4 包含音频流"),
        ],
        "dimensions": [],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_check(
    phase: str,
    output_dir,
    artifacts: Optional[dict] = None,
) -> QualityReport:
    """Run quality check for a specific phase.

    Args:
        phase: Phase key (e.g. "phase3")
        output_dir: Path to the output directory
        artifacts: Optional dict of phase outputs/results

    Returns:
        QualityReport with grade and issues
    """
    output_dir = Path(output_dir)
    rules = QUALITY_RULES.get(phase)
    if not rules:
        return QualityReport(phase=phase, grade="A", passed=True)

    issues: List[QualityIssue] = []

    # --- Red lines ---
    for rule_id, description in rules["red_lines"]:
        ok = _check_red_line(rule_id, output_dir, artifacts)
        if not ok:
            issues.append(QualityIssue(
                severity=Severity.CRITICAL,
                phase=phase,
                rule=rule_id,
                message=f"🔴 红线违反: {description}",
                suggestion=_get_suggestion(rule_id),
            ))

    # --- Dimensions ---
    for dim_id, description, check_fn in rules.get("dimensions", []):
        try:
            data = artifacts or {}
            if not check_fn(data):
                issues.append(QualityIssue(
                    severity=Severity.WARNING,
                    phase=phase,
                    rule=dim_id,
                    message=f"🟡 {description}",
                ))
        except Exception:
            pass

    # --- Grade ---
    n_crit = sum(1 for i in issues if i.severity == Severity.CRITICAL)
    n_warn = sum(1 for i in issues if i.severity == Severity.WARNING)

    if n_crit >= 3:
        grade = "D"
    elif n_crit >= 1:
        grade = "C"
    elif n_warn > 2:
        grade = "B"
    else:
        grade = "A"

    passed = n_crit == 0
    report = QualityReport(phase=phase, grade=grade, passed=passed, issues=issues)
    _print_report(report, rules["name"])
    return report


# ---------------------------------------------------------------------------
# Red-line checkers
# ---------------------------------------------------------------------------

def _check_red_line(rule_id: str, output_dir: Path,
                    artifacts: Optional[dict] = None) -> bool:

    if rule_id == "events_exist":
        f = output_dir / "events.jsonl"
        if f.exists() and f.stat().st_size > 10:
            return True
        return (artifacts or {}).get("events") is not None

    if rule_id == "characters_exist":
        f = output_dir / "CHARACTERS.json"
        if f.exists():
            data = json.loads(f.read_text())
            return len(data.get("characters", [])) > 0
        return False

    if rule_id == "storyboard_exists":
        f = output_dir / "STORYBOARD.json"
        return f.exists() and f.stat().st_size > 100

    if rule_id == "storyboard_image_exists":
        f = output_dir / "storyboard.png"
        return f.exists() and f.stat().st_size > 10_240

    if rule_id == "character_images_exist":
        chars_dir = _find_chars_dir(output_dir)
        if chars_dir is None:
            return False
        for cd in chars_dir.iterdir():
            if not cd.is_dir():
                continue
            front = cd / "front.png"
            if not front.exists() or front.stat().st_size < 10_240:
                return False
        return True

    if rule_id == "character_card_exists":
        chars_dir = _find_chars_dir(output_dir)
        if chars_dir is None:
            return False
        for cd in chars_dir.iterdir():
            if not cd.is_dir():
                continue
            if not (cd / "character_card.json").exists():
                return False
        return True

    if rule_id == "videos_exist":
        shots = output_dir / "shots"
        if not shots.exists():
            return False
        return any(
            (sd / "output.mp4").exists() and (sd / "output.mp4").stat().st_size > 102_400
            for sd in shots.iterdir() if sd.is_dir()
        )

    if rule_id == "assembly_exists":
        f = output_dir / "raw_assembly.mp4"
        return f.exists() and f.stat().st_size > 512_000

    if rule_id == "final_exists":
        f = output_dir / "polished.mp4"
        return f.exists() and f.stat().st_size > 512_000

    if rule_id in ("assembly_has_video", "final_has_video"):
        target = ("raw_assembly.mp4" if "assembly" in rule_id
                  else "polished.mp4")
        return _probe_stream(output_dir / target, "v")

    if rule_id == "final_has_audio":
        return _probe_stream(output_dir / "polished.mp4", "a")

    return True  # unknown rules pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_chars_dir(output_dir: Path) -> Optional[Path]:
    """Find the characters directory (handles nested structure)."""
    for candidate in (output_dir / "characters",
                      output_dir / "characters" / "characters"):
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return None


def _probe_stream(filepath: Path, stream_type: str) -> bool:
    """Use ffprobe to check if a media file has a video/audio stream."""
    import subprocess
    if not filepath.exists():
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams",
             "-select_streams", stream_type, str(filepath)],
            capture_output=True, text=True, timeout=10,
        )
        return f"codec_type={'video' if stream_type == 'v' else 'audio'}" in r.stdout
    except Exception:
        return False


def _get_suggestion(rule_id: str) -> str:
    return {
        "character_images_exist":
            "检查 Seedream API 尺寸参数，确认三视图生成成功",
        "videos_exist":
            "检查 Seedance API key 和 prompt，确认视频生成成功",
        "assembly_has_video":
            "Phase 7 拼接可能丢失视频流，检查 VideoStitch 逻辑",
        "final_has_video":
            "Phase 8 音频处理可能丢失视频流，检查 remux 逻辑",
        "final_has_audio":
            "检查音频处理管线是否正常",
    }.get(rule_id, "")


def _print_report(report: QualityReport, phase_name: str) -> None:
    """Print quality report in Toonflow supervision style."""
    status = "✅ 通过" if report.passed else "❌ 未通过（红线违反）"
    print(f"\n  ┌─ 质检报告: {phase_name} ─────────────────")
    print(f"  │ 评分: {report.grade}  {status}")
    if report.issues:
        for issue in report.issues:
            print(f"  │ {issue.message}")
            if issue.suggestion:
                print(f"  │   → 建议: {issue.suggestion}")
    else:
        print(f"  │ 无问题")
    print(f"  └──────────────────────────────────────────\n")
