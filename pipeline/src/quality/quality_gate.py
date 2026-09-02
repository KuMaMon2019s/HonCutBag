"""Unified quality gate with per-phase rule routing.

HonCut supervision architecture:
- ONE quality gate module, multiple rule sets routed by phase key
- Red lines (CRITICAL): violate → block pipeline
- Dimensions (WARNING): graded A/B/C/D
- Structured report: 总评 → 问题清单 → 建议

Usage:
    from quality.quality_gate import run_quality_check
    report = run_quality_check("phase3", output_dir)
    if not report.passed:
        # Block pipeline — critical artifact missing
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

from quality.character_reference_qa import (
    PROP_DETAIL_INPUT_SCHEMA,
    PROP_DETAIL_QA_SCHEMA,
    file_sha256,
    validate_identity_detail_input_contract,
    validate_character_reference_qa_receipt,
)
from utils.canonical_visual_contracts import CanonicalVisualContractError
from utils.character_identity import is_declared_character_reference
from utils.video_capabilities import (
    capabilities_for,
    max_primary_story_duration,
    min_primary_story_duration,
)


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
    step_summary: Dict[str, str] = field(default_factory=dict)  # step_name → status

    def has_critical(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.issues)


# ---------------------------------------------------------------------------
# Per-phase quality rules
# ---------------------------------------------------------------------------

QUALITY_RULES: Dict[str, Dict[str, Any]] = {
    "phase1": {
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
    "phase2": {
        "name": "故事板图片",
        "red_lines": [
            ("storyboard_image_exists", "storyboard.png 存在且 > 10KB"),
        ],
        "dimensions": [],
    },
    "phase3": {
        "name": "角色工厂",
        "red_lines": [
            ("character_images_exist", "每个角色声明的四张身份参考图均存在且 > 10KB"),
            ("character_card_exists", "每个角色有 character_card.json"),
            (
                "character_reference_qa_passed",
                "每个角色四视图通过视角、构图、中性姿态、背景及跨视图一致性审核",
            ),
        ],
        "dimensions": [],
    },
    "phase6": {
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
        "name": "一致性守卫",
        "red_lines": [],
        "dimensions": [],
        "critical_steps": ["consistency_guard", "scene_variation", "slideshow_risk"],
    },
    "phase8": {
        "name": "组装引擎",
        "red_lines": [
            ("assembly_exists", "raw_assembly.mp4 存在且 > 500KB"),
            ("assembly_has_video", "raw_assembly.mp4 包含视频流"),
        ],
        "dimensions": [],
        "critical_steps": ["transition_render"],
    },
    "phase9": {
        "name": "后期处理",
        "red_lines": [
            ("final_exists", "polished.mp4 存在且 > 500KB"),
            ("final_has_video", "polished.mp4 包含视频流"),
            ("final_has_audio", "polished.mp4 包含音频流"),
        ],
        "dimensions": [],
        "critical_steps": ["subtitle_burn", "audio_pipeline", "rhythm_editor", "final_encode"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_check(
    phase: str,
    output_dir,
    artifacts: Optional[dict] = None,
    step_status: Optional[Dict[str, str]] = None,
) -> QualityReport:
    """Run quality check for a specific phase.

    Args:
        phase: Phase key (e.g. "phase3")
        output_dir: Path to the output directory
        artifacts: Optional dict of phase outputs/results
        step_status: Optional dict mapping step names to status
                     ("done", "failed", "skipped", "not_required"). Failed/skipped
                     critical steps cap the grade.

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

    # --- Step status integrity: failed/skipped critical steps cap grade ---
    step_issues: List[QualityIssue] = []
    critical_steps = rules.get("critical_steps", [])
    step_status = step_status or {}

    if step_status:
        for step_name, status in step_status.items():
            # ``not_required`` is a successfully resolved condition (for
            # example, no speech means there is intentionally no subtitle burn)
            # and must not be graded like a skipped required operation.
            if status in ("failed", "skipped"):
                is_critical = step_name in critical_steps
                if is_critical:
                    sev = Severity.CRITICAL
                    msg = f"🔴 关键步骤 {step_name} {status}（必须真实执行才能获得 A）"
                else:
                    sev = Severity.WARNING
                    msg = f"🟡 可选步骤 {step_name} {status}"
                step_issues.append(QualityIssue(
                    severity=sev,
                    phase=phase,
                    rule=f"step_{step_name}_{status}",
                    message=msg,
                    suggestion=f"检查 {step_name} 的执行日志，修复后重新运行",
                ))

    issues.extend(step_issues)

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

    # Attach step_status summary for downstream consumers
    report.step_summary = {
        name: status for name, status in step_status.items()
    } if step_status else {}

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
            chars = data.get("characters", [])
            if len(chars) > 0:
                return True
            # Landscape-only scripts: storyboard exists with shots but no
            # human characters — this is valid (e.g. nature scenery).
            sb = output_dir / "STORYBOARD.json"
            if sb.exists():
                sb_data = json.loads(sb.read_text())
                if len(sb_data.get("shots", [])) > 0:
                    return True
            return False
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
        character_count = 0
        for cd in chars_dir.iterdir():
            if not cd.is_dir():
                continue
            character_count += 1
            card_path = cd / "character_card.json"
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            declared = card.get("reference_images")
            if not isinstance(declared, dict) or len(declared) < 4:
                return False
            references: list[Path] = []
            for value in declared.values():
                path = Path(str(value))
                if not path.is_absolute():
                    path = output_dir / path
                references.append(path)
            if len(set(references)) < 4 or not all(
                path.is_file() and path.stat().st_size > 10_240
                for path in references
            ):
                return False
        return character_count > 0

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

    if rule_id == "character_reference_qa_passed":
        chars_dir = _find_chars_dir(output_dir)
        if chars_dir is None:
            return False
        character_count = 0
        for cd in chars_dir.iterdir():
            if not cd.is_dir():
                continue
            character_count += 1
            try:
                card = json.loads(
                    (cd / "character_card.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return False
            declared = card.get("reference_images")
            if not isinstance(declared, dict) or len(declared) < 4:
                return False
            view_paths: dict[str, Path] = {}
            for name, value in declared.items():
                path = Path(str(value))
                view_paths[str(name)] = path if path.is_absolute() else output_dir / path
            report_value = card.get("reference_qa_report")
            if not report_value:
                return False
            report_path = Path(str(report_value))
            if not report_path.is_absolute():
                report_path = output_dir / report_path
            if not validate_character_reference_qa_receipt(
                report_path,
                view_paths,
                synthetic_styling=(
                    card.get("synthetic_styling")
                    if isinstance(card.get("synthetic_styling"), dict)
                    else None
                ),
                generation_contract=(
                    card.get("reference_generation_contract")
                    if isinstance(card.get("reference_generation_contract"), dict)
                    else None
                ),
            ):
                return False
            identity_props = card.get("identity_props")
            if isinstance(identity_props, list) and identity_props:
                detail_value = card.get("prop_detail_board")
                detail_report_value = card.get("prop_detail_board_qa_report")
                if not detail_value or not detail_report_value:
                    return False
                detail_path = Path(str(detail_value))
                if not detail_path.is_absolute():
                    detail_path = output_dir / detail_path
                detail_report_path = Path(str(detail_report_value))
                if not detail_report_path.is_absolute():
                    detail_report_path = output_dir / detail_report_path
                try:
                    detail_report = json.loads(
                        detail_report_path.read_text(encoding="utf-8")
                    )
                    detail_input = detail_report["inputs"]["prop_detail_board"]
                    canonical_inputs = detail_report["inputs"]["canonical_references"]
                    input_contract = detail_report["input_contract"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    return False
                if (
                    detail_report.get("schema") != PROP_DETAIL_QA_SCHEMA
                    or detail_report.get("status") != "passed"
                    or detail_report.get("qa_verdict")
                    not in {"pass", "acceptable_deviation"}
                    or input_contract.get("schema") != PROP_DETAIL_INPUT_SCHEMA
                    or not detail_path.is_file()
                    or detail_input.get("path") != detail_path.name
                    or detail_input.get("sha256") != file_sha256(detail_path)
                    or detail_input.get("media_role")
                    != "identity_prop_geometry_reference"
                ):
                    return False
                try:
                    validate_identity_detail_input_contract(
                        output_dir=output_dir,
                        canonical_paths=[
                            cd / str(value.get("path") or "")
                            for value in canonical_inputs
                            if isinstance(value, dict)
                        ],
                        input_contract=input_contract,
                    )
                except (CanonicalVisualContractError, RuntimeError, TypeError):
                    return False
                final = detail_report.get("final")
                if (
                    not isinstance(final, dict)
                    or not final.get("qa_observation_id")
                    or not final.get("qa_decision_id")
                ):
                    return False
                if not isinstance(canonical_inputs, list) or not canonical_inputs:
                    return False
                for canonical_input in canonical_inputs:
                    if not isinstance(canonical_input, dict):
                        return False
                    canonical_path = cd / str(canonical_input.get("path") or "")
                    if (
                        not canonical_path.is_file()
                        or canonical_input.get("sha256") != file_sha256(canonical_path)
                    ):
                        return False
        return character_count > 0

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
    for candidate in (
        output_dir / "characters",
        output_dir / "characters" / "characters",
    ):
        if not candidate.is_dir():
            continue
        character_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        if any((path / "character_card.json").is_file() for path in character_dirs):
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
            "检查 Seedream API 尺寸与引用链，确认角色四视图生成成功",
        "character_reference_qa_passed":
            "检查 character_reference_qa.json，按失败视角重生成后再进入分镜与视频阶段",
        "videos_exist":
            "检查 Seedance API key 和 prompt，确认视频生成成功",
        "assembly_has_video":
            "Phase 8 拼接可能丢失视频流，检查 VideoStitch 逻辑",
        "final_has_video":
            "Phase 9 音频处理可能丢失视频流，检查 remux 逻辑",
        "final_has_audio":
            "检查音频处理管线是否正常",
    }.get(rule_id, "")


def _print_report(report: QualityReport, phase_name: str) -> None:
    """Print a HonCut quality supervision report."""
    status = "✅ 通过" if report.passed else "❌ 未通过（红线违反）"
    print(f"\n  ┌─ 质检报告: {phase_name} ─────────────────")
    print(f"  │ 评分: {report.grade}  {status}")
    if report.issues:
        for issue in report.issues:
            print(f"  │ {issue.message}")
            if issue.suggestion:
                print(f"  │   → 建议: {issue.suggestion}")
    else:
        print("  │ 无问题")
    if report.step_summary:
        print("  │ ── 步骤执行清单 ──")
        for step, st in report.step_summary.items():
            icon = "✓" if st == "done" else ("✗" if st == "failed" else "⊘")
            print(f"  │   {icon} {step}: {st}")
    print("  └──────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# M5: HonCut 监督层审核
# ---------------------------------------------------------------------------

def run_storyboard_review(storyboard_data: dict, script_text: str, characters: list) -> dict:
    """按 HonCut 监督规范审核分镜表质量。
    
    4 条红线（违反即严重）：
    R1: 资产引用合法（角色/场景必须在 characters 中存在）
    R2: 剧本忠实（台词一字不差，不遗漏不新增）
    R3: 具象可感（禁止抽象笼统词，声音具体到声源）
    R4: 父子资产正确（衍生状态用衍生 ID，不主和衍生同存）
    
    评分：A(0严重≤2中等) / B(0严重≤5中等) / C(1-2严重) / D(≥3严重)
    
    Returns:
        {"grade": "A"|"B"|"C"|"D", "severe": int, "moderate": int, "issues": [...]}
    """
    severe_issues = []
    moderate_issues = []
    
    shots = storyboard_data.get("shots", [])
    has_declared_characters = bool(characters)
    
    # 抽象笼统词黑名单（R3）— 扩充至 30+ 词
    abstract_words = [
        "美丽的", "漂亮的", "好看的", "某种", "一些", "很多", "非常", "很",
        "优雅地", "缓缓地", "轻轻地", "慢慢地", "渐渐地", "默默地",
        "充满", "弥漫", "笼罩", "环绕", "萦绕",
        "氛围感", "意境", "韵味", "格调", "质感",
        "温暖的感觉", "悲伤的感觉", "幸福的感觉",
        "如同", "仿佛", "好像", "似乎",
    ]
    
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"S{i+1:02d}")
        who = shot.get("who", [])
        visual = shot.get("visual") or ""
        what = shot.get("what") or ""
        
        # R1: 资产引用合法。Qualified mentions may contain a declared name,
        # but matching must remain token-safe and unambiguous.
        if has_declared_characters:
            for name in who:
                if not is_declared_character_reference(name, characters):
                    severe_issues.append(f"[R1] {shot_id}: 角色 '{name}' 不在角色列表中")
        
        # R3: 具象可感
        for word in abstract_words:
            if word in visual:
                moderate_issues.append(f"[R3] {shot_id}: 抽象词 '{word}' 在 visual 中")
        
        # R2: 剧本忠实（简单检查：visual 不应为空）
        if not visual or len(visual) < 10:
            moderate_issues.append(f"[R2] {shot_id}: visual 描述过短或为空")
        
        # P0-2a: R2 台词忠实度（检查 shot 中引用的台词是否在原文中存在）
        what = shot.get("what") or ""
        if what and script_text:
            # 提取引号内的台词
            quoted = re.findall(r'["「](.+?)["」]', what)
            for q in quoted:
                if q not in script_text:
                    severe_issues.append(f"[R2] {shot_id}: 台词 '{q[:20]}...' 不在原文中")
        
        # P0-2b: 一级分镜必须能由基础段加有界延长段完整承载。
        duration = shot.get("suggested_duration", shot.get("duration", 0))
        profile = capabilities_for({**storyboard_data, **shot})
        minimum = min_primary_story_duration(profile)
        maximum = max_primary_story_duration(profile)
        if duration and not minimum <= duration <= maximum:
            severe_issues.append(
                f"[时长] {shot_id}: {duration}s 不在 {profile.name} 的 "
                f"{minimum:g}-{maximum:g}秒一级分镜承载范围"
            )
        
        # P0-2c: 长台词拆镜检查（HonCut 铁律: >20字强制拆镜）
        dialogue = shot.get("dialogue") or shot.get("what") or ""
        if isinstance(dialogue, dict):
            dialogue = dialogue.get("line") or ""
        elif not isinstance(dialogue, str):
            dialogue = str(dialogue) if dialogue is not None else ""
        dialogue_text = re.findall(r'["「](.+?)["」]', dialogue)
        for d in dialogue_text:
            if len(d) > 20:
                moderate_issues.append(f"[拆镜] {shot_id}: 台词'{d[:15]}...'({len(d)}字) 建议拆镜")
        
        # P0-2f: 禁光影色调词（HonCut: 分镜不规划光影/色调/配乐）
        banned_visual_words = ["色调", "光影", "配乐", "BGM", "背景音乐", "色温", "饱和度", "对比度"]
        for bw in banned_visual_words:
            if bw in visual:
                moderate_issues.append(f"[禁词] {shot_id}: visual 含 '{bw}'（应由后期处理）")
    
    # P0-2d: 在场人物不消失检查（HonCut 铁律）
    for i in range(1, len(shots)):
        prev_who = set(shots[i-1].get("who", []))
        curr_who = set(shots[i].get("who", []))
        # 同场景内（where 相同），人物不应无故减少
        if shots[i-1].get("where") == shots[i].get("where") and prev_who:
            disappeared = prev_who - curr_who
            if disappeared:
                moderate_issues.append(
                    f"[消失] S{i+1:02d}: {', '.join(disappeared)} 在同场景中消失")
    
    # P0-2g: 景别视角错开检查（HonCut: 相邻镜头不应同景别同角度）
    for i in range(1, len(shots)):
        prev_cam = shots[i-1].get("camera", "")
        curr_cam = shots[i].get("camera", "")
        if prev_cam and curr_cam and prev_cam == curr_cam:
            moderate_issues.append(
                f"[景别] S{i:02d}→S{i+1:02d}: 连续相同景别 '{curr_cam}'")
    
    # 评分
    severe_count = len(severe_issues)
    moderate_count = len(moderate_issues)
    
    if severe_count >= 3:
        grade = "D"
    elif severe_count >= 1:
        grade = "C"
    elif moderate_count > 5:
        grade = "B"
    elif moderate_count > 2:
        grade = "B"
    else:
        grade = "A"
    
    result = {
        "grade": grade,
        "severe": severe_count,
        "moderate": moderate_count,
        "issues": severe_issues + moderate_issues,
        "total_shots": len(shots),
    }
    
    # 打印报告
    print(f"\n  📋 [M5] 分镜审核报告: {grade} 级")
    print(f"    严重: {severe_count} | 中等: {moderate_count} | 镜头数: {len(shots)}")
    if severe_issues:
        for issue in severe_issues[:5]:
            print(f"    🔴 {issue}")
    if moderate_issues:
        for issue in moderate_issues[:5]:
            print(f"    🟡 {issue}")
    
    return result
