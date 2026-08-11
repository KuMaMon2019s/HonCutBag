#!/usr/bin/env python3
"""HonCut pipeline CLI and backward-compatible orchestration facade.

Phase entry points live in :mod:`phases`; the implementation core is kept as
one module so the existing LangGraph nodes and monkeypatch-based integrations
continue to share a single module namespace.
"""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.deps import check_dependencies

check_dependencies()

from phases import pipeline_core as _core
from utils.pipeline_config import DEFAULT_CONFIG, load_config
from utils.run_memory import RunMemory

# Phase IDs renumbered to contiguous integers on 2026-08-10.
PHASES = (
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
)
PHASE_NUMBERS = {
    "phase1": "1",
    "phase2": "2",
    "phase3": "3",
    "phase4": "4",
    "phase5": "5",
    "phase6": "6",
    "phase7": "7",
    "phase8": "8",
    "phase9": "9",
}


def _record_report_checkpoints(report: dict, output_dir: str | Path) -> None:
    """Persist every successful phase returned by the live runner path."""
    phase_names = {value: key for key, value in PHASE_NUMBERS.items()}
    for report_key, result in report.get("phases", {}).items():
        if not isinstance(result, dict) or result.get("status") not in ("done", "skipped"):
            continue
        phase_name = report_key if report_key in PHASES else phase_names.get(str(report_key))
        if phase_name:
            _core._record_stage_checkpoint(Path(output_dir), phase_name, result)


def _compact_phase_record(phase_id: str, result: dict) -> str:
    """Build generic phase metadata without copying large result payloads."""
    metrics = {
        key: value
        for key, value in result.items()
        if key not in {"status", "outputs", "artifacts"}
        and isinstance(value, (int, float, bool))
    }
    artifact_values = result.get("outputs") or result.get("artifacts") or []
    if isinstance(artifact_values, str):
        artifact_values = [artifact_values]
    artifacts = [Path(value).name for value in artifact_values if isinstance(value, str)]
    record = {
        "phase": phase_id,
        "status": result.get("status", "unknown"),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    if result.get("error"):
        record["error"] = str(result["error"])[:160]
    return json.dumps(record, ensure_ascii=False, sort_keys=True)[:499]


def _record_run_memory(
    report: dict,
    output_dir: str | Path,
    config: dict | None = None,
    *,
    memory_factory=RunMemory,
) -> None:
    """Record every reported phase status in run-scoped memory."""
    config = load_config() if config is None else config
    if not config.get("memory_enabled", DEFAULT_CONFIG["memory_enabled"]):
        return
    memory = memory_factory(
        Path(output_dir),
        messages_per_summary=int(
            config.get(
                "memory_messages_per_summary",
                DEFAULT_CONFIG["memory_messages_per_summary"],
            )
        ),
    )
    phase_names = {value: key for key, value in PHASE_NUMBERS.items()}
    for report_key, result in report.get("phases", {}).items():
        if not isinstance(result, dict):
            continue
        # A selected-phase run reports every other phase as skipped. Persisting
        # those no-op records creates noise and used to trigger paid LLM
        # summaries after the pipeline had already printed COMPLETED.
        if result.get("status") == "skipped":
            continue
        phase_name = report_key if report_key in PHASES else phase_names.get(str(report_key))
        if phase_name:
            memory.add("phase", _compact_phase_record(phase_name, result))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Honcut AI Video Pipeline — 端到端管线")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="故事文本")
    input_group.add_argument("--input", type=str, help="故事文本文件路径")
    parser.add_argument("--duration", type=int, default=60, help="目标视频时长（秒），默认 60")
    parser.add_argument("--shot-duration", type=int, default=_core.AVG_SHOT_DURATION,
                        help=f"每镜平均时长（秒），默认 {_core.AVG_SHOT_DURATION}")
    parser.add_argument("--chain-mode", action="store_true",
                        help="Seedance 尾帧接力模式（镜头串行生成）")
    parser.add_argument("--dry-run", action="store_true", help="dry-run 模式")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录，默认当前目录")
    parser.add_argument("--skip-phase", type=float, nargs="+", default=[], help="跳过指定 Phase")
    parser.add_argument(
        "--transition", choices=["crossfade", "fade", "cut"], default="crossfade", help="Phase 8 转场模式"
    )
    parser.add_argument("--transition-duration", type=float, default=0.5, help="Phase 8 转场时长（秒）")
    reshoot_group = parser.add_mutually_exclusive_group()
    reshoot_group.add_argument(
        "--enable-reshoot", dest="enable_reshoot", action="store_true",
        help="允许 Phase 8 对视觉缺陷/时长不足镜头补录（默认开启，最多两轮）",
    )
    reshoot_group.add_argument(
        "--disable-reshoot", dest="enable_reshoot", action="store_false",
        help="禁止付费补录；检测到必须补录的坏镜头时阻断组装",
    )
    parser.set_defaults(enable_reshoot=True)
    parser.add_argument(
        "--media-profile", choices=_core.AVAILABLE_PROFILES, default="1080p", help="编码配置（默认 1080p）"
    )
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser.add_argument("--auto-approve", action="store_true", help="自动批准人工审核节点")
    parser.add_argument("--resume-from", help="从指定阶段恢复（如 phase5）")
    parser.add_argument(
        "--phase", choices=PHASES, help="Execute single phase only (e.g., 'phase1', 'phase6')"
    )
    parser.add_argument(
        "--start-phase", choices=PHASES, help="Start from this phase (skip earlier phases)"
    )
    parser.add_argument(
        "--end-phase", choices=PHASES, help="End at this phase (skip later phases)"
    )
    return parser


def _phase_skip_list(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[float]:
    """Resolve phase range flags into the core runner's existing skip list."""
    selection = args

    if selection.phase and (selection.start_phase or selection.end_phase):
        parser.error("--phase cannot be combined with --start-phase or --end-phase")

    if not (selection.phase or selection.start_phase or selection.end_phase):
        return selection.skip_phase

    if selection.skip_phase:
        parser.error("phase selection flags cannot be combined with --skip-phase")

    if selection.phase:
        start_index = end_index = PHASES.index(selection.phase)
    else:
        start_index = PHASES.index(selection.start_phase) if selection.start_phase else 0
        end_index = PHASES.index(selection.end_phase) if selection.end_phase else len(PHASES) - 1
        if start_index > end_index:
            parser.error("--start-phase must not come after --end-phase")

    selected = set(PHASES[start_index : end_index + 1])
    skipped = [float(PHASE_NUMBERS[phase]) for phase in PHASES if phase not in selected]
    # Phase 9.5 is outside the supported phase-level monitoring sequence.
    skipped.append(9.5)
    return skipped


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    report = _core.run_pipeline(
        text=args.text,
        input_file=args.input,
        duration=args.duration,
        shot_duration=args.shot_duration,
        chain_mode=args.chain_mode,
        dry_run=args.dry_run,
        skip_phase=_phase_skip_list(args, parser),
        output_dir=args.output_dir,
        transition=args.transition,
        transition_duration=args.transition_duration,
        media_profile=args.media_profile,
        enable_reshoot=args.enable_reshoot,
        resume=args.resume,
        auto_approve=args.auto_approve,
        resume_from=args.resume_from,
    )
    _record_report_checkpoints(report, args.output_dir)
    try:
        _record_run_memory(report, args.output_dir)
    except Exception as exc:
        # Durable run memory is auxiliary and must never turn a completed
        # render/report into a failed CLI invocation.
        print(f"Warning: run memory recording skipped: {exc}", flush=True)
    selected_result = None
    if args.phase:
        selected_result = report.get("phases", {}).get(args.phase)
    phase_failed = selected_result and selected_result.get("status") == "error"
    if phase_failed:
        print(f"Phase {args.phase} failed: {selected_result.get('error', 'unknown error')}", flush=True)
    success = report["status"] in ("completed", "partial") and not phase_failed
    raise SystemExit(0 if success else 1)

# Preserve the historical ``import pipeline_runner`` module identity. This is
# important to callers that patch phase functions before invoking run_pipeline.
if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = _core
