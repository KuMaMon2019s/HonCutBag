#!/usr/bin/env python3
"""HonCut pipeline CLI and backward-compatible orchestration facade.

Phase entry points live in :mod:`phases`; the implementation core is kept as
one module so the existing LangGraph nodes and monkeypatch-based integrations
continue to share a single module namespace.
"""

import argparse
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

PHASES = (
    "phase1",
    "phase2",
    "phase2_5",
    "phase3",
    "phase4",
    "phase4_5",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
)
PHASE_NUMBERS = {
    "phase1": "1",
    "phase2": "2",
    "phase2_5": "2.5",
    "phase3": "3",
    "phase4": "4",
    "phase4_5": "4.5",
    "phase5": "5",
    "phase6": "6",
    "phase7": "7",
    "phase8": "8",
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
        "--transition", choices=["crossfade", "fade", "cut"], default="crossfade", help="Phase 7 转场模式"
    )
    parser.add_argument("--transition-duration", type=float, default=0.5, help="Phase 7 转场时长（秒）")
    parser.add_argument("--enable-reshoot", action="store_true",
                        help="允许 Phase 7 时长不足时真实补录（默认关闭）")
    parser.add_argument(
        "--media-profile", choices=_core.AVAILABLE_PROFILES, default="1080p", help="编码配置（默认 1080p）"
    )
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser.add_argument("--auto-approve", action="store_true", help="自动批准人工审核节点")
    parser.add_argument("--resume-from", help="从指定阶段恢复（如 phase5）")
    parser.add_argument(
        "--phase", choices=PHASES, help="Execute single phase only (e.g., 'phase2', 'phase5')"
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
    # Phase 8.5 is outside the supported phase-level monitoring sequence.
    skipped.append(8.5)
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
    selected_result = None
    if args.phase:
        report_key = "phase1" if args.phase == "phase1" else PHASE_NUMBERS[args.phase]
        selected_result = report.get("phases", {}).get(report_key)
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
