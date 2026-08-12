#!/usr/bin/env python3
"""Run HonCut phases sequentially and persist monitoring progress."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Phase IDs renumbered to contiguous integers on 2026-08-10.
PHASES = [
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
    "phase9_5",
]
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
    "phase9_5": "9.5",
}
PIPELINE_DIR = Path(__file__).resolve().parents[1]
RUNNER = PIPELINE_DIR / "src" / "pipeline_runner.py"
if str(RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(RUNNER.parent))

from phases.phase1.adaptation_engine import (  # noqa: E402
    AVG_SHOT_DURATION,
    MAX_SHOT_DURATION,
    MIN_SHOT_DURATION,
)


def _normalize_shot_duration(config: dict) -> int:
    """Validate and clamp the optional per-shot duration from config."""
    shot_duration = config.get("shot_duration", AVG_SHOT_DURATION)
    if isinstance(shot_duration, bool) or not isinstance(shot_duration, int):
        raise ValueError("config shot_duration must be an integer number of seconds")
    clamped = max(MIN_SHOT_DURATION, min(MAX_SHOT_DURATION, shot_duration))
    if clamped != shot_duration:
        print(
            f"Warning: shot_duration {shot_duration}s is outside "
            f"[{MIN_SHOT_DURATION}, {MAX_SHOT_DURATION}]; clamped to {clamped}s",
            file=sys.stderr,
            flush=True,
        )
    config["shot_duration"] = clamped
    return clamped


def _normalize_chain_mode(config: dict) -> bool:
    """Validate the optional Seedance chain-mode switch."""
    chain_mode = config.get("chain_mode", False)
    if not isinstance(chain_mode, bool):
        raise ValueError("config chain_mode must be a boolean")
    config["chain_mode"] = chain_mode
    return chain_mode


def _normalize_enable_reshoot(config: dict) -> bool:
    """Validate bounded automatic visual/duration reshoots (enabled by default)."""
    enable_reshoot = config.get("enable_reshoot", True)
    if not isinstance(enable_reshoot, bool):
        raise ValueError("config enable_reshoot must be a boolean")
    config["enable_reshoot"] = enable_reshoot
    return enable_reshoot


def _write_progress(progress_file: Path, payload: dict) -> None:
    """Atomically replace the progress file so cron never reads partial JSON."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=progress_file.parent,
            prefix=f".{progress_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, progress_file)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError as error:
                print(f"Warning: unable to remove progress temp file: {error}", file=sys.stderr)


def _resume_results(output_dir: Path, progress_file: Path, resume_from: str) -> list[dict]:
    """Merge successful pre-resume phases from progress and pipeline reports."""
    resume_index = PHASES.index(resume_from)
    prior_phases = set(PHASES[:resume_index])
    merged: dict[str, dict] = {}

    if progress_file.exists():
        previous = _read_json(progress_file, {})
        for result in previous.get("results", []):
            phase = result.get("phase")
            if phase in prior_phases and result.get("exit_code") == 0:
                merged[phase] = result

    report_path = output_dir / "pipeline_report.json"
    if report_path.exists():
        report = _read_json(report_path, {})
        report_phases = report.get("phases", {})
        for phase in PHASES[:resume_index]:
            phase_number = PHASE_NUMBERS[phase]
            phase_report = report_phases.get(phase, report_phases.get(phase_number, {}))
            if phase_report.get("status") in {"success", "completed", "done"}:
                merged.setdefault(
                    phase,
                    {
                        "phase": phase,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "timestamp": phase_report.get("timestamp", report.get("generated_at", "historical")),
                    },
                )

    return [merged[phase] for phase in PHASES[:resume_index] if phase in merged]


def _read_json(path: Path, default: dict) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except (OSError, json.JSONDecodeError):
        return default


def _merge_phase_report(report_path: Path, existing_report: dict, phase: str) -> None:
    """Keep prior phase results when the single-phase runner rewrites its report."""
    generated_report = _read_json(report_path, {})
    phase_number = PHASE_NUMBERS[phase]
    generated_phases = generated_report.get("phases", {})
    current_result = generated_phases.get(phase_number, generated_phases.get(phase, {}))

    merged = {**existing_report, **generated_report}
    merged_phases = dict(existing_report.get("phases", {}))
    is_synthetic_skip = (
        current_result.get("status") == "skipped"
        and current_result.get("reason") == "user-specified"
    )
    if current_result and not is_synthetic_skip:
        # Canonicalize orchestrator reports to phase names and do not import the
        # runner's synthetic "skipped user-specified" entries.
        merged_phases[phase] = current_result
    merged["phases"] = merged_phases
    merged["status"] = "running"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)


def _set_report_status(report_path: Path, status: str) -> None:
    report = _read_json(report_path, {"phases": {}})
    report["status"] = status
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)


def run_phase(phase: str, config: dict) -> dict:
    """Run a single phase and return its process status."""
    report_path = Path(config["output_dir"]) / "pipeline_report.json"
    existing_report = _read_json(report_path, {"phases": {}, "status": "running"})
    cmd = [
        sys.executable,
        str(RUNNER),
        "--input",
        str(config["input"]),
        "--duration",
        str(config["duration"]),
        "--shot-duration",
        str(config["shot_duration"]),
        "--output-dir",
        str(config["output_dir"]),
        "--phase",
        phase,
        "--media-profile",
        config.get("media_profile", "720p"),
    ]
    if config.get("transition"):
        cmd.extend(["--transition", str(config["transition"])])
    if config.get("auto_approve"):
        cmd.append("--auto-approve")
    if config.get("dry_run"):
        cmd.append("--dry-run")
    if config.get("chain_mode"):
        cmd.append("--chain-mode")
    if config.get("enable_reshoot"):
        cmd.append("--enable-reshoot")

    log_dir = Path(config["output_dir"]) / "phase_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{phase}_run.log"
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=RUNNER.parent)
    log_path.write_text(
        f"=== {phase} phase run log ===\n"
        f"{result.stdout}"
        "\n=== STDERR ===\n"
        f"{result.stderr}",
        encoding="utf-8",
    )
    _merge_phase_report(report_path, existing_report, phase)
    return {
        "phase": phase,
        "exit_code": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-1000:],
        "log_path": str(log_path),
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON config file")
    parser.add_argument("--resume-from", choices=PHASES, help="Resume from specific phase")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required = {"input", "duration", "output_dir"}
    missing = sorted(required.difference(config))
    if missing:
        parser.error(f"config is missing required keys: {', '.join(missing)}")
    try:
        _normalize_shot_duration(config)
        _normalize_chain_mode(config)
        _normalize_enable_reshoot(config)
    except ValueError as error:
        parser.error(str(error))

    # Match direct runner usage from pipeline/src: relative input/output paths
    # are interpreted from that directory for both execution and monitoring.
    for path_key in ("input", "output_dir"):
        configured_path = Path(config[path_key]).expanduser()
        if not configured_path.is_absolute():
            configured_path = RUNNER.parent / configured_path
        config[path_key] = str(configured_path.resolve())

    progress_file = Path(config["output_dir"]) / "phase_progress.json"
    phases = PHASES[PHASES.index(args.resume_from) :] if args.resume_from else PHASES
    results = _resume_results(Path(config["output_dir"]), progress_file, args.resume_from) if args.resume_from else []
    _write_progress(
        progress_file,
        {"results": results, "current_phase": None, "status": "pending", "phases": PHASES},
    )

    for phase in phases:
        print(f"\n{'=' * 60}\n  Running {phase}...\n{'=' * 60}\n", flush=True)
        _write_progress(
            progress_file,
            {"results": results, "current_phase": phase, "status": "running", "phases": PHASES},
        )

        result = run_phase(phase, config)
        results.append(result)
        state = "failed" if result["exit_code"] != 0 else "running"
        current_phase = phase if result["exit_code"] != 0 else None
        _write_progress(
            progress_file,
            {"results": results, "current_phase": current_phase, "status": state, "phases": PHASES},
        )

        if result["exit_code"] != 0:
            _set_report_status(Path(config["output_dir"]) / "pipeline_report.json", "failed")
            print(f"\n❌ {phase} FAILED!", flush=True)
            print(f"Exit code: {result['exit_code']}", flush=True)
            print(f"Stdout tail: {result['stdout'][-1500:]}", flush=True)
            print(f"Stderr tail: {result['stderr'][-500:]}", flush=True)
            print(f"Error: {result['stderr'][:500]}", flush=True)
            raise SystemExit(1)

        print(f"\n✅ {phase} completed", flush=True)

    _write_progress(
        progress_file,
        {"results": results, "current_phase": None, "status": "completed", "phases": PHASES},
    )
    _set_report_status(Path(config["output_dir"]) / "pipeline_report.json", "completed")
    print(f"\n{'=' * 60}\n  All phases completed successfully!\n{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
