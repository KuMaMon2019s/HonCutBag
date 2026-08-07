#!/usr/bin/env python3
"""Run HonCut phases sequentially and persist monitoring progress."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PHASES = [
    "phase2",
    "phase2_5",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
]
PIPELINE_DIR = Path(__file__).resolve().parents[1]
RUNNER = PIPELINE_DIR / "src" / "pipeline_runner.py"


def _write_progress(progress_file: Path, payload: dict) -> None:
    """Atomically replace the progress file so cron never reads partial JSON."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(progress_file)


def run_phase(phase: str, config: dict) -> dict:
    """Run a single phase and return its process status."""
    cmd = [
        sys.executable,
        str(RUNNER),
        "--input",
        str(config["input"]),
        "--duration",
        str(config["duration"]),
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

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=RUNNER.parent)
    return {
        "phase": phase,
        "exit_code": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-1000:],
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

    # Match direct runner usage from pipeline/src: relative input/output paths
    # are interpreted from that directory for both execution and monitoring.
    for path_key in ("input", "output_dir"):
        configured_path = Path(config[path_key]).expanduser()
        if not configured_path.is_absolute():
            configured_path = RUNNER.parent / configured_path
        config[path_key] = str(configured_path.resolve())

    progress_file = Path(config["output_dir"]) / "phase_progress.json"
    phases = PHASES[PHASES.index(args.resume_from) :] if args.resume_from else PHASES
    results = []
    if args.resume_from and progress_file.exists():
        try:
            previous = json.loads(progress_file.read_text(encoding="utf-8"))
            resume_index = PHASES.index(args.resume_from)
            results = [
                result
                for result in previous.get("results", [])
                if result.get("exit_code") == 0
                and result.get("phase") in PHASES[:resume_index]
            ]
        except (OSError, json.JSONDecodeError):
            results = []
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
            print(f"\n❌ {phase} FAILED!", flush=True)
            print(f"Exit code: {result['exit_code']}", flush=True)
            print(f"Error: {result['stderr'][:500]}", flush=True)
            raise SystemExit(1)

        print(f"\n✅ {phase} completed", flush=True)

    _write_progress(
        progress_file,
        {"results": results, "current_phase": None, "status": "completed", "phases": PHASES},
    )
    print(f"\n{'=' * 60}\n  All phases completed successfully!\n{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
