#!/usr/bin/env python3
"""Report the status of an independently executed HonCut phase sequence."""

import argparse
import json
import subprocess
from pathlib import Path

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


def check_process(process_id: str) -> dict:
    """Check whether a background process is still running."""
    try:
        pid_number = int(process_id)
        if pid_number <= 0:
            raise ValueError("process ID must be a positive integer")
        result = subprocess.run(
            ["ps", "-p", str(pid_number), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        state = result.stdout.strip()
        return {"running": result.returncode == 0 and bool(state) and not state.startswith("Z")}
    except (ValueError, OSError) as error:
        return {"running": False, "error": str(error)}


def get_phase_status(output_dir: str) -> dict:
    """Read and summarize the current phase progress file."""
    progress_file = Path(output_dir) / "phase_progress.json"
    if not progress_file.exists():
        return {
            "status": "not_started",
            "completed_phases": [],
            "skipped_phases": [],
            "failed_phases": [],
        }

    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "unavailable",
            "error": str(error),
            "completed_phases": [],
            "skipped_phases": [],
            "failed_phases": [],
        }

    summary = {
        "status": data.get("status", "running"),
        "current_phase": data.get("current_phase"),
        "dry_run": bool(data.get("dry_run")),
        "completed_phases": [],
        "skipped_phases": [],
        "failed_phases": [],
        "total_phases": len(data.get("phases") or PHASES),
    }
    for result in data.get("results", []):
        if not isinstance(result, dict) or result.get("phase") not in PHASES:
            continue
        if result.get("exit_code", 1) != 0:
            summary["failed_phases"].append(
                {
                    "phase": result["phase"],
                    "exit_code": result.get("exit_code", 1),
                    "timestamp": result.get("timestamp", "unknown"),
                }
            )
        elif result.get("phase_status") == "skipped":
            summary["skipped_phases"].append(
                {
                    "phase": result["phase"],
                    "reason": result.get("phase_reason", "unspecified"),
                    "timestamp": result.get("timestamp", "unknown"),
                }
            )
        else:
            # Older progress files did not persist the authoritative phase
            # status. Preserve their historical exit-code interpretation.
            summary["completed_phases"].append(
                {
                    "phase": result["phase"],
                    "exit_code": 0,
                    "timestamp": result.get("timestamp", "unknown"),
                }
            )
    return summary


def format_report(status: dict) -> str:
    """Format a concise status report suitable for Discord."""
    lines = ["🐼 **HonCut Pipeline Monitor**\n"]
    completed = {item["phase"] for item in status.get("completed_phases", [])}
    skipped_items = status.get("skipped_phases", [])
    skipped = {item["phase"]: item for item in skipped_items}
    failed = {item["phase"] for item in status.get("failed_phases", [])}
    current = status.get("current_phase")

    for phase in PHASES:
        if phase in failed:
            lines.append(f"❌ {phase} FAILED")
        elif phase == current and status.get("status") == "running":
            lines.append(f"⏳ {phase} (running)")
        elif phase in skipped:
            lines.append(f"⏭️ {phase} ({skipped[phase]['reason']})")
        elif phase in completed:
            lines.append(f"✅ {phase}")
        else:
            lines.append(f"⏸️ {phase} (pending)")

    resolved = len(completed) + len(skipped)
    progress_detail = (
        f" ({len(completed)} completed, {len(skipped)} skipped)" if skipped else ""
    )
    lines.append(f"\n**Progress**: {resolved}/{len(PHASES)} phases{progress_detail}")
    if failed:
        lines.extend([f"\n❌ **Error in {next(reversed(status['failed_phases']))['phase']}**", "Suggestion: Check logs and fix before continuing"])
    elif status.get("status") == "completed" or resolved == len(PHASES):
        if status.get("dry_run"):
            lines.extend(
                [
                    "\n🎉 **Dry-run structural validation completed**",
                    "Skipped phases are not production evidence.",
                ]
            )
        else:
            lines.extend(["\n🎉 **All phases completed!**", "Next: Review output and push to GitHub"])
    elif status.get("status") in {"not_started", "unavailable"}:
        lines.append(f"\n⚠️ **Status: {status['status']}**")
    if status.get("process_running") is False and status.get("status") not in {"completed", "failed"}:
        lines.append("\n⚠️ **Orchestrator process is not running**")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--process-id", help="Background process ID to check")
    args = parser.parse_args()

    status = get_phase_status(args.output_dir)
    if args.process_id:
        status["process_running"] = check_process(args.process_id)["running"]
    print(format_report(status))

    completed = len(status.get("completed_phases", []))
    skipped = len(status.get("skipped_phases", []))
    failed = len(status.get("failed_phases", []))
    still_running = status.get("process_running", True)
    finished = status.get("status") in {"completed", "failed"}
    raise SystemExit(
        0
        if finished or completed + skipped == len(PHASES) or failed or not still_running
        else 1
    )


if __name__ == "__main__":
    main()
