#!/usr/bin/env python3
"""Run HonCut phases sequentially and persist monitoring progress."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
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

#: Phase subprocesses currently running, so a SIGINT/SIGTERM on the
#: orchestrator terminates them too instead of leaving orphaned paid work.
_ACTIVE_SUBPROCESSES: set = set()


def _terminate_children(signum, frame):
    for process in list(_ACTIVE_SUBPROCESSES):
        try:
            process.terminate()
        except OSError:
            pass
    raise SystemExit(128 + int(signum))


signal.signal(signal.SIGINT, _terminate_children)
signal.signal(signal.SIGTERM, _terminate_children)

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION  # noqa: E402
from utils.phase_policy import get_policy  # noqa: E402
from utils.video_capabilities import (  # noqa: E402
    get_video_capabilities,
    max_primary_story_duration,
    min_primary_story_duration,
)
from utils.canonical_visual_contracts import (  # noqa: E402
    CHARACTER_VISUAL_POLICIES,
    SOURCE_DERIVED_POLICY,
    SYNTHETIC_STYLIZED_POLICY,
)


def _normalize_shot_duration(config: dict) -> int:
    """Validate and clamp the optional per-shot duration from config."""
    shot_duration = config.get("shot_duration", AVG_SHOT_DURATION)
    if isinstance(shot_duration, bool) or not isinstance(shot_duration, int):
        raise ValueError("config shot_duration must be an integer number of seconds")
    profile = get_video_capabilities(
        model=config.get("video_model"),
        provider=config.get("video_provider"),
    )
    minimum = int(min_primary_story_duration(profile))
    maximum = int(max_primary_story_duration(profile))
    clamped = max(minimum, min(maximum, shot_duration))
    if clamped != shot_duration:
        print(
            f"Warning: shot_duration {shot_duration}s is outside "
            f"[{minimum}, {maximum}] for {profile.name}; clamped to {clamped}s",
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


def _normalize_transition_duration(config: dict) -> float:
    """Validate the Phase 8 transition duration before spawning children."""

    try:
        duration = float(config.get("transition_duration", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("config transition_duration must be a number") from exc
    if duration < 0:
        raise ValueError("config transition_duration must not be negative")
    config["transition_duration"] = duration
    return duration


def _normalize_character_visual_policy(config: dict) -> str:
    """Migrate an old config boolean at this CLI boundary, then remove it."""
    explicit = config.get("character_visual_policy")
    legacy_present = "no_real_person" in config
    legacy = config.pop("no_real_person", None)
    if explicit is not None and legacy_present:
        raise ValueError(
            "config character_visual_policy conflicts with legacy person flag"
        )
    if explicit is None:
        explicit = (
            SYNTHETIC_STYLIZED_POLICY
            if legacy is True
            else SOURCE_DERIVED_POLICY
        )
    if explicit not in CHARACTER_VISUAL_POLICIES:
        raise ValueError("config character_visual_policy is unsupported")
    config["character_visual_policy"] = explicit
    return explicit


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


def _merge_phase_report(report_path: Path, existing_report: dict, phase: str) -> dict:
    """Merge the child report and return the selected phase's real outcome."""
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
    return current_result if isinstance(current_result, dict) else {}


def _set_report_status(report_path: Path, status: str) -> None:
    report = _read_json(report_path, {"phases": {}})
    report["status"] = status
    if status == "completed":
        report.pop("error", None)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)


def _stream_subprocess(cmd: list, log_path: Path, cwd: Path, env: dict, monitor=None) -> dict:
    """Run ``cmd`` with live, line-buffered log capture.

    Output is tee'd to ``log_path`` as it arrives (so external monitors can
    watch a phase in real time) while a bounded in-memory tail is kept to
    preserve the historical result contract (stdout tail / stderr tail).

    ``monitor`` is an optional :class:`PhaseMonitor` whose watchdog thread
    lives exactly as long as this subprocess (created when the phase starts,
    joined when it ends) — one sentinel per phase, never a global one.
    """
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        _ACTIVE_SUBPROCESSES.add(process)
        monitor_stop = threading.Event()
        monitor_thread = None
        if monitor is not None:
            monitor_thread = threading.Thread(
                target=monitor.run, args=(process, monitor_stop), daemon=True
            )
            monitor_thread.start()
        try:
            def _drain(stream, sink: list[str]) -> None:
                for line in iter(stream.readline, ""):
                    sink.append(line)
                    if len(sink) > 4000:
                        del sink[:2000]
                    log_file.write(line)

            stdout_thread = threading.Thread(
                target=_drain, args=(process.stdout, stdout_chunks), daemon=True
            )
            stderr_thread = threading.Thread(
                target=_drain, args=(process.stderr, stderr_chunks), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            stdout_thread.join()
            stderr_thread.join()
            returncode = process.wait()
        finally:
            monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=5.0)
            _ACTIVE_SUBPROCESSES.discard(process)

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)
    return {"returncode": returncode, "stdout": stdout_text, "stderr": stderr_text}


class PhaseMonitor:
    """Per-phase stall sentinel.

    Lifecycle is bound to one phase subprocess: the orchestrator creates it
    when the phase starts and joins it when the phase ends (the "create a
    sentinel per stage, retire it on stage exit" contract).

    Healthy heartbeat = the newest mtime among events.jsonl, progress.json,
    the live phase log, and any artifact matching the phase policy. Past
    ``soft_stall_s`` an alert file is written (observation only, never a
    kill); past ``hard_stall_s`` the subprocess receives SIGTERM so the
    pipeline fails closed at a resumable checkpoint. A grace window covers
    cold starts where no artifact exists yet.
    """

    def __init__(
        self,
        output_dir: Path,
        phase: str,
        policy,
        log_path: Path,
        grace_s: int = 60,
        interval_s: float = 15.0,
    ):
        self.output_dir = Path(output_dir)
        self.phase = phase
        self.policy = policy
        self.log_path = Path(log_path)
        self.grace_s = grace_s
        self.interval_s = interval_s
        self.alert_file = self.output_dir / "monitor_alert.json"
        self.stall_killed = False
        self.kill_reason = ""

    def heartbeat_paths(self) -> list:
        candidates = [
            self.output_dir / "events.jsonl",
            self.output_dir / "progress.json",
            self.log_path,
        ]
        for pattern in self.policy.artifacts:
            candidates.extend(self.output_dir.glob(pattern))
        return [p for p in candidates if p.exists()]

    def newest_heartbeat_age(self, now=None) -> float:
        import time as _time

        now = now if now is not None else _time.time()
        newest = 0.0
        for path in self.heartbeat_paths():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            newest = max(newest, mtime)
        if newest == 0.0:
            # Nothing to observe yet — treat as "no heartbeat recorded" and
            # let the caller's grace window decide.
            return float("inf")
        return max(0.0, now - newest)

    def _write_alert(self, age: float) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "phase": self.phase,
            "kind": "soft_stall",
            "heartbeat_age_s": round(age, 1),
            "soft_stall_s": self.policy.soft_stall_s,
            "hard_stall_s": self.policy.hard_stall_s,
            "note": "observation only; pipeline not killed",
        }
        _write_progress(self.alert_file, payload)

    def _clear_alert(self) -> None:
        try:
            self.alert_file.unlink(missing_ok=True)
        except OSError:
            pass

    def run(self, process, stop_event) -> None:
        import time as _time

        started = _time.monotonic()
        while not stop_event.wait(self.interval_s):
            if process.poll() is not None:
                break
            elapsed = _time.monotonic() - started
            if elapsed < self.grace_s:
                continue
            age = self.newest_heartbeat_age()
            if age > self.policy.hard_stall_s:
                self.stall_killed = True
                self.kill_reason = (
                    f"hard stall: no heartbeat for {age:.0f}s "
                    f"(threshold {self.policy.hard_stall_s}s)"
                )
                self._write_alert(age)
                try:
                    process.terminate()
                except OSError:
                    pass
                break
            if age > self.policy.soft_stall_s:
                self._write_alert(age)
            else:
                self._clear_alert()
        self._clear_alert()


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
        config.get("media_profile", "480p"),
        "--transition-duration",
        str(config.get("transition_duration", 0.5)),
    ]
    if config.get("project_id"):
        cmd.extend(["--project-id", str(config["project_id"])])
    if config.get("character_library_dir"):
        cmd.extend(
            ["--character-library-dir", str(config["character_library_dir"])]
        )
    if config.get("transition"):
        cmd.extend(["--transition", str(config["transition"])])
    if config.get("auto_approve"):
        cmd.append("--auto-approve")
    if config.get("dry_run"):
        cmd.append("--dry-run")
    if config.get("chain_mode"):
        cmd.append("--chain-mode")
    cmd.extend(
        [
            "--character-visual-policy",
            config.get("character_visual_policy", SOURCE_DERIVED_POLICY),
        ]
    )
    if config.get("_resume"):
        cmd.append("--resume")
    if config.get("_resume_from"):
        cmd.extend(["--resume-from", str(config["_resume_from"])])
    if config.get("_accept_code_change"):
        cmd.append("--accept-code-change")
    cmd.append(
        "--enable-reshoot"
        if config.get("enable_reshoot", True)
        else "--disable-reshoot"
    )

    log_dir = Path(config["output_dir"]) / "phase_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{phase}_run.log"

    # Per-phase sentinel: created here, retired when the subprocess exits.
    policy = get_policy(phase, config.get("monitor_overrides"))
    monitor = PhaseMonitor(
        output_dir=Path(config["output_dir"]),
        phase=phase,
        policy=policy,
        log_path=log_path,
    )

    # Child reporters append to the shared events.jsonl instead of clearing
    # it, preserving cross-phase history for monitors and post-mortems.
    child_env = {**os.environ, "HONCUT_APPEND_EVENTS": "1"}
    if "video_provider" in config:
        child_env["VIDEO_PROVIDER"] = str(config["video_provider"])
    if "video_model" in config:
        child_env["SEEDANCE_MODEL"] = str(config["video_model"])
    result = _stream_subprocess(cmd, log_path, RUNNER.parent, child_env, monitor=monitor)
    authoritative_result = _merge_phase_report(report_path, existing_report, phase)
    phase_result = {
        "phase": phase,
        "exit_code": result["returncode"],
        "stdout": result["stdout"][-2000:],
        "stderr": result["stderr"][-1000:],
        "log_path": str(log_path),
        "timestamp": datetime.now().isoformat(),
    }
    if authoritative_result.get("status"):
        phase_result["phase_status"] = authoritative_result["status"]
    if authoritative_result.get("reason"):
        phase_result["phase_reason"] = authoritative_result["reason"]
    if monitor.stall_killed:
        phase_result["stall_killed"] = True
        phase_result["stall_reason"] = monitor.kill_reason
        print(f"🚨 {phase} monitor: {monitor.kill_reason}", flush=True)
    return phase_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON config file")
    parser.add_argument("--resume-from", choices=PHASES, help="Resume from specific phase")
    parser.add_argument(
        "--accept-code-change",
        action="store_true",
        help="Explicitly admit a code-only identity change at --resume-from",
    )
    args = parser.parse_args()

    if args.accept_code_change and not args.resume_from:
        parser.error("--accept-code-change requires --resume-from")

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
        _normalize_transition_duration(config)
        _normalize_character_visual_policy(config)
    except ValueError as error:
        parser.error(str(error))

    # Match direct runner usage from pipeline/src: relative input/output paths
    # are interpreted from that directory for both execution and monitoring.
    for path_key in ("input", "output_dir", "character_library_dir"):
        if not config.get(path_key):
            continue
        configured_path = Path(config[path_key]).expanduser()
        if not configured_path.is_absolute():
            configured_path = RUNNER.parent / configured_path
        config[path_key] = str(configured_path.resolve())
    config["_resume"] = bool(args.resume_from)

    progress_file = Path(config["output_dir"]) / "phase_progress.json"
    phases = PHASES[PHASES.index(args.resume_from) :] if args.resume_from else PHASES
    code_change_acceptance_phase = phases[0] if args.accept_code_change else None
    results = _resume_results(Path(config["output_dir"]), progress_file, args.resume_from) if args.resume_from else []
    _write_progress(
        progress_file,
        {
            "results": results,
            "current_phase": None,
            "status": "pending",
            "phases": PHASES,
            "dry_run": bool(config.get("dry_run")),
        },
    )

    for phase in phases:
        # Admit the transition exactly once. Later children resume normally so
        # a source edit during the monitored run cannot be silently accepted.
        config["_accept_code_change"] = phase == code_change_acceptance_phase
        # Every child must invalidate its own completed checkpoint before it
        # runs. Passing only --resume would make a completed child report a
        # synthetic success without executing any phase code.
        config["_resume_from"] = phase if args.resume_from else None
        print(f"\n{'=' * 60}\n  Running {phase}...\n{'=' * 60}\n", flush=True)
        _write_progress(
            progress_file,
            {
                "results": results,
                "current_phase": phase,
                "status": "running",
                "phases": PHASES,
                "dry_run": bool(config.get("dry_run")),
            },
        )

        result = run_phase(phase, config)
        results.append(result)
        state = "failed" if result["exit_code"] != 0 else "running"
        current_phase = phase if result["exit_code"] != 0 else None
        _write_progress(
            progress_file,
            {
                "results": results,
                "current_phase": current_phase,
                "status": state,
                "phases": PHASES,
                "dry_run": bool(config.get("dry_run")),
            },
        )

        if result["exit_code"] != 0:
            _set_report_status(Path(config["output_dir"]) / "pipeline_report.json", "failed")
            print(f"\n❌ {phase} FAILED!", flush=True)
            print(f"Exit code: {result['exit_code']}", flush=True)
            print(f"Stdout tail: {result['stdout'][-1500:]}", flush=True)
            print(f"Stderr tail: {result['stderr'][-500:]}", flush=True)
            print(f"Error: {result['stderr'][:500]}", flush=True)
            raise SystemExit(1)

        phase_status = result.get("phase_status", "unknown")
        if phase_status == "skipped":
            reason = result.get("phase_reason", "unspecified")
            print(f"\n⏭️ {phase} skipped ({reason})", flush=True)
        elif phase_status in {"success", "completed", "done"}:
            print(f"\n✅ {phase} completed", flush=True)
        else:
            print(
                f"\nℹ️ {phase} subprocess exited successfully; "
                f"authoritative outcome is {phase_status}",
                flush=True,
            )

    _write_progress(
        progress_file,
        {
            "results": results,
            "current_phase": None,
            "status": "completed",
            "phases": PHASES,
            "dry_run": bool(config.get("dry_run")),
        },
    )
    _set_report_status(Path(config["output_dir"]) / "pipeline_report.json", "completed")
    final_message = (
        "Dry-run structural validation completed; skipped production operations "
        "are not production evidence."
        if config.get("dry_run")
        else "All phases completed successfully!"
    )
    print(f"\n{'=' * 60}\n  {final_message}\n{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
