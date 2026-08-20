#!/usr/bin/env python3
"""Launch the HonCut pipeline as a detached daemon.

Usage: ``python3 detached_pipeline_launch.py <config_path> <tag>``
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_SAFE_TAG = re.compile(r"^[A-Za-z0-9_.-]+$")


def build_launch_command(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str | None = None,
    resume_from: str | None = None,
    accept_code_change: bool = False,
) -> list[str]:
    """Build a portable command without assuming conda or a machine path."""

    if accept_code_change and not resume_from:
        raise ValueError("accept_code_change requires resume_from")

    root = Path(project_root).resolve()
    config = Path(config_path).expanduser().resolve()
    executable = python_executable or os.environ.get("HONCUT_PYTHON") or sys.executable
    command = [
        executable,
        "-u",
        str(root / "pipeline" / "scripts" / "phase_orchestrator.py"),
        "--config",
        str(config),
    ]
    if resume_from:
        command.extend(["--resume-from", resume_from])
    if accept_code_change:
        command.append("--accept-code-change")
    return command


def _read_daemon_pid(fd: int) -> int:
    """Read the second-fork PID announced by the intermediate child."""

    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 64)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks).decode("ascii", errors="strict").strip()
    if not payload.isdigit() or int(payload) <= 0:
        raise RuntimeError("detached launcher did not report a daemon PID")
    return int(payload)


def launch(
    config_path: str | Path,
    tag: str,
    *,
    resume_from: str | None = None,
    accept_code_change: bool = False,
) -> int:
    """Double-fork and return the actual long-running daemon PID."""

    if not _SAFE_TAG.fullmatch(tag):
        raise ValueError("tag may contain only letters, digits, '.', '_' and '-'")

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config file does not exist: {config}")

    log_path = Path("/tmp") / f"honcut_{tag}.log"
    pid_path = Path("/tmp") / f"honcut_{tag}.pid"
    command = build_launch_command(
        config,
        resume_from=resume_from,
        accept_code_change=accept_code_change,
    )
    read_fd, write_fd = os.pipe()

    first_pid = os.fork()
    if first_pid > 0:
        os.close(write_fd)
        try:
            daemon_pid = _read_daemon_pid(read_fd)
        finally:
            os.close(read_fd)
            os.waitpid(first_pid, 0)
        pid_path.write_text(str(daemon_pid), encoding="ascii")
        print(f"✅ 管线已脱离进程树，后台进程 PID: {daemon_pid}")
        print(f"   日志: {log_path}")
        return daemon_pid

    os.close(read_fd)
    try:
        os.setsid()
        daemon_pid = os.fork()
        if daemon_pid > 0:
            os.write(write_fd, str(daemon_pid).encode("ascii"))
            os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        os.chdir(PROJECT_ROOT)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
        finally:
            if log_fd > 2:
                os.close(log_fd)
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull_fd, 0)
        finally:
            if devnull_fd > 2:
                os.close(devnull_fd)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Provider and routing remain owned by the config/environment.  The
        # launcher must not silently force Seedance or bypass Bridge mode.
        os.execvpe(command[0], command, env)
    except BaseException as exc:
        try:
            os.write(2, f"detached launcher failed: {exc}\n".encode())
        finally:
            os._exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("tag")
    parser.add_argument("--resume-from")
    parser.add_argument("--accept-code-change", action="store_true")
    args = parser.parse_args(argv)
    try:
        launch(
            args.config_path,
            args.tag,
            resume_from=args.resume_from,
            accept_code_change=args.accept_code_change,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
