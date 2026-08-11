#!/usr/bin/env python3
"""HonCut 全链路启动器 — 双 fork + setsid 完全脱离进程树，gateway 重启杀不死
用法: python3 detached_pipeline_launch.py <config_path> <tag>
"""
import os
import sys

if len(sys.argv) != 3:
    print("用法: detached_pipeline_launch.py <config_path> <tag>")
    sys.exit(1)

CONFIG = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]  # e.g. 2026-08-10_02
LOG = f"/tmp/honcut_{TAG}.log"
PIDFILE = f"/tmp/honcut_{TAG}.pid"
CWD = "/Users/soda/projects/honcut"

def launch():
    pid = os.fork()
    if pid > 0:
        with open(PIDFILE, "w") as f:
            f.write(str(pid))
        print(f"✅ 管线启动器已脱离进程树，子进程 PID: {pid}")
        sys.exit(0)

    # 子进程：setsid 创建新会话
    os.setsid()

    # 第二次 fork 防重获控制终端
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    os.chdir(CWD)
    log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VIDEO_PROVIDER"] = "seedance"

    os.execvp("conda", [
        "conda", "run", "--no-capture-output", "-n", "honcut",
        "python", "-u", "pipeline/scripts/phase_orchestrator.py",
        "--config", CONFIG
    ])

if __name__ == "__main__":
    launch()
