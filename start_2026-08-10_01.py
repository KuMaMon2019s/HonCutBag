#!/usr/bin/env python3
"""HonCut 2026-08-10_01 全链路启动器 — 双 fork + setsid 完全脱离进程树"""
import os
import sys

LOG = "/tmp/honcut_2026-08-10_01.log"
PIDFILE = "/tmp/honcut_2026-08-10_01.pid"
CWD = "/Users/soda/projects/honcut"

def launch():
    # 第一次 fork
    pid = os.fork()
    if pid > 0:
        # 父进程写入 PID 文件后退出
        with open(PIDFILE, "w") as f:
            f.write(str(pid))
        print(f"✅ 管线启动器已脱离进程树，子进程 PID: {pid}")
        sys.exit(0)

    # 子进程：setsid 创建新会话，完全脱离控制终端
    os.setsid()

    # 第二次 fork（防止重新获得控制终端）
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # 孙进程：重定向日志并执行
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
        "--config", "/tmp/honcut_2026-08-10_01_config.json"
    ])

if __name__ == "__main__":
    launch()
