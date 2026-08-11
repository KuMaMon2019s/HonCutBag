#!/bin/bash
# HonCut 2026-08-10_01 全链路启动器 — setsid 完全脱离进程树，gateway 重启杀不死
cd /Users/soda/projects/honcut

LOG=/tmp/honcut_2026-08-10_01.log
PIDFILE=/tmp/honcut_2026-08-10_01.pid

# setsid 创建新会话，完全脱离父进程树；nohup 防 SIGHUP
setsid bash -c '
export PYTHONUNBUFFERED=1
export VIDEO_PROVIDER=seedance
conda run --no-capture-output -n honcut python -u pipeline/scripts/phase_orchestrator.py \
  --config /tmp/honcut_2026-08-10_01_config.json
' > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
sleep 2
echo "✅ 管线已启动 PID: $(cat $PIDFILE)"
