#!/usr/bin/env python3
"""
progress_reporter.py — 管线进度报告系统

输出两个文件到 output_dir:
  - events.jsonl   追加式事件流（每行一个 JSON）
  - progress.json  当前进度快照（原子覆盖写入）

Usage:
    reporter = ProgressReporter("./output", total_phases=8)
    reporter.phase_start("phase1", "编剧引擎")
    reporter.step("phase1", "提取 47 个事件", progress_pct=20)
    reporter.phase_done("phase1", "完成", duration_s=61.3)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class ProgressReporter:
    """管线进度报告器 — events.jsonl + progress.json"""

    def __init__(self, output_dir: str, total_phases: int = 8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.output_dir / "events.jsonl"
        self.progress_file = self.output_dir / "progress.json"
        self.total_phases = total_phases
        self.phases_done: list = []
        self.start_time = time.time()
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._current_phase: Optional[str] = None
        self._current_phase_name: Optional[str] = None
        self._current_step: Optional[str] = None
        self._progress_pct: int = 0
        self._status: str = "running"

        # 清空旧的 events 文件（新管线运行 = 新日志）
        self.events_file.write_text("")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phase_start(self, phase_id: str, phase_name: str):
        """Phase 开始时调用"""
        self._current_phase = phase_id
        self._current_phase_name = phase_name
        self._current_step = "启动中..."
        self._progress_pct = self._calc_base_pct()
        self._write_event({
            "ts": self._now_iso(),
            "phase": phase_id,
            "event": "start",
            "msg": phase_name,
        })
        self._flush_progress()

    def step(self, phase_id: str, msg: str, progress_pct: Optional[int] = None):
        """Phase 内的步骤更新"""
        self._current_step = msg
        if progress_pct is not None:
            self._progress_pct = max(0, min(100, progress_pct))
        else:
            # 自动递增（在当前 phase 的基础 pct 范围内）
            base = self._calc_base_pct()
            next_pct = base + (self._calc_phase_range()) // 3
            self._progress_pct = min(next_pct, self._calc_base_pct() + self._calc_phase_range() - 1)
        self._write_event({
            "ts": self._now_iso(),
            "phase": phase_id,
            "event": "step",
            "msg": msg,
        })
        self._flush_progress()

    def phase_done(self, phase_id: str, msg: str, duration_s: Optional[float] = None):
        """Phase 完成时调用"""
        if phase_id not in self.phases_done:
            self.phases_done.append(phase_id)
        # 更新进度到该 phase 的结束百分比
        idx = self._phase_index(phase_id)
        if idx >= 0:
            self._progress_pct = int((idx + 1) / self.total_phases * 100)
        self._current_step = msg
        event: dict = {
            "ts": self._now_iso(),
            "phase": phase_id,
            "event": "done",
            "msg": msg,
        }
        if duration_s is not None:
            event["duration_s"] = round(duration_s, 2)
        self._write_event(event)
        self._flush_progress()

    def mark_completed(self):
        """管线全部完成"""
        self._status = "completed"
        self._progress_pct = 100
        self._current_step = "全部完成"
        self._write_event({
            "ts": self._now_iso(),
            "phase": "pipeline",
            "event": "completed",
            "msg": "Pipeline 全部完成",
        })
        self._flush_progress()

    def mark_failed(self, error: str):
        """管线失败"""
        self._status = "failed"
        self._current_step = f"失败: {error}"
        self._write_event({
            "ts": self._now_iso(),
            "phase": "pipeline",
            "event": "failed",
            "msg": error,
        })
        self._flush_progress()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _phase_index(self, phase_id: str) -> int:
        """返回 phase 在 PHASE_ORDER 中的索引，找不到返回 -1"""
        order = ["phase1", "phase2", "phase3", "phase4", "phase6", "phase7", "phase8", "phase9"]
        try:
            return order.index(phase_id)
        except ValueError:
            return -1

    def _calc_base_pct(self) -> int:
        """当前 phase 起始百分比"""
        idx = self._phase_index(self._current_phase) if self._current_phase else -1
        if idx < 0:
            return 0
        return int(idx / self.total_phases * 100)

    def _calc_phase_range(self) -> int:
        """当前 phase 占的百分比范围"""
        return max(1, int(100 / self.total_phases))

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _write_event(self, event: dict):
        """追加一行到 events.jsonl"""
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 进度报告不应阻断管线

    def _flush_progress(self):
        """覆盖写 progress.json（原子写入）"""
        data = {
            "current_phase": self._current_phase or "",
            "phase_name": self._current_phase_name or "",
            "step": self._current_step or "",
            "progress_pct": self._progress_pct,
            "started_at": self._started_at,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "phases_done": list(self.phases_done),
            "phases_total": self.total_phases,
            "status": self._status,
        }
        self._atomic_write_json(self.progress_file, data)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict):
        """原子写入：先写 .tmp 再 rename"""
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except OSError:
            pass
