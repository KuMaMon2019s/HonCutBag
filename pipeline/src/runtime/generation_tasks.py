"""SQLite persistence for generation tasks that must survive process restarts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = ("queued", "running", "submission_uncertain")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _encode_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("generation task JSON must contain an object")
    return decoded


@dataclass(frozen=True)
class GenerationTask:
    task_id: str
    run_id: str
    task_type: str
    media_type: str
    resource_id: str
    status: str
    payload: dict[str, Any]
    outcome: dict[str, Any]
    error_message: str | None
    provider_id: str | None
    provider_job_id: str | None
    provider_endpoint: str | None
    attempt_count: int
    queued_at: str
    started_at: str | None
    updated_at: str
    finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GenerationTask:
        return cls(
            task_id=row["task_id"],
            run_id=row["run_id"],
            task_type=row["task_type"],
            media_type=row["media_type"],
            resource_id=row["resource_id"],
            status=row["status"],
            payload=_decode_json(row["payload_json"]),
            outcome=_decode_json(row["outcome_json"]),
            error_message=row["error_message"],
            provider_id=row["provider_id"],
            provider_job_id=row["provider_job_id"],
            provider_endpoint=row["provider_endpoint"],
            attempt_count=int(row["attempt_count"]),
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )


@dataclass(frozen=True)
class EnqueuedTask:
    task: GenerationTask
    deduped: bool


class GenerationTaskStore:
    """Own the small SQLite task ledger for one HonCut output directory."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'queued', 'running', 'submission_uncertain',
                        'succeeded', 'failed'
                    )),
                    payload_json TEXT NOT NULL,
                    outcome_json TEXT,
                    error_message TEXT,
                    provider_id TEXT,
                    provider_job_id TEXT,
                    provider_endpoint TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_tasks_status_queued
                ON generation_tasks(status, queued_at)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_tasks_active_dedupe
                ON generation_tasks(run_id, task_type, resource_id)
                WHERE status IN (
                    'queued', 'running', 'submission_uncertain'
                )
                """
            )

    def enqueue(
        self,
        *,
        run_id: str,
        task_type: str,
        media_type: str,
        resource_id: str,
        payload: dict[str, Any],
        provider_id: str | None = None,
    ) -> EnqueuedTask:
        """Insert one active task or return the matching active task."""
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._find_active(
                connection,
                run_id=run_id,
                task_type=task_type,
                resource_id=resource_id,
            )
            if existing is not None:
                connection.commit()
                return EnqueuedTask(task=existing, deduped=True)

            task_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO generation_tasks (
                    task_id, run_id, task_type, media_type, resource_id, status,
                    payload_json, provider_id, queued_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    task_type,
                    media_type,
                    resource_id,
                    _encode_json(payload),
                    provider_id,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError(f"generation task {task_id} disappeared after enqueue")
        return EnqueuedTask(task=GenerationTask.from_row(row), deduped=False)

    def _find_active(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        task_type: str,
        resource_id: str,
    ) -> GenerationTask | None:
        placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
        row = connection.execute(
            f"""
            SELECT * FROM generation_tasks
            WHERE run_id = ? AND task_type = ? AND resource_id = ?
              AND status IN ({placeholders})
            ORDER BY queued_at DESC
            LIMIT 1
            """,
            (run_id, task_type, resource_id, *ACTIVE_STATUSES),
        ).fetchone()
        return GenerationTask.from_row(row) if row is not None else None

    def get(self, task_id: str) -> GenerationTask | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return GenerationTask.from_row(row) if row is not None else None

    def claim(self, task_id: str) -> GenerationTask | None:
        """Atomically move one queued task to running."""
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'running', started_at = COALESCE(started_at, ?),
                    updated_at = ?, attempt_count = attempt_count + 1
                WHERE task_id = ? AND status = 'queued'
                """,
                (now, now, task_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def persist_provider_job(
        self,
        task_id: str,
        *,
        provider_job_id: str,
        provider_endpoint: str,
    ) -> GenerationTask:
        if not provider_job_id:
            raise ValueError("provider_job_id must not be empty")
        return self._update_running(
            task_id,
            """
            provider_job_id = ?, provider_endpoint = ?, error_message = NULL,
            updated_at = ?
            """,
            (provider_job_id, provider_endpoint, _utc_now()),
        )

    def note_resumable_error(self, task_id: str, message: str) -> GenerationTask:
        return self._update_running(
            task_id,
            "error_message = ?, updated_at = ?",
            (message, _utc_now()),
        )

    def mark_submission_uncertain(self, task_id: str, message: str) -> GenerationTask:
        return self._update_running(
            task_id,
            "status = 'submission_uncertain', error_message = ?, updated_at = ?",
            (message, _utc_now()),
        )

    def mark_succeeded(self, task_id: str, outcome: dict[str, Any]) -> GenerationTask:
        now = _utc_now()
        return self._update_running(
            task_id,
            """
            status = 'succeeded', outcome_json = ?, error_message = NULL,
            updated_at = ?, finished_at = ?
            """,
            (_encode_json(outcome), now, now),
        )

    def mark_failed(self, task_id: str, message: str) -> GenerationTask:
        now = _utc_now()
        return self._update_running(
            task_id,
            "status = 'failed', error_message = ?, updated_at = ?, finished_at = ?",
            (message, now, now),
        )

    def _update_running(
        self,
        task_id: str,
        assignments: str,
        parameters: tuple[Any, ...],
    ) -> GenerationTask:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"UPDATE generation_tasks SET {assignments} "
                "WHERE task_id = ? AND status = 'running'",
                (*parameters, task_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                current = self.get(task_id)
                current_status = current.status if current is not None else "missing"
                raise RuntimeError(
                    f"cannot update generation task {task_id}: status={current_status}"
                )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)
