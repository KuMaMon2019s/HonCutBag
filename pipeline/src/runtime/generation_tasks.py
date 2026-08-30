"""SQLite persistence for generation tasks that must survive process restarts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = ("queued", "running", "submission_uncertain")
GENERATION_TASK_SCHEMA_VERSION = 3


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


def _payload_fingerprint(
    payload: dict[str, Any], explicit: str | None = None
) -> str:
    candidate = explicit or payload.get("input_fingerprint")
    if candidate is not None and str(candidate).strip():
        return str(candidate).strip()
    return hashlib.sha256(_encode_json(payload).encode("utf-8")).hexdigest()


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
    input_fingerprint: str
    output_artifact_id: str | None
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
            input_fingerprint=row["input_fingerprint"],
            output_artifact_id=row["output_artifact_id"],
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


@dataclass(frozen=True)
class GenerationTaskEvent:
    event_id: str
    task_id: str
    run_id: str
    event_type: str
    from_status: str | None
    to_status: str | None
    details: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GenerationTaskEvent:
        return cls(
            event_id=row["event_id"],
            task_id=row["task_id"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            from_status=row["from_status"],
            to_status=row["to_status"],
            details=_decode_json(row["details_json"]),
            created_at=row["created_at"],
        )


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
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > GENERATION_TASK_SCHEMA_VERSION:
                raise RuntimeError(
                    "generation task database schema is newer than this runtime: "
                    f"{version} > {GENERATION_TASK_SCHEMA_VERSION}"
                )
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
                    input_fingerprint TEXT,
                    output_artifact_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS generation_task_events (
                    event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    task_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES generation_tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_generation_task_events_task
                ON generation_task_events(task_id, event_sequence);
                CREATE INDEX IF NOT EXISTS idx_generation_task_events_run_type
                ON generation_task_events(run_id, event_type, event_sequence);
                CREATE TRIGGER IF NOT EXISTS generation_task_events_no_update
                BEFORE UPDATE ON generation_task_events BEGIN
                    SELECT RAISE(ABORT, 'generation_task_events is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS generation_task_events_no_delete
                BEFORE DELETE ON generation_task_events BEGIN
                    SELECT RAISE(ABORT, 'generation_task_events is append-only');
                END;
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(generation_tasks)"
                ).fetchall()
            }
            if "input_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE generation_tasks ADD COLUMN input_fingerprint TEXT"
                )
            if "output_artifact_id" not in columns:
                connection.execute(
                    "ALTER TABLE generation_tasks ADD COLUMN output_artifact_id TEXT"
                )
            rows = connection.execute(
                """
                SELECT task_id, payload_json, outcome_json, input_fingerprint,
                       output_artifact_id
                FROM generation_tasks
                WHERE input_fingerprint IS NULL OR input_fingerprint = ''
                   OR output_artifact_id IS NULL
                """
            ).fetchall()
            for row in rows:
                payload = _decode_json(row["payload_json"])
                outcome = _decode_json(row["outcome_json"])
                connection.execute(
                    """
                    UPDATE generation_tasks
                    SET input_fingerprint = ?,
                        output_artifact_id = COALESCE(output_artifact_id, ?)
                    WHERE task_id = ?
                    """,
                    (
                        _payload_fingerprint(payload, row["input_fingerprint"]),
                        outcome.get("output_artifact_id"),
                        row["task_id"],
                    ),
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_tasks_status_queued
                ON generation_tasks(status, queued_at)
                """
            )
            if version < 3:
                legacy_rows = connection.execute(
                    "SELECT * FROM generation_tasks ORDER BY queued_at, task_id"
                ).fetchall()
                for legacy_row in legacy_rows:
                    already_imported = connection.execute(
                        """
                        SELECT 1 FROM generation_task_events
                        WHERE task_id = ? AND event_type = 'LegacySnapshotImported'
                        """,
                        (legacy_row["task_id"],),
                    ).fetchone()
                    if already_imported is not None:
                        continue
                    self._append_event(
                        connection,
                        task_id=legacy_row["task_id"],
                        run_id=legacy_row["run_id"],
                        event_type="LegacySnapshotImported",
                        from_status=None,
                        to_status=legacy_row["status"],
                        details={
                            "attempt_count": int(legacy_row["attempt_count"]),
                            "provider_job_id_present": bool(
                                legacy_row["provider_job_id"]
                            ),
                            "snapshot_status": legacy_row["status"],
                        },
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generation_tasks_input_fingerprint
                ON generation_tasks(provider_id, input_fingerprint)
                """
            )
            # The original index omitted provider_id, so a failed Bridge task
            # captured a later direct-Seedance request for the same shot. Drop
            # it explicitly so existing databases receive the corrected key.
            connection.execute("DROP INDEX IF EXISTS idx_generation_tasks_active_dedupe")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_tasks_active_dedupe
                ON generation_tasks(
                    run_id, task_type, resource_id, COALESCE(provider_id, '')
                )
                WHERE status IN (
                    'queued', 'running', 'submission_uncertain'
                )
                """
            )
            connection.execute(
                f"PRAGMA user_version={GENERATION_TASK_SCHEMA_VERSION}"
            )

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        task_id: str,
        run_id: str,
        event_type: str,
        from_status: str | None,
        to_status: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO generation_task_events (
                event_id, task_id, run_id, event_type, from_status,
                to_status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                task_id,
                run_id,
                event_type,
                from_status,
                to_status,
                _encode_json(details or {}),
                _utc_now(),
            ),
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
        input_fingerprint: str | None = None,
    ) -> EnqueuedTask:
        """Insert one active task or return the matching active task."""
        now = _utc_now()
        resolved_fingerprint = _payload_fingerprint(payload, input_fingerprint)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._find_active(
                connection,
                run_id=run_id,
                task_type=task_type,
                resource_id=resource_id,
                provider_id=provider_id,
            )
            if existing is not None:
                if existing.payload != payload:
                    raise RuntimeError(
                        "active generation task payload changed for "
                        f"{resource_id}: task_id={existing.task_id}"
                    )
                connection.commit()
                return EnqueuedTask(task=existing, deduped=True)

            task_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO generation_tasks (
                    task_id, run_id, task_type, media_type, resource_id, status,
                    payload_json, provider_id, input_fingerprint,
                    queued_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    task_type,
                    media_type,
                    resource_id,
                    _encode_json(payload),
                    provider_id,
                    resolved_fingerprint,
                    now,
                    now,
                ),
            )
            self._append_event(
                connection,
                task_id=task_id,
                run_id=run_id,
                event_type="TaskQueued",
                from_status=None,
                to_status="queued",
                details={
                    "input_fingerprint": resolved_fingerprint,
                    "provider_id": provider_id,
                    "resource_id": resource_id,
                    "task_type": task_type,
                },
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
        provider_id: str | None,
    ) -> GenerationTask | None:
        placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
        row = connection.execute(
            f"""
            SELECT * FROM generation_tasks
            WHERE run_id = ? AND task_type = ? AND resource_id = ?
              AND provider_id IS ?
              AND status IN ({placeholders})
            ORDER BY queued_at DESC
            LIMIT 1
            """,
            (run_id, task_type, resource_id, provider_id, *ACTIVE_STATUSES),
        ).fetchone()
        return GenerationTask.from_row(row) if row is not None else None

    def find_active(
        self,
        *,
        run_id: str,
        task_type: str,
        resource_id: str,
        provider_id: str | None = None,
    ) -> GenerationTask | None:
        """Return one active task, optionally scoped to a provider."""
        placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
        provider_clause = "AND provider_id IS ?" if provider_id is not None else ""
        parameters: tuple[Any, ...] = (run_id, task_type, resource_id)
        if provider_id is not None:
            parameters += (provider_id,)
        parameters += ACTIVE_STATUSES
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM generation_tasks
                WHERE run_id = ? AND task_type = ? AND resource_id = ?
                  {provider_clause}
                  AND status IN ({placeholders})
                ORDER BY queued_at DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
        return GenerationTask.from_row(row) if row is not None else None

    def find_succeeded(
        self,
        *,
        run_id: str,
        task_type: str,
        resource_id: str,
        payload: dict[str, Any],
        provider_id: str,
    ) -> GenerationTask | None:
        """Return the newest successful task for exactly the same immutable input."""
        input_fingerprint = _payload_fingerprint(payload)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_tasks
                WHERE run_id = ? AND task_type = ? AND resource_id = ?
                  AND provider_id = ? AND payload_json = ? AND status = 'succeeded'
                  AND input_fingerprint = ?
                ORDER BY finished_at DESC, queued_at DESC
                LIMIT 1
                """,
                (
                    run_id,
                    task_type,
                    resource_id,
                    provider_id,
                    _encode_json(payload),
                    input_fingerprint,
                ),
            ).fetchone()
        return GenerationTask.from_row(row) if row is not None else None

    def find_failed(
        self,
        *,
        run_id: str,
        task_type: str,
        resource_id: str,
        payload: dict[str, Any],
        provider_id: str,
    ) -> GenerationTask | None:
        """Return the newest terminal failure for the exact immutable input."""
        input_fingerprint = _payload_fingerprint(payload)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_tasks
                WHERE run_id = ? AND task_type = ? AND resource_id = ?
                  AND provider_id = ? AND payload_json = ? AND status = 'failed'
                  AND input_fingerprint = ?
                ORDER BY finished_at DESC, queued_at DESC
                LIMIT 1
                """,
                (
                    run_id,
                    task_type,
                    resource_id,
                    provider_id,
                    _encode_json(payload),
                    input_fingerprint,
                ),
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
            claimed_row = connection.execute(
                "SELECT run_id FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            self._append_event(
                connection,
                task_id=task_id,
                run_id=claimed_row["run_id"],
                event_type="TaskClaimed",
                from_status="queued",
                to_status="running",
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def reserve_submission_attempt(
        self,
        task_id: str,
        *,
        provider_endpoint: str,
    ) -> GenerationTask:
        """Atomically reserve the one paid submission before network I/O."""
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] != "running" or row["provider_job_id"]:
                connection.rollback()
                status = row["status"] if row is not None else "missing"
                raise RuntimeError(
                    f"cannot reserve submission for task {task_id}: status={status}"
                )
            prior_attempt = connection.execute(
                """
                SELECT 1 FROM generation_task_events
                WHERE task_id = ? AND event_type = 'SubmissionAttempted'
                """,
                (task_id,),
            ).fetchone()
            if prior_attempt is not None:
                connection.rollback()
                raise RuntimeError(
                    f"submission attempt already exists for task {task_id}"
                )
            connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'submission_uncertain', provider_endpoint = ?,
                    error_message = 'provider submission reserved before network I/O',
                    updated_at = ?
                WHERE task_id = ? AND status = 'running'
                """,
                (provider_endpoint, now, task_id),
            )
            self._append_event(
                connection,
                task_id=task_id,
                run_id=row["run_id"],
                event_type="SubmissionAttempted",
                from_status="running",
                to_status="submission_uncertain",
                details={"provider_endpoint": provider_endpoint},
            )
            updated = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(updated)

    def confirm_provider_job(
        self,
        task_id: str,
        *,
        provider_job_id: str,
        provider_endpoint: str,
    ) -> GenerationTask:
        if not provider_job_id:
            raise ValueError("provider_job_id must not be empty")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] != "submission_uncertain":
                connection.rollback()
                status = row["status"] if row is not None else "missing"
                raise RuntimeError(
                    f"cannot confirm provider job for task {task_id}: status={status}"
                )
            connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'running', provider_job_id = ?, provider_endpoint = ?,
                    error_message = NULL, updated_at = ?
                WHERE task_id = ? AND status = 'submission_uncertain'
                """,
                (provider_job_id, provider_endpoint, now, task_id),
            )
            self._append_event(
                connection,
                task_id=task_id,
                run_id=row["run_id"],
                event_type="ProviderAccepted",
                from_status="submission_uncertain",
                to_status="running",
                details={
                    "provider_endpoint": provider_endpoint,
                    "provider_job_id": provider_job_id,
                },
            )
            updated = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(updated)

    def persist_provider_job(
        self,
        task_id: str,
        *,
        provider_job_id: str,
        provider_endpoint: str,
    ) -> GenerationTask:
        """Compatibility name for the canonical provider acceptance transition."""
        return self.confirm_provider_job(
            task_id,
            provider_job_id=provider_job_id,
            provider_endpoint=provider_endpoint,
        )

    def note_resumable_error(self, task_id: str, message: str) -> GenerationTask:
        return self._update_running(
            task_id,
            "error_message = ?, updated_at = ?",
            (message, _utc_now()),
            event_type="ResumableErrorRecorded",
            to_status="running",
            details={"message": message},
        )

    def mark_submission_uncertain(self, task_id: str, message: str) -> GenerationTask:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if current is None or current["status"] not in {
                "running",
                "submission_uncertain",
            }:
                connection.rollback()
                status = current["status"] if current is not None else "missing"
                raise RuntimeError(
                    f"cannot mark task {task_id} uncertain: status={status}"
                )
            connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'submission_uncertain', error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (message, now, task_id),
            )
            self._append_event(
                connection,
                task_id=task_id,
                run_id=current["run_id"],
                event_type="SubmissionOutcomeUncertain",
                from_status=current["status"],
                to_status="submission_uncertain",
                details={"message": message},
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def mark_succeeded(self, task_id: str, outcome: dict[str, Any]) -> GenerationTask:
        now = _utc_now()
        return self._update_running(
            task_id,
            """
            status = 'succeeded', outcome_json = ?, output_artifact_id = ?,
            error_message = NULL,
            updated_at = ?, finished_at = ?
            """,
            (_encode_json(outcome), outcome.get("output_artifact_id"), now, now),
            event_type="TaskSucceeded",
            to_status="succeeded",
            details={
                "output_artifact_id": outcome.get("output_artifact_id"),
                "output_sha256": outcome.get("output_sha256"),
            },
        )

    def mark_failed(
        self,
        task_id: str,
        message: str,
        *,
        provider_terminal: bool = False,
    ) -> GenerationTask:
        current = self.get(task_id)
        if (
            current is not None
            and current.provider_job_id
            and not provider_terminal
        ):
            raise RuntimeError(
                "refusing to fail a submitted provider job without an explicit "
                f"terminal provider state: task_id={task_id}, "
                f"provider_job_id={current.provider_job_id}"
            )
        if current is None or current.status not in {"running", "submission_uncertain"}:
            status = current.status if current is not None else "missing"
            raise RuntimeError(f"cannot fail generation task {task_id}: status={status}")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'failed', error_message = ?, updated_at = ?, finished_at = ?
                WHERE task_id = ? AND status = ?
                """,
                (message, now, now, task_id, current.status),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError(f"generation task {task_id} changed concurrently")
            self._append_event(
                connection,
                task_id=task_id,
                run_id=current.run_id,
                event_type=(
                    "ProviderTerminalFailure"
                    if current.provider_job_id
                    else "SubmissionRejected"
                ),
                from_status=current.status,
                to_status="failed",
                details={"message": message, "provider_terminal": provider_terminal},
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def resolve_unsubmitted_uncertain_as_failed(
        self, task_id: str, message: str
    ) -> GenerationTask:
        """Release an uncertain task only when no provider job was persisted.

        This is for a later-confirmed HTTP rejection that an older classifier
        mislabeled as submission uncertainty. A task with any provider job ID
        remains protected from resubmission.
        """
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE generation_tasks
                SET status = 'failed', error_message = ?, updated_at = ?,
                    finished_at = ?
                WHERE task_id = ? AND status = 'submission_uncertain'
                  AND provider_job_id IS NULL
                """,
                (message, now, now, task_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                current = self.get(task_id)
                current_status = current.status if current is not None else "missing"
                provider_job = current.provider_job_id if current is not None else None
                raise RuntimeError(
                    f"cannot release uncertain task {task_id}: "
                    f"status={current_status}, provider_job_id={provider_job!r}"
                )
            current = connection.execute(
                "SELECT run_id FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            self._append_event(
                connection,
                task_id=task_id,
                run_id=current["run_id"],
                event_type="SubmissionRejectedAfterReview",
                from_status="submission_uncertain",
                to_status="failed",
                details={"message": message},
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def _update_running(
        self,
        task_id: str,
        assignments: str,
        parameters: tuple[Any, ...],
        *,
        event_type: str,
        to_status: str,
        details: dict[str, Any] | None = None,
    ) -> GenerationTask:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT run_id FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
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
            self._append_event(
                connection,
                task_id=task_id,
                run_id=current["run_id"],
                event_type=event_type,
                from_status="running",
                to_status=to_status,
                details=details,
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
        return GenerationTask.from_row(row)

    def events(self, task_id: str) -> list[GenerationTaskEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generation_task_events
                WHERE task_id = ? ORDER BY event_sequence
                """,
                (task_id,),
            ).fetchall()
        return [GenerationTaskEvent.from_row(row) for row in rows]

    def submission_attempt_count(
        self,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> int:
        clauses = ["event_type = 'SubmissionAttempted'"]
        parameters: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            parameters.append(task_id)
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM generation_task_events WHERE "
                + " AND ".join(clauses),
                parameters,
            ).fetchone()[0])
