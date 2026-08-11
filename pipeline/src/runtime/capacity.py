"""Provider/media capacity configuration and local slot accounting."""

from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path


class CapacityUnavailableError(RuntimeError):
    """A provider/media lane has no runnable capacity."""


class CapacityWaitTimeoutError(TimeoutError):
    """A shared provider/media slot did not become available in time."""


class CapacityLeaseLostError(RuntimeError):
    """A shared slot lease disappeared or could no longer be renewed."""


def _read_positive_capacity(variable: str, fallback: int) -> int:
    raw = os.environ.get(variable)
    if raw is None:
        return fallback
    try:
        capacity = int(raw)
    except ValueError as error:
        raise ValueError(f"{variable} must be a positive integer, got {raw!r}") from error
    if capacity < 1:
        raise ValueError(f"{variable} must be at least 1, got {capacity}")
    return capacity


def _read_positive_seconds(variable: str, fallback: float) -> float:
    raw = os.environ.get(variable)
    if raw is None:
        return fallback
    try:
        seconds = float(raw)
    except ValueError as error:
        raise ValueError(f"{variable} must be a positive number, got {raw!r}") from error
    if seconds <= 0:
        raise ValueError(f"{variable} must be greater than 0, got {seconds}")
    return seconds


def default_capacity_lease_path() -> Path:
    """Return the machine-local database shared by independent HonCut runs."""

    configured = os.environ.get("HONCUT_CAPACITY_DB")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".honcut" / "provider-capacity.db"


class CapacityTable:
    """Read-only concurrency limits keyed by provider and media type."""

    def __init__(
        self,
        limits: Mapping[tuple[str, str], int],
        defaults: Mapping[str, int],
    ) -> None:
        self._limits = dict(limits)
        self._defaults = dict(defaults)
        for key, capacity in self._limits.items():
            if capacity < 0:
                raise ValueError(f"capacity for {key!r} must not be negative")
        for media_type, capacity in self._defaults.items():
            if capacity < 0:
                raise ValueError(
                    f"default capacity for {media_type!r} must not be negative"
                )

    @classmethod
    def for_seedance_video(cls, fallback: int) -> CapacityTable:
        if fallback < 1:
            raise ValueError(
                f"video concurrency fallback must be at least 1, got {fallback}"
            )
        capacity = _read_positive_capacity(
            "HONCUT_SEEDANCE_VIDEO_CONCURRENCY", fallback
        )
        return cls({("seedance", "video"): capacity}, {"video": fallback})

    def get(self, provider_id: str, media_type: str) -> int:
        known_providers = {provider for provider, _ in self._limits}
        if provider_id in known_providers:
            return self._limits.get((provider_id, media_type), 0)
        return self._defaults.get(media_type, 0)


class SlotTable:
    """Thread-safe in-process occupancy for provider/media capacity lanes."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._occupants: dict[tuple[str, str], set[str]] = {}

    def acquire(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise CapacityUnavailableError(
                f"no capacity for provider={provider_id!r}, media={media_type!r}"
            )
        key = (provider_id, media_type)
        with self._condition:
            while True:
                bucket = self._occupants.setdefault(key, set())
                if task_id in bucket:
                    raise RuntimeError(
                        f"task {task_id!r} already occupies provider={provider_id!r}, "
                        f"media={media_type!r}"
                    )
                if len(bucket) < capacity:
                    bucket.add(task_id)
                    return
                self._condition.wait()

    def release(self, provider_id: str, media_type: str, task_id: str) -> None:
        key = (provider_id, media_type)
        with self._condition:
            bucket = self._occupants.get(key)
            if bucket is None:
                return
            bucket.discard(task_id)
            if not bucket:
                del self._occupants[key]
            self._condition.notify_all()

    def occupied(self, provider_id: str, media_type: str) -> int:
        with self._condition:
            return len(self._occupants.get((provider_id, media_type), ()))

    @contextmanager
    def reserve(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
    ) -> Iterator[None]:
        self.acquire(provider_id, media_type, task_id, capacity=capacity)
        try:
            yield
        finally:
            self.release(provider_id, media_type, task_id)


@dataclass(frozen=True)
class CapacityLease:
    """One machine-local provider/media slot held by a running task."""

    lease_id: str
    provider_id: str
    media_type: str
    task_id: str
    slot_index: int


class CrossProcessSlotTable:
    """SQLite leases that coordinate provider capacity across processes."""

    def __init__(
        self,
        database_path: Path,
        *,
        lease_ttl: float | None = None,
        heartbeat_interval: float | None = None,
        poll_interval: float = 0.1,
        owner_id: str | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.lease_ttl = (
            _read_positive_seconds("HONCUT_CAPACITY_LEASE_TTL", 120.0)
            if lease_ttl is None
            else lease_ttl
        )
        self.heartbeat_interval = (
            min(self.lease_ttl / 3, 30.0)
            if heartbeat_interval is None
            else heartbeat_interval
        )
        self.poll_interval = poll_interval
        self.owner_id = owner_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        )
        if self.lease_ttl <= 0:
            raise ValueError("lease_ttl must be greater than 0")
        if not 0 < self.heartbeat_interval < self.lease_ttl:
            raise ValueError("heartbeat_interval must be between 0 and lease_ttl")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            timeout=5,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capacity_leases (
                    lease_id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    slot_index INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    UNIQUE(provider_id, media_type, slot_index)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_capacity_leases_expiry
                ON capacity_leases(provider_id, media_type, expires_at)
                """
            )

    def _try_acquire(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
        lease_id: str,
    ) -> CapacityLease | None:
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    DELETE FROM capacity_leases
                    WHERE provider_id = ? AND media_type = ? AND expires_at <= ?
                    """,
                    (provider_id, media_type, now),
                )
                rows = connection.execute(
                    """
                    SELECT slot_index
                    FROM capacity_leases
                    WHERE provider_id = ? AND media_type = ?
                    """,
                    (provider_id, media_type),
                ).fetchall()
                if len(rows) >= capacity:
                    connection.commit()
                    return None
                occupied_slots = {row[0] for row in rows}
                slot_index = next(
                    index for index in range(capacity) if index not in occupied_slots
                )
                connection.execute(
                    """
                    INSERT INTO capacity_leases (
                        lease_id, provider_id, media_type, task_id, owner_id,
                        slot_index, acquired_at, heartbeat_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease_id,
                        provider_id,
                        media_type,
                        task_id,
                        self.owner_id,
                        slot_index,
                        now,
                        now,
                        now + self.lease_ttl,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return CapacityLease(
            lease_id=lease_id,
            provider_id=provider_id,
            media_type=media_type,
            task_id=task_id,
            slot_index=slot_index,
        )

    def acquire(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
        wait_timeout: float | None = None,
    ) -> CapacityLease:
        if capacity < 1:
            raise CapacityUnavailableError(
                f"no capacity for provider={provider_id!r}, media={media_type!r}"
            )
        if wait_timeout is not None and wait_timeout <= 0:
            raise ValueError("wait_timeout must be greater than 0")
        lease_id = uuid.uuid4().hex
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        while True:
            lease = self._try_acquire(
                provider_id,
                media_type,
                task_id,
                capacity=capacity,
                lease_id=lease_id,
            )
            if lease is not None:
                return lease
            if deadline is not None and time.monotonic() >= deadline:
                raise CapacityWaitTimeoutError(
                    f"timed out waiting for provider={provider_id!r}, "
                    f"media={media_type!r}, capacity={capacity}"
                )
            time.sleep(self.poll_interval)

    def renew(self, lease: CapacityLease) -> bool:
        now = time.time()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE capacity_leases
                SET heartbeat_at = ?, expires_at = ?
                WHERE lease_id = ? AND owner_id = ?
                """,
                (now, now + self.lease_ttl, lease.lease_id, self.owner_id),
            )
            return cursor.rowcount == 1

    def release(self, lease: CapacityLease) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                DELETE FROM capacity_leases
                WHERE lease_id = ? AND owner_id = ?
                """,
                (lease.lease_id, self.owner_id),
            )

    def occupied(self, provider_id: str, media_type: str) -> int:
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    DELETE FROM capacity_leases
                    WHERE provider_id = ? AND media_type = ? AND expires_at <= ?
                    """,
                    (provider_id, media_type, now),
                )
                count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM capacity_leases
                    WHERE provider_id = ? AND media_type = ?
                    """,
                    (provider_id, media_type),
                ).fetchone()[0]
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return count

    @contextmanager
    def reserve(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
        wait_timeout: float | None = None,
    ) -> Iterator[CapacityLease]:
        lease = self.acquire(
            provider_id,
            media_type,
            task_id,
            capacity=capacity,
            wait_timeout=wait_timeout,
        )
        stop_heartbeat = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            while not stop_heartbeat.wait(self.heartbeat_interval):
                try:
                    if not self.renew(lease):
                        raise CapacityLeaseLostError(
                            f"capacity lease {lease.lease_id} was lost"
                        )
                except Exception as error:
                    heartbeat_errors.append(error)
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"capacity-heartbeat-{lease.lease_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield lease
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=self.heartbeat_interval + 1)
            self.release(lease)
        if heartbeat_errors:
            raise CapacityLeaseLostError(
                f"failed to keep capacity lease {lease.lease_id} alive"
            ) from heartbeat_errors[0]
