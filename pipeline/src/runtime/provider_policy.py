"""Runtime-owned timeout, retry, cooldown, backoff, and capacity policy."""

from __future__ import annotations

import os
import sqlite3
import time
from inspect import Parameter, signature
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Callable, TypeVar

from runtime.capacity import CapacityTable
from runtime.video_provider import ProviderErrorKind, classify_provider_error


T = TypeVar("T")


def _positive_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    value = fallback if raw is None else float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    value = fallback if raw is None else int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ProviderRetryDecision:
    should_retry: bool
    error_kind: ProviderErrorKind
    delay_seconds: float = 0.0


@dataclass(frozen=True)
class ProviderExecutionPolicy:
    """One transport policy shared by Phase 6 and continuity runtimes."""

    provider_id: str
    submit_timeout_seconds: float = 30.0
    status_timeout_seconds: float = 30.0
    poll_deadline_seconds: float = 1800.0
    max_rate_limit_retries: int = 3
    initial_backoff_seconds: float = 30.0
    max_backoff_seconds: float = 120.0
    cooldown_seconds: int = 900

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id must not be empty")
        for name in (
            "submit_timeout_seconds",
            "status_timeout_seconds",
            "poll_deadline_seconds",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_rate_limit_retries < 0 or self.cooldown_seconds < 0:
            raise ValueError("retry and cooldown counts must be non-negative")

    @classmethod
    def from_environment(cls, provider_id: str) -> "ProviderExecutionPolicy":
        return cls(
            provider_id=provider_id,
            submit_timeout_seconds=_positive_float(
                "HONCUT_PROVIDER_SUBMIT_TIMEOUT_SECONDS", 30.0
            ),
            status_timeout_seconds=_positive_float(
                "HONCUT_PROVIDER_STATUS_TIMEOUT_SECONDS", 30.0
            ),
            poll_deadline_seconds=_positive_float(
                "HONCUT_PROVIDER_POLL_DEADLINE_SECONDS", 1800.0
            ),
            max_rate_limit_retries=_nonnegative_int(
                "HONCUT_PROVIDER_RATE_LIMIT_RETRIES", 3
            ),
            initial_backoff_seconds=_positive_float(
                "HONCUT_PROVIDER_BACKOFF_SECONDS", 30.0
            ),
            max_backoff_seconds=_positive_float(
                "HONCUT_PROVIDER_MAX_BACKOFF_SECONDS", 120.0
            ),
            cooldown_seconds=_nonnegative_int(
                "HONCUT_PROVIDER_COOLDOWN_SECONDS", 900
            ),
        )

    def capacity(self, fallback: int) -> int:
        if self.provider_id == "seedance":
            return CapacityTable.for_seedance_video(fallback).get(
                self.provider_id, "video"
            )
        return max(1, fallback)

    def retry_decision(
        self, error: BaseException, retries_completed: int
    ) -> ProviderRetryDecision:
        kind = classify_provider_error(error)
        retryable = kind in {ProviderErrorKind.QUOTA, ProviderErrorKind.RATE_LIMIT}
        if not retryable or retries_completed >= self.max_rate_limit_retries:
            return ProviderRetryDecision(False, kind)
        delay = min(
            self.initial_backoff_seconds * (2**retries_completed),
            self.max_backoff_seconds,
        )
        return ProviderRetryDecision(True, kind, delay)

    def wait(
        self,
        decision: ProviderRetryDecision,
        *,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not decision.should_retry:
            raise ValueError("cannot wait for a non-retry decision")
        (sleeper or time.sleep)(decision.delay_seconds)

    def execute_rate_limited(
        self,
        operation: Callable[[], T],
        *,
        sleeper: Callable[[float], None] | None = None,
    ) -> T:
        """Retry only explicit provider quota/rate rejections."""
        retries_completed = 0
        while True:
            try:
                return operation()
            except Exception as error:
                decision = self.retry_decision(error, retries_completed)
                if not decision.should_retry:
                    raise
                self.wait(decision, sleeper=sleeper)
                retries_completed += 1

    def bind_poll(
        self,
        operation: Callable[..., T],
        *,
        interval_seconds: int,
        **credentials: Any,
    ) -> Callable[[str], T]:
        """Bind runtime poll limits while retaining narrow test/legacy callables."""
        maximum_attempts = max(
            1,
            int(self.poll_deadline_seconds // interval_seconds),
        )
        policy_kwargs = {
            **credentials,
            "max_attempts": maximum_attempts,
            "interval": interval_seconds,
            "request_timeout": self.status_timeout_seconds,
        }
        parameters = signature(operation).parameters.values()
        accepts_arbitrary_keywords = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters
        )
        accepted_names = {parameter.name for parameter in parameters}
        supported_kwargs = (
            policy_kwargs
            if accepts_arbitrary_keywords
            else {
                name: value
                for name, value in policy_kwargs.items()
                if name in accepted_names
            }
        )

        def poll(provider_job_id: str) -> T:
            return operation(provider_job_id, **supported_kwargs)

        return poll


def provider_cooldown_state(
    database_path: str | Path,
    resource_id: str,
    *,
    provider_id: str,
    cooldown_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Return a no-network receipt for a recent pre-job quota rejection."""
    path = Path(database_path)
    if cooldown_seconds <= 0 or not path.is_file():
        return None
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT finished_at, error_message
            FROM generation_tasks
            WHERE resource_id = ? AND provider_id = ?
              AND provider_job_id IS NULL AND status = 'failed'
            ORDER BY finished_at DESC, queued_at DESC
            LIMIT 1
            """,
            (resource_id, provider_id),
        ).fetchone()
    if not row or not row[0] or classify_provider_error(
        RuntimeError(str(row[1] or ""))
    ) not in {ProviderErrorKind.QUOTA, ProviderErrorKind.RATE_LIMIT}:
        return None
    finished_at = datetime.fromisoformat(str(row[0]))
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    remaining = cooldown_seconds - (current - finished_at).total_seconds()
    if remaining <= 0:
        return None
    return {
        "status": "provider_cooldown",
        "provider_id": provider_id,
        "last_failed_at": finished_at.isoformat(),
        "retry_after_seconds": ceil(remaining),
        "provider_request_sent": False,
        "reason": f"recent {provider_id} quota rejection before job creation",
    }


__all__ = [
    "ProviderExecutionPolicy",
    "ProviderRetryDecision",
    "provider_cooldown_state",
]
