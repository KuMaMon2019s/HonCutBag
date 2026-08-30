"""Narrow transport hooks installed by Runtime acceptance policy.

Provider transports import this module without depending back on Runtime.  The
hooks are process-wide so Phase worker threads share one durable request ledger.
Outside an explicitly installed scope every function is a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Callable, Iterator


BeforeProviderRequest = Callable[[dict[str, Any]], Any]
AfterProviderRequest = Callable[[Any, dict[str, Any]], None]
FailedProviderRequest = Callable[[Any, dict[str, Any]], None]


@dataclass(frozen=True)
class _ProviderRequestGuard:
    max_retries: int | None
    before_provider_request: BeforeProviderRequest | None
    after_provider_request: AfterProviderRequest | None
    failed_provider_request: FailedProviderRequest | None


_guard_lock = threading.RLock()
_active_guard: _ProviderRequestGuard | None = None


@contextmanager
def provider_request_guard_scope(
    *,
    max_retries: int | None = None,
    before_provider_request: BeforeProviderRequest | None = None,
    after_provider_request: AfterProviderRequest | None = None,
    failed_provider_request: FailedProviderRequest | None = None,
) -> Iterator[None]:
    """Install one transport guard for the isolated acceptance process."""

    global _active_guard
    if (
        max_retries is not None
        and (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        )
    ):
        raise ValueError("transport max_retries must be a non-negative integer")
    guard = _ProviderRequestGuard(
        max_retries=max_retries,
        before_provider_request=before_provider_request,
        after_provider_request=after_provider_request,
        failed_provider_request=failed_provider_request,
    )
    with _guard_lock:
        if _active_guard is not None:
            raise RuntimeError("a Provider request guard is already active")
        _active_guard = guard
    try:
        yield
    finally:
        with _guard_lock:
            if _active_guard is not guard:
                raise RuntimeError("Provider request guard ownership changed")
            _active_guard = None


def _current_guard() -> _ProviderRequestGuard | None:
    with _guard_lock:
        return _active_guard


def effective_transport_retries(default_retries: int) -> int:
    """Apply the Runtime-installed ceiling to a transport-owned retry loop."""

    if (
        isinstance(default_retries, bool)
        or not isinstance(default_retries, int)
        or default_retries < 0
    ):
        raise ValueError("transport retries must be a non-negative integer")
    guard = _current_guard()
    if guard is None or guard.max_retries is None:
        return default_retries
    return min(default_retries, guard.max_retries)


def provider_request_started(payload: dict[str, Any]) -> Any:
    """Persist safe request metadata immediately before network submission."""

    guard = _current_guard()
    if guard is None or guard.before_provider_request is None:
        return None
    return guard.before_provider_request(payload)


def provider_request_completed(token: Any, outcome: dict[str, Any]) -> None:
    """Settle a request only after its complete Provider response is usable."""

    guard = _current_guard()
    if guard is None or guard.after_provider_request is None or token is None:
        return
    guard.after_provider_request(token, outcome)


def provider_request_failed(token: Any, outcome: dict[str, Any]) -> None:
    """Persist a known failure or unresolved submission without resubmitting."""

    guard = _current_guard()
    if guard is None or guard.failed_provider_request is None or token is None:
        return
    guard.failed_provider_request(token, outcome)
