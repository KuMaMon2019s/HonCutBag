"""Runtime-owned Provider attempt controls for bounded live acceptance.

Normal production execution has no active override.  A dedicated acceptance
process may install one process-wide scope so Phase-owned worker threads share
the same retry ceiling and durable request callbacks.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator

from utils.provider_request_guard import provider_request_guard_scope

BeforeProviderRequest = Callable[[dict[str, Any]], Any]
AfterProviderRequest = Callable[[Any, dict[str, Any]], None]
FailedProviderRequest = Callable[[Any, dict[str, Any]], None]


_scope_lock = threading.RLock()
_active_retry_limit: int | None = None
_scope_active = False


def _validated_retry_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Provider max_retries must be a non-negative integer")
    return value


@contextmanager
def provider_attempt_scope(
    *,
    max_retries: int | None = None,
    before_provider_request: BeforeProviderRequest | None = None,
    after_provider_request: AfterProviderRequest | None = None,
    failed_provider_request: FailedProviderRequest | None = None,
) -> Iterator[None]:
    """Install one process-wide policy visible to Phase worker threads.

    The scope deliberately rejects nesting.  It is intended for the isolated
    full-chain acceptance process, not as mutable application configuration.
    """

    global _active_retry_limit, _scope_active
    retry_limit = _validated_retry_limit(max_retries)
    with _scope_lock:
        if _scope_active:
            raise RuntimeError("a Provider attempt scope is already active")
        _active_retry_limit = retry_limit
        _scope_active = True
    try:
        with provider_request_guard_scope(
            max_retries=retry_limit,
            before_provider_request=before_provider_request,
            after_provider_request=after_provider_request,
            failed_provider_request=failed_provider_request,
        ):
            yield
    finally:
        with _scope_lock:
            if not _scope_active or _active_retry_limit != retry_limit:
                raise RuntimeError("Provider attempt scope ownership changed")
            _active_retry_limit = None
            _scope_active = False


def _current_retry_limit() -> int | None:
    with _scope_lock:
        return _active_retry_limit if _scope_active else None


def effective_provider_retries(default_retries: int) -> int:
    """Return the Runtime-capped retry count for one Provider operation."""

    default = _validated_retry_limit(default_retries)
    assert default is not None
    retry_limit = _current_retry_limit()
    if retry_limit is None:
        return default
    return min(default, retry_limit)


def effective_provider_attempts(default_attempts: int) -> int:
    """Return a positive attempt count after applying the retry ceiling."""

    if (
        isinstance(default_attempts, bool)
        or not isinstance(default_attempts, int)
        or default_attempts < 1
    ):
        raise ValueError("Provider attempts must be a positive integer")
    return effective_provider_retries(default_attempts - 1) + 1
