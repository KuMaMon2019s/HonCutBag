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
PrepareMediaUpload = Callable[[dict[str, Any]], tuple[Any, str]]
MediaUploadTransition = Callable[[Any, dict[str, Any]], None]
MediaUploadTimeoutResolver = Callable[[int], "MediaUploadTimeouts"]


@dataclass(frozen=True)
class MediaUploadTimeouts:
    """Primitive timeout values passed from Runtime to a media transport."""

    connect_seconds: float
    read_seconds: float
    write_seconds: float
    pool_seconds: float
    reconciliation_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "connect_seconds",
            "read_seconds",
            "write_seconds",
            "pool_seconds",
            "reconciliation_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class _ProviderRequestGuard:
    max_retries: int | None
    before_provider_request: BeforeProviderRequest | None
    after_provider_request: AfterProviderRequest | None
    failed_provider_request: FailedProviderRequest | None


@dataclass(frozen=True)
class _MediaUploadGuard:
    timeout_resolver: MediaUploadTimeoutResolver
    prepare_upload: PrepareMediaUpload | None
    submission_started: MediaUploadTransition | None
    upload_completed: MediaUploadTransition | None
    upload_failed: MediaUploadTransition | None


_guard_lock = threading.RLock()
_active_guard: _ProviderRequestGuard | None = None
_media_guard_lock = threading.RLock()
_active_media_guard: _MediaUploadGuard | None = None


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
    if max_retries is not None and (
        isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0
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


@contextmanager
def media_upload_guard_scope(
    *,
    timeout_resolver: MediaUploadTimeoutResolver,
    prepare_upload: PrepareMediaUpload | None = None,
    submission_started: MediaUploadTransition | None = None,
    upload_completed: MediaUploadTransition | None = None,
    upload_failed: MediaUploadTransition | None = None,
) -> Iterator[None]:
    """Install one Runtime-owned upload policy for all worker threads."""

    global _active_media_guard
    guard = _MediaUploadGuard(
        timeout_resolver=timeout_resolver,
        prepare_upload=prepare_upload,
        submission_started=submission_started,
        upload_completed=upload_completed,
        upload_failed=upload_failed,
    )
    with _media_guard_lock:
        if _active_media_guard is not None:
            raise RuntimeError("a media upload guard is already active")
        _active_media_guard = guard
    try:
        yield
    finally:
        with _media_guard_lock:
            if _active_media_guard is not guard:
                raise RuntimeError("media upload guard ownership changed")
            _active_media_guard = None


def _current_media_guard() -> _MediaUploadGuard | None:
    with _media_guard_lock:
        return _active_media_guard


def effective_media_upload_timeouts(
    payload_bytes: int,
    *,
    fallback: MediaUploadTimeouts,
) -> MediaUploadTimeouts:
    """Resolve payload-aware timeouts from the active Runtime policy."""

    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise ValueError("payload_bytes must be a non-negative integer")
    guard = _current_media_guard()
    if guard is None:
        return fallback
    return guard.timeout_resolver(payload_bytes)


def media_upload_prepared(payload: dict[str, Any]) -> tuple[Any, str]:
    """Read or create a durable upload record before any network operation."""

    guard = _current_media_guard()
    if guard is None or guard.prepare_upload is None:
        return None, "prepared"
    return guard.prepare_upload(payload)


def media_upload_submission_started(token: Any, payload: dict[str, Any]) -> None:
    """Persist submission uncertainty immediately before the authoritative PUT."""

    guard = _current_media_guard()
    if guard is None or guard.submission_started is None or token is None:
        return
    guard.submission_started(token, payload)


def media_upload_completed(token: Any, outcome: dict[str, Any]) -> None:
    """Settle a verified upload or content-addressed reconciliation."""

    guard = _current_media_guard()
    if guard is None or guard.upload_completed is None or token is None:
        return
    guard.upload_completed(token, outcome)


def media_upload_failed(token: Any, outcome: dict[str, Any]) -> None:
    """Persist a known rejection or unresolved upload without resubmitting."""

    guard = _current_media_guard()
    if guard is None or guard.upload_failed is None or token is None:
        return
    guard.upload_failed(token, outcome)


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
