"""Provider-neutral contracts for durable video generation jobs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from runtime.execution_errors import (
    ProviderEndpointChangedError,
    ProviderJobFailedError,
    ProviderPreparationError,
    SubmissionUncertainError,
)


class VideoJobState(str, Enum):
    """Normalized provider job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class ProviderErrorKind(str, Enum):
    """Stable error classes used by runtime policy and persistence."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    MODERATION = "moderation"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CAPACITY = "capacity"
    TRANSIENT = "transient"
    ENDPOINT_CHANGED = "endpoint_changed"
    SUBMISSION_UNCERTAIN = "submission_uncertain"
    JOB_FAILED = "job_failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VideoProviderCapabilities:
    """Declared provider features; policy must not infer them from names."""

    provider_id: str
    generation_modes: frozenset[str]
    supports_cancel: bool = False
    async_submission: bool = True
    paid: bool = True

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id must not be empty")
        if not self.generation_modes:
            raise ValueError("generation_modes must not be empty")


@dataclass(frozen=True)
class VideoGenerationRequest:
    """Immutable provider input bound to its precomputed semantic fingerprint."""

    resource_id: str
    input_fingerprint: str
    payload_json: str

    @classmethod
    def from_payload(
        cls,
        *,
        resource_id: str,
        input_fingerprint: str,
        payload: dict[str, Any],
    ) -> "VideoGenerationRequest":
        if not resource_id.strip():
            raise ValueError("resource_id must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", input_fingerprint):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 hex digest")
        return cls(
            resource_id=resource_id,
            input_fingerprint=input_fingerprint,
            payload_json=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise ValueError("video generation payload must decode to an object")
        return value


@dataclass(frozen=True)
class VideoSubmission:
    """Provider acknowledgement that must be persisted before polling."""

    provider_job_id: str
    provider_endpoint: str

    def __post_init__(self) -> None:
        if not self.provider_job_id.strip():
            raise ValueError("provider_job_id must not be empty")
        if not self.provider_endpoint.strip():
            raise ValueError("provider_endpoint must not be empty")


@dataclass(frozen=True)
class VideoJobStatus:
    """Normalized status result without embedding provider response bodies."""

    state: VideoJobState
    output_uri: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.state is VideoJobState.SUCCEEDED and not self.output_uri:
            raise ValueError("succeeded video job status requires output_uri")
        if self.state is VideoJobState.FAILED and not self.error_message:
            raise ValueError("failed video job status requires error_message")


class ProviderOperationUnsupported(RuntimeError):
    """The provider does not advertise the requested optional operation."""


@runtime_checkable
class VideoProvider(Protocol):
    """Narrow asynchronous video-provider boundary used by the runtime."""

    provider_id: str

    @property
    def capabilities(self) -> VideoProviderCapabilities: ...

    def submit(self, request: VideoGenerationRequest) -> VideoSubmission: ...

    def status(self, provider_job_id: str) -> VideoJobStatus: ...

    def cancel(self, provider_job_id: str) -> VideoJobStatus: ...


def classify_provider_error(error: BaseException) -> ProviderErrorKind:
    """Classify existing and provider-native exceptions deterministically."""

    if isinstance(error, SubmissionUncertainError):
        return ProviderErrorKind.SUBMISSION_UNCERTAIN
    if isinstance(error, ProviderEndpointChangedError):
        return ProviderErrorKind.ENDPOINT_CHANGED
    if isinstance(error, ProviderJobFailedError):
        return ProviderErrorKind.JOB_FAILED
    if isinstance(error, ProviderOperationUnsupported):
        return ProviderErrorKind.UNSUPPORTED
    if isinstance(error, ProviderPreparationError | ValueError | TypeError):
        return ProviderErrorKind.VALIDATION
    if isinstance(error, TimeoutError):
        return ProviderErrorKind.TIMEOUT

    message = str(error).casefold()
    if any(marker in message for marker in ("policyviolation", "privacyinformation", "moderation", "nsfw")):
        return ProviderErrorKind.MODERATION
    if any(marker in message for marker in ("quota", "insufficient balance", "credit exhausted")):
        return ProviderErrorKind.QUOTA
    if any(marker in message for marker in ("rate limit", "too many requests", "http 429", "api 429")):
        return ProviderErrorKind.RATE_LIMIT
    if any(marker in message for marker in ("unauthorized", "forbidden", "http 401", "http 403", "api 401", "api 403")):
        return ProviderErrorKind.AUTHENTICATION
    if any(marker in message for marker in ("capacity", "no slot", "concurrency limit")):
        return ProviderErrorKind.CAPACITY
    if any(marker in message for marker in ("timeout", "timed out")):
        return ProviderErrorKind.TIMEOUT
    if any(marker in message for marker in ("connection", "temporarily unavailable", "http 500", "http 502", "http 503", "http 504")):
        return ProviderErrorKind.TRANSIENT
    return ProviderErrorKind.UNKNOWN


__all__ = [
    "ProviderErrorKind",
    "ProviderOperationUnsupported",
    "VideoGenerationRequest",
    "VideoJobState",
    "VideoJobStatus",
    "VideoProvider",
    "VideoProviderCapabilities",
    "VideoSubmission",
    "classify_provider_error",
]
