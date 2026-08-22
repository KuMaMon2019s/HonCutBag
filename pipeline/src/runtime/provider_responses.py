"""Validated response envelopes for external video providers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError


class ProviderResponseError(RuntimeError):
    """A provider returned JSON that cannot satisfy the transport contract."""


class VideoSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("task_id", "id"),
    )


class SeedanceTaskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("task_id", "id"),
    )
    status: Literal[
        "queued",
        "pending",
        "running",
        "succeeded",
        "failed",
        "error",
        "cancelled",
    ]
    content: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


def parse_video_submission(
    payload: Any,
    *,
    provider_id: str,
) -> VideoSubmissionResponse:
    try:
        return VideoSubmissionResponse.model_validate(payload)
    except ValidationError as error:
        raise ProviderResponseError(
            f"{provider_id} submission response is missing a valid task ID"
        ) from error


def parse_seedance_task(payload: Any) -> SeedanceTaskResponse:
    try:
        return SeedanceTaskResponse.model_validate(payload)
    except ValidationError as error:
        raise ProviderResponseError("Seedance task response has an invalid schema") from error


__all__ = [
    "ProviderResponseError",
    "SeedanceTaskResponse",
    "VideoSubmissionResponse",
    "parse_seedance_task",
    "parse_video_submission",
]
