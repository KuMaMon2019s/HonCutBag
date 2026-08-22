"""Stable Pydantic contracts for workflow configuration and phase receipts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class PhaseStatus(StrEnum):
    """Statuses currently emitted by individual HonCut phases."""

    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"
    WARNING = "warning"


class RunStatus(StrEnum):
    """Top-level workflow lifecycle statuses."""

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class PhaseError(BaseModel):
    """Structured error evidence that routing can inspect deterministically."""

    model_config = ConfigDict(extra="forbid")

    phase: str = Field(min_length=1)
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    shot_id: str | None = None
    cause_type: str | None = None


class PhaseResult(BaseModel):
    """Compatibility envelope around existing per-phase dictionaries.

    Phase-specific metrics remain allowed while common workflow fields are
    validated. This lets schemas land before business producers are migrated.
    """

    model_config = ConfigDict(extra="allow")

    status: PhaseStatus
    duration_s: float | None = Field(default=None, ge=0)
    outputs: list[str] = Field(default_factory=list)
    error: str | None = None
    reason: str | None = None

    @field_validator("outputs", mode="before")
    @classmethod
    def normalize_outputs(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class GraphRunConfig(BaseModel):
    """Validated, serializable inputs used to seed a future graph run."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(default="pipeline_run", min_length=1)
    project_id: str = Field(default="local", min_length=1)
    input_text: str = Field(
        default="",
        validation_alias=AliasChoices("input_text", "text"),
    )
    output_dir: str = Field(min_length=1)
    target_duration_s: int = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("target_duration_s", "duration"),
    )
    shot_duration_s: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices("shot_duration_s", "shot_duration"),
    )
    dry_run: bool = False
    chain_mode: bool = False
    # Human storyboard review is permanently disabled. Keep this field for
    # checkpoint/config compatibility; the workflow treats either value as
    # an unconditional skip and new configurations serialize the true value.
    auto_approve: bool = True
    transition: str = "crossfade"
    transition_duration_s: float = Field(
        default=0.5,
        ge=0,
        validation_alias=AliasChoices("transition_duration_s", "transition_duration"),
    )
    media_profile: str = Field(default="480p", min_length=1)
    project_video_spec: dict[str, Any] = Field(default_factory=dict)
    enable_reshoot: bool = True
    resume: bool = False
    resume_from: str | None = None
    skip_phase: list[float] = Field(default_factory=list)
