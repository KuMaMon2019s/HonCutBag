"""Strict artifact identity and run-manifest schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


class ArtifactRef(BaseModel):
    """Immutable metadata for one filesystem artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    schema_version: Literal[1] = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    artifact_type: str = Field(alias="type", min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    relative_path: str = Field(min_length=1)
    producer_node: str = Field(min_length=1)
    producer_task_id: str | None = Field(default=None, min_length=1)
    parent_artifact_ids: tuple[str, ...] = ()
    created_at: datetime

    @field_validator(
        "artifact_id",
        "run_id",
        "project_id",
        "artifact_type",
        "producer_node",
        "producer_task_id",
    )
    @classmethod
    def reject_surrounding_whitespace(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("artifact identifiers must not contain surrounding whitespace")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or value in {"", "."} or ".." in path.parts:
            raise ValueError("artifact path must remain relative to its run directory")
        return path.as_posix()

    @field_validator("parent_artifact_ids")
    @classmethod
    def validate_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("parent artifact IDs must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("parent artifact IDs must be unique")
        return value


class ArtifactManifest(BaseModel):
    """Versioned, project-bound registry for one run's artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = ARTIFACT_MANIFEST_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    artifacts: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def validate_lineage(self) -> "ArtifactManifest":
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("artifact IDs must be unique within a run manifest")
        known = set(identifiers)
        for artifact in self.artifacts:
            if artifact.run_id != self.run_id or artifact.project_id != self.project_id:
                raise ValueError("artifact identity does not match its run manifest")
            if artifact.artifact_id in artifact.parent_artifact_ids:
                raise ValueError("an artifact cannot be its own parent")
            missing = set(artifact.parent_artifact_ids) - known
            if missing:
                raise ValueError(
                    "artifact references missing parents: " + ", ".join(sorted(missing))
                )
        return self


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactManifest",
    "ArtifactRef",
]
