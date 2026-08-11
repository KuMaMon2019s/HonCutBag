"""Pydantic v2 contracts for QA, supervision, consistency, and reshoots."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Grade = Literal["A", "B", "C", "D"]


class QualityIssue(BaseModel):
    """Shared issue evidence without forcing every QA producer to one taxonomy."""

    model_config = ConfigDict(extra="allow")

    category: str = "unspecified"
    severity: Literal["info", "low", "medium", "warning", "high", "critical"]
    description: str = ""
    message: str = ""
    shot_id: str | None = None
    shot_order: int | None = Field(default=None, ge=1)


class QAResult(BaseModel):
    """Deterministic QA envelope used by graph routing."""

    model_config = ConfigDict(extra="allow")

    passed: bool | None = None
    grade: Grade | None = None
    verdict: Literal["pass", "warn", "block", "revise", "fail"] | None = None
    issues: list[QualityIssue | dict[str, Any]] = Field(default_factory=list)


class SupervisionIssue(BaseModel):
    """One issue returned by independent storyboard supervision."""

    model_config = ConfigDict(extra="allow")

    shot_order: int | None = Field(default=None, ge=1)
    category: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"]
    description: str = Field(min_length=1)


class SupervisionResult(BaseModel):
    """Validated replacement for ad-hoc supervision JSON parsing."""

    model_config = ConfigDict(extra="allow")

    grade: Grade
    verdict: Literal["pass", "warn", "block"]
    issues: list[SupervisionIssue] = Field(default_factory=list)
    summary: str


class ConsistencyResult(BaseModel):
    """Cross-shot consistency evidence consumed by deterministic routing."""

    model_config = ConfigDict(extra="allow")

    passed: bool = True
    consistency_score: float | None = Field(default=None, ge=0, le=100)
    failed_shots: list[str] = Field(default_factory=list)
    slideshow_risk: float | None = Field(default=None, ge=0)
    variation_score: float | None = Field(default=None, ge=0)


class ReshootDecision(BaseModel):
    """Explicit assembly decision for a bounded reshoot loop."""

    model_config = ConfigDict(extra="allow")

    required: bool = False
    shot_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    attempt: int = Field(default=0, ge=0)
