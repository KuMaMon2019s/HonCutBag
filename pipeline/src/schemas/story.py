"""Pydantic v2 contracts for story artifacts produced by LLM stages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DirectorPlan(BaseModel):
    """Compatible envelope for the current director-plan JSON."""

    model_config = ConfigDict(extra="allow")

    scenes: list[dict[str, Any]] = Field(default_factory=list)
    scene_transitions: list[dict[str, Any]] = Field(default_factory=list)


class CharacterDefinition(BaseModel):
    """Character fields shared by discovery, storyboard, and asset phases."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str = Field(min_length=1)
    description: str = ""
    appearance: dict[str, Any] = Field(default_factory=dict)
    style: str = ""


class StoryboardShot(BaseModel):
    """Gradual contract covering both legacy and current shot identifiers."""

    model_config = ConfigDict(extra="allow")

    shot_id: str | int | None = None
    id: str | int | None = None
    shot_order: int | None = Field(default=None, ge=1)
    duration: float | None = Field(default=None, gt=0)
    suggested_duration: float | None = Field(default=None, gt=0)
    visual: str = ""
    action: str = ""
    where: str = ""
    who: str | list[str] = Field(default_factory=list)
    dialogue: Any = None


class Storyboard(BaseModel):
    """Checkpoint-safe storyboard model that preserves unknown legacy fields."""

    model_config = ConfigDict(extra="allow")

    title: str = ""
    style: str | dict[str, Any] = ""
    shots: list[StoryboardShot]
    metadata: dict[str, Any] = Field(default_factory=dict)
