"""Pydantic v2 contracts for story artifacts produced by LLM stages."""

from __future__ import annotations

from typing import Any, Literal

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
    shot_size: str = ""
    camera_movement: str = ""
    lens_mm: int | None = Field(default=None, ge=50, le=85)
    camera_motion_contract: dict[str, Any] = Field(default_factory=dict)
    lighting_key: str = ""
    time: str = ""
    time_of_day: str = ""
    time_window: str = ""
    source_time_values: list[str] = Field(default_factory=list)
    temporal_visual_contract: dict[str, Any] = Field(default_factory=dict)
    shot_intent: str = ""
    hero_moment: bool = False
    texture_keywords: list[str] = Field(default_factory=list)
    dialogue: Any = None
    boundary_before: Literal["cut", "continuous"] = "cut"
    continuity_reason: str = ""
    continuity_subject: str = ""
    source_excerpt: str = ""
    source_sequence_ids: list[str] = Field(default_factory=list)
    source_action_unit_ids: list[str] = Field(default_factory=list)
    source_event_roles: list[str] = Field(default_factory=list)
    micro_actions: list[str] = Field(default_factory=list)
    generation_action_categories: list[str] = Field(default_factory=list)
    generation_action_units: list[dict[str, Any]] = Field(default_factory=list)
    speaker_attribution: list[dict[str, Any]] = Field(default_factory=list)


class Storyboard(BaseModel):
    """Checkpoint-safe storyboard model that preserves unknown legacy fields."""

    model_config = ConfigDict(extra="allow")

    title: str = ""
    style: str | dict[str, Any] = ""
    shots: list[StoryboardShot]
    metadata: dict[str, Any] = Field(default_factory=dict)
