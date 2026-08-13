"""Contracts for editorial shots split into provider-sized generation chunks."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BoundaryKind = Literal["cut", "continuous"]
ContinuityMode = Literal["fresh", "native_extend"]


class ContinuityAnchors(BaseModel):
    """Long-lived anchors that must survive every chunk in one editorial shot."""

    model_config = ConfigDict(extra="allow")

    characters: list[str] = Field(default_factory=list)
    scene: str = ""
    screen_direction: str = ""
    camera_motion: str = ""
    style: str = ""


class GenerationChunk(BaseModel):
    """One provider-sized generation unit within an editorial shot."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    target_duration_s: float = Field(gt=0)
    mode: ContinuityMode
    depends_on: str | None = None

    @model_validator(mode="after")
    def dependency_matches_mode(self) -> GenerationChunk:
        if self.mode == "fresh" and self.depends_on is not None:
            raise ValueError("fresh chunks cannot depend on a previous chunk")
        if self.mode == "native_extend" and not self.depends_on:
            raise ValueError("native_extend chunks require depends_on")
        return self


class ContinuityShot(BaseModel):
    """An editorial shot and the generation chunks hidden inside it."""

    model_config = ConfigDict(extra="allow")

    shot_id: str = Field(min_length=1)
    target_duration_s: float = Field(gt=0)
    boundary_before: BoundaryKind = "cut"
    anchors: ContinuityAnchors = Field(default_factory=ContinuityAnchors)
    chunks: list[GenerationChunk] = Field(min_length=1)

    @model_validator(mode="after")
    def chunks_form_a_linear_chain(self) -> ContinuityShot:
        expected_dependency: str | None = None
        seen: set[str] = set()
        for index, chunk in enumerate(self.chunks, 1):
            if chunk.sequence != index:
                raise ValueError("chunk sequence must be contiguous and start at 1")
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            expected_mode = "fresh" if index == 1 else "native_extend"
            if chunk.mode != expected_mode or chunk.depends_on != expected_dependency:
                raise ValueError("chunks must form one ordered fresh -> native_extend chain")
            seen.add(chunk.chunk_id)
            expected_dependency = chunk.chunk_id
        if not math.isclose(
            sum(chunk.target_duration_s for chunk in self.chunks),
            self.target_duration_s,
            abs_tol=1e-6,
        ):
            raise ValueError("chunk durations must add up to the editorial shot duration")
        return self


class ContinuityPlan(BaseModel):
    """Phase 4 artifact consumed by the future continuity-aware Phase 6 runner."""

    model_config = ConfigDict(extra="allow")

    version: Literal[1] = 1
    provider_chunk_limit_s: float = Field(gt=0)
    shots: list[ContinuityShot] = Field(default_factory=list)

    @model_validator(mode="after")
    def resource_ids_are_globally_unique(self) -> ContinuityPlan:
        shot_ids: set[str] = set()
        chunk_ids: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in shot_ids:
                raise ValueError(f"duplicate shot_id: {shot.shot_id}")
            shot_ids.add(shot.shot_id)
            for chunk in shot.chunks:
                if chunk.target_duration_s > self.provider_chunk_limit_s + 1e-6:
                    raise ValueError(f"{chunk.chunk_id} exceeds provider chunk duration limit")
                if chunk.chunk_id in chunk_ids:
                    raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
                chunk_ids.add(chunk.chunk_id)
        return self
