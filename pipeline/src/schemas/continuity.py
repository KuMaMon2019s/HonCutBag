"""Contracts for editorial shots split into provider-sized generation chunks."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BoundaryKind = Literal["cut", "continuous"]
ContinuityMode = Literal["fresh", "native_extend"]
ChunkExecutionStrategy = Literal[
    "legacy",
    "multi_image",
    "tail_video_extend",
    "first_last_frame_bridge",
]


class ContinuityAnchors(BaseModel):
    """Long-lived anchors that must survive every chunk in one editorial shot."""

    model_config = ConfigDict(extra="allow")

    characters: list[str] = Field(default_factory=list)
    scene: str = ""
    screen_direction: str = ""
    camera_motion: str = ""
    style: str = ""
    tracking_prompt: str = ""


class GenerationChunk(BaseModel):
    """One provider-sized generation unit within an editorial shot."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    target_duration_s: float = Field(gt=0)
    requested_frames: int | None = Field(default=None, gt=0)
    expected_overlap_frames: int = Field(default=0, ge=0)
    expected_provider_padding_frames: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    expected_unique_frames: int | None = Field(default=None, gt=0)
    mode: ContinuityMode
    depends_on: str | None = None
    execution_strategy: ChunkExecutionStrategy = Field(
        default="legacy", exclude_if=lambda value: value == "legacy"
    )
    storyboard_beat_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    storyboard_image: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bridge_target_shot_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bridge_target_beat_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bridge_target_storyboard_image: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_prompt: str = Field(default="", exclude_if=lambda value: not value)
    start_state: str = Field(default="", exclude_if=lambda value: not value)
    end_state: str = Field(default="", exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def dependency_matches_mode(self) -> GenerationChunk:
        if self.mode == "fresh" and self.depends_on is not None:
            raise ValueError("fresh chunks cannot depend on a previous chunk")
        if self.mode == "native_extend" and not self.depends_on:
            raise ValueError("native_extend chunks require depends_on")
        if self.mode == "fresh" and self.expected_overlap_frames:
            raise ValueError("fresh chunks cannot reserve replay overlap")
        if self.execution_strategy == "multi_image" and self.mode != "fresh":
            raise ValueError("multi_image chunks must start fresh")
        if self.execution_strategy == "tail_video_extend" and self.mode != "native_extend":
            raise ValueError("tail_video_extend chunks must use native_extend dependency")
        if self.execution_strategy == "first_last_frame_bridge":
            if self.mode != "native_extend":
                raise ValueError(
                    "first_last_frame_bridge chunks must use native_extend dependency"
                )
            if not self.bridge_target_shot_id:
                raise ValueError(
                    "first_last_frame_bridge requires the target primary shot"
                )
            if self.expected_overlap_frames:
                raise ValueError(
                    "first_last_frame_bridge must not reserve reference-video replay"
                )
        if self.expected_provider_padding_frames and (
            self.requested_frames is None or self.expected_unique_frames is None
        ):
            raise ValueError(
                "provider-minimum padding requires requested and expected unique frames"
            )
        if self.requested_frames is not None and self.expected_unique_frames is not None:
            if self.expected_unique_frames != (
                self.requested_frames
                - self.expected_overlap_frames
                - self.expected_provider_padding_frames
            ):
                raise ValueError(
                    "expected unique frames must equal requested frames minus overlap "
                    "and provider-minimum padding"
                )
        return self


class ContinuityShot(BaseModel):
    """An editorial shot and the generation chunks hidden inside it."""

    model_config = ConfigDict(extra="allow")

    shot_id: str = Field(min_length=1)
    target_duration_s: float = Field(gt=0)
    target_frames: int | None = Field(default=None, gt=0)
    boundary_before: BoundaryKind = "cut"
    continuity_group_id: str = Field(default="", max_length=120)
    extends_from_shot_id: str | None = None
    extends_from_chunk_id: str | None = None
    continuity_reason: str = ""
    anchors: ContinuityAnchors = Field(default_factory=ContinuityAnchors)
    chunks: list[GenerationChunk] = Field(min_length=1)

    @model_validator(mode="after")
    def chunks_form_a_linear_chain(self) -> ContinuityShot:
        if bool(self.extends_from_shot_id) != bool(self.extends_from_chunk_id):
            raise ValueError("cross-shot continuation requires both predecessor ids")
        if self.extends_from_chunk_id and self.boundary_before != "continuous":
            raise ValueError("only continuous boundaries may extend a previous shot")
        expected_dependency: str | None = self.extends_from_chunk_id
        seen: set[str] = set()
        for index, chunk in enumerate(self.chunks, 1):
            if chunk.sequence != index:
                raise ValueError("chunk sequence must be contiguous and start at 1")
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            expected_mode = "fresh" if expected_dependency is None else "native_extend"
            if chunk.mode != expected_mode or chunk.depends_on != expected_dependency:
                raise ValueError(
                    "chunks must form one ordered fresh/previous-shot -> native_extend chain"
                )
            seen.add(chunk.chunk_id)
            expected_dependency = chunk.chunk_id
        unique_frames = [chunk.expected_unique_frames for chunk in self.chunks]
        if self.target_frames is not None and all(value is not None for value in unique_frames):
            if sum(int(value) for value in unique_frames) != self.target_frames:
                raise ValueError("chunk unique frames must add up to the editorial shot frame budget")
        elif not math.isclose(
            sum(chunk.target_duration_s for chunk in self.chunks),
            self.target_duration_s,
            abs_tol=1e-6,
        ):
            raise ValueError("chunk durations must add up to the editorial shot duration")
        return self


class PrimaryShotBridge(BaseModel):
    """Post-primary FLF2V bridge between two completed continuous shots."""

    model_config = ConfigDict(extra="forbid")

    bridge_id: str = Field(min_length=1)
    source_shot_id: str = Field(min_length=1)
    target_shot_id: str = Field(min_length=1)
    target_duration_s: float = Field(ge=3, le=6)
    requested_frames: int = Field(gt=0)
    execution_strategy: Literal["first_last_frame_bridge"] = (
        "first_last_frame_bridge"
    )
    boundary_kind: Literal["continuous"] = "continuous"
    continuity_reason: str = ""
    action_prompt: str = ""
    start_state: str = ""
    end_state: str = ""
    storyboard_transition_image: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    storyboard_transition_prompt: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    storyboard_transition_usage: Literal[
        "visual_continuity_plan_not_video_endpoint"
    ] | None = Field(default=None, exclude_if=lambda value: value is None)
    generation_phase: Literal["post_primary_shots"] = "post_primary_shots"
    first_frame_source: Literal["source_primary_video_tail_frame"] = (
        "source_primary_video_tail_frame"
    )
    last_frame_source: Literal["target_primary_video_first_frame"] = (
        "target_primary_video_first_frame"
    )


class ContinuityPlan(BaseModel):
    """Phase 4 artifact consumed by the future continuity-aware Phase 6 runner."""

    model_config = ConfigDict(extra="allow")

    version: Literal[1] = 1
    provider_chunk_limit_s: float = Field(gt=0)
    timeline_fps: int = Field(default=24, gt=0)
    shots: list[ContinuityShot] = Field(default_factory=list)
    bridges: list[PrimaryShotBridge] = Field(default_factory=list)

    @model_validator(mode="after")
    def resource_ids_are_globally_unique(self) -> ContinuityPlan:
        shot_ids: set[str] = set()
        chunk_ids: set[str] = set()
        ordered_shot_ids: list[str] = []
        for shot in self.shots:
            if shot.shot_id in shot_ids:
                raise ValueError(f"duplicate shot_id: {shot.shot_id}")
            if shot.extends_from_shot_id and shot.extends_from_shot_id not in shot_ids:
                raise ValueError(
                    f"{shot.shot_id} must extend an earlier shot, got {shot.extends_from_shot_id}"
                )
            if shot.extends_from_chunk_id and shot.extends_from_chunk_id not in chunk_ids:
                raise ValueError(
                    f"{shot.shot_id} references unknown predecessor chunk "
                    f"{shot.extends_from_chunk_id}"
                )
            shot_ids.add(shot.shot_id)
            ordered_shot_ids.append(shot.shot_id)
            if shot.target_frames is not None:
                expected_target = round(shot.target_duration_s * self.timeline_fps)
                if abs(shot.target_frames - expected_target) > 1:
                    raise ValueError(f"{shot.shot_id} target frame budget disagrees with duration")
            for chunk in shot.chunks:
                if chunk.target_duration_s > self.provider_chunk_limit_s + 1e-6:
                    raise ValueError(f"{chunk.chunk_id} exceeds provider chunk duration limit")
                if chunk.requested_frames is not None:
                    expected_requested = round(chunk.target_duration_s * self.timeline_fps)
                    if chunk.requested_frames != expected_requested:
                        raise ValueError(f"{chunk.chunk_id} requested frames disagree with duration")
                if chunk.chunk_id in chunk_ids:
                    raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
                chunk_ids.add(chunk.chunk_id)
        bridge_ids: set[str] = set()
        shot_positions = {shot_id: index for index, shot_id in enumerate(ordered_shot_ids)}
        for bridge in self.bridges:
            if bridge.bridge_id in bridge_ids:
                raise ValueError(f"duplicate bridge_id: {bridge.bridge_id}")
            bridge_ids.add(bridge.bridge_id)
            if bridge.source_shot_id not in shot_positions:
                raise ValueError(
                    f"{bridge.bridge_id} references unknown source shot "
                    f"{bridge.source_shot_id}"
                )
            if bridge.target_shot_id not in shot_positions:
                raise ValueError(
                    f"{bridge.bridge_id} references unknown target shot "
                    f"{bridge.target_shot_id}"
                )
            source_position = shot_positions[bridge.source_shot_id]
            if source_position + 1 >= len(ordered_shot_ids) or (
                ordered_shot_ids[source_position + 1] != bridge.target_shot_id
            ):
                raise ValueError(
                    f"{bridge.bridge_id} must connect adjacent primary shots"
                )
            expected_frames = round(bridge.target_duration_s * self.timeline_fps)
            if bridge.requested_frames != expected_frames:
                raise ValueError(
                    f"{bridge.bridge_id} requested frames disagree with duration"
                )
            target_shot = self.shots[source_position + 1]
            if target_shot.boundary_before != "continuous":
                raise ValueError(
                    f"{bridge.bridge_id} is forbidden across a cut/transition boundary"
                )
        return self
