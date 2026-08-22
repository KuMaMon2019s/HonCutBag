"""Checkpoint-safe LangGraph channel definitions.

Only JSON-serializable IDs, paths, status, and necessary metadata belong here.
Images, frames, videos, base64 payloads, and logs remain filesystem artifacts.
"""

from __future__ import annotations

from typing import Any, TypedDict


class HonCutState(TypedDict, total=False):
    """Target State shape plus temporary aliases used by the live graph."""

    # Run identity and validated configuration.
    run_id: str
    run_fingerprint: str
    input_text: str
    output_dir: str
    target_duration_s: int
    shot_duration_s: int
    dry_run: bool
    chain_mode: bool
    auto_approve: bool
    transition: str
    transition_duration_s: float
    media_profile: str
    project_video_spec: dict[str, Any]
    enable_reshoot: bool
    resume: bool
    resume_from: str | None
    skip_phase: list[float]

    # Story and artifact metadata. Large media content is never stored here.
    director_plan: dict[str, Any]
    storyboard: dict[str, Any]
    characters: list[dict[str, Any]]
    storyboard_image: str
    shot_ids: list[str]
    generated_shots: list[str]
    failed_shots: list[str]

    # Quality evidence kept separate by lifecycle stage.
    storyboard_qa: dict[str, Any]
    supervision: dict[str, Any]
    consistency: dict[str, Any]
    assembly_qa: dict[str, Any]
    final_qa: dict[str, Any]

    # Durable artifact paths.
    assembled_video: str
    final_video: str

    # Workflow bookkeeping.
    phase_results: dict[str, dict[str, Any]]
    completed_phases: list[str]
    current_phase: str
    status: str
    errors: list[dict[str, Any]]
    quality_attempts: int
    reshoot_attempts: int

    # Compatibility aliases used by the current pipeline_core graph. Remove
    # only after checkpoint/resume migration is proven end to end.
    text: str
    duration: int
    shot_duration: int
    transition_duration: float
    events: list[dict[str, Any]]
    shots: list[dict[str, Any]]
    videos: list[str]
    quality_report: dict[str, Any]
    video_generation_mode: str
    retry_count: int
    error: str
