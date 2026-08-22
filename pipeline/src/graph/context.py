"""Pure run-context construction for the gradual graph migration.

This module deliberately does not open a checkpointer or import Phase business
implementations. Workflow compilation and runner injection arrive in later,
separately verified slices.
"""

from __future__ import annotations

from graph.migrations import CURRENT_STATE_SCHEMA_VERSION
from graph.state import HonCutState
from schemas.workflow import GraphRunConfig, RunStatus


def initial_state_from_config(
    config: GraphRunConfig,
    *,
    include_legacy_aliases: bool = True,
) -> HonCutState:
    """Create a fresh checkpoint-safe State without touching external systems."""

    state = HonCutState(
        state_schema_version=CURRENT_STATE_SCHEMA_VERSION,
        run_id=config.run_id,
        run_fingerprint=config.run_id,
        project_id=config.project_id,
        input_text=config.input_text,
        output_dir=config.output_dir,
        target_duration_s=config.target_duration_s,
        shot_duration_s=config.shot_duration_s,
        dry_run=config.dry_run,
        chain_mode=config.chain_mode,
        auto_approve=config.auto_approve,
        transition=config.transition,
        transition_duration_s=config.transition_duration_s,
        media_profile=config.media_profile,
        project_video_spec=dict(config.project_video_spec),
        enable_reshoot=config.enable_reshoot,
        resume=config.resume,
        resume_from=config.resume_from,
        skip_phase=list(config.skip_phase),
        storyboard={},
        characters=[],
        storyboard_image="",
        shot_ids=[],
        generated_shots=[],
        failed_shots=[],
        assembled_video="",
        final_video="",
        phase_results={},
        completed_phases=[],
        current_phase="phase1",
        status=RunStatus.RUNNING.value,
        errors=[],
        quality_attempts=0,
        reshoot_attempts=0,
        retry_count=0,
    )
    if include_legacy_aliases:
        state.update(
            text=config.input_text,
            duration=config.target_duration_s,
            shot_duration=config.shot_duration_s,
            transition_duration=config.transition_duration_s,
            events=[],
            shots=[],
            videos=[],
            quality_report={},
        )
    return state
