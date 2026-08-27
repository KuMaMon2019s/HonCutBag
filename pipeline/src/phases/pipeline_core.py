"""Backward-compatible exports for the decomposed HonCut pipeline."""

from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from graph.routing import quality_gate_router, route_phase5
from graph.state import HonCutState
from phases.phase1 import phase1_pipeline as _phase1_pipeline_owner
from phases.phase1 import phase1_screenwriter as _phase1_screenwriter_owner
from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from phases.phase1.phase1_director import run_phase1_director
from phases.phase2 import storyboard_assets as _phase2_assets_owner
from phases.phase2 import phase2_storyboard as _phase2_storyboard_owner
from phases.phase2.storyboard_assets import (
    _derive_end_state,
    _end_frame_sidecar_path,
    _generate_flf2v_end_frame,
    _generate_shot_images as _owned_generate_shot_images,
    _normalize_shot_id,
    _read_end_frame_sidecar,
    _shot_storyboard_reference,
    _storyboard_canvas,
    _storyboard_image_size,
    _storyboard_keyframe_description,
    _validate_end_frame,
    _validate_storyboard_image_composition,
    _write_end_frame_sidecar,
    build_end_frame_prompt,
    fill_storyboard_template,
    fit_to_aspect,
    load_storyboard_prompt_techniques,
)
from phases.phase3 import phase3_character as _phase3_character_owner
from phases.phase3.phase3_character import detect_derive_assets
from phases.phase4 import phase4_orchestrator as _phase4_orchestrator_owner
from phases.phase5.supervision import run_storyboard_supervision
from phases.phase6 import direct_generation as _phase6_direct_owner
from phases.phase6 import phase6_video_gen as _phase6_video_owner
from phases.phase6.direct_generation import (
    _apply_chain_relay,
    _generation_input_fingerprint,
    _phase6_output_failure,
    _prepare_phase6_prompt,
    _privacy_fallback_strategy,
    _prompt_assets_for_shot,
    _rejected_privacy_image_url,
    _run_phase6_fallback as _owned_phase6_fallback,
    _without_rejected_privacy_images,
)
from phases.phase7 import phase7_consistency as _phase7_consistency_owner
from phases.phase8 import phase8_assembly as _phase8_assembly_owner
from phases.phase9 import phase9_post as _phase9_post_owner
from phases.phase9.captions import (
    _caption_segments_from_final_asr,
    _fmt_srt_time,
    _merge_shot_transcripts,
    _probe_shot_duration,
    _write_srt,
    clean_subtitle_text,
)
from phases.phase9.delivery_encoding import (
    _final_encode_duration_gate,
    _final_encode_filters,
    _validated_reviewed_delivery_contract,
)
from phases.phase9.score_and_mix import (
    _detect_bgm,
    _phase9_real_audio_mix_request,
    _phase9_real_audio_tracks,
    _prepare_continuous_bgm,
)
from prompt.shot_prompt_builder import build_batch_prompts
from prompt.speech_pacing import annotate_shot_pacing
from quality.quality_gate import run_quality_check
from runtime import pipeline_execution as _pipeline_execution
from runtime.phase_timing import _banner, _elapsed, _now
from runtime.pipeline_checkpoints import (
    PHASE_ORDER,
    _resume_skip_phases,
    load_state_from_sqlite,
)
from runtime.pipeline_reports import _write_report
from runtime.retry_execution import _retry_with_policy
from utils.ark_llm import call_llm_stream
from utils.config import get_api_key
from utils.file_integrity import _file_sha256
from utils.media_probe import _assert_duration_conserved, _probe_av_durations
from utils.media_profiles import _project_video_spec
from utils.progress_reporter import ProgressReporter
from utils.source_paths import PIPELINE_SRC_DIR, PROJECT_ROOT

SCRIPT_DIR = PIPELINE_SRC_DIR
STYLE_SUMMARY_WALL_TIMEOUT = _phase1_screenwriter_owner.STYLE_SUMMARY_WALL_TIMEOUT
STYLE_SUMMARY_IDLE_TIMEOUT = _phase1_screenwriter_owner.STYLE_SUMMARY_IDLE_TIMEOUT
PHASE1_CHECKPOINT_SCHEMA_VERSION = (
    _phase1_screenwriter_owner.PHASE1_CHECKPOINT_SCHEMA_VERSION
)
_integrate_storyboard_prompts = _phase1_screenwriter_owner._integrate_storyboard_prompts
_attach_director_storyboard = _phase1_screenwriter_owner._attach_director_storyboard
_extract_visual_style_text = _phase1_screenwriter_owner._extract_visual_style_text
_phase1_input_hash = _phase1_screenwriter_owner._phase1_input_hash
_atomic_write_phase1_json = _phase1_screenwriter_owner._atomic_write_phase1_json
_load_phase1_checkpoint = _phase1_screenwriter_owner._load_phase1_checkpoint
_write_project_visual_style = _phase1_screenwriter_owner._write_project_visual_style
_continuity_mode_from_text = _phase1_screenwriter_owner._continuity_mode_from_text
_PipelineVideoTool = _phase6_video_owner._PipelineVideoTool
_LocalVideoVendorAdapter = _phase6_video_owner._LocalVideoVendorAdapter
_select_transition = _phase8_assembly_owner._select_transition


def _run_storyboard_supervision(storyboard: dict, output_dir: Path) -> dict:
    return run_storyboard_supervision(storyboard, output_dir)


def _summarize_visual_style_with_llm(script_text: str) -> Optional[str]:
    _phase1_screenwriter_owner.call_llm_stream = call_llm_stream
    return _phase1_screenwriter_owner._summarize_visual_style_with_llm(script_text)


def run_phase1_screenwriter(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
    screenplay_rewrite_request: dict[str, Any] | None = None,
    shot_policy: str = "continuity",
    *,
    _director_runner=None,
) -> dict:
    _phase1_screenwriter_owner._integrate_storyboard_prompts = _integrate_storyboard_prompts
    _phase1_screenwriter_owner._attach_director_storyboard = _attach_director_storyboard
    _phase1_screenwriter_owner._summarize_visual_style_with_llm = (
        _summarize_visual_style_with_llm
    )
    _phase1_screenwriter_owner.annotate_shot_pacing = annotate_shot_pacing
    _phase1_screenwriter_owner.run_quality_check = run_quality_check
    return _phase1_screenwriter_owner.run_phase1_screenwriter(
        text,
        output_dir,
        duration,
        dry_run,
        reporter=reporter,
        shot_duration=shot_duration,
        project_video_spec=project_video_spec,
        screenplay_rewrite_request=screenplay_rewrite_request,
        shot_policy=shot_policy,
        _director_runner=_director_runner or run_phase1_director,
    )


def run_phase1(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
    screenplay_rewrite_request: dict[str, Any] | None = None,
    shot_policy: str = "continuity",
) -> dict:
    return _phase1_pipeline_owner.run_phase1(
        text,
        output_dir,
        duration,
        dry_run,
        reporter=reporter,
        shot_duration=shot_duration,
        project_video_spec=project_video_spec,
        screenplay_rewrite_request=screenplay_rewrite_request,
        shot_policy=shot_policy,
        _director_runner=run_phase1_director,
        _screenwriter_runner=run_phase1_screenwriter,
    )


def _generate_shot_images(*args, **kwargs) -> int:
    _phase2_assets_owner.build_batch_prompts = build_batch_prompts
    return _owned_generate_shot_images(*args, **kwargs)


def _run_phase6_fallback(*args, **kwargs) -> dict:
    _phase6_direct_owner.get_api_key = get_api_key
    return _owned_phase6_fallback(*args, **kwargs)


def run_phase2(storyboard_data: dict, characters_data: dict, output_dir: Path, dry_run: bool) -> dict:
    _phase2_storyboard_owner.run_quality_check = run_quality_check
    _phase2_storyboard_owner.fill_storyboard_template = fill_storyboard_template
    _phase2_storyboard_owner._generate_shot_images = _generate_shot_images
    _phase2_storyboard_owner._storyboard_canvas = _storyboard_canvas
    _phase2_storyboard_owner._storyboard_image_size = _storyboard_image_size
    _phase2_storyboard_owner._validate_storyboard_image_composition = _validate_storyboard_image_composition
    return _phase2_storyboard_owner.run_phase2(
        storyboard_data, characters_data, output_dir, dry_run
    )


def run_phase3(output_dir: Path, characters_data: dict, dry_run: bool) -> dict:
    _phase3_character_owner.detect_derive_assets = detect_derive_assets
    _phase3_character_owner.run_quality_check = run_quality_check
    _phase3_character_owner._retry_with_policy = _retry_with_policy
    _phase3_character_owner._storyboard_canvas = _storyboard_canvas
    _phase3_character_owner._storyboard_image_size = _storyboard_image_size
    _phase3_character_owner._validate_storyboard_image_composition = _validate_storyboard_image_composition
    return _phase3_character_owner.run_phase3(output_dir, characters_data, dry_run)


def run_phase4(output_dir: Path, dry_run: bool) -> dict:
    return _phase4_orchestrator_owner.run_phase4(output_dir, dry_run)


def run_phase6(
    storyboard_data: dict,
    output_dir: str | Path,
    dry_run: bool,
    chain_mode: bool = False,
    media_profile: str = "480p",
) -> dict:
    _phase6_video_owner._run_phase6_fallback = _run_phase6_fallback
    return _phase6_video_owner.run_phase6(
        storyboard_data,
        output_dir,
        dry_run,
        chain_mode,
        media_profile,
        _adapter_cls=_LocalVideoVendorAdapter,
        _quality_runner=run_quality_check,
    )


def run_phase7(output_dir: Path, dry_run: bool, storyboard_data: dict | None = None) -> dict:
    return _phase7_consistency_owner.run_phase7(output_dir, dry_run, storyboard_data)


def _finish_phase8(
    assembly: dict,
    output_dir: Path,
    target_duration: Optional[float],
    enable_reshoot: bool,
    transition: str,
    transition_duration: float,
    media_profile: str,
    reshoot_round: int,
    reshoot_history: list[dict],
    chain_mode: bool,
) -> dict:
    _phase8_assembly_owner.run_phase6 = run_phase6
    _phase8_assembly_owner.run_quality_check = run_quality_check
    return _phase8_assembly_owner._finish_phase8(
        assembly,
        output_dir,
        target_duration,
        enable_reshoot,
        transition,
        transition_duration,
        media_profile,
        reshoot_round,
        reshoot_history,
        chain_mode,
    )


def run_phase8(*args, **kwargs) -> dict:
    _phase8_assembly_owner.run_phase6 = run_phase6
    _phase8_assembly_owner.run_quality_check = run_quality_check
    return _phase8_assembly_owner.run_phase8(*args, **kwargs)


def run_phase9(*args, **kwargs) -> dict:
    _phase9_post_owner.run_quality_check = run_quality_check
    return _phase9_post_owner.run_phase9(*args, **kwargs)


def build_pipeline_graph(
    auto_approve: bool = True,
    reporter: Optional[ProgressReporter] = None,
):
    from graph.composition import build_pipeline_graph as build_composed_graph

    return build_composed_graph(
        auto_approve=auto_approve,
        reporter=reporter,
        phase_owner=sys.modules[__name__],
        legacy_compat=True,
    )


def _composed_node(name: str, state: HonCutState, **kwargs) -> dict:
    from graph import composition

    return getattr(composition, name)(
        state,
        phase_owner=sys.modules[__name__],
        **kwargs,
    )


def node_phase1(state: HonCutState, reporter: Optional[ProgressReporter] = None) -> dict:
    return _composed_node("node_phase1", state, reporter=reporter)


def node_phase2(state: HonCutState) -> dict:
    return _composed_node("node_phase2", state)


def node_phase3(state: HonCutState) -> dict:
    return _composed_node("node_phase3", state)


def node_phase4(state: HonCutState) -> dict:
    return _composed_node("node_phase4", state)


def node_phase5_quality(state: HonCutState) -> dict:
    return _composed_node("node_phase5_quality", state)


def node_phase6_txt2vid(state: HonCutState) -> dict:
    return _composed_node("node_phase6_txt2vid", state)


def node_phase6_img2vid(state: HonCutState) -> dict:
    return _composed_node("node_phase6_img2vid", state)


def node_phase6_reference(state: HonCutState) -> dict:
    return _composed_node("node_phase6_reference", state)


def node_phase7(state: HonCutState) -> dict:
    return _composed_node("node_phase7", state)


def node_phase8(state: HonCutState) -> dict:
    return _composed_node("node_phase8", state)


def node_phase9(state: HonCutState) -> dict:
    return _composed_node("node_phase9", state)


def node_phase9_5(state: HonCutState) -> dict:
    from graph.composition import node_phase9_5 as composed

    return composed(state)


def run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    shot_duration: int = AVG_SHOT_DURATION,
    shot_policy: str | None = None,
    chain_mode: bool = False,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = "480p",
    enable_reshoot: bool = True,
    no_real_person: bool = False,
    resume: bool = False,
    auto_approve: bool = True,
    resume_from: str = None,
    accept_code_change_from: str = None,
    project_id: str = "local",
) -> dict:
    return _pipeline_execution.run_pipeline(
        text=text,
        input_file=input_file,
        duration=duration,
        shot_duration=shot_duration,
        shot_policy=shot_policy,
        chain_mode=chain_mode,
        dry_run=dry_run,
        skip_phase=skip_phase,
        output_dir=output_dir,
        transition=transition,
        transition_duration=transition_duration,
        media_profile=media_profile,
        enable_reshoot=enable_reshoot,
        no_real_person=no_real_person,
        resume=resume,
        auto_approve=auto_approve,
        resume_from=resume_from,
        accept_code_change_from=accept_code_change_from,
        project_id=project_id,
        _phase_owner=sys.modules[__name__],
    )


def _run_pipeline(*args, **kwargs) -> dict:
    return _pipeline_execution._run_pipeline(
        *args,
        _phase_owner=sys.modules[__name__],
        **kwargs,
    )
