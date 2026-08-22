#!/usr/bin/env python3
"""
pipeline_runner.py — Phase 1-9 端到端集成脚本
串联 Phase 1-9 所有模块，一键运行完整管线：任意文本 → polished.mp4

Usage:
    python pipeline_runner.py --text "故事文本" --duration 30 --dry-run
    python pipeline_runner.py --input story.txt --duration 60 --dry-run
    python pipeline_runner.py --text "..." --duration 30 --output-dir ./my_project
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import re
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from utils.config import get_api_key
from utils.progress_reporter import ProgressReporter
from quality.quality_gate import run_quality_check
from utils.timing_estimator import estimate_phase_duration, estimate_total, estimate_remaining
from quality.delivery_promise import classify_from_brief
from prompt.speech_pacing import annotate_shot_pacing
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.checkpoint import (
    invalidate_checkpoint_from as invalidate_stage_checkpoint,
)
from tools.provider_scoring import rank_providers
from tools.video_composer import lock_runtime
from prompt.shot_prompt_builder import build_batch_prompts
from tools.video_stitcher import build_stitch_plan
from tools.asset_binder import bind_assets
from prompt.prompt_sanitizer import sanitize_quality_prompt
from prompt.three_part_prompt import build_three_part_prompt
from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from quality.composition_validator import validate_composition
from tools.vendor_adapter import VendorAdapter, VendorModel
from utils.style_slices import get_slice
from utils.ark_llm import call_llm_stream, configure_heartbeat_callback
from utils.character_body_contracts import character_visual_description
from runtime.phase_timing import _banner, _elapsed, _ensure_dir, _now
from runtime.pipeline_checkpoints import (
    PHASE_ORDER,
    SqliteSaver,
    _checkpoint_path,
    _get_completed_stages,
    _get_next_stage,
    _read_checkpoint,
    _record_stage_checkpoint,
    _resume_skip_phases,
    get_sqlite_checkpointer,
    load_state_from_sqlite,
    save_state_to_sqlite,
)
from runtime.pipeline_reports import _write_report
from runtime.retry_execution import _retry_with_policy
from utils.file_integrity import _file_sha256
from utils.media_probe import _assert_duration_conserved, _probe_av_durations
from utils.source_paths import (
    LEGACY_TOOLS_DIR,
    OM_TOOLS_DIR,
    PIPELINE_SRC_DIR,
    PROJECT_ROOT,
)


STYLE_SUMMARY_WALL_TIMEOUT = 180.0
STYLE_SUMMARY_IDLE_TIMEOUT = 75.0


from phases.phase5.supervision import run_storyboard_supervision


def _run_storyboard_supervision(storyboard: dict, output_dir: Path) -> dict:
    """Compatibility facade for independent Phase 5 supervision."""
    return run_storyboard_supervision(storyboard, output_dir)

# Keep progress visible when invoked through ``conda run | tee``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# LangGraph integration: @task + RetryPolicy, Send fan-out, SqliteSaver
# ---------------------------------------------------------------------------
try:
    from langgraph.func import task
    from langgraph.types import RetryPolicy, Send, Command
    from langgraph.errors import GraphInterrupt
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"⚠ LangGraph not available: {e}")
    print("  Install with: pip install langgraph langgraph-checkpoint-sqlite langchain-core")
    LANGGRAPH_AVAILABLE = False
    # Fallback decorators
    import functools
    def task(func=None, **kwargs):
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kw):
                return f(*args, **kw)
            return wrapper
        if func:
            return decorator(func)
        return decorator
    RetryPolicy = None
    Send = None
    SqliteSaver = None
    Command = None
    GraphInterrupt = None

# ---------------------------------------------------------------------------
# Path setup — 让本脚本能 import 同目录及 2026-07-27_05/scripts 下的模块
# ---------------------------------------------------------------------------
SCRIPT_DIR = PIPELINE_SRC_DIR

# 优先加载当前源码目录，然后是兼容工具目录。insert(0) 必须反序执行。
for d in reversed((PIPELINE_SRC_DIR, LEGACY_TOOLS_DIR, str(OM_TOOLS_DIR))):
    s = str(d)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Media Profiles — delegate to the bundled compatibility module
# ---------------------------------------------------------------------------
import dataclasses as _dc

OM_PROFILES_AVAILABLE = False
try:
    from lib.media_profiles import get_profile as _om_get_profile
    from lib.media_profiles import MediaProfile, ALL_PROFILES
    OM_PROFILES_AVAILABLE = True
except ImportError:
    _om_get_profile = None  # type: ignore
    ALL_PROFILES = {}

# Backward-compatible name mapping: legacy names → OM profile names
_PROFILE_NAME_MAP = {
    "480p": None,             # no OM equivalent — use hardcoded fallback
    "720p": None,             # no OM equivalent — use hardcoded fallback
    "1080p": "generic_hd",
    "youtube": "youtube_landscape",
    "youtube_shorts": "youtube_shorts",
    "tiktok": "tiktok",
    "instagram_reels": "instagram_reels",
    "cinematic": "cinematic",
}

# Hardcoded fallbacks for profiles not in OM
_LEGACY_FALLBACKS = {
    "480p": {"name": "480p", "width": 854, "height": 480, "fps": 30, "codec": "libx264",
             "audio_codec": "aac", "crf": 23, "pixel_format": "yuv420p", "aspect_ratio": "16:9"},
    "720p": {"name": "720p", "width": 1280, "height": 720, "fps": 30, "codec": "libx264",
             "audio_codec": "aac", "crf": 23, "pixel_format": "yuv420p", "aspect_ratio": "16:9"},
}

# All available profile names for argparse choices
AVAILABLE_PROFILES = ["480p", "720p", "1080p", "youtube", "youtube_shorts",
                      "tiktok", "instagram_reels", "cinematic"]
if OM_PROFILES_AVAILABLE:
    AVAILABLE_PROFILES = sorted(set(AVAILABLE_PROFILES) | set(ALL_PROFILES.keys()))


def _get_profile_dict(profile_name: str = "1080p") -> dict:
    """Get media profile as a dict, using OM if available, else legacy fallback.

    Returns a dict with keys: name, width, height, fps, codec, audio_codec, crf, pixel_format.
    """
    # Check legacy fallbacks first (480p, 720p — not in OM)
    if profile_name in _LEGACY_FALLBACKS:
        return _LEGACY_FALLBACKS[profile_name]

    if OM_PROFILES_AVAILABLE and _om_get_profile is not None:
        # Map legacy name → OM name
        om_name = _PROFILE_NAME_MAP.get(profile_name, profile_name)
        if om_name is not None:
            try:
                profile = _om_get_profile(om_name)
                return _dc.asdict(profile)
            except ValueError:
                pass
        # Try direct name (for OM-native names)
        try:
            profile = _om_get_profile(profile_name)
            return _dc.asdict(profile)
        except ValueError:
            pass

    # Ultimate fallback: 1080p generic
    return {"name": "1080p", "width": 1920, "height": 1080, "fps": 30,
            "codec": "libx264", "audio_codec": "aac", "crf": 23,
            "pixel_format": "yuv420p", "aspect_ratio": "16:9"}


def _project_video_spec(media_profile: str) -> dict[str, Any]:
    """Resolve the one geometry/delivery contract shared by every phase."""
    profile = _get_profile_dict(media_profile)
    width = int(profile["width"])
    height = int(profile["height"])
    semantic_ratio = profile.get("aspect_ratio")
    if hasattr(semantic_ratio, "value"):
        semantic_ratio = semantic_ratio.value
    if not semantic_ratio:
        divisor = math.gcd(width, height)
        semantic_ratio = f"{width // divisor}:{height // divisor}"
    return {
        "aspect_ratio": str(semantic_ratio),
        "width": width,
        "height": height,
        "fps": float(profile.get("fps") or 30),
        "delivery_profile": media_profile,
    }



# ---------------------------------------------------------------------------
if LANGGRAPH_AVAILABLE:
    # Define state for LangGraph-based execution (Phase 1 StateGraph migration)
    class PipelineState(TypedDict):
        """State for LangGraph pipeline execution."""
        text: str
        output_dir: str
        duration: int
        dry_run: bool
        storyboard_data: Optional[dict]
        characters_data: Optional[dict]
        shots: list
        videos: list
        completed_phases: list
        status: str
        errors: list

    # @task wrappers with RetryPolicy — for future StateGraph execution
    # These are defined here so they're ready when we migrate to StateGraph.
    # Currently, the linear pipeline uses _retry_with_policy() instead.
    @task(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0))
    def task_call_seedream(prompt: str, output_path: str, size: str = "1920x1920", timeout: int = 180) -> dict:
        """⚠️ 此函数为 LangGraph @task 包装，仅在 StateGraph 执行上下文中有效，不可直接调用。
        Task-wrapped Seedream API call with automatic retry (for StateGraph)."""
        from clients.seedream_client import SeedreamClient
        client = SeedreamClient()
        client.text_to_image(
            prompt=prompt,
            output_path=output_path,
            size=size,
            timeout=timeout,
        )
        if Path(output_path).exists():
            return {"status": "done", "output_path": output_path}
        else:
            return {"status": "error", "error": "File not created"}

    @task(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0))
    def task_generate_character(char_dict: dict, chars_dir: str, skip_images: bool = False) -> dict:
        """⚠️ 此函数为 LangGraph @task 包装，仅在 StateGraph 执行上下文中有效，不可直接调用。
        Task-wrapped character generation with retry (for StateGraph)."""
        from phases.phase3.character_factory import generate_single
        result = generate_single(char_dict, chars_dir, skip_images=skip_images)
        return {"status": "done", "result": result}

    @task(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0))
    def task_generate_video(shot: dict, storyboard_image: Optional[str], output_dir: str, style_context: Optional[dict] = None) -> dict:
        """⚠️ 此函数为 LangGraph @task 包装，仅在 StateGraph 执行上下文中有效，不可直接调用。
        Task-wrapped video generation with retry (for StateGraph)."""
        from vendor.video_tools.tools.video.seedance_video import SeedanceVideo
        sv = SeedanceVideo()
        
        shot_id = shot.get("id", "?")
        # 标准化 shot_id 格式为零填充（S01, S02, S03）
        if isinstance(shot_id, int):
            shot_id_str = f"S{shot_id:02d}"
        else:
            shot_id_str = f"S{str(shot_id).zfill(2)}"
        prompt_items = build_batch_prompts([shot], style_context)
        prompt = (prompt_items[0]["prompt"] if prompt_items else "") or shot.get("prompt", "")
        duration = str(shot.get("duration", 5))
        aspect_ratio = shot.get("aspect_ratio", "16:9")
        
        shot_dir = Path(output_dir) / f"shots/{shot_id_str}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        video_path = shot_dir / "output.mp4"
        
        if storyboard_image and Path(storyboard_image).exists():
            result = sv.execute({
                "operation": "reference_to_video",
                "prompt": prompt,
                "reference_image_paths": [storyboard_image],
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "output_path": str(video_path),
            })
            if result.success:
                return {"status": "done", "output_path": str(video_path), "shot_id": shot_id}
        
        # Fallback to text_to_video
        result = sv.execute({
            "operation": "text_to_video",
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "output_path": str(video_path),
        })
        
        if result.success:
            return {"status": "done", "output_path": str(video_path), "shot_id": shot_id}
        else:
            return {"status": "error", "error": result.error, "shot_id": shot_id}

    # Send fan-out functions (for future StateGraph parallel execution)
    def fan_out_characters(state: dict) -> list:
        """TODO: 待接入 StateGraph 并行执行（当前为预留接口）。
        Generate Send objects for parallel character generation."""
        characters = state.get("characters_data", {}).get("characters", [])
        chars_dir = str(Path(state["output_dir"]) / "characters")
        skip_images = state.get("dry_run", False)
        
        sends = []
        for char in characters:
            char_dict = {
                "id": char.get("id", f"char_{len(sends)}"),
                "name": char.get("name", f"角色{len(sends)}"),
                "description": character_visual_description(char),
                "appearance": char.get("appearance", {}),  # 传递完整 appearance dict
                "style": char.get("style", ""),
                "negative": ", ".join(filter(None, (
                    str(char.get("negative", "")).strip(),
                    str(char.get("negative_guardrails", "")).strip(),
                ))),
            }
            sends.append(Send("generate_character", {"char_dict": char_dict, "chars_dir": chars_dir, "skip_images": skip_images}))
        
        return sends

    def fan_out_shots(state: dict) -> list:
        """TODO: 待接入 StateGraph 并行执行（当前为预留接口）。
        Generate Send objects for parallel video generation."""
        shots = state.get("shots", [])
        output_dir = state["output_dir"]
        style_context = None
        
        if state.get("storyboard_data", {}).get("style"):
            style_context = {"mood": state["storyboard_data"]["style"]}
        
        sends = []
        for shot in shots:
            shot_reference = _shot_storyboard_reference(Path(output_dir), shot.get("id"))
            sends.append(Send("generate_video", {
                "shot": shot,
                "storyboard_image": str(shot_reference) if shot_reference else None,
                "output_dir": output_dir,
                "style_context": style_context,
            }))
        
        return sends

# Phase 1: 导演规划 (M1 增量模块)
# ---------------------------------------------------------------------------

from phases.phase1.phase1_director import run_phase1_director
from phases.phase1 import phase1_pipeline as _phase1_pipeline_owner
from phases.phase1 import phase1_screenwriter as _phase1_screenwriter_owner

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


from phases.phase9.delivery_encoding import (
    _final_encode_duration_gate,
    _final_encode_filters,
    _validated_reviewed_delivery_contract,
)

def _summarize_visual_style_with_llm(script_text: str) -> Optional[str]:
    """Compatibility facade preserving call-time stream monkeypatches."""
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
) -> dict:
    """Compatibility facade for the Phase 1 screenwriter owner."""
    _phase1_screenwriter_owner._integrate_storyboard_prompts = (
        _integrate_storyboard_prompts
    )
    _phase1_screenwriter_owner._attach_director_storyboard = (
        _attach_director_storyboard
    )
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
    )


def run_phase1(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
) -> dict:
    """Compatibility facade for the Phase 1 composition owner."""
    return _phase1_pipeline_owner.run_phase1(
        text,
        output_dir,
        duration,
        dry_run,
        reporter=reporter,
        shot_duration=shot_duration,
        project_video_spec=project_video_spec,
        _director_runner=run_phase1_director,
        _screenwriter_runner=run_phase1_screenwriter,
    )


from phases.phase2 import phase2_storyboard as _phase2_storyboard_owner
from phases.phase2.storyboard_assets import (
    _derive_end_state,
    _end_frame_sidecar_path,
    _file_sha256,
    _generate_flf2v_end_frame,
    _generate_shot_images,
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


from phases.phase6.direct_generation import _phase6_output_failure


def run_phase2(
    storyboard_data: dict,
    characters_data: dict,
    output_dir: Path,
    dry_run: bool,
) -> dict:
    """Compatibility facade for the Phase 2 storyboard owner."""
    _phase2_storyboard_owner.run_quality_check = run_quality_check
    _phase2_storyboard_owner.fill_storyboard_template = fill_storyboard_template
    _phase2_storyboard_owner._generate_shot_images = _generate_shot_images
    _phase2_storyboard_owner._storyboard_canvas = _storyboard_canvas
    _phase2_storyboard_owner._storyboard_image_size = _storyboard_image_size
    _phase2_storyboard_owner._validate_storyboard_image_composition = (
        _validate_storyboard_image_composition
    )
    return _phase2_storyboard_owner.run_phase2(
        storyboard_data,
        characters_data,
        output_dir,
        dry_run,
    )

from phases.phase3 import phase3_character as _phase3_character_owner
from phases.phase3.phase3_character import detect_derive_assets


def run_phase3(output_dir: Path, characters_data: dict, dry_run: bool) -> dict:
    """Compatibility facade for the Phase 3 character owner."""
    _phase3_character_owner.detect_derive_assets = detect_derive_assets
    _phase3_character_owner.run_quality_check = run_quality_check
    _phase3_character_owner._retry_with_policy = _retry_with_policy
    _phase3_character_owner._storyboard_canvas = _storyboard_canvas
    _phase3_character_owner._storyboard_image_size = _storyboard_image_size
    _phase3_character_owner._validate_storyboard_image_composition = (
        _validate_storyboard_image_composition
    )
    return _phase3_character_owner.run_phase3(output_dir, characters_data, dry_run)

from phases.phase4 import phase4_orchestrator as _phase4_orchestrator_owner


def run_phase4(output_dir: Path, dry_run: bool) -> dict:
    """Compatibility facade for the Phase 4 orchestration owner."""
    _phase4_orchestrator_owner._normalize_shot_id = _normalize_shot_id
    _phase4_orchestrator_owner.rank_providers = rank_providers
    _phase4_orchestrator_owner.lock_runtime = lock_runtime
    return _phase4_orchestrator_owner.run_phase4(output_dir, dry_run)

from phases.phase6 import phase6_video_gen as _phase6_video_owner
from phases.phase6.direct_generation import (
    _apply_chain_relay,
    _generation_input_fingerprint,
    _prepare_phase6_prompt,
    _privacy_fallback_strategy,
    _prompt_assets_for_shot,
    _rejected_privacy_image_url,
    _run_phase6_fallback,
    _without_rejected_privacy_images,
)
from phases.phase6.legacy_vendor_generation import _run_phase6_om_seedance

_PipelineVideoTool = _phase6_video_owner._PipelineVideoTool
_LocalVideoVendorAdapter = _phase6_video_owner._LocalVideoVendorAdapter


def run_phase6(
    storyboard_data: dict,
    output_dir: str | Path,
    dry_run: bool,
    chain_mode: bool = False,
) -> dict:
    """Compatibility facade for the Phase 6 generation owner."""
    _phase6_video_owner._run_phase6_fallback = _run_phase6_fallback
    return _phase6_video_owner.run_phase6(
        storyboard_data,
        output_dir,
        dry_run,
        chain_mode,
        _adapter_cls=_LocalVideoVendorAdapter,
        _quality_runner=run_quality_check,
    )

from phases.phase7 import phase7_consistency as _phase7_consistency_owner


def run_phase7(
    output_dir: Path,
    dry_run: bool,
    storyboard_data: dict | None = None,
) -> dict:
    """Compatibility facade for the Phase 7 consistency handoff owner."""
    return _phase7_consistency_owner.run_phase7(
        output_dir,
        dry_run,
        storyboard_data,
    )

from phases.phase8 import phase8_assembly as _phase8_assembly_owner

_select_transition = _phase8_assembly_owner._select_transition


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
    """Compatibility facade for the Phase 8 finish/reshoot transaction."""
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


def run_phase8(
    output_dir: Path,
    dry_run: bool,
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = "1080p",
    target_duration: Optional[float] = None,
    enable_reshoot: bool = True,
    chain_mode: bool = False,
    _reshoot_round: int = 0,
    _reshoot_history: Optional[list[dict]] = None,
    _continuity_round: int = 0,
) -> dict:
    """Compatibility facade for the Phase 8 assembly owner."""
    _phase8_assembly_owner.run_phase6 = run_phase6
    _phase8_assembly_owner.run_quality_check = run_quality_check
    return _phase8_assembly_owner.run_phase8(
        output_dir,
        dry_run,
        transition=transition,
        transition_duration=transition_duration,
        media_profile=media_profile,
        target_duration=target_duration,
        enable_reshoot=enable_reshoot,
        chain_mode=chain_mode,
        _reshoot_round=_reshoot_round,
        _reshoot_history=_reshoot_history,
        _continuity_round=_continuity_round,
    )

from phases.phase9 import phase9_post as _phase9_post_owner
from phases.phase9.captions import (
    _caption_segments_from_final_asr,
    _fmt_srt_time,
    _merge_shot_transcripts,
    _probe_shot_duration,
    _write_srt,
    clean_subtitle_text,
)
from phases.phase9.score_and_mix import (
    _detect_bgm,
    _phase9_real_audio_mix_request,
    _phase9_real_audio_tracks,
    _prepare_continuous_bgm,
)


def run_phase9(
    output_dir: Path,
    dry_run: bool,
    color_grade: Optional[str] = None,
    upscale: Optional[int] = None,
    media_profile: str = "1080p",
    target_duration: Optional[float] = None,
) -> dict:
    """Compatibility facade for the Phase 9 post-production owner."""
    _phase9_post_owner.run_quality_check = run_quality_check
    return _phase9_post_owner.run_phase9(
        output_dir,
        dry_run,
        color_grade=color_grade,
        upscale=upscale,
        media_profile=media_profile,
        target_duration=target_duration,
    )



# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LangGraph Phase 1: StateGraph + Command + Conditional Edges
# ---------------------------------------------------------------------------

if LANGGRAPH_AVAILABLE:
    from graph.state import HonCutState

    def build_pipeline_graph(auto_approve: bool = True, reporter: Optional[ProgressReporter] = None):
        """Deprecated facade for :func:`graph.composition.build_pipeline_graph`."""
        from graph.composition import build_pipeline_graph as build_composed_graph

        return build_composed_graph(
            auto_approve=auto_approve,
            reporter=reporter,
            phase_owner=sys.modules[__name__],
            legacy_compat=True,
        )

    # --- Node functions (wrappers around existing run_phase* functions) ---
    
    def node_phase1(state: HonCutState, reporter: Optional[ProgressReporter] = None) -> dict:
        """Compatibility facade for the migrated Phase 1 graph node."""
        from graph.composition import node_phase1 as composed_node

        return composed_node(
            state,
            reporter=reporter,
            phase_owner=sys.modules[__name__],
        )

    def node_phase2(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 2 graph node."""
        from graph.composition import node_phase2 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase3(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 3 graph node."""
        from graph.composition import node_phase3 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase4(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 4 graph node."""
        from graph.composition import node_phase4 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def route_phase5(state: HonCutState) -> str:
        """Deprecated facade for the canonical Phase 5 router."""
        from graph.routing import route_phase5 as route

        return route(state)

    def node_phase5_quality(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 5 graph node."""
        from graph.composition import node_phase5_quality as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase6_txt2vid(state: HonCutState) -> dict:
        """Compatibility facade for the migrated txt2vid node."""
        from graph.composition import node_phase6_txt2vid as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase6_img2vid(state: HonCutState) -> dict:
        """Compatibility facade for the migrated img2vid node."""
        from graph.composition import node_phase6_img2vid as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase6_reference(state: HonCutState) -> dict:
        """Compatibility facade for the migrated reference node."""
        from graph.composition import node_phase6_reference as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase7(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 7 graph node."""
        from graph.composition import node_phase7 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def quality_gate_router(state: HonCutState) -> str:
        """Deprecated facade for the canonical structural quality router."""
        from graph.routing import quality_gate_router as route

        return route(state)

    def node_phase8(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 8 graph node."""
        from graph.composition import node_phase8 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase9(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 9 graph node."""
        from graph.composition import node_phase9 as composed_node

        return composed_node(state, phase_owner=sys.modules[__name__])

    def node_phase9_5(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 9.5 graph node."""
        from graph.composition import node_phase9_5 as composed_node

        return composed_node(state)

else:
    # Fallback when LangGraph is not available
    def build_pipeline_graph(auto_approve: bool = True, reporter: Optional[ProgressReporter] = None):
        return None


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    shot_duration: int = AVG_SHOT_DURATION,
    chain_mode: bool = False,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = "1080p",
    enable_reshoot: bool = True,
    no_real_person: bool = False,
    resume: bool = False,
    auto_approve: bool = True,
    resume_from: str = None,
    accept_code_change_from: str = None,
    project_id: str = "local",
) -> dict:
    """Run the pipeline without leaking its privacy mode into later runs."""
    previous_no_real_person = os.environ.get("HONCUT_NO_REAL_PERSON")
    try:
        return _run_pipeline(
            text=text,
            input_file=input_file,
            duration=duration,
            shot_duration=shot_duration,
            chain_mode=chain_mode,
            dry_run=dry_run,
            skip_phase=skip_phase,
            output_dir=output_dir,
            project_id=project_id,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            enable_reshoot=enable_reshoot,
            no_real_person=no_real_person,
            resume=resume,
            auto_approve=auto_approve,
            resume_from=resume_from,
            accept_code_change_from=accept_code_change_from,
        )
    finally:
        if previous_no_real_person is None:
            os.environ.pop("HONCUT_NO_REAL_PERSON", None)
        else:
            os.environ["HONCUT_NO_REAL_PERSON"] = previous_no_real_person


def _run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    shot_duration: int = AVG_SHOT_DURATION,
    chain_mode: bool = False,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = "1080p",
    enable_reshoot: bool = True,
    no_real_person: bool = False,
    resume: bool = False,
    auto_approve: bool = True,
    resume_from: str = None,
    accept_code_change_from: str = None,
    project_id: str = "local",
) -> dict:
    """
    主入口：端到端管线

    Args:
        text: 故事文本（直接传入）
        input_file: 故事文本文件路径（与 text 二选一）
        duration: 目标视频时长（秒）
        shot_duration: 每镜平均时长（秒）
        chain_mode: Seedance 尾帧接力模式
        dry_run: dry-run 模式（Phase 1 实际调 LLM，Phase 3 skip-images，Phase 4 dry-run，Phase 6-8 跳过）
        skip_phase: 跳过指定 phase 列表，如 [3, 8]
        output_dir: 输出目录
        project_id: 项目隔离标识；默认 `local`
        transition: Phase 8 转场模式 ("crossfade" | "fade" | "cut")
        transition_duration: Phase 8 转场时长（秒），默认 0.5
        media_profile: 编码配置名称，从 MEDIA_PROFILES 中选择（默认 "1080p"）
        enable_reshoot: 视觉缺陷或时长不足时是否允许调用 Phase 6 补录（默认 True，最多两轮）
        no_real_person: 将所有角色锁定为带多样化可见妆造锚点的虚构 CGI 设计
        resume: 从检查点恢复，跳过已完成的 Phase
        accept_code_change_from: 显式接受代码变更并从指定 Phase 继续；其他身份变化仍拒绝

    Returns:
        pipeline_report dict
    """
    skip_phase = list(skip_phase or [])
    output_path = Path(output_dir).resolve()
    _ensure_dir(output_path)
    os.environ["HONCUT_NO_REAL_PERSON"] = "1" if no_real_person else "0"

    if accept_code_change_from is not None:
        from utils.artifact_chain import (
            PHASE_SEQUENCE,
            can_resume_from,
            invalidate_checkpoints_from,
        )

        if not resume:
            raise ValueError("code change acceptance requires resume mode")
        if accept_code_change_from not in PHASE_SEQUENCE:
            raise ValueError(
                f"code change acceptance has unknown Phase: {accept_code_change_from}"
            )
        if resume_from and accept_code_change_from != resume_from:
            raise ValueError(
                "code change acceptance Phase must match --resume-from"
            )
        if not can_resume_from(accept_code_change_from, output_path):
            raise RuntimeError(
                "code change acceptance refused: prerequisite artifacts are "
                f"incomplete for {accept_code_change_from}"
            )

    # Resolve source and run identity before consulting any checkpoint. This
    # prevents an old "all phases complete" record from short-circuiting a new
    # script, model, provider, geometry, or code version.
    if text is None and input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    if not text and not resume:
        raise ValueError("必须提供 --text 或 --input 参数")
    text = text or ""
    project_video_spec = _project_video_spec(media_profile)
    from runtime.run_manifest import prepare_run_manifest
    from utils.config import get_video_route

    configured_video_provider = os.environ.get("VIDEO_PROVIDER", "seedance").lower()
    effective_video_provider = (
        "seedance"
        if configured_video_provider in {"bridge", "ark"}
        else configured_video_provider
    )
    effective_video_route = get_video_route(configured_video_provider)

    run_manifest = prepare_run_manifest(
        output_path,
        source_text=text,
        resolved_config={
            "project_id": project_id,
            "duration": duration,
            "shot_duration": shot_duration,
            "chain_mode": chain_mode,
            "transition": transition,
            "transition_duration": transition_duration,
            "media_profile": media_profile,
            "enable_reshoot": enable_reshoot,
            "no_real_person": no_real_person,
            "dry_run": dry_run,
            "video_provider": effective_video_provider,
            "video_generation_mode": effective_video_route,
            "video_model": os.environ.get(
                "SEEDANCE_MODEL",
                os.environ.get("VIDEO_MODEL", "doubao-seedance-2.0-mini"),
            ),
            "project_video_spec": project_video_spec,
        },
        repo_root=PROJECT_ROOT,
        resume=resume,
        accepted_code_change_from=accept_code_change_from,
    )
    spec_path = output_path / "PROJECT_VIDEO_SPEC.json"
    spec_temporary = spec_path.with_suffix(".json.tmp")
    spec_temporary.write_text(
        json.dumps(project_video_spec, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(spec_temporary, spec_path)

    # --- M6: --resume-from 支持 ---
    if resume_from:
        from utils.artifact_chain import PHASE_SEQUENCE, can_resume_from

        if resume_from not in PHASE_SEQUENCE:
            raise ValueError(f"未知 Phase: {resume_from}")
        if not can_resume_from(resume_from, output_path):
            raise RuntimeError(
                f"Resume-from {resume_from} refused: prerequisite artifacts are incomplete"
            )
        invalidated = invalidate_stage_checkpoint(
            _checkpoint_path(output_path),
            resume_from,
            PHASE_SEQUENCE,
        )
        invalidated_artifact_receipts = invalidate_checkpoints_from(
            resume_from,
            output_path,
        )
        skip_phase = _resume_skip_phases(skip_phase, resume_from)
        print(f"  🔄 [M6] Resume-from {resume_from}: 跳过 {skip_phase}")
        stale_phases = list(dict.fromkeys([*invalidated, *invalidated_artifact_receipts]))
        if stale_phases:
            print(
                "  ♻ [M6] 已将目标阶段及下游 checkpoint 标记为 stale: "
                + ", ".join(stale_phases)
            )

    # ---- 进度报告系统初始化 ----
    # 编排器为每个 Phase 子进程设置 HONCUT_APPEND_EVENTS=1，跨阶段 events 历史保留。
    reporter = ProgressReporter(
        str(output_path),
        total_phases=len(PHASE_ORDER),
        clear_events=not os.environ.get("HONCUT_APPEND_EVENTS"),
    )

    # --- M6: 产物链（增量）---
    try:
        from utils.artifact_chain import save_checkpoint as save_artifact_checkpoint, can_resume_from
        M6_AVAILABLE = True
    except ImportError:
        M6_AVAILABLE = False

    # ---- Resume: 读取检查点 ----
    completed_phases = set()
    resume_snapshot = None
    resume_uses_graph = False
    if resume:
        from runtime.checkpoint_resolution import resolve_resume_snapshot

        graph_states = []
        if not resume_from:
            graph_states = [
                (
                    "graph",
                    load_state_from_sqlite(
                        output_path,
                        thread_id=run_manifest["run_fingerprint"],
                    ),
                ),
                (
                    "sqlite-stage",
                    load_state_from_sqlite(output_path, thread_id="pipeline_run"),
                ),
            ]
        resume_snapshot = resolve_resume_snapshot(
            output_path,
            run_fingerprint=run_manifest["run_fingerprint"],
            project_id=project_id,
            graph_states=graph_states,
        )
        completed_phases = set(resume_snapshot.completed_phases)
        resume_uses_graph = resume_snapshot.source == "graph"
        if completed_phases:
            print(
                f"\n  🔄 Resume 模式 ({resume_snapshot.source}): "
                f"跳过已完成的 Phase: {sorted(completed_phases)}"
            )
        else:
            print("\n  🔄 Resume 模式: 无可信检查点，从头开始")

        if len(completed_phases) == len(PHASE_ORDER):
            print("  ✓ 所有 Phase 已完成，无需重新运行")
            cp = _read_checkpoint(output_path)
            reporter.mark_completed()
            return {
                "status": "completed",
                "resumed": True,
                "completed_phases": sorted(completed_phases),
                "output_dir": str(output_dir),
                "timestamp": cp.get("timestamp", "") if cp else "",
            }

    total_start = _now()
    report = {
        "status": "completed",
        "input_text_length": len(text),
        "duration_target_s": duration,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "resumed": resume,
        "phases": {},
    }

    print(f"\n{'#'*60}")
    print(f"  Honcut AI Video Pipeline")
    print(f"  文本长度: {len(text)} 字 | 目标时长: {duration}s | dry-run: {dry_run}")
    print(f"  输出目录: {output_dir}")
    if resume and completed_phases:
        print(f"  🔄 Resume: 已完成 {len(completed_phases)}/{len(PHASE_ORDER)} Phase")
    print("  ⏭️ 人工故事板复查已禁用（100% 跳过）")
    
    # 打印预估总耗时
    if not dry_run:
        _est = estimate_total(num_characters=3, num_shots=10)  # 默认值，实际运行时会根据数据调整
        print(f"  ⏱ 预估总耗时: {_est['total_human']} (基于历史数据)")
    
    print(f"{'#'*60}")

    # --- LangGraph StateGraph execution path ---
    if LANGGRAPH_AVAILABLE and not skip_phase and (
        not resume or not completed_phases or resume_uses_graph
    ):
        print(f"\n  🚀 Using LangGraph StateGraph for pipeline execution")
        try:
            # Build the graph through the production composition root.
            from graph.composition import build_pipeline_graph as build_composed_graph

            graph = build_composed_graph(
                auto_approve=auto_approve,
                reporter=reporter,
                phase_owner=sys.modules[__name__],
            )
            
            if graph is None:
                raise RuntimeError("Failed to build pipeline graph")
            
            # Create SQLite checkpointer
            saver = get_sqlite_checkpointer(output_path)
            checkpointer = None
            if saver:
                try:
                    checkpointer = saver.__enter__()
                    app = graph.compile(checkpointer=checkpointer)
                except Exception as e:
                    print(f"  ⚠ SQLite checkpointer failed: {e}; compiling without checkpointer")
                    checkpointer = None
                    app = graph.compile()
            else:
                app = graph.compile()
            
            # Seed the live graph through the validated, checkpoint-safe contract.
            from graph.context import initial_state_from_config
            from graph.migrations import latest_error_message, migrate_state
            from schemas.workflow import GraphRunConfig

            run_config = GraphRunConfig(
                run_id=run_manifest["run_fingerprint"],
                project_id=project_id,
                input_text=text,
                output_dir=str(output_path),
                target_duration_s=duration,
                shot_duration_s=shot_duration,
                dry_run=dry_run,
                chain_mode=chain_mode,
                auto_approve=auto_approve,
                transition=transition,
                transition_duration_s=transition_duration,
                media_profile=media_profile,
                project_video_spec=project_video_spec,
                enable_reshoot=enable_reshoot,
                resume=resume,
                resume_from=resume_from,
                skip_phase=skip_phase,
            )
            initial_state = initial_state_from_config(
                run_config,
                include_legacy_aliases=False,
            )
            
            # Config for threading
            config = {
                "configurable": {
                    "thread_id": run_manifest["run_fingerprint"],
                }
            }
            
            # Handle resume: if resuming, try to get existing state
            invocation_input = initial_state
            if resume and checkpointer and resume_uses_graph:
                try:
                    existing_state = app.get_state(config)
                    if existing_state:
                        # Safely check for values attribute
                        state_values = getattr(existing_state, 'values', None)
                        if state_values and isinstance(state_values, dict):
                            print(f"  🔄 Resuming from LangGraph checkpoint")
                            migrate_state(state_values)
                            invocation_input = None
                except Exception as e:
                    raise RuntimeError(
                        f"failed to load trusted graph checkpoint: {e}"
                    ) from e
            
            # Execute the graph
            try:
                final_state = app.invoke(invocation_input, config=config)

                pending_interrupts = final_state.get("__interrupt__", ())
                if pending_interrupts:
                    raise RuntimeError(
                        "unexpected graph interrupt: human review is disabled"
                    )

                final_status = final_state.get("status", "completed")
                if final_status == "running":
                    final_status = "failed"
                
                # Build report from final state
                report = {
                    "status": final_status,
                    "input_text_length": len(text),
                    "duration_target_s": duration,
                    "dry_run": dry_run,
                    "output_dir": str(output_dir),
                    "resumed": resume,
                    "phases": final_state.get("phase_results", {}),
                    "total_duration_s": _elapsed(total_start),
                    "final_video": final_state.get("final_video", ""),
                    "langgraph": True,
                }
                
                if report["status"] == "completed":
                    reporter.mark_completed()
                else:
                    reporter.mark_failed(
                        latest_error_message(final_state)
                        or f"Pipeline ended with status: {report['status']}"
                    )
                
                # Write report
                _write_report(report, output_dir)
                
                # Print summary
                print(f"\n{'#'*60}")
                print(f"  Pipeline {report['status'].upper()} (LangGraph)")
                print(f"  总耗时: {report['total_duration_s']}s")
                for pid, pdata in report["phases"].items():
                    status_icon = {"done": "✓", "skipped": "⊘", "error": "✗"}.get(pdata.get("status", ""), "?")
                    dur = pdata.get("duration_s", "-")
                    print(f"    {status_icon} Phase {pid}: {pdata.get('status', '?')} ({dur}s)")
                print(f"{'#'*60}\n")
                
                return report
                
            except GraphInterrupt as e:
                raise RuntimeError(
                    "unexpected graph interrupt: human review is disabled"
                ) from e
                
        except Exception as e:
            print(f"\n  ⚠ LangGraph execution failed: {e}")
            traceback.print_exc()
            reporter.mark_failed(f"LangGraph execution failed: {e}")
            report.update(
                status="failed",
                error=f"LangGraph execution failed: {e}",
                total_duration_s=_elapsed(total_start),
            )
            _write_report(report, output_dir)
            return report
    
    # --- Sequential execution (fallback or when skip_phase is used) ---
    if LANGGRAPH_AVAILABLE and not skip_phase and (
        not resume or not completed_phases or resume_uses_graph
    ):
        pass  # Already tried above
    else:
        print(f"\n  📋 Using sequential execution mode")

    # ---- Phase 1: 导演拆解 + 编剧引擎 (必须成功) ----
    storyboard_data = None
    characters_data = None
    if 1 in skip_phase:
        report["phases"]["phase1"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase1" in completed_phases:
        # Resume: 从 checkpoint 加载 Phase 1 结果
        cp = _read_checkpoint(output_path)
        p2 = dict(cp["results"].get("phase1", {"status": "done"}))
        p2.setdefault("status", "done")
        storyboard_data = None
        characters_data = None
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())
        report["phases"]["phase1"] = {**p2, "resumed": True}
        print(f"  🔄 Phase 1: 从 checkpoint 恢复 (已跳过)")
    else:
        reporter.phase_start("phase1", "导演拆解 + 编剧引擎")
        p2 = run_phase1(
            text,
            output_path,
            duration,
            dry_run,
            reporter=reporter,
            shot_duration=shot_duration,
            project_video_spec=project_video_spec,
        )
        # 提取内部数据（不写入 report）
        storyboard_data = p2.pop("_storyboard", None)
        characters_data = p2.pop("_characters", None)
        report["phases"]["phase1"] = p2

        if p2["status"] == "error":
            reporter.mark_failed(f"Phase 1 failed: {p2.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 1 failed: {p2.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        reporter.phase_done("phase1", "导演拆解 + 编剧引擎完成", duration_s=p2.get("duration_s"))
        # 写入 checkpoint
        _record_stage_checkpoint(output_path, "phase1", p2)

    # --- P2-5b: HonCut 质检阻断 ---
    try:
        p2_result = report["phases"].get("phase1", {})
        review = p2_result.get("storyboard_review", {})
        if review.get("grade") == "D":
            print(f"  🚫 [P2-5b] 分镜审核 D 级，管线中止（节省后续 token）")
            report["status"] = "aborted_quality"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        elif review.get("grade") == "C":
            print(f"  ⚠ [P2-5b] 分镜审核 C 级，继续但需注意质量")
    except Exception as e:
        print(f"  ⚠ [P2-5b] 质检阻断检查失败（降级跳过）: {e}")

    # 如果 Phase 1 被跳过或数据为空，尝试从文件读
    if 1 in skip_phase or storyboard_data is None:
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())

    # ---- Phase 2: 故事板图片生成 (OM image_selector) ----
    if 2 in skip_phase:
        report["phases"]["phase2"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase2" in completed_phases:
        cp = _read_checkpoint(output_path)
        p2_5 = dict(cp["results"].get("phase2", {"status": "done"}))
        p2_5.setdefault("status", "done")
        report["phases"]["phase2"] = {**p2_5, "resumed": True}
        print(f"  🔄 Phase 2: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None or characters_data is None:
        report["phases"]["phase2"] = {"status": "skipped", "reason": "no storyboard/characters data"}
    else:
        reporter.phase_start("phase2", "故事板图片生成")
        p2_5 = run_phase2(storyboard_data, characters_data, Path(output_dir), dry_run)
        report["phases"]["phase2"] = p2_5
        if p2_5["status"] == "error":
            reporter.phase_done("phase2", f"故事板图片生成失败: {p2_5.get('error')}", duration_s=p2_5.get("duration_s"))
            reporter.mark_failed(f"Phase 2 failed: {p2_5.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 2 failed: {p2_5.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase2", "故事板图片生成完成", duration_s=p2_5.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase2", p2_5)

    # ---- Phase 3: 角色工厂 ----
    # Checkpoints created before the semantic four-view contract may contain
    # four existing files but no valid angle/background/identity evidence.
    # Never let resume bypass the current blocking gate.
    phase3_resume_quality = None
    if 3 not in skip_phase and resume and "phase3" in completed_phases:
        phase3_resume_quality = run_quality_check("phase3", output_path)
        if not phase3_resume_quality.passed:
            print(
                "  ⚠ Phase 3 checkpoint 四视图审核凭证缺失、失败或已过期；"
                "本次恢复将重新执行 Phase 3"
            )
    if 3 in skip_phase:
        report["phases"]["phase3"] = {"status": "skipped", "reason": "user-specified"}
    elif (
        resume
        and "phase3" in completed_phases
        and phase3_resume_quality is not None
        and phase3_resume_quality.passed
    ):
        cp = _read_checkpoint(output_path)
        p3 = dict(cp["results"].get("phase3", {"status": "done"}))
        p3.setdefault("status", "done")
        report["phases"]["phase3"] = {**p3, "resumed": True}
        print(f"  🔄 Phase 3: 从 checkpoint 恢复 (已跳过)")
    elif characters_data is None:
        report["phases"]["phase3"] = {"status": "skipped", "reason": "no characters data"}
    else:
        reporter.phase_start("phase3", "角色工厂")
        p3 = run_phase3(output_dir, characters_data, dry_run)
        report["phases"]["phase3"] = p3
        if p3["status"] == "error":
            reporter.phase_done("phase3", f"角色工厂失败: {p3.get('error')}", duration_s=p3.get("duration_s"))
            reporter.mark_failed(f"Phase 3 failed: {p3.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 3 failed: {p3.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase3", "角色工厂完成", duration_s=p3.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase3", p3)

    # ---- Phase 4: 编排器 ----
    if 4 in skip_phase:
        report["phases"]["phase4"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase4" in completed_phases:
        cp = _read_checkpoint(output_path)
        p4 = dict(cp["results"].get("phase4", {"status": "done"}))
        p4.setdefault("status", "done")
        report["phases"]["phase4"] = {**p4, "resumed": True}
        print(f"  🔄 Phase 4: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["phase4"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        reporter.phase_start("phase4", "编排器")
        p4 = run_phase4(output_dir, dry_run)
        report["phases"]["phase4"] = p4
        if p4["status"] == "error":
            reporter.phase_done("phase4", f"编排器失败: {p4.get('error')}", duration_s=p4.get("duration_s"))
            reporter.mark_failed(f"Phase 4 failed: {p4.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 4 failed: {p4.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase4", "编排器完成", duration_s=p4.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase4", p4)

    # ---- Phase 5: 分镜质检闸门 ----
    # This is deliberately immediately before Phase 6: a resumed or partially
    # selected run must not bypass the last zero/video-cost checkpoint.
    if 5 in skip_phase:
        if 6 not in skip_phase:
            from utils.artifact_chain import can_resume_from

            checkpoint = _read_checkpoint(output_path)
            phase5_receipt = (
                checkpoint.get("results", {}).get("phase5")
                if isinstance(checkpoint, dict)
                else None
            )
            try:
                gate_report = json.loads(
                    (output_path / "storyboard_qa_report.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                gate_report = None
            gate_is_current_and_passing = bool(
                isinstance(phase5_receipt, dict)
                and phase5_receipt.get("status") == "done"
                and isinstance(gate_report, dict)
                and gate_report.get("gate_passed") is True
                and can_resume_from("phase6", output_path)
            )
            if not gate_is_current_and_passing:
                error = (
                    "Phase 6 refused: the current run has no passing Phase 5 "
                    "checkpoint and storyboard QA receipt"
                )
                report["phases"]["phase5"] = {
                    "status": "error",
                    "error": error,
                }
                report["status"] = "failed"
                report["error"] = error
                report["total_duration_s"] = _elapsed(total_start)
                reporter.mark_failed(error)
                _write_report(report, output_dir)
                return report
            report["phases"]["phase5"] = {
                **phase5_receipt,
                "resumed": True,
                "gate_validation": "current-run checkpoint",
            }
        else:
            report["phases"]["phase5"] = {
                "status": "skipped",
                "reason": "user-specified",
            }
    elif storyboard_data is None:
        report["phases"]["phase5"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        from phases.phase5 import storyboard_qa_gate

        reporter.phase_start("phase5", "分镜质检闸门")
        p4_5 = storyboard_qa_gate.run_storyboard_qa_with_correction(
            output_path,
            qa_runner=storyboard_qa_gate.run_storyboard_qa_gate,
        )
        report["phases"]["phase5"] = p4_5
        reporter.phase_done("phase5", f"分镜质检 {p4_5.get('grade', '?')} 级", duration_s=p4_5.get("duration_s"))
        if p4_5["status"] == "error":
            reporter.mark_failed(p4_5.get("error", "Phase 5 blocked Phase 6"))
            report["status"] = "failed"
            report["error"] = p4_5.get("error", "Phase 5 blocked Phase 6")
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        from quality.supervision_agent import SupervisionBlockedError
        try:
            supervision = _run_storyboard_supervision(storyboard_data, output_path)
            report["phases"]["phase5"]["supervision"] = supervision
        except SupervisionBlockedError as exc:
            reporter.mark_failed(str(exc))
            report["status"] = "failed"
            report["error"] = str(exc)
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        _record_stage_checkpoint(output_path, "phase5", p4_5)

    # ---- Phase 6: 视频生成 ----
    if 6 in skip_phase:
        report["phases"]["phase6"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase6" in completed_phases:
        cp = _read_checkpoint(output_path)
        p5 = dict(cp["results"].get("phase6", {"status": "done"}))
        p5.setdefault("status", "done")
        report["phases"]["phase6"] = {**p5, "resumed": True}
        print(f"  🔄 Phase 6: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["phase6"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        p5 = run_phase6(storyboard_data, output_dir, dry_run, chain_mode=chain_mode)
        report["phases"]["phase6"] = p5
        if p5["status"] == "error":
            report["status"] = "failed"
            report["error"] = p5.get("error", "Phase 6 video generation failed")
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase6", p5)

    # ---- Phase 7: handoff into Phase 8 pixel-level QA ----
    if 7 in skip_phase:
        report["phases"]["phase7"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase7" in completed_phases:
        cp = _read_checkpoint(output_path)
        p6 = dict(cp["results"].get("phase7", {"status": "done"}))
        p6.setdefault("status", "done")
        report["phases"]["phase7"] = {**p6, "resumed": True}
        print(f"  🔄 Phase 7: 从 checkpoint 恢复 (已跳过)")
    else:
        # Ensure storyboard_data is available (may be None if Phase 1 was skipped)
        if storyboard_data is None:
            sb_path = output_path / "STORYBOARD.json"
            if sb_path.exists():
                storyboard_data = json.loads(sb_path.read_text())
        
        p6 = run_phase7(Path(output_dir), dry_run, storyboard_data=storyboard_data)
        report["phases"]["phase7"] = p6
        if p6["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 7 failed: {p6.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase7", p6)

        report["quality_gate"] = {
            "passed": True,
            "video_quality_owner": "phase8",
        }

    # ---- Phase 8: 组装引擎 ----
    if 8 in skip_phase:
        report["phases"]["phase8"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase8" in completed_phases:
        cp = _read_checkpoint(output_path)
        p7 = dict(cp["results"].get("phase8", {"status": "done"}))
        p7.setdefault("status", "done")
        report["phases"]["phase8"] = {**p7, "resumed": True}
        print(f"  🔄 Phase 8: 从 checkpoint 恢复 (已跳过)")
    else:
        p7 = run_phase8(
            output_dir, dry_run, transition=transition,
            transition_duration=transition_duration, media_profile=media_profile,
            target_duration=duration, enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
        )
        report["phases"]["phase8"] = p7
        if p7["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 8 failed: {p7.get('error', 'unknown assembly error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase8", p7)

    # ---- Phase 9: 后期处理 ----
    if 9 in skip_phase:
        report["phases"]["phase9"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase9" in completed_phases:
        cp = _read_checkpoint(output_path)
        p8 = dict(cp["results"].get("phase9", {"status": "done"}))
        p8.setdefault("status", "done")
        report["phases"]["phase9"] = {**p8, "resumed": True}
        print(f"  🔄 Phase 9: 从 checkpoint 恢复 (已跳过)")
    else:
        p8 = run_phase9(
            output_dir, dry_run, media_profile=media_profile,
            target_duration=duration,
        )
        report["phases"]["phase9"] = p8
        if p8["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 9 failed: {p8.get('error', 'unknown post-processing error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase9", p8)

    # ---- Phase 9.5: Video QA 硬性质检 ----
    if 9.5 in skip_phase:
        report["phases"]["phase9_5"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase9_5" in completed_phases:
        cp = _read_checkpoint(output_path)
        p9_5 = dict(cp["results"].get("phase9_5", {"status": "done"}))
        p9_5.setdefault("status", "done")
        report["phases"]["phase9_5"] = {**p9_5, "resumed": True}
        print(f"  🔄 Phase 9.5: 从 checkpoint 恢复 (已跳过)")
    else:
        try:
            from quality.video_qa import run_video_qa
            delivery_profile = _get_profile_dict(media_profile)
            qa_report = run_video_qa(
                output_dir,
                storyboard_data=storyboard_data,
                expected_width=int(delivery_profile["width"]),
                expected_height=int(delivery_profile["height"]),
                expected_min_duration=float(duration) - 1.0,
                expected_max_duration=float(duration) + 1.0,
            )
            qa_passed = qa_report.verdict == "pass"
            p9_5 = {
                "status": "done" if qa_passed else "error",
                "verdict": qa_report.verdict,
                "grade": qa_report.grade,
                "issues_count": len(qa_report.issues),
                "duration_s": 0,
            }
            report["phases"]["phase9_5"] = p9_5
            if not qa_passed:
                report["status"] = "failed"
                report["quality_gate"] = {
                    "passed": False,
                    "reason": f"Phase 9.5 delivery QA requires revision: {qa_report.grade} grade",
                    "issues": [i.message for i in qa_report.issues],
                }
            else:
                _record_stage_checkpoint(output_path, "phase9_5", p9_5)
        except ImportError:
            report["phases"]["phase9_5"] = {"status": "error", "reason": "video_qa module not available"}
            report["status"] = "failed"
            report["quality_gate"] = {"passed": False, "reason": "Phase 9.5 delivery QA is unavailable"}
        except Exception as e:
            report["phases"]["phase9_5"] = {"status": "error", "error": str(e)}
            report["status"] = "failed"
            report["quality_gate"] = {"passed": False, "reason": f"Phase 9.5 delivery QA failed to run: {e}"}

    report["total_duration_s"] = _elapsed(total_start)

    if report["status"] == "completed":
        report.pop("error", None)
        reporter.mark_completed()
    else:
        reporter.mark_failed(
            report.get("quality_gate", {}).get("reason")
            or report.get("error")
            or f"Pipeline ended with status: {report['status']}"
        )

    # --- M6: 产物链验证 ---
    if M6_AVAILABLE:
        try:
            from utils.artifact_chain import verify_artifacts, save_checkpoint as save_artifact_checkpoint
            for phase_name in PHASE_ORDER:
                phase_report = report.get("phases", {}).get(phase_name, {})
                if phase_report.get("status") != "done":
                    continue
                va = verify_artifacts(phase_name, output_path)
                if va["exists"]:
                    save_artifact_checkpoint(phase_name, output_path, va)
        except Exception as e:
            print(f"  ⚠ [M6] 产物链验证跳过: {e}")

    # 写报告
    _write_report(report, output_dir)

    # 打印总结
    print(f"\n{'#'*60}")
    print(f"  Pipeline {report['status'].upper()}")
    print(f"  总耗时: {report['total_duration_s']}s")
    for pid, pdata in report["phases"].items():
        status_icon = {"done": "✓", "skipped": "⊘", "error": "✗"}.get(pdata["status"], "?")
        dur = pdata.get("duration_s", "-")
        print(f"    {status_icon} Phase {pid}: {pdata['status']} ({dur}s)")
    print(f"{'#'*60}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Honcut AI Video Pipeline — 端到端管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # dry-run 模式（验证流程，不生成视频）
  python pipeline_runner.py --text "艾米在雪地里找到了一只受伤的小狼" --duration 30 --dry-run

  # 从文件输入
  python pipeline_runner.py --input story.txt --duration 60 --dry-run

  # 完整模式（实际生成，需要 ARK_AGENT_API_KEY）
  python pipeline_runner.py --text "..." --duration 30 --output-dir ./my_project

  # 跳过指定 Phase
  python pipeline_runner.py --text "..." --duration 30 --dry-run --skip-phase 5 6 7 8
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="故事文本")
    group.add_argument("--input", type=str, help="故事文本文件路径")
    parser.add_argument("--duration", type=int, default=60, help="目标视频时长（秒），默认 60")
    parser.add_argument("--dry-run", action="store_true", help="dry-run 模式")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录，默认当前目录")
    parser.add_argument("--skip-phase", type=float, nargs="+", default=[], help="跳过指定 Phase（支持 9.5）")
    parser.add_argument("--transition", type=str, default="crossfade", choices=["crossfade", "fade", "cut"],
                        help="Phase 8 转场模式: crossfade (默认), fade (fade-through-black), cut (硬切)")
    parser.add_argument("--transition-duration", type=float, default=0.5,
                        help="Phase 8 转场时长（秒），默认 0.5")
    reshoot_group = parser.add_mutually_exclusive_group()
    reshoot_group.add_argument(
        "--enable-reshoot", dest="enable_reshoot", action="store_true",
        help="允许 Phase 8 对视觉缺陷/时长不足镜头补录（默认开启，最多两轮）",
    )
    reshoot_group.add_argument(
        "--disable-reshoot", dest="enable_reshoot", action="store_false",
        help="禁止补录；检测到必须补录的坏镜头时阻断组装",
    )
    parser.set_defaults(enable_reshoot=True)
    parser.add_argument(
        "--no-real-person",
        action="store_true",
        help="只生成带多样化、可见非真人妆造锚点的虚构 CGI 角色",
    )
    parser.add_argument("--media-profile", type=str, default="1080p",
                        choices=AVAILABLE_PROFILES,
                        help="编码配置（默认 1080p）")
    parser.add_argument("--resume", action="store_true",
                        help="从检查点恢复，跳过已完成的 Phase（读取 output_dir/checkpoint.json）")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=True,
        help="兼容参数；人工故事板复查已永久禁用并始终跳过",
    )
    parser.add_argument("--resume-from", type=str, default=None,
                        help="从指定阶段恢复（如 phase5），跳过之前的阶段")
    parser.add_argument(
        "--accept-code-change",
        action="store_true",
        help="显式接受代码变更后续跑；必须与 --resume-from 同用",
    )

    args = parser.parse_args()

    if args.accept_code_change and not args.resume:
        parser.error("--accept-code-change requires --resume")
    if args.accept_code_change and not args.resume_from:
        parser.error("--accept-code-change requires --resume-from")

    report = run_pipeline(
        text=args.text,
        input_file=args.input,
        duration=args.duration,
        dry_run=args.dry_run,
        skip_phase=args.skip_phase,
        output_dir=args.output_dir,
        transition=args.transition,
        transition_duration=args.transition_duration,
        media_profile=args.media_profile,
        enable_reshoot=args.enable_reshoot,
        no_real_person=args.no_real_person,
        resume=args.resume,
        auto_approve=args.auto_approve,
        resume_from=args.resume_from,
        accept_code_change_from=(
            args.resume_from if args.accept_code_change else None
        ),
    )

    sys.exit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
