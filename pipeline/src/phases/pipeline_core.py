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


def _final_encode_duration_gate(
    encode_input_durations: dict[str, float | None],
    encoded_durations: dict[str, float | None],
    *,
    delivery_contract: dict[str, Any],
    requested_duration: float | None,
    fps: float,
) -> dict[str, Any]:
    """Build the non-contradictory Phase 9 final-encode duration receipt.

    Phase 8 and the rhythm editor own the reviewed edit timeline. Phase 9's
    encoder must preserve that input and match the cryptographically bound
    delivery receipt rather than silently re-trimming it to an earlier target.
    """
    duration_tolerance_s = 2 / float(fps)
    audio_duration_tolerance_s = max(duration_tolerance_s, 0.05)
    comparison_epsilon_s = 1e-6
    duration_deltas = {
        kind: (
            None
            if encode_input_durations[kind] is None or encoded_durations[kind] is None
            else round(
                abs(encoded_durations[kind] - encode_input_durations[kind]),
                6,
            )
        )
        for kind in ("video", "audio")
    }
    encode_conserved = all(
        encode_input_durations[kind] is None
        or (
            encoded_durations[kind] is not None
            and duration_deltas[kind]
            <= (
                audio_duration_tolerance_s
                if kind == "audio"
                else duration_tolerance_s
            )
            + comparison_epsilon_s
        )
        for kind in ("video", "audio")
    )
    reviewed_duration = float(delivery_contract["duration_s"])
    encode_input_to_reviewed_delta = round(
        abs(float(encode_input_durations["video"]) - reviewed_duration),
        6,
    )
    encoded_to_reviewed_delta = (
        None
        if encoded_durations["video"] is None
        else round(abs(float(encoded_durations["video"]) - reviewed_duration), 6)
    )
    reviewed_timeline_matched = (
        encode_input_to_reviewed_delta
        <= duration_tolerance_s + comparison_epsilon_s
        and encoded_to_reviewed_delta is not None
        and encoded_to_reviewed_delta
        <= duration_tolerance_s + comparison_epsilon_s
    )
    requested_duration_delta = (
        None
        if requested_duration is None or encoded_durations["video"] is None
        else round(
            abs(encoded_durations["video"] - float(requested_duration)),
            6,
        )
    )
    requested_duration_within_tolerance = (
        None
        if requested_duration_delta is None
        else requested_duration_delta
        <= duration_tolerance_s + comparison_epsilon_s
    )
    return {
        "passed": encode_conserved and reviewed_timeline_matched,
        "artifact": "polished.mp4",
        "expected": encode_input_durations,
        "actual": encoded_durations,
        "absolute_delta_s": duration_deltas,
        "authoritative_duration_s": reviewed_duration,
        "authoritative_duration_source": "delivery_timeline.json",
        "encode_input_to_authoritative_delta_s": encode_input_to_reviewed_delta,
        "encoded_to_authoritative_delta_s": encoded_to_reviewed_delta,
        "reviewed_timeline": delivery_contract,
        "requested_duration_s": requested_duration,
        "requested_duration_delta_s": requested_duration_delta,
        "requested_duration_within_tolerance": requested_duration_within_tolerance,
        "requested_duration_enforced_by_final_encode": False,
        "tolerance_s": {
            "video": duration_tolerance_s,
            "audio": audio_duration_tolerance_s,
        },
        "tolerance_frames": 2,
        "basis": (
            "Phase 9 encode input is SHA-256-bound to delivery_timeline.json; "
            "the earlier requested duration is diagnostic only"
        ),
    }


def _final_encode_filters(profile: dict[str, Any]) -> tuple[str, str]:
    """Return delivery filters that normalize format without changing runtime."""
    video_filters = (
        "setpts=PTS-STARTPTS,"
        f"scale={profile['width']}:{profile['height']}:"
        "force_original_aspect_ratio=increase,"
        f"crop={profile['width']}:{profile['height']},setsar=1,"
        f"fps={profile['fps']}"
    )
    audio_filters = "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0"
    return video_filters, audio_filters


def _validated_reviewed_delivery_contract(
    output_dir: Path,
    encode_input: Path,
    encode_input_durations: dict[str, float | None],
    *,
    fps: float,
) -> dict[str, Any]:
    """Validate that final encoding consumes the current reviewed timeline."""
    from phases.phase9.rhythm_editor import DELIVERY_TIMELINE_SCHEMA

    receipt_path = Path(output_dir) / "delivery_timeline.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Final encode requires a readable delivery_timeline.json from the current rhythm edit"
        ) from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != DELIVERY_TIMELINE_SCHEMA:
        raise RuntimeError(
            f"Final encode requires delivery timeline schema {DELIVERY_TIMELINE_SCHEMA}"
        )
    if receipt.get("artifact") != encode_input.name:
        raise RuntimeError(
            "Delivery timeline artifact does not match the final encode input: "
            f"{receipt.get('artifact')!r} != {encode_input.name!r}"
        )
    expected_sha = str(receipt.get("source_sha256") or "")
    actual_sha = _file_sha256(encode_input)
    if len(expected_sha) != 64 or expected_sha != actual_sha:
        raise RuntimeError(
            "Delivery timeline SHA-256 does not match the final encode input"
        )
    source_size = receipt.get("source_size_bytes")
    if not isinstance(source_size, int) or source_size != encode_input.stat().st_size:
        raise RuntimeError(
            "Delivery timeline byte size does not match the final encode input"
        )

    try:
        duration = float(receipt["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Delivery timeline has no valid duration_s") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Delivery timeline duration_s must be finite and positive")
    shots = receipt.get("shots")
    if not isinstance(shots, list) or not shots:
        raise RuntimeError("Delivery timeline must contain at least one reviewed shot")

    comparison_epsilon_s = 1e-6
    previous_end = 0.0
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not str(shot.get("shot_id") or "").strip():
            raise RuntimeError(f"Delivery timeline shot {index} has no shot_id")
        try:
            start = float(shot["output_start_s"])
            end = float(shot["output_end_s"])
            item_duration = float(shot["output_duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Delivery timeline shot {index} has invalid output timing"
            ) from exc
        if not all(math.isfinite(value) for value in (start, end, item_duration)):
            raise RuntimeError(f"Delivery timeline shot {index} timing must be finite")
        if abs(start - previous_end) > comparison_epsilon_s:
            raise RuntimeError(
                f"Delivery timeline shot {index} is not contiguous with its predecessor"
            )
        if end <= start or abs((end - start) - item_duration) > comparison_epsilon_s:
            raise RuntimeError(f"Delivery timeline shot {index} duration is inconsistent")
        previous_end = end
    if abs(previous_end - duration) > comparison_epsilon_s:
        raise RuntimeError("Delivery timeline final boundary does not match duration_s")

    video_duration = encode_input_durations.get("video")
    if video_duration is None:
        raise RuntimeError("Final encode input has no measurable video duration")
    tolerance_s = 2 / float(fps)
    input_delta = abs(float(video_duration) - duration)
    if input_delta > tolerance_s + comparison_epsilon_s:
        raise RuntimeError(
            "Delivery timeline duration does not match the final encode input: "
            f"timeline={duration:.6f}s input={float(video_duration):.6f}s"
        )
    return {
        "schema": DELIVERY_TIMELINE_SCHEMA,
        "artifact": encode_input.name,
        "source_sha256": actual_sha,
        "source_size_bytes": source_size,
        "duration_s": duration,
        "shot_count": len(shots),
        "timing_contiguous": True,
        "input_duration_delta_s": round(input_delta, 6),
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

def _select_transition(shot_meta: dict, default_transition: str = "dissolve") -> str:
    """Select transition type based on shot emotion and context."""
    # Check if shot already has a transition_to_next field (from adaptation_engine)
    explicit = shot_meta.get("transition_to_next", "")
    if explicit in ("cut", "dissolve", "fade"):
        return explicit
    
    # Emotion-based selection
    emotion = shot_meta.get("emotion", "").lower()
    
    # Gentle emotions → dissolve
    gentle = ["温柔", "深情", "心动", "欣喜", "喜悦", "暧昧", "羞涩"]
    if any(e in emotion for e in gentle):
        return "dissolve"
    
    # Intense emotions → cut
    intense = ["紧张", "愤怒", "惊讶", "震惊", "慌乱", "压迫"]
    if any(e in emotion for e in intense):
        return "cut"
    
    # Scene change indicators → fade
    # (detected by comparing 'where' fields between consecutive shots)
    
    return default_transition


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 8: 组装引擎 (Assembly) — delegates to OM VideoStitch
# ---------------------------------------------------------------------------

def _finish_phase8(
    phase_result: dict,
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
    """Apply the duration gate and fail closed when required footage is missing."""
    from phases.phase8.duration_gate import evaluate_duration_gate, trim_excess_to_target
    from phases.phase8.reshoot_transaction import (
        ReshootTransaction,
        mark_cycle_completed,
    )

    outputs = list(phase_result.get("outputs", []))
    for artifact in (
        "storyboard_order_review.json",
        "frame_analysis.json",
        "duration_gate.json",
    ):
        if artifact not in outputs:
            outputs.append(artifact)
    phase_result["outputs"] = outputs

    try:
        gate, reshoot_plan = evaluate_duration_gate(
            output_dir,
            target_duration,
            round_number=reshoot_round,
            reshoots=reshoot_history,
        )
        if gate.get("status") != "OVERLONG":
            duration_trim = trim_excess_to_target(output_dir, target_duration)
            if duration_trim:
                phase_result["duration_trim"] = duration_trim
                if "duration_trim.json" not in phase_result["outputs"]:
                    phase_result["outputs"].append("duration_trim.json")
                print(
                    "  ✂ [8.3] 组装时长归一化: "
                    f"{duration_trim['original_s']:.2f}s → {duration_trim['trimmed_s']:.2f}s",
                    flush=True,
                )
                gate, reshoot_plan = evaluate_duration_gate(
                    output_dir,
                    target_duration,
                    round_number=reshoot_round,
                    reshoots=reshoot_history,
                )
    except Exception as exc:
        print(f"  ⚠⚠ [8.3] 时长闸门执行失败: {exc}；阻止交付", flush=True)
        phase_result["duration_gate_error"] = str(exc)
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration gate failed: {exc}"
        return phase_result

    phase_result["duration_gate"] = gate
    if target_duration is None:
        print("  ⊘ [8.3] target_duration=None，跳过时长闸门", flush=True)
        mark_cycle_completed(output_dir)
        return phase_result
    if gate["passed"]:
        print(
            f"  ✓ [8.3] 时长闸门通过: {gate['actual_s']:.2f}s / {gate['target_s']:.2f}s",
            flush=True,
        )
        mark_cycle_completed(output_dir)
        return phase_result

    if gate.get("status") == "OVERLONG":
        print(
            f"  ⚠⚠ [8.3] 成片过长: 实际 {gate['actual_s']:.2f}s，"
            f"目标 {gate['target_s']:.2f}s，超出 {gate['excess_s']:.2f}s；"
            "必须重新剪辑，禁止生成补拍计划",
            flush=True,
        )
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration gate requires re-edit for overlong assembly: "
            f"excess {gate['excess_s']:.2f}s"
        )
        return phase_result

    print(
        f"  ⚠⚠ [8.3] 时长不足: 实际 {gate['actual_s']:.2f}s，"
        f"目标 {gate['target_s']:.2f}s，缺口 {gate['gap_s']:.2f}s",
        flush=True,
    )
    if not enable_reshoot:
        print("  ⊘ [8.3] enable_reshoot=false，时长缺口未修复，阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration gate requires reshoot but enable_reshoot=false: "
            f"missing {gate['gap_s']:.2f}s"
        )
        return phase_result
    if reshoot_round >= 2:
        print("  ⚠⚠ [8.3] 已达补录上限 2 轮；阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration still fails after 2 reshoot rounds: "
            f"missing {gate['gap_s']:.2f}s"
        )
        return phase_result
    selected = (reshoot_plan or {}).get("shots", [])
    if not selected:
        print("  ⚠⚠ [8.3] 未找到 requested > actual 的短板镜头，无法自动补录", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = "Phase 8 duration gate failed and no reshoot candidates were found"
        return phase_result

    selected_ids = [str(shot["shot_id"]) for shot in selected]
    try:
        transaction = ReshootTransaction.begin(
            output_dir,
            kind="duration_shortfall",
            shot_ids=selected_ids,
        )
        transaction.remove_sources()
    except Exception as exc:
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 could not prepare recoverable duration reshoot: {exc}"
        return phase_result
    print(
        f"  🔄 [8.3] 补录第 {reshoot_round + 1}/2 轮: {', '.join(selected_ids)}；"
        "其余镜头由 Phase 6 自动跳过",
        flush=True,
    )
    storyboard_path = output_dir / "STORYBOARD.json"
    try:
        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        generation = run_phase6(storyboard, output_dir, dry_run=False, chain_mode=chain_mode)
    except Exception as exc:
        transaction.rollback(str(exc))
        print(f"  ⚠⚠ [8.3] 补录调用 Phase 6 失败: {exc}；阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration reshoot could not run Phase 6: {exc}"
        return phase_result
    if generation.get("status") == "error":
        failure = str(generation.get("error") or generation.get("errors"))
        transaction.rollback(failure)
        print(f"  ⚠⚠ [8.3] Phase 6 补录失败: {generation.get('error') or generation.get('errors')}", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration reshoot failed in Phase 6: "
            f"{generation.get('error') or generation.get('errors')}"
        )
        return phase_result

    try:
        transaction.commit()
    except Exception as exc:
        transaction.rollback(str(exc))
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration reshoot validation failed: {exc}"
        return phase_result

    history = reshoot_history + [{**reshoot_plan, "phase6_status": generation.get("status")}]
    return run_phase8(
        output_dir,
        dry_run=False,
        transition=transition,
        transition_duration=transition_duration,
        media_profile=media_profile,
        target_duration=target_duration,
        enable_reshoot=enable_reshoot,
        _reshoot_round=reshoot_round + 1,
        _reshoot_history=history,
        chain_mode=chain_mode,
    )


def run_phase8(output_dir: Path, dry_run: bool,
               transition: str = "crossfade",
               transition_duration: float = 0.5,
               media_profile: str = "1080p",
               target_duration: Optional[float] = None,
               enable_reshoot: bool = True,
               chain_mode: bool = False,
               _reshoot_round: int = 0,
               _reshoot_history: Optional[list[dict]] = None,
               _continuity_round: int = 0) -> dict:
    """Phase 8: 逐镜质检、裁切/补录闭环与受审组装。"""
    _banner(8, 9, f"组装引擎 (Assembly) — {transition}", dry_run)
    start = _now()
    phase8_estimate = estimate_phase_duration("phase8")
    print(f"  ⏱ Phase 8 开始 (预估 ~{int(phase8_estimate)}s)")
    output_dir = Path(output_dir)
    reshoot_history = list(_reshoot_history or [])
    exhausted_reshoot_policy = os.environ.get(
        "HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY", "fail"
    ).strip().lower()
    if exhausted_reshoot_policy not in {"fail", "assemble_best"}:
        raise ValueError(
            "HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY must be 'fail' or "
            "'assemble_best'"
        )

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频组装")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # Prove the complete storyboard/video/metadata identity before any pixel
    # analysis or paid reshoot can begin. Resume and manual artifact repair can
    # bypass Phase 4, so Phase 8 owns this invariant too.
    from phases.phase8.inventory import Phase8InventoryError, load_phase8_inventory
    from phases.phase8.reshoot_transaction import durable_attempt_count

    try:
        clip_paths, shot_metas = load_phase8_inventory(output_dir)
        _reshoot_round = max(_reshoot_round, durable_attempt_count(output_dir))
    except (Phase8InventoryError, RuntimeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "duration_s": _elapsed(start),
        }

    shots_dir = output_dir / "shots"

    # Step 8.1: compare storyboard narrative order with the current clip order.
    from phases.phase8.story_order_reviewer import reorder_shots, review_story_order

    current_order = [Path(path).parent.name for path in clip_paths]
    order_review = review_story_order(output_dir, current_order)
    if not order_review["matches_current_order"]:
        clip_paths, shot_metas, changed = reorder_shots(
            clip_paths, shot_metas, order_review["suggested_order"]
        )
        if changed:
            print(
                "  🔀 [8.1] 按剧情审稿建议重排镜头: "
                + " → ".join(Path(path).parent.name for path in clip_paths),
                flush=True,
            )
    if not order_review["narrative_consistent"]:
        print(f"  ⚠ [8.1] 剧情连贯性问题: {order_review['issues']}", flush=True)

    # Step 8.15: the complete chunk trajectory is now available.  Revisit
    # provisional Phase 6 boundaries before per-shot QA or formal assembly.
    from phases.phase8.continuity_adjudication import adjudicate_continuity_seams
    from quality.sam3_sidecar import phase8_sam3_endpoint

    try:
        with phase8_sam3_endpoint(output_dir) as sam3_url:
            continuity_adjudication = adjudicate_continuity_seams(
                output_dir,
                sam3_base_url=sam3_url,
            )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Phase 8 continuity adjudication failed: {exc}",
            "duration_s": _elapsed(start),
        }
    if continuity_adjudication.get("requires_human_review"):
        review_boundaries = [
            boundary["boundary_id"]
            for shot in continuity_adjudication.get("shots", [])
            for boundary in shot.get("boundaries", [])
            if boundary.get("action") == "human_review"
        ]
        return {
            "status": "error",
            "error": (
                "Phase 8 found appearance-level rollback evidence but requires "
                "object-trajectory or human corroboration: "
                + ", ".join(review_boundaries)
            ),
            "duration_s": _elapsed(start),
            "continuity_adjudication": continuity_adjudication,
            "review_artifact": "CONTINUITY_ADJUDICATION.json",
        }
    if continuity_adjudication.get("requires_phase6"):
        requests = [
            request
            for request in json.loads(
                (output_dir / "CONTINUITY_TOPUP_REQUESTS.json").read_text(encoding="utf-8")
            ).get("requests", [])
        ]
        summary = ", ".join(
            f"{item['shot_id']} 缺 {item['deficit_frames']} 帧" for item in requests
        )
        print(
            f"  ↩ [8.15] 检出内部回退，已写入硬裁剪裁决；{summary}，回流 Phase 6",
            flush=True,
        )
        if not enable_reshoot:
            return {
                "status": "error",
                "error": (
                    "Phase 8 temporal seam adjudication requires continuation top-up "
                    f"but enable_reshoot=false: {summary}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        if _continuity_round >= 2:
            return {
                "status": "error",
                "error": (
                    "Phase 8 continuity still requires top-up after 2 feedback rounds: "
                    f"{summary}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        storyboard_path = output_dir / "STORYBOARD.json"
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            generation = run_phase6(
                storyboard,
                output_dir,
                dry_run=False,
                chain_mode=chain_mode,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Phase 8 continuity top-up could not run Phase 6: {exc}",
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        if generation.get("status") == "error":
            return {
                "status": "error",
                "error": (
                    "Phase 8 continuity top-up failed in Phase 6: "
                    f"{generation.get('error') or generation.get('errors')}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
                "phase6_topup": generation,
            }
        history = reshoot_history + [
            {
                "kind": "continuity_topup",
                "round": _continuity_round + 1,
                "requests": requests,
                "phase6_status": generation.get("status"),
            }
        ]
        return run_phase8(
            output_dir,
            dry_run=False,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            target_duration=target_duration,
            enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
            _reshoot_round=_reshoot_round,
            _reshoot_history=history,
            _continuity_round=_continuity_round + 1,
        )

    # Step 8.2: dense per-shot review with actionable keep/trim/reshoot decisions.
    from phases.phase8.frame_analysis import analyze_shot_frames

    frame_report = analyze_shot_frames(shots_dir, output_dir / "frame_analysis.json")
    reshoot_shots = list(frame_report.get("summary", {}).get("reshoot", []))
    if (
        reshoot_shots
        and _reshoot_round >= 2
        and exhausted_reshoot_policy == "assemble_best"
    ):
        unresolved = list(reshoot_shots)
        print(
            "  ⚠ [8.2] 已达补录上限；按显式 assemble_best 策略组装最佳现有素材: "
            + ", ".join(unresolved),
            flush=True,
        )
        frame_report.setdefault("summary", {})["delivery_policy"] = "assemble_best"
        frame_report["summary"]["unresolved_after_reshoot_limit"] = unresolved
        reshoot_history.append(
            {
                "kind": "visual_quality_limit",
                "round": _reshoot_round,
                "shots": unresolved,
                "policy": "assemble_best",
            }
        )
        reshoot_shots = []
    if reshoot_shots:
        if not enable_reshoot:
            return {
                "status": "error",
                "error": (
                    "Phase 8 visual QA requires reshoot but enable_reshoot=false: "
                    + ", ".join(reshoot_shots)
                ),
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
            }
        if _reshoot_round >= 2:
            return {
                "status": "error",
                "error": f"Phase 8 visual QA still fails after 2 reshoot rounds: {', '.join(reshoot_shots)}",
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
                "continuity_adjudication": continuity_adjudication,
                "reshoot_history": reshoot_history,
            }

        from phases.phase8.reshoot_transaction import ReshootTransaction

        try:
            transaction = ReshootTransaction.begin(
                output_dir,
                kind="visual_quality",
                shot_ids=reshoot_shots,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Phase 8 could not prepare recoverable visual reshoot: {exc}",
                "duration_s": _elapsed(start),
            }

        for shot_id in reshoot_shots:
            video_path = shots_dir / shot_id / "output.mp4"
            # A rejected FLF2V result can be caused by an inconsistent
            # generated endpoint. Change the route only after its metadata and
            # source clip have both been backed up by the transaction.
            meta_path = shots_dir / shot_id / "SHOT_META.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            review_entry = frame_report.get("shots", {}).get(shot_id, {})
            semantic_review = review_entry.get("semantic_review") or {}
            reasons = review_entry.get("reasons") or semantic_review.get("issues") or []
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            meta["phase8_reshoot"] = {
                "round": _reshoot_round + 1,
                "qa_contract": semantic_review.get("qa_contract"),
                "issues": [str(reason) for reason in reasons if str(reason).strip()],
            }
            from utils.camera_motion_contracts import apply_camera_motion_contract

            apply_camera_motion_contract(meta)
            if meta.get("gen_strategy") == "flf2v":
                meta["gen_strategy"] = "phantom"
                meta["phase8_reshoot_route_reason"] = (
                    "FLF2V visual QA failure; avoid reusing a possibly "
                    "inconsistent generated endpoint"
                )
                print(
                    f"  ↪ [8.2] {shot_id}: FLF2V 补录改用 Phantom 角色参考路由",
                    flush=True,
                )
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        transaction.remove_sources()

        print(
            f"  🔄 [8.2] 视觉质检补录第 {_reshoot_round + 1}/2 轮: {', '.join(reshoot_shots)}",
            flush=True,
        )
        storyboard_path = output_dir / "STORYBOARD.json"
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            generation = run_phase6(
                storyboard, output_dir, dry_run=False, chain_mode=chain_mode
            )
        except Exception as exc:
            transaction.rollback(str(exc))
            return {
                "status": "error",
                "error": f"Phase 8 visual reshoot could not run Phase 6: {exc}",
                "duration_s": _elapsed(start),
            }
        if generation.get("status") == "error":
            failure = str(generation.get("error") or generation.get("errors"))
            transaction.rollback(failure)
            return {
                "status": "error",
                "error": (
                    "Phase 8 visual reshoot failed in Phase 6: "
                    f"{generation.get('error') or generation.get('errors')}"
                ),
                "duration_s": _elapsed(start),
            }
        missing_outputs = [
            shot_id for shot_id in reshoot_shots
            if not (shots_dir / shot_id / "output.mp4").is_file()
        ]
        if missing_outputs:
            failure = "Phase 6 reported success without regenerated clips: " + ", ".join(missing_outputs)
            transaction.rollback(failure)
            return {
                "status": "error",
                "error": failure,
                "duration_s": _elapsed(start),
            }
        try:
            transaction.commit()
        except Exception as exc:
            transaction.rollback(str(exc))
            return {
                "status": "error",
                "error": f"Phase 8 visual reshoot validation failed: {exc}",
                "duration_s": _elapsed(start),
            }
        history = reshoot_history + [{
            "kind": "visual_quality",
            "round": _reshoot_round + 1,
            "shots": reshoot_shots,
            "phase6_status": generation.get("status"),
        }]
        return run_phase8(
            output_dir,
            dry_run=False,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            target_duration=target_duration,
            enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
            _reshoot_round=_reshoot_round + 1,
            _reshoot_history=history,
        )

    reviewed_order = [Path(path).parent.name for path in clip_paths]

    # Intelligent transition selection based on shot emotions
    print(f"  → 发现 {len(clip_paths)} 个视频片段")
    
    # ── Smart transition: visual similarity + three-layer voting ──
    smart_decisions = None
    try:
        from utils.shot_embedder import embed_all_shots, compute_transition_similarity
        from tools.smart_transition import decide_all_transitions
        
        print("  → 智能转场: 抽帧 + 向量化 + 三层决策...")
        embeddings = embed_all_shots(str(shots_dir), run_id=str(output_dir.name))
        if embeddings:
            similarities = compute_transition_similarity(embeddings, reviewed_order)
            smart_decisions = decide_all_transitions(
                shot_metas,
                similarities,
                shot_ids=reviewed_order,
            )
            
            # Log decisions
            for d in smart_decisions:
                sim_str = f"{d['layers']['visual']['similarity']:.2f}" if d['layers']['visual']['similarity'] >= 0 else "N/A"
                print(f"    • {d['pair']}: {d['decision']} "
                      f"(语义={d['layers']['semantic']['choice']}, "
                      f"视觉={d['layers']['visual']['choice']}[{sim_str}], "
                      f"节奏={d['layers']['rhythm']['choice']})")
    except Exception as e:
        print(f"  ⚠ 智能转场不可用: {e}，降级为情绪映射")
        smart_decisions = None
    
    # Select transition for each shot (except last, which has no "next")
    selected_transitions = []
    for i, shot_meta in enumerate(shot_metas[:-1]):  # Last shot doesn't need a transition
        if smart_decisions and i < len(smart_decisions):
            sel_transition = smart_decisions[i]["decision"]
        else:
            sel_transition = _select_transition(shot_meta, default_transition=transition)
        selected_transitions.append(sel_transition)
        shot_name = reviewed_order[i]
        emotion = shot_meta.get("emotion", "N/A")
        source = "智能" if (smart_decisions and i < len(smart_decisions)) else "情绪"
        print(f"    • {shot_name} → {sel_transition} ({source}, emotion: {emotion})")
    
    # Determine the most common transition type for batch processing
    if selected_transitions:
        from collections import Counter
        transition_counts = Counter(selected_transitions)
        batch_transition = transition_counts.most_common(1)[0][0]
        
        # Check if all transitions are the same
        all_same = len(transition_counts) == 1
        
        if all_same:
            print(f"  → 拼接模式: {batch_transition} (所有镜头统一)")
        else:
            print(f"  → 拼接模式: {batch_transition} (混合模式，使用最常用类型)")
            print(f"    分布: {dict(transition_counts)}")
    else:
        batch_transition = transition
        print(f"  → 拼接模式: {batch_transition} (duration={transition_duration}s)")

    stitch_transition = {
        "dissolve": "crossfade",
        "fade": "fade_through_black",
    }.get(batch_transition, batch_transition)
    if stitch_transition not in {"cut", "crossfade", "fade_through_black"}:
        stitch_transition = "crossfade"
    stitch_plan = build_stitch_plan(
        [
            {"path": path, "duration": shot_metas[index].get("duration", 0) if index < len(shot_metas) else 0}
            for index, path in enumerate(clip_paths)
        ],
        stitch_transition,
        transition_duration,
    )
    clip_paths = stitch_plan.clips

    # The reviewed edit-decision path is primary: this is where per-shot trim
    # decisions become real frame-accurate cuts. Generic concat is fallback.
    transition_dicts = (
        [{"decision": value} for value in selected_transitions]
        if selected_transitions else None
    )
    continuity_plan_path = output_dir / "CONTINUITY_PLAN.json"
    continuity_plan_for_edit = (
        json.loads(continuity_plan_path.read_text(encoding="utf-8"))
        if continuity_plan_path.is_file()
        else None
    )
    has_continuity_transition_lock = any(
        shot.get("boundary_before") == "continuous"
        for shot in (continuity_plan_for_edit or {}).get("shots", [])
    )
    reviewed_edit_error = "reviewed edit path did not complete"
    reviewed_edit_execution_started = False
    try:
        from phases.phase8.edit_decisions import build_edit_decisions, execute_edit_decisions

        print("  → 构建 reviewed edit_decisions（质检裁切 + 音频归一化）...")
        assembly_profile = _get_profile_dict(media_profile)
        edit_decisions = build_edit_decisions(
            shots_dir=shots_dir,
            target_width=int(assembly_profile["width"]),
            target_height=int(assembly_profile["height"]),
            transition_decisions=transition_dicts,
            quality_report=frame_report,
            shot_order=reviewed_order,
            target_duration=target_duration,
            transition_duration=transition_duration,
            fit_mode="cover",
            continuity_plan=continuity_plan_for_edit,
            allow_unresolved_reshoots=(
                exhausted_reshoot_policy == "assemble_best" and _reshoot_round >= 2
            ),
        )
        print(f"  → 执行 reviewed edit_decisions（{len(edit_decisions['cuts'])} 个片段）...")
        reviewed_edit_execution_started = True
        reviewed_edit = execute_edit_decisions(
            edit_decisions,
            output_path=str(output_dir / "raw_assembly.mp4"),
        )
        if reviewed_edit.get("success"):
            print("  ✓ Phase 8 完成: raw_assembly.mp4 (reviewed_edit_decisions)")
            from phases.phase9.audio_mixer import prepare_phase9_audio_assets

            audio_receipt = prepare_phase9_audio_assets(output_dir)
            qg_report = run_quality_check("phase8", output_dir)
            if not qg_report.passed:
                return {
                    "status": "error",
                    "error": f"Phase 8 质检未通过: {qg_report.grade}",
                    "quality_report": qg_report,
                    "duration_s": _elapsed(start),
                }
            return _finish_phase8({
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4", "edit_timeline.json"],
                "method": "reviewed_edit_decisions",
                "transition": batch_transition,
                "transition_duration": transition_duration,
                "clip_count": len(edit_decisions["cuts"]),
                "transition_selections": selected_transitions or None,
                "edit_decisions_segments": reviewed_edit.get("segments"),
                "audio_transition_policy": edit_decisions.get("metadata", {}).get(
                    "audio_transition_policy"
                ),
                "transition_locks": edit_decisions.get("metadata", {}).get(
                    "transition_locks", []
                ),
                "audio_transition_counts": {
                    kind: sum(
                        item.get("audio_transition") == kind
                        for item in edit_decisions.get("transitions", [])
                    )
                    for kind in ("edge_fade", "crossfade")
                },
                "frame_analysis": frame_report.get("summary", {}),
                "reshoot_history": reshoot_history,
                "audio_layer": audio_receipt,
            }, output_dir, target_duration, enable_reshoot, transition,
               transition_duration, media_profile, _reshoot_round, reshoot_history,
               chain_mode)
        reviewed_edit_error = str(reviewed_edit.get("error", "unknown error"))
        return {
            "status": "error",
            "error": f"Phase 8 reviewed edit execution failed: {reviewed_edit_error}",
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }
    except Exception as exc:
        reviewed_edit_error = str(exc)
        if reviewed_edit_execution_started:
            return {
                "status": "error",
                "error": f"Phase 8 reviewed edit execution failed: {reviewed_edit_error}",
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
            }
        print(f"  ⚠ reviewed edit_decisions 构建异常: {exc}；降级为 VideoEdit", flush=True)

    if frame_report.get("summary", {}).get("trim"):
        return {
            "status": "error",
            "error": (
                "Phase 8 cannot safely fall back to raw concat because reviewed trims are required: "
                f"{reviewed_edit_error}"
            ),
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }
    if has_continuity_transition_lock:
        return {
            "status": "error",
            "error": (
                "Phase 8 cannot safely fall back to batch transitions because "
                f"continuous boundary locks are required: {reviewed_edit_error}"
            ),
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }

    # Generic concat cannot apply per-shot decisions, so it is only reached
    # when the reviewed editor itself is unavailable or fails technically.
    try:
        from tools.video.video_edit import VideoEdit

        editor = VideoEdit()
        concat_output = output_dir / ".video_edit_concat.mp4"
        final_output = output_dir / "raw_assembly.mp4"
        video_edit_transition = "cut" if stitch_plan.transition == "cut" else "crossfade"
        call_started = time.time()
        print(
            f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] VideoEdit.concat: "
            f"{len(clip_paths)} clips, transition={video_edit_transition}, "
            f"crossfade={stitch_plan.duration}s"
        )
        concat_result = editor.execute({
            "operation": "concat",
            "input_paths": clip_paths,
            "output_path": str(concat_output),
            "transition": video_edit_transition,
            "crossfade_duration": stitch_plan.duration,
        })
        print(
            f"    VideoEdit.concat result: success={concat_result.success}, "
            f"elapsed={time.time() - call_started:.1f}s, "
            f"output={concat_output if concat_result.success else None}, "
            f"error={concat_result.error}"
        )
        if not concat_result.success:
            raise RuntimeError(concat_result.error or "VideoEdit.concat failed")

        # Trim the assembled container to its exact computed timeline.  Using
        # actual probed clip durations avoids stale SHOT_META duration values.
        clip_durations = []
        for clip_path in clip_paths:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=30, check=True,
            )
            clip_durations.append(float(probe.stdout.strip().splitlines()[0]))
        trim_end = sum(clip_durations)
        if video_edit_transition == "crossfade":
            trim_end -= stitch_plan.duration * (len(clip_paths) - 1)

        trim_started = time.time()
        print(
            f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] VideoEdit.trim: "
            f"start=0.0s, end={trim_end:.3f}s"
        )
        trim_result = editor.execute({
            "operation": "trim",
            "input_path": str(concat_output),
            "output_path": str(final_output),
            "start_time": 0.0,
            "end_time": trim_end,
        })
        print(
            f"    VideoEdit.trim result: success={trim_result.success}, "
            f"elapsed={time.time() - trim_started:.1f}s, output={final_output}, "
            f"error={trim_result.error}"
        )
        if not trim_result.success:
            raise RuntimeError(trim_result.error or "VideoEdit.trim failed")
        concat_output.unlink(missing_ok=True)

        print(f"  ✓ Phase 8 完成: raw_assembly.mp4 (VideoEdit)")
        from phases.phase9.audio_mixer import prepare_phase9_audio_assets
        audio_receipt = prepare_phase9_audio_assets(output_dir)
        qg_report = run_quality_check("phase8", output_dir)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 8 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
        return _finish_phase8({
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": ["raw_assembly.mp4"],
            "method": "VideoEdit",
            "transition": video_edit_transition,
            "transition_duration": stitch_plan.duration,
            "clip_count": len(clip_paths),
            "transition_selections": selected_transitions if selected_transitions else None,
            "trim_end_s": round(trim_end, 3),
            "frame_analysis": frame_report.get("summary", {}),
            "reshoot_history": reshoot_history,
            "audio_layer": audio_receipt,
        }, output_dir, target_duration, enable_reshoot, transition,
           transition_duration, media_profile, _reshoot_round, reshoot_history,
           chain_mode)
    except Exception as e:
        try:
            (output_dir / ".video_edit_concat.mp4").unlink(missing_ok=True)
        except OSError:
            pass
        print(f"  ⚠ VideoEdit 失败: {e}，降级为 VideoStitch")

    # Final fallback: OM VideoStitch for keep-only projects.
    try:
        from vendor.video_tools.tools.video.video_stitch import VideoStitch
        stitcher = VideoStitch()
        result = stitcher.execute({
            "operation": "stitch",
            "clips": clip_paths,
            "output_path": str(output_dir / "raw_assembly.mp4"),
            "transition": stitch_plan.transition,
            "transition_duration": stitch_plan.duration,
            "auto_normalize": True,
            "profile": media_profile,
        })

        if result.success:
            print(f"  ✓ Phase 8 完成: raw_assembly.mp4 (VideoStitch fallback)")
            from phases.phase9.audio_mixer import prepare_phase9_audio_assets
            audio_receipt = prepare_phase9_audio_assets(output_dir)
            
            # Quality gate: Phase 8
            qg_report = run_quality_check("phase8", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 8 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
            return _finish_phase8({
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4"],
                "method": f"VideoStitch_{batch_transition}_fallback",
                "transition": stitch_plan.transition,
                "transition_duration": stitch_plan.duration,
                "clip_count": len(clip_paths),
                "transition_selections": selected_transitions if selected_transitions else None,
                "stitch_offsets": stitch_plan.offsets,
                "frame_analysis": frame_report.get("summary", {}),
                "reshoot_history": reshoot_history,
                "audio_layer": audio_receipt,
            }, output_dir, target_duration, enable_reshoot, transition,
               transition_duration, media_profile, _reshoot_round, reshoot_history,
               chain_mode)
        else:
            return {"status": "error", "error": result.error, "duration_s": _elapsed(start)}

    except ImportError as e:
        return {"status": "error", "error": f"VideoStitch unavailable: {e}", "duration_s": _elapsed(start)}


# Phase 9: 后期处理 (audio + visual + rhythm → polished.mp4)
# ---------------------------------------------------------------------------

def _detect_bgm(output_dir: Path, storyboard_path: Optional[Path] = None) -> Optional[str]:
    """
    Detect background music file for Phase 9 audio processing.

    Search order:
    1. BGM referenced in STORYBOARD.json (metadata.bgm_path)
    2. Common BGM filenames in output_dir (bgm.mp3, bg_music.mp3, etc.)
    3. Any .mp3/.wav/.aac file in output_dir/audio/ subdirectory

    Returns:
        Path to BGM file as string, or None if not found.
    """
    # 1. Check storyboard metadata
    if storyboard_path and storyboard_path.exists():
        try:
            sb_data = json.loads(storyboard_path.read_text())
            bgm = sb_data.get("metadata", {}).get("bgm_path")
            if bgm and Path(bgm).exists():
                return str(bgm)
            # Also check top-level bgm field
            bgm = sb_data.get("bgm_path") or sb_data.get("bgm")
            if bgm and Path(bgm).exists():
                return str(bgm)
        except (json.JSONDecodeError, KeyError):
            pass

    # 2. Check common BGM filenames in output_dir
    common_bgm_names = ["bgm.mp3", "bgm.wav", "bg_music.mp3", "background_music.mp3",
                        "music.mp3", "soundtrack.mp3", "ost.mp3"]
    for name in common_bgm_names:
        candidate = output_dir / name
        if candidate.exists():
            return str(candidate)

    # 3. Check audio subdirectory
    audio_dir = output_dir / "audio"
    if audio_dir.exists():
        for ext in ("*.mp3", "*.wav", "*.aac", "*.m4a"):
            matches = list(audio_dir.glob(ext))
            if matches:
                return str(matches[0])

    return None


def _prepare_continuous_bgm(
    bgm_path: str,
    target_duration: float,
    output_path: Path,
    *,
    crossfade_s: float = 2.0,
) -> str:
    """Extend one score across the film with equal-power loop crossfades."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", bgm_path],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    source_duration = float(probe.stdout.strip().splitlines()[0])
    if source_duration <= 0 or target_duration <= 0:
        raise ValueError("BGM and target durations must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fade = min(crossfade_s, max(0.1, source_duration / 4))
    count = max(1, math.ceil((target_duration - fade) / max(0.1, source_duration - fade)))
    inputs = [item for _ in range(count) for item in ("-i", bgm_path)]
    filters = [f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}]" for index in range(count)]
    current = "a0"
    for index in range(1, count):
        output = f"mix{index}"
        filters.append(f"[{current}][a{index}]acrossfade=d={fade}:c1=qsin:c2=qsin[{output}]")
        current = output
    filters.append(
        f"[{current}]atrim=duration={target_duration},"
        f"afade=t=in:d=1,afade=t=out:st={max(0.0, target_duration - 2.0)}:d=2[out]"
    )
    command = [
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", "[out]", "-c:a", "aac", "-b:a", "192k", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=600)
    if completed.returncode != 0 or not output_path.is_file():
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"Failed to prepare continuous BGM: {detail}")
    return str(output_path)


def _probe_shot_duration(shots_dir: Path, shot_id: int) -> float:
    """Probe the real duration of a shot video via ffprobe.

    Falls back to 2.0s if the file is missing or ffprobe fails.
    """
    shot_video = shots_dir / f"S{shot_id:02d}" / "output.mp4"
    if not shot_video.exists():
        return 2.0
    try:
        import subprocess as _sp
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(shot_video),
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return 2.0


def _write_srt(segments: list, srt_path: str) -> None:
    """Write segments to an SRT subtitle file as fallback."""
    import os
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    lines = []
    for idx, seg in enumerate(segments, 1):
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)
        text = seg.get("text", "")
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}")
        lines.append(text)
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_subtitle_text(text: str) -> str:
    """去除中英文标点并规范空白。

    中文分词由 ASR word segment 边界提供；本函数不猜测中文词界，
    因此没有 word 级数据时只去标点，不强行切字。
    """
    import unicodedata

    if not isinstance(text, str):
        return ""
    without_punctuation = "".join(
        character for character in text
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _merge_shot_transcripts(
    sb_shots: list,
    durations_ms: list[int],
    shot_transcripts: list[dict],
    edit_timeline: Optional[dict] = None,
) -> dict:
    """Offset per-shot ASR words and create caption-burn segments.

    In a scripted-dialogue scene, the authored dialogue contract is the source
    of truth: ASR may validate audible speech but cannot replace the line or
    invent dialogue on a shot explicitly authored without one. In a fully
    unscripted scene, ASR remains authoritative.
    """
    merged_words = []
    caption_segments = []
    cumulative_ms = 0
    shot_entries = []
    timeline_by_shot = {
        str(item.get("shot_id")): item
        for item in (edit_timeline or {}).get("shots", [])
        if isinstance(item, dict) and item.get("shot_id")
    }
    scripted_scene = any(
        clean_subtitle_text(
            (shot.get("dialogue") or {}).get("line", "")
            if isinstance(shot.get("dialogue"), dict)
            else str(shot.get("dialogue") or "")
        )
        for shot in sb_shots
    )
    for index, (shot, duration_ms, transcription) in enumerate(
        zip(sb_shots, durations_ms, shot_transcripts), 1
    ):
        shot_id = str(shot.get("shot_id") or f"S{index:02d}")
        timeline_item = timeline_by_shot.get(shot_id)
        if timeline_item:
            shot_start_ms = round(float(timeline_item.get("output_start_s", 0.0)) * 1000)
            shot_duration_ms = round(float(timeline_item.get("output_duration_s", 0.0)) * 1000)
            source_in_ms = round(float(timeline_item.get("source_in_s", 0.0)) * 1000)
            speed = float(timeline_item.get("speed", 1.0) or 1.0)
        else:
            shot_start_ms = cumulative_ms
            shot_duration_ms = duration_ms
            source_in_ms = 0
            speed = 1.0
        if duration_ms <= 0 or transcription.get("skipped"):
            # Shot missing output.mp4 — skip caption generation entirely
            shot_entries.append({
                "shot_id": shot_id,
                "text": "",
                "source": "skipped",
                "start_ms": shot_start_ms,
                "end_ms": shot_start_ms,
                "segments": [],
            })
            continue
        local_words = transcription.get("segments") or []
        asr_text = clean_subtitle_text(transcription.get("text") or "")
        dialogue = shot.get("dialogue")
        dialogue_line = (
            dialogue.get("line", "") if isinstance(dialogue, dict) else str(dialogue or "")
        )
        scripted_text = clean_subtitle_text(dialogue_line)
        if scripted_text:
            text = scripted_text
            source = "dialogue_script"
            words = [{
                "word": text,
                "start_ms": round(shot_start_ms + shot_duration_ms * 0.2),
                "end_ms": round(shot_start_ms + shot_duration_ms * 0.8),
                "source": source,
            }]
        elif scripted_scene:
            # Do not let ASR hallucinations introduce dialogue into a shot that
            # the authored scripted scene explicitly leaves silent.
            words = []
            text = ""
            source = "none"
        elif local_words or asr_text:
            words = []
            for item in local_words:
                cleaned_word = clean_subtitle_text(item.get("word") or item.get("text") or "")
                if not cleaned_word:
                    continue
                source_start_ms = int(item["start_ms"])
                source_end_ms = int(item["end_ms"])
                # Words outside the retained source window must not leak into
                # the edited timeline after head/tail trims.
                source_out_ms = source_in_ms + round(shot_duration_ms * speed)
                if source_end_ms <= source_in_ms or source_start_ms >= source_out_ms:
                    continue
                mapped_start_ms = shot_start_ms + max(
                    0, round((source_start_ms - source_in_ms) / speed)
                )
                mapped_end_ms = min(
                    shot_start_ms + shot_duration_ms,
                    shot_start_ms + max(
                        0, round((source_end_ms - source_in_ms) / speed)
                    ),
                )
                if mapped_end_ms <= mapped_start_ms:
                    continue
                words.append({
                    "word": cleaned_word,
                    "start_ms": mapped_start_ms,
                    "end_ms": mapped_end_ms,
                    "source": "asr",
                })
            text = " ".join(item["word"] for item in words) if words else asr_text
            if not words and text:
                words = [{
                    "word": text,
                    "start_ms": shot_start_ms,
                    "end_ms": shot_start_ms + shot_duration_ms,
                    "source": "asr",
                }]
            source = "asr"
        else:
            words = []
            text = ""
            source = "none"

        merged_words.extend(words)
        shot_entries.append({
            "shot_id": shot_id,
            "text": text,
            "source": source,
            "start_ms": shot_start_ms,
            "end_ms": shot_start_ms + shot_duration_ms,
            "segments": words,
        })
        if words:
            caption_segments.append({
                "text": text,
                "start": words[0]["start_ms"] / 1000,
                "end": words[-1]["end_ms"] / 1000,
                "source": source,
                "words": [{
                    "word": item["word"],
                    "start": item["start_ms"] / 1000,
                    "end": item["end_ms"] / 1000,
                    "source": item["source"],
                } for item in words],
            })
        cumulative_ms += duration_ms
    output_duration_ms = round(
        float((edit_timeline or {}).get("duration_s", 0.0)) * 1000
    ) or cumulative_ms
    return {
        "text": "".join(entry["text"] for entry in shot_entries if entry["text"]),
        "duration_ms": output_duration_ms,
        "segments": merged_words,
        "shots": shot_entries,
        "caption_segments": caption_segments,
    }


def _caption_segments_from_final_asr(transcription: dict) -> list[dict]:
    """Build cue-preserving captions from the audible final mix."""
    captions = []
    utterances = transcription.get("utterances") or []
    if not utterances and (transcription.get("segments") or transcription.get("text")):
        utterances = [{
            "text": transcription.get("text", ""),
            "words": transcription.get("segments") or [],
        }]
    for utterance in utterances:
        cleaned_words = []
        for item in utterance.get("words") or []:
            cleaned = clean_subtitle_text(item.get("word") or item.get("text") or "")
            start_ms = int(item.get("start_ms", item.get("start_time", -1)))
            end_ms = int(item.get("end_ms", item.get("end_time", -1)))
            if not cleaned or start_ms < 0 or end_ms <= start_ms:
                continue
            cleaned_words.append({
                "word": cleaned,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
                "source": "final_mix_asr",
            })
        word_groups = []
        for word in cleaned_words:
            if word_groups and word["start"] - word_groups[-1][-1]["end"] >= 0.25:
                word_groups.append([])
            if not word_groups:
                word_groups.append([])
            word_groups[-1].append(word)
        if not word_groups:
            text = clean_subtitle_text(utterance.get("text") or "")
            start_ms = int(utterance.get("start_ms", utterance.get("start_time", -1)))
            end_ms = int(utterance.get("end_ms", utterance.get("end_time", -1)))
            if not text or start_ms < 0 or end_ms <= start_ms:
                continue
            word_groups = [[{
                "word": text,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
                "source": "final_mix_asr",
            }]]
        for group in word_groups:
            # Generated impacts and metallic transients can be decoded as a
            # run of implausibly short syllables. Without confidence scores,
            # reject only the narrow high-speed pattern that repeatedly caused
            # unstable text across ASR passes; normal short interjections stay.
            mean_word_duration = sum(
                item["end"] - item["start"] for item in group
            ) / len(group)
            if len(group) >= 2 and mean_word_duration < 0.1:
                continue
            captions.append({
                "text": "".join(item["word"] for item in group),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "source": "final_mix_asr",
                "words": group,
            })
    return captions


def _phase9_real_audio_tracks(
    output_dir: Path,
    storyboard_data: Optional[dict],
    transcript_data: Optional[dict],
    raw_video: Path,
    bgm_path: Optional[str] = None,
) -> tuple[list[dict], int]:
    """Build the real-audio base and narration tracks for Phase 9."""
    tracks = [{"path": str(raw_video), "role": "music"}]
    if bgm_path:
        tracks.append({"path": bgm_path, "role": "music", "volume": 0.18})
    audio_options = storyboard_data.get("audio", {}) if storyboard_data else {}
    if not storyboard_data or not audio_options.get("enabled", False) or not audio_options.get("tts", True):
        return tracks, 0

    from phases.phase9.audio_mixer import AudioMixer as Phase7AudioMixer

    transcript_shots = (transcript_data or {}).get("shots", [])
    skipped = 0
    for index, shot in enumerate(storyboard_data.get("shots", []), 1):
        narration_path = output_dir / "audio_layer" / f"narration_{index:03d}.mp3"
        spoken_text = Phase7AudioMixer._spoken_text(
            shot.get("narration") or shot.get("voiceover") or shot.get("dialogue")
        )
        if not spoken_text or not narration_path.is_file():
            continue

        transcript_shot = transcript_shots[index - 1] if index <= len(transcript_shots) else {}
        # Only ASR text proves the line exists in source audio. Script fallback
        # text is not evidence and must never suppress an overlay.
        asr_text = transcript_shot.get("text", "") if transcript_shot.get("source") == "asr" else ""
        normalized_line = "".join(clean_subtitle_text(spoken_text).casefold().split())
        normalized_asr = "".join(clean_subtitle_text(asr_text).casefold().split())
        if normalized_line and normalized_asr and normalized_line in normalized_asr:
            skipped += 1
            shot_id = shot.get("shot_id") or f"S{index:02d}"
            print(f"    ⊘ [P0-D3] {shot_id}: TTS skipped (dialogue already in source audio)")
            continue

        tracks.append({
            "path": str(narration_path),
            "role": "speech",
            "start_seconds": float(transcript_shot.get("start_ms", 0)) / 1000,
        })
    return tracks, skipped


def _phase9_real_audio_mix_request(tracks: list[dict], audio_out: Path) -> dict:
    """Return the AudioMixer request for a preserved real-audio base track."""
    has_tts = any(track.get("role") == "speech" for track in tracks)
    return {
        "operation": "full_mix" if has_tts else "mix",
        "tracks": tracks,
        "ducking": {
            "enabled": True,
            "music_volume_during_speech": 0.15,
        } if has_tts else None,
        "normalize": True,
        "loudnorm_target": -14,
        "output_path": str(audio_out),
    }



def run_phase9(output_dir: Path, dry_run: bool, color_grade: Optional[str] = None,
               upscale: Optional[int] = None, media_profile: str = "1080p",
               target_duration: Optional[float] = None) -> dict:
    """Phase 9: audio_pipeline + visual_post + [color_grade] + [upscale] + rhythm_editor → polished.mp4

    Audio processing (enhanced with OM AudioMixer capabilities):
    - Loudness normalization (loudnorm filter, target -14 LUFS)
    - Background music ducking (sidechaincompress when BGM detected)
    - Fade in/out (1s fade-in, 2s fade-out)
    - Falls back to basic FFmpeg processing if enhanced pipeline fails
    - Uses lib.media_profiles (OM) for final output encoding parameters

    Args:
        color_grade: Optional color profile name (cinematic_warm, cinematic_cool, moody_dark,
                     bright_clean, vintage_film, high_contrast, neutral)
        upscale: Optional target height in pixels (e.g. 720 for 720p output)
        media_profile: encoding profile name (default: "1080p")
    """
    _banner(9, 9, "后期处理 (Post-Production)", dry_run)
    start = _now()
    _p8_est = estimate_phase_duration("phase9")
    print(f"  ⏱ Phase 9 开始 (预估 ~{int(_p8_est)}s)")
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过后期处理")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    raw_video = output_dir / "raw_assembly.mp4"
    if not raw_video.exists():
        return {"status": "skipped", "reason": "no raw_assembly.mp4", "duration_s": _elapsed(start)}

    outputs = []
    storyboard_path = output_dir / "STORYBOARD.json"
    sb_path_str = str(storyboard_path) if storyboard_path.exists() else None
    storyboard_data = None

    # --- P0-D3: Check whether the audio track is genuinely audible ──────────
    # Previously this only checked "has audio stream" — but local Wan2.2 videos
    # have an anullsrc-injected silent track from edit_decisions normalisation.
    # Now we run volumedetect: mean_volume < -60 dB → treat as silent → run
    # ambient fallback so the final video is never silent.
    has_real_audio = False
    try:
        from tools.audio_pipeline import is_silent_audio
        import subprocess as _sp
        # First check: does an audio stream exist at all?
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                     "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                     str(raw_video)]
        probe_result = _sp.run(probe_cmd, capture_output=True, text=True, timeout=10)
        has_stream = bool(probe_result.stdout.strip())
        if has_stream:
            # Second check: is it actually audible?
            has_real_audio = not is_silent_audio(str(raw_video))
    except Exception:
        pass

    if has_real_audio:
        print("  → [P0-D3] 视频已有真实音轨（Seedance generate_audio），跳过环境音合成")

    # Step 9.1: transcribe every source shot before subtitle rendering. Keep
    # this outside Phase 9's broad post-processing guard so extraction/API
    # failures propagate instead of being reported as an ordinary soft failure.
    transcript_data = None
    if sb_path_str:
        from clients.asr_client import transcribe_audio
        from tools.audio_pipeline import extract_audio_track, is_silent_audio

        storyboard_data = json.loads(storyboard_path.read_text(encoding="utf-8"))
        sb_shots = storyboard_data.get("shots", [])
        shots_dir = output_dir / "shots"
        asr_receipts_dir = output_dir / "asr_transcripts"
        asr_receipts_dir.mkdir(parents=True, exist_ok=True)
        durations_ms = []
        shot_transcripts = []
        print("  → asr_transcription: 逐镜提取音轨并转写...")
        for index, _shot in enumerate(sb_shots, 1):
            shot_dir = shots_dir / f"S{index:02d}"
            shot_video = shot_dir / "output.mp4"
            wav_path = shot_dir / "audio.wav"
            if not shot_video.is_file():
                print(f"    ⚠ S{index:02d}: output.mp4 缺失，跳过 ASR（该镜未进入成片）")
                durations_ms.append(0)
                shot_transcripts.append({"text": "", "segments": [], "skipped": True})
                continue
            durations_ms.append(round(_probe_shot_duration(shots_dir, index) * 1000))
            shot_id = _shot.get("shot_id") or _shot.get("id") or f"S{index:02d}"
            if is_silent_audio(str(shot_video)):
                print(f"    ⊘ S{index:02d}: 无可听音轨，跳过 ASR")
                transcription = {
                    "text": "",
                    "segments": [],
                    "skipped": True,
                    "reason": "no_audible_audio",
                }
                shot_transcripts.append(transcription)
                receipt = {
                    "shot_id": str(shot_id),
                    "audio_path": None,
                    "duration_ms": durations_ms[-1],
                    "transcription": transcription,
                }
                (asr_receipts_dir / f"S{index:02d}.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                continue
            extract_audio_track(str(shot_video), str(wav_path))
            transcription = transcribe_audio(str(wav_path))
            shot_transcripts.append(transcription)
            receipt = {
                "shot_id": str(shot_id),
                "audio_path": str(wav_path),
                "duration_ms": durations_ms[-1],
                "transcription": transcription,
            }
            (asr_receipts_dir / f"S{index:02d}.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        timeline_path = output_dir / "edit_timeline.json"
        edit_timeline = None
        if timeline_path.is_file():
            edit_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        transcript_data = _merge_shot_transcripts(
            sb_shots,
            durations_ms,
            shot_transcripts,
            edit_timeline=edit_timeline,
        )
        transcript_data["asr_summary"] = {
            "shots_considered": len(shot_transcripts),
            "shots_submitted": sum(not item.get("skipped") for item in shot_transcripts),
            "shots_skipped_no_audio": sum(
                item.get("reason") == "no_audible_audio" for item in shot_transcripts
            ),
            "shots_with_text": sum(bool(item.get("text") or item.get("segments"))
                                   for item in shot_transcripts),
            "raw_word_segments": sum(len(item.get("segments") or [])
                                     for item in shot_transcripts),
            "caption_segments": len(transcript_data["caption_segments"]),
            "receipts_dir": "asr_transcripts",
        }
        transcript_path = output_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs.append("transcript.json")
        outputs.append("asr_transcripts/")
        summary = transcript_data["asr_summary"]
        print(
            "    ✓ ASR 完成: "
            f"{summary['shots_with_text']}/{summary['shots_considered']} 镜有语音, "
            f"{summary['raw_word_segments']} 个原始词段; "
            f"生成 {summary['caption_segments']} 条字幕"
        )

    try:
        from phases.phase9.visual_post import process_visual
        from phases.phase9.rhythm_editor import edit_rhythm

        # Track step statuses for quality gate integrity
        step_status = {}
        if transcript_data is not None:
            step_status["asr_transcription"] = "done"

        # Step 9.1: Audio processing via OM AudioMixer
        bgm_path = _detect_bgm(output_dir, storyboard_path)
        if (
            not bgm_path
            and storyboard_data
            and storyboard_data.get("audio", {}).get("enabled", False)
        ):
            try:
                from phases.phase9.audio_mixer import AudioMixer as Phase9MaterialMixer

                mood = (
                    storyboard_data.get("audio", {}).get("mood")
                    or storyboard_data.get("metadata", {}).get("mood")
                )
                selected_bgm = Phase9MaterialMixer().select_bgm(mood, target_duration)
                bgm_path = selected_bgm.path if selected_bgm else None
            except Exception as exc:
                print(f"    ⚠ 全局配乐选择不可用: {exc}")
        if bgm_path:
            try:
                bgm_path = _prepare_continuous_bgm(
                    bgm_path,
                    float(
                        target_duration
                        or _probe_av_durations(raw_video)["video"]
                        or 0.0
                    ),
                    output_dir / "audio_layer" / "continuous_bgm.m4a",
                )
                outputs.append("audio_layer/continuous_bgm.m4a")
                print("    ✓ 全局配乐已跨全片延展，并对循环点做等功率交叉淡化")
            except Exception as exc:
                print(f"    ⚠ 全局配乐延展失败，使用原始曲目: {exc}")
        if has_real_audio:
            audio_out = output_dir / "audio_processed.mp4"
            from vendor.video_tools.tools.audio.audio_mixer import AudioMixer

            base_audio = output_dir / "audio_layer" / "source_audio.m4a"
            base_audio.parent.mkdir(parents=True, exist_ok=True)
            extract_base = [
                "ffmpeg", "-y", "-i", str(raw_video), "-vn",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                str(base_audio),
            ]
            import subprocess as _sp
            extracted = _sp.run(extract_base, capture_output=True, text=True)
            base_track = base_audio if extracted.returncode == 0 and base_audio.is_file() else raw_video
            tracks, skipped_tts = _phase9_real_audio_tracks(
                output_dir, storyboard_data, transcript_data, base_track, bgm_path
            )
            overlay_count = sum(track.get("role") == "speech" for track in tracks)
            mixer = AudioMixer()
            mix_result = mixer.execute(_phase9_real_audio_mix_request(tracks, audio_out))
            audio_success = bool(mix_result.success)
            if audio_success:
                remux_tmp = output_dir / "audio_remux_tmp.mp4"
                remux_cmd = [
                    "ffmpeg", "-y", "-i", str(raw_video), "-i", str(audio_out),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-af", "apad", "-shortest", str(remux_tmp),
                ]
                import subprocess as _sp
                _sp.run(remux_cmd, capture_output=True, check=True)
                import shutil
                shutil.move(str(remux_tmp), str(audio_out))
                outputs.append("audio_processed.mp4")
                step_status["audio_pipeline"] = "done"
                if overlay_count:
                    print(
                        f"  ✓ [P0-D3] base track preserved + {overlay_count} TTS overlays "
                        f"({skipped_tts} skipped: dialogue already in source audio)"
                    )
                else:
                    print("  ✓ [P0-D3] base track only, loudnorm applied")
            else:
                print(f"  ⚠ [P0-D3] real-audio processing failed: {mix_result.error}")
                step_status["audio_pipeline"] = "failed"
                import shutil
                shutil.copy2(str(raw_video), audio_out)
        else:
            print("  → audio_pipeline: 音频处理 (AudioMixer: loudnorm + ducking)...")
            audio_out = output_dir / "audio_processed.mp4"

            # Detect background music for ducking
            if bgm_path:
                print(f"    ✓ BGM detected: {Path(bgm_path).name}")
            else:
                print(f"    ⊘ No BGM detected (skipping ducking)")

            audio_success = False
            try:
                from vendor.video_tools.tools.audio.audio_mixer import AudioMixer
                mixer = AudioMixer()

                # Prepare tracks
                tracks = [{"path": str(raw_video), "role": "speech"}]
                if bgm_path:
                    tracks.append({"path": bgm_path, "role": "music", "volume": 0.2})

                mix_result = mixer.execute({
                    "operation": "full_mix" if bgm_path else "mix",
                    "tracks": tracks,
                    "ducking": {"enabled": True, "music_volume_during_speech": 0.15} if bgm_path else None,
                    "normalize": True,
                    "loudnorm_target": -14,  # YouTube/TikTok standard
                    "output_path": str(audio_out),
                })

                if mix_result.success:
                    outputs.append("audio_processed.mp4")
                    audio_success = True
                    print(f"  ✓ Audio processing complete")
                    step_status["audio_pipeline"] = "done"

                    # AudioMixer outputs audio-only (-vn); remux processed audio
                    # back into the original video stream so downstream steps
                    # (visual_post, rhythm_editor, final_encode) still have video.
                    remux_tmp = output_dir / "audio_remux_tmp.mp4"
                    remux_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(raw_video),       # original video with video stream
                        "-i", str(audio_out),        # processed audio-only file
                        "-map", "0:v",              # take video from original
                        "-map", "1:a",              # take audio from processed
                        "-c:v", "copy",             # don't re-encode video
                        "-c:a", "aac",
                        "-af", "apad", "-shortest",
                        str(remux_tmp),
                    ]
                    import subprocess as _sp
                    try:
                        _sp.run(remux_cmd, capture_output=True, check=True)
                        import shutil
                        shutil.move(str(remux_tmp), str(audio_out))
                        print(f"  ✓ Audio remuxed into video stream")
                    except Exception as remux_err:
                        print(f"  ⚠ Audio remux failed: {remux_err}, using original video")
                        import shutil
                        if remux_tmp.exists():
                            remux_tmp.unlink()
                        shutil.copy2(str(raw_video), str(audio_out))
                else:
                    print(f"  ⚠ AudioMixer failed: {mix_result.error}")
                    step_status["audio_pipeline"] = "failed"
                    # Fallback: just copy video
                    import shutil
                    shutil.copy2(str(raw_video), audio_out)
            except ImportError as e:
                print(f"  ⚠ AudioMixer unavailable: {e}")
                step_status["audio_pipeline"] = "failed"
                # Fallback: just copy video
                import shutil
                shutil.copy2(str(raw_video), audio_out)

            # ── Ambient fallback: if AudioMixer path produced a silent track ──
            # AudioMixer may not be available or may fail, leaving audio_out as
            # a copy of the silent raw_video.  Detect and inject generated ambience.
            try:
                from tools.audio_pipeline import is_silent_audio, generate_ambient_audio
                if audio_out.exists() and is_silent_audio(str(audio_out)):
                    print("  → [ambient-fallback] AudioMixer output still silent, generating ambient audio...")
                    from phases.phase8.edit_decisions import probe_video
                    vid_info = probe_video(str(raw_video))
                    ambient_dur = vid_info.get("duration", 12.0)
                    # Pick scene hint from storyboard if available
                    scene_hint = "generic"
                    if storyboard_data:
                        scene_desc = str(storyboard_data.get("metadata", {}).get("scene", "")).lower()
                        if "forest" in scene_desc or "林" in scene_desc:
                            scene_hint = "forest"
                        elif "city" in scene_desc or "城" in scene_desc:
                            scene_hint = "city"
                    ambient_tmp = output_dir / ".ambient_fallback.m4a"
                    if generate_ambient_audio(ambient_dur, str(ambient_tmp), scene_hint=scene_hint, target_db=-10.0):
                        # Mix ambient audio into the video
                        ambient_out = output_dir / ".ambient_remux.mp4"
                        import subprocess as _sp
                        mix_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(audio_out),
                            "-i", str(ambient_tmp),
                            "-filter_complex",
                            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                            "-map", "0:v", "-map", "[aout]",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-shortest",
                            str(ambient_out),
                        ]
                        try:
                            _sp.run(mix_cmd, capture_output=True, check=True, timeout=60)
                            import shutil
                            shutil.move(str(ambient_out), str(audio_out))
                            print(f"  ✓ [ambient-fallback] Ambient audio mixed in ({scene_hint}, {ambient_dur:.1f}s)")
                            step_status["audio_pipeline"] = "done"
                            if "audio_processed.mp4" not in outputs:
                                outputs.append("audio_processed.mp4")
                        except Exception as mix_err:
                            print(f"  ⚠ [ambient-fallback] Mix failed: {mix_err}")
                        finally:
                            if ambient_tmp.exists():
                                ambient_tmp.unlink()
                            if ambient_out.exists():
                                ambient_out.unlink()
                    else:
                        print("  ⚠ [ambient-fallback] Ambient generation failed")
            except ImportError:
                pass  # audio_pipeline not available, skip fallback

        audio_out = str(audio_out)

        # Subtitle timing must reflect the audible final mix, including TTS
        # overlays and generated speech that was absent from storyboard fields.
        if sb_path_str:
            final_mix_wav = output_dir / "asr_transcripts" / "final_mix.wav"
            final_mix_wav.parent.mkdir(parents=True, exist_ok=True)
            extract_audio_track(str(audio_out), str(final_mix_wav))
            final_mix_transcription = transcribe_audio(str(final_mix_wav))
            final_mix_receipt = {
                "audio_path": str(final_mix_wav),
                "transcription": final_mix_transcription,
            }
            (final_mix_wav.parent / "final_mix.json").write_text(
                json.dumps(final_mix_receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            final_mix_captions = _caption_segments_from_final_asr(final_mix_transcription)
            transcript_data["shot_caption_segments"] = transcript_data["caption_segments"]
            transcript_data["caption_segments"] = final_mix_captions
            transcript_data["final_mix_transcription"] = final_mix_transcription
            transcript_data["asr_summary"]["final_mix_caption_segments"] = len(
                final_mix_captions
            )
            (output_dir / "transcript.json").write_text(
                json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"  ✓ final_mix_asr: 最终混音识别出 {len(final_mix_captions)} 条对白字幕"
            )

        # Step 9.2: visual_post
        print("  → visual_post: 视觉后期...")
        visual_out = str(output_dir / "visual_processed.mp4")
        process_visual(
            video_path=audio_out,
            output_path=visual_out,
            enable_outro=False,
        )
        outputs.append("visual_processed.mp4")

        current_video = visual_out

        # Step 9.2.1: Subtitle burn (optional, from OM RemotionCaptionBurn)
        if sb_path_str:
            print("  → subtitle_burn: 字幕烧录 (RemotionCaptionBurn)...")
            subtitled_out = str(output_dir / "subtitled.mp4")
            try:
                from vendor.video_tools.tools.video.remotion_caption_burn import RemotionCaptionBurn
                caption_burner = RemotionCaptionBurn()

                segments = transcript_data["caption_segments"] if transcript_data else []

                # Also generate SRT as fallback
                srt_path = str(output_dir / "subtitles.srt")
                _write_srt(segments, srt_path)

                if segments:
                    burn_result = caption_burner.execute({
                        "input_path": str(current_video),
                        "output_path": str(subtitled_out),
                        "segments": segments,
                        "srt_path": srt_path,
                        "font_size": 48,
                        "font_color": "#FFFFFF",
                        "outline_color": "#000000",
                        "outline_width": 3,
                        "margin_bottom": 60,
                        "fade_in_ms": 180,
                        "fade_out_ms": 220,
                        "force_ffmpeg": True,
                    })

                    if burn_result.success:
                        current_video = subtitled_out
                        outputs.append("subtitled.mp4")
                        outputs.append("subtitles.srt")
                        print(f"    ✓ 字幕烧录完成: {len(segments)} 条字幕")
                        step_status["subtitle_burn"] = "done"
                    else:
                        print(f"    ⚠ 字幕烧录失败: {burn_result.error}")
                        step_status["subtitle_burn"] = "failed"
                else:
                    print(f"    ⊘ No subtitle data available, skipping subtitle burn")
                    step_status["subtitle_burn"] = "not_required"
            except ImportError as e:
                print(f"    ⚠ RemotionCaptionBurn unavailable: {e}")
                step_status["subtitle_burn"] = "failed"
            except Exception as e:
                print(f"    ⚠ 字幕烧录异常: {e}")
                step_status["subtitle_burn"] = "failed"

        # Step 9.2.5: Color grade (optional, from OM ColorGrade)
        if color_grade:
            print(f"  → color_grade: 应用调色 ({color_grade})...")
            graded_out = str(output_dir / "color_graded.mp4")
            try:
                from vendor.video_tools.tools.enhancement.color_grade import ColorGrade
                grader = ColorGrade()
                grade_result = grader.execute({
                    "input_path": str(current_video),
                    "output_path": str(graded_out),
                    "profile": color_grade,
                    "intensity": 1.0,
                })
                if grade_result.success:
                    current_video = graded_out
                    outputs.append("color_graded.mp4")
                    print(f"    ✓ 调色完成: {color_grade}")
                else:
                    print(f"    ⚠ 调色失败: {grade_result.error}")
            except ImportError as e:
                print(f"    ⚠ ColorGrade unavailable: {e}")

        # Step 9.2.6: Upscale (optional, from OM Upscale — lanczos)
        if upscale:
            print(f"  → upscale: 超分到 {upscale}p (lanczos)...")
            upscaled_out = str(output_dir / "upscaled.mp4")
            try:
                from vendor.video_tools.tools.enhancement.upscale import Upscale
                upscaler = Upscale()
                upscale_result = upscaler.execute({
                    "input_path": str(current_video),
                    "output_path": str(upscaled_out),
                    "target_height": upscale,
                })
                if upscale_result.success:
                    current_video = upscaled_out
                    outputs.append("upscaled.mp4")
                    print(f"    ✓ 超分完成: {upscale}p")
                else:
                    print(f"    ⚠ 超分失败: {upscale_result.error}")
            except ImportError as e:
                print(f"    ⚠ Upscale unavailable: {e}")

        # Step 9.3: rhythm_editor → polished.mp4
        print("  → rhythm_editor: 节奏编辑...")
        final_out = str(output_dir / "polished.mp4")
        # A failed rerun must not inherit a receipt from an older artifact.
        (output_dir / "delivery_timeline.json").unlink(missing_ok=True)
        try:
            edit_rhythm(
                video_path=current_video,
                storyboard_path=sb_path_str,
                timeline_path=str(output_dir / "edit_timeline.json"),
                output_path=final_out,
            )
            outputs.append("polished.mp4")
            step_status["rhythm_editor"] = "done"
        except Exception as e:
            print(f"  ⚠ rhythm_editor failed: {e}")
            step_status["rhythm_editor"] = "failed"
            # Fallback: just copy video
            import shutil
            shutil.copy2(current_video, final_out)
            outputs.append("polished.mp4")

        # Step 9.4: Final encoding with media profile
        print(f"  → final_encode: 使用 {media_profile} 配置重新编码...")
        final_encoded = str(output_dir / "polished_final.mp4")
        profile = _get_profile_dict(media_profile)
        encode_input_durations = _probe_av_durations(Path(final_out))
        delivery_contract = _validated_reviewed_delivery_contract(
            output_dir,
            Path(final_out),
            encode_input_durations,
            fps=float(profile["fps"]),
        )

        video_filters, audio_filters = _final_encode_filters(profile)

        cmd = [
            "ffmpeg", "-y",
            "-i", final_out,
            "-vf", video_filters,
            "-af", audio_filters,
            "-c:v", profile["codec"],
            "-crf", str(profile["crf"]),
            "-preset", "medium",
            "-c:a", profile["audio_codec"],
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", profile["pixel_format"],
            final_encoded,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            step_status["final_encode"] = "failed"
            raise RuntimeError(f"Final encoding failed: {result.stderr[-1000:]}")

        encoded_durations = _probe_av_durations(Path(final_encoded))
        final_duration_gate = _final_encode_duration_gate(
            encode_input_durations,
            encoded_durations,
            delivery_contract=delivery_contract,
            requested_duration=target_duration,
            fps=float(profile["fps"]),
        )
        duration_tolerance_s = final_duration_gate["tolerance_s"]["video"]
        audio_duration_tolerance_s = final_duration_gate["tolerance_s"]["audio"]
        _assert_duration_conserved(
            encode_input_durations,
            encoded_durations,
            tolerance_s=duration_tolerance_s,
            audio_tolerance_s=audio_duration_tolerance_s,
        )
        if not final_duration_gate["passed"]:
            raise RuntimeError(
                "Final duration gate rejected the encoded candidate before promotion: "
                f"{final_duration_gate}"
            )

        # Only promote the encoded artifact after its independent A/V duration
        # assertions pass.  The delivery gate deliberately probes polished.mp4.
        import shutil
        shutil.move(final_encoded, final_out)
        polished_durations = _probe_av_durations(Path(final_out))
        final_duration_gate = _final_encode_duration_gate(
            encode_input_durations,
            polished_durations,
            delivery_contract=delivery_contract,
            requested_duration=target_duration,
            fps=float(profile["fps"]),
        )
        (output_dir / "final_duration_gate.json").write_text(
            json.dumps(final_duration_gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _assert_duration_conserved(
            encode_input_durations,
            polished_durations,
            tolerance_s=duration_tolerance_s,
            audio_tolerance_s=audio_duration_tolerance_s,
        )
        if not final_duration_gate["passed"]:
            raise RuntimeError(
                "Final duration gate failed to conserve the reviewed encode input: "
                f"expected={encode_input_durations}, actual={polished_durations}"
            )
        outputs.extend([
            f"polished.mp4 (encoded with {media_profile})",
            "final_duration_gate.json",
        ])
        print(f"    ✓ 最终编码完成: {profile['width']}x{profile['height']} @ {profile['fps']}fps")
        step_status["final_encode"] = "done"
        step_status["final_duration_gate"] = "done"

        # Final character-animation QA runs against the delivered video and
        # persists its complete structured result for later inspection.
        character_qa_result = None
        try:
            from quality.character_qa import CharacterAnimationQA

            character_video = output_dir / "polished.mp4"
            characters_json = output_dir / "CHARACTERS.json"
            qa_started = time.time()
            print(
                f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"CharacterAnimationQA.check: video={character_video}, "
                f"characters={characters_json}"
            )
            character_tool_result = CharacterAnimationQA().execute({
                "operation": "full_qa",
                "video_path": str(character_video),
                "characters_json_path": str(characters_json),
                "output_dir": str(output_dir / "character_qa_samples"),
            })
            if character_tool_result.success:
                qa_data = character_tool_result.data or {}
                verdict = qa_data.get("verdict", "unknown")
                grade = {"pass": "A", "revise": "C", "fail": "D"}.get(verdict, "N/A")
                character_qa_result = {
                    "status": "success",
                    "grade": grade,
                    "duration_seconds": character_tool_result.duration_seconds,
                    **qa_data,
                }
                (output_dir / "character_qa_report.json").write_text(
                    json.dumps(character_qa_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                outputs.append("character_qa_report.json")
                step_status["character_qa"] = "done" if verdict == "pass" else "failed"
                print(
                    f"  ✓ Character QA 完成: elapsed={time.time() - qa_started:.1f}s, "
                    f"grade={grade}, verdict={verdict}, issues={len(qa_data.get('issues', []))}"
                )
                print(f"    CharacterAnimationQA result: {json.dumps(qa_data, ensure_ascii=False, default=str)}")
            else:
                character_qa_result = {"status": "failed", "error": character_tool_result.error}
                step_status["character_qa"] = "failed"
                print(
                    f"  ⚠ Character QA 失败: elapsed={time.time() - qa_started:.1f}s, "
                    f"error={character_tool_result.error}"
                )
        except Exception as e:
            character_qa_result = {"status": "skipped", "reason": str(e)}
            step_status["character_qa"] = "skipped"
            print(f"  ⚠ Character QA 不可用: {e}")

        print(f"  ✓ Phase 9 完成: polished.mp4")
        
        # Quality gate: Phase 9
        qg_report = run_quality_check("phase9", output_dir, step_status=step_status)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 9 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
        
        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "color_grade": color_grade,
            "upscale": upscale,
            "audio_enhanced": audio_success,
            "bgm_detected": bgm_path is not None,
            "media_profile": media_profile,
            "step_status": step_status,
            "character_qa": character_qa_result,
        }
    except ImportError as e:
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


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
