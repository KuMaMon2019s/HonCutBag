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
import os
import random
import subprocess
import sys
import time
import traceback
import re
from pathlib import Path
from typing import Optional, TypedDict, Any

from utils.config import get_api_key
from utils.progress_reporter import ProgressReporter
from quality.quality_gate import run_quality_check
from utils.timing_estimator import estimate_phase_duration, estimate_total, estimate_remaining
from quality.slideshow_risk import score_slideshow_risk
from quality.variation_checker import check_scene_variation
from quality.delivery_promise import classify_from_brief
from prompt.speech_pacing import annotate_shot_pacing
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.checkpoint import write_checkpoint as write_stage_checkpoint
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


STYLE_SUMMARY_WALL_TIMEOUT = 180.0
STYLE_SUMMARY_IDLE_TIMEOUT = 75.0


def _run_storyboard_supervision(storyboard: dict, output_dir: Path) -> dict:
    """Run the optional independent review at the Phase 5/6 boundary."""
    from quality.supervision_agent import run_supervision
    from utils.pipeline_config import load_config

    config = load_config()
    style_path = output_dir / "visual-style.md"
    visual_style = (
        style_path.read_text(encoding="utf-8")
        if style_path.is_file()
        else str(storyboard.get("style", ""))
    )
    return run_supervision(storyboard, visual_style, output_dir, config)

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
    from langgraph.types import RetryPolicy, Send, Command, interrupt
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.base import empty_checkpoint
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
    interrupt = None
    GraphInterrupt = None

# ---------------------------------------------------------------------------
# Path setup — 让本脚本能 import 同目录及 2026-07-27_05/scripts 下的模块
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_SRC_DIR = SCRIPT_DIR
LEGACY_TOOLS_DIR = SCRIPT_DIR.parent.parent / "vendor" / "legacy"
OM_TOOLS_DIR = SCRIPT_DIR.parent.parent / "vendor" / "video_tools"

# 优先加载当前源码目录，然后是兼容工具目录
for d in (PIPELINE_SRC_DIR, LEGACY_TOOLS_DIR, str(OM_TOOLS_DIR)):
    s = str(d)
    if s not in sys.path:
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
             "audio_codec": "aac", "crf": 23, "pixel_format": "yuv420p"},
    "720p": {"name": "720p", "width": 1280, "height": 720, "fps": 30, "codec": "libx264",
             "audio_codec": "aac", "crf": 23, "pixel_format": "yuv420p"},
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
            "codec": "libx264", "audio_codec": "aac", "crf": 23, "pixel_format": "yuv420p"}


def _probe_av_durations(path: Path) -> dict[str, float | None]:
    """Return independent video/audio stream durations; probing failures block delivery."""
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    durations: dict[str, float | None] = {"video": None, "audio": None}
    for stream in streams:
        kind = stream.get("codec_type")
        if kind in durations and durations[kind] is None and stream.get("duration") not in (None, "N/A"):
            durations[kind] = float(stream["duration"])
    if durations["video"] is None:
        raise RuntimeError(f"No measurable video stream duration: {path}")
    return durations


def _assert_duration_conserved(before: dict[str, float | None], after: dict[str, float | None], tolerance_s: float = 1.0) -> None:
    """Assert final encoding conserved video and audio durations independently."""
    for kind in ("video", "audio"):
        expected, actual = before.get(kind), after.get(kind)
        if expected is None:
            continue
        if actual is None or abs(actual - expected) > tolerance_s:
            raise RuntimeError(
                f"Final {kind} duration changed from {expected:.3f}s to "
                f"{actual if actual is not None else 'missing'} (tolerance ±{tolerance_s:.1f}s)"
            )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _elapsed(start: float) -> float:
    return round(_now() - start, 2)


def _banner(phase_num, total: int, name: str, dry_run: bool = False):
    tag = " [DRY-RUN]" if dry_run else ""
    print(f"\n{'='*60}")
    print(f"  [Phase {phase_num}/{total}] {name}{tag}")
    print(f"{'='*60}")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Checkpoint — HonCut 断点续跑
# ---------------------------------------------------------------------------
# 格式: {
#   "completed": ["phase1", "phase2", ...],
#   "results": {"phase1": {...}, ...},
#   "timestamp": "2026-07-28T..."
# }
# ---------------------------------------------------------------------------

# Phase 顺序定义（用于 resume 时判断哪些已完成）
PHASE_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9", "phase9_5"]


def _checkpoint_path(output_dir: Path) -> Path:
    """返回 checkpoint.json 路径"""
    return Path(output_dir) / "checkpoint.json"


def _record_stage_checkpoint(output_dir: Path, phase_name: str, result: dict) -> Path:
    """Persist a successful phase through the shared checkpoint module."""
    cp_path = _checkpoint_path(output_dir)
    status = result.get("status", "")
    if status not in ("done", "skipped"):
        return cp_path
    safe_result = {}
    for k, v in result.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v, default=str)
            safe_result[k] = v
        except (TypeError, ValueError):
            safe_result[k] = str(v)
    checkpoint = write_stage_checkpoint(cp_path, phase_name, safe_result)

    if LANGGRAPH_AVAILABLE:
        try:
            save_state_to_sqlite(checkpoint, output_dir, thread_id="pipeline_run")
        except Exception as e:
            print(f"  ⚠ SQLite checkpoint 写入失败: {e}")

    return cp_path


def _read_checkpoint(output_dir: Path) -> Optional[dict]:
    """读取检查点。返回 None 如果不存在或损坏。"""
    cp_path = _checkpoint_path(output_dir)
    cp_path = Path(cp_path)
    if not cp_path.exists():
        return None
    try:
        with open(cp_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        # 基本校验
        if not isinstance(checkpoint.get("completed"), list):
            return None
        return checkpoint
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# LangGraph Phase 1 Integration: @task + RetryPolicy, Send fan-out, SqliteSaver
# ---------------------------------------------------------------------------

# --- Retry helper (works outside StateGraph context) ---
def _retry_with_policy(func, max_attempts=3, backoff_factor=2.0, *args, **kwargs):
    """Execute func with retry logic matching LangGraph's RetryPolicy semantics.
    
    This is a standalone retry helper that works outside StateGraph context.
    When running inside a StateGraph, the @task decorator provides equivalent
    retry behavior automatically.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_text = str(e)
            if attempt < max_attempts:
                is_429 = (
                    "429" in error_text
                    or "Too Many Requests" in error_text
                    or "QuotaExceeded" in error_text
                    or getattr(getattr(e, "response", None), "status_code", None) == 429
                )
                if is_429:
                    base_wait = [120, 240, 480][attempt - 1]
                    wait_time = base_wait + random.uniform(0, 30)
                else:
                    wait_time = backoff_factor ** (attempt - 1)
                print(
                    f"    ⚠ Attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                print(
                    f"    ✗ All {max_attempts} attempts failed. Last error: {e}",
                    flush=True,
                )
    raise last_error


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
                "description": char.get("appearance", {}).get("summary", char.get("description", "")),
                "appearance": char.get("appearance", {}),  # 传递完整 appearance dict
                "style": char.get("style", ""),
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

    # SqliteSaver checkpoint integration
    _sqlite_saver_instance = None
    _sqlite_saver_path = None

    def get_sqlite_checkpointer(output_dir: Path):
        """Create SqliteSaver checkpointer for the pipeline (module-level singleton).
        
        Returns the SqliteSaver context manager itself. Use it as:
            saver = get_sqlite_checkpointer(output_dir)
            with saver as checkpointer:
                checkpointer.setup()
                # use checkpointer...
        """
        global _sqlite_saver_instance, _sqlite_saver_path
        db_path = Path(output_dir) / "checkpoint.db"
        db_path_str = str(db_path)
        if _sqlite_saver_instance is not None and _sqlite_saver_path == db_path_str:
            return _sqlite_saver_instance
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _sqlite_saver_instance = SqliteSaver.from_conn_string(db_path_str)
            _sqlite_saver_path = db_path_str
            return _sqlite_saver_instance
        except Exception as e:
            print(f"⚠ SqliteSaver initialization failed: {e}")
            return None

    def save_state_to_sqlite(state: dict, output_dir: Path, thread_id: str = "default") -> bool:
        """Save the stage checkpoint as LangGraph channel values in SQLite."""
        try:
            db_path = Path(output_dir) / "checkpoint.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = state
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
                checkpointer.put(
                    config,
                    checkpoint,
                    {"source": "update", "step": len(state.get("completed", [])), "writes": state},
                    {},
                )
            return True
        except Exception as e:
            print(f"⚠ Failed to save state to SQLite: {e}")
            return False

    def load_state_from_sqlite(output_dir: Path, thread_id: str = "default") -> Optional[dict]:
        """Load pipeline state from SQLite checkpoint."""
        try:
            db_path = Path(output_dir) / "checkpoint.db"
            if not db_path.exists():
                return None
            with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
                config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                checkpoint = checkpointer.get_tuple(config)
            
                if checkpoint and checkpoint.checkpoint:
                    return checkpoint.checkpoint.get("channel_values", {})
            return None
        except Exception as e:
            print(f"⚠ Failed to load state from SQLite: {e}")
            return None

else:
    # Fallback when LangGraph is not available
    def get_sqlite_checkpointer(output_dir: Path):
        return None
    
    def save_state_to_sqlite(state: dict, output_dir: Path, thread_id: str = "default") -> bool:
        return False
    
    def load_state_from_sqlite(output_dir: Path, thread_id: str = "default") -> Optional[dict]:
        return None


def _get_completed_stages(output_dir: Path) -> list:
    """获取已完成的阶段列表。"""
    cp = _read_checkpoint(output_dir)
    if cp is None:
        return []
    return cp.get("completed", [])


def _get_next_stage(output_dir: Path, all_phases: list = None) -> Optional[str]:
    """获取下一个要执行的阶段。

    返回第一个不在 completed 列表中的 phase，或 None（全部完成）。
    """
    if all_phases is None:
        all_phases = PHASE_ORDER
    completed = set(_get_completed_stages(output_dir))
    for phase in all_phases:
        if phase not in completed:
            return phase
    return None


# ---------------------------------------------------------------------------
# Phase 1: 导演规划 (M1 增量模块)
# ---------------------------------------------------------------------------

def run_phase1_director(text: str, output_dir: Path, dry_run: bool) -> dict:
    """Phase 1: 导演规划（M1 增量模块）"""
    _banner("1", 9, "导演规划 (Director Planner)", dry_run)
    start = _now()
    try:
        from phases.phase1.director_planner import plan_director
        result = plan_director(text, output_dir, dry_run)
        # Lock the intended production medium before providers can downgrade it.
        delivery_promise = classify_from_brief("cinematic", {}).to_dict()
        result.setdefault("delivery_promise", delivery_promise)
        plan = result.get("plan")
        if isinstance(plan, dict):
            plan.setdefault("delivery_promise", delivery_promise)
            scenes = plan.get("scenes", [])
            pacing_inputs = []
            for scene in scenes:
                dialogue = scene.get("dialogue") or scene.get("lines")
                if not dialogue and scene.get("dialogue_words"):
                    dialogue = "字" * int(scene["dialogue_words"])
                pacing_inputs.append({
                    "dialogue": dialogue or "",
                    "emotion": scene.get("emotion_arc", ""),
                })
            pacing = annotate_shot_pacing(pacing_inputs)
            for scene, annotation in zip(scenes, pacing):
                scene["speech_pacing"] = {
                    "duration_s": annotation["speech_duration_s"],
                    "emotion": annotation["emotion"],
                }
            plan_path = Path(result.get("output", output_dir / "director_plan.json"))
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        result["duration_s"] = _elapsed(start)
        return result
    except Exception as e:
        print(f"  ⚠ [M1] Phase 1 降级跳过: {e}")
        return {"status": "skipped", "reason": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 1: 编剧引擎 (text → STORYBOARD.json + CHARACTERS.json)
# ---------------------------------------------------------------------------

def _integrate_storyboard_prompts(storyboard: dict, characters: list[dict]) -> dict:
    """Normalize generated prompts through the shared Phase 1 contracts."""
    assets = [
        {"id": character.get("id", index), "name": character.get("name", ""), "type": "角色"}
        for index, character in enumerate(characters, 1)
        if character.get("name")
    ]
    for shot in storyboard.get("shots", []):
        visual = shot.get("prompt") or shot.get("visual") or shot.get("what") or "scene"
        lighting = shot.get("lighting_key") or "natural cinematic lighting"
        style = storyboard.get("style") or "cinematic"
        prompt = build_three_part_prompt(str(visual), str(lighting), str(style))
        referenced = shot.get("associate_assets") or shot.get("who") or []
        referenced_names = {str(value) for value in referenced}
        shot_assets = [
            asset for asset in assets
            if str(asset["id"]) in referenced_names or asset["name"] in referenced_names
        ]
        shot["prompt"] = sanitize_quality_prompt(bind_assets(prompt, shot_assets))
    return storyboard


def _extract_visual_style_text(script_text: str) -> Optional[str]:
    """Extract a declared art-style paragraph without interpreting the script."""
    match = re.search(
        r"(?im)^\s*(?:美术风格|Art\s+style)\s*[：:]\s*(.+(?:\n(?!\s*(?:角色设定|剧情|人物设定|Characters?|Plot|Story)\s*[：:]).+)*)",
        script_text,
    )
    if not match:
        return None
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return " ".join(lines).strip() or None


def _summarize_visual_style_with_llm(script_text: str) -> Optional[str]:
    """Best-effort style summary; deliberately isolated so tests can mock it."""
    api_key = os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ 风格总结不可用，降级使用 default_visual_style")
        return None
    try:
        return call_llm_stream(
            messages=[{
                "role": "user",
                "content": "用一句话总结以下剧本的美术风格，只输出风格描述：\n" + script_text,
            }],
            max_tokens=1024,
            wall_timeout=STYLE_SUMMARY_WALL_TIMEOUT,
            read_timeout=STYLE_SUMMARY_IDLE_TIMEOUT,
            idle_timeout=STYLE_SUMMARY_IDLE_TIMEOUT,
        ).strip() or None
    except Exception as exc:
        print(f"  ⚠ 风格总结失败，降级使用 default_visual_style: {exc}")
        return None


PHASE1_CHECKPOINT_SCHEMA_VERSION = 2


def _phase1_input_hash(items: list) -> str:
    serialized = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_write_phase1_json(
    path: Path,
    payload: dict,
    *,
    collection_key: str,
    input_hash: str,
) -> None:
    """Persist a completed Phase 1 substage without exposing partial JSON."""
    stored = dict(payload)
    stored["_checkpoint"] = {
        "schema_version": PHASE1_CHECKPOINT_SCHEMA_VERSION,
        "collection_key": collection_key,
        "input_hash": input_hash,
        "item_count": len(stored.get(collection_key, [])),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _load_phase1_checkpoint(
    path: Path,
    collection_key: str,
    *,
    input_hash: str,
) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get(collection_key), list):
        return None
    metadata = payload.get("_checkpoint")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("schema_version") != PHASE1_CHECKPOINT_SCHEMA_VERSION:
        return None
    if metadata.get("collection_key") != collection_key:
        return None
    if metadata.get("input_hash") != input_hash:
        return None
    if metadata.get("item_count") != len(payload[collection_key]):
        return None
    return payload


def _write_project_visual_style(output_dir: Path, style_text: str) -> Path:
    """Write the minimal frontmatter accepted by parse_visual_style."""
    import yaml
    payload = {
        "name": "Script-derived project style",
        "version": "1.0",
        "style_prompt_short": style_text,
        "style_prompt_full": style_text,
    }
    style_path = Path(output_dir) / "visual-style.md"
    style_path.write_text(
        "---\n" + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000) + "---\n",
        encoding="utf-8",
    )
    return style_path

def run_phase1_screenwriter(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
) -> dict:
    """Phase 1: text_parser → event_extractor → character_discoverer → adaptation_engine → storyboard_generator"""
    _banner(1, 9, "编剧引擎 (Screenwriter)", dry_run)
    start = _now()
    _p2_est = estimate_phase_duration("phase1")
    print(f"  ⏱ Phase 1 开始 (预估 ~{int(_p2_est)}s)")
    output_dir = Path(output_dir)

    try:
        from prompt.text_parser import parse_text
    except ImportError as e:
        return {"status": "error", "error": f"Phase 1 import failed: {e}", "duration_s": _elapsed(start)}

    outputs = []
    if reporter:
        reporter.start_heartbeat("phase1")
        configure_heartbeat_callback(
            lambda: reporter.step(
                "phase1", "LLM 流式响应", progress_pct=reporter._progress_pct
            )
        )
    try:
        # Step 2.1: text_parser → segments list
        print("  → text_parser: 解析文本结构...")
        if reporter:
            reporter.step("phase1", "解析文本结构", progress_pct=10)
        parsed = parse_text(text)
        segments = parsed.get("segments", [])
        print(f"    ✓ 解析出 {len(segments)} 个段落")
        if reporter:
            reporter.step("phase1", f"解析出 {len(segments)} 个段落", progress_pct=20)

        # dry-run 模式：生成模拟数据，不调用 API
        if dry_run:
            print("  ⊘ dry-run 模式，生成模拟数据（跳过 API 调用）...")
            if reporter:
                reporter.step("phase1", "dry-run: 生成模拟事件", progress_pct=30)
            
            # 模拟事件数据
            mock_events = [
                {
                    "id": 1,
                    "who": ["主角"],
                    "where": "场景A",
                    "what": "发现关键线索",
                    "emotion": "紧张",
                    "visual": "主角在昏暗的房间中发现了一张神秘的地图",
                    "time": "夜晚",
                    "action_type": "discovery"
                },
                {
                    "id": 2,
                    "who": ["主角", "配角"],
                    "where": "场景B",
                    "what": "展开行动",
                    "emotion": "激动",
                    "visual": "两人在阳光下讨论计划，充满希望",
                    "time": "白天",
                    "action_type": "action"
                },
                {
                    "id": 3,
                    "who": ["主角"],
                    "where": "场景C",
                    "what": "面临挑战",
                    "emotion": "坚定",
                    "visual": "主角独自站在山顶，眺望远方",
                    "time": "黄昏",
                    "action_type": "resolution"
                }
            ]
            
            if reporter:
                reporter.step("phase1", f"dry-run: 提取 {len(mock_events)} 个事件", progress_pct=45)
            
            # 模拟角色数据
            mock_characters = {
                "characters": [
                    {
                        "id": "protagonist",
                        "name": "主角",
                        "aliases": ["他", "主人公"],
                        "role": "protagonist",
                        "appearance": {
                            "gender": "male",
                            "age_range": "25-35",
                            "height": "中等身高",
                            "build": "athletic",
                            "hair": "黑色短发",
                            "face": "坚毅的面容",
                            "clothing": "休闲装",
                            "distinguishing": "无明显特征",
                            "summary": "25-35岁男性，黑色短发，身材健壮，面容坚毅"
                        },
                        "personality": {
                            "traits": ["勇敢", "坚定", "善良"],
                            "speech_style": "简洁有力",
                            "motivation": "寻找真相"
                        },
                        "style": "写实风格, 35mm film, 自然光",
                        "negative": "卡通, 3D渲染, 过度饱和",
                        "size": "1920x1920",
                        "first_appearance": 1,
                        "appearance_count": 3
                    },
                    {
                        "id": "supporting",
                        "name": "配角",
                        "aliases": ["朋友"],
                        "role": "supporting",
                        "appearance": {
                            "gender": "female",
                            "age_range": "20-30",
                            "height": "中等身高",
                            "build": "slim",
                            "hair": "棕色长发",
                            "face": "温和的面容",
                            "clothing": "职业装",
                            "distinguishing": "戴眼镜",
                            "summary": "20-30岁女性，棕色长发，身材纤细，戴眼镜，面容温和"
                        },
                        "personality": {
                            "traits": ["聪明", "细心", "支持"],
                            "speech_style": "理性分析",
                            "motivation": "帮助主角"
                        },
                        "style": "写实风格, 35mm film, 自然光",
                        "negative": "卡通, 3D渲染, 过度饱和",
                        "size": "1920x1920",
                        "first_appearance": 2,
                        "appearance_count": 1
                    }
                ]
            }
            
            if reporter:
                reporter.step("phase1", f"dry-run: 发现 {len(mock_characters['characters'])} 个角色", progress_pct=60)
            
            # 模拟分镜数据
            mock_storyboard = {
                "shots": [
                    {
                        "id": 1,
                        "prompt": "A young man discovers a mysterious map in a dimly lit room, cinematic lighting, 35mm film, natural light, tense atmosphere",
                        "caption": "发现神秘地图",
                        "duration": 5,
                        "aspect_ratio": "16:9",
                        "scene": "昏暗的房间",
                        "action": "发现地图",
                        "camera": "中景",
                        "emotion": "紧张"
                    },
                    {
                        "id": 2,
                        "prompt": "Two people discussing plans under bright sunlight, hopeful atmosphere, cinematic composition, natural lighting",
                        "caption": "讨论计划",
                        "duration": 5,
                        "aspect_ratio": "16:9",
                        "scene": "阳光明媚的户外",
                        "action": "讨论计划",
                        "camera": "双人镜头",
                        "emotion": "激动"
                    },
                    {
                        "id": 3,
                        "prompt": "A determined man standing alone on a mountain top at sunset, looking into the distance, epic cinematic shot, golden hour lighting",
                        "caption": "眺望远方",
                        "duration": 5,
                        "aspect_ratio": "16:9",
                        "scene": "山顶",
                        "action": "眺望",
                        "camera": "远景",
                        "emotion": "坚定"
                    }
                ],
                "total_duration": 15,
                "style": "写实电影风格"
            }
            _integrate_storyboard_prompts(mock_storyboard, mock_characters["characters"])
            
            if reporter:
                reporter.step("phase1", f"dry-run: 生成 {len(mock_storyboard['shots'])} 个分镜", progress_pct=80)
            
            # 写出文件
            storyboard_path = output_dir / "STORYBOARD.json"
            characters_path = output_dir / "CHARACTERS.json"
            events_path = output_dir / "events.json"
            
            storyboard_path.write_text(json.dumps(mock_storyboard, ensure_ascii=False, indent=2))
            characters_path.write_text(json.dumps(mock_characters, ensure_ascii=False, indent=2))
            events_path.write_text(json.dumps({"events": mock_events}, ensure_ascii=False, indent=2))
            
            outputs = ["STORYBOARD.json", "CHARACTERS.json", "events.json"]
            print(f"  ✓ Phase 1 完成 (dry-run): {outputs}")
            
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "_storyboard": mock_storyboard,
                "_characters": mock_characters,
            }
        
        # 正常模式：调用 API
        try:
            from prompt.event_extractor import extract_events
            from phases.phase1.character_discoverer import (
                CHARACTER_CONTEXT_SCHEMA_VERSION,
                _is_human_character,
                discover_characters,
            )
            from phases.phase1.adaptation_engine import adapt_events
            from phases.phase1.storyboard_generator import generate_storyboard
        except ImportError as e:
            return {"status": "error", "error": f"Phase 1 import failed: {e}", "duration_s": _elapsed(start)}

        # Step 2.2: event_extractor → events list
        print("  → event_extractor: 提取事件...")
        if reporter:
            reporter.step("phase1", "提取事件", progress_pct=30)
        events_checkpoint = output_dir / "phase1_events.json"
        nonempty_segments = [
            segment for segment in segments if str(segment.get("content", "")).strip()
        ]
        events_input_hash = _phase1_input_hash(nonempty_segments)
        expected_segment_ids = [segment.get("id", 0) for segment in nonempty_segments]
        events_result = _load_phase1_checkpoint(
            events_checkpoint,
            "events",
            input_hash=events_input_hash,
        )
        if events_result is not None:
            complete_coverage = (
                events_result.get("source_segments_hash") == events_input_hash
                and events_result.get("source_segment_count") == len(nonempty_segments)
                and events_result.get("covered_segment_ids") == expected_segment_ids
                and events_result.get("total_events") == len(events_result["events"])
            )
            if not complete_coverage:
                events_result = None
        if events_result is not None:
            print("    ↻ 复用 phase1_events.json，跳过事件提取")
        else:
            events_result = dict(extract_events(segments))
            events_result.setdefault("schema_version", "2.0")
            events_result.setdefault("source_segments_hash", events_input_hash)
            events_result.setdefault("source_segment_count", len(nonempty_segments))
            events_result.setdefault("covered_segment_ids", expected_segment_ids)
            events_result.setdefault("total_events", len(events_result.get("events", [])))
            _atomic_write_phase1_json(
                events_checkpoint,
                events_result,
                collection_key="events",
                input_hash=events_input_hash,
            )
        events = events_result.get("events", [])
        print(f"    ✓ 提取 {len(events)} 个事件")
        if reporter:
            reporter.step("phase1", f"提取 {len(events)} 个事件", progress_pct=40)

        # Step 2.3: character_discoverer → characters dict
        print("  → character_discoverer: 发现角色...")
        if reporter:
            reporter.step("phase1", "发现角色", progress_pct=50)
        characters_checkpoint = output_dir / "phase1_characters.json"
        characters_input_hash = _phase1_input_hash([
            {
                "character_context_schema": CHARACTER_CONTEXT_SCHEMA_VERSION,
                "events": events,
            }
        ])
        characters_result = _load_phase1_checkpoint(
            characters_checkpoint,
            "characters",
            input_hash=characters_input_hash,
        )
        if characters_result is not None:
            characters = characters_result["characters"]
            valid_characters = (
                characters_result.get("source_text_hash") == characters_input_hash
                and characters_result.get("total_characters") == len(characters)
                and all(
                    isinstance(character, dict)
                    and bool(str(character.get("name", "")).strip())
                    and _is_human_character(str(character.get("name", "")).strip())
                    for character in characters
                )
            )
            if not valid_characters:
                characters_result = None
        if characters_result is not None:
            print("    ↻ 复用 phase1_characters.json，跳过角色发现")
        else:
            characters_result = dict(discover_characters(events))
            characters_result["source_text_hash"] = characters_input_hash
            characters_result.setdefault(
                "total_characters", len(characters_result.get("characters", []))
            )
            _atomic_write_phase1_json(
                characters_checkpoint,
                characters_result,
                collection_key="characters",
                input_hash=characters_input_hash,
            )
        characters_list = characters_result.get("characters", [])
        print(f"    ✓ 发现 {len(characters_list)} 个角色")
        if reporter:
            reporter.step("phase1", f"发现 {len(characters_list)} 个角色", progress_pct=60)

        # Step 2.4: adaptation_engine → adapted shots list
        print("  → adaptation_engine: 影视化改编...")
        if reporter:
            reporter.step("phase1", "影视化改编", progress_pct=70)
        adapted = adapt_events(
            events,
            characters_list,
            target_duration=duration,
            shot_duration=shot_duration,
            source_text=text,
            output_dir=output_dir,
        )
        adapted_shots = adapted.get("shots", [])
        print(f"    ✓ 改编完成，{len(adapted_shots)} 个镜头")
        if reporter:
            reporter.step("phase1", f"改编完成，{len(adapted_shots)} 个镜头", progress_pct=80)

        # Phase 1 storyboard_generator step → storyboard dict
        style_source = "剧本提取"
        visual_style_text = _extract_visual_style_text(text)
        if not visual_style_text:
            style_source = "LLM 总结"
            visual_style_text = _summarize_visual_style_with_llm(text)
        visual_style_path = None
        if visual_style_text:
            visual_style_path = _write_project_visual_style(output_dir, visual_style_text)
            print(f"  ✓ 项目风格: visual-style.md（{style_source}）")
        print("  → storyboard_generator: 生成分镜...")
        if reporter:
            reporter.step("phase1", "生成分镜", progress_pct=90)
        storyboard = generate_storyboard(
            adapted_shots,
            characters_list,
            visual_style_path=str(visual_style_path) if visual_style_path else None,
            visual_style_text=visual_style_text,
        )
        _integrate_storyboard_prompts(storyboard, characters_list)
        annotate_shot_pacing(storyboard.get("shots", []))

        # 写出文件
        storyboard_path = output_dir / "STORYBOARD.json"
        characters_path = output_dir / "CHARACTERS.json"

        storyboard_path.write_text(json.dumps(storyboard, ensure_ascii=False, indent=2))
        characters_path.write_text(json.dumps(characters_result, ensure_ascii=False, indent=2))

        outputs = ["STORYBOARD.json", "CHARACTERS.json"]
        print(f"  ✓ Phase 1 完成: {outputs}")

        # Quality gate: Phase 1
        qg_report = run_quality_check("phase1", output_dir, {
            "events": storyboard.get("events", []),
            "shots": storyboard.get("shots", []),
        })
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 1 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start), "outputs": outputs}

        # --- M5: 监督层审核（增量，失败不影响后续）---
        try:
            from quality.quality_gate import run_storyboard_review
            review = run_storyboard_review(
                storyboard_data=storyboard,
                script_text=text,
                characters=characters_result.get("characters", []),
            )
            result_data = {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "_storyboard": storyboard,
                "_characters": characters_result,
                "storyboard_review": review,
            }
            if review.get("grade") == "D":
                print(f"  ⚠ [M5] 分镜审核 D 级，建议重做（但不阻断管线）")
            return result_data
        except Exception as e:
            print(f"  ⚠ [M5] 分镜审核跳过: {e}")

        phase1_result = {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "_storyboard": storyboard,
            "_characters": characters_result,
        }
        return phase1_result

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start), "outputs": outputs}
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()


def run_phase1(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
) -> dict:
    """Phase 1: director planning followed by the screenwriter engine."""
    started = _now()
    if reporter:
        # LangGraph enters the combined runner directly, so establish visible
        # progress before the director's first network request.  Sequential
        # execution may already have called phase_start; avoid a duplicate.
        if (
            getattr(reporter, "_current_phase", None) != "phase1"
            and hasattr(reporter, "phase_start")
        ):
            reporter.phase_start("phase1", "导演拆解 + 编剧引擎")
        reporter.step("phase1", "导演规划", progress_pct=1)
        reporter.start_heartbeat("phase1")
        configure_heartbeat_callback(
            lambda: reporter.step(
                "phase1",
                "导演规划 LLM 流式响应",
                progress_pct=getattr(reporter, "_progress_pct", 1),
            )
        )
    try:
        director = run_phase1_director(text, Path(output_dir), dry_run)
        screenwriter = run_phase1_screenwriter(
            text,
            output_dir,
            duration,
            dry_run,
            reporter=reporter,
            shot_duration=shot_duration,
        )
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()
    combined = dict(screenwriter)
    combined["director"] = director
    combined["duration_s"] = _elapsed(started)
    return combined


# ---------------------------------------------------------------------------
# Phase 2: 故事板图片生成 (OM image_selector)
# ---------------------------------------------------------------------------

def load_storyboard_prompt_techniques() -> str:
    """加载 HonCut 分镜提示词技巧。

    返回精简版提示词技巧文本，追加到分镜生成 prompt 中，
    增强镜头语言、构图规则和画质控制。
    """
    techniques_path = SCRIPT_DIR.parent / "prompts" / "storyboard_prompt_techniques.md"
    if not techniques_path.exists():
        return ""
    try:
        return techniques_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def fill_storyboard_template(template: str, storyboard_data: dict, characters_data: dict) -> str:
    """填充故事板提示词模板

    从 STORYBOARD.json 提取镜头描述，从 CHARACTERS.json 提取角色描述，
    替换模板中的占位符。
    集成 HonCut 分镜提示词技巧（镜头语言、构图规则、画质控制）。

    注意：只提取代码块内的提示词部分，忽略文档说明。
    """
    # 提取代码块内的提示词（忽略文档说明）
    import re
    code_block_match = re.search(r'```\n(.*?)```', template, re.DOTALL)
    if code_block_match:
        prompt_template = code_block_match.group(1).strip()
    else:
        # 如果没有代码块，使用整个模板
        prompt_template = template.strip()

    # 提取镜头描述
    shots = storyboard_data.get("shots", [])
    storyboard_lines = []
    for i, shot in enumerate(shots, 1):
        scene = (
            shot.get("visual")
            or shot.get("scene")
            or shot.get("description")
            or shot.get("prompt")
            or ""
        )
        action = shot.get("action_description") or shot.get("action") or shot.get("what") or ""
        camera = (
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or shot.get("shot_type")
            or ""
        )
        scene = str(scene).strip()
        action = str(action).strip()
        camera = str(camera).strip()
        if not scene and not action:
            raise ValueError(f"storyboard shot {i} has no visual content")
        line = f"面板 {i}: {scene or action}"
        if action:
            line += f" — 动作: {action}"
        if camera:
            line += f" — 镜头: {camera}"
        storyboard_lines.append(line)
    storyboard_content = "\n".join(storyboard_lines) if storyboard_lines else "无分镜内容"

    # 提取角色描述
    characters = characters_data.get("characters", [])
    char_lines = []
    for c in characters:
        name = c.get("name", "未知角色")
        appearance = c.get("appearance", {})
        summary = appearance.get("summary", c.get("description", ""))
        if summary:
            char_lines.append(f"- {name}: {summary}")
    character_reference = "\n".join(char_lines) if char_lines else "无角色描述"

    # 面板数量
    panel_count = len(shots) if shots else 12

    # 风格（默认值）
    style = "粗铅笔线条，细节最少，快速手势绘画能量"

    # 替换占位符
    prompt = prompt_template.replace("{{STORYBOARD_CONTENT}}", storyboard_content)
    prompt = prompt.replace("{{CHARACTER_REFERENCE}}", character_reference)
    prompt = prompt.replace("{{PANEL_COUNT}}", str(panel_count))
    prompt = prompt.replace("{{STYLE}}", style)

    # 追加 HonCut 分镜提示词技巧参考
    techniques = load_storyboard_prompt_techniques()
    if techniques:
        prompt += "\n\n---\n# 分镜提示词技巧参考\n\n"
        # 提取核心段落（跳过 YAML frontmatter 和主标题）
        tech_lines = techniques.split("\n")
        core_lines = []
        in_core = False
        for line in tech_lines:
            if line.startswith("## 核心原则"):
                in_core = True
            if in_core:
                core_lines.append(line)
        # 限制长度避免 prompt 过长（取前60行核心内容）
        prompt += "\n".join(core_lines[:60])

    return prompt


def _normalize_shot_id(shot_item: dict) -> Optional[str]:
    """Return a zero-padded shot ID, or ``None`` when no usable ID exists."""
    raw = (
        shot_item.get("shot_id")
        or shot_item.get("id")
        or shot_item.get("shot_order", 0)
    )
    if not raw:
        return None
    if isinstance(raw, int):
        return f"S{raw:02d}"

    raw_str = str(raw)
    # Legacy storyboards already store IDs such as ``S01``.
    if raw_str.upper().startswith("S"):
        raw_str = raw_str[1:]
    return f"S{raw_str.zfill(2)}"


def _shot_storyboard_reference(output_dir: Path, shot_id: Any) -> Optional[Path]:
    """Return a single-shot reference; never return the overview contact sheet."""
    normalized = _normalize_shot_id({"id": shot_id})
    if normalized is None:
        return None
    path = Path(output_dir) / "storyboard_images" / f"{normalized}.png"
    if path.is_file() and path.stat().st_size > 1024:
        return path
    return None


# Action verb → end-state mapping (for FLF2V end frames)
_ACTION_END_STATES = {
    # Chinese
    "抬手": "hand raised to its highest point, arm extended",
    "抬手拂发": "hand lowered after brushing hair aside, hair now clear of the face",
    "走来": "has arrived at the destination, standing steadily",
    "坐下": "seated steadily on the chair, posture relaxed",
    "转身": "has completed the turn, now facing the new direction",
    "拥抱": "arms wrapped around each other in a warm embrace",
    "牵手": "hands clasped together, fingers interlocked",
    "回头": "head turned to look back over the shoulder",
    "起身": "standing upright, fully risen from the seated position",
    "挥手": "hand raised in a waving gesture, arm extended",
    # English (base forms)
    "raise hand": "hand raised to its highest point, arm extended",
    "walk over": "has arrived at the destination, standing steadily",
    "sit down": "seated steadily on the chair, posture relaxed",
    "turn around": "has completed the turn, now facing the new direction",
    "embrace": "arms wrapped around each other in a warm embrace",
    "hold hands": "hands clasped together, fingers interlocked",
    "look back": "head turned to look back over the shoulder",
    "stand up": "standing upright, fully risen from the seated position",
    "wave": "hand raised in a waving gesture, arm extended",
    # English (conjugated variants — S05 "raises her hand to brush away hair")
    "raises her hand": "hand lowered back to resting position, hair now clear of the face",
    "raises his hand": "hand lowered back to resting position",
    "brush away": "hand lowered after brushing, action completed",
    "brushes away": "hand lowered after brushing, action completed",
    "walks over": "has arrived at the destination, standing steadily",
    "walks toward": "has arrived at the destination, standing steadily",
    "sits down": "seated steadily on the chair, posture relaxed",
    "stands up": "standing upright, fully risen from the seated position",
    "turns around": "has completed the turn, now facing the new direction",
    "embraces": "arms wrapped around each other in a warm embrace",
    "hugs": "arms wrapped around each other in a warm embrace",
    "holds hands": "hands clasped together, fingers interlocked",
    "runs toward": "has arrived at the destination, standing steadily",
    "looks back": "head turned to look back over the shoulder",
    "waves": "hand raised in a waving gesture, arm extended",
}


def _derive_end_state(shot: dict) -> str:
    """Derive explicit end-state description from action verbs.
    
    Returns a concrete end-state sentence for known action verbs,
    or a generic fallback for unknown actions.
    """
    action_text = " ".join(
        str(shot.get(field, "")) for field in ("visual", "what", "description")
        if shot.get(field)
    ).lower()
    
    # Check for known action verbs (longest match first to avoid partial matches)
    sorted_verbs = sorted(_ACTION_END_STATES.keys(), key=len, reverse=True)
    for verb in sorted_verbs:
        if verb.lower() in action_text:
            return _ACTION_END_STATES[verb]
    
    # Generic fallback
    return "the action described is fully completed, natural resting pose afterwards"


def build_end_frame_prompt(shot: dict) -> str:
    """Build rich t2i prompt for end frame generation.

    M3 fix: switched from i2i (copies reference) to t2i with rich description.
    Includes: end-state pose, scene context, character appearance, style anchors.
    """
    prompt = shot.get("prompt", shot.get("visual", ""))
    end_state = _derive_end_state(shot)

    # Extract character appearance from prompt (simple heuristic)
    char_desc = ""
    if "少女" in prompt or "girl" in prompt.lower():
        char_desc = "young woman in traditional attire"
    elif "少年" in prompt or "boy" in prompt.lower():
        char_desc = "young man"

    return (
        f"{end_state}.\n\n"
        f"Scene: {prompt}.\n\n"
        f"Character: {char_desc if char_desc else 'the character'}.\n\n"
        f"Style: maintain the same artistic style, lighting, and composition as the start frame.\n\n"
        f"The action has just completed; the character is in the final resting position.\n\n"
        f"Background, camera angle, and environment must match the start frame exactly."
    )


def fit_to_aspect(image_path: Path, target_w: int, target_h: int, output_path: Path) -> Path:
    """Resize image to exact target dimensions WITHOUT stretching.

    M5: If image already matches target aspect ratio (within 1% tolerance):
    simple high-quality resize. Otherwise: resize to COVER target aspect,
    then CENTER-CROP to exact dimensions. NEVER stretch/distort.

    Args:
        image_path: Source image path
        target_w: Target width in pixels
        target_h: Target height in pixels
        output_path: Where to save result (PNG)

    Returns:
        output_path (for chaining)
    """
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        src_w, src_h = img.size

        src_aspect = src_w / src_h
        target_aspect = target_w / target_h
        aspect_tolerance = 0.01  # 1% tolerance

        if abs(src_aspect - target_aspect) / target_aspect <= aspect_tolerance:
            # Already matches aspect ratio — simple resize
            resized = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            # Need to cover + center-crop
            scale_w = target_w / src_w
            scale_h = target_h / src_h
            scale = max(scale_w, scale_h)  # COVER = use larger scale

            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            resized_cover = img.resize((new_w, new_h), Image.LANCZOS)

            # Center-crop to exact target dimensions
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            resized = resized_cover.crop((left, top, left + target_w, top + target_h))

        resized.save(str(output_path), 'PNG')

    return output_path


# Seedream API minimum pixel requirement (Agent Plan).
# HTTP 400 if WxH < 3686400. Empirically verified 2026-08-06.
SEEDREAM_MIN_PIXELS = 3686400


def _storyboard_image_size(image_path: Optional[Path] = None, video_width: int = 1280, video_height: int = 720) -> str:
    """Return Seedream's WxH size string for storyboard/end-frame generation.

    M7 fix: ensure returned size meets Seedream's minimum pixel requirement
    (SEEDREAM_MIN_PIXELS = 3686400). The old 1920x1080 (2,073,600 px) was
    rejected with HTTP 400 InvalidParameter.

    Formula: for aspect a=w/h, width = ceil(sqrt(min_pixels * a)), then
    round to even. Height = width / a, also rounded to even.

    For 16:9: 2560x1440 = 3,686,400 px (exactly at minimum).
    """
    import math

    aspect = video_width / video_height  # a = w/h

    # Compute smallest width >= sqrt(min_pixels * aspect) at correct aspect
    raw_w = math.sqrt(SEEDREAM_MIN_PIXELS * aspect)
    w = math.ceil(raw_w)
    # Round to even
    w = w if w % 2 == 0 else w + 1

    # Compute height from width and aspect, round to even
    h = math.ceil(w / aspect)
    h = h if h % 2 == 0 else h + 1

    # Safety check: if rounding pushed us below minimum, bump width
    if w * h < SEEDREAM_MIN_PIXELS:
        w += 2  # next even number
        h = math.ceil(w / aspect)
        h = h if h % 2 == 0 else h + 1

    return f"{w}x{h}"


# ── M4: FLF2V end-frame validation thresholds (t2i-adapted) ──────────────
# Tuned after S05 smoke test: t2i generation (M3) produces genuinely different
# images from first frame, so i2i-era thresholds were too strict.
#
# similarity_high: 0.97 allows genuine action progress whose broad grayscale
#   composition remains similar (observed at 0.9324 in the 240s xianxia run),
#   while still catching true copies (0.99+). The old 0.93 threshold rejected
#   visibly different camera staging and hand positions.
# similarity_low: 0.3 unchanged — scene drift detection still needed.
# sharpness_floor_ratio: 0.15 — t2i without reference injection is inherently
#   softer than i2i with character refs.实测 88.5/439.7=0.20 > 0.15 passes.
#   Old value 0.3 rejected visually acceptable t2i output.
FLF2V_SIMILARITY_LOW: float = 0.3
FLF2V_SIMILARITY_HIGH: float = 0.97
FLF2V_SHARPNESS_RATIO: float = 0.15


def _validate_end_frame(
    first_frame_path: Path,
    end_frame_path: Path,
    similarity_low: float = FLF2V_SIMILARITY_LOW,
    similarity_high: float = FLF2V_SIMILARITY_HIGH,
    sharpness_floor_ratio: float = FLF2V_SHARPNESS_RATIO,
    brightness_range: tuple = (15, 240),
) -> dict:
    """Validate end frame against first frame using metric-based checks.
    
    Returns dict with keys: passed, similarity, sharpness_ok, brightness_ok,
    resolution_ok, reason (if failed).
    
    Thresholds (M4, t2i-adapted):
      similarity_low=0.3   — scene drift floor (unchanged from M2)
      similarity_high=0.97 — catch true copies (0.99+), allow changed staging that
                             retains the same subject and luminous background
      sharpness_floor_ratio=0.15 — t2i is softer than i2i; 0.20× ratio is acceptable
    
    No VLM required — deterministic metric checks:
    1. Resolution identical to first frame
    2. Non-black, non-blank (mean brightness in range)
    3. Sharpness: Laplacian variance above floor (first_frame_variance × ratio)
    4. Similarity: perceptual distance must be in band [low, high]
       - Too similar (> high) → copy of first frame (no action progress)
       - Too different (< low) → scene/camera drifted
    """
    from PIL import Image
    import numpy as np
    
    result = {
        "passed": False,
        "similarity": None,
        "sharpness_ok": False,
        "brightness_ok": False,
        "resolution_ok": False,
        "reason": None,
        "thresholds": {
            "similarity_low": similarity_low,
            "similarity_high": similarity_high,
            "sharpness_ratio": sharpness_floor_ratio,
        },
    }
    
    try:
        first_img = Image.open(first_frame_path).convert("RGB")
        end_img = Image.open(end_frame_path).convert("RGB")
    except Exception as e:
        result["reason"] = f"cannot open images: {e}"
        return result
    
    # 1. Resolution normalization (M8): if sizes differ, normalize first frame
    #    to end frame dimensions via fit_to_aspect (COVER + center-crop, no stretch).
    #    This handles legacy square first frames (1920×1920) vs new 16:9 end frames (2560×1440).
    end_w, end_h = end_img.size
    if first_img.size != end_img.size:
        import tempfile
        tmp_first = Path(tempfile.mktemp(suffix=".png"))
        try:
            fit_to_aspect(first_frame_path, end_w, end_h, tmp_first)
            first_img = Image.open(tmp_first).convert("RGB")
        finally:
            tmp_first.unlink(missing_ok=True)
    result["resolution_ok"] = True
    
    # Convert to numpy arrays for metric computation
    first_arr = np.array(first_img)
    end_arr = np.array(end_img)
    
    # 2. Brightness check (non-black, non-blank)
    mean_brightness = float(np.mean(end_arr))
    result["brightness_ok"] = brightness_range[0] <= mean_brightness <= brightness_range[1]
    if not result["brightness_ok"]:
        result["reason"] = f"brightness out of range: {mean_brightness:.1f} (expected {brightness_range})"
        return result
    
    # 3. Sharpness check (Laplacian variance)
    def _laplacian_variance(arr):
        """Compute Laplacian variance as sharpness proxy."""
        gray = np.mean(arr, axis=2)  # RGB → grayscale
        # Simple 3×3 Laplacian kernel via array slicing
        lap = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        return float(np.var(lap))
    
    first_sharpness = _laplacian_variance(first_arr)
    end_sharpness = _laplacian_variance(end_arr)
    sharpness_floor = first_sharpness * sharpness_floor_ratio
    result["sharpness_ok"] = end_sharpness >= sharpness_floor
    if not result["sharpness_ok"]:
        result["reason"] = (
            f"too blurry: sharpness={end_sharpness:.1f} < floor={sharpness_floor:.1f} "
            f"(first_frame={first_sharpness:.1f} × {sharpness_floor_ratio})"
        )
        return result
    
    # 4. Similarity check (downsampled grayscale MSE → normalized similarity)
    def _downsample_gray(arr, target_size=64):
        """Downsample to small grayscale image for perceptual comparison."""
        gray = np.mean(arr, axis=2)
        # Simple box downsample
        h, w = gray.shape
        step_h = max(1, h // target_size)
        step_w = max(1, w // target_size)
        small = gray[::step_h, ::step_w][:target_size, :target_size]
        return small.astype(np.float64)
    
    first_small = _downsample_gray(first_arr)
    end_small = _downsample_gray(end_arr)
    
    # Pad to same shape if needed
    min_h = min(first_small.shape[0], end_small.shape[0])
    min_w = min(first_small.shape[1], end_small.shape[1])
    first_small = first_small[:min_h, :min_w]
    end_small = end_small[:min_h, :min_w]
    
    # MSE → similarity (1.0 = identical, 0.0 = maximally different)
    mse = float(np.mean((first_small - end_small) ** 2))
    # Normalize: max possible MSE for 8-bit images is 255^2 = 65025
    similarity = max(0.0, 1.0 - mse / 65025.0)
    result["similarity"] = round(similarity, 4)
    
    if similarity < similarity_low:
        result["reason"] = (
            f"too different (scene drift): similarity={similarity:.4f} < {similarity_low}"
        )
        return result
    
    if similarity > similarity_high:
        result["reason"] = (
            f"too similar (no action progress): similarity={similarity:.4f} > {similarity_high}"
        )
        return result
    
    result["passed"] = True
    return result


def _end_frame_sidecar_path(end_frame_path: Path) -> Path:
    """Return the sidecar meta JSON path for an end frame."""
    return end_frame_path.with_name(end_frame_path.stem + "_end.meta.json")


def _write_end_frame_sidecar(
    end_frame_path: Path,
    first_frame_sha: str,
    prompt_sha: str,
    validation: dict,
):
    """Write cache sidecar for end frame."""
    sidecar = _end_frame_sidecar_path(end_frame_path)
    sidecar.write_text(json.dumps({
        "first_frame_sha256": first_frame_sha,
        "prompt_sha256": prompt_sha,
        "validation": validation,
    }, indent=2))


def _read_end_frame_sidecar(end_frame_path: Path) -> Optional[dict]:
    """Read cache sidecar, or None if missing/invalid."""
    sidecar = _end_frame_sidecar_path(end_frame_path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_flf2v_end_frame(
    shot_item: dict,
    shot_id: str,
    first_frame_path: Path,
    ref_image_path: Optional[Path],
) -> bool:
    """Generate one idempotent Seedream end frame for an FLF2V shot.

    M3 fix: prefer text_to_image (NOT image_to_image) to avoid near-identical copies.
    t2i generates a fresh image from the rich prompt — end-state pose + scene + style.
    The first frame is still used for VALIDATION (similarity band check) only.
    Falls back to i2i if t2i fails.

    Cache: uses sidecar meta JSON with first_frame_sha + prompt_sha + validation.
    Validation: metric-based checks (resolution, brightness, sharpness, similarity band).
    """
    if shot_item.get("gen_strategy", "i2v") != "flf2v":
        return False

    # First frame MUST exist — used for validation and size reference
    if not first_frame_path.exists():
        raise FileNotFoundError(
            f"[FLF2V] {shot_id}: first frame {first_frame_path} not found. "
            "Cannot generate end frame without the start frame as reference."
        )

    end_path = first_frame_path.with_name(f"{shot_id}_end.png")
    prompt = build_end_frame_prompt(shot_item)
    first_frame_sha = _file_sha256(first_frame_path)
    import hashlib as _hashlib
    prompt_sha = _hashlib.sha256(prompt.encode()).hexdigest()

    # Check cache sidecar
    sidecar = _read_end_frame_sidecar(end_path)
    if (
        sidecar is not None
        and end_path.exists()
        and sidecar.get("first_frame_sha256") == first_frame_sha
        and sidecar.get("prompt_sha256") == prompt_sha
        and sidecar.get("validation", {}).get("passed")
    ):
        print(f"    ⏭ [FLF2V] {end_path.name} cached+validated, skipping")
        return False

    # Generate end frame — M3: prefer t2i over i2i
    from clients.seedream_client import SeedreamClient
    client = SeedreamClient()
    # M5: use video target aspect ratio (16:9), not first frame's dimensions
    video_w = shot_item.get("width", 1280)
    video_h = shot_item.get("height", 720)
    size = _storyboard_image_size(video_width=video_w, video_height=video_h)

    # Primary: text_to_image (no reference → no copy problem)
    try:
        client.text_to_image(
            prompt=prompt,
            output_path=str(end_path),
            size=size,
        )
        print(f"    [FLF2V] 终帧 {end_path.name} ✓ (t2i)")
    except Exception as e:
        # Fallback: i2i with first frame as reference
        print(f"    ⚠ [FLF2V] t2i failed ({e}), falling back to i2i")
        client.image_to_image(
            prompt=prompt,
            ref_image=str(first_frame_path),
            output_path=str(end_path),
            size=size,
        )
        print(f"    [FLF2V] 终帧 {end_path.name} ✓ (i2i fallback)")
    
    # Validate the generated end frame
    validation = _validate_end_frame(first_frame_path, end_path)
    _write_end_frame_sidecar(end_path, first_frame_sha, prompt_sha, validation)
    
    if not validation["passed"]:
        reason = validation.get("reason", "unknown")
        print(f"    ⚠ [FLF2V] {end_path.name} validation FAILED: {reason}")
        # Retry once with same approach (t2i)
        print(f"    [FLF2V] retrying {end_path.name}...")
        client.text_to_image(
            prompt=prompt,
            output_path=str(end_path),
            size=size,
        )
        validation = _validate_end_frame(first_frame_path, end_path)
        _write_end_frame_sidecar(end_path, first_frame_sha, prompt_sha, validation)
        
        if not validation["passed"]:
            reason = validation.get("reason", "unknown")
            print(f"    ✗ [FLF2V] {end_path.name} retry FAILED: {reason}")
            raise RuntimeError(
                f"[FLF2V] {shot_id}: end frame validation failed after retry: {reason}"
            )
    
    sim = validation.get("similarity", "N/A")
    print(f"    ✓ [FLF2V] {end_path.name} validated (similarity={sim})")
    return True


def _storyboard_keyframe_description(shot: dict) -> str:
    """Build an identity-locked, single-moment prompt for one storyboard frame."""
    who_declared = "who" in shot
    who = shot.get("who") or []
    identity = str(shot.get("subject_description") or "").strip()
    action = str(
        shot.get("action_description") or shot.get("what") or ""
    ).strip()
    staging = str(shot.get("visual") or "").strip()
    # Only explicit who=[] is an environment contract. Legacy storyboards may
    # omit ``who`` while carrying character identity and action in the older
    # subject/action fields.
    if who_declared and not who:
        parts = [
            "Environment-only cinematic keyframe.",
            "Depict exclusively the described clouds, landscape, architecture, light, and atmosphere.",
            f"Exact environment contract: {action}." if action else "",
            f"Visual staging: {staging}." if staging else "",
            "The frame is uninhabited: zero people, zero humanoid figures, and zero unrelated objects.",
        ]
    else:
        parts = [
            "Single decisive cinematic keyframe.",
            (
                "Character identity lock (gender, hair, face, clothing, and body proportions "
                f"must remain exact): {identity}."
                if identity
                else ""
            ),
            f"Exact action contract: {action}." if action else "",
            "Show one decisive final pose of that exact action contract.",
            f"Visual staging: {staging}." if staging else "",
            "Depict only subjects, props, and actions explicitly named in the contract.",
            "No exposed midriff unless the identity contract explicitly requires it.",
        ]
    return " ".join(part for part in parts if part)


def _generate_shot_images(
    output_dir: Path,
    storyboard_data: dict,
    regenerate_shot_ids: set[str] | None = None,
) -> int:
    """Generate storyboard images for each shot (M2 task).
    
    Args:
        output_dir: Project output directory
        storyboard_data: Storyboard data with shots list
        
    Returns:
        Number of successfully generated images
    """
    try:
        storyboard_images_dir = output_dir / "storyboard_images"
        storyboard_images_dir.mkdir(exist_ok=True)
        shots = storyboard_data.get("shots", [])
        prompt_scenes = []
        for shot in shots:
            prompt_scene = dict(shot)
            prompt_scene["description"] = _storyboard_keyframe_description(shot)
            prompt_scene.setdefault("shot_language", {
                "shot_size": shot.get("shot_size"),
                "camera_movement": shot.get("camera_movement"),
                "lighting_key": shot.get("lighting_key"),
            })
            prompt_scenes.append(prompt_scene)
        # Each shot's visual/action contract already contains its intended style.
        # A project-level LLM style summary may mention plot objects (palace,
        # character, props); injecting it into every shot leaks future content.
        batch_prompts = build_batch_prompts(
            prompt_scenes,
            None,
        )
        prompt_by_id = {str(item["scene_id"]): item["prompt"] for item in batch_prompts}

        # --- P0-1a: Load character reference images (for shot image consistency) ---
        char_ref_map = {}  # {char_name_lower: preferred_reference_path}
        protagonist_ref = None
        chars_path = output_dir / "CHARACTERS.json"
        if chars_path.exists():
            try:
                chars_data = json.loads(chars_path.read_text())
                for char in chars_data.get("characters", []):
                    reference_path = None
                    for char_dir in (
                        output_dir / "characters" / char["id"],
                        output_dir / "characters" / "characters" / char["id"],
                    ):
                        candidates = [
                            char_dir / "face_closeup.png",
                            char_dir / "full_body.png",
                            *sorted(char_dir.glob("variant_*.png")),
                            char_dir / "front.png",  # legacy fallback
                        ]
                        reference_path = next(
                            (path for path in candidates if path.exists()), None
                        )
                        if reference_path is not None:
                            break
                    if reference_path is not None:
                        char_ref_map[char["name"].lower()] = reference_path
                        char_ref_map[char["id"].lower()] = reference_path
                        if protagonist_ref is None:
                            protagonist_ref = reference_path
                if char_ref_map:
                    print(f"  → [P0-1] 已加载 {len(char_ref_map)//2} 个角色参考图")
            except Exception as e:
                print(f"  ⚠ [P0-1] 角色参考图加载失败: {e}")

        generated_count = 0

        # Phase 2 runs before the character factory. Character shots must wait
        # until Phase 3 has produced references instead of silently using t2i.
        has_character_shots = any(shot.get("who") for shot in shots)
        if has_character_shots and not char_ref_map:
            print(
                "  → [M2] 角色参考图尚未生成；逐镜分镜图延后到 Phase 3",
                flush=True,
            )
            return 0
        
        # --- P2-5d: HonCut concurrent shot image generation ---
        def _gen_shot_image(shot_item):
            """Single shot image generation logic (for concurrent calls)"""
            shot_id = _normalize_shot_id(shot_item)
            if shot_id is None:
                print("    ⚠ [M2] 分镜缺少有效 shot_id/id/shot_order，跳过")
                return None
            shot_prompt = prompt_by_id.get(str(shot_item.get("id", ""))) or shot_item.get("prompt", shot_item.get("visual", ""))
            if not shot_prompt:
                return None
            shot_image_path = storyboard_images_dir / f"{shot_id}.png"

            # --- P0-1c: Match character reference image ---
            # Support structured who[] from storyboard:
            # - Empty who [] → pure landscape/no_character → NO reference injection
            # - Single character → use that character's preferred reference
            # - Multiple characters → use first character's preferred reference
            ref_image_paths = []
            shot_who = shot_item.get("who", [])
            if not isinstance(shot_who, list):
                shot_who = [shot_who] if shot_who else []

            if len(shot_who) == 0:
                # Pure landscape / no_character shot — do NOT inject any character reference
                print(f"    [M2] {shot_id}: 纯风景镜头(who=[]), 不注入角色参考")
            else:
                # Match first available character reference
                for name in shot_who:
                    reference = char_ref_map.get(str(name).lower())
                    if reference is not None and reference not in ref_image_paths:
                        ref_image_paths.append(reference)
                # Fallback to protagonist only if who[] is non-empty but no match found
                if not ref_image_paths and protagonist_ref:
                    ref_image_paths.append(protagonist_ref)

            ref_image_path = ref_image_paths[0] if ref_image_paths else None

            stale_for_references = bool(
                shot_image_path.exists()
                and ref_image_paths
                and any(
                    reference.stat().st_mtime > shot_image_path.stat().st_mtime
                    for reference in ref_image_paths
                )
            )
            force_regenerate = bool(
                regenerate_shot_ids and shot_id in regenerate_shot_ids
            )
            if (
                shot_image_path.exists()
                and not stale_for_references
                and not force_regenerate
            ):
                _generate_flf2v_end_frame(
                    shot_item, shot_id, shot_image_path, ref_image_path
                )
                return shot_id
            if stale_for_references:
                print(
                    f"    [M2] {shot_id}: 角色参考图较新，刷新旧分镜图",
                    flush=True,
                )
            elif force_regenerate:
                print(f"    [M2] {shot_id}: 按质检清单定向重绘", flush=True)

            # --- 429 retry with exponential backoff ---
            import time as _time
            _m2_max_retries = 3
            _m2_wait_times = [120, 240, 480]
            for _m2_attempt in range(1, _m2_max_retries + 1):
                try:
                    # Preserve 16:9 while meeting Seedream's minimum pixel count.
                    _m2_size = _storyboard_image_size(video_width=1920, video_height=1080)
                    if ref_image_paths and all(path.exists() for path in ref_image_paths):
                        # P0-1c: Use image_to_image mode (with reference image)
                        from clients.seedream_client import SeedreamClient
                        client = SeedreamClient()
                        client.image_to_image(
                            prompt=shot_prompt,
                            ref_image=[str(path) for path in ref_image_paths],
                            output_path=str(shot_image_path),
                            size=_m2_size,
                        )
                        refs = ", ".join(path.name for path in ref_image_paths)
                        print(f"    [M2] 分镜图 {shot_id}.png ✓ (refs: {refs})")
                    else:
                        # No reference image, pure text-to-image
                        from clients.seedream_client import text_to_image
                        text_to_image(prompt=shot_prompt, output_path=str(shot_image_path), size=_m2_size)
                    print(f"    [M2] 分镜图 {shot_id}.png ✓")
                    _generate_flf2v_end_frame(
                        shot_item, shot_id, shot_image_path, ref_image_path
                    )
                    return shot_id
                except Exception as e:
                    _err_str = str(e)
                    _is_429 = (
                        "429" in _err_str
                        or "Too Many Requests" in _err_str
                        or (hasattr(e, "response") and getattr(getattr(e, "response", None), "status_code", None) == 429)
                    )
                    if _is_429 and _m2_attempt < _m2_max_retries:
                        _wait = _m2_wait_times[_m2_attempt - 1]
                        print(f"    [M2] {shot_id} retry {_m2_attempt}/{_m2_max_retries} (429, wait {_wait}s)...")
                        _time.sleep(_wait)
                        continue
                    else:
                        # Non-429 or retries exhausted → raise
                        print(f"    [M2] {shot_id}.png ✗ → {e}")
                        raise
            return None

        # Concurrent execution (max_workers=3)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_gen_shot_image, s): s for s in shots}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        generated_count += 1
                except Exception as e:
                    shot = futures[future]
                    shot_id = _normalize_shot_id(shot) or "<missing>"
                    print(f"    [M2] 分镜图 {shot_id}.png 并发失败（降级跳过）: {e}")
        print(f"  → [M2] 分镜图序列: {generated_count}/{len(shots)} 张")
        return generated_count
    except Exception as e:
        print(f"  ⚠ [M2] 分镜图序列生成失败（降级跳过）: {e}")
        return 0


def _validate_storyboard_image_composition(output_dir: Path, storyboard_data: dict) -> dict:
    """Validate that every generated storyboard cut has its required image asset."""
    cuts = []
    cursor = 0.0
    for shot in storyboard_data.get("shots", []):
        shot_id = _normalize_shot_id(shot)
        if shot_id is None:
            continue
        duration = float(shot.get("duration", 5))
        cuts.append({
            "id": shot_id,
            "source": f"storyboard_images/{shot_id}.png",
            "in_seconds": cursor,
            "out_seconds": cursor + duration,
        })
        cursor += duration
    report = validate_composition({"cuts": cuts}, output_dir)
    (output_dir / "storyboard_composition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_phase2(storyboard_data: dict, characters_data: dict, output_dir: Path, dry_run: bool) -> dict:
    """Phase 2: 使用 OM image_selector 生成故事板图片，不可用时降级到 Seedream API"""
    _banner("2", 9, "故事板图片生成 (ImageSelector / Seedream)", dry_run)
    start = _now()
    _p25_est = estimate_phase_duration("phase2")
    print(f"  ⏱ Phase 2 开始 (预估 ~{int(_p25_est)}s)")
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过故事板图片生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    print("[cooldown] 等待 120s 让 Agent Plan 限流窗口重置...", flush=True)
    time.sleep(120)

    # 1. 加载模板
    template_path = SCRIPT_DIR.parent / "prompts" / "storyboard_template.md"
    if not template_path.exists():
        return {"status": "error", "error": f"storyboard_template.md not found at {template_path}", "duration_s": _elapsed(start)}

    template = template_path.read_text(encoding="utf-8")

    # 2. 填充模板
    prompt = fill_storyboard_template(template, storyboard_data, characters_data)
    (output_dir / "storyboard_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"  → 提示词已生成 ({len(prompt)} 字符)")

    storyboard_path = output_dir / "storyboard.png"
    om_error = None

    # 3. 尝试调用 OM image_selector
    try:
        from vendor.video_tools.tools.graphics.image_selector import ImageSelector
        selector = ImageSelector()

        print(f"  → image_selector: 生成故事板图片...")

        result = selector.execute({
            "prompt": prompt,
            "width": 1920,
            "height": 1080,
            "aspect_ratio": "16:9",
            "output_path": str(storyboard_path),
        })

        if result.success:
            # 从 result.data 中提取输出路径
            out_path = result.data.get("output_path") or result.data.get("image_path")
            if out_path and Path(out_path).exists():
                # 如果输出不在目标位置，复制过去
                if Path(out_path) != storyboard_path:
                    import shutil
                    shutil.copy2(out_path, storyboard_path)
                print(f"  ✓ Phase 2 完成: storyboard.png (provider: OM)")
                
                # Quality gate: Phase 2
                qg_report = run_quality_check("phase2", output_dir)
                if not qg_report.passed:
                    return {"status": "error", "error": f"Phase 2 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
                
                # --- M2: 分镜图序列（每镜头一张）---
                generated = _generate_shot_images(output_dir, storyboard_data)
                if generated:
                    composition_report = _validate_storyboard_image_composition(output_dir, storyboard_data)
                    if not composition_report["valid"]:
                        return {"status": "error", "error": "Storyboard composition validation failed", "composition_report": composition_report, "duration_s": _elapsed(start)}

                # --- P0-A: HonCut 场景参考图生成 ---
                try:
                    scenes_dir = output_dir / "scenes"
                    scenes_dir.mkdir(exist_ok=True)
                    from phases.phase4.scene_consistency import (
                        _load_style as _load_scene_visual_style,
                        build_scene_reference_prompt,
                    )
                    scene_visual_style = _load_scene_visual_style(
                        output_dir / "visual-style.md"
                    )
                    # 提取所有唯一场景
                    unique_wheres = list(set(
                        shot.get("where", "") for shot in storyboard_data.get("shots", [])
                        if shot.get("where")
                    ))
                    scene_count = 0
                    for where in unique_wheres:
                        scene_id = where.replace(" ", "_").replace("/", "_")[:30]
                        scene_dir = scenes_dir / scene_id
                        scene_dir.mkdir(exist_ok=True)
                        ref_path = scene_dir / "reference.png"
                        if ref_path.exists():
                            scene_count += 1
                            continue
                        try:
                            scene_prompt = build_scene_reference_prompt(
                                where,
                                list(storyboard_data.get("shots", [])),
                                scene_visual_style,
                            )
                            from clients.seedream_client import text_to_image
                            text_to_image(prompt=scene_prompt, output_path=str(ref_path))
                            scene_count += 1
                            print(f"    [P0-A] 场景参考图 {scene_id}/reference.png ✓")
                        except Exception as e:
                            print(f"    [P0-A] 场景参考图 {scene_id} 失败（降级跳过）: {e}")
                    print(f"  → [P0-A] 场景参考图: {scene_count}/{len(unique_wheres)} 个")
                except Exception as e:
                    print(f"  ⚠ [P0-A] 场景参考图生成失败（降级跳过）: {e}")

                return {
                    "status": "done",
                    "duration_s": _elapsed(start),
                    "outputs": ["storyboard.png"],
                    "provider": result.data.get("provider", "unknown"),
                }
            else:
                # result 成功但没有文件路径 — 可能有 URL
                image_url = result.data.get("image_url") or result.data.get("url")
                if image_url:
                    import urllib.request
                    print(f"  → 下载图片: {image_url[:80]}...")
                    urllib.request.urlretrieve(image_url, str(storyboard_path))
                    print(f"  ✓ Phase 2 完成: storyboard.png (provider: OM)")
                    return {"status": "done", "duration_s": _elapsed(start), "outputs": ["storyboard.png"], "provider": "om"}
                else:
                    om_error = "No output file or URL in result"
                    print(f"  ⚠ OM 生成成功但无输出文件/URL: {om_error}")
        else:
            om_error = result.error or "image_selector returned failure"
            print(f"  ⚠ OM image_selector 失败: {om_error}")

    except ImportError as e:
        om_error = f"OM tools unavailable: {e}"
        print(f"  ⚠ OM image_selector 不可用: {e}")
    except Exception as e:
        om_error = str(e)
        print(f"  ⚠ OM image_selector 异常: {e}")

    # 4. 降级到 Seedream API
    print(f"  → 降级到 Seedream API (ARK_AGENT_API_KEY)...")
    try:
        from clients.seedream_client import SeedreamClient
        client = SeedreamClient()
        seedream_size = _storyboard_image_size(video_width=1920, video_height=1080)
        print(f"  → seedream: 生成故事板图片 ({seedream_size}, timeout=180s, retry=3)...")

        # Use retry policy for API call
        def _call_seedream():
            client.text_to_image(
                prompt=prompt,
                output_path=str(storyboard_path),
                size=seedream_size,
                timeout=180,
            )
            if not storyboard_path.exists():
                raise RuntimeError("Seedream 调用成功但未生成文件")
        
        _retry_with_policy(_call_seedream, max_attempts=3, backoff_factor=2.0)

        if storyboard_path.exists():
            print(f"  ✓ Phase 2 完成: storyboard.png (provider: Seedream, fallback from OM: {om_error})")
            
            # Quality gate: Phase 2
            qg_report = run_quality_check("phase2", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 2 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
            # --- M2: 分镜图序列（每镜头一张）---
            generated = _generate_shot_images(output_dir, storyboard_data)
            if generated:
                composition_report = _validate_storyboard_image_composition(output_dir, storyboard_data)
                if not composition_report["valid"]:
                    return {"status": "error", "error": "Storyboard composition validation failed", "composition_report": composition_report, "duration_s": _elapsed(start)}
            
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["storyboard.png"],
                "provider": "seedream",
                "fallback_reason": om_error,
            }
        else:
            print(f"  ✗ Seedream 调用成功但未生成文件")
            return {"status": "error", "error": f"OM failed ({om_error}), Seedream succeeded but no file produced", "duration_s": _elapsed(start)}

    except ImportError as e:
        print(f"  ✗ seedream_client 也不可用: {e}")
        return {"status": "error", "error": f"OM failed ({om_error}), Seedream import failed: {e}", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": f"OM failed ({om_error}), Seedream failed: {e}", "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 3: 角色工厂 (CHARACTERS.json → characters/*/character_card.json)
# ---------------------------------------------------------------------------

def detect_derive_assets(characters_data: dict) -> list:
    """按 HonCut 规范检测衍生资产。

    读取 CHARACTERS.json，检测角色是否有变身/换装描述，
    返回衍生资产列表。

    规则（精简版）:
    - 角色: 仅变身状态（服装/变身特效/变形）
    - 场景: 仅时间变体（日景→夜景）
    - 道具: 不衍生

    Returns:
        list of dict: [{"parent_id": "char_001", "name": "战斗服", "desc": "...", "type": "role"}, ...]
    """
    derive_assets = []
    # 兼容 dict（含 "characters" key）和直接 list 两种输入
    if isinstance(characters_data, list):
        characters_list = characters_data
    else:
        characters_list = characters_data.get("characters", [])

    # 变身/换装关键词（中文 + 英文）
    transformation_keywords = [
        "变身", "换装", "战斗服", "礼服", "盔甲", "兽化", "巨大化", "变形",
        "能量", "光效", "transform", "costume", "armor", "beast", "giant",
        "battle", "ceremony", "formal", "magic", "power"
    ]

    for char in characters_list:
        char_id = char.get("id", "")
        char_name = char.get("name", "")
        description = char.get("description", "")
        appearance = char.get("appearance", {})
        summary = appearance.get("summary", "")
        clothing = appearance.get("clothing", "")

        # 合并所有文本字段进行检测
        all_text = f"{description} {summary} {clothing}".lower()

        # 检测是否包含变身/换装关键词
        detected_keywords = [kw for kw in transformation_keywords if kw.lower() in all_text]

        if detected_keywords:
            # 为每个检测到的变身状态创建衍生资产
            for keyword in detected_keywords[:3]:  # 限制每个角色最多3个衍生
                # 生成衍生资产名称（2-6字）
                derive_name = keyword if len(keyword) <= 6 else keyword[:6]

                # 生成描述
                derive_desc = f"{char_name}的{keyword}形态 · 与默认态有明显视觉差异"

                derive_assets.append({
                    "parent_id": char_id,
                    "parent_name": char_name,
                    "name": derive_name,
                    "desc": derive_desc,
                    "type": "role",
                    "keyword": keyword,
                })

    return derive_assets


def run_phase3(output_dir: Path, characters_data: dict, dry_run: bool) -> dict:
    """Phase 3: character_factory — 生成角色三视图 + 衍生资产检测"""
    _banner(3, 9, "角色工厂 (Character Factory + Derive Assets)", dry_run)
    start = _now()
    outputs = []
    output_dir = Path(output_dir)

    try:
        from phases.phase3.character_factory import batch_generate

        chars_dir = _ensure_dir(output_dir / "characters")
        characters_list = characters_data.get("characters", [])

        if not characters_list:
            print("  ⊘ 无角色数据，跳过")
            return {"status": "skipped", "reason": "no characters", "duration_s": _elapsed(start)}

        # Step 3.1: HonCut 衍生资产检测
        print("  → 检测衍生资产（变身/换装状态）...")
        derive_assets = detect_derive_assets(characters_data)
        if derive_assets:
            print(f"    ✓ 检测到 {len(derive_assets)} 个衍生资产:")
            for da in derive_assets:
                print(f"      - {da['parent_name']}·{da['name']}: {da['desc']}")

        # Step 3.2: 生成基础角色三视图
        visual_style_path = output_dir / "visual-style.md"
        character_style = ""
        if visual_style_path.is_file():
            character_style = get_slice(
                visual_style_path.read_text(encoding="utf-8"), "character"
            )
        # 为每个角色准备 id/name/description
        char_dicts = []
        for i, c in enumerate(characters_list):
            char_dicts.append({
                "id": c.get("id", f"char_{i}"),
                "name": c.get("name", f"角色{i}"),
                "description": c.get("appearance", {}).get("summary", c.get("description", "")),
                "appearance": c.get("appearance", {}),  # 传递完整 appearance dict
                "style": "\n\n".join(
                    part for part in (c.get("style", ""), character_style) if part
                ),
            })

        _p3_est = estimate_phase_duration("phase3", num_characters=len(char_dicts))
        print(f"  ⏱ Phase 3 开始 (预估 ~{int(_p3_est)}s)")
        print(f"  → batch_generate: {len(char_dicts)} 个角色, skip_images={dry_run}")

        if not dry_run:
            print("[cooldown] 等待 120s 让 Agent Plan 限流窗口重置...", flush=True)
            time.sleep(120)
        
        # Use retry policy for each character generation
        results = []
        _p3_char_start = _now()
        for i, char_dict in enumerate(char_dicts):
            char_name = char_dict.get("name", f"角色{i}")
            print(f"    → [{i+1}/{len(char_dicts)}] {char_name}...")
            _char_t0 = _now()
            
            def _gen_char():
                # Pass output_dir (not chars_dir) — generate_character appends /characters/ internally
                return batch_generate([char_dict], str(output_dir), skip_images=dry_run)
            
            try:
                result = _retry_with_policy(_gen_char, max_attempts=3, backoff_factor=2.0)
                results.extend(result or [])
            except Exception as e:
                print(f"    ✗ {char_name} 生成失败: {e}")
                results.append(None)
            _char_elapsed = round(_now() - _char_t0, 1)
            _char_cumulative = round(_now() - _p3_char_start, 1)
            print(f"  ⏱ {char_name} 完成 (耗时 {_char_elapsed}s, 累计 {_char_cumulative}s / 预估 {int(_p3_est)}s)")

        # 统计输出
        for r in (results or []):
            if isinstance(r, dict):
                name = r.get("name", r.get("id", "unknown"))
                outputs.append(f"characters/{name}/")
            elif isinstance(r, str):
                outputs.append(r)

        if not outputs:
            # fallback: 扫描目录
            for d in chars_dir.iterdir():
                if d.is_dir():
                    outputs.append(f"characters/{d.name}/")

        print(f"  ✓ Phase 3 完成: {len(outputs)} 角色卡 + {len(derive_assets)} 衍生资产")
        
        # Quality gate: Phase 3 (CRITICAL — blocks pipeline if character images missing)
        qg_report = run_quality_check("phase3", output_dir)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 3 质检未通过: {qg_report.grade} — 角色图片缺失，不能继续", "quality_report": qg_report, "duration_s": _elapsed(start)}

        # Phase 2 deliberately defers character-bearing shot images until the
        # reference packs exist. Refresh older t2i artifacts from legacy runs.
        storyboard_path = output_dir / "STORYBOARD.json"
        if storyboard_path.is_file():
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            expected_shots = len(storyboard.get("shots", []))
            generated_shots = _generate_shot_images(output_dir, storyboard)
            if generated_shots != expected_shots:
                return {
                    "status": "error",
                    "error": (
                        "Phase 3 could not produce all character-locked storyboard "
                        f"images: {generated_shots}/{expected_shots}"
                    ),
                    "duration_s": _elapsed(start),
                }
            composition_report = _validate_storyboard_image_composition(
                output_dir, storyboard
            )
            if not composition_report["valid"]:
                return {
                    "status": "error",
                    "error": "Storyboard composition validation failed after Phase 3",
                    "composition_report": composition_report,
                    "duration_s": _elapsed(start),
                }
        
        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs or ["characters/"],
            "derive_assets_count": len(derive_assets),
            "derive_assets": derive_assets,
        }

    except ImportError as e:
        print(f"  ⚠ Phase 3 import 失败: {e}")
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 4: 编排器 (STORYBOARD.json → shots/S*/SHOT_META.json)
# ---------------------------------------------------------------------------

def run_phase4(output_dir: Path, dry_run: bool) -> dict:
    """Phase 4: orchestrator — 镜头编排（通过 CLI 调用）"""
    _banner(4, 9, "编排器 (Orchestrator)", dry_run)
    start = _now()
    _p4_est = estimate_phase_duration("phase4")
    print(f"  ⏱ Phase 4 开始 (预估 ~{int(_p4_est)}s)")
    outputs = []
    output_dir = Path(output_dir)

    storyboard_path = output_dir / "STORYBOARD.json"
    if not storyboard_path.exists():
        return {"status": "error", "error": "STORYBOARD.json not found", "duration_s": _elapsed(start)}

    try:
        from phases.phase4.scene_consistency import write_scene_consistency

        storyboard_for_consistency = json.loads(storyboard_path.read_text(encoding="utf-8"))
        characters_path = output_dir / "CHARACTERS.json"
        characters_for_consistency = (
            json.loads(characters_path.read_text(encoding="utf-8"))
            if characters_path.exists() else {"characters": []}
        )
        visual_style_path = next(
            (
                candidate for candidate in (
                    output_dir / "visual-style.md",
                    output_dir / "visual_style_spec.md",
                ) if candidate.exists()
            ),
            None,
        )
        write_scene_consistency(
            output_dir / "SCENE_CONSISTENCY.json",
            storyboard_for_consistency,
            characters_for_consistency,
            visual_style_path,
        )
        outputs.append("SCENE_CONSISTENCY.json")
        print("  ✓ 场景一致性契约: SCENE_CONSISTENCY.json")

        orchestrator_script = LEGACY_TOOLS_DIR / "orchestrator.py"
        if not orchestrator_script.exists():
            return {"status": "error", "error": f"orchestrator.py not found at {orchestrator_script}", "duration_s": _elapsed(start)}

        shots_dir = output_dir / "shots"
        cmd = [
            sys.executable, str(orchestrator_script),
            "--storyboard", str(storyboard_path.resolve()),
            "--skip-assembly",
            "--shots-dir", str(shots_dir.resolve()),
            # Phase 4 owns routing and SHOT_META creation only.  The legacy
            # orchestrator's live mode also submits video jobs, which belongs
            # exclusively to Phase 6 and can otherwise double-submit work.
            "--dry-run",
        ]

        print(f"  → orchestrator: {' '.join(cmd[-4:])}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(LEGACY_TOOLS_DIR),
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (str(PIPELINE_SRC_DIR), os.environ.get("PYTHONPATH", "")),
                    )
                ),
            },
        )

        print(f"  → orchestrator return code: {result.returncode}")

        if result.returncode != 0:
            print(f"  ⚠ orchestrator stdout tail: {result.stdout[-1500:]}")
            print(f"  ⚠ orchestrator stderr tail: {result.stderr[-1000:]}")
            print(f"  ⚠ orchestrator stderr: {result.stderr[-500:]}")
            # 非致命：可能只是没有视频文件
            if "shots" in str(result.stdout) or dry_run:
                print("  → 继续（dry-run 模式下部分错误可接受）")

        # 扫描输出
        shots_dir = output_dir / "shots"
        if shots_dir.exists():
            for d in sorted(shots_dir.iterdir()):
                if d.is_dir() and d.name.startswith("S"):
                    outputs.append(f"shots/{d.name}/")

        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        provider_candidates = [
            {"name": "local_video", "provider": "local", "capabilities": ["i2v", "flf2v"], "quality": .8, "control": .9, "reliability": .8, "cost": 0, "latency_score": .7},
            {"name": "seedance", "provider": "volcengine", "capabilities": ["i2v", "t2v", "reference_image"], "quality": .9, "control": .7, "reliability": .75, "cost": .4, "latency_score": .6},
        ]
        required_capabilities = sorted({
            shot.get("gen_strategy", "i2v") for shot in storyboard.get("shots", [])
        })
        rankings = rank_providers(provider_candidates, {"capabilities": required_capabilities})
        selected_provider = rankings[0].tool_name if rankings else "local_video"
        composition = {
            "cuts": [{"id": shot.get("id"), "type": shot.get("type", "video")} for shot in storyboard.get("shots", [])],
            "provider": selected_provider,
            "provider_rankings": [score.to_dict() for score in rankings],
        }
        locked_composition = lock_runtime(composition, available={"ffmpeg", "remotion"})

        director_scenes = []
        director_plan_path = output_dir / "director_plan.json"
        if director_plan_path.exists():
            director_scenes = json.loads(director_plan_path.read_text(encoding="utf-8")).get("scenes", [])
        for index, shot_dir in enumerate(sorted(shots_dir.glob("S*")) if shots_dir.exists() else []):
            meta_path = shot_dir / "SHOT_META.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["provider"] = selected_provider
            meta["provider_rankings"] = locked_composition["provider_rankings"]
            meta["render_runtime"] = locked_composition["render_runtime"]
            if index < len(director_scenes) and director_scenes[index].get("speech_pacing"):
                meta["speech_pacing"] = director_scenes[index]["speech_pacing"]
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        shot_output_count = sum(item.startswith("shots/") for item in outputs)
        print(f"  ✓ Phase 4 完成: {shot_output_count} 镜头目录")
        status = "done" if shot_output_count or dry_run else "error"
        return {"status": status, "duration_s": _elapsed(start), "outputs": outputs or ["shots/"], "provider": selected_provider, "render_runtime": locked_composition["render_runtime"], **({"error": "orchestrator produced no shot directories"} if status == "error" else {})}

    except subprocess.TimeoutExpired as e:
        timeout_stdout = e.stdout or ""
        timeout_stderr = e.stderr or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode(errors="replace")
        if isinstance(timeout_stderr, bytes):
            timeout_stderr = timeout_stderr.decode(errors="replace")
        print("  ⚠ orchestrator timed out after 120s")
        print(f"  ⚠ orchestrator stdout tail: {timeout_stdout[-1500:]}")
        print(f"  ⚠ orchestrator stderr tail: {timeout_stderr[-1000:]}")
        return {"status": "error", "error": "orchestrator timed out", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 6: 视频生成 (OM SeedanceVideo — reference_to_video)
# ---------------------------------------------------------------------------

def _run_phase6_om_seedance(storyboard_data: dict, output_dir: Path, characters_data: Optional[dict] = None, _timing_ctx: Optional[dict] = None) -> dict:
    """使用 OM SeedanceVideo 生成视频（支持 reference_to_video）
    
    Args:
        storyboard_data: STORYBOARD.json 的内容
        output_dir: 输出目录
        characters_data: CHARACTERS.json 的内容（可选，用于注入角色参考图）
        _timing_ctx: 可选计时上下文 {start, estimate}，用于打印子节点进度
    """
    from vendor.video_tools.tools.video.seedance_video import SeedanceVideo

    sv = SeedanceVideo()

    # 检查工具是否可用
    status = sv.get_status()
    if status.value != "available":
        raise ImportError(f"SeedanceVideo not available (status={status})")

    shots = storyboard_data.get("shots", [])
    has_shot_references = any(
        _shot_storyboard_reference(output_dir, shot.get("id")) is not None
        for shot in shots
    )

    # 构建角色参考图映射：character_id -> preferred reference path
    character_ref_images = {}
    if characters_data:
        characters = characters_data.get("characters", [])
        for char in characters:
            char_id = char.get("id", "")
            char_name = char.get("name", "")
            char_dirs = [
                output_dir / "characters" / char_id,
                output_dir / "characters" / "characters" / char_id,
            ]
            reference_path = None
            for char_dir in char_dirs:
                candidates = [
                    char_dir / "face_closeup.png",
                    char_dir / "full_body.png",
                    *sorted(char_dir.glob("variant_*.png")),
                    char_dir / "front.png",  # legacy fallback
                ]
                reference_path = next((path for path in candidates if path.exists()), None)
                if reference_path is not None:
                    break
            if reference_path is not None:
                character_ref_images[char_id] = str(reference_path)
                character_ref_images[char_name] = str(reference_path)  # 也支持按名称匹配
                print(f"  ✓ 角色参考图: {char_name} -> {reference_path.name}")

    if has_shot_references:
        print("  → 模式: reference_to_video (逐镜分镜图存在)")
    else:
        print("  → 模式: 角色参考图或 text_to_video")

    outputs = []
    errors = []

    # Optional style context from storyboard
    style_context = None
    if storyboard_data.get("style"):
        style_context = {"mood": storyboard_data["style"]}

    # Sequential generation with retry policy per shot
    print(f"  → 生成 {len(shots)} 个镜头 (retry=3, backoff=2.0)...")
    
    for shot in shots:
        shot_id = shot.get("id", "?")
        raw_prompt = shot.get("prompt", "")
        duration = str(shot.get("duration", 5))
        aspect_ratio = shot.get("aspect_ratio", "16:9")

        # Use OM build_shot_prompt for standardized prompt construction
        # Falls back to raw prompt if no shot_language metadata present
        prompt_items = build_batch_prompts([shot], style_context)
        prompt = prompt_items[0]["prompt"] if prompt_items else ""
        if not prompt or len(prompt) < 5:
            prompt = raw_prompt  # fallback to original prompt
        elif raw_prompt and len(raw_prompt) > len(prompt):
            # If original prompt is richer, append structured layers
            prompt = f"{prompt}. {raw_prompt}"

        if not prompt:
            continue

        shot_dir = output_dir / f"shots/S{shot_id}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        video_path = shot_dir / "output.mp4"

        def _generate_shot():
            """Inner function for retry logic."""
            # 优先使用角色参考图
            character_ref = None
            if character_ref_images:
                # 从 shot 中提取角色信息
                shot_characters = shot.get("characters", [])
                if not shot_characters:
                    # 尝试从 prompt 中匹配角色名称
                    for char_key, char_path in character_ref_images.items():
                        if char_key.lower() in prompt.lower():
                            character_ref = char_path
                            print(f"    ✓ 匹配到角色参考图: {char_key}")
                            break
                else:
                    # 使用 shot 中明确指定的角色
                    for char_id in shot_characters:
                        if char_id in character_ref_images:
                            character_ref = character_ref_images[char_id]
                            print(f"    ✓ 使用角色参考图: {char_id}")
                            break
            
            # 单镜构图图已经通过角色参考生成，因此优先级最高。总览网格
            # storyboard.png 绝不能作为单镜视频参考，否则会传播分格构图。
            shot_reference = _shot_storyboard_reference(output_dir, shot_id)
            reference_image = str(shot_reference) if shot_reference else character_ref
            
            if reference_image:
                # 优先使用 reference_to_video
                try:
                    result = sv.execute({
                        "operation": "reference_to_video",
                        "prompt": prompt,
                        "reference_image_paths": [reference_image],
                        "duration": duration,
                        "aspect_ratio": aspect_ratio,
                        "output_path": str(video_path),
                    })
                    if result.success:
                        return result
                    else:
                        # reference_to_video 失败，降级到 text_to_video
                        print(f"    ⚠ reference_to_video 失败: {result.error}, 降级到 text_to_video...")
                except Exception as e:
                    print(f"    ⚠ reference_to_video 异常: {e}, 降级到 text_to_video...")

            # text_to_video（降级或无参考图片）
            result = sv.execute({
                "operation": "text_to_video",
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "output_path": str(video_path),
            })
            return result

        try:
            _p5_est_val = int(_timing_ctx["estimate"]) if _timing_ctx else 0
            print(f"  → S{shot_id}: 生成视频...")
            _shot_t0 = _now()
            result = _retry_with_policy(_generate_shot, max_attempts=3, backoff_factor=2.0)
            _shot_elapsed = round(_now() - _shot_t0, 1)
            _p5_cumulative = round(_now() - (_timing_ctx["start"] if _timing_ctx else _now()), 1)

            if result.success:
                # 如果输出不在目标位置，复制过去
                out_path = result.data.get("output_path") or result.data.get("output")
                if out_path and Path(out_path) != video_path and Path(out_path).exists():
                    import shutil
                    shutil.copy2(out_path, video_path)
                outputs.append(f"shots/S{shot_id}/output.mp4")
                print(f"    ✓ S{shot_id}: 视频已生成")
                if _timing_ctx:
                    print(f"  ⏱ S{shot_id} 完成 (耗时 {_shot_elapsed}s, 累计 {_p5_cumulative}s / 预估 {_p5_est_val}s)")
            else:
                error_msg = result.error or "unknown error"
                errors.append(f"S{shot_id}: {error_msg}")
                print(f"    ✗ S{shot_id}: 生成失败 — {error_msg}")
                if _timing_ctx:
                    print(f"  ⏱ S{shot_id} 失败 (耗时 {_shot_elapsed}s, 累计 {_p5_cumulative}s / 预估 {_p5_est_val}s)")

        except Exception as e:
            errors.append(f"S{shot_id}: {e}")
            print(f"    ✗ S{shot_id}: 所有重试均失败 — {e}")
            _shot_elapsed = round(_now() - _shot_t0, 1)
            _p5_cumulative = round(_now() - (_timing_ctx["start"] if _timing_ctx else _now()), 1)
            if _timing_ctx:
                print(f"  ⏱ S{shot_id} 失败 (耗时 {_shot_elapsed}s, 累计 {_p5_cumulative}s / 预估 {_p5_est_val}s)")
            continue

    return {
        "status": "done" if outputs else "error",
        "outputs": outputs,
        "errors": errors,
        "provider": "seedance",
        "mode": "reference_to_video" if has_shot_references else "text_to_video",
    }


def _apply_chain_relay(content_list, first_frame_b64, shot_id):
    """Replace the first frame with a relay frame unless content is reference-only."""
    if any(item.get("role") == "reference_image" for item in (content_list or [])):
        print(
            f"    [chain] {shot_id}: reference-only shot, skipping tail-frame relay",
            flush=True,
        )
        return content_list
    if any(
        item.get("type") == "text"
        and "[identity-lock: text-only; no reference media]" in item.get("text", "")
        for item in (content_list or [])
    ):
        print(
            f"    [chain] {shot_id}: identity-locked FLF2V shot, "
            "keeping its storyboard first frame",
            flush=True,
        )
        return content_list
    content_list = [
        item for item in (content_list or [])
        if item.get("role") != "first_frame"
    ]
    content_list.insert(1 if content_list else 0, {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{first_frame_b64}"},
        "role": "first_frame",
        "priority": "high",
    })
    return content_list


def _prompt_assets_for_shot(shot_meta: dict, characters_data: dict) -> list[dict]:
    """Return only character prompt assets explicitly bound to this shot."""
    requested = shot_meta.get("who") or shot_meta.get("characters") or []
    if not isinstance(requested, list):
        requested = [requested] if requested else []
    requested_keys = {str(value).casefold() for value in requested if value}
    for asset_id in shot_meta.get("associate_assets", []):
        if isinstance(asset_id, str) and asset_id.startswith("char:"):
            requested_keys.add(asset_id[5:].split(":", 1)[0].casefold())

    selected = []
    for character in characters_data.get("characters", []):
        keys = {
            str(character.get("id", "")).casefold(),
            str(character.get("name", "")).casefold(),
            *{
                str(alias).casefold()
                for alias in character.get("aliases", [])
                if alias
            },
        }
        if requested_keys.intersection(keys):
            selected.append({
                "name": character.get("name", ""),
                "description": character.get("appearance", {}).get("summary", ""),
            })
    return selected


def _rejected_privacy_image_url(content: list[dict] | None, error: object) -> str | None:
    """Return the stable URL key for the provider-rejected content[N] image."""
    import re
    from urllib.parse import urlsplit

    match = re.search(r"content\[(\d+)\]", str(error))
    if not match or not content:
        return None
    index = int(match.group(1))
    if not 0 <= index < len(content):
        return None
    item = content[index]
    if item.get("type") != "image_url":
        return None
    url = item.get("image_url", {}).get("url")
    if not url:
        return None
    parsed = urlsplit(str(url))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _without_rejected_privacy_images(
    content: list[dict], rejected_urls: set[str]
) -> list[dict]:
    """Remove only previously rejected images while retaining safe references."""
    if not rejected_urls:
        return content
    from urllib.parse import urlsplit

    filtered = []
    for item in content:
        url = item.get("image_url", {}).get("url") if item.get("type") == "image_url" else None
        if url:
            parsed = urlsplit(str(url))
            stable_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if stable_url in rejected_urls:
                continue
        filtered.append(item)
    return filtered


def _privacy_fallback_strategy(gen_strategy: str) -> str:
    """Return a structurally valid route after an endpoint-frame rejection."""
    # FLF2V requires first_frame + last_frame as an inseparable pair. Removing
    # only one endpoint creates an invalid request, so fall back to Phantom's
    # identity references instead. Phantom/i2v can drop one rejected image.
    return "phantom" if gen_strategy == "flf2v" else gen_strategy


def _run_phase6_fallback(output_dir: Path, chain_mode: bool = False) -> dict:
    """Generate Phase 6 video through direct ARK or the explicit local Bridge."""
    output_dir = Path(output_dir)

    shots_dir = output_dir / "shots"
    if not shots_dir.exists():
        return {"status": "skipped", "reason": "no shots directory"}

    video_provider = os.environ.get("VIDEO_PROVIDER", "seedance").lower()
    use_local = video_provider in {"local", "wan", "bridge"}
    if use_local:
        try:
            from clients import local_video_client
        except ImportError:
            print("  ✗ Phase 6 前置检查失败: local_video_client 未找到", flush=True)
            return {"status": "error", "error": "local_video_client not found"}
        if not local_video_client.is_available(timeout=3.0):
            print("  ✗ Phase 6 前置检查失败: 本地视频 API 不可达", flush=True)
            return {"status": "error", "error": "local video API unreachable"}
        if video_provider == "bridge":
            print("  → 路由: 通过 Bridge 使用 Seedance 在线模型", flush=True)
        else:
            print("  → 路由: 仅使用本地视频 API (192.168.31.221:9100)", flush=True)
        if chain_mode and video_provider != "bridge":
            print("  [chain] 当前 provider 不是 seedance，Wan2.2 本地不支持接力；按普通模式执行", flush=True)
            chain_mode = False
    else:
        print("  → 路由: 直连 ARK Agent Plan", flush=True)

    from runtime.generation_tasks import GenerationTaskStore

    generation_tasks = GenerationTaskStore(output_dir / "runtime.db")

    # Load character reference images for consistency
    import base64 as _b64
    char_ref_map = {}   # {match_key_lower: base64_of_front_png}
    char_list = []      # [(char_id, char_name, b64)] for fallback
    chars_path = output_dir / "CHARACTERS.json"
    chars_data = {"characters": []}
    declared_character_ids = set()
    missing_character_fronts = set()
    if chars_path.exists():
        chars_data = json.loads(chars_path.read_text())
        for char in chars_data.get("characters", []):
            declared_character_ids.add(char["id"])
            # Try both directory structures: characters/{id}/ and characters/characters/{id}/
            reference_path = None
            for char_dir in (
                output_dir / "characters" / char["id"],
                output_dir / "characters" / "characters" / char["id"],
            ):
                candidates = [
                    char_dir / "face_closeup.png",
                    char_dir / "full_body.png",
                    *sorted(char_dir.glob("variant_*.png")),
                    char_dir / "front.png",  # legacy fallback
                ]
                reference_path = next((path for path in candidates if path.exists()), None)
                if reference_path is not None:
                    break
            if reference_path is not None:
                b64 = _b64.b64encode(reference_path.read_bytes()).decode()
                # Map multiple keys for matching: Chinese name, pinyin id, id without underscores
                char_ref_map[char["name"].lower()] = b64
                char_ref_map[char["id"].lower()] = b64
                char_ref_map[char["id"].replace("_", "").lower()] = b64
                char_list.append((char["id"], char["name"], b64))
            else:
                missing_character_fronts.add(char["id"])
        if char_ref_map:
            print(f"  → 已加载 {len(char_list)} 个角色参考图")
        if missing_character_fronts:
            print(
                "  ⚠ Phase 6 前置检查: 缺少角色参考图 "
                "(face_closeup.png/full_body.png/variant_*.png): "
                + ", ".join(sorted(missing_character_fronts)),
                flush=True,
            )

    # Determine protagonist (first character) for default injection
    protagonist_b64 = char_list[0][2] if char_list else None
    protagonist_name = char_list[0][1] if char_list else None

    outputs = []
    # --- P1-C: Seed Locking（参考 HonCut asset_manifest seed）---
    # 同场景镜头使用相同 seed，确保背景一致性
    scene_seed_map = {}  # {where: seed}
    prev_shot_dir = None  # --- P1-D2: 上一镜头视频作为运动参考 ---
    scene_consistency_path = output_dir / "SCENE_CONSISTENCY.json"
    scene_consistency_data = (
        json.loads(scene_consistency_path.read_text(encoding="utf-8"))
        if scene_consistency_path.exists() else {}
    )

    # --- 并发配置 ---
    try:
        from utils.config import VIDEO_GEN_CONCURRENCY
        concurrency = VIDEO_GEN_CONCURRENCY
    except ImportError:
        concurrency = int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1"))
    provider_capacity = max(1, concurrency)
    provider_slots = None
    provider_leases = None
    if not use_local:
        from runtime.capacity import (
            CapacityTable,
            CrossProcessSlotTable,
            SlotTable,
            default_capacity_lease_path,
        )

        provider_slots = SlotTable()
        provider_leases = CrossProcessSlotTable(default_capacity_lease_path())
        if not chain_mode:
            capacities = CapacityTable.for_seedance_video(provider_capacity)
            provider_capacity = capacities.get("seedance", "video")
    if chain_mode:
        concurrency = 1
        provider_capacity = 1
        print("  [chain] Seedance 尾帧接力已启用，强制按 shot_id 串行生成", flush=True)
    elif not use_local:
        concurrency = provider_capacity
    else:
        concurrency = max(1, concurrency)
    capacity_source = (
        "seedance video capacity" if not use_local else "VIDEO_GEN_CONCURRENCY"
    )
    print(
        f"  → 并发模式: {capacity_source}={concurrency} "
        f"({'串行' if concurrency == 1 else f'并行 workers={concurrency}'})"
    )

    shot_dirs = [d for d in sorted(shots_dir.iterdir()) if d.is_dir() and d.name.startswith("S")]
    pending_shot_dirs = []
    for shot_dir in shot_dirs:
        existing_output = shot_dir / "output.mp4"
        if existing_output.exists() and existing_output.stat().st_size > 10 * 1024:
            print(f"  ⏭ {shot_dir.name}: output.mp4 exists, skipping")
            outputs.append(f"shots/{shot_dir.name}/output.mp4")
        else:
            pending_shot_dirs.append(shot_dir)

    task_dir_id = None
    if use_local and os.environ.get("HONCUT_TASK_DIR_MODE") == "1" and pending_shot_dirs:
        from tools import task_dir_exporter

        export_shots = {}
        for pending_dir in pending_shot_dirs:
            pending_meta_path = pending_dir / "SHOT_META.json"
            if not pending_meta_path.exists():
                raise FileNotFoundError(f"Missing shot metadata for task export: {pending_meta_path}")
            pending_meta = json.loads(pending_meta_path.read_text(encoding="utf-8"))
            pending_meta["_char_ids"] = sorted({
                asset_id[5:].split(":", 1)[0]
                for asset_id in pending_meta.get("associate_assets", [])
                if isinstance(asset_id, str) and asset_id.startswith("char:")
            })
            export_shots[pending_dir.name] = pending_meta
        local_task_dir = task_dir_exporter.build_task_dir(
            output_dir,
            [directory.name for directory in pending_shot_dirs],
            {
                "shots": export_shots,
                "chain_mode": chain_mode,
                "model": os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2.0-mini"),
                "resolution": "720p",
            },
        )
        task_dir_id = task_dir_exporter.upload_task_dir(local_task_dir, "tasks")
        print(f"  [task_dir] uploaded tasks/{task_dir_id}", flush=True)
    
    def _process_shot(
        shot_dir: Path,
        chain_source: Optional[tuple[str, Path]] = None,
        chain_allowed: bool = True,
    ) -> Optional[dict]:
        """处理单个镜头的视频生成，返回 output.mp4 路径或 None"""
        meta_path = shot_dir / "SHOT_META.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        active_source = chain_source if chain_mode and chain_allowed else None
        meta["chain_source"] = active_source[0] if active_source else None
        meta["chain_active"] = bool(active_source)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt = meta.get("prompt", "")
        if scene_consistency_data:
            from phases.phase6.video_generator import build_video_prompt

            scene_contract = scene_consistency_data.get("shots", {}).get(
                shot_dir.name, {}
            )
            if scene_contract:
                meta["lighting_description"] = (
                    scene_contract.get("lighting_description")
                    or scene_contract.get("lighting_note")
                    or meta.get("lighting_description")
                )
                meta["style_anchor"] = (
                    scene_contract.get("style_anchor")
                    or scene_contract.get("style_suffix")
                    or scene_consistency_data.get("global_style_lock")
                    or meta.get("style_anchor")
                )

            routed_prompt = build_video_prompt(
                meta,
                chars_data,
                scene_consistency_data,
                os.environ.get("VIDEO_MODEL", "seedance"),
            )
            if isinstance(routed_prompt, dict):
                prompt = routed_prompt["prompt"]
                meta["negative_prompt"] = routed_prompt["negative_prompt"]
            else:
                prompt = routed_prompt
            meta["prompt"] = prompt
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        gen_strategy = meta.get("gen_strategy", "i2v")
        if gen_strategy not in {"flf2v", "phantom", "i2v"}:
            gen_strategy = "i2v"
        route_reason = {
            "flf2v": "action shot",
            "phantom": "dialogue/emotion shot",
            "i2v": "scenery/ambient or default",
        }[gen_strategy]
        video_provider = os.environ.get("VIDEO_PROVIDER", "seedance").lower()
        if video_provider == "bridge":
            bridge_model = "seedance"
        else:
            bridge_model = {"flf2v": "flf2v", "phantom": "phantom", "i2v": "wan22"}[gen_strategy]
        print(f"    [route] {shot_dir.name} → {gen_strategy} ({route_reason})")
        associated_character_ids = {
            asset_id[5:].split(":", 1)[0]
            for asset_id in meta.get("associate_assets", [])
            if isinstance(asset_id, str) and asset_id.startswith("char:")
        }
        missing_for_shot = associated_character_ids & missing_character_fronts
        if missing_for_shot or (declared_character_ids and not char_list):
            missing_ids = missing_for_shot or missing_character_fronts
            print(
                f"    ✗ {shot_dir.name}: 缺少角色参考图 "
                "characters/*/{face_closeup.png,full_body.png,variant_*.png} "
                f"({', '.join(sorted(missing_ids))})，跳过镜头",
                flush=True,
            )
            return None
        # --- M4: 模型路由（增量，失败用原始 prompt）---
        try:
            from prompt.prompt_router import route_prompt
            try:
                from utils.config import SEEDANCE_MODEL
                model_name = SEEDANCE_MODEL
            except ImportError:
                model_name = os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2.0-mini")
            routed_prompt = route_prompt(
                model_name=model_name,
                mode="single_shot",
                shot_data=meta,
                assets=_prompt_assets_for_shot(meta, chars_data),
            )
            if routed_prompt:
                prompt = routed_prompt
                print(f"    [M4] 提示词路由: {model_name} → single_shot")
        except Exception as e:
            pass  # 降级用原始 prompt
        duration = meta.get("duration")  # 从 SHOT_META 读取；缺失时由模型 profile 选中间档
        if not prompt:
            return None

        # Find character reference for this shot
        # Strategy: match by name/id in prompt → fallback to protagonist
        first_frame_b64 = None
        prompt_lower = prompt.lower()

        # Canonical visual reference is the one-image-per-shot artifact.
        # Never inject storyboard.png: it is a multi-panel overview sheet.
        shot_image = _shot_storyboard_reference(output_dir, shot_dir.name)
        if shot_image is not None:
            first_frame_b64 = _b64.b64encode(shot_image.read_bytes()).decode()
            print(f"    [M2] 注入逐镜分镜图: {shot_image.name}")

        # --- P0-C: HonCut 资产ID绑定匹配（associateAssetsIds）---
        # --- P1-A4: 衍生参考图匹配（char:id:state → variant_state.png）---
        associate_assets = meta.get("associate_assets", [])
        if associate_assets and first_frame_b64 is None:
            for asset_id in associate_assets:
                if asset_id.startswith("char:"):
                    parts = asset_id[5:].split(":")
                    char_id = parts[0]
                    variant_state = parts[1] if len(parts) > 1 else None

                    if variant_state:
                        # 衍生参考图匹配（P1-A4）
                        variant_png = output_dir / "characters" / char_id / f"variant_{variant_state}.png"
                        if not variant_png.exists():
                            variant_png = output_dir / "characters" / "characters" / char_id / f"variant_{variant_state}.png"
                        if variant_png.exists():
                            first_frame_b64 = _b64.b64encode(variant_png.read_bytes()).decode()
                            print(f"    [P1-A] 衍生参考图匹配: {char_id}:{variant_state}")
                            break

                    # 基准参考图匹配（新资产优先，旧 front.png 仅兼容）
                    reference_path = None
                    for char_dir in (
                        output_dir / "characters" / char_id,
                        output_dir / "characters" / "characters" / char_id,
                    ):
                        candidates = [
                            char_dir / "face_closeup.png",
                            char_dir / "full_body.png",
                            *sorted(char_dir.glob("variant_*.png")),
                            char_dir / "front.png",
                        ]
                        reference_path = next(
                            (path for path in candidates if path.exists()), None
                        )
                        if reference_path is not None:
                            break
                    if reference_path is not None:
                        first_frame_b64 = _b64.b64encode(reference_path.read_bytes()).decode()
                        print(f"    [P0-C] 资产绑定匹配角色: {char_id}")
                        break

        # Strategy: match by name/id in prompt only when no canonical shot
        # frame was found. Global style text may mention a protagonist even for
        # explicit who=[] scenery shots; it must never replace Sxx.png.
        if first_frame_b64 is None:
            for char_name, b64 in char_ref_map.items():
                if char_name in prompt_lower:
                    first_frame_b64 = b64
                    print(f"    [ref] 注入角色参考: {char_name}")
                    break
        # Fallback: inject protagonist for shots with human activity keywords
        shot_who = meta.get("who") or meta.get("characters") or []
        if first_frame_b64 is None and protagonist_b64 and shot_who:
            human_keywords = ["woman", "man", "girl", "boy", "person", "she", "he",
                              "her", "his", "lin xia", "shen yu", "xia", "yu"]
            if any(kw in prompt_lower for kw in human_keywords):
                first_frame_b64 = protagonist_b64
                print(f"    [ref] 注入主角参考 (fallback): {protagonist_name}")

        # --- P0-A3: 场景参考图（逐镜图/角色参考缺失时使用）---
        if first_frame_b64 is None:
            shot_where = meta.get("where", "")
            if shot_where:
                scene_id = shot_where.replace(" ", "_").replace("/", "_")[:30]
                scene_ref = output_dir / "scenes" / scene_id / "reference.png"
                if scene_ref.exists() and scene_ref.stat().st_size > 1024:
                    first_frame_b64 = _b64.b64encode(scene_ref.read_bytes()).decode()
                    print(f"    [P0-A] 注入场景参考图: {scene_id}")

        if active_source:
            try:
                first_frame_b64 = _b64.b64encode(active_source[1].read_bytes()).decode()
                print(f"    [chain] {shot_dir.name}: 首帧接力自 {active_source[0]}", flush=True)
            except Exception as error:
                print(f"    [chain] {shot_dir.name}: 无法读取 {active_source[0]} 尾帧，回退独立首帧 — {error}", flush=True)
                active_source = None
                meta["chain_source"] = None
                meta["chain_active"] = False
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # --- P1-C2: 同场景同 seed ---
        shot_where = meta.get("where", "")
        shot_seed = None
        if shot_where:
            if shot_where not in scene_seed_map:
                import hashlib
                scene_seed_map[shot_where] = int(hashlib.md5(shot_where.encode()).hexdigest()[:8], 16) % 2147483647
            shot_seed = scene_seed_map[shot_where]

        # --- P1-D2: 上一镜头视频作为运动参考（可选，仅串行模式）---
        prev_video_ref = None
        if concurrency == 1 and prev_shot_dir is not None:
            prev_output = prev_shot_dir / "output.mp4"
            if prev_output.exists() and prev_output.stat().st_size > 10240:
                try:
                    prev_video_ref = _b64.b64encode(prev_output.read_bytes()).decode()
                    # 视频参考大小限制（>5MB 不传，避免超限）
                    if len(prev_video_ref) > 5 * 1024 * 1024 * 4 // 3:
                        prev_video_ref = None
                except Exception as error:
                    print(
                        f"  ⚠ {shot_dir.name}: 无法读取上一镜头运动参考 — {error}",
                        flush=True,
                    )

        max_retries = 3
        quota_retries = 0
        privacy_retries = 0
        policy_retries = 0
        privacy_rejected_urls: set[str] = set()
        privacy_retry_strategy = gen_strategy
        content_list = None
        # Retry budgets are failure-class specific. A temporary quota burst
        # must not consume the later privacy-fallback opportunity (or vice
        # versa), otherwise the first different error after backoff becomes a
        # false terminal failure.
        while True:
            try:
                out_path = str(shot_dir / "output.mp4")
                
                # --- 本地 API 路由 ---
                if use_local:
                    try:
                        print(f"  → {shot_dir.name}: 提交本地 API 视频生成...")
                        from clients import local_video_client
                        from tools import asset_packager
                        
                        # [LEGACY-KEEP v2.0] Build content[] for Windows Bridges not yet on task_dir.
                        shot_id = shot_dir.name  # e.g., "S01"
                        content_meta = dict(meta)
                        content_meta["prompt"] = prompt
                        content_meta["gen_strategy"] = gen_strategy
                        content_meta["_char_ids"] = sorted(associated_character_ids)
                        zip_path = None
                        base64_list = []
                        content_list = None
                        if task_dir_id is None:
                            content_list = asset_packager.build_content_for_shot(
                                output_dir=output_dir,
                                shot_id=shot_id,
                                shot_meta=content_meta,
                            )
                            if active_source and first_frame_b64:
                                content_list = _apply_chain_relay(
                                    content_list, first_frame_b64, shot_id
                                )

                            # [LEGACY-KEEP v2.0] zip/base64 fallback for old Bridges.
                            if not content_list or len(content_list) <= 1:
                                zip_path, base64_list = asset_packager.package_shot_assets(
                                    output_dir=output_dir,
                                    shot_id=shot_id,
                                    shot_meta=meta,
                                )
                                content_list = None

                        generate = (
                            local_video_client.generate_video_with_fallback
                            if bridge_model == "seedance"
                            else local_video_client.generate_video
                        )
                        from functools import partial

                        from runtime.bridge_execution import execute_bridge_video_task

                        bridge_generate = partial(
                            generate,
                            prompt=prompt,
                            output_path=out_path,
                            reference_image_base64=first_frame_b64,
                            seed=shot_seed if shot_seed is not None else -1,
                            duration=duration,
                            width=1280,
                            height=720,
                            fps=24,
                            asset_zip_path=zip_path,
                            image_base64_list=base64_list,
                            content=content_list,
                            batch_id=output_dir.name,
                            model=bridge_model,
                            return_last_frame=chain_mode and chain_allowed,
                            task_dir=task_dir_id,
                        )
                        execution = execute_bridge_video_task(
                            generation_tasks,
                            run_id=str(output_dir.resolve()),
                            resource_id=shot_id,
                            payload={
                                "shot_id": shot_id,
                                "output_path": f"shots/{shot_id}/output.mp4",
                                "model": bridge_model,
                                "duration": duration,
                                "seed": shot_seed if shot_seed is not None else -1,
                                "task_dir": task_dir_id,
                            },
                            provider_endpoint=local_video_client.get_api_url(),
                            output_path=out_path,
                            generate=bridge_generate,
                        )
                        generation_result = execution.generation_result

                        if shot_seed is not None:
                            meta["seed"] = shot_seed
                            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                        print(f"    ✓ {shot_dir.name}: 视频已生成 (本地 API)")
                        if isinstance(generation_result, str):
                            generation_result = {
                                "output_path": generation_result,
                                "last_frame_path": None,
                                "actual_model": bridge_model,
                            }
                        generation_result["relative_output"] = f"shots/{shot_dir.name}/output.mp4"
                        return generation_result
                    except Exception as local_err:
                        print(f"    ✗ {shot_dir.name}: 本地 API 失败 — {local_err}")
                        print("    ⚠ 不降级到 ARK（零成本测试模式），跳过此镜头")
                        return None

                print(f"  → {shot_dir.name}: 提交 ARK Agent Plan 视频生成...")
                from clients import seedance_client
                from tools import asset_packager

                shot_id = shot_dir.name
                content_meta = dict(meta)
                content_meta["prompt"] = prompt
                content_meta["gen_strategy"] = privacy_retry_strategy
                content_meta["_char_ids"] = sorted(associated_character_ids)
                content_list = asset_packager.build_content_for_shot(
                    output_dir=output_dir,
                    shot_id=shot_id,
                    shot_meta=content_meta,
                )
                content_list = _without_rejected_privacy_images(
                    content_list, privacy_rejected_urls
                )
                if active_source and first_frame_b64:
                    content_list = _apply_chain_relay(
                        content_list, first_frame_b64, shot_id
                    )

                api_key = get_api_key("ARK_AGENT")
                if not api_key:
                    raise RuntimeError("缺少 ARK_AGENT_API_KEY；检查 ARK_AGENT_API_KEY 或 Agent Plan 权限")
                try:
                    from utils.config import SEEDANCE_MODEL
                    direct_model = SEEDANCE_MODEL
                except ImportError:
                    direct_model = os.environ.get(
                        "SEEDANCE_MODEL", "doubao-seedance-2.0-mini"
                    )
                from functools import partial

                from runtime.seedance_execution import execute_seedance_video_task

                execution = execute_seedance_video_task(
                    generation_tasks,
                    run_id=str(output_dir.resolve()),
                    resource_id=shot_id,
                    payload={
                        "shot_id": shot_id,
                        "output_path": f"shots/{shot_id}/output.mp4",
                        "model": direct_model,
                        "duration": duration or 12,
                        "seed": shot_seed,
                    },
                    provider_endpoint=seedance_client.BASE_URL,
                    output_path=out_path,
                    submit=partial(
                        seedance_client.submit_content,
                        content_list,
                        api_key=api_key,
                        model=direct_model,
                        duration=duration or 12,
                        ratio="16:9",
                        seed=shot_seed,
                    ),
                    poll=partial(seedance_client.poll, api_key=api_key),
                    download=seedance_client.download,
                )
                task_id = execution.provider_job_id

                last_frame_path = shot_dir / "last_frame.jpg"
                if chain_mode and chain_allowed:
                    try:
                        subprocess.run(
                            [
                                "ffmpeg", "-sseof", "-0.1", "-i", out_path,
                                "-frames:v", "1", "-q:v", "1",
                                str(last_frame_path), "-y",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except Exception as frame_error:
                        print(
                            f"    ⚠ {shot_dir.name}: 尾帧提取失败 — {frame_error}",
                            flush=True,
                        )

                actual_duration = None
                try:
                    probe = subprocess.run(
                        [
                            "ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", out_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    actual_duration = float(probe.stdout.strip())
                except Exception:
                    pass

                meta.update({
                    "task_id": task_id,
                    "status": "completed",
                    "video_path": out_path,
                    "actual_model": direct_model,
                    "actual_duration": actual_duration,
                })
                if shot_seed is not None:
                    meta["seed"] = shot_seed
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                return {
                    "output_path": out_path,
                    "last_frame_path": (
                        str(last_frame_path) if last_frame_path.exists() else None
                    ),
                    "actual_model": direct_model,
                    "relative_output": f"shots/{shot_dir.name}/output.mp4",
                }
            except Exception as e:
                err_str = str(e)
                # 429 QuotaExceeded — exponential backoff retry
                if "QuotaExceeded" in err_str or "429" in err_str:
                    wait_sec = min(30 * (2 ** quota_retries), 120)
                    if quota_retries < max_retries:
                        quota_retries += 1
                        print(f"    ⚠ {shot_dir.name}: 配额超限(429)，等待 {wait_sec}s 后重试 ({quota_retries}/{max_retries})...")
                        import time as _time
                        _time.sleep(wait_sec)
                        continue
                    else:
                        print(f"    ✗ {shot_dir.name}: 配额超限，已重试 {max_retries} 次，跳过")
                        return None
                # Check auth only after classifying quota errors. Provider
                # request ids are arbitrary hexadecimal-ish strings and can
                # contain the substring "401" or "403" inside a genuine 429.
                if "401" in err_str or "403" in err_str:
                    raise RuntimeError(
                        f"{err_str}；检查 ARK_AGENT_API_KEY 或 Agent Plan 权限"
                    ) from e
                if "PrivacyInformation" in err_str and privacy_retries < max_retries:
                    privacy_retries += 1
                    fallback_strategy = _privacy_fallback_strategy(
                        privacy_retry_strategy
                    )
                    if fallback_strategy != privacy_retry_strategy:
                        privacy_retry_strategy = fallback_strategy
                        privacy_rejected_urls.clear()
                        print(
                            f"    ⚠ {shot_dir.name}: FLF2V 首尾帧有图片被隐私检测拒绝，"
                            "整组切换为 Phantom 身份参考模式后重试"
                        )
                        continue
                    rejected_url = _rejected_privacy_image_url(content_list, e)
                    if rejected_url:
                        privacy_rejected_urls.add(rejected_url)
                        print(
                            f"    ⚠ {shot_dir.name}: 参考图被隐私检测拒绝，"
                            "仅剔除被拒图片并保留其余安全参考后重试"
                        )
                        continue
                    print(
                        f"    ⚠ {shot_dir.name}: 无法定位被拒参考图，"
                        "停止自动重提以避免重复无效请求"
                    )
                if "PolicyViolation" in err_str and policy_retries < max_retries:
                    policy_retries += 1
                    print(f"    ⚠ {shot_dir.name}: 版权误报，重试 ({policy_retries}/{max_retries})...")
                    prompt = prompt.replace("Cinematic", "Original fictional")
                    prompt += ", original character design, non-copyrighted"
                    first_frame_b64 = None
                    continue
                print(f"    ✗ {shot_dir.name}: 异常 — {e}")
                return None

        return None

    def _process_shot_with_capacity(
        shot_dir: Path,
        chain_source: tuple[str, Path] | None = None,
        chain_allowed: bool = True,
    ) -> dict | None:
        if provider_slots is None or provider_leases is None:
            return _process_shot(shot_dir, chain_source, chain_allowed)
        with provider_slots.reserve(
            "seedance",
            "video",
            shot_dir.name,
            capacity=provider_capacity,
        ):
            lease_task_id = f"{output_dir.resolve()}:{shot_dir.name}"
            with provider_leases.reserve(
                "seedance",
                "video",
                lease_task_id,
                capacity=provider_capacity,
            ):
                return _process_shot(shot_dir, chain_source, chain_allowed)

    # --- 执行模式：串行或并发 ---
    if concurrency == 1:
        # 串行模式（默认，保持原有逻辑和状态更新）
        chain_source = None
        chain_allowed = True
        for shot_dir in pending_shot_dirs:
            result = _process_shot_with_capacity(
                shot_dir, chain_source, chain_allowed
            )
            if result:
                outputs.append(result["relative_output"])
            if chain_mode:
                actual_model = result.get("actual_model") if result else None
                if actual_model == "wan22":
                    print(f"    [chain] {shot_dir.name}: 降级 Wan2.2，接力链中断，后续镜头回退独立首帧", flush=True)
                    chain_allowed = False
                    chain_source = None
                else:
                    last_frame_path = result.get("last_frame_path") if result else None
                    chain_source = (
                        (shot_dir.name, Path(last_frame_path))
                        if last_frame_path and Path(last_frame_path).exists()
                        else None
                    )
            prev_shot_dir = shot_dir
    else:
        # 并发模式（VIDEO_GEN_CONCURRENCY > 1）
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_shot_with_capacity, shot_dir): shot_dir
                for shot_dir in pending_shot_dirs
            }
            for future in as_completed(futures):
                shot_dir = futures[future]
                try:
                    result = future.result()
                    if result:
                        outputs.append(result["relative_output"])
                except Exception as e:
                    print(f"    ✗ {shot_dir.name}: 并发处理异常 — {e}")

    provider = "local_video_client" if use_local else "seedance_client"

    # --- Bug 3 fix: detect shots with missing/invalid output.mp4 ---
    errors = []
    missing_shots = []
    for sd in shot_dirs:
        out_mp4 = sd / "output.mp4"
        if not out_mp4.exists():
            missing_shots.append(sd.name)
            errors.append({"shot": sd.name, "error": "output.mp4 missing after Phase 6"})
        elif out_mp4.stat().st_size < 10 * 1024:
            missing_shots.append(sd.name)
            errors.append({"shot": sd.name, "error": f"output.mp4 too small ({out_mp4.stat().st_size} bytes) after Phase 6"})
    if missing_shots:
        print(f"  ⚠ Phase 6 部分镜头无产出: {', '.join(missing_shots)}")

    return {
        "status": "error" if missing_shots or not outputs else "done",
        "outputs": outputs,
        "errors": errors,
        "missing_shots": missing_shots,
        "error": (
            "Phase 6 missing required shot outputs: " + ", ".join(missing_shots)
            if missing_shots
            else "Phase 6 produced no videos"
            if not outputs
            else None
        ),
        "provider": provider,
        "mode": "text_to_video",
    }


class _PipelineVideoTool(BaseTool):
    """BaseTool-conforming wrapper around the pipeline's video generator."""
    name = "pipeline_video_generation"
    runtime = ToolRuntime.API
    capabilities = ["i2v", "flf2v"]
    input_schema = {"output_dir": "path"}

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = _now()
        try:
            data = _run_phase6_fallback(
                Path(inputs["output_dir"]),
                chain_mode=bool(inputs.get("chain_mode", False)),
            )
            return ToolResult(data.get("status") == "done", data=data, error=data.get("error"), duration_seconds=_elapsed(started))
        except Exception as exc:
            return ToolResult(False, error=str(exc), duration_seconds=_elapsed(started))


class _LocalVideoVendorAdapter(VendorAdapter):
    id = "honcut-local-video"
    name = "HonCut Local Video"

    def video_request(self, config: dict[str, Any], model: VendorModel) -> Any:
        return _PipelineVideoTool().execute(config)


def run_phase6(storyboard_data: dict, output_dir: Path, dry_run: bool, chain_mode: bool = False) -> dict:
    """Phase 6: video generation through the configured provider route."""
    _banner(6, 9, "视频生成 (Seedance — reference_to_video)", dry_run)
    start = _now()
    
    # Estimate based on shot count
    _num_shots = len(storyboard_data.get("shots", [])) if storyboard_data else 0
    if _num_shots == 0:
        # 如果没有 storyboard_data，使用默认值（但不写死 10）
        _num_shots = 5  # 合理的默认值，实际会根据剧本长度计算
    _p5_est = estimate_phase_duration("phase6", num_shots=_num_shots)
    print(f"  ⏱ Phase 6 开始 (预估 ~{int(_p5_est)}s, {_num_shots} 镜头)")

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    print("  → Phase 6 使用当前配置的视频提供方路由", flush=True)
    try:
        adapter = _LocalVideoVendorAdapter([
            VendorModel("Local Bridge", "local-video-bridge", "video", ("i2v", "flf2v"))
        ])
        tool_result = adapter.request(
            "local-video-bridge",
            {"output_dir": str(output_dir), "chain_mode": chain_mode},
        )
        result = tool_result.data or {"status": "error", "error": tool_result.error}
        result["duration_s"] = _elapsed(start)
        provider = result.get("provider", "unknown_provider")
        if result["status"] == "done":
            print(f"  ✓ Phase 6 完成: {len(result['outputs'])} 视频 ({provider})")
            
            # Quality gate: Phase 6
            qg_report = run_quality_check("phase6", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 6 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
        else:
            print(f"  ✗ Phase 6 失败 ({provider})")
        return result

    except ImportError as e:
        return {"status": "error", "error": f"All video generation methods unavailable: {e}", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 7: 一致性守卫 + 场景变化检测 + 幻灯片风险评分
def run_phase7(output_dir: Path, dry_run: bool, storyboard_data: dict = None) -> dict:
    """Phase 7: consistency_guard + scene_variation_check + slideshow_risk_score
    
    集成三个质检模块：
    1. consistency_guard — 角色一致性检查与修复
    2. scene_variation_check — 场景变化检测（抄自 OM variation_checker）
    3. slideshow_risk_score — 幻灯片风险评分（抄自 OM slideshow_risk）
    """
    _banner(7, 9, "一致性守卫 + 场景变化检测 + 幻灯片风险评分", dry_run)
    start = _now()
    _p6_est = estimate_phase_duration("phase7")
    print(f"  ⏱ Phase 7 开始 (预估 ~{int(_p6_est)}s)")
    output_dir = Path(output_dir)
    outputs = []
    consistency_result = None

    if dry_run:
        print("  ⊘ dry-run 模式，跳过一致性检查")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # --- 1. consistency_guard (原有逻辑) ---
    try:
        from quality.consistency_guard import run_consistency_check

        print("  → run_consistency_check: 检查角色一致性...")
        result = run_consistency_check(output_dir=output_dir)
        consistency_result = result
        outputs.append("consistency_report.json")
        print(f"  ✓ 角色一致性检查完成")

    except ImportError as e:
        print(f"  ⚠ consistency_guard 不可用: {e}")
        consistency_result = {"passed": False, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        print(f"  ⚠ 角色一致性检查失败: {e}")
        consistency_result = {"passed": False, "error": str(e)}

    # --- 2. scene_variation_check (新增，抄自 OM variation_checker) ---
    if storyboard_data:
        print("  → scene_variation_check: 检查场景变化...")
        scenes = storyboard_data.get("shots", [])
        if scenes:
            variation_result = check_scene_variation(scenes)
            variation_report_path = output_dir / "variation_report.json"
            variation_report_path.write_text(json.dumps(variation_result, ensure_ascii=False, indent=2))
            outputs.append("variation_report.json")
            
            print(f"    评分: {variation_result['score']}/5.0 ({variation_result['verdict']})")
            if variation_result["violations"]:
                print(f"    发现 {len(variation_result['violations'])} 个问题:")
                for v in variation_result["violations"][:3]:
                    print(f"      - {v}")
            else:
                print(f"    ✓ 场景变化良好")
        else:
            print(f"  ⊘ 无场景数据，跳过场景变化检测")
    else:
        print(f"  ⊘ 无 storyboard_data，跳过场景变化检测")

    # --- 3. slideshow_risk_score (新增，抄自 OM slideshow_risk) ---
    if storyboard_data:
        print("  → slideshow_risk_score: 评估幻灯片风险...")
        scenes = storyboard_data.get("shots", [])
        if scenes:
            slideshow_result = score_slideshow_risk(scenes)
            slideshow_report_path = output_dir / "slideshow_risk_report.json"
            slideshow_report_path.write_text(json.dumps(slideshow_result, ensure_ascii=False, indent=2))
            outputs.append("slideshow_risk_report.json")
            
            print(f"    平均评分: {slideshow_result['average']}/5.0 ({slideshow_result['verdict']})")
            print(f"    维度评分:")
            for dim_name, dim_data in slideshow_result["dimensions"].items():
                print(f"      - {dim_name}: {dim_data['score']}/5.0 — {dim_data['reason']}")
        else:
            print(f"  ⊘ 无场景数据，跳过幻灯片风险评分")
    else:
        print(f"  ⊘ 无 storyboard_data，跳过幻灯片风险评分")

    if not consistency_result or not consistency_result.get("passed", False):
        score = (consistency_result or {}).get("consistency_score", 0)
        error = (consistency_result or {}).get("error")
        reason = error or f"角色一致性分数 {score} < 70"
        print(f"  ✗ Phase 7 质检未通过: {reason}")
        return {
            "status": "error",
            "error": reason,
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "consistency_score": score,
        }

    print(f"  ✓ Phase 7 完成")
    
    # 提取质检指标供 quality_gate 使用
    quality_metrics = {}
    if storyboard_data:
        # 从已生成的报告中提取评分
        variation_report_path = output_dir / "variation_report.json"
        slideshow_report_path = output_dir / "slideshow_risk_report.json"
        
        if variation_report_path.exists():
            try:
                variation_data = json.loads(variation_report_path.read_text())
                # variation_score 范围 0-5，需要归一化到 0-1
                quality_metrics["variation_score"] = variation_data.get("score", 5.0)
            except (json.JSONDecodeError, KeyError):
                pass
        
        if slideshow_report_path.exists():
            try:
                slideshow_data = json.loads(slideshow_report_path.read_text())
                # slideshow_risk 的 average 范围 0-5，归一化到 0-1
                avg = slideshow_data.get("average", 0.0)
                quality_metrics["slideshow_risk"] = avg / 5.0
            except (json.JSONDecodeError, KeyError):
                pass
    
    return {
        "status": "done", 
        "duration_s": _elapsed(start), 
        "outputs": outputs,
        **quality_metrics,  # 将质检指标合并到返回值
    }


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
        return phase_result
    if gate["passed"]:
        print(
            f"  ✓ [8.3] 时长闸门通过: {gate['actual_s']:.2f}s / {gate['target_s']:.2f}s",
            flush=True,
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

    deleted: list[str] = []
    for shot in selected:
        video_path = output_dir / "shots" / shot["shot_id"] / "output.mp4"
        if video_path.is_file():
            video_path.unlink()
            deleted.append(shot["shot_id"])
    print(
        f"  🔄 [8.3] 补录第 {reshoot_round + 1}/2 轮: {', '.join(deleted)}；"
        "其余镜头由 Phase 6 自动跳过",
        flush=True,
    )
    storyboard_path = output_dir / "STORYBOARD.json"
    try:
        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        generation = run_phase6(storyboard, output_dir, dry_run=False, chain_mode=chain_mode)
    except Exception as exc:
        print(f"  ⚠⚠ [8.3] 补录调用 Phase 6 失败: {exc}；阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration reshoot could not run Phase 6: {exc}"
        return phase_result
    if generation.get("status") == "error":
        print(f"  ⚠⚠ [8.3] Phase 6 补录失败: {generation.get('error') or generation.get('errors')}", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration reshoot failed in Phase 6: "
            f"{generation.get('error') or generation.get('errors')}"
        )
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
               _reshoot_history: Optional[list[dict]] = None) -> dict:
    """Phase 8: 逐镜质检、裁切/补录闭环与受审组装。"""
    _banner(8, 9, f"组装引擎 (Assembly) — {transition}", dry_run)
    start = _now()
    phase8_estimate = estimate_phase_duration("phase8")
    print(f"  ⏱ Phase 8 开始 (预估 ~{int(phase8_estimate)}s)")
    output_dir = Path(output_dir)
    reshoot_history = list(_reshoot_history or [])

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频组装")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # 收集视频片段和对应的 SHOT_META
    shots_dir = output_dir / "shots"
    clip_paths = []
    shot_metas = []
    if shots_dir.exists():
        for shot_d in sorted(shots_dir.iterdir()):
            if shot_d.is_dir() and shot_d.name.startswith("S"):
                video = shot_d / "output.mp4"
                if video.exists():
                    clip_paths.append(str(video))
                    # Load SHOT_META.json for this shot
                    meta_path = shot_d / "SHOT_META.json"
                    if meta_path.exists():
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            shot_metas.append(json.load(f))
                    else:
                        shot_metas.append({})  # Empty meta if file doesn't exist

    if not clip_paths:
        return {"status": "error", "error": "No video clips found", "duration_s": _elapsed(start)}

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

    # Step 8.2: dense per-shot review with actionable keep/trim/reshoot decisions.
    from phases.phase8.frame_analysis import analyze_shot_frames

    frame_report = analyze_shot_frames(shots_dir, output_dir / "frame_analysis.json")
    reshoot_shots = list(frame_report.get("summary", {}).get("reshoot", []))
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
                "reshoot_history": reshoot_history,
            }

        deleted: list[str] = []
        for shot_id in reshoot_shots:
            video_path = shots_dir / shot_id / "output.mp4"
            if video_path.is_file():
                # A rejected FLF2V result can be caused by an inconsistent
                # generated endpoint (identity/framing drift). Repeating the
                # exact first/last-frame route reproduces the same defect, so
                # use the character-reference route for the reshoot.
                meta_path = shots_dir / shot_id / "SHOT_META.json"
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    meta = {}
                if meta.get("gen_strategy") == "flf2v":
                    meta["gen_strategy"] = "phantom"
                    meta["phase8_reshoot_route_reason"] = (
                        "FLF2V visual QA failure; avoid reusing a possibly "
                        "inconsistent generated endpoint"
                    )
                    meta_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"  ↪ [8.2] {shot_id}: FLF2V 补录改用 Phantom 角色参考路由",
                        flush=True,
                    )
                video_path.unlink()
                deleted.append(shot_id)
        if deleted != reshoot_shots:
            missing = sorted(set(reshoot_shots).difference(deleted))
            return {
                "status": "error",
                "error": f"Phase 8 cannot reshoot missing source clips: {', '.join(missing)}",
                "duration_s": _elapsed(start),
            }

        print(
            f"  🔄 [8.2] 视觉质检补录第 {_reshoot_round + 1}/2 轮: {', '.join(deleted)}",
            flush=True,
        )
        storyboard_path = output_dir / "STORYBOARD.json"
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            generation = run_phase6(
                storyboard, output_dir, dry_run=False, chain_mode=chain_mode
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Phase 8 visual reshoot could not run Phase 6: {exc}",
                "duration_s": _elapsed(start),
            }
        if generation.get("status") == "error":
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
            return {
                "status": "error",
                "error": f"Phase 6 reported success without regenerated clips: {', '.join(missing_outputs)}",
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
            similarities = compute_transition_similarity(embeddings)
            smart_decisions = decide_all_transitions(shot_metas, similarities)
            
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
        shot_name = f"S{i+1:03d}"
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
    reviewed_edit_error = "reviewed edit path did not complete"
    try:
        from phases.phase8.edit_decisions import build_edit_decisions, execute_edit_decisions

        print("  → 构建 reviewed edit_decisions（质检裁切 + 音频归一化）...")
        edit_decisions = build_edit_decisions(
            shots_dir=shots_dir,
            target_width=1920,
            target_height=1080,
            transition_decisions=transition_dicts,
            quality_report=frame_report,
            shot_order=reviewed_order,
            target_duration=target_duration,
            transition_duration=transition_duration,
        )
        print(f"  → 执行 reviewed edit_decisions（{len(edit_decisions['cuts'])} 个片段）...")
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
                "outputs": ["raw_assembly.mp4"],
                "method": "reviewed_edit_decisions",
                "transition": batch_transition,
                "transition_duration": transition_duration,
                "clip_count": len(edit_decisions["cuts"]),
                "transition_selections": selected_transitions or None,
                "edit_decisions_segments": reviewed_edit.get("segments"),
                "frame_analysis": frame_report.get("summary", {}),
                "reshoot_history": reshoot_history,
                "audio_layer": audio_receipt,
            }, output_dir, target_duration, enable_reshoot, transition,
               transition_duration, media_profile, _reshoot_round, reshoot_history,
               chain_mode)
        print(
            f"  ⚠ reviewed edit_decisions 失败: {reviewed_edit.get('error', 'unknown error')}；"
            "降级为 VideoEdit",
            flush=True,
        )
        reviewed_edit_error = str(reviewed_edit.get("error", "unknown error"))
    except Exception as exc:
        print(f"  ⚠ reviewed edit_decisions 异常: {exc}；降级为 VideoEdit", flush=True)
        reviewed_edit_error = str(exc)

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


def _merge_shot_transcripts(sb_shots: list, durations_ms: list[int], shot_transcripts: list[dict]) -> dict:
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
        if duration_ms <= 0 or transcription.get("skipped"):
            # Shot missing output.mp4 — skip caption generation entirely
            shot_entries.append({
                "shot_id": shot.get("shot_id") or f"S{index:02d}",
                "text": "",
                "source": "skipped",
                "start_ms": cumulative_ms,
                "end_ms": cumulative_ms,
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
                "start_ms": round(cumulative_ms + duration_ms * 0.2),
                "end_ms": round(cumulative_ms + duration_ms * 0.8),
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
                words.append({
                    "word": cleaned_word,
                    "start_ms": cumulative_ms + int(item["start_ms"]),
                    "end_ms": cumulative_ms + int(item["end_ms"]),
                    "source": "asr",
                })
            text = " ".join(item["word"] for item in words) if words else asr_text
            if not words and text:
                words = [{
                    "word": text,
                    "start_ms": cumulative_ms,
                    "end_ms": cumulative_ms + duration_ms,
                    "source": "asr",
                }]
            source = "asr"
        else:
            words = []
            text = ""
            source = "none"

        merged_words.extend(words)
        shot_entries.append({
            "shot_id": shot.get("shot_id") or f"S{index:02d}",
            "text": text,
            "source": source,
            "start_ms": cumulative_ms,
            "end_ms": cumulative_ms + duration_ms,
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
    return {
        "text": "".join(entry["text"] for entry in shot_entries if entry["text"]),
        "duration_ms": cumulative_ms,
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
) -> tuple[list[dict], int]:
    """Build the real-audio base and narration tracks for Phase 9."""
    tracks = [{"path": str(raw_video), "role": "music"}]
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
    has_tts = len(tracks) > 1
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
        from tools.audio_pipeline import extract_audio_track

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
            extract_audio_track(str(shot_video), str(wav_path))
            durations_ms.append(round(_probe_shot_duration(shots_dir, index) * 1000))
            transcription = transcribe_audio(str(wav_path))
            shot_transcripts.append(transcription)
            shot_id = _shot.get("shot_id") or _shot.get("id") or f"S{index:02d}"
            receipt = {
                "shot_id": str(shot_id),
                "audio_path": str(wav_path),
                "duration_ms": durations_ms[-1],
                "transcription": transcription,
            }
            (asr_receipts_dir / f"S{index:02d}.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        transcript_data = _merge_shot_transcripts(sb_shots, durations_ms, shot_transcripts)
        transcript_data["asr_summary"] = {
            "shots_submitted": len(shot_transcripts),
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
            f"{summary['shots_with_text']}/{summary['shots_submitted']} 镜有语音, "
            f"{summary['raw_word_segments']} 个原始词段; "
            f"生成 {summary['caption_segments']} 条字幕"
        )

    try:
        from phases.phase9.visual_post import process_visual
        from phases.phase9.rhythm_editor import edit_rhythm

        # Track step statuses for quality gate integrity
        step_status = {}
        storyboard_data = storyboard_data if sb_path_str else None
        if transcript_data is not None:
            step_status["asr_transcription"] = "done"

        # Step 9.1: Audio processing via OM AudioMixer
        bgm_path = None
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
                output_dir, storyboard_data, transcript_data, base_track
            )
            overlay_count = len(tracks) - 1
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
            bgm_path = _detect_bgm(output_dir, storyboard_path)
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
                    scene_hint = "lake_evening"
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
        try:
            edit_rhythm(
                video_path=current_video,
                storyboard_path=sb_path_str,
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

        cmd = [
            "ffmpeg", "-y",
            "-i", final_out,
            "-vf", (
                "setpts=PTS-STARTPTS,"
                f"scale={profile['width']}:{profile['height']},"
                f"fps={profile['fps']}"
            ),
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
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
        _assert_duration_conserved(encode_input_durations, encoded_durations, tolerance_s=1.0)

        # Only promote the encoded artifact after its independent A/V duration
        # assertions pass.  The delivery gate deliberately probes polished.mp4.
        import shutil
        shutil.move(final_encoded, final_out)
        polished_durations = _probe_av_durations(Path(final_out))
        duration_deltas = {
            kind: (None if encode_input_durations[kind] is None or polished_durations[kind] is None
                   else round(abs(polished_durations[kind] - encode_input_durations[kind]), 6))
            for kind in ("video", "audio")
        }
        duration_gate_passed = all(
            encode_input_durations[kind] is None
            or (polished_durations[kind] is not None and duration_deltas[kind] <= 1.0)
            for kind in ("video", "audio")
        )
        requested_duration_delta = (
            None if target_duration is None or polished_durations["video"] is None
            else round(abs(polished_durations["video"] - float(target_duration)), 6)
        )
        if requested_duration_delta is not None:
            duration_gate_passed = duration_gate_passed and requested_duration_delta <= 1.0
        final_duration_gate = {
            "passed": duration_gate_passed,
            "artifact": "polished.mp4",
            "expected": encode_input_durations,
            "actual": polished_durations,
            "absolute_delta_s": duration_deltas,
            "requested_duration_s": target_duration,
            "requested_duration_delta_s": requested_duration_delta,
            "tolerance_s": 1.0,
            "basis": "Phase 9 encode input plus the requested delivery duration",
        }
        (output_dir / "final_duration_gate.json").write_text(
            json.dumps(final_duration_gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _assert_duration_conserved(encode_input_durations, polished_durations, tolerance_s=1.0)
        if not duration_gate_passed:
            raise RuntimeError(
                f"Final duration gate failed: target={target_duration}, actual={polished_durations}"
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
# LangGraph Phase 1: StateGraph + Interrupt + Command + Conditional Edges
# ---------------------------------------------------------------------------

if LANGGRAPH_AVAILABLE:
    class HonCutState(TypedDict):
        """State schema for LangGraph StateGraph pipeline."""
        text: str
        events: list
        characters: list
        storyboard: dict
        storyboard_image: str
        shots: list
        videos: list
        quality_report: dict
        final_video: str
        status: str
        error: str
        # Internal fields (not in original plan but needed for compatibility)
        output_dir: str
        duration: int
        shot_duration: int
        chain_mode: bool
        dry_run: bool
        transition: str
        transition_duration: float
        enable_reshoot: bool
        media_profile: str
        skip_phase: list
        resume: bool
        auto_approve: bool
        # Tracking fields
        phase_results: dict
        retry_count: int
        completed_phases: list

    def build_pipeline_graph(auto_approve: bool = False, reporter: Optional[ProgressReporter] = None):
        """Compatibility facade for the migrated uncompiled workflow builder."""
        from graph.nodes.phase8 import route_after_phase8
        from graph.nodes.phase9 import route_after_phase9
        from graph.workflow import build_workflow

        return build_workflow(
            state_schema=HonCutState,
            nodes={
                "phase1": lambda state: node_phase1(state, reporter=reporter),
                "phase2": node_phase2,
                "phase3": node_phase3,
                "phase4": node_phase4,
                "phase5": node_phase5_quality,
                "phase6_txt2vid": node_phase6_txt2vid,
                "phase6_img2vid": node_phase6_img2vid,
                "phase6_reference": node_phase6_reference,
                "phase7": node_phase7,
                "phase8": node_phase8,
                "phase9": node_phase9,
                "phase9_5": node_phase9_5,
            },
            review_storyboard_node=node_review_storyboard,
            route_phase5=route_phase5,
            quality_gate_router=quality_gate_router,
            route_after_phase8=route_after_phase8,
            route_after_phase9=route_after_phase9,
            auto_approve=auto_approve,
        )

    # --- Node functions (wrappers around existing run_phase* functions) ---
    
    def node_phase1(state: HonCutState, reporter: Optional[ProgressReporter] = None) -> dict:
        """Compatibility facade for the migrated Phase 1 graph node."""
        from graph.nodes.phase1 import phase1_node

        # Resolve the module global at call time so existing monkeypatches of
        # pipeline_core.run_phase1 keep working during the migration.
        return phase1_node(
            state,
            runner=run_phase1,
            reporter=reporter,
            default_shot_duration=AVG_SHOT_DURATION,
        )

    def node_phase2(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 2 graph node."""
        from graph.nodes.phase2 import phase2_node

        # Resolve the module global at call time so existing monkeypatches of
        # pipeline_core.run_phase2 keep working during the migration.
        return phase2_node(state, runner=run_phase2)

    def node_review_storyboard(state: HonCutState) -> dict:
        """Interrupt node: pause for human review of storyboard
        
        注意: Command(goto=...) 在 interrupt 恢复后的行为依赖 LangGraph 版本（>=0.6）。
        如果 reject 不生效，备选方案是在 quality_gate_router 中检测 storyboard_rejected 状态。
        """
        print("\n" + "="*60)
        print("  📋 人工审核节点：请审核分镜故事板")
        print("="*60)
        if state.get("storyboard_image"):
            print(f"  故事板图片: {state['storyboard_image']}")
        
        # Use interrupt() to pause execution
        # User will resume with: graph.invoke(None, config)
        decision = interrupt({
            "type": "review_storyboard",
            "storyboard_image": state.get("storyboard_image"),
            "message": "请审核分镜故事板，确认后继续",
        })
        
        # decision can be "approve" or "reject"
        if decision == "reject":
            # Rollback to Phase 2
            return Command(goto="phase2", update={"status": "storyboard_rejected"})
        
        return {}

    def node_phase3(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 3 graph node."""
        from graph.nodes.phase3 import phase3_node

        # Resolve the module global at call time so existing monkeypatches of
        # pipeline_core.run_phase3 keep working during the migration.
        return phase3_node(state, runner=run_phase3)

    def node_phase4(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 4 graph node."""
        from graph.nodes.phase4 import phase4_node

        # Resolve the module global at call time so existing monkeypatches of
        # pipeline_core.run_phase4 keep working during the migration.
        return phase4_node(state, runner=run_phase4)

    def route_phase5(state: HonCutState) -> str:
        """根据镜头属性路由到不同的 Phase 6 生成器"""
        storyboard = state.get("storyboard", {})
        shots = storyboard.get("shots", [])

        # 统计镜头类型
        has_reference = any(s.get("ref_type") == "reference" for s in shots)
        has_storyboard_image = bool(state.get("storyboard_image"))

        if has_reference:
            return "reference"
        elif has_storyboard_image:
            return "img2vid"
        else:
            return "txt2vid"

    def node_phase5_quality(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 5 graph node."""
        from graph.nodes.phase5 import phase5_node
        from phases.phase5.storyboard_qa_gate import run_storyboard_qa_gate
        from quality.supervision_agent import SupervisionBlockedError

        # Resolve all existing callables at invocation time so monkeypatches
        # remain effective even when the graph was built earlier.
        return phase5_node(
            state,
            qa_runner=run_storyboard_qa_gate,
            supervision_runner=_run_storyboard_supervision,
            supervision_blocked_error=SupervisionBlockedError,
        )

    def node_phase6_txt2vid(state: HonCutState) -> dict:
        """Compatibility facade for the migrated txt2vid node."""
        from graph.nodes.phase6 import phase6_txt2vid_node

        return phase6_txt2vid_node(state, runner=run_phase6)

    def node_phase6_img2vid(state: HonCutState) -> dict:
        """Compatibility facade preserving delegation to txt2vid."""
        from graph.nodes.phase6 import phase6_img2vid_node

        return phase6_img2vid_node(state, txt2vid_node=node_phase6_txt2vid)

    def node_phase6_reference(state: HonCutState) -> dict:
        """Compatibility facade preserving delegation to txt2vid."""
        from graph.nodes.phase6 import phase6_reference_node

        return phase6_reference_node(state, txt2vid_node=node_phase6_txt2vid)

    def node_phase7(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 7 graph node."""
        from graph.nodes.phase7 import phase7_node

        return phase7_node(state, runner=run_phase7)

    def quality_gate_router(state: HonCutState) -> str:
        """Quality gate: decide whether to pass or retry Phase 6"""
        quality = state.get("quality_report", {})
        retry_count = state.get("retry_count", 0)
        
        slideshow_risk = quality.get("slideshow_risk", 0.0)
        variation_score = quality.get("variation_score", 5.0)
        
        # Check if quality is acceptable
        if slideshow_risk > 0.7 or variation_score < 3.0:
            # Quality failed
            if retry_count < 2:
                # Retry Phase 6 (max 2 times)
                print(f"\n  ⚠ 质检不通过 (slideshow_risk={slideshow_risk}, variation={variation_score})")
                print(f"  🔄 回退到 Phase 6 重新生成 (retry {retry_count + 1}/2)")
                return "retry"
            else:
                print(f"\n  ⚠ 质检不通过，但已达最大重试次数，继续执行")
        
        # Quality passed or max retries reached
        return "pass"

    def node_phase8(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 8 graph node."""
        from graph.nodes.phase8 import phase8_node

        return phase8_node(state, runner=run_phase8)

    def node_phase9(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 9 graph node."""
        from graph.nodes.phase9 import phase9_node

        return phase9_node(state, runner=run_phase9)

    def node_phase9_5(state: HonCutState) -> dict:
        """Compatibility facade for the migrated Phase 9.5 graph node."""
        from graph.nodes.final_qa import final_qa_node
        try:
            from quality.video_qa import run_video_qa
        except ImportError:
            run_video_qa = None

        return final_qa_node(state, runner=run_video_qa)

else:
    # Fallback when LangGraph is not available
    def build_pipeline_graph(auto_approve: bool = False, reporter: Optional[ProgressReporter] = None):
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
    resume: bool = False,
    auto_approve: bool = False,
    resume_from: str = None,
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
        transition: Phase 8 转场模式 ("crossfade" | "fade" | "cut")
        transition_duration: Phase 8 转场时长（秒），默认 0.5
        media_profile: 编码配置名称，从 MEDIA_PROFILES 中选择（默认 "1080p"）
        enable_reshoot: 视觉缺陷或时长不足时是否允许调用 Phase 6 补录（默认 True，最多两轮）
        resume: 从检查点恢复，跳过已完成的 Phase

    Returns:
        pipeline_report dict
    """
    skip_phase = list(skip_phase or [])
    output_path = Path(output_dir).resolve()
    _ensure_dir(output_path)

    # --- M6: --resume-from 支持 ---
    if resume_from:
        try:
            from utils.artifact_chain import PHASE_SEQUENCE, can_resume_from, phase_numbers_before
            if resume_from not in PHASE_SEQUENCE:
                raise ValueError(f"未知 Phase: {resume_from}")
            if can_resume_from(resume_from, output_path):
                skip_phase = phase_numbers_before(resume_from)
                print(f"  🔄 [M6] Resume-from {resume_from}: 跳过 {skip_phase}")
            else:
                print(f"  ⚠ [M6] Resume-from {resume_from}: 前置依赖不满足，从头开始")
        except Exception as e:
            print(f"  ⚠ [M6] resume-from 解析失败: {e}")

    # ---- 进度报告系统初始化 ----
    # 编排器为每个 Phase 子进程设置 HONCUT_APPEND_EVENTS=1，跨阶段 events 历史保留。
    reporter = ProgressReporter(
        str(output_path),
        total_phases=len(PHASE_ORDER),
        clear_events=not os.environ.get("HONCUT_APPEND_EVENTS"),
    )

    # --- M6: 产物链（增量）---
    try:
        from utils.artifact_chain import save_checkpoint as save_artifact_checkpoint, can_resume_from, get_resumable_phase
        M6_AVAILABLE = True
    except ImportError:
        M6_AVAILABLE = False

    # ---- Resume: 读取检查点 ----
    completed_phases = set()
    if resume:
        # Try JSON checkpoint first, then fall back to SQLite checkpoint
        completed_phases = set(_get_completed_stages(output_path))
        if not completed_phases:
            # Fallback: try to read completed phases from SQLite checkpoint
            sqlite_state = load_state_from_sqlite(output_path, thread_id="pipeline_run")
            if sqlite_state and isinstance(sqlite_state, dict):
                sqlite_completed = sqlite_state.get("completed_phases", [])
                if sqlite_completed:
                    completed_phases = set(sqlite_completed)
                    print(f"\n  🔄 Resume 模式: 从 SQLite checkpoint 恢复已完成的 Phase: {sorted(completed_phases)}")
                    # Also write a JSON checkpoint so future resume calls can read it
                    # Reconstruct a minimal checkpoint.json from SQLite state
                    phase_results = sqlite_state.get("phase_results", {})
                    for phase_name in completed_phases:
                        phase_result = phase_results.get(phase_name, {"status": "done"})
                        _record_stage_checkpoint(output_path, phase_name, phase_result)
                else:
                    print(f"\n  🔄 Resume 模式: 无检查点，从头开始")
            else:
                print(f"\n  🔄 Resume 模式: 无检查点，从头开始")
        else:
            print(f"\n  🔄 Resume 模式: 跳过已完成的 Phase: {sorted(completed_phases)}")
        
        if completed_phases:
            next_stage = _get_next_stage(output_path)
            if next_stage is None:
                print(f"  ✓ 所有 Phase 已完成，无需重新运行")
                cp = _read_checkpoint(output_path)
                reporter.mark_completed()
                return {
                    "status": "completed",
                    "resumed": True,
                    "completed_phases": sorted(completed_phases),
                    "output_dir": str(output_dir),
                    "timestamp": cp.get("timestamp", "") if cp else "",
                }

    # 读取文本（resume 模式下如果文本未提供，尝试从检查点恢复）
    if text is None and input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    if not text and resume:
        # resume 模式下文本可以不提供（Phase 1 会被跳过如果已完成）
        text = ""
    if not text:
        raise ValueError("必须提供 --text 或 --input 参数")

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
    if auto_approve:
        print("  ⏭️ 自动跳过人工审核节点 (--auto-approve)")
    
    # 打印预估总耗时
    if not dry_run:
        _est = estimate_total(num_characters=3, num_shots=10)  # 默认值，实际运行时会根据数据调整
        print(f"  ⏱ 预估总耗时: {_est['total_human']} (基于历史数据)")
    
    print(f"{'#'*60}")

    # --- LangGraph StateGraph execution path ---
    if LANGGRAPH_AVAILABLE and not skip_phase:
        print(f"\n  🚀 Using LangGraph StateGraph for pipeline execution")
        try:
            # Build the graph
            graph = build_pipeline_graph(auto_approve=auto_approve, reporter=reporter)
            
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
                # P0-3 fix: interrupt nodes require a checkpointer.
                # If auto_approve is False but no SQLite saver available,
                # fall back to auto_approve mode with a warning.
                if not auto_approve:
                    print("  ⚠ interrupt nodes require checkpointer but SQLite unavailable; falling back to --auto-approve mode")
                    # Rebuild graph without interrupt nodes
                    graph = build_pipeline_graph(auto_approve=True, reporter=reporter)
                    if graph is None:
                        raise RuntimeError("Failed to rebuild pipeline graph (auto_approve fallback)")
                app = graph.compile()
            
            # Prepare initial state
            initial_state = {
                "text": text,
                "events": [],
                "characters": [],
                "storyboard": {},
                "storyboard_image": "",
                "shots": [],
                "videos": [],
                "quality_report": {},
                "final_video": "",
                "status": "running",
                "output_dir": str(output_path),
                "duration": duration,
                "shot_duration": shot_duration,
                "chain_mode": chain_mode,
                "dry_run": dry_run,
                "transition": transition,
                "transition_duration": transition_duration,
                "media_profile": media_profile,
                "enable_reshoot": enable_reshoot,
                "skip_phase": skip_phase or [],
                "resume": resume,
                "auto_approve": auto_approve,
                "phase_results": {},
                "retry_count": 0,
                "completed_phases": [],
            }
            
            # Config for threading
            config = {"configurable": {"thread_id": "pipeline_run"}}
            
            # Handle resume: if resuming, try to get existing state
            if resume and checkpointer:
                try:
                    existing_state = app.get_state(config)
                    if existing_state:
                        # Safely check for values attribute
                        state_values = getattr(existing_state, 'values', None)
                        if state_values and isinstance(state_values, dict):
                            print(f"  🔄 Resuming from LangGraph checkpoint")
                            # Merge existing state with initial state
                            for key, value in state_values.items():
                                if key not in ("text", "duration", "dry_run"):  # Don't override CLI args
                                    initial_state[key] = value
                except Exception as e:
                    print(f"  ⚠ Failed to load checkpoint state: {e}")
            
            # Execute the graph
            try:
                final_state = app.invoke(initial_state, config=config)

                # LangGraph versions that implement interrupt() as a returned
                # value do not raise GraphInterrupt.  Treat the marker as a
                # paused run so a process that has exited never leaves a
                # misleading "running" pipeline_report.json behind.
                pending_interrupts = final_state.get("__interrupt__", ())
                if pending_interrupts:
                    print(f"\n  ⏸ Pipeline paused for human review")
                    print(f"  Resume with: python pipeline_runner.py --resume --output-dir {output_dir}")
                    report = {
                        "status": "interrupted",
                        "input_text_length": len(text),
                        "duration_target_s": duration,
                        "dry_run": dry_run,
                        "output_dir": str(output_dir),
                        "resumed": resume,
                        "phases": final_state.get("phase_results", {}),
                        "total_duration_s": _elapsed(total_start),
                        "interrupt_info": str(pending_interrupts),
                        "langgraph": True,
                    }
                    _write_report(report, output_dir)
                    return report

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
                
                if report["status"] in ("completed", "partial"):
                    reporter.mark_completed()
                else:
                    reporter.mark_failed(
                        final_state.get("error") or f"Pipeline ended with status: {report['status']}"
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
                # Interrupt occurred (human review needed)
                print(f"\n  ⏸ Pipeline paused for human review")
                print(f"  Resume with: python pipeline_runner.py --resume --output-dir {output_dir}")
                
                report = {
                    "status": "interrupted",
                    "input_text_length": len(text),
                    "duration_target_s": duration,
                    "dry_run": dry_run,
                    "output_dir": str(output_dir),
                    "resumed": resume,
                    "phases": {},
                    "total_duration_s": _elapsed(total_start),
                    "interrupt_info": str(e),
                    "langgraph": True,
                }
                
                _write_report(report, output_dir)
                return report
                
        except Exception as e:
            print(f"\n  ⚠ LangGraph execution failed: {e}")
            print(f"  Falling back to sequential execution")
            traceback.print_exc()
            # Fall through to sequential execution
    
    # --- Sequential execution (fallback or when skip_phase is used) ---
    if LANGGRAPH_AVAILABLE and not skip_phase:
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
    if 3 in skip_phase:
        report["phases"]["phase3"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase3" in completed_phases:
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
        report["phases"]["phase5"] = {"status": "skipped", "reason": "user-specified"}
    elif storyboard_data is None:
        report["phases"]["phase5"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        from phases.phase5.storyboard_qa_gate import run_storyboard_qa_gate

        reporter.phase_start("phase5", "分镜质检闸门")
        p4_5 = run_storyboard_qa_gate(output_path)
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
            report["status"] = "partial"
        else:
            _record_stage_checkpoint(output_path, "phase6", p5)

    # ---- Phase 7: 一致性守卫 + 场景变化检测 + 幻灯片风险评分 ----
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

        # ---- 质检门：检查 Phase 7 结果，决定是否回退到 Phase 6 ----
        quality_gate_passed = True
        if p6.get("status") != "error" and not dry_run:
            # 从 consistency_report.json 读取角色一致性分数
            consistency_report_path = output_path / "consistency_report.json"
            if consistency_report_path.exists():
                try:
                    consistency_data = json.loads(consistency_report_path.read_text())
                    consistency_score = consistency_data.get("consistency_score", 100)
                    # 阈值：70 分以下视为不通过
                    if consistency_score < 70:
                        quality_gate_passed = False
                        print(f"  ⚠ 质检门未通过: 角色一致性分数 {consistency_score} < 70")
                        print(f"  🔄 建议回退到 Phase 6 重新生成视频")
                except (json.JSONDecodeError, KeyError):
                    pass

        # 兼容旧报告的二次门禁；run_phase7 已经对新执行失败关闭。
        if not quality_gate_passed:
            report["quality_gate"] = {
                "passed": False,
                "reason": "角色一致性分数低于阈值",
                "recommendation": "回退到 Phase 6"
            }
        else:
            report["quality_gate"] = {"passed": True}

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
            qa_report = run_video_qa(
                output_dir,
                storyboard_data=storyboard_data,
                expected_width=None,  # Let QA infer from media profile
                expected_height=None,
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


def _write_report(report: dict, output_dir: Path):
    """写出 pipeline_report.json"""
    output_dir = Path(output_dir)
    report_path = output_dir / "pipeline_report.json"
    # 深拷贝，确保可序列化
    clean = json.loads(json.dumps(report, default=str))
    if clean.get("status") == "completed":
        clean.pop("error", None)
    report_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))
    print(f"\n  📄 报告已写入: {report_path}")


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
    parser.add_argument("--media-profile", type=str, default="1080p",
                        choices=AVAILABLE_PROFILES,
                        help="编码配置（默认 1080p）")
    parser.add_argument("--resume", action="store_true",
                        help="从检查点恢复，跳过已完成的 Phase（读取 output_dir/checkpoint.json）")
    parser.add_argument("--auto-approve", action="store_true",
                        help="自动批准人工审核节点（用于 CI/测试，跳过 interrupt）")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="从指定阶段恢复（如 phase5），跳过之前的阶段")

    args = parser.parse_args()

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
        resume=args.resume,
        auto_approve=args.auto_approve,
        resume_from=args.resume_from,
    )

    sys.exit(0 if report["status"] in ("completed", "partial") else 1)


if __name__ == "__main__":
    main()
