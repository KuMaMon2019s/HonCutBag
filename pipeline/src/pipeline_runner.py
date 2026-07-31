#!/usr/bin/env python3
"""
pipeline_runner.py — Phase 9: 端到端集成脚本
串联 Phase 2-8 所有模块，一键运行完整管线：任意文本 → polished.mp4

Usage:
    python pipeline_runner.py --text "故事文本" --duration 30 --dry-run
    python pipeline_runner.py --input story.txt --duration 60 --dry-run
    python pipeline_runner.py --text "..." --duration 30 --output-dir ./my_project
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, TypedDict, Any

from config import get_api_key
from progress_reporter import ProgressReporter
from quality_gate import run_quality_check
from timing_estimator import estimate_phase_duration, estimate_total, estimate_remaining

# ---------------------------------------------------------------------------
# LangGraph Integration (Phase 1: @task + RetryPolicy, Send fan-out, SqliteSaver)
# ---------------------------------------------------------------------------
try:
    from langgraph.func import task
    from langgraph.types import RetryPolicy, Send, Command, interrupt
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import StateGraph, START, END
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
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE28_DIR = SCRIPT_DIR  # Phase 2,3,8 模块所在
PHASE47_DIR = SCRIPT_DIR.parent.parent / "vendor" / "legacy"
OM_TOOLS_DIR = SCRIPT_DIR.parent.parent / "vendor" / "openmontage"

# 优先加载当前目录（2026-07-28_01/scripts），然后是旧目录
for d in (PHASE28_DIR, PHASE47_DIR, str(OM_TOOLS_DIR)):
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Media Profiles — delegate to OpenMontage lib.media_profiles
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
# Checkpoint — 断点续跑（灵感来自 OpenMontage/lib/checkpoint.py）
# ---------------------------------------------------------------------------
# 格式: {
#   "completed": ["phase2", "phase2_5", ...],
#   "results": {"phase2": {...}, ...},
#   "timestamp": "2026-07-28T..."
# }
# ---------------------------------------------------------------------------

# Phase 顺序定义（用于 resume 时判断哪些已完成）
PHASE_ORDER = ["phase1", "phase2", "phase2_5", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8"]


def _checkpoint_path(output_dir: Path) -> Path:
    """返回 checkpoint.json 路径"""
    return Path(output_dir) / "checkpoint.json"


def _write_checkpoint(output_dir: Path, phase_name: str, result: dict) -> Path:
    """每个 Phase 完成后写入检查点。

    更新 output_dir/checkpoint.json：
    - 将 phase_name 加入 completed 列表（去重）
    - 将 result 存入 results[phase_name]
    - 更新 timestamp

    使用原子写入（tmp + rename）防止写坏。
    
    如果 LangGraph 可用，同时写入 SQLite checkpoint。
    """
    cp_path = _checkpoint_path(output_dir)
    cp_path = Path(cp_path)

    # 读取已有 checkpoint（如果有）
    checkpoint = {"completed": [], "results": {}, "timestamp": ""}
    if cp_path.exists():
        try:
            with open(cp_path, encoding="utf-8") as f:
                checkpoint = json.load(f)
        except (json.JSONDecodeError, OSError):
            checkpoint = {"completed": [], "results": {}, "timestamp": ""}

    # 更新（只保存成功的 Phase）
    status = result.get("status", "")
    if status not in ("done", "skipped"):
        # 失败的 Phase 不写入 checkpoint，避免 resume 时跳过
        return cp_path
    
    if phase_name not in checkpoint["completed"]:
        checkpoint["completed"].append(phase_name)
    # 只存可序列化的 result 摘要（去掉内部大对象）
    safe_result = {}
    for k, v in result.items():
        if k.startswith("_"):
            continue  # 跳过内部数据（如 _storyboard, _characters）
        try:
            json.dumps(v, default=str)
            safe_result[k] = v
        except (TypeError, ValueError):
            safe_result[k] = str(v)
    checkpoint["results"][phase_name] = safe_result
    checkpoint["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    # 原子写入 JSON checkpoint
    tmp_path = cp_path.with_suffix(".json.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    import os
    os.replace(tmp_path, cp_path)

    # 如果 LangGraph 可用，同时写入 SQLite checkpoint
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
            if attempt < max_attempts:
                wait_time = backoff_factor ** (attempt - 1)
                print(f"    ⚠ Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ✗ All {max_attempts} attempts failed. Last error: {e}")
    raise last_error


if LANGGRAPH_AVAILABLE:
    # Define state for LangGraph-based execution (Phase 2 StateGraph migration)
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
        from seedream_client import SeedreamClient
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
        from character_factory import generate_single
        result = generate_single(char_dict, chars_dir, skip_images=skip_images)
        return {"status": "done", "result": result}

    @task(retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0))
    def task_generate_video(shot: dict, storyboard_image: Optional[str], output_dir: str, style_context: Optional[dict] = None) -> dict:
        """⚠️ 此函数为 LangGraph @task 包装，仅在 StateGraph 执行上下文中有效，不可直接调用。
        Task-wrapped video generation with retry (for StateGraph)."""
        from tools.video.seedance_video import SeedanceVideo
        sv = SeedanceVideo()
        
        shot_id = shot.get("id", "?")
        # 标准化 shot_id 格式为零填充（S01, S02, S03）
        if isinstance(shot_id, int):
            shot_id_str = f"S{shot_id:02d}"
        else:
            shot_id_str = f"S{str(shot_id).zfill(2)}"
        prompt = _build_shot_prompt(shot, style_context) or shot.get("prompt", "")
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
        storyboard_image = str(Path(output_dir) / "storyboard.png")
        style_context = None
        
        if state.get("storyboard_data", {}).get("style"):
            style_context = {"mood": state["storyboard_data"]["style"]}
        
        sends = []
        for shot in shots:
            sends.append(Send("generate_video", {
                "shot": shot,
                "storyboard_image": storyboard_image,
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
        """Save pipeline state to JSON file.
        
        P0-4 fix: SQLite checkpoint management is now handled automatically by
        StateGraph.compile(checkpointer=...) during graph execution. This function
        only writes state to JSON for external inspection/debugging.
        """
        try:
            # Write state to JSON for inspection (not for graph checkpoint)
            state_file = output_dir / "state.json"
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"⚠ Failed to save state to JSON: {e}")
            return False

    def load_state_from_sqlite(output_dir: Path, thread_id: str = "default") -> Optional[dict]:
        """Load pipeline state from SQLite checkpoint."""
        try:
            saver = get_sqlite_checkpointer(output_dir)
            if not saver:
                return None
            with saver as checkpointer:  # type: ignore[attr-defined]
                config = {"configurable": {"thread_id": thread_id}}
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

def run_phase1(text: str, output_dir: Path, dry_run: bool) -> dict:
    """Phase 1: 导演规划（M1 增量模块）"""
    _banner("1", 9, "导演规划 (Director Planner)", dry_run)
    start = _now()
    try:
        from director_planner import plan_director
        result = plan_director(text, output_dir, dry_run)
        result["duration_s"] = _elapsed(start)
        return result
    except Exception as e:
        print(f"  ⚠ [M1] Phase 1 降级跳过: {e}")
        return {"status": "skipped", "reason": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 2: 编剧引擎 (text → STORYBOARD.json + CHARACTERS.json)
# ---------------------------------------------------------------------------

def run_phase2(text: str, output_dir: Path, duration: int, dry_run: bool, reporter: Optional[ProgressReporter] = None) -> dict:
    """Phase 2: text_parser → event_extractor → character_discoverer → adaptation_engine → storyboard_generator"""
    _banner(2, 8, "编剧引擎 (Screenwriter)", dry_run)
    start = _now()
    _p2_est = estimate_phase_duration("phase2")
    print(f"  ⏱ Phase 2 开始 (预估 ~{int(_p2_est)}s)")
    output_dir = Path(output_dir)

    try:
        from text_parser import parse_text
    except ImportError as e:
        return {"status": "error", "error": f"Phase 2 import failed: {e}", "duration_s": _elapsed(start)}

    outputs = []
    try:
        # Step 2.1: text_parser → segments list
        print("  → text_parser: 解析文本结构...")
        if reporter:
            reporter.step("phase2", "解析文本结构", progress_pct=10)
        parsed = parse_text(text)
        segments = parsed.get("segments", [])
        print(f"    ✓ 解析出 {len(segments)} 个段落")
        if reporter:
            reporter.step("phase2", f"解析出 {len(segments)} 个段落", progress_pct=20)

        # dry-run 模式：生成模拟数据，不调用 API
        if dry_run:
            print("  ⊘ dry-run 模式，生成模拟数据（跳过 API 调用）...")
            if reporter:
                reporter.step("phase2", "dry-run: 生成模拟事件", progress_pct=30)
            
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
                reporter.step("phase2", f"dry-run: 提取 {len(mock_events)} 个事件", progress_pct=45)
            
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
                reporter.step("phase2", f"dry-run: 发现 {len(mock_characters['characters'])} 个角色", progress_pct=60)
            
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
            
            if reporter:
                reporter.step("phase2", f"dry-run: 生成 {len(mock_storyboard['shots'])} 个分镜", progress_pct=80)
            
            # 写出文件
            storyboard_path = output_dir / "STORYBOARD.json"
            characters_path = output_dir / "CHARACTERS.json"
            events_path = output_dir / "events.json"
            
            storyboard_path.write_text(json.dumps(mock_storyboard, ensure_ascii=False, indent=2))
            characters_path.write_text(json.dumps(mock_characters, ensure_ascii=False, indent=2))
            events_path.write_text(json.dumps({"events": mock_events}, ensure_ascii=False, indent=2))
            
            outputs = ["STORYBOARD.json", "CHARACTERS.json", "events.json"]
            print(f"  ✓ Phase 2 完成 (dry-run): {outputs}")
            
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "_storyboard": mock_storyboard,
                "_characters": mock_characters,
            }
        
        # 正常模式：调用 API
        try:
            from event_extractor import extract_events
            from character_discoverer import discover_characters
            from adaptation_engine import adapt_events
            from storyboard_generator import generate_storyboard
        except ImportError as e:
            return {"status": "error", "error": f"Phase 2 import failed: {e}", "duration_s": _elapsed(start)}

        # Step 2.2: event_extractor → events list
        print("  → event_extractor: 提取事件...")
        if reporter:
            reporter.step("phase2", "提取事件", progress_pct=30)
        events_result = extract_events(segments)
        events = events_result.get("events", [])
        print(f"    ✓ 提取 {len(events)} 个事件")
        if reporter:
            reporter.step("phase2", f"提取 {len(events)} 个事件", progress_pct=40)

        # Step 2.3: character_discoverer → characters dict
        print("  → character_discoverer: 发现角色...")
        if reporter:
            reporter.step("phase2", "发现角色", progress_pct=50)
        characters_result = discover_characters(events)
        characters_list = characters_result.get("characters", [])
        print(f"    ✓ 发现 {len(characters_list)} 个角色")
        if reporter:
            reporter.step("phase2", f"发现 {len(characters_list)} 个角色", progress_pct=60)

        # Step 2.4: adaptation_engine → adapted shots list
        print("  → adaptation_engine: 影视化改编...")
        if reporter:
            reporter.step("phase2", "影视化改编", progress_pct=70)
        adapted = adapt_events(events, characters_list, target_duration=duration)
        adapted_shots = adapted.get("shots", [])
        print(f"    ✓ 改编完成，{len(adapted_shots)} 个镜头")
        if reporter:
            reporter.step("phase2", f"改编完成，{len(adapted_shots)} 个镜头", progress_pct=80)

        # Step 2.5: storyboard_generator → storyboard dict
        print("  → storyboard_generator: 生成分镜...")
        if reporter:
            reporter.step("phase2", "生成分镜", progress_pct=90)
        storyboard = generate_storyboard(adapted_shots, characters_list)

        # 写出文件
        storyboard_path = output_dir / "STORYBOARD.json"
        characters_path = output_dir / "CHARACTERS.json"

        storyboard_path.write_text(json.dumps(storyboard, ensure_ascii=False, indent=2))
        characters_path.write_text(json.dumps(characters_result, ensure_ascii=False, indent=2))

        outputs = ["STORYBOARD.json", "CHARACTERS.json"]
        print(f"  ✓ Phase 2 完成: {outputs}")

        # Quality gate: Phase 2
        qg_report = run_quality_check("phase2", output_dir, {
            "events": storyboard.get("events", []),
            "shots": storyboard.get("shots", []),
        })
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 2 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start), "outputs": outputs}

        # --- M5: 监督层审核（增量，失败不影响后续）---
        try:
            from quality_gate import run_storyboard_review
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

        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "_storyboard": storyboard,
            "_characters": characters_result,
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start), "outputs": outputs}


# ---------------------------------------------------------------------------
# Phase 2.5: 故事板图片生成 (OM image_selector)
# ---------------------------------------------------------------------------

def load_storyboard_prompt_techniques() -> str:
    """加载分镜提示词技巧（来源: ToonFlow storyboard_prompt_techniques.md）

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
    集成 ToonFlow 分镜提示词技巧（镜头语言、构图规则、画质控制）。

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
        scene = shot.get("scene", shot.get("description", ""))
        action = shot.get("action", "")
        camera = shot.get("camera", "")
        line = f"面板 {i}: {scene}"
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

    # 追加 ToonFlow 分镜提示词技巧参考
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


def run_phase2_5(storyboard_data: dict, characters_data: dict, output_dir: Path, dry_run: bool) -> dict:
    """Phase 2.5: 使用 OM image_selector 生成故事板图片，不可用时降级到 Seedream API"""
    _banner("2.5", 8, "故事板图片生成 (ImageSelector / Seedream)", dry_run)
    start = _now()
    _p25_est = estimate_phase_duration("phase2_5")
    print(f"  ⏱ Phase 2.5 开始 (预估 ~{int(_p25_est)}s)")
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过故事板图片生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # 1. 加载模板
    template_path = SCRIPT_DIR.parent / "prompts" / "storyboard_template.md"
    if not template_path.exists():
        return {"status": "error", "error": f"storyboard_template.md not found at {template_path}", "duration_s": _elapsed(start)}

    template = template_path.read_text(encoding="utf-8")

    # 2. 填充模板
    prompt = fill_storyboard_template(template, storyboard_data, characters_data)
    print(f"  → 提示词已生成 ({len(prompt)} 字符)")

    storyboard_path = output_dir / "storyboard.png"
    om_error = None

    # 3. 尝试调用 OM image_selector
    try:
        from tools.graphics.image_selector import ImageSelector
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
                print(f"  ✓ Phase 2.5 完成: storyboard.png (provider: OM)")
                
                # Quality gate: Phase 2.5
                qg_report = run_quality_check("phase2_5", output_dir)
                if not qg_report.passed:
                    return {"status": "error", "error": f"Phase 2.5 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
                
                # --- M2: 分镜图序列（每镜头一张）---
                try:
                    storyboard_images_dir = output_dir / "storyboard_images"
                    storyboard_images_dir.mkdir(exist_ok=True)
                    shots = storyboard_data.get("shots", [])
                    generated_count = 0
                    for shot in shots:
                        shot_id = shot.get("shot_id", f"S{shot.get('shot_order', 0):02d}")
                        shot_prompt = shot.get("prompt", shot.get("visual", ""))
                        if not shot_prompt:
                            continue
                        shot_image_path = storyboard_images_dir / f"{shot_id}.png"
                        if shot_image_path.exists():
                            generated_count += 1
                            continue
                        try:
                            # 复用现有 seedream_client
                            from seedream_client import text_to_image
                            text_to_image(prompt=shot_prompt, output_path=str(shot_image_path))
                            generated_count += 1
                            print(f"    [M2] 分镜图 {shot_id}.png ✓")
                        except Exception as e:
                            print(f"    [M2] 分镜图 {shot_id}.png 失败（降级跳过）: {e}")
                    print(f"  → [M2] 分镜图序列: {generated_count}/{len(shots)} 张")
                except Exception as e:
                    print(f"  ⚠ [M2] 分镜图序列生成失败（降级跳过）: {e}")

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
                    print(f"  ✓ Phase 2.5 完成: storyboard.png (provider: OM)")
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
        from seedream_client import SeedreamClient
        client = SeedreamClient()
        print(f"  → seedream: 生成故事板图片 (1920x1920, timeout=180s, retry=3)...")

        # Use retry policy for API call
        def _call_seedream():
            client.text_to_image(
                prompt=prompt,
                output_path=str(storyboard_path),
                size="1920x1920",
                timeout=180,
            )
            if not storyboard_path.exists():
                raise RuntimeError("Seedream 调用成功但未生成文件")
        
        _retry_with_policy(_call_seedream, max_attempts=3, backoff_factor=2.0)

        if storyboard_path.exists():
            print(f"  ✓ Phase 2.5 完成: storyboard.png (provider: Seedream, fallback from OM: {om_error})")
            
            # Quality gate: Phase 2.5
            qg_report = run_quality_check("phase2_5", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 2.5 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
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
    """检测衍生资产（来源: ToonFlow derive_assets 方法论）

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
    _banner(3, 8, "角色工厂 (Character Factory + Derive Assets)", dry_run)
    start = _now()
    outputs = []
    output_dir = Path(output_dir)

    try:
        from character_factory import batch_generate

        chars_dir = _ensure_dir(output_dir / "characters")
        characters_list = characters_data.get("characters", [])

        if not characters_list:
            print("  ⊘ 无角色数据，跳过")
            return {"status": "skipped", "reason": "no characters", "duration_s": _elapsed(start)}

        # Step 3.1: 衍生资产检测（ToonFlow 方法论）
        print("  → 检测衍生资产（变身/换装状态）...")
        derive_assets = detect_derive_assets(characters_data)
        if derive_assets:
            print(f"    ✓ 检测到 {len(derive_assets)} 个衍生资产:")
            for da in derive_assets:
                print(f"      - {da['parent_name']}·{da['name']}: {da['desc']}")

        # Step 3.2: 生成基础角色三视图
        # 为每个角色准备 id/name/description
        char_dicts = []
        for i, c in enumerate(characters_list):
            char_dicts.append({
                "id": c.get("id", f"char_{i}"),
                "name": c.get("name", f"角色{i}"),
                "description": c.get("appearance", {}).get("summary", c.get("description", "")),
                "appearance": c.get("appearance", {}),  # 传递完整 appearance dict
                "style": c.get("style", ""),
            })

        _p3_est = estimate_phase_duration("phase3", num_characters=len(char_dicts))
        print(f"  ⏱ Phase 3 开始 (预估 ~{int(_p3_est)}s)")
        print(f"  → batch_generate: {len(char_dicts)} 个角色, skip_images={dry_run}")
        
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
    _banner(4, 8, "编排器 (Orchestrator)", dry_run)
    start = _now()
    _p4_est = estimate_phase_duration("phase4")
    print(f"  ⏱ Phase 4 开始 (预估 ~{int(_p4_est)}s)")
    outputs = []
    output_dir = Path(output_dir)

    storyboard_path = output_dir / "STORYBOARD.json"
    if not storyboard_path.exists():
        return {"status": "error", "error": "STORYBOARD.json not found", "duration_s": _elapsed(start)}

    try:
        orchestrator_script = PHASE47_DIR / "orchestrator.py"
        if not orchestrator_script.exists():
            return {"status": "error", "error": f"orchestrator.py not found at {orchestrator_script}", "duration_s": _elapsed(start)}

        shots_dir = output_dir / "shots"
        cmd = [
            sys.executable, str(orchestrator_script),
            "--storyboard", str(storyboard_path.resolve()),
            "--skip-assembly",
            "--shots-dir", str(shots_dir.resolve()),
        ]
        if dry_run:
            cmd.append("--dry-run")

        print(f"  → orchestrator: {' '.join(cmd[-4:])}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PHASE47_DIR),
        )

        if result.returncode != 0:
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

        print(f"  ✓ Phase 4 完成: {len(outputs)} 镜头目录")
        status = "done" if outputs or dry_run else "error"
        return {"status": status, "duration_s": _elapsed(start), "outputs": outputs or ["shots/"]}

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "orchestrator timed out", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# OM build_shot_prompt — 标准化提示词构建（抄自 OpenMontage/lib/shot_prompt_builder.py）
# ---------------------------------------------------------------------------

_SHOT_SIZE_PHRASES = {
    "extreme_wide": "extreme wide shot showing vast environment",
    "wide": "wide shot capturing full scene",
    "medium_wide": "medium-wide shot framing subject with surroundings",
    "medium": "medium shot from waist up",
    "medium_close": "medium close-up from chest up",
    "close_up": "close-up focusing on face or detail",
    "extreme_close_up": "extreme close-up on fine detail",
    "over_shoulder": "over-the-shoulder perspective",
    "insert": "insert shot of specific detail",
    "establishing": "establishing shot setting the location",
}

_MOVEMENT_PHRASES = {
    "static": "locked-off static camera",
    "pan_left": "smooth pan to the left",
    "pan_right": "smooth pan to the right",
    "tilt_up": "gentle tilt upward",
    "tilt_down": "gentle tilt downward",
    "dolly_in": "slow dolly in toward subject",
    "dolly_out": "slow dolly out from subject",
    "tracking_left": "tracking shot moving left alongside subject",
    "tracking_right": "tracking shot moving right alongside subject",
    "crane_up": "crane shot rising upward",
    "crane_down": "crane shot descending",
    "handheld": "handheld camera with natural movement",
    "steadicam": "smooth steadicam following movement",
    "whip_pan": "fast whip pan",
    "orbital": "orbital camera circling subject",
    "zoom_in": "slow zoom in",
    "zoom_out": "slow zoom out",
    "rack_focus": "rack focus shift between foreground and background",
}

_LIGHTING_PHRASES = {
    "high_key": "bright high-key lighting, minimal shadows",
    "low_key": "dramatic low-key lighting with deep shadows",
    "natural": "natural ambient lighting",
    "golden_hour": "warm golden hour sunlight",
    "blue_hour": "cool blue hour twilight",
    "tungsten_warm": "warm tungsten interior lighting",
    "neon": "neon-lit with vibrant color spill",
    "silhouette": "backlit silhouette",
    "rim_lit": "rim lighting highlighting edges",
    "volumetric": "volumetric light with visible rays",
    "overcast_soft": "soft overcast diffused light",
}

_DOF_PHRASES = {
    "shallow": "shallow depth of field with bokeh",
    "medium": "medium depth of field",
    "deep": "deep focus with everything sharp",
}

_COLOR_TEMP_PHRASES = {
    "cool": "cool blue-toned color palette",
    "neutral": "neutral balanced colors",
    "warm": "warm amber-toned color palette",
    "mixed": "mixed color temperatures for contrast",
}


def _build_shot_prompt(shot: dict, style_context: Optional[dict] = None) -> str:
    """Build standardized prompt from shot metadata (OM 5-layer framework).

    Layers:
      1. Camera (lens, depth of field)
      2. Movement (shot size, camera movement)
      3. Subject (description + texture keywords)
      4. Lighting (lighting key, color temperature)
      5. Style (from style_context)
    """
    sl = shot.get("shot_language", {}) or {}
    layers: list[str] = []

    # Layer 1: Camera
    camera_parts = []
    if sl.get("lens_mm"):
        camera_parts.append(f"{sl['lens_mm']}mm lens")
    if sl.get("depth_of_field"):
        camera_parts.append(_DOF_PHRASES.get(sl["depth_of_field"], ""))
    if camera_parts:
        layers.append(", ".join(filter(None, camera_parts)))

    # Layer 2: Movement
    movement_parts = []
    if sl.get("shot_size"):
        movement_parts.append(_SHOT_SIZE_PHRASES.get(sl["shot_size"], sl["shot_size"]))
    if sl.get("camera_movement") and sl["camera_movement"] != "static":
        movement_parts.append(_MOVEMENT_PHRASES.get(sl["camera_movement"], sl["camera_movement"]))
    if movement_parts:
        layers.append(", ".join(movement_parts))

    # Layer 3: Subject
    description = shot.get("description", "") or shot.get("scene", "") or shot.get("prompt", "")
    texture = shot.get("texture_keywords", []) or []
    subject_parts = [description]
    if texture:
        subject_parts.append(", ".join(texture) if isinstance(texture, list) else str(texture))
    action = shot.get("action", "")
    if action:
        subject_parts.append(f"action: {action}")
    layers.append(". ".join(filter(None, subject_parts)))

    # Layer 4: Lighting
    lighting_parts = []
    if sl.get("lighting_key"):
        lighting_parts.append(_LIGHTING_PHRASES.get(sl["lighting_key"], sl["lighting_key"]))
    if sl.get("color_temperature"):
        lighting_parts.append(_COLOR_TEMP_PHRASES.get(sl["color_temperature"], ""))
    # Fallback: use shot-level emotion as lighting hint
    if not lighting_parts and shot.get("emotion"):
        emotion = shot["emotion"]
        emotion_map = {
            "紧张": "tense dramatic lighting",
            "激动": "energetic vibrant lighting",
            "坚定": "determined strong lighting",
            "悲伤": "melancholic muted lighting",
            "快乐": "bright cheerful lighting",
        }
        lighting_parts.append(emotion_map.get(emotion, f"{emotion} mood"))
    if lighting_parts:
        layers.append(", ".join(filter(None, lighting_parts)))

    # Layer 5: Style
    if style_context:
        mood = style_context.get("mood", "")
        visual_lang = style_context.get("visual_language", {}) or {}
        style_hint = visual_lang.get("aesthetic", "") or mood
        if style_hint:
            layers.append(f"Style: {style_hint}")
    # Fallback: use storyboard-level style
    elif shot.get("style"):
        layers.append(f"Style: {shot['style']}")

    return ". ".join(filter(None, layers))


# ---------------------------------------------------------------------------
# Phase 5: 视频生成 (OM SeedanceVideo — reference_to_video)
# ---------------------------------------------------------------------------

def _run_phase5_om_seedance(storyboard_data: dict, output_dir: Path, characters_data: Optional[dict] = None, _timing_ctx: Optional[dict] = None) -> dict:
    """使用 OM SeedanceVideo 生成视频（支持 reference_to_video）
    
    Args:
        storyboard_data: STORYBOARD.json 的内容
        output_dir: 输出目录
        characters_data: CHARACTERS.json 的内容（可选，用于注入角色参考图）
        _timing_ctx: 可选计时上下文 {start, estimate}，用于打印子节点进度
    """
    from tools.video.seedance_video import SeedanceVideo

    sv = SeedanceVideo()

    # 检查工具是否可用
    status = sv.get_status()
    if status.value != "available":
        raise ImportError(f"SeedanceVideo not available (status={status})")

    shots = storyboard_data.get("shots", [])
    storyboard_image = output_dir / "storyboard.png"
    has_storyboard_image = storyboard_image.exists()

    # 构建角色参考图映射：character_id -> front.png 路径
    character_ref_images = {}
    if characters_data:
        characters = characters_data.get("characters", [])
        for char in characters:
            char_id = char.get("id", "")
            char_name = char.get("name", "")
            # 查找角色目录中的 front.png
            char_dir = output_dir / "characters" / char_id
            front_png = char_dir / "front.png"
            if front_png.exists():
                character_ref_images[char_id] = str(front_png)
                character_ref_images[char_name] = str(front_png)  # 也支持按名称匹配
                print(f"  ✓ 角色参考图: {char_name} -> {front_png.name}")

    if has_storyboard_image:
        print(f"  → 模式: reference_to_video (storyboard.png 存在)")
    else:
        print(f"  → 模式: text_to_video (storyboard.png 不存在)")

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
        prompt = _build_shot_prompt(shot, style_context)
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
            
            # 确定参考图：角色参考图 > storyboard.png
            reference_image = character_ref if character_ref else (str(storyboard_image) if has_storyboard_image else None)
            
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
        "mode": "reference_to_video" if has_storyboard_image else "text_to_video",
    }


def _run_phase5_fallback(output_dir: Path) -> dict:
    """降级：使用手写的 seedance_client"""
    output_dir = Path(output_dir)
    from seedance_client import submit, poll, download

    shots_dir = output_dir / "shots"
    if not shots_dir.exists():
        return {"status": "skipped", "reason": "no shots directory"}

    api_key = get_api_key("ARK_AGENT_API_KEY") or os.environ.get("ARK_AGENT_API_KEY", "")
    if not api_key:
        return {"status": "error", "error": "ARK_AGENT_API_KEY not set"}

    print("  → 模式: text_to_video (seedance_client fallback)")

    # Load character reference images for consistency
    import base64 as _b64
    char_ref_map = {}   # {match_key_lower: base64_of_front_png}
    char_list = []      # [(char_id, char_name, b64)] for fallback
    chars_path = output_dir / "CHARACTERS.json"
    if chars_path.exists():
        chars_data = json.loads(chars_path.read_text())
        for char in chars_data.get("characters", []):
            # Try both directory structures: characters/{id}/ and characters/characters/{id}/
            front_png = output_dir / "characters" / char["id"] / "front.png"
            if not front_png.exists():
                front_png = output_dir / "characters" / "characters" / char["id"] / "front.png"
            if front_png.exists():
                b64 = _b64.b64encode(front_png.read_bytes()).decode()
                # Map multiple keys for matching: Chinese name, pinyin id, id without underscores
                char_ref_map[char["name"].lower()] = b64
                char_ref_map[char["id"].lower()] = b64
                char_ref_map[char["id"].replace("_", "").lower()] = b64
                char_list.append((char["id"], char["name"], b64))
        if char_ref_map:
            print(f"  → 已加载 {len(char_list)} 个角色参考图")

    # Determine protagonist (first character) for default injection
    protagonist_b64 = char_list[0][2] if char_list else None
    protagonist_name = char_list[0][1] if char_list else None

    # Load storyboard image as style reference
    storyboard_b64 = None
    storyboard_path = output_dir / "storyboard.png"
    if storyboard_path.exists() and storyboard_path.stat().st_size > 10240:
        storyboard_b64 = _b64.b64encode(storyboard_path.read_bytes()).decode()
        print(f"  → 已加载故事板风格参考图")

    outputs = []
    for shot_dir in sorted(shots_dir.iterdir()):
        if not shot_dir.is_dir() or not shot_dir.name.startswith("S"):
            continue
        meta_path = shot_dir / "SHOT_META.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        prompt = meta.get("prompt", "")
        # --- M4: 模型路由（增量，失败用原始 prompt）---
        try:
            from prompt_router import route_prompt
            model_name = os.environ.get("SEEDANCE_MODEL", "seedance-2-0-mini")
            routed_prompt = route_prompt(
                model_name=model_name,
                mode="single_shot",
                shot_data=meta,
                assets=[{"name": c.get("name", ""), "description": c.get("appearance", {}).get("summary", "")} 
                        for c in (json.loads((output_dir / "CHARACTERS.json").read_text()).get("characters", [])
                                  if (output_dir / "CHARACTERS.json").exists() else [])]
            )
            if routed_prompt:
                prompt = routed_prompt
                print(f"    [M4] 提示词路由: {model_name} → single_shot")
        except Exception as e:
            pass  # 降级用原始 prompt
        duration = meta.get("duration", 5)  # 从 SHOT_META 读取 duration
        if not prompt:
            continue

        # Find character reference for this shot
        # Strategy: match by name/id in prompt → fallback to protagonist
        first_frame_b64 = None
        prompt_lower = prompt.lower()
        for char_name, b64 in char_ref_map.items():
            if char_name in prompt_lower:
                first_frame_b64 = b64
                print(f"    [ref] 注入角色参考: {char_name}")
                break
        # Fallback: inject protagonist for shots with human activity keywords
        if first_frame_b64 is None and protagonist_b64:
            human_keywords = ["woman", "man", "girl", "boy", "person", "she", "he",
                              "her", "his", "lin xia", "shen yu", "xia", "yu"]
            if any(kw in prompt_lower for kw in human_keywords):
                first_frame_b64 = protagonist_b64
                print(f"    [ref] 注入主角参考 (fallback): {protagonist_name}")

        # If still no reference, use storyboard as style reference
        if first_frame_b64 is None and storyboard_b64:
            first_frame_b64 = storyboard_b64
            print(f"    [ref] 注入故事板风格参考")

        # --- M2: 分镜图构图参考（优先级低于角色参考图）---
        if first_frame_b64 is None:
            shot_image = output_dir / "storyboard_images" / f"{shot_dir.name}.png"
            if shot_image.exists() and shot_image.stat().st_size > 1024:
                first_frame_b64 = _b64.b64encode(shot_image.read_bytes()).decode()
                print(f"    [M2] 注入分镜图构图参考: {shot_dir.name}.png")

        # Add style prefix to every prompt
        style_prefix = "Photorealistic, urban, Jiangnan aesthetic, lifelike skin texture, natural lighting, "
        if not prompt.lower().startswith("photorealistic"):
            prompt = style_prefix + prompt

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                print(f"  → {shot_dir.name}: 提交视频生成...")
                task_id = submit(prompt=prompt, api_key=api_key, duration=duration, ratio="16:9",
                                 reference_image_base64=first_frame_b64)
                video_url = poll(task_id, api_key=api_key)
                if video_url:
                    out_path = str(shot_dir / "output.mp4")
                    download(video_url, out_path)
                    outputs.append(f"shots/{shot_dir.name}/output.mp4")
                    print(f"    ✓ {shot_dir.name}: 视频已生成")
                    break
                else:
                    print(f"    ✗ {shot_dir.name}: poll 返回空 URL")
                    break
            except Exception as e:
                err_str = str(e)
                # 429 QuotaExceeded — exponential backoff retry
                if "QuotaExceeded" in err_str or "429" in err_str:
                    wait_sec = min(30 * (2 ** attempt), 120)
                    if attempt < max_retries:
                        print(f"    ⚠ {shot_dir.name}: 配额超限(429)，等待 {wait_sec}s 后重试 ({attempt+1}/{max_retries})...")
                        import time as _time
                        _time.sleep(wait_sec)
                        continue
                    else:
                        print(f"    ✗ {shot_dir.name}: 配额超限，已重试 {max_retries} 次，跳过")
                        break
                if "PrivacyInformation" in err_str and first_frame_b64 is not None:
                    # Seedance rejects real-person reference images — drop and retry text-only
                    print(f"    ⚠ {shot_dir.name}: 参考图被隐私检测拒绝，降级为纯文本生成")
                    first_frame_b64 = None
                    continue
                if "PolicyViolation" in err_str and attempt < max_retries:
                    print(f"    ⚠ {shot_dir.name}: 版权误报，重试 ({attempt+1}/{max_retries})...")
                    prompt = prompt.replace("Cinematic", "Original fictional")
                    prompt += ", original character design, non-copyrighted"
                    first_frame_b64 = None
                    continue
                print(f"    ✗ {shot_dir.name}: 异常 — {e}")
                break

    return {
        "status": "done" if outputs else "error",
        "outputs": outputs,
        "provider": "seedance_client",
        "mode": "text_to_video",
    }


def run_phase5(storyboard_data: dict, output_dir: Path, dry_run: bool) -> dict:
    """Phase 5: 视频生成 — OM SeedanceVideo (reference_to_video) 带降级"""
    _banner(5, 8, "视频生成 (Seedance — reference_to_video)", dry_run)
    start = _now()
    
    # Estimate based on shot count
    _num_shots = len(storyboard_data.get("shots", [])) if storyboard_data else 10
    _p5_est = estimate_phase_duration("phase5", num_shots=_num_shots)
    print(f"  ⏱ Phase 5 开始 (预估 ~{int(_p5_est)}s, {_num_shots} 镜头)")

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # 优先尝试 OM SeedanceVideo
    try:
        print("  → 尝试 OM SeedanceVideo...")
        result = _run_phase5_om_seedance(storyboard_data, output_dir, _timing_ctx={"start": start, "estimate": _p5_est})
        result["duration_s"] = _elapsed(start)
        if result["status"] == "done":
            print(f"  ✓ Phase 5 完成: {len(result['outputs'])} 视频 (OM SeedanceVideo)")
            
            # Quality gate: Phase 5
            qg_report = run_quality_check("phase5", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 5 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
        else:
            print(f"  ⚠ Phase 5 部分完成: {len(result.get('outputs', []))} 视频, {len(result.get('errors', []))} 错误")
        return result

    except ImportError as e:
        print(f"  ⚠ OM SeedanceVideo 不可用: {e}")
        print("  → 降级到 seedance_client...")
    except Exception as e:
        print(f"  ⚠ OM SeedanceVideo 异常: {e}")
        print("  → 降级到 seedance_client...")

    # 降级到手写的 seedance_client
    try:
        result = _run_phase5_fallback(output_dir)
        result["duration_s"] = _elapsed(start)
        if result["status"] == "done":
            print(f"  ✓ Phase 5 完成: {len(result['outputs'])} 视频 (seedance_client fallback)")
            
            # Quality gate: Phase 5
            qg_report = run_quality_check("phase5", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 5 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
        else:
            print(f"  ✗ Phase 5 失败 (seedance_client fallback)")
        return result

    except ImportError as e:
        return {"status": "error", "error": f"All video generation methods unavailable: {e}", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}


# ---------------------------------------------------------------------------
# Phase 6: 一致性守卫 + 场景变化检测 + 幻灯片风险评分
# ---------------------------------------------------------------------------

# --- 抄自 OpenMontage/lib/variation_checker.py ---
_GENERIC_PHRASES = {
    "a person", "a beautiful", "modern", "futuristic", "cutting-edge",
    "in today's world", "sleek design", "innovative", "state-of-the-art",
    "next-generation", "revolutionary", "a professional", "dynamic",
    "vibrant", "stunning", "breathtaking", "amazing", "incredible",
    "powerful", "seamless", "elegant solution",
}

def _check_scene_variation(scenes: list) -> dict:
    """检查场景计划的重复模式（抄自 OM variation_checker.check_scene_variation）
    
    Returns:
        {
            "score": float (0-5, lower is better),
            "verdict": "strong" | "acceptable" | "revise" | "fail",
            "violations": list of specific issues,
            "suggestions": list of improvement suggestions,
        }
    """
    if not scenes:
        return {"score": 5.0, "verdict": "fail", "violations": ["No scenes to check"], "suggestions": []}

    violations = []
    suggestions = []

    # Check 1: Shot size variety
    shot_sizes = [
        s.get("shot_language", {}).get("shot_size", "unspecified")
        for s in scenes
    ]
    from collections import Counter
    size_counts = Counter(shot_sizes)
    if len(scenes) >= 4:
        most_common_size, most_common_count = size_counts.most_common(1)[0]
        if most_common_count / len(scenes) > 0.5:
            violations.append(
                f"Shot size '{most_common_size}' used in {most_common_count}/{len(scenes)} scenes "
                f"({most_common_count/len(scenes):.0%}). Vary shot sizes for visual interest."
            )
            suggestions.append("Mix wide establishing shots with close-ups for visual rhythm.")

    # Check 2: Consecutive same-size shots
    longest_run = 1 if shot_sizes else 0
    current_run = 1
    for i in range(1, len(shot_sizes)):
        if shot_sizes[i] == shot_sizes[i-1] and shot_sizes[i] != "unspecified":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    if longest_run >= 3:
        violations.append(
            f"{longest_run} consecutive same-size shots. "
            f"Vary shot sizes between scenes for editorial rhythm."
        )

    # Check 3: Static shot overuse
    movements = [
        s.get("shot_language", {}).get("camera_movement", "unspecified")
        for s in scenes
    ]
    static_count = sum(1 for m in movements if m in ("static", "unspecified"))
    if len(scenes) >= 4 and static_count / len(scenes) > 0.6:
        violations.append(
            f"{static_count}/{len(scenes)} scenes are static or unspecified movement. "
            f"Add intentional camera movement to at least 40% of scenes."
        )
        suggestions.append("Consider dolly_in for emphasis, tracking for energy, or crane for scale.")

    # Check 4: Lighting variety
    lightings = {
        s.get("shot_language", {}).get("lighting_key")
        for s in scenes
        if s.get("shot_language", {}).get("lighting_key")
    }
    if len(scenes) >= 4 and len(lightings) <= 1:
        violations.append(
            f"Only {len(lightings)} unique lighting setup(s) across {len(scenes)} scenes. "
            f"Vary lighting to create mood shifts."
        )

    # Check 5: Hero moment exists and is visually distinct
    hero_scenes = [s for s in scenes if s.get("hero_moment")]
    if len(scenes) >= 4 and not hero_scenes:
        violations.append(
            "No hero_moment flagged. Every video should have at least one visual peak."
        )
        suggestions.append("Mark the most impactful scene as hero_moment=true.")

    if hero_scenes:
        for hero in hero_scenes:
            hero_idx = scenes.index(hero)
            hero_size = hero.get("shot_language", {}).get("shot_size")
            for offset in (-1, 1):
                neighbor_idx = hero_idx + offset
                if 0 <= neighbor_idx < len(scenes):
                    neighbor_size = scenes[neighbor_idx].get("shot_language", {}).get("shot_size")
                    if hero_size and neighbor_size and hero_size == neighbor_size:
                        violations.append(
                            f"Hero scene '{hero.get('id')}' has same shot size as neighbor. "
                            f"Hero moments should be visually distinct from surrounding scenes."
                        )

    # Check 6: Description specificity
    generic_count = 0
    for scene in scenes:
        desc = scene.get("description", "").lower()
        for phrase in _GENERIC_PHRASES:
            if phrase in desc:
                generic_count += 1
                break
    if generic_count >= len(scenes) * 0.3:
        violations.append(
            f"{generic_count}/{len(scenes)} scenes use generic language. "
            f"Replace vague descriptions with specific visual details."
        )
        suggestions.append(
            "Instead of 'a beautiful cityscape', try 'rain-slicked Tokyo intersection "
            "at night, neon reflections in puddles, pedestrians with translucent umbrellas'."
        )

    # Check 7: Texture keywords presence
    textured = sum(1 for s in scenes if s.get("texture_keywords"))
    if len(scenes) >= 4 and textured < len(scenes) * 0.3:
        violations.append(
            f"Only {textured}/{len(scenes)} scenes have texture_keywords. "
            f"Add texture descriptors to visual scenes for richer generation prompts."
        )

    # Check 8: Shot intent completeness
    intented = sum(1 for s in scenes if s.get("shot_intent"))
    if len(scenes) >= 4 and intented < len(scenes) * 0.5:
        violations.append(
            f"Only {intented}/{len(scenes)} scenes have shot_intent. "
            f"Every scene should explain WHY it exists in the video."
        )

    # Score
    score = min(5.0, len(violations) * 0.6)
    if score < 2.0:
        verdict = "strong"
    elif score < 3.0:
        verdict = "acceptable"
    elif score < 4.0:
        verdict = "revise"
    else:
        verdict = "fail"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "violations": violations,
        "suggestions": suggestions,
    }


# --- 抄自 OpenMontage/lib/slideshow_risk.py ---
def _score_slideshow_risk(scenes: list) -> dict:
    """评估幻灯片风险（抄自 OM slideshow_risk.score_slideshow_risk）
    
    Returns:
        {
            "average": float,
            "verdict": str,
            "dimensions": {dimension_name: {"score": float, "reason": str}},
        }
    """
    if not scenes:
        return {"average": 5.0, "verdict": "fail", "dimensions": {}}

    dimensions = {
        "repetition": _score_repetition(scenes),
        "decorative_visuals": _score_decorative(scenes),
        "weak_motion": _score_weak_motion(scenes),
        "weak_shot_intent": _score_weak_intent(scenes),
        "typography_overreliance": _score_typography(scenes),
    }

    scores = [d["score"] for d in dimensions.values()]
    average = sum(scores) / len(scores)

    if average < 2.0:
        verdict = "strong"
    elif average < 3.0:
        verdict = "acceptable"
    elif average < 4.0:
        verdict = "revise"
    else:
        verdict = "fail"

    return {
        "average": round(average, 2),
        "verdict": verdict,
        "dimensions": dimensions,
    }


def _score_repetition(scenes: list) -> dict:
    """评估视觉重复度"""
    if len(scenes) < 3:
        return {"score": 0.0, "reason": "Too few scenes to assess repetition"}

    from collections import Counter
    types = Counter(s.get("type", "unknown") for s in scenes)
    most_common_type, most_common_count = types.most_common(1)[0]
    type_ratio = most_common_count / len(scenes)

    descriptions = [s.get("description", "").lower()[:50] for s in scenes]
    unique_desc_ratio = len(set(descriptions)) / len(descriptions)

    sizes = [s.get("shot_language", {}).get("shot_size", "none") for s in scenes]
    size_ratio = Counter(sizes).most_common(1)[0][1] / len(scenes)

    score = 0.0
    reasons = []

    if type_ratio > 0.7:
        score += 2.0
        reasons.append(f"Scene type '{most_common_type}' dominates at {type_ratio:.0%}")
    if unique_desc_ratio < 0.6:
        score += 1.5
        reasons.append(f"Only {unique_desc_ratio:.0%} unique descriptions")
    if size_ratio > 0.6:
        score += 1.5
        reasons.append(f"Same shot size in {size_ratio:.0%} of scenes")

    return {"score": min(5.0, score), "reason": "; ".join(reasons) or "Good variety"}


def _score_decorative(scenes: list) -> dict:
    """评估场景是否装饰性而非传达信息"""
    decorative_count = 0
    for scene in scenes:
        has_info_role = bool(scene.get("information_role"))
        has_narrative_role = bool(scene.get("narrative_role"))
        has_intent = bool(scene.get("shot_intent"))

        if not has_info_role and not has_narrative_role and not has_intent:
            decorative_count += 1

    ratio = decorative_count / len(scenes)
    score = min(5.0, ratio * 5.0)

    if ratio > 0.5:
        reason = f"{decorative_count}/{len(scenes)} scenes have no stated purpose"
    elif ratio > 0.2:
        reason = f"{decorative_count}/{len(scenes)} scenes lack stated purpose"
    else:
        reason = "Most scenes have clear communicative purpose"

    return {"score": round(score, 1), "reason": reason}


def _score_weak_motion(scenes: list) -> dict:
    """评估摄像机运动是否有目的"""
    total_moving = 0
    purposeless_moving = 0

    for scene in scenes:
        sl = scene.get("shot_language", {})
        movement = sl.get("camera_movement", "static")
        if movement not in ("static", "unspecified", None):
            total_moving += 1
            if not scene.get("shot_intent"):
                purposeless_moving += 1

    if total_moving == 0:
        return {"score": 1.5, "reason": "No camera movement defined"}

    ratio = purposeless_moving / total_moving
    score = min(5.0, ratio * 4.0)

    if ratio > 0.5:
        reason = f"{purposeless_moving}/{total_moving} moving shots lack shot_intent"
    else:
        reason = "Camera movement appears purposeful"

    return {"score": round(score, 1), "reason": reason}


def _score_weak_intent(scenes: list) -> dict:
    """评估 shot_intent 完整性"""
    with_intent = sum(1 for s in scenes if s.get("shot_intent"))
    ratio = with_intent / len(scenes)

    score = min(5.0, (1.0 - ratio) * 5.0)

    if ratio < 0.3:
        reason = f"Only {with_intent}/{len(scenes)} scenes have shot_intent"
    elif ratio < 0.6:
        reason = f"{with_intent}/{len(scenes)} scenes have shot_intent"
    else:
        reason = "Strong shot intent coverage"

    return {"score": round(score, 1), "reason": reason}


def _score_typography(scenes: list) -> dict:
    """评估文字主导过度"""
    text_scenes = sum(
        1 for s in scenes
        if s.get("type") in ("text_card", "stat_card", "kpi_grid")
    )
    ratio = text_scenes / len(scenes)

    if ratio > 0.6:
        score = 4.0
        reason = f"{text_scenes}/{len(scenes)} scenes are text/stat cards"
    elif ratio > 0.4:
        score = 2.5
        reason = f"{text_scenes}/{len(scenes)} scenes are text-based"
    elif ratio > 0.2:
        score = 1.0
        reason = "Balanced text and visual content"
    else:
        score = 0.0
        reason = "Visual-first approach"

    return {"score": score, "reason": reason}


def run_phase6(output_dir: Path, dry_run: bool, storyboard_data: dict = None) -> dict:
    """Phase 6: consistency_guard + scene_variation_check + slideshow_risk_score
    
    集成三个质检模块：
    1. consistency_guard — 角色一致性检查与修复
    2. scene_variation_check — 场景变化检测（抄自 OM variation_checker）
    3. slideshow_risk_score — 幻灯片风险评分（抄自 OM slideshow_risk）
    """
    _banner(6, 8, "一致性守卫 + 场景变化检测 + 幻灯片风险评分", dry_run)
    start = _now()
    _p6_est = estimate_phase_duration("phase6")
    print(f"  ⏱ Phase 6 开始 (预估 ~{int(_p6_est)}s)")
    output_dir = Path(output_dir)
    outputs = []

    if dry_run:
        print("  ⊘ dry-run 模式，跳过一致性检查")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # --- 1. consistency_guard (原有逻辑) ---
    try:
        from consistency_guard import run_consistency_check

        print("  → run_consistency_check: 检查角色一致性...")
        result = run_consistency_check(output_dir=output_dir)
        outputs.append("consistency_report.json")
        print(f"  ✓ 角色一致性检查完成")

    except ImportError as e:
        print(f"  ⚠ consistency_guard 不可用: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"  ⚠ 角色一致性检查失败: {e}")

    # --- 2. scene_variation_check (新增，抄自 OM variation_checker) ---
    if storyboard_data:
        print("  → scene_variation_check: 检查场景变化...")
        scenes = storyboard_data.get("shots", [])
        if scenes:
            variation_result = _check_scene_variation(scenes)
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
            slideshow_result = _score_slideshow_risk(scenes)
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

    print(f"  ✓ Phase 6 完成")
    
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
# Phase 7: 组装引擎 (Assembly) — delegates to OM VideoStitch
# ---------------------------------------------------------------------------

def run_phase7(output_dir: Path, dry_run: bool,
               transition: str = "crossfade",
               transition_duration: float = 0.5,
               media_profile: str = "1080p") -> dict:
    """Phase 7: 组装引擎 — 直接调用 OM VideoStitch"""
    _banner(7, 8, f"组装引擎 (Assembly) — {transition}", dry_run)
    start = _now()
    _p7_est = estimate_phase_duration("phase7")
    print(f"  ⏱ Phase 7 开始 (预估 ~{int(_p7_est)}s)")
    output_dir = Path(output_dir)

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
                        import json
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            shot_metas.append(json.load(f))
                    else:
                        shot_metas.append({})  # Empty meta if file doesn't exist

    if not clip_paths:
        return {"status": "error", "error": "No video clips found", "duration_s": _elapsed(start)}

    if len(clip_paths) < 2:
        import shutil
        output_final = output_dir / "raw_assembly.mp4"
        shutil.copy2(clip_paths[0], str(output_final))
        print(f"  ✓ Phase 7 完成: 仅 1 个片段，直接复制")
        
        # Quality gate: Phase 7
        qg_report = run_quality_check("phase7", output_dir)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 7 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
        
        return {"status": "done", "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4"], "method": "single_clip_copy"}

    # Intelligent transition selection based on shot emotions
    print(f"  → 发现 {len(clip_paths)} 个视频片段")
    
    # ── Smart transition: visual similarity + three-layer voting ──
    smart_decisions = None
    try:
        from shot_embedder import embed_all_shots, compute_transition_similarity
        from smart_transition import decide_all_transitions
        
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

    # 调用 edit_decisions 架构（从 OpenMontage VideoCompose 学习）
    try:
        from edit_decisions import build_edit_decisions, execute_edit_decisions
        
        print("  → 构建 edit_decisions（帧精确裁切 + 音频归一化）...")
        edit_decisions = build_edit_decisions(
            shots_dir=shots_dir,
            target_width=1920,
            target_height=1080,
            transition_decisions=smart_decisions,
        )
        
        print(f"  → 执行 edit_decisions（{len(edit_decisions['cuts'])} 个片段）...")
        ed_result = execute_edit_decisions(
            edit_decisions,
            output_path=str(output_dir / "raw_assembly.mp4")
        )
        
        if ed_result.get("success"):
            print(f"  ✓ Phase 7 完成: raw_assembly.mp4 (edit_decisions)")
            
            # Quality gate: Phase 7
            qg_report = run_quality_check("phase7", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 7 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4"],
                "method": "edit_decisions",
                "transition": batch_transition,
                "transition_duration": transition_duration,
                "clip_count": len(clip_paths),
                "transition_selections": selected_transitions if selected_transitions else None,
                "edit_decisions_segments": ed_result.get("segments"),
            }
        else:
            error_msg = ed_result.get("error", "Unknown error")
            print(f"  ⚠ edit_decisions 失败: {error_msg}，降级为 VideoStitch")
            # Fall through to VideoStitch fallback
            
    except Exception as e:
        print(f"  ⚠ edit_decisions 异常: {e}，降级为 VideoStitch")
        # Fall through to VideoStitch fallback
    
    # Fallback: OM VideoStitch（如果 edit_decisions 失败）
    try:
        from tools.video.video_stitch import VideoStitch
        stitcher = VideoStitch()
        result = stitcher.execute({
            "operation": "stitch",
            "clips": clip_paths,
            "output_path": str(output_dir / "raw_assembly.mp4"),
            "transition": batch_transition,
            "transition_duration": transition_duration,
            "auto_normalize": True,
            "profile": media_profile,
        })

        if result.success:
            print(f"  ✓ Phase 7 完成: raw_assembly.mp4 (VideoStitch fallback)")
            
            # Quality gate: Phase 7
            qg_report = run_quality_check("phase7", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 7 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
            
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4"],
                "method": f"VideoStitch_{batch_transition}_fallback",
                "transition": batch_transition,
                "transition_duration": transition_duration,
                "clip_count": len(clip_paths),
                "transition_selections": selected_transitions if selected_transitions else None,
            }
        else:
            return {"status": "error", "error": result.error, "duration_s": _elapsed(start)}

    except ImportError as e:
        return {"status": "error", "error": f"VideoStitch unavailable: {e}", "duration_s": _elapsed(start)}


# Phase 8: 后期处理 (audio + visual + rhythm → polished.mp4)
# ---------------------------------------------------------------------------

def _detect_bgm(output_dir: Path, storyboard_path: Optional[Path] = None) -> Optional[str]:
    """
    Detect background music file for Phase 8 audio processing.

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



def run_phase8(output_dir: Path, dry_run: bool, color_grade: Optional[str] = None, upscale: Optional[int] = None, media_profile: str = "1080p") -> dict:
    """Phase 8: audio_pipeline + visual_post + [color_grade] + [upscale] + rhythm_editor → polished.mp4

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
    _banner(8, 8, "后期处理 (Post-Production)", dry_run)
    start = _now()
    _p8_est = estimate_phase_duration("phase8")
    print(f"  ⏱ Phase 8 开始 (预估 ~{int(_p8_est)}s)")
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

    try:
        from visual_post import process_visual
        from rhythm_editor import edit_rhythm

        # Step 8.1: Audio processing via OM AudioMixer
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
            from tools.audio.audio_mixer import AudioMixer
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
                    "-shortest",
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
                # Fallback: just copy video
                import shutil
                shutil.copy2(raw_video, audio_out)
        except ImportError as e:
            print(f"  ⚠ AudioMixer unavailable: {e}")
            # Fallback: just copy video
            import shutil
            shutil.copy2(raw_video, audio_out)

        audio_out = str(audio_out)

        # Step 8.2: visual_post
        print("  → visual_post: 视觉后期...")
        visual_out = str(output_dir / "visual_processed.mp4")
        process_visual(
            video_path=audio_out,
            output_path=visual_out,
        )
        outputs.append("visual_processed.mp4")

        current_video = visual_out

        # Step 8.2.1: Subtitle burn (optional, from OM RemotionCaptionBurn)
        if sb_path_str:
            print("  → subtitle_burn: 字幕烧录 (RemotionCaptionBurn)...")
            subtitled_out = str(output_dir / "subtitled.mp4")
            try:
                from tools.video.remotion_caption_burn import RemotionCaptionBurn
                caption_burner = RemotionCaptionBurn()

                # Read captions from storyboard
                import json
                with open(sb_path_str, 'r', encoding='utf-8') as f:
                    storyboard_data = json.load(f)

                # Extract captions from shots
                captions = []
                for shot in storyboard_data.get('shots', []):
                    if 'caption' in shot and shot['caption']:
                        captions.append({
                            "text": shot['caption'],
                            "start": shot.get('start_time', 0),
                            "end": shot.get('end_time', 0),
                        })

                if captions:
                    burn_result = caption_burner.execute({
                        "input_path": str(current_video),
                        "output_path": str(subtitled_out),
                        "captions": captions,
                        "style": {
                            "font_size": 48,
                            "font_color": "#FFFFFF",
                            "background_color": "#000000",
                            "background_opacity": 0.7,
                            "position": "bottom",
                        },
                    })

                    if burn_result.success:
                        current_video = subtitled_out
                        outputs.append("subtitled.mp4")
                        print(f"    ✓ 字幕烧录完成: {len(captions)} 条字幕")
                    else:
                        print(f"    ⚠ 字幕烧录失败: {burn_result.error}")
                else:
                    print(f"    ⊘ No subtitle data available, skipping subtitle burn")
            except ImportError as e:
                print(f"    ⚠ RemotionCaptionBurn unavailable: {e}")
            except Exception as e:
                print(f"    ⚠ 字幕烧录异常: {e}")

        # Step 8.2.5: Color grade (optional, from OM ColorGrade)
        if color_grade:
            print(f"  → color_grade: 应用调色 ({color_grade})...")
            graded_out = str(output_dir / "color_graded.mp4")
            try:
                from tools.enhancement.color_grade import ColorGrade
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

        # Step 8.2.6: Upscale (optional, from OM Upscale — lanczos)
        if upscale:
            print(f"  → upscale: 超分到 {upscale}p (lanczos)...")
            upscaled_out = str(output_dir / "upscaled.mp4")
            try:
                from tools.enhancement.upscale import Upscale
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

        # Step 8.3: rhythm_editor → polished.mp4
        print("  → rhythm_editor: 节奏编辑...")
        final_out = str(output_dir / "polished.mp4")
        edit_rhythm(
            video_path=current_video,
            storyboard_path=sb_path_str,
            output_path=final_out,
        )
        outputs.append("polished.mp4")

        # Step 8.4: Final encoding with media profile
        print(f"  → final_encode: 使用 {media_profile} 配置重新编码...")
        final_encoded = str(output_dir / "polished_final.mp4")
        profile = _get_profile_dict(media_profile)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", final_out,
            "-vf", f"scale={profile['width']}:{profile['height']}",
            "-r", str(profile["fps"]),
            "-c:v", profile["codec"],
            "-crf", str(profile["crf"]),
            "-preset", "medium",
            "-c:a", profile["audio_codec"],
            "-b:a", "192k",
            "-pix_fmt", profile["pixel_format"],
            final_encoded,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                # Replace polished.mp4 with final encoded version
                import shutil
                shutil.move(final_encoded, final_out)
                outputs.append(f"polished.mp4 (encoded with {media_profile})")
                print(f"    ✓ 最终编码完成: {profile['width']}x{profile['height']} @ {profile['fps']}fps")
            else:
                print(f"    ⚠ 最终编码失败，使用原始 polished.mp4")
        except Exception as e:
            print(f"    ⚠ 最终编码异常: {e}，使用原始 polished.mp4")

        print(f"  ✓ Phase 8 完成: polished.mp4")
        
        # Quality gate: Phase 8
        qg_report = run_quality_check("phase8", output_dir)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 8 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
        
        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "color_grade": color_grade,
            "upscale": upscale,
            "audio_enhanced": audio_success,
            "bgm_detected": bgm_path is not None,
            "media_profile": media_profile,
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
# LangGraph Phase 2: StateGraph + Interrupt + Command + Conditional Edges
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
        # Internal fields (not in original plan but needed for compatibility)
        output_dir: str
        duration: int
        dry_run: bool
        transition: str
        transition_duration: float
        media_profile: str
        skip_phase: list
        resume: bool
        auto_approve: bool
        # Tracking fields
        phase_results: dict
        retry_count: int
        completed_phases: list

    def build_pipeline_graph(auto_approve: bool = False):
        """Build the LangGraph StateGraph for the pipeline.
        
        Args:
            auto_approve: If True, skip interrupt nodes (for CI/testing)
        
        Returns:
            CompiledStateGraph instance
        """
        graph = StateGraph(HonCutState)
        
        # Add nodes (each phase is a node)
        graph.add_node("phase2", node_phase2)
        graph.add_node("phase2_5", node_phase2_5)
        
        # Interrupt node for human review (Phase 2.5 → Phase 3)
        if not auto_approve:
            graph.add_node("review_storyboard", node_review_storyboard)
        
        graph.add_node("phase3", node_phase3)
        graph.add_node("phase4", node_phase4)
        
        # Conditional routing for Phase 4 → Phase 5 (different generators)
        graph.add_node("phase5_txt2vid", node_phase5_txt2vid)
        graph.add_node("phase5_img2vid", node_phase5_img2vid)
        graph.add_node("phase5_reference", node_phase5_reference)
        
        graph.add_node("phase6", node_phase6)
        graph.add_node("phase7", node_phase7)
        graph.add_node("phase8", node_phase8)
        
        # Define edges
        graph.add_edge(START, "phase2")
        graph.add_edge("phase2", "phase2_5")
        
        if not auto_approve:
            graph.add_edge("phase2_5", "review_storyboard")
            graph.add_edge("review_storyboard", "phase3")
        else:
            graph.add_edge("phase2_5", "phase3")
        
        graph.add_edge("phase3", "phase4")
        
        # Conditional routing: Phase 4 routes to different Phase 5 variants
        graph.add_conditional_edges(
            "phase4",
            route_phase5,
            {
                "txt2vid": "phase5_txt2vid",
                "img2vid": "phase5_img2vid",
                "reference": "phase5_reference",
            }
        )
        
        # All Phase 5 variants converge to Phase 6
        graph.add_edge("phase5_txt2vid", "phase6")
        graph.add_edge("phase5_img2vid", "phase6")
        graph.add_edge("phase5_reference", "phase6")
        
        # Quality gate: Phase 6 can rollback to Phase 5 via Command
        graph.add_conditional_edges(
            "phase6",
            quality_gate_router,
            {
                "pass": "phase7",
                "retry": "phase5_txt2vid",  # Default retry path
            }
        )
        
        graph.add_edge("phase7", "phase8")
        graph.add_edge("phase8", END)
        
        # Compile with SQLite checkpointer
        # Note: checkpointer is created per-invocation in run_pipeline()
        return graph

    # --- Node functions (wrappers around existing run_phase* functions) ---
    
    def node_phase2(state: HonCutState) -> dict:
        """Phase 2 node: 编剧引擎"""
        output_dir = Path(state["output_dir"])
        result = run_phase2(
            text=state["text"],
            output_dir=output_dir,
            duration=state["duration"],
            dry_run=state["dry_run"],
        )
        
        # Extract internal data
        storyboard_data = result.pop("_storyboard", None)
        characters_data = result.pop("_characters", None)
        
        return {
            "storyboard": storyboard_data or {},
            "characters": characters_data.get("characters", []) if characters_data else [],
            "phase_results": {**state.get("phase_results", {}), "phase2": result},
            "completed_phases": state.get("completed_phases", []) + ["phase2"],
            "skip_phase": state.get("skip_phase", []),
        }

    def node_phase2_5(state: HonCutState) -> dict:
        """Phase 2.5 node: 故事板图片生成"""
        output_dir = Path(state["output_dir"])
        storyboard_data = state.get("storyboard")
        characters_data = {"characters": state.get("characters", [])}
        
        result = run_phase2_5(
            storyboard_data=storyboard_data,
            characters_data=characters_data,
            output_dir=output_dir,
            dry_run=state["dry_run"],
        )
        
        # Extract storyboard image path
        storyboard_image = ""
        if result.get("status") == "done" and result.get("outputs"):
            storyboard_image = result["outputs"][0]
        
        return {
            "storyboard_image": storyboard_image,
            "phase_results": {**state.get("phase_results", {}), "phase2_5": result},
            "completed_phases": state.get("completed_phases", []) + ["phase2_5"],
            "skip_phase": state.get("skip_phase", []),
        }

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
            # Rollback to Phase 2.5
            return Command(goto="phase2_5", update={"status": "storyboard_rejected"})
        
        return {}

    def node_phase3(state: HonCutState) -> dict:
        """Phase 3 node: 角色工厂"""
        output_dir = Path(state["output_dir"])
        characters_data = {"characters": state.get("characters", [])}
        
        result = run_phase3(
            output_dir=output_dir,
            characters_data=characters_data,
            dry_run=state["dry_run"],
        )
        
        return {
            "phase_results": {**state.get("phase_results", {}), "phase3": result},
            "completed_phases": state.get("completed_phases", []) + ["phase3"],
            "skip_phase": state.get("skip_phase", []),
        }

    def node_phase4(state: HonCutState) -> dict:
        """Phase 4 node: 编排器"""
        output_dir = Path(state["output_dir"])
        
        result = run_phase4(
            output_dir=output_dir,
            dry_run=state["dry_run"],
        )
        
        return {
            "phase_results": {**state.get("phase_results", {}), "phase4": result},
            "completed_phases": state.get("completed_phases", []) + ["phase4"],
            "skip_phase": state.get("skip_phase", []),
        }

    def route_phase5(state: HonCutState) -> str:
        """根据镜头属性路由到不同的 Phase 5 生成器"""
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

    def node_phase5_txt2vid(state: HonCutState) -> dict:
        """Phase 5 variant: text-to-video generation"""
        output_dir = Path(state["output_dir"])
        storyboard_data = state.get("storyboard")
        
        result = run_phase5(
            storyboard_data=storyboard_data,
            output_dir=output_dir,
            dry_run=state["dry_run"],
        )
        
        # P0-2 fix: increment retry_count so quality_gate_router can terminate
        return {
            "videos": result.get("outputs", []),
            "phase_results": {**state.get("phase_results", {}), "phase5": result},
            "completed_phases": state.get("completed_phases", []) + ["phase5"],
            "retry_count": state.get("retry_count", 0) + 1,
            "skip_phase": state.get("skip_phase", []),
        }

    def node_phase5_img2vid(state: HonCutState) -> dict:
        """Phase 5 variant: image-to-video generation (uses storyboard image).
        当前委托给 node_phase5_txt2vid，待实现差异化逻辑。"""
        # Same as txt2vid for now, but could use different logic
        return node_phase5_txt2vid(state)

    def node_phase5_reference(state: HonCutState) -> dict:
        """Phase 5 variant: reference-to-video generation.
        当前委托给 node_phase5_txt2vid，待实现差异化逻辑。"""
        # Same as txt2vid for now, but could use different logic
        return node_phase5_txt2vid(state)

    def node_phase6(state: HonCutState) -> dict:
        """Phase 6 node: 一致性守卫 + 质检"""
        output_dir = Path(state["output_dir"])
        storyboard_data = state.get("storyboard")
        
        result = run_phase6(
            output_dir=output_dir,
            dry_run=state["dry_run"],
            storyboard_data=storyboard_data,
        )
        
        # Extract quality metrics for gating
        quality_report = {
            "slideshow_risk": result.get("slideshow_risk", 0.0),
            "variation_score": result.get("variation_score", 5.0),
        }
        
        return {
            "quality_report": quality_report,
            "phase_results": {**state.get("phase_results", {}), "phase6": result},
            "completed_phases": state.get("completed_phases", []) + ["phase6"],
            "skip_phase": state.get("skip_phase", []),
        }

    def quality_gate_router(state: HonCutState) -> str:
        """Quality gate: decide whether to pass or retry Phase 5"""
        quality = state.get("quality_report", {})
        retry_count = state.get("retry_count", 0)
        
        slideshow_risk = quality.get("slideshow_risk", 0.0)
        variation_score = quality.get("variation_score", 5.0)
        
        # Check if quality is acceptable
        if slideshow_risk > 0.7 or variation_score < 3.0:
            # Quality failed
            if retry_count < 2:
                # Retry Phase 5 (max 2 times)
                print(f"\n  ⚠ 质检不通过 (slideshow_risk={slideshow_risk}, variation={variation_score})")
                print(f"  🔄 回退到 Phase 5 重新生成 (retry {retry_count + 1}/2)")
                return "retry"
            else:
                print(f"\n  ⚠ 质检不通过，但已达最大重试次数，继续执行")
        
        # Quality passed or max retries reached
        return "pass"

    def node_phase7(state: HonCutState) -> dict:
        """Phase 7 node: 组装引擎"""
        output_dir = Path(state["output_dir"])
        
        result = run_phase7(
            output_dir=output_dir,
            dry_run=state["dry_run"],
            transition=state.get("transition", "crossfade"),
            transition_duration=state.get("transition_duration", 0.5),
            media_profile=state.get("media_profile", "1080p"),
        )
        
        return {
            "phase_results": {**state.get("phase_results", {}), "phase7": result},
            "completed_phases": state.get("completed_phases", []) + ["phase7"],
            "skip_phase": state.get("skip_phase", []),
        }

    def node_phase8(state: HonCutState) -> dict:
        """Phase 8 node: 后期处理"""
        output_dir = Path(state["output_dir"])
        
        result = run_phase8(
            output_dir=output_dir,
            dry_run=state["dry_run"],
            media_profile=state.get("media_profile", "1080p"),
        )
        
        final_video = ""
        if result.get("status") == "done" and result.get("outputs"):
            final_video = result["outputs"][0]
        
        return {
            "final_video": final_video,
            "status": "completed",
            "phase_results": {**state.get("phase_results", {}), "phase8": result},
            "completed_phases": state.get("completed_phases", []) + ["phase8"],
            "skip_phase": state.get("skip_phase", []),
        }

else:
    # Fallback when LangGraph is not available
    def build_pipeline_graph(auto_approve: bool = False):
        return None


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = "1080p",
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
        dry_run: dry-run 模式（Phase 2 实际调 LLM，Phase 3 skip-images，Phase 4 dry-run，Phase 5-8 跳过）
        skip_phase: 跳过指定 phase 列表，如 [3, 8]
        output_dir: 输出目录
        transition: Phase 7 转场模式 ("crossfade" | "fade" | "cut")
        transition_duration: Phase 7 转场时长（秒），默认 0.5
        media_profile: 编码配置名称，从 MEDIA_PROFILES 中选择（默认 "1080p"）
        resume: 从检查点恢复，跳过已完成的 Phase

    Returns:
        pipeline_report dict
    """
    skip_phase = skip_phase or []
    skip_phases = set(skip_phase)
    output_path = Path(output_dir).resolve()
    _ensure_dir(output_path)

    # --- M6: --resume-from 支持 ---
    if resume_from:
        try:
            from artifact_chain import PHASE_SEQUENCE, can_resume_from
            if can_resume_from(resume_from, output_path):
                idx = PHASE_SEQUENCE.index(resume_from) if resume_from in PHASE_SEQUENCE else 0
                skip_phases = set(PHASE_SEQUENCE[:idx])
                print(f"  🔄 [M6] Resume-from {resume_from}: 跳过 {sorted(skip_phases)}")
            else:
                print(f"  ⚠ [M6] Resume-from {resume_from}: 前置依赖不满足，从头开始")
        except Exception as e:
            print(f"  ⚠ [M6] resume-from 解析失败: {e}")

    # ---- 进度报告系统初始化 ----
    reporter = ProgressReporter(str(output_path), total_phases=len(PHASE_ORDER))

    # --- M6: 产物链（增量）---
    try:
        from artifact_chain import save_checkpoint as save_artifact_checkpoint, can_resume_from, get_resumable_phase
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
                        _write_checkpoint(output_path, phase_name, phase_result)
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
        # resume 模式下文本可以不提供（Phase 2 会被跳过如果已完成）
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
        print(f"  ✓ Auto-approve: 跳过人工审核节点")
    
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
            graph = build_pipeline_graph(auto_approve=auto_approve)
            
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
                    graph = build_pipeline_graph(auto_approve=True)
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
                "dry_run": dry_run,
                "transition": transition,
                "transition_duration": transition_duration,
                "media_profile": media_profile,
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
                
                # Build report from final state
                report = {
                    "status": final_state.get("status", "completed"),
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
                
                # Mark pipeline complete
                reporter.mark_completed()
                
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

    # --- M1: Phase 1 导演规划（增量，失败不影响后续）---
    if 1 not in skip_phase:
        try:
            p1_result = run_phase1(text, output_path, dry_run)
            report["phases"]["phase1"] = p1_result
            _write_checkpoint(output_path, "phase1", p1_result)
        except Exception as e:
            print(f"  ⚠ Phase 1 降级跳过: {e}")
            report["phases"]["phase1"] = {"status": "skipped", "reason": str(e)}

    # ---- Phase 2: 编剧引擎 (必须成功) ----
    storyboard_data = None
    characters_data = None
    if 2 in skip_phase:
        report["phases"]["2"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase2" in completed_phases:
        # Resume: 从 checkpoint 加载 Phase 2 结果
        cp = _read_checkpoint(output_path)
        p2 = dict(cp["results"].get("phase2", {"status": "done"}))
        p2.setdefault("status", "done")
        storyboard_data = None
        characters_data = None
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())
        report["phases"]["2"] = {**p2, "resumed": True}
        print(f"  🔄 Phase 2: 从 checkpoint 恢复 (已跳过)")
    else:
        reporter.phase_start("phase2", "编剧引擎")
        p2 = run_phase2(text, output_dir, duration, dry_run)
        # 提取内部数据（不写入 report）
        storyboard_data = p2.pop("_storyboard", None)
        characters_data = p2.pop("_characters", None)
        report["phases"]["2"] = p2

        if p2["status"] == "error":
            reporter.mark_failed(f"Phase 2 failed: {p2.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 2 failed: {p2.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        reporter.phase_done("phase2", "编剧引擎完成", duration_s=p2.get("duration_s"))
        # 写入 checkpoint
        _write_checkpoint(output_path, "phase2", p2)
    # 如果 Phase 2 被跳过或数据为空，尝试从文件读
    if 2 in skip_phase or storyboard_data is None:
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())

    # ---- Phase 2.5: 故事板图片生成 (OM image_selector) ----
    if 2.5 in skip_phase:
        report["phases"]["2.5"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase2_5" in completed_phases:
        cp = _read_checkpoint(output_path)
        p2_5 = dict(cp["results"].get("phase2_5", {"status": "done"}))
        p2_5.setdefault("status", "done")
        report["phases"]["2.5"] = {**p2_5, "resumed": True}
        print(f"  🔄 Phase 2.5: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None or characters_data is None:
        report["phases"]["2.5"] = {"status": "skipped", "reason": "no storyboard/characters data"}
    else:
        reporter.phase_start("phase2_5", "故事板图片生成")
        p2_5 = run_phase2_5(storyboard_data, characters_data, Path(output_dir), dry_run)
        report["phases"]["2.5"] = p2_5
        if p2_5["status"] == "error":
            # Phase 2.5 失败不阻断管线，仅标记 partial
            report["status"] = "partial"
            reporter.phase_done("phase2_5", f"故事板图片生成失败: {p2_5.get('error')}", duration_s=p2_5.get("duration_s"))
        else:
            reporter.phase_done("phase2_5", "故事板图片生成完成", duration_s=p2_5.get("duration_s"))
            _write_checkpoint(output_path, "phase2_5", p2_5)

    # ---- Phase 3: 角色工厂 ----
    if 3 in skip_phase:
        report["phases"]["3"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase3" in completed_phases:
        cp = _read_checkpoint(output_path)
        p3 = dict(cp["results"].get("phase3", {"status": "done"}))
        p3.setdefault("status", "done")
        report["phases"]["3"] = {**p3, "resumed": True}
        print(f"  🔄 Phase 3: 从 checkpoint 恢复 (已跳过)")
    elif characters_data is None:
        report["phases"]["3"] = {"status": "skipped", "reason": "no characters data"}
    else:
        reporter.phase_start("phase3", "角色工厂")
        p3 = run_phase3(output_dir, characters_data, dry_run)
        report["phases"]["3"] = p3
        if p3["status"] == "error":
            report["status"] = "partial"
            reporter.phase_done("phase3", f"角色工厂失败: {p3.get('error')}", duration_s=p3.get("duration_s"))
        else:
            reporter.phase_done("phase3", "角色工厂完成", duration_s=p3.get("duration_s"))
            _write_checkpoint(output_path, "phase3", p3)

    # ---- Phase 4: 编排器 ----
    if 4 in skip_phase:
        report["phases"]["4"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase4" in completed_phases:
        cp = _read_checkpoint(output_path)
        p4 = dict(cp["results"].get("phase4", {"status": "done"}))
        p4.setdefault("status", "done")
        report["phases"]["4"] = {**p4, "resumed": True}
        print(f"  🔄 Phase 4: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["4"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        reporter.phase_start("phase4", "编排器")
        p4 = run_phase4(output_dir, dry_run)
        report["phases"]["4"] = p4
        if p4["status"] == "error":
            report["status"] = "partial"
            reporter.phase_done("phase4", f"编排器失败: {p4.get('error')}", duration_s=p4.get("duration_s"))
        else:
            reporter.phase_done("phase4", "编排器完成", duration_s=p4.get("duration_s"))
            _write_checkpoint(output_path, "phase4", p4)

    # ---- Phase 5: 视频生成 ----
    if 5 in skip_phase:
        report["phases"]["5"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase5" in completed_phases:
        cp = _read_checkpoint(output_path)
        p5 = dict(cp["results"].get("phase5", {"status": "done"}))
        p5.setdefault("status", "done")
        report["phases"]["5"] = {**p5, "resumed": True}
        print(f"  🔄 Phase 5: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["5"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        p5 = run_phase5(storyboard_data, output_dir, dry_run)
        report["phases"]["5"] = p5
        if p5["status"] == "error":
            report["status"] = "partial"
        else:
            _write_checkpoint(output_path, "phase5", p5)

    # ---- Phase 6: 一致性守卫 + 场景变化检测 + 幻灯片风险评分 ----
    if 6 in skip_phase:
        report["phases"]["6"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase6" in completed_phases:
        cp = _read_checkpoint(output_path)
        p6 = dict(cp["results"].get("phase6", {"status": "done"}))
        p6.setdefault("status", "done")
        report["phases"]["6"] = {**p6, "resumed": True}
        print(f"  🔄 Phase 6: 从 checkpoint 恢复 (已跳过)")
    else:
        # Ensure storyboard_data is available (may be None if Phase 2 was skipped)
        if storyboard_data is None:
            sb_path = output_path / "STORYBOARD.json"
            if sb_path.exists():
                storyboard_data = json.loads(sb_path.read_text())
        
        p6 = run_phase6(Path(output_dir), dry_run, storyboard_data=storyboard_data)
        report["phases"]["6"] = p6
        if p6["status"] == "error":
            report["status"] = "partial"
        else:
            _write_checkpoint(output_path, "phase6", p6)

        # ---- 质检门：检查 Phase 6 结果，决定是否回退到 Phase 5 ----
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
                        print(f"  🔄 建议回退到 Phase 5 重新生成视频")
                except (json.JSONDecodeError, KeyError):
                    pass

        # 如果质检门未通过，记录到报告但不阻断管线（当前版本仅警告）
        if not quality_gate_passed:
            report["quality_gate"] = {
                "passed": False,
                "reason": "角色一致性分数低于阈值",
                "recommendation": "回退到 Phase 5"
            }
        else:
            report["quality_gate"] = {"passed": True}

    # ---- Phase 7: 组装引擎 ----
    if 7 in skip_phase:
        report["phases"]["7"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase7" in completed_phases:
        cp = _read_checkpoint(output_path)
        p7 = dict(cp["results"].get("phase7", {"status": "done"}))
        p7.setdefault("status", "done")
        report["phases"]["7"] = {**p7, "resumed": True}
        print(f"  🔄 Phase 7: 从 checkpoint 恢复 (已跳过)")
    else:
        p7 = run_phase7(output_dir, dry_run, transition=transition, transition_duration=transition_duration, media_profile=media_profile)
        report["phases"]["7"] = p7
        if p7["status"] == "error":
            report["status"] = "partial"
        else:
            _write_checkpoint(output_path, "phase7", p7)

    # ---- Phase 8: 后期处理 ----
    if 8 in skip_phase:
        report["phases"]["8"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase8" in completed_phases:
        cp = _read_checkpoint(output_path)
        p8 = dict(cp["results"].get("phase8", {"status": "done"}))
        p8.setdefault("status", "done")
        report["phases"]["8"] = {**p8, "resumed": True}
        print(f"  🔄 Phase 8: 从 checkpoint 恢复 (已跳过)")
    else:
        p8 = run_phase8(output_dir, dry_run, media_profile=media_profile)
        report["phases"]["8"] = p8
        if p8["status"] == "error":
            report["status"] = "partial"
        else:
            _write_checkpoint(output_path, "phase8", p8)

    report["total_duration_s"] = _elapsed(total_start)

    # 标记管线完成
    reporter.mark_completed()

    # --- M6: 产物链验证 ---
    if M6_AVAILABLE:
        try:
            from artifact_chain import verify_artifacts, save_checkpoint as save_artifact_checkpoint
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
    parser.add_argument("--skip-phase", type=int, nargs="+", default=[], help="跳过指定 Phase")
    parser.add_argument("--transition", type=str, default="crossfade", choices=["crossfade", "fade", "cut"],
                        help="Phase 7 转场模式: crossfade (默认), fade (fade-through-black), cut (硬切)")
    parser.add_argument("--transition-duration", type=float, default=0.5,
                        help="Phase 7 转场时长（秒），默认 0.5")
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
        resume=args.resume,
        auto_approve=args.auto_approve,
        resume_from=args.resume_from,
    )

    sys.exit(0 if report["status"] in ("completed", "partial") else 1)


if __name__ == "__main__":
    main()
