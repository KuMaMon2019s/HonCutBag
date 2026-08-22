"""Legacy OM Seedance generation path retained for compatibility."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any, Optional

from phases.phase2.storyboard_assets import _shot_storyboard_reference
from prompt.shot_prompt_builder import build_batch_prompts
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.vendor_adapter import VendorAdapter, VendorModel
from utils.character_body_contracts import character_visual_description
from utils.config import get_api_key
from utils.file_integrity import _file_sha256
from utils.timing_estimator import estimate_phase_duration


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

    # Legacy compatibility route performs one transport attempt. Runtime owns
    # all retry/backoff policy for production providers.
    print(f"  → 生成 {len(shots)} 个镜头 (legacy single-attempt route)...")

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
            """Perform the legacy route's single provider attempt."""
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
            result = _generate_shot()
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
