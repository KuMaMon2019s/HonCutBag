"""Phase 6 provider adapters for calibrated continuity chunk execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from quality.seam_calibration import SeamCalibration
from runtime.bridge_execution import execute_bridge_video_task
from runtime.capacity import (
    CapacityTable,
    CrossProcessSlotTable,
    SlotTable,
    default_capacity_lease_path,
)
from runtime.continuity_chunks import (
    ChunkExecutionRequest,
    ChunkExecutionResult,
    execute_continuity_plan,
)
from runtime.execution_errors import ProviderPreparationError
from runtime.generation_tasks import GenerationTaskStore
from runtime.seedance_execution import execute_seedance_video_task
from schemas.continuity import ContinuityPlan, GenerationChunk
from utils.storyboard_motion_policy import apply_storyboard_motion_policy
from utils.video_capabilities import SEEDANCE_2_CAPABILITIES
from utils.video_geometry import resolve_video_geometry

CONTINUITY_BRIDGE_ENV = "HONCUT_CONTINUITY_BRIDGE"
CONTINUITY_BRIDGE_MODES = {"off", "auto"}
SEAM_DECISIONS_KIND = "honcut.continuity_seam_decisions.v1"
SEEDANCE_MAX_REFERENCE_IMAGES = SEEDANCE_2_CAPABILITIES.max_reference_images or 9
CONTINUITY_ANCHOR_FRAME_COUNT = SEEDANCE_2_CAPABILITIES.continuity_anchor_frame_count
SEEDANCE_MIN_IMAGE_ASPECT = 0.40
SEEDANCE_MAX_IMAGE_ASPECT = 2.50
SEEDANCE_IMAGE_ASPECT_MARGIN = 0.01
MAX_COPYRIGHT_POLICY_REPAIRS = 2
COPYRIGHT_POLICY_REPAIR_VERSION = "original_audio_frame_fallback_v1"
_COPYRIGHT_SAFE_AUDIO_CONTRACT = (
    "[copyright-safe audio contract] Generate original ambient location sounds only: "
    "natural footsteps, clothing movement, crowd presence, and location ambience. "
    "No music, soundtrack, song, melody, lyrics, or recognizable tune. No copyrighted "
    "audio, sampled recording, or imitation of any existing artist. This instruction overrides "
    "any earlier soundtrack, music, or rhythm-audio request; visual body-motion timing remains unchanged.\n"
)
_FRAME_ONLY_CONTINUITY_CONTRACT = (
    "[copyright-safe frame-only continuity fallback] The rejected reference-video item "
    "has been removed. Continue only from the remaining ordered reference images and the "
    "authored storyboard start/action/end states. Preserve their visible subject identity, "
    "position, direction, camera, and lighting without reconstructing or quoting any audio "
    "from the rejected media.\n"
)


def probe_continuity_frames(path: Path, timeline_fps: int) -> dict[str, Any]:
    """Count decoded video frames; duration metadata is only a fallback."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot count continuity frames in {path}")
    streams = json.loads(completed.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"continuity artifact has no video stream: {path}")
    stream = streams[0]
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frames in (None, "N/A"):
        duration = float(stream.get("duration") or 0.0)
        frames = round(duration * timeline_fps)
    numerator, denominator = str(stream.get("avg_frame_rate") or f"{timeline_fps}/1").split(
        "/", 1
    )
    source_fps = float(numerator) / max(float(denominator), 1.0)
    return {
        "frames": int(frames),
        "duration_s": round(int(frames) / source_fps, 6),
        "source_fps": round(source_fps, 6),
    }


def finalize_continuity_shot(
    output_path: Path,
    target_frames: int,
    timeline_fps: int,
) -> dict[str, Any]:
    """Enforce the pre-Phase-8 minimum duration without discarding valid frames."""
    before = probe_continuity_frames(output_path, timeline_fps)
    actual_frames = int(before["frames"])
    delta = target_frames - actual_frames
    if delta == 0:
        return {
            "method": "none",
            "before_frames": actual_frames,
            "target_frames": target_frames,
            "after_frames": actual_frames,
        }
    if delta < 0:
        return {
            "method": "deferred_phase8_excess_trim",
            "before_frames": actual_frames,
            "target_frames": target_frames,
            "after_frames": actual_frames,
            "excess_frames": -delta,
            "minimum_target_met": True,
        }
    if delta > max(2, math.ceil(target_frames * 0.02)):
        raise RuntimeError(
            f"{output_path.parent.name} is short by {delta} frames; "
            "additional continuation generation is required"
        )

    temporary = output_path.with_name(f"{output_path.stem}.duration_closure{output_path.suffix}")
    stretch = target_frames / actual_frames
    video_filter = (
        f"setpts={stretch:.12f}*PTS,fps={timeline_fps},"
        f"trim=end_frame={target_frames},setpts=PTS-STARTPTS"
    )
    method = "bounded_micro_retime"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_path),
            "-vf",
            video_filter,
            "-frames:v",
            str(target_frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot close {output_path.parent.name} duration: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    after = probe_continuity_frames(temporary, timeline_fps)
    if int(after["frames"]) != target_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"duration closure produced {after['frames']} frames, expected {target_frames}"
        )
    os.replace(temporary, output_path)
    return {
        "method": method,
        "before_frames": actual_frames,
        "target_frames": target_frames,
        "after_frames": int(after["frames"]),
        "adjustment_frames": delta,
    }


def normalize_provider_minimum_padding(
    input_path: Path,
    chunk: GenerationChunk,
    timeline_fps: int,
) -> dict[str, Any]:
    """Retime provider-minimum padding away while preserving both endpoints."""
    padding_frames = int(chunk.expected_provider_padding_frames)
    if padding_frames <= 0:
        return {
            "method": "none",
            "output_path": str(input_path),
            "provider_padding_frames": 0,
        }
    if chunk.expected_unique_frames is None:
        raise ValueError(
            f"{chunk.chunk_id} provider padding requires expected_unique_frames"
        )
    before = probe_continuity_frames(input_path, timeline_fps)
    source_duration = float(before["duration_s"])
    target_frames = int(chunk.expected_unique_frames)
    target_duration = target_frames / timeline_fps
    if source_duration <= 0:
        raise RuntimeError(f"{chunk.chunk_id} provider output has no positive duration")
    speed_ratio = target_duration / source_duration
    destination = input_path.with_name(
        f"{input_path.stem}.story_clock{input_path.suffix}"
    )
    temporary = destination.with_name(
        f"{destination.stem}.tmp{destination.suffix}"
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            (
                f"setpts={speed_ratio:.12f}*PTS,fps={timeline_fps},"
                f"trim=end_frame={target_frames},setpts=PTS-STARTPTS"
            ),
            "-frames:v",
            str(target_frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot normalize provider padding for {chunk.chunk_id}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    after = probe_continuity_frames(temporary, timeline_fps)
    if int(after["frames"]) != target_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{chunk.chunk_id} provider-padding normalization produced "
            f"{after['frames']} frames, expected {target_frames}"
        )
    os.replace(temporary, destination)
    return {
        "method": "provider_minimum_endpoint_preserving_retime",
        "output_path": str(destination),
        "source_frames": int(before["frames"]),
        "target_frames": target_frames,
        "provider_padding_frames": padding_frames,
    }


def _read_shot_meta(output_dir: Path, shot_id: str) -> dict[str, Any]:
    path = output_dir / "shots" / shot_id / "SHOT_META.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"continuity auto requires {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _generation_seed(request: ChunkExecutionRequest) -> int | None:
    scene = str(request.anchors.get("scene") or "").strip()
    if not scene:
        return None
    seed_material = json.dumps(
        {
            "scene": scene,
            "chunk_id": request.chunk.chunk_id,
            "repair_attempt": request.repair_attempt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % 2_147_483_647


def _chunk_unique_duration(chunk: GenerationChunk) -> float:
    """Recover effective story time without counting replay/reference overlap."""
    request_duration = float(chunk.target_duration_s)
    if chunk.requested_frames and chunk.expected_unique_frames:
        return request_duration * chunk.expected_unique_frames / chunk.requested_frames
    return request_duration


def _validated_seedance_chunk_duration(
    chunk: GenerationChunk,
    resource_id: str,
) -> int:
    request_duration, _unique_duration = SEEDANCE_2_CAPABILITIES.validate_chunk_durations(
        float(chunk.target_duration_s),
        _chunk_unique_duration(chunk),
        chunk.execution_strategy,
        resource_id=resource_id,
    )
    return int(request_duration)


def _chunk_duration(request: ChunkExecutionRequest) -> int:
    return _validated_seedance_chunk_duration(request.chunk, request.resource_id)


def _validate_seedance_continuity_plan(plan: ContinuityPlan) -> None:
    """Reject every invalid request before initializing a paid provider route."""
    errors: list[str] = []
    for shot in plan.shots:
        for chunk in shot.chunks:
            try:
                _validated_seedance_chunk_duration(chunk, chunk.chunk_id)
            except ValueError as exc:
                errors.append(str(exc))
    for bridge in plan.bridges:
        try:
            SEEDANCE_2_CAPABILITIES.validate_chunk_durations(
                bridge.target_duration_s,
                bridge.target_duration_s,
                "first_last_frame_bridge",
                resource_id=bridge.bridge_id,
            )
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        detail = "; ".join(errors[:8])
        if len(errors) > 8:
            detail += f"; and {len(errors) - 8} more"
        raise RuntimeError(
            "Seedance continuity preflight failed before any paid provider "
            f"submission: {detail}. Regenerate Phase 1-4 artifacts with the current "
            "secondary-storyboard contract"
        )


def _video_geometry(shot_meta: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve provider ratio and Bridge dimensions from the authored shot."""
    return resolve_video_geometry(shot_meta)


def _storyboard_group_for_shot(
    output_dir: Path,
    shot_id: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    path = output_dir / "STORYBOARD_GROUPS.json"
    if not path.is_file():
        return None, None
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    group_id = (contract.get("shot_to_group") or {}).get(shot_id)
    group = next(
        (
            item for item in contract.get("groups", [])
            if isinstance(item, dict) and item.get("group_id") == group_id
        ),
        None,
    )
    if group is None:
        return None, None
    board_value = group.get("storyboard_board")
    board = output_dir / str(board_value) if board_value else None
    return group, board if board is not None and board.is_file() else None


def _storyboard_group_prompt(group: dict[str, Any] | None, shot_id: str) -> str:
    if not group:
        return ""
    beats = [beat for beat in group.get("beats", []) if isinstance(beat, dict)]
    position = next((index for index, beat in enumerate(beats) if beat.get("shot_id") == shot_id), None)
    if position is None:
        return ""
    current = beats[position]
    previous = beats[position - 1] if position > 0 else None
    following = beats[position + 1] if position + 1 < len(beats) else None
    handoff = group.get("handoff_from_previous") or {}
    has_inner_beats = bool(current.get("storyboard_beats"))
    actions = " -> ".join(str(value) for value in current.get("generation_actions", []))
    lines = [
        f"[storyboard group {group.get('group_id')}; step {position + 1}/{len(beats)}]",
        "The group board is a chronological narrative map, not a request to perform every panel now.",
        (
            f"Previous shot final state: {previous.get('end_state', '')}"
            if previous
            else (
                f"Previous generation group {group.get('previous_group_id')} ended at: "
                f"{handoff.get('previous_end_state', '')}. Begin this fresh editorial cut from the declared current start state."
                if handoff
                else "This is the fresh story entry."
            )
        ),
        f"Current shot starting state: {current.get('start_state', '')}",
        (
            "The authoritative Pxx contract below defines the only action to execute now."
            if has_inner_beats
            else f"Execute only this current shot action contract: {actions or 'no authored body action'}"
        ),
        f"Current shot required result: {current.get('end_state', '')}",
        f"Next shot will begin from: {following.get('start_state', '')}" if following else "This is the final shot in the group.",
        "Do not jump to a later panel, combine later actions, replay the previous panel, or render a collage.",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def _chunk_prompt(
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
    group_prompt: str = "",
) -> str:
    prompt = str(shot_meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"{request.shot_id} SHOT_META.json has no prompt")
    strategy = request.chunk.execution_strategy
    continuation = {
        "multi_image": (
            "Generate the opening secondary beat from the ordered multi-image "
            "identity, environment, and current-storyboard references."
        ),
        "tail_video_extend": (
            "Extend the extracted tail section of the previous secondary-beat "
            "video without a reset, cut, replay, or re-entry."
        ),
        "first_last_frame_bridge": (
            "Generate only the transition between the completed source primary "
            "video's actual final frame and the completed target primary video's "
            "actual first frame. Both primary actions are already complete: do not "
            "add, finish, repeat, or reinterpret plot action."
        ),
    }.get(
        strategy,
        (
            "Generate the opening chunk of this editorial shot."
            if request.chunk.mode == "fresh"
            else "Continue directly from the reference video's final state without a reset or cut."
        ),
    )
    repair = ""
    if request.repair_attempt > 0:
        repair = (
            "\n[approved seam repair] Preserve subject position, scale, orientation, "
            "screen direction, camera framing, lighting, and surrounding motion state. "
            "Do not skip forward in time or reposition the subject before continuing."
        )
    beat_contract = ""
    if request.chunk.storyboard_beat_id:
        beat_contract = (
            f"\n[authoritative storyboard beat {request.chunk.storyboard_beat_id}] "
            f"Start state: {request.chunk.start_state or 'continue the supplied state'}. "
            f"Execute only this visible action: {request.chunk.action_prompt or 'natural scene progression'}. "
            f"Required end state: {request.chunk.end_state or 'complete that action'}. "
            "Do not execute another Pxx panel, skip ahead, or replay an earlier panel."
        )
    return apply_storyboard_motion_policy(
        f"{prompt}\n\n[continuity chunk {request.chunk.sequence}] {continuation}\n"
        f"{group_prompt}\n{request.memory_context}{beat_contract}{repair}"
    )


def _seedance_reference_image_payload(board_path: Path) -> tuple[bytes, str]:
    """Return a complete, uncropped board inside Seedance's aspect limits."""
    raw = board_path.read_bytes()
    try:
        import io

        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw)) as source:
            image_format = str(source.format or "PNG").upper()
            image = ImageOps.exif_transpose(source).convert("RGB")
        content_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(image_format, "image/png")
        width, height = image.size
        aspect = width / height
        if SEEDANCE_MIN_IMAGE_ASPECT <= aspect <= SEEDANCE_MAX_IMAGE_ASPECT:
            return raw, content_type

        safe_min = SEEDANCE_MIN_IMAGE_ASPECT + SEEDANCE_IMAGE_ASPECT_MARGIN
        safe_max = SEEDANCE_MAX_IMAGE_ASPECT - SEEDANCE_IMAGE_ASPECT_MARGIN
        target_width = max(width, math.ceil(height * safe_min))
        target_height = max(height, math.ceil(width / safe_max))
        corners = (
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        )
        fill = tuple(
            round(sum(pixel[channel] for pixel in corners) / 4)
            for channel in range(3)
        )
        canvas = Image.new("RGB", (target_width, target_height), fill)
        canvas.paste(image, ((target_width - width) // 2, (target_height - height) // 2))

        encoded = io.BytesIO()
        output_format = image_format if image_format in {"JPEG", "PNG", "WEBP"} else "PNG"
        save_options = (
            {"quality": 95, "optimize": True}
            if output_format in {"JPEG", "WEBP"}
            else {"optimize": True}
        )
        canvas.save(encoded, format=output_format, **save_options)
        print(
            "  [continuity] padded storyboard board for Seedance: "
            f"{width}x{height} ({aspect:.3f}) -> "
            f"{target_width}x{target_height} ({target_width / target_height:.3f})"
        )
        return encoded.getvalue(), content_type
    except (ImportError, OSError, ValueError):
        import mimetypes

        content_type = mimetypes.guess_type(board_path.name)[0] or "application/octet-stream"
        return raw, content_type


def _append_group_board_reference(
    content: list[dict[str, Any]],
    board_path: Path | None,
) -> list[dict[str, Any]]:
    if board_path is None:
        return content
    frame_roles = {
        item.get("role")
        for item in content
        if item.get("type") == "image_url"
    }
    if frame_roles & {"first_frame", "last_frame"}:
        # Seedance rejects first/last-frame control mixed with reference media.
        # The same group contract is already embedded in the text prompt, so the
        # board remains a media reference only for reference-only requests.
        return content
    from clients.tos_uploader import upload_image

    board_payload, content_type = _seedance_reference_image_payload(board_path)
    board_url = upload_image(board_payload, content_type)
    if not board_url:
        raise RuntimeError(f"failed to upload storyboard group board {board_path}")
    image_number = sum(item.get("type") == "image_url" for item in content) + 1
    directive = (
        f"图片{image_number}是本连续组按时间从左到右、从上到下排列的总体分镜板，"
        "仅用于确认当前镜头在故事中的位置和前后因果；必须按当前格中的主体动作箭头和摄影机箭头"
        "执行相应的运动方向与轨迹，但箭头、轨迹线和提示文字都是不可见的制作标注；"
        "不得复制整张拼图、不得同时演完其他格，不得在成片中生成箭头、辅助线、分格边框、编号或板中文字。"
    )
    text_item = next((item for item in content if item.get("type") == "text"), None)
    if text_item is None:
        content.insert(0, {"type": "text", "text": directive})
    else:
        text_item["text"] = f"{directive}\n{text_item.get('text', '')}"
    board_item = {
        "type": "image_url",
        "image_url": {"url": board_url},
        "role": "reference_image",
        "priority": "low",
    }
    video_index = next(
        (index for index, item in enumerate(content) if item.get("type") == "video_url"),
        len(content),
    )
    content.insert(video_index, board_item)
    return content


def _base_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    from tools.asset_packager import build_content_for_shot

    content_meta = dict(shot_meta)
    group, board_path = _storyboard_group_for_shot(output_dir, request.shot_id)
    content_meta["prompt"] = _chunk_prompt(
        request,
        shot_meta,
        _storyboard_group_prompt(group, request.shot_id),
    )
    from utils.privacy_visual_policy import (
        is_no_real_person_enabled,
        no_real_person_prompt_contract,
    )

    if is_no_real_person_enabled():
        content_meta["prompt"] = (
            f"{no_real_person_prompt_contract()}\n{content_meta['prompt']}"
        )
    strategy = request.chunk.execution_strategy
    if strategy == "first_last_frame_bridge":
        # Frame roles cannot be mixed with reference media in Seedance.  The
        # bridge helper adds the actual predecessor tail and next P01 below.
        return [{"type": "text", "text": content_meta["prompt"]}]
    if request.chunk.storyboard_image:
        content_meta["_storyboard_frame_path"] = request.chunk.storyboard_image
        content_meta["_storyboard_beat_id"] = request.chunk.storyboard_beat_id
        content_meta["generation_actions"] = [request.chunk.action_prompt]
        content_meta["gen_strategy"] = (
            "phantom"
            if strategy in {"multi_image", "tail_video_extend"}
            else "i2v"
        )
    reserved_group_board = 1 if board_path is not None else 0
    if request.chunk.mode == "native_extend":
        content_meta["_max_reference_images"] = (
            SEEDANCE_MAX_REFERENCE_IMAGES
            - CONTINUITY_ANCHOR_FRAME_COUNT
            - reserved_group_board
        )
    elif reserved_group_board:
        content_meta["_max_reference_images"] = (
            SEEDANCE_MAX_REFERENCE_IMAGES - reserved_group_board
        )
    content = build_content_for_shot(
        output_dir=output_dir,
        shot_id=request.shot_id,
        shot_meta=content_meta,
    )
    if not content:
        content = [{"type": "text", "text": content_meta["prompt"]}]
    return content


def _extension_content(
    content: Sequence[dict[str, Any]],
    previous_output_path: Path,
) -> list[dict[str, Any]]:
    from clients.tos_uploader import upload_media_file
    from quality.continuity_seam import (
        extract_ordered_video_frames,
        extract_video_tail_window,
    )

    with previous_output_path.open("rb") as predecessor:
        predecessor_hash = hashlib.file_digest(predecessor, "sha256").hexdigest()
    anchor_dir = previous_output_path.parent / "continuity_anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    asset_stem = f"{previous_output_path.stem}_{predecessor_hash[:16]}_tail_2000ms"
    tail_video_path = anchor_dir / f"{asset_stem}.mp4"
    if not tail_video_path.is_file() or tail_video_path.stat().st_size == 0:
        extract_video_tail_window(previous_output_path, tail_video_path, window_s=2.0)
    frame_paths = tuple(anchor_dir / f"{asset_stem}_frame_{index:02d}.jpg" for index in range(1, 4))
    if any(not path.is_file() or path.stat().st_size == 0 for path in frame_paths):
        extract_ordered_video_frames(tail_video_path, frame_paths)
    video_url = upload_media_file(tail_video_path, prefix="volcengine/video")
    if not video_url:
        raise RuntimeError(f"failed to upload continuity tail window {tail_video_path}")
    frame_urls = []
    for path in frame_paths:
        url = upload_media_file(path, prefix="volcengine/image")
        if not url:
            raise RuntimeError(f"failed to upload ordered continuity anchor {path}")
        frame_urls.append(url)

    base_images = [dict(item) for item in content if item.get("type") == "image_url"]
    base_images = base_images[
        : SEEDANCE_MAX_REFERENCE_IMAGES - CONTINUITY_ANCHOR_FRAME_COUNT
    ]
    first_anchor_number = len(base_images) + 1
    anchor_numbers = tuple(
        range(first_anchor_number, first_anchor_number + CONTINUITY_ANCHOR_FRAME_COUNT)
    )
    anchor_labels = "、".join(f"图片{number}" for number in anchor_numbers)
    directive = (
        f"向后延长视频1。{anchor_labels}是视频1末段按时间先后顺序截取的状态帧，"
        "只用于判断主体的速度、方向和连续运动。新生成内容的第一帧必须紧接视频1最后一帧，"
        f"并严格参考图片{anchor_numbers[-1]}的最终状态；保持主体位置、大小、朝向、水波相位、机位和灯光，"
        f"继续最后的速度与方向。不得重播视频1中的运动轨迹，不得回到图片{anchor_numbers[0]}或更早位置，"
        "不得让主体重新入画。\n"
    )
    text_items = [dict(item) for item in content if item.get("type") == "text"]
    normalized: list[dict[str, Any]] = text_items or [{"type": "text", "text": ""}]
    text_item = next((item for item in normalized if item.get("type") == "text"), None)
    if text_item is not None:
        text_item["text"] = f"{directive}{text_item.get('text', '')}"
    for item in base_images:
        item["role"] = "reference_image"
        item.pop("priority", None)
        normalized.append(item)
    normalized.extend(
        {
            "type": "image_url",
            "image_url": {"url": url},
            "role": "reference_image",
        }
        for url in frame_urls
    )
    normalized.append(
        {
            "type": "video_url",
            "video_url": {"url": video_url},
            "role": "reference_video",
        }
    )
    return normalized


def _first_last_bridge_content(
    content: Sequence[dict[str, Any]],
    previous_output_path: Path,
    target_output_path: Path,
) -> list[dict[str, Any]]:
    """Bind a transition to completed source-tail and target-head frames."""
    from clients.tos_uploader import upload_image
    from quality.continuity_seam import (
        extract_video_head_frame,
        extract_video_tail_frame,
    )

    for label, path in (
        ("source primary video", previous_output_path),
        ("target primary video", target_output_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} missing: {path}")
    with previous_output_path.open("rb") as predecessor:
        predecessor_hash = hashlib.file_digest(predecessor, "sha256").hexdigest()
    anchor_dir = previous_output_path.parent / "continuity_anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    tail_frame = anchor_dir / (
        f"{previous_output_path.stem}_{predecessor_hash[:16]}_final.jpg"
    )
    if not tail_frame.is_file() or tail_frame.stat().st_size == 0:
        extract_video_tail_frame(previous_output_path, tail_frame)
    with target_output_path.open("rb") as target:
        target_hash = hashlib.file_digest(target, "sha256").hexdigest()
    head_frame = anchor_dir / (
        f"{target_output_path.parent.name}_{target_hash[:16]}_first.jpg"
    )
    if not head_frame.is_file() or head_frame.stat().st_size == 0:
        extract_video_head_frame(target_output_path, head_frame)

    first_payload, first_content_type = _seedance_reference_image_payload(tail_frame)
    last_payload, last_content_type = _seedance_reference_image_payload(head_frame)
    first_url = upload_image(first_payload, first_content_type)
    last_url = upload_image(last_payload, last_content_type)
    if not first_url or not last_url:
        raise RuntimeError(
            "failed to upload first/last frames for secondary-storyboard bridge"
        )
    directive = (
        "图片1来自已完成的上一一级分镜视频真实尾帧，必须作为新视频第一帧；"
        "图片2来自已完成的下一一级分镜视频真实首帧，必须作为新视频最后一帧。"
        "只生成两帧之间的连续摄影机与动作过渡，不得重复两侧已完成剧情，"
        "不得新增剧情、角色、道具、文字、箭头、轨迹线、编号或分格标记。\n"
    )
    text_value = next(
        (str(item.get("text") or "") for item in content if item.get("type") == "text"),
        "",
    )
    return [
        {"type": "text", "text": f"{directive}{text_value}"},
        {
            "type": "image_url",
            "image_url": {"url": first_url},
            "role": "first_frame",
            "priority": "high",
        },
        {
            "type": "image_url",
            "image_url": {"url": last_url},
            "role": "last_frame",
            "priority": "high",
        },
    ]


def _provider_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any], int | None, int]:
    shot_meta = _read_shot_meta(output_dir, request.shot_id)
    content = _base_content(output_dir, request, shot_meta)
    strategy = request.chunk.execution_strategy
    if strategy == "first_last_frame_bridge":
        if request.previous_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no predecessor video")
        if request.target_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no completed target video")
        content = _first_last_bridge_content(
            content,
            request.previous_output_path,
            request.target_output_path,
        )
    elif request.chunk.mode == "native_extend":
        if request.previous_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no predecessor video")
        content = _extension_content(content, request.previous_output_path)
    if strategy != "first_last_frame_bridge":
        _, board_path = _storyboard_group_for_shot(output_dir, request.shot_id)
        content = _append_group_board_reference(content, board_path)
    return content, shot_meta, _generation_seed(request), _chunk_duration(request)


def _task_payload(
    request: ChunkExecutionRequest,
    *,
    model: str,
    duration: int,
    seed: int | None,
) -> dict[str, Any]:
    unique_duration = _chunk_unique_duration(request.chunk)
    requested_frames = request.chunk.requested_frames
    overlap_duration = (
        duration * request.chunk.expected_overlap_frames / requested_frames
        if requested_frames
        else 0.0
    )
    padding_duration = (
        duration * request.chunk.expected_provider_padding_frames / requested_frames
        if requested_frames
        else 0.0
    )
    return {
        "shot_id": request.shot_id,
        "chunk_id": request.chunk.chunk_id,
        "resource_id": request.resource_id,
        "input_fingerprint": request.input_fingerprint,
        "output_path": str(request.output_path),
        "model": model,
        "mode": request.chunk.mode,
        "execution_strategy": request.chunk.execution_strategy,
        "continuation_contract": (
            "next_primary_p01_first_last_frame_v1"
            if request.chunk.execution_strategy == "first_last_frame_bridge"
            else "tail_window_ordered_frames_v1"
            if request.chunk.mode == "native_extend"
            else "multi_image_storyboard_identity_v1"
            if request.chunk.execution_strategy == "multi_image"
            else None
        ),
        "tail_window_seconds": (
            SEEDANCE_2_CAPABILITIES.tail_reference_window_s
            if request.chunk.execution_strategy == "tail_video_extend"
            else None
        ),
        "ordered_frame_fractions": (
            list(SEEDANCE_2_CAPABILITIES.tail_reference_frame_fractions)
            if request.chunk.execution_strategy == "tail_video_extend"
            else None
        ),
        "bridge_target_beat_id": request.chunk.bridge_target_beat_id,
        "duration": duration,
        "provider_request_duration_s": duration,
        "effective_story_duration_s": round(unique_duration, 6),
        "reference_overlap_duration_s": round(overlap_duration, 6),
        "provider_minimum_padding_duration_s": round(padding_duration, 6),
        "seed": seed,
        "repair_attempt": request.repair_attempt,
        "privacy_fallback": "drop_provider_rejected_images_once_v1",
    }


def _provider_input_context(
    output_dir: Path,
    shot_id: str,
    chunk_id: str,
    *,
    storyboard_image: str | None = None,
    bridge_target_storyboard_image: str | None = None,
) -> dict[str, str | None]:
    shot_meta = output_dir / "shots" / shot_id / "SHOT_META.json"
    storyboard_frame = output_dir / "storyboard_images" / f"{shot_id}.png"
    group, group_board = _storyboard_group_for_shot(output_dir, shot_id)
    group_contract = output_dir / "STORYBOARD_GROUPS.json"
    from utils.privacy_visual_policy import (
        NO_REAL_PERSON_POLICY,
        is_no_real_person_enabled,
    )

    def optional_hash(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = output_dir / path
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    return {
        "shot_meta_sha256": hashlib.sha256(shot_meta.read_bytes()).hexdigest(),
        "storyboard_frame_sha256": (
            hashlib.sha256(storyboard_frame.read_bytes()).hexdigest()
            if storyboard_frame.is_file()
            else None
        ),
        "storyboard_group_id": str(group.get("group_id")) if group else None,
        "storyboard_groups_sha256": (
            hashlib.sha256(group_contract.read_bytes()).hexdigest()
            if group_contract.is_file()
            else None
        ),
        "storyboard_group_board_sha256": (
            hashlib.sha256(group_board.read_bytes()).hexdigest()
            if group_board is not None
            else None
        ),
        "chunk_id": chunk_id,
        "secondary_storyboard_image_sha256": optional_hash(storyboard_image),
        "bridge_target_storyboard_image_sha256": optional_hash(
            bridge_target_storyboard_image
        ),
        "generation_seed_strategy": "scene_chunk_repair_v1",
        "continuation_contract": "tail_window_ordered_frames_v1",
        "tail_window_seconds": "2.0",
        "ordered_frame_fractions": "0.2,0.6,0.95",
        "visual_identity_policy": (
            NO_REAL_PERSON_POLICY if is_no_real_person_enabled() else None
        ),
    }


def _privacy_rejected_image_indices(
    content: Sequence[dict[str, Any]],
    error: BaseException,
) -> tuple[int, ...]:
    """Resolve only provider-addressed image items from a privacy rejection."""
    message = str(error)
    privacy_markers = (
        "PrivacyInformation",
        "InputImageSensitiveContentDetected",
        "real person",
    )
    if not any(marker.casefold() in message.casefold() for marker in privacy_markers):
        return ()
    indices = []
    for raw_index in re.findall(r"content\[(\d+)\]", message):
        index = int(raw_index)
        if (
            0 <= index < len(content)
            and content[index].get("type") == "image_url"
            and index not in indices
        ):
            indices.append(index)
    return tuple(indices)


def _without_content_indices(
    content: Sequence[dict[str, Any]],
    rejected_indices: Sequence[int],
) -> list[dict[str, Any]]:
    rejected = set(rejected_indices)
    return [dict(item) for index, item in enumerate(content) if index not in rejected]


def _copyright_policy_violation_kind(error: BaseException) -> str | None:
    """Classify only the two copyright-policy failures with safe remediations."""
    message = str(error).casefold()
    if "policyviolation" not in message:
        return None
    if "outputaudiosensitivecontentdetected" in message:
        return "output_audio"
    if "inputvideosensitivecontentdetected" in message:
        return "input_video"
    return None


def _copyright_rejected_video_indices(
    content: Sequence[dict[str, Any]],
    error: BaseException,
) -> tuple[int, ...]:
    """Resolve provider-addressed video items without dropping unrelated inputs."""
    if _copyright_policy_violation_kind(error) != "input_video":
        return ()
    indices: list[int] = []
    for raw_index in re.findall(r"content\[(\d+)\]", str(error)):
        index = int(raw_index)
        if (
            0 <= index < len(content)
            and content[index].get("type") == "video_url"
            and index not in indices
        ):
            indices.append(index)
    return tuple(indices)


def _sanitize_music_language(value: str) -> str:
    """Remove soundtrack requests while preserving the authored body action."""
    substitutions = (
        (
            r"(?i)\b(?:soundtrack|music|song|melody|lyrics?|bpm|rhythm)\b",
            "visual motion timing",
        ),
        (r"配乐|音乐|歌曲|旋律|歌词", "动作时序提示"),
        (r"节奏", "动作时序"),
    )
    sanitized = value
    for pattern, replacement in substitutions:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def _copyright_repair_content(
    content: Sequence[dict[str, Any]],
    *,
    audio_safe: bool,
    rejected_video_indices: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Build a bounded compliant retry without reusing a rejected video item."""
    corrected = _without_content_indices(content, rejected_video_indices)
    if rejected_video_indices:
        removed = [
            content[index]
            for index in rejected_video_indices
            if 0 <= index < len(content)
        ]
        if not removed or any(item.get("type") != "video_url" for item in removed):
            raise RuntimeError("copyright fallback did not resolve a rejected video item")
    frame_only = bool(rejected_video_indices) or any(
        _FRAME_ONLY_CONTINUITY_CONTRACT in str(item.get("text") or "")
        for item in content
        if item.get("type") == "text"
    )
    text_contract = (
        (_FRAME_ONLY_CONTINUITY_CONTRACT if frame_only else "")
        + (_COPYRIGHT_SAFE_AUDIO_CONTRACT if audio_safe else "")
    )
    for item in corrected:
        if item.get("type") != "text":
            continue
        original = str(item.get("text") or "")
        original = original.replace(_COPYRIGHT_SAFE_AUDIO_CONTRACT, "").replace(
            _FRAME_ONLY_CONTINUITY_CONTRACT,
            "",
        )
        item["text"] = text_contract + (
            _sanitize_music_language(original) if audio_safe else original
        )
    if text_contract and not any(item.get("type") == "text" for item in corrected):
        corrected.insert(0, {"type": "text", "text": text_contract})
    return corrected


def _copyright_repair_seed(
    seed: int | None,
    repairs: Sequence[dict[str, Any]],
) -> int | None:
    """Deterministically vary repaired output while keeping reruns reproducible."""
    if seed is None or not repairs:
        return seed
    material = json.dumps(
        {"seed": seed, "repairs": list(repairs)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) % 2_147_483_647


def _direct_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import seedance_client
    from utils.config import SEEDANCE_MODEL, get_api_key_or_raise
    from utils.video_validation import is_valid_video

    api_key = get_api_key_or_raise("ARK_AGENT")
    model = SEEDANCE_MODEL
    fallback_workers = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    capacity = CapacityTable.for_seedance_video(fallback_workers).get("seedance", "video")
    slots = SlotTable()
    leases = CrossProcessSlotTable(default_capacity_lease_path())

    def execute(request: ChunkExecutionRequest) -> ChunkExecutionResult:
        seed = _generation_seed(request)
        duration = _chunk_duration(request)
        ratio, _width, _height = _video_geometry(
            _read_shot_meta(output_dir, request.shot_id)
        )
        payload = _task_payload(request, model=model, duration=duration, seed=seed)
        payload["ratio"] = ratio
        run_id = str(output_dir.resolve())
        repairs: list[dict[str, Any]] = []
        content_override: list[dict[str, Any]] | None = None
        execution = None
        with slots.reserve("seedance", "video", request.resource_id, capacity=capacity):
            with leases.reserve(
                "seedance",
                "video",
                request.resource_id,
                capacity=capacity,
            ):
                for policy_attempt in range(MAX_COPYRIGHT_POLICY_REPAIRS + 1):
                    try:
                        if content_override is None:
                            content, _shot_meta, _seed, _duration = _provider_content(
                                output_dir,
                                request,
                            )
                        else:
                            content = [dict(item) for item in content_override]
                    except Exception as exc:
                        raise ProviderPreparationError(
                            "cannot prepare provider content for "
                            f"{request.resource_id}: {exc}"
                        ) from exc

                    attempt_seed = _copyright_repair_seed(seed, repairs)
                    submitted_content: list[dict[str, Any]] = []
                    attempt_payload = (
                        payload
                        if policy_attempt == 0
                        else {
                            **payload,
                            "seed": attempt_seed,
                            "copyright_policy_repair_version": (
                                COPYRIGHT_POLICY_REPAIR_VERSION
                            ),
                            "copyright_policy_repair_attempt": policy_attempt,
                            "copyright_policy_repairs": [
                                dict(item) for item in repairs
                            ],
                        }
                    )
                    attempt_resource_id = (
                        request.resource_id
                        if policy_attempt == 0
                        else f"{request.resource_id}_CP{policy_attempt:02d}"
                    )

                    def submit() -> str:
                        def submit_selected(
                            selected: Sequence[dict[str, Any]],
                        ) -> str:
                            submitted_content[:] = [dict(item) for item in selected]
                            return seedance_client.submit_content(
                                selected,
                                api_key=api_key,
                                model=model,
                                duration=duration,
                                ratio=ratio,
                                seed=attempt_seed,
                            )

                        try:
                            return submit_selected(content)
                        except Exception as exc:
                            rejected_indices = _privacy_rejected_image_indices(
                                content,
                                exc,
                            )
                            if not rejected_indices:
                                raise
                            corrected_content = _without_content_indices(
                                content,
                                rejected_indices,
                            )
                            print(
                                "  🛡 Seedance 隐私纠偏：移除服务商明确拒绝的参考图 "
                                + ", ".join(
                                    f"content[{index}]"
                                    for index in rejected_indices
                                )
                                + "，限次重试 1 次",
                                flush=True,
                            )
                            return submit_selected(corrected_content)

                    try:
                        succeeded = task_store.find_succeeded(
                            run_id=run_id,
                            task_type="video.generate",
                            resource_id=attempt_resource_id,
                            payload=attempt_payload,
                            provider_id="seedance",
                        )
                        failed = (
                            None
                            if succeeded is not None
                            else task_store.find_failed(
                                run_id=run_id,
                                task_type="video.generate",
                                resource_id=attempt_resource_id,
                                payload=attempt_payload,
                                provider_id="seedance",
                            )
                        )
                        if failed is not None and _copyright_policy_violation_kind(
                            RuntimeError(failed.error_message or "")
                        ):
                            raise RuntimeError(failed.error_message or "policy violation")
                        execution = execute_seedance_video_task(
                            task_store,
                            run_id=run_id,
                            resource_id=attempt_resource_id,
                            payload=attempt_payload,
                            provider_endpoint=seedance_client.BASE_URL,
                            output_path=request.output_path,
                            submit=submit,
                            poll=partial(seedance_client.poll, api_key=api_key),
                            download=seedance_client.download,
                            validate_output=is_valid_video,
                        )
                        break
                    except Exception as exc:
                        violation_kind = _copyright_policy_violation_kind(exc)
                        if (
                            violation_kind is None
                            or len(repairs) >= MAX_COPYRIGHT_POLICY_REPAIRS
                        ):
                            raise
                        submitted = submitted_content or content
                        if violation_kind == "output_audio":
                            if any(
                                item.get("reason_code")
                                == "OutputAudioSensitiveContentDetected.PolicyViolation"
                                for item in repairs
                            ):
                                raise
                            content_override = _copyright_repair_content(
                                submitted,
                                audio_safe=True,
                            )
                            repair = {
                                "attempt": len(repairs) + 1,
                                "reason_code": (
                                    "OutputAudioSensitiveContentDetected.PolicyViolation"
                                ),
                                "policy": "original_ambient_no_music_v1",
                                "removed_content_indices": [],
                            }
                        else:
                            rejected_video_indices = (
                                _copyright_rejected_video_indices(submitted, exc)
                            )
                            if not rejected_video_indices:
                                raise
                            content_override = _copyright_repair_content(
                                submitted,
                                audio_safe=True,
                                rejected_video_indices=rejected_video_indices,
                            )
                            repair = {
                                "attempt": len(repairs) + 1,
                                "reason_code": (
                                    "InputVideoSensitiveContentDetected.PolicyViolation"
                                ),
                                "policy": "drop_rejected_video_keep_ordered_frames_v1",
                                "removed_content_indices": list(
                                    rejected_video_indices
                                ),
                                "retained_reference_images": sum(
                                    item.get("type") == "image_url"
                                    for item in content_override
                                ),
                            }
                        repairs.append(repair)
                        print(
                            "  🛡 Seedance 版权审核合规重生成："
                            f"{repair['reason_code']} → {repair['policy']} "
                            f"({len(repairs)}/{MAX_COPYRIGHT_POLICY_REPAIRS})",
                            flush=True,
                        )
                if execution is None:
                    raise RuntimeError(
                        f"{request.resource_id} copyright policy retry exited without result"
                    )
        return ChunkExecutionResult(
            output_path=Path(execution.output_path),
            provider_task_id=execution.provider_job_id,
            copyright_policy_repairs=tuple(dict(item) for item in repairs),
        )

    return execute


def _bridge_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import local_video_client
    from utils.video_validation import is_valid_video

    capacity = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    slots = SlotTable()

    def execute(request: ChunkExecutionRequest) -> ChunkExecutionResult:
        seed = _generation_seed(request)
        duration = _chunk_duration(request)
        model = "seedance"
        ratio, width, height = _video_geometry(
            _read_shot_meta(output_dir, request.shot_id)
        )
        payload = _task_payload(request, model=model, duration=duration, seed=seed)
        payload.update({"ratio": ratio, "width": width, "height": height})

        def generate(**runtime_kwargs: Any) -> str | dict[str, Any]:
            content, _shot_meta, _seed, _duration = _provider_content(output_dir, request)
            prompt = next(
                (str(item.get("text")) for item in content if item.get("type") == "text"),
                "",
            )
            return local_video_client.generate_video(
                prompt=prompt,
                output_path=str(request.output_path),
                seed=seed if seed is not None else -1,
                duration=duration,
                width=width,
                height=height,
                fps=24,
                content=content,
                batch_id=output_dir.name,
                model=model,
                **runtime_kwargs,
            )

        with slots.reserve("bridge", "video", request.resource_id, capacity=capacity):
                execution = execute_bridge_video_task(
                task_store,
                run_id=str(output_dir.resolve()),
                resource_id=request.resource_id,
                payload=payload,
                provider_endpoint=local_video_client.get_api_url(),
                    output_path=request.output_path,
                    generate=generate,
                    validate_output=is_valid_video,
                )
        return ChunkExecutionResult(
            output_path=Path(execution.output_path),
            provider_task_id=execution.provider_job_id,
        )

    return execute


def materialize_continuity_shot(
    chunk_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Concatenate accepted internal boundaries into one editorial-shot clip."""
    if not chunk_paths:
        raise ValueError("cannot materialize a shot without chunks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    command = ["ffmpeg", "-y"]
    for path in chunk_paths:
        command.extend(["-i", str(path)])
    streams = "".join(f"[{index}:v:0]" for index in range(len(chunk_paths)))
    command.extend(
        [
            "-filter_complex",
            f"{streams}concat=n={len(chunk_paths)}:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot materialize continuity shot {output_path}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    os.replace(temporary, output_path)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_phase8_seam_decisions(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "CONTINUITY_SEAM_DECISIONS.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != SEAM_DECISIONS_KIND:
        raise ValueError(f"unsupported Phase 8 continuity decisions in {path}")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"{path} must contain a decisions object")
    for boundary_id, decision in decisions.items():
        if not isinstance(boundary_id, str) or not isinstance(decision, dict):
            raise ValueError(f"{path} contains an invalid boundary decision")
        if decision.get("action") != "hard_trim":
            raise ValueError(f"{boundary_id} has unsupported Phase 8 action")
        if int(decision.get("trim_frames") or 0) <= 0:
            raise ValueError(f"{boundary_id} requires a positive trim_frames value")
    return decisions


def _render_phase8_hard_trim(
    following: Path,
    output_path: Path,
    *,
    trim_frames: int,
    timeline_fps: int,
) -> None:
    """Materialize a frame-addressed, interpolation-free Phase 8 cut."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.phase8{output_path.suffix}")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(following),
            "-vf",
            (
                f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS,"
                f"fps={timeline_fps},format=yuv420p"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            "cannot apply Phase 8 continuity trim: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    os.replace(temporary, output_path)


def _continuity_bridge_preparer(
    output_dir: Path,
    *,
    planned_overlap_seconds: float = 0.0,
    timeline_fps: int = 24,
) -> Callable[[Path, Path, str], dict[str, Any]] | None:
    """Build the optional local trim/interpolation step used before seam scoring."""
    phase8_decisions = _load_phase8_seam_decisions(output_dir)
    mode = os.environ.get(CONTINUITY_BRIDGE_ENV, "off").strip().lower()
    if mode not in CONTINUITY_BRIDGE_MODES:
        expected = ", ".join(sorted(CONTINUITY_BRIDGE_MODES))
        raise ValueError(f"{CONTINUITY_BRIDGE_ENV} must be one of {expected}, got {mode!r}")
    if mode == "off" and not phase8_decisions:
        return None
    raw_candidates = os.environ.get("HONCUT_CONTINUITY_BRIDGE_FRAMES", "4,6,8")
    try:
        candidate_frames = tuple(int(item.strip()) for item in raw_candidates.split(","))
    except ValueError as exc:
        raise ValueError(
            "HONCUT_CONTINUITY_BRIDGE_FRAMES must be comma-separated integers"
        ) from exc
    candidate_frames = tuple(sorted(set(candidate_frames)))
    if not candidate_frames or any(frame < 4 or frame > 24 for frame in candidate_frames):
        raise ValueError("HONCUT_CONTINUITY_BRIDGE_FRAMES values must be between 4 and 24")

    def prepare(previous: Path, following: Path, boundary_id: str) -> dict[str, Any]:
        from quality.continuity_bridge import detect_replayed_prefix, repair_continuity_boundary

        boundary_dir = output_dir / "continuity_bridges" / boundary_id
        output_path = boundary_dir / "effective_following.mp4"
        receipt_path = boundary_dir / "CONTINUITY_BRIDGE.json"
        phase8_decision = phase8_decisions.get(boundary_id)
        input_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "previous_sha256": _file_sha256(previous),
                    "following_sha256": _file_sha256(following),
                    "candidate_frames": candidate_frames,
                    "planned_overlap_seconds": planned_overlap_seconds,
                    "phase8_decision": phase8_decision,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if receipt_path.is_file():
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            cached_output = Path(str(cached.get("output_path") or following))
            if (
                cached.get("status") != "fallback"
                and cached.get("input_fingerprint") == input_fingerprint
                and cached_output.is_file()
                and cached_output.stat().st_size > 0
            ):
                return cached

        boundary_dir.mkdir(parents=True, exist_ok=True)
        if phase8_decision is not None:
            expected = phase8_decision.get("source_fingerprint") or {}
            actual = {
                "previous_sha256": _file_sha256(previous),
                "following_sha256": _file_sha256(following),
            }
            if expected != actual:
                raise RuntimeError(
                    f"{boundary_id} Phase 8 decision is stale; rerun seam adjudication"
                )
            trim_frames = int(phase8_decision["trim_frames"])
            _render_phase8_hard_trim(
                following,
                output_path,
                trim_frames=trim_frames,
                timeline_fps=timeline_fps,
            )
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "adjudicated_trim",
                "reason": phase8_decision["reason"],
                "input_fingerprint": input_fingerprint,
                "output_path": str(output_path),
                "selected_bridge_frames": None,
                "ghost_safe_fallback": True,
                "phase8_decision": phase8_decision,
                "overlap": {
                    "detected": True,
                    "overlap_seconds": trim_frames / timeline_fps,
                    "overlap_frames": trim_frames,
                    "source": "phase8_temporal_adjudication",
                    "reason": phase8_decision["reason"],
                },
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return receipt
        if mode == "off":
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "skipped",
                "reason": "continuity bridge is disabled for this unadjudicated boundary",
                "input_fingerprint": input_fingerprint,
                "output_path": str(following),
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return receipt
        try:
            overlap = detect_replayed_prefix(previous, following)
            if not overlap["detected"]:
                if planned_overlap_seconds <= 0:
                    receipt = {
                        "kind": "honcut.continuity_bridge.v1",
                        "status": "skipped",
                        "reason": overlap["reason"],
                        "overlap": overlap,
                        "input_fingerprint": input_fingerprint,
                        "output_path": str(following),
                    }
                else:
                    trial = repair_continuity_boundary(
                        previous,
                        following,
                        output_path,
                        work_dir=boundary_dir / "work",
                        overlap_seconds=planned_overlap_seconds,
                        candidate_frames=candidate_frames,
                    )
                    baseline = float(trial["baseline_boundary_frame_mae"])
                    selected = float(trial["selected_boundary_frame_mae"])
                    trimmed = float(trial["trimmed_boundary_frame_mae"])
                    if (
                        trial.get("improved")
                        and selected <= baseline * 0.5
                        and trimmed < baseline
                    ):
                        # The interpolation result is evidence that the planned
                        # cut point is useful, not necessarily a safe output.
                        # With an uncertain trajectory, synthesized in-betweens
                        # can create double contours around moving subjects.
                        # Prefer the decoded hard trim: one small discontinuity
                        # is less objectionable than a multi-frame ghost trail.
                        trimmed_path = boundary_dir / "work" / "trimmed_hard_cut.mp4"
                        temporary = output_path.with_name(
                            f"{output_path.stem}.ghost_safe{output_path.suffix}"
                        )
                        shutil.copy2(trimmed_path, temporary)
                        os.replace(temporary, output_path)
                        receipt = trial
                        receipt["status"] = "trimmed"
                        receipt["output_path"] = str(output_path)
                        receipt["selected_bridge_frames"] = None
                        receipt["selected_boundary_frame_mae"] = trimmed
                        receipt["ghost_safe_fallback"] = True
                        receipt["detector_overlap"] = overlap
                        receipt["overlap"] = {
                            "detected": True,
                            "overlap_seconds": planned_overlap_seconds,
                            "source": "phase4_planned_budget",
                            "reason": (
                                "planned overlap trial reduced boundary-frame error "
                                "by at least fifty percent; emitted a hard trim to "
                                "avoid interpolation ghosting"
                            ),
                        }
                    else:
                        receipt = {
                            "kind": "honcut.continuity_bridge.v1",
                            "status": "skipped",
                            "reason": (
                                "planned overlap trial did not reduce boundary-frame "
                                "error by at least fifty percent"
                            ),
                            "overlap": overlap,
                            "planned_overlap_trial": trial,
                            "input_fingerprint": input_fingerprint,
                            "output_path": str(following),
                        }
            else:
                receipt = repair_continuity_boundary(
                    previous,
                    following,
                    output_path,
                    work_dir=boundary_dir / "work",
                    overlap_seconds=float(overlap["overlap_seconds"]),
                    candidate_frames=candidate_frames,
                )
                receipt["input_fingerprint"] = input_fingerprint
                receipt["overlap"] = overlap
            receipt["input_fingerprint"] = input_fingerprint
        except Exception as exc:
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "fallback",
                "reason": f"local continuity bridge failed: {exc}",
                "input_fingerprint": input_fingerprint,
                "output_path": str(following),
            }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt

    return prepare


def execute_phase6_auto_continuity(
    output_dir: str | Path,
    plan: ContinuityPlan,
    calibration: SeamCalibration | None,
) -> dict[str, Any]:
    """Run continuity groups through exactly one configured provider."""
    root = Path(output_dir)
    provider = os.environ.get("VIDEO_PROVIDER", "seedance").strip().lower()
    if provider == "seedance":
        executor_factory = _direct_seedance_executor
    elif provider == "bridge":
        executor_factory = _bridge_seedance_executor
    else:
        raise RuntimeError(
            "continuity auto requires VIDEO_PROVIDER=seedance or bridge; "
            f"{provider!r} has no verified native video-extension contract"
        )

    _validate_seedance_continuity_plan(plan)

    requires_video_upload = any(
        chunk.mode == "native_extend"
        and chunk.execution_strategy != "first_last_frame_bridge"
        for shot in plan.shots
        for chunk in shot.chunks
    )
    if requires_video_upload:
        from clients.tos_uploader import is_media_upload_configured

        if not is_media_upload_configured():
            raise RuntimeError(
                "continuity auto preflight requires TOS_ACCESS_KEY, "
                "TOS_SECRET_KEY, and TOS_BUCKET before any paid provider "
                "submission for native_extend chunks"
            )

    planned_overlap_seconds = max(
        (
            chunk.expected_overlap_frames / plan.timeline_fps
            for shot in plan.shots
            for chunk in shot.chunks
        ),
        default=0.0,
    )
    prepare_seam = _continuity_bridge_preparer(
        root,
        planned_overlap_seconds=planned_overlap_seconds,
        timeline_fps=plan.timeline_fps,
    )
    missing_replay_overlap = [
        chunk.chunk_id
        for shot in plan.shots
        for chunk in shot.chunks
        if (
            chunk.mode == "native_extend"
            and chunk.execution_strategy == "legacy"
            and chunk.expected_overlap_frames <= 0
        )
    ]
    if prepare_seam is not None and missing_replay_overlap:
        raise RuntimeError(
            "continuity bridge requires an overlap-budgeted CONTINUITY_PLAN.json; "
            "rerun Phase 4 with HONCUT_CONTINUITY_BRIDGE=auto; missing overlap on "
            + ", ".join(missing_replay_overlap[:6])
        )
    task_store = GenerationTaskStore(root / "runtime.db")
    execute_chunk = executor_factory(root, task_store)

    max_repairs = int(os.environ.get("HONCUT_CONTINUITY_MAX_REPAIRS", "1"))
    if not 0 <= max_repairs <= 3:
        raise ValueError("HONCUT_CONTINUITY_MAX_REPAIRS must be between 0 and 3")
    workers = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    report = execute_continuity_plan(
        plan,
        root,
        execute_chunk=execute_chunk,
        prepare_seam=prepare_seam,
        materialize_shot=materialize_continuity_shot,
        probe_frames=probe_continuity_frames,
        finalize_shot=finalize_continuity_shot,
        normalize_chunk=normalize_provider_minimum_padding,
        chunk_context=lambda shot, chunk: _provider_input_context(
            root,
            shot.shot_id,
            chunk.chunk_id,
            storyboard_image=chunk.storyboard_image,
            bridge_target_storyboard_image=chunk.bridge_target_storyboard_image,
        ),
        seam_calibration=(
            calibration.model_dump(mode="json") if calibration is not None else None
        ),
        max_seam_repairs=max_repairs,
        max_workers=workers,
    )
    report.update(
        {
            "provider": provider,
            "mode": "continuity_auto",
            "continuity_bridge": os.environ.get(CONTINUITY_BRIDGE_ENV, "off").strip().lower(),
            "calibration_fingerprint": (
                calibration.dataset_fingerprint if calibration is not None else None
            ),
        }
    )
    return report
