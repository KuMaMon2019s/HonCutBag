"""Phase 6 provider adapters for calibrated continuity chunk execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from schemas.continuity import ContinuityPlan

CONTINUITY_BRIDGE_ENV = "HONCUT_CONTINUITY_BRIDGE"
CONTINUITY_BRIDGE_MODES = {"off", "auto"}
SEAM_DECISIONS_KIND = "honcut.continuity_seam_decisions.v1"
SEEDANCE_MAX_REFERENCE_IMAGES = 9
CONTINUITY_ANCHOR_FRAME_COUNT = 3


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
    """Close a materialized editorial shot to its exact frame budget."""
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
    if delta > max(2, math.ceil(target_frames * 0.02)):
        raise RuntimeError(
            f"{output_path.parent.name} is short by {delta} frames; "
            "additional continuation generation is required"
        )

    temporary = output_path.with_name(f"{output_path.stem}.duration_closure{output_path.suffix}")
    if delta < 0:
        video_filter = f"trim=end_frame={target_frames},setpts=PTS-STARTPTS"
        method = "exact_tail_trim"
    else:
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


def _chunk_duration(request: ChunkExecutionRequest) -> int:
    duration = float(request.chunk.target_duration_s)
    rounded = round(duration)
    if not math.isclose(duration, rounded, abs_tol=1e-6):
        raise ValueError(
            f"{request.resource_id} duration {duration} cannot be represented by Seedance seconds"
        )
    return int(rounded)


def _video_geometry(shot_meta: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve provider ratio and Bridge dimensions from the authored shot."""
    width = int(shot_meta.get("width") or 0)
    height = int(shot_meta.get("height") or 0)
    ratio = str(
        shot_meta.get("aspect_ratio") or shot_meta.get("ratio") or ""
    ).strip()
    if width > 0 and height > 0:
        divisor = math.gcd(width, height)
        return ratio or f"{width // divisor}:{height // divisor}", width, height
    import re

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*", ratio)
    if match:
        left, right = float(match.group(1)), float(match.group(2))
        if left > 0 and right > 0:
            if left >= right:
                width, height = 1280, max(2, round(1280 * right / left / 2) * 2)
            else:
                height, width = 1280, max(2, round(1280 * left / right / 2) * 2)
            return f"{match.group(1)}:{match.group(2)}", width, height
    return "16:9", 1280, 720


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
    continuation = (
        "Generate the opening chunk of this editorial shot."
        if request.chunk.mode == "fresh"
        else "Continue directly from the reference video's final state without a reset or cut."
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
    return (
        f"{prompt}\n\n[continuity chunk {request.chunk.sequence}] {continuation}\n"
        f"{group_prompt}\n{request.memory_context}{beat_contract}{repair}"
    )


def _append_group_board_reference(
    content: list[dict[str, Any]],
    board_path: Path | None,
) -> list[dict[str, Any]]:
    if board_path is None:
        return content
    from clients.tos_uploader import upload_image

    try:
        from PIL import Image

        with Image.open(board_path) as board_image:
            image_format = str(board_image.format or "").upper()
        content_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(image_format, "application/octet-stream")
    except (ImportError, OSError, ValueError):
        import mimetypes

        content_type = mimetypes.guess_type(board_path.name)[0] or "application/octet-stream"
    board_url = upload_image(board_path.read_bytes(), content_type)
    if not board_url:
        raise RuntimeError(f"failed to upload storyboard group board {board_path}")
    image_number = sum(item.get("type") == "image_url" for item in content) + 1
    directive = (
        f"图片{image_number}是本连续组按时间从左到右、从上到下排列的总体分镜板，"
        "仅用于确认当前镜头在故事中的位置和前后因果；不得复制整张拼图、不得同时演完其他格、"
        "不得生成分格边框或板中文字。"
    )
    text_item = next((item for item in content if item.get("type") == "text"), None)
    if text_item is None:
        content.insert(0, {"type": "text", "text": directive})
    else:
        text_item["text"] = f"{directive}\n{text_item.get('text', '')}"
    content.append({
        "type": "image_url",
        "image_url": {"url": board_url},
        "role": "reference_image",
        "priority": "low",
    })
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
    if request.chunk.storyboard_image:
        content_meta["_storyboard_frame_path"] = request.chunk.storyboard_image
        content_meta["_storyboard_beat_id"] = request.chunk.storyboard_beat_id
        content_meta["generation_actions"] = [request.chunk.action_prompt]
        content_meta["gen_strategy"] = "i2v"
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
    return _append_group_board_reference(content, board_path)


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


def _provider_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any], int | None, int]:
    shot_meta = _read_shot_meta(output_dir, request.shot_id)
    content = _base_content(output_dir, request, shot_meta)
    if request.chunk.mode == "native_extend":
        if request.previous_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no predecessor video")
        content = _extension_content(content, request.previous_output_path)
    return content, shot_meta, _generation_seed(request), _chunk_duration(request)


def _task_payload(
    request: ChunkExecutionRequest,
    *,
    model: str,
    duration: int,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "shot_id": request.shot_id,
        "chunk_id": request.chunk.chunk_id,
        "resource_id": request.resource_id,
        "input_fingerprint": request.input_fingerprint,
        "output_path": str(request.output_path),
        "model": model,
        "mode": request.chunk.mode,
        "continuation_contract": (
            "tail_window_ordered_frames_v1" if request.chunk.mode == "native_extend" else None
        ),
        "tail_window_seconds": 2.0 if request.chunk.mode == "native_extend" else None,
        "ordered_frame_fractions": (
            [0.2, 0.6, 0.95] if request.chunk.mode == "native_extend" else None
        ),
        "duration": duration,
        "seed": seed,
        "repair_attempt": request.repair_attempt,
    }


def _provider_input_context(
    output_dir: Path,
    shot_id: str,
    chunk_id: str,
) -> dict[str, str | None]:
    shot_meta = output_dir / "shots" / shot_id / "SHOT_META.json"
    storyboard_frame = output_dir / "storyboard_images" / f"{shot_id}.png"
    group, group_board = _storyboard_group_for_shot(output_dir, shot_id)
    group_contract = output_dir / "STORYBOARD_GROUPS.json"
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
        "generation_seed_strategy": "scene_chunk_repair_v1",
        "continuation_contract": "tail_window_ordered_frames_v1",
        "tail_window_seconds": "2.0",
        "ordered_frame_fractions": "0.2,0.6,0.95",
    }


def _direct_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import seedance_client
    from utils.config import SEEDANCE_MODEL, get_api_key_or_raise

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

        def submit() -> str:
            try:
                content, _shot_meta, _seed, _duration = _provider_content(
                    output_dir,
                    request,
                )
            except Exception as exc:
                raise ProviderPreparationError(
                    f"cannot prepare provider content for {request.resource_id}: {exc}"
                ) from exc
            return seedance_client.submit_content(
                content,
                api_key=api_key,
                model=model,
                duration=duration,
                ratio=ratio,
                seed=seed,
            )

        with slots.reserve("seedance", "video", request.resource_id, capacity=capacity):
            with leases.reserve(
                "seedance",
                "video",
                request.resource_id,
                capacity=capacity,
            ):
                execution = execute_seedance_video_task(
                    task_store,
                    run_id=str(output_dir.resolve()),
                    resource_id=request.resource_id,
                    payload=payload,
                    provider_endpoint=seedance_client.BASE_URL,
                    output_path=request.output_path,
                    submit=submit,
                    poll=partial(seedance_client.poll, api_key=api_key),
                    download=seedance_client.download,
                )
        return ChunkExecutionResult(
            output_path=Path(execution.output_path),
            provider_task_id=execution.provider_job_id,
        )

    return execute


def _bridge_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import local_video_client

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

    requires_video_upload = any(
        chunk.mode == "native_extend" for shot in plan.shots for chunk in shot.chunks
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
    if prepare_seam is not None and any(
        len(shot.chunks) > 1
        and all(chunk.expected_overlap_frames == 0 for chunk in shot.chunks[1:])
        for shot in plan.shots
    ):
        raise RuntimeError(
            "continuity bridge requires an overlap-budgeted CONTINUITY_PLAN.json; "
            "rerun Phase 4 with HONCUT_CONTINUITY_BRIDGE=auto"
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
        chunk_context=lambda shot, chunk: _provider_input_context(
            root,
            shot.shot_id,
            chunk.chunk_id,
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
