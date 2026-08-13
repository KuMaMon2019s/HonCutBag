"""Phase 6 provider adapters for calibrated continuity chunk execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from runtime.generation_tasks import GenerationTaskStore
from runtime.seedance_execution import execute_seedance_video_task
from schemas.continuity import ContinuityPlan


def _read_shot_meta(output_dir: Path, shot_id: str) -> dict[str, Any]:
    path = output_dir / "shots" / shot_id / "SHOT_META.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"continuity auto requires {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scene_seed(request: ChunkExecutionRequest) -> int | None:
    scene = str(request.anchors.get("scene") or "").strip()
    if not scene:
        return None
    return int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16) % 2_147_483_647


def _chunk_duration(request: ChunkExecutionRequest) -> int:
    duration = float(request.chunk.target_duration_s)
    rounded = round(duration)
    if not math.isclose(duration, rounded, abs_tol=1e-6):
        raise ValueError(
            f"{request.resource_id} duration {duration} cannot be represented by Seedance seconds"
        )
    return int(rounded)


def _chunk_prompt(request: ChunkExecutionRequest, shot_meta: dict[str, Any]) -> str:
    prompt = str(shot_meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"{request.shot_id} SHOT_META.json has no prompt")
    continuation = (
        "Generate the opening chunk of this editorial shot."
        if request.chunk.mode == "fresh"
        else "Continue directly from the reference video's final state without a reset or cut."
    )
    return (
        f"{prompt}\n\n[continuity chunk {request.chunk.sequence}] {continuation}\n"
        f"{request.memory_context}"
    )


def _base_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    from tools.asset_packager import build_content_for_shot

    content_meta = dict(shot_meta)
    content_meta["prompt"] = _chunk_prompt(request, shot_meta)
    content = build_content_for_shot(
        output_dir=output_dir,
        shot_id=request.shot_id,
        shot_meta=content_meta,
    )
    if not content:
        return [{"type": "text", "text": content_meta["prompt"]}]
    return content


def _extension_content(
    content: Sequence[dict[str, Any]],
    previous_output_path: Path,
) -> list[dict[str, Any]]:
    from clients.tos_uploader import upload_media_file

    video_url = upload_media_file(previous_output_path, prefix="volcengine/video")
    if not video_url:
        raise RuntimeError(f"failed to upload continuity predecessor {previous_output_path}")
    normalized: list[dict[str, Any]] = []
    for item in content:
        copied = dict(item)
        if copied.get("type") == "image_url":
            copied["role"] = "reference_image"
            copied.pop("priority", None)
        normalized.append(copied)
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
    return content, shot_meta, _scene_seed(request), _chunk_duration(request)


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
        "duration": duration,
        "seed": seed,
        "repair_attempt": request.repair_attempt,
    }


def _provider_input_context(output_dir: Path, shot_id: str) -> dict[str, str | None]:
    shot_meta = output_dir / "shots" / shot_id / "SHOT_META.json"
    storyboard_frame = output_dir / "storyboard_images" / f"{shot_id}.png"
    return {
        "shot_meta_sha256": hashlib.sha256(shot_meta.read_bytes()).hexdigest(),
        "storyboard_frame_sha256": (
            hashlib.sha256(storyboard_frame.read_bytes()).hexdigest()
            if storyboard_frame.is_file()
            else None
        ),
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
        seed = _scene_seed(request)
        duration = _chunk_duration(request)
        payload = _task_payload(request, model=model, duration=duration, seed=seed)

        def submit() -> str:
            content, _shot_meta, _seed, _duration = _provider_content(output_dir, request)
            return seedance_client.submit_content(
                content,
                api_key=api_key,
                model=model,
                duration=duration,
                ratio="16:9",
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
        seed = _scene_seed(request)
        duration = _chunk_duration(request)
        model = "seedance"
        payload = _task_payload(request, model=model, duration=duration, seed=seed)

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
                width=1280,
                height=720,
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


def execute_phase6_auto_continuity(
    output_dir: str | Path,
    plan: ContinuityPlan,
    calibration: SeamCalibration,
) -> dict[str, Any]:
    """Run calibrated continuity generation through exactly one configured provider."""
    root = Path(output_dir)
    provider = os.environ.get("VIDEO_PROVIDER", "seedance").strip().lower()
    task_store = GenerationTaskStore(root / "runtime.db")
    if provider == "seedance":
        execute_chunk = _direct_seedance_executor(root, task_store)
    elif provider == "bridge":
        execute_chunk = _bridge_seedance_executor(root, task_store)
    else:
        raise RuntimeError(
            "continuity auto requires VIDEO_PROVIDER=seedance or bridge; "
            f"{provider!r} has no verified native video-extension contract"
        )

    max_repairs = int(os.environ.get("HONCUT_CONTINUITY_MAX_REPAIRS", "1"))
    if not 0 <= max_repairs <= 2:
        raise ValueError("HONCUT_CONTINUITY_MAX_REPAIRS must be between 0 and 2")
    workers = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    report = execute_continuity_plan(
        plan,
        root,
        execute_chunk=execute_chunk,
        materialize_shot=materialize_continuity_shot,
        chunk_context=lambda shot, _chunk: _provider_input_context(root, shot.shot_id),
        seam_calibration=calibration.model_dump(mode="json"),
        max_seam_repairs=max_repairs,
        max_workers=workers,
    )
    report.update(
        {
            "provider": provider,
            "mode": "continuity_auto",
            "calibration_fingerprint": calibration.dataset_fingerprint,
        }
    )
    return report
