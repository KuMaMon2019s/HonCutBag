"""Local repair for overlapping generated-video seams."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BRIDGE_KIND = "honcut.continuity_bridge.v1"


@dataclass(frozen=True)
class VideoShape:
    duration_s: float
    fps: float
    width: int
    height: int
    frame_count: int


def _run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_checked(command: list[str], output_path: Path, *, action: str) -> None:
    output_path.unlink(missing_ok=True)
    completed = _run(command)
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(f"cannot {action}: {detail[-1] if detail else 'unknown ffmpeg error'}")


def _probe_video(path: Path) -> VideoShape:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration,nb_read_frames,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot probe continuity bridge input {path}")
    document = json.loads(completed.stdout)
    streams = document.get("streams") or []
    if not streams:
        raise RuntimeError(f"continuity bridge input has no video stream: {path}")
    stream = streams[0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    shape = VideoShape(
        duration_s=float(stream.get("duration") or document["format"]["duration"]),
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=int(
            stream.get("nb_read_frames")
            or stream.get("nb_frames")
            or round(float(stream.get("duration") or document["format"]["duration"]) * fps)
        ),
    )
    if shape.duration_s <= 0 or shape.fps <= 0 or shape.width <= 0 or shape.height <= 0:
        raise RuntimeError(f"continuity bridge input has invalid video metadata: {path}")
    return shape


def _sample_window(
    path: Path,
    *,
    start_s: float,
    duration_s: float,
    fps: int,
    width: int = 96,
    height: int = 54,
) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            str(path),
            "-t",
            f"{duration_s:.6f}",
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    frame_bytes = width * height * 3
    usable = len(completed.stdout) - len(completed.stdout) % frame_bytes
    if completed.returncode != 0 or usable < frame_bytes * 2:
        detail = completed.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"cannot sample continuity overlap from {path}: "
            f"{detail[-1] if detail else 'fewer than two decoded frames'}"
        )
    return np.frombuffer(completed.stdout[:usable], dtype=np.uint8).reshape(-1, height, width, 3)


def detect_replayed_prefix(
    previous_video: str | Path,
    following_video: str | Path,
    *,
    search_seconds: float = 3.0,
    sample_fps: int = 12,
    min_overlap_seconds: float = 0.5,
    max_frame_mae: float = 0.08,
    min_motion_cosine: float = 0.5,
    min_margin: float = 0.004,
) -> dict[str, Any]:
    """Find whether the following prefix replays the previous tail trajectory."""
    previous = Path(previous_video)
    following = Path(following_video)
    if search_seconds <= 0 or sample_fps < 2 or min_overlap_seconds <= 0:
        raise ValueError("overlap search durations and sample fps must be positive")
    previous_shape = _probe_video(previous)
    following_shape = _probe_video(following)
    window_s = min(search_seconds, previous_shape.duration_s, following_shape.duration_s)
    if window_s < min_overlap_seconds:
        return {
            "detected": False,
            "overlap_seconds": 0.0,
            "reason": "videos are shorter than the minimum overlap window",
            "candidates": [],
        }
    previous_frames = (
        _sample_window(
            previous,
            start_s=max(0.0, previous_shape.duration_s - window_s),
            duration_s=window_s,
            fps=sample_fps,
        ).astype(np.float32)
        / 255.0
    )
    following_frames = (
        _sample_window(
            following,
            start_s=0.0,
            duration_s=window_s,
            fps=sample_fps,
        ).astype(np.float32)
        / 255.0
    )
    maximum = min(len(previous_frames), len(following_frames))
    minimum = max(2, math.ceil(min_overlap_seconds * sample_fps))
    candidates: list[dict[str, float | int]] = []
    for count in range(minimum, maximum + 1):
        tail = previous_frames[-count:]
        head = following_frames[:count]
        frame_mae = float(np.abs(tail - head).mean())
        tail_motion = np.diff(tail, axis=0)
        head_motion = np.diff(head, axis=0)
        denominator = float(np.linalg.norm(tail_motion) * np.linalg.norm(head_motion))
        motion_cosine = (
            float(np.sum(tail_motion * head_motion) / denominator) if denominator > 1e-9 else 0.0
        )
        candidates.append(
            {
                "frames": count,
                "seconds": round(count / sample_fps, 6),
                "frame_mae": round(frame_mae, 6),
                "motion_cosine": round(motion_cosine, 6),
            }
        )
    best = min(candidates, key=lambda item: float(item["frame_mae"]))
    separated = [
        float(item["frame_mae"])
        for item in candidates
        if abs(int(item["frames"]) - int(best["frames"])) >= sample_fps * 0.25
    ]
    runner_up = min(separated) if separated else float(best["frame_mae"])
    margin = runner_up - float(best["frame_mae"])
    detected = bool(
        float(best["frame_mae"]) <= max_frame_mae
        and float(best["motion_cosine"]) >= min_motion_cosine
        and margin >= min_margin
    )
    return {
        "detected": detected,
        "overlap_seconds": float(best["seconds"]) if detected else 0.0,
        "best": best,
        "confidence_margin": round(margin, 6),
        "thresholds": {
            "max_frame_mae": max_frame_mae,
            "min_motion_cosine": min_motion_cosine,
            "min_margin": min_margin,
        },
        "candidates": candidates,
        "reason": (
            "unique aligned trajectory replay detected"
            if detected
            else "no sufficiently similar, directionally aligned, unique replay window"
        ),
    }


def _extract_frame(video: Path, timestamp_s: float, output_path: Path) -> None:
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{timestamp_s:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output_path),
        ],
        output_path,
        action=f"extract continuity bridge anchor at {timestamp_s:.3f}s from {video}",
    )


def _sample_one_frame(video: Path, timestamp_s: float) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{timestamp_s:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=96:54",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    frame_bytes = 96 * 54 * 3
    if completed.returncode != 0 or len(completed.stdout) < frame_bytes:
        raise RuntimeError(f"cannot sample continuity bridge boundary frame from {video}")
    return np.frombuffer(completed.stdout[:frame_bytes], dtype=np.uint8).reshape(54, 96, 3)


def _measure_cross_boundary_mae(previous_video: Path, following_video: Path) -> float:
    previous_shape = _probe_video(previous_video)
    previous = _sample_one_frame(
        previous_video, max(0.0, previous_shape.duration_s - 2 / previous_shape.fps)
    )
    following = _sample_one_frame(following_video, 0.0)
    return round(
        float(np.abs(following.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0),
        6,
    )


def _render_trimmed_following(
    following_video: Path,
    output_path: Path,
    *,
    trim_seconds: float,
    fps: float,
) -> None:
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{trim_seconds:.9f}",
            "-i",
            str(following_video),
            "-vf",
            f"fps={fps:.9f},format=yuv420p,setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        output_path,
        action=f"trim {trim_seconds:.3f}s replayed continuity prefix",
    )


def _render_bridge_candidate(
    previous_video: Path,
    following_video: Path,
    *,
    output_path: Path,
    work_dir: Path,
    trim_seconds: float,
    bridge_frames: int,
    fps: float,
) -> dict[str, Any]:
    if bridge_frames < 4:
        raise ValueError("continuity bridge requires at least four frames")
    bridge_seconds = bridge_frames / fps
    following_shape = _probe_video(following_video)
    if trim_seconds + 2 * bridge_seconds >= following_shape.duration_s:
        raise ValueError("following video is too short for the requested continuity bridge")
    previous_shape = _probe_video(previous_video)
    work_dir.mkdir(parents=True, exist_ok=True)
    previous_final_s = max(0.0, previous_shape.duration_s - 1 / fps)
    timestamps = (
        max(0.0, previous_final_s - bridge_seconds),
        previous_final_s,
        trim_seconds + bridge_seconds,
        trim_seconds + 2 * bridge_seconds,
    )
    for index, (source, timestamp) in enumerate(
        (
            (previous_video, timestamps[0]),
            (previous_video, timestamps[1]),
            (following_video, timestamps[2]),
            (following_video, timestamps[3]),
        )
    ):
        _extract_frame(source, timestamp, work_dir / f"anchor_{index:02d}.png")
    for index in range(4, 8):
        shutil.copy2(work_dir / "anchor_03.png", work_dir / f"anchor_{index:02d}.png")

    # Give the two seam endpoints N intermediate output ticks. The following
    # endpoint remains in the source clip, so the bridge replaces exactly N
    # discarded source frames without adding or losing one frame.
    source_rate = fps / (bridge_frames + 1)
    interpolated = work_dir / "interpolated.mp4"
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            f"{source_rate:.9f}",
            "-start_number",
            "0",
            "-i",
            str(work_dir / "anchor_%02d.png"),
            "-vf",
            (
                f"minterpolate=fps={fps:.9f}:mi_mode=mci:mc_mode=aobmc:"
                "me_mode=bidir:me=epzs:vsbmc=1"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(interpolated),
        ],
        interpolated,
        action="interpolate continuity bridge anchors",
    )
    bridge = work_dir / "bridge.mp4"
    bridge_start = (bridge_frames + 2) / fps
    bridge_end = 2 * (bridge_frames + 1) / fps
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(interpolated),
            "-vf",
            f"trim=start={bridge_start:.9f}:end={bridge_end:.9f},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(bridge),
        ],
        bridge,
        action="extract interpolated continuity bridge frames",
    )
    remaining_start = trim_seconds + bridge_seconds
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(bridge),
            "-ss",
            f"{remaining_start:.9f}",
            "-i",
            str(following_video),
            "-filter_complex",
            (
                f"[0:v]fps={fps:.9f},format=yuv420p,setpts=PTS-STARTPTS[v0];"
                f"[1:v]fps={fps:.9f},format=yuv420p,setpts=PTS-STARTPTS[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[outv]"
            ),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        output_path,
        action=f"render {bridge_frames}-frame continuity bridge candidate",
    )
    return {
        "bridge_frames": bridge_frames,
        "bridge_seconds": round(bridge_seconds, 6),
        "trim_seconds": round(trim_seconds, 6),
        "output_path": str(output_path),
    }


def repair_continuity_boundary(
    previous_video: str | Path,
    following_video: str | Path,
    output_path: str | Path,
    *,
    work_dir: str | Path,
    overlap_seconds: float | None = None,
    candidate_frames: tuple[int, ...] = (4, 6, 8),
) -> dict[str, Any]:
    """Trim a replayed prefix and choose the smoothest local interpolation bridge."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("continuity bridge requires ffmpeg and ffprobe")
    previous = Path(previous_video)
    following = Path(following_video)
    output = Path(output_path)
    workspace = Path(work_dir)
    if output.resolve() in {previous.resolve(), following.resolve()}:
        raise ValueError("continuity bridge output must not overwrite either provider input")
    previous_shape = _probe_video(previous)
    following_shape = _probe_video(following)
    if (
        previous_shape.width != following_shape.width
        or previous_shape.height != following_shape.height
    ):
        raise ValueError("continuity bridge inputs must have the same resolution")
    if abs(previous_shape.fps - following_shape.fps) > 0.05:
        raise ValueError("continuity bridge inputs must have the same frame rate")
    valid_frames = tuple(sorted(set(candidate_frames)))
    if not valid_frames or any(frame < 4 or frame > 24 for frame in valid_frames):
        raise ValueError("continuity bridge candidate frames must be between 4 and 24")

    overlap = detect_replayed_prefix(previous, following)
    trim_seconds = overlap["overlap_seconds"] if overlap_seconds is None else overlap_seconds
    if trim_seconds < 0:
        raise ValueError("continuity overlap trim must not be negative")
    baseline_boundary_mae = _measure_cross_boundary_mae(previous, following)
    candidates: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    trimmed_path = workspace / "trimmed_hard_cut.mp4"
    _render_trimmed_following(
        following,
        trimmed_path,
        trim_seconds=float(trim_seconds),
        fps=previous_shape.fps,
    )
    trimmed_boundary_mae = _measure_cross_boundary_mae(previous, trimmed_path)
    for frames in valid_frames:
        candidate_path = workspace / f"bridge_{frames:02d}_frames.mp4"
        candidate = _render_bridge_candidate(
            previous,
            following,
            output_path=candidate_path,
            work_dir=workspace / f"candidate_{frames:02d}",
            trim_seconds=float(trim_seconds),
            bridge_frames=frames,
            fps=previous_shape.fps,
        )
        candidate["boundary_frame_mae"] = _measure_cross_boundary_mae(previous, candidate_path)
        candidates.append(candidate)
    selected = min(candidates, key=lambda item: float(item["boundary_frame_mae"]))
    if float(selected["boundary_frame_mae"]) < trimmed_boundary_mae:
        selected_path = Path(selected["output_path"])
        status = "repaired"
        selected_frames: int | None = int(selected["bridge_frames"])
        selected_mae = float(selected["boundary_frame_mae"])
    else:
        selected_path = trimmed_path
        status = "trimmed"
        selected_frames = None
        selected_mae = trimmed_boundary_mae
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    shutil.copy2(selected_path, temporary)
    os.replace(temporary, output)
    output_shape = _probe_video(output)
    receipt = {
        "kind": BRIDGE_KIND,
        "status": status,
        "engine": "ffmpeg_minterpolate",
        "previous_video": str(previous),
        "following_video": str(following),
        "output_path": str(output),
        "overlap": overlap,
        "trim_seconds": round(float(trim_seconds), 6),
        "baseline_boundary_frame_mae": baseline_boundary_mae,
        "trimmed_boundary_frame_mae": trimmed_boundary_mae,
        "candidates": candidates,
        "selected_bridge_frames": selected_frames,
        "selected_boundary_frame_mae": selected_mae,
        "improved": selected_mae < baseline_boundary_mae,
        "source_following_duration_seconds": round(following_shape.duration_s, 6),
        "source_following_frames": following_shape.frame_count,
        "effective_following_duration_seconds": round(output_shape.duration_s, 6),
        "effective_following_frames": output_shape.frame_count,
        "removed_replay_frames": following_shape.frame_count - output_shape.frame_count,
        "removed_replay_duration_seconds": round(
            following_shape.duration_s - output_shape.duration_s, 6
        ),
    }
    (workspace / "CONTINUITY_BRIDGE.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
