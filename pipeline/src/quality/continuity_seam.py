"""Observe-only visual metrics for immediate generated-chunk boundaries."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _pixels(value: Path | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        pixels = value
    else:
        pixels = np.asarray(Image.open(value).convert("RGB"))
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).resize((96, 54))
    return np.asarray(image, dtype=np.float32)


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return pixels[:, :, 0] * 0.299 + pixels[:, :, 1] * 0.587 + pixels[:, :, 2] * 0.114


def _perceptual_hash(pixels: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(pixels.astype(np.uint8)).convert("L").resize((9, 8))
    values = np.asarray(gray, dtype=np.float32)
    return values[:, 1:] > values[:, :-1]


def _centroid(pixels: np.ndarray) -> np.ndarray | None:
    gray = _luminance(pixels)
    weights = np.abs(gray - float(np.median(gray)))
    total = float(weights.sum())
    if total < 1e-6:
        return None
    yy, xx = np.indices(weights.shape)
    return np.asarray([float((xx * weights).sum() / total), float((yy * weights).sum() / total)])


def _motion_vector(frames: Sequence[np.ndarray]) -> np.ndarray | None:
    if len(frames) < 2:
        return None
    first = _centroid(frames[-2])
    second = _centroid(frames[-1])
    if first is None or second is None:
        return None
    vector = second - first
    return vector if float(np.linalg.norm(vector)) >= 0.25 else None


def _motion_direction_change(
    tail_frames: Sequence[np.ndarray], head_frames: Sequence[np.ndarray]
) -> float | None:
    tail_vector = _motion_vector(tail_frames)
    head_vector = _motion_vector(head_frames[:2])
    if tail_vector is None and head_vector is None:
        return 0.0
    if tail_vector is None or head_vector is None:
        return 0.5
    cosine = float(
        np.dot(tail_vector, head_vector)
        / (np.linalg.norm(tail_vector) * np.linalg.norm(head_vector))
    )
    return round(float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0)), 6)


def compare_frame_sequences(
    tail_frames: Sequence[Path | np.ndarray],
    head_frames: Sequence[Path | np.ndarray],
) -> dict[str, Any]:
    """Return normalized raw evidence without assigning a pass/fail threshold."""
    if not tail_frames or not head_frames:
        raise ValueError("tail_frames and head_frames must not be empty")
    tail = [_pixels(frame) for frame in tail_frames]
    head = [_pixels(frame) for frame in head_frames]
    previous = tail[-1]
    following = head[0]
    pixel_mae = float(np.abs(previous - following).mean() / 255.0)
    brightness_delta = float(
        abs(float(_luminance(previous).mean()) - float(_luminance(following).mean())) / 255.0
    )
    previous_color = previous.mean(axis=(0, 1))
    following_color = following.mean(axis=(0, 1))
    color_mean_delta = float(
        np.linalg.norm(previous_color - following_color) / (255.0 * np.sqrt(3.0))
    )
    hash_distance = float(
        np.not_equal(_perceptual_hash(previous), _perceptual_hash(following)).mean()
    )
    motion_change = _motion_direction_change(tail, head)
    motion_component = 0.0 if motion_change is None else motion_change
    provisional_risk = (
        0.35 * pixel_mae
        + 0.20 * brightness_delta
        + 0.20 * color_mean_delta
        + 0.15 * hash_distance
        + 0.10 * motion_component
    )
    return {
        "pixel_mae": round(pixel_mae, 6),
        "brightness_delta": round(brightness_delta, 6),
        "color_mean_delta": round(color_mean_delta, 6),
        "perceptual_hash_distance": round(hash_distance, 6),
        "motion_direction_change": motion_change,
        "provisional_risk_score": round(provisional_risk, 6),
        "policy": "observe_only",
    }


def _video_duration(video_path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip().splitlines()[0])


def _sample_video_frames(
    video_path: Path,
    *,
    frames_per_second: int = 4,
    width: int = 96,
    height: int = 54,
) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={frames_per_second},scale={width}:{height}",
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
    if completed.returncode != 0 or len(completed.stdout) < frame_bytes * 2:
        detail = completed.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"cannot sample continuation trajectory from {video_path}: "
            f"{detail[-1] if detail else 'fewer than two decoded frames'}"
        )
    usable_bytes = len(completed.stdout) - (len(completed.stdout) % frame_bytes)
    return np.frombuffer(completed.stdout[:usable_bytes], dtype=np.uint8).reshape(
        -1,
        height,
        width,
        3,
    )


def measure_video_replay_similarity(
    previous_video: Path,
    following_video: Path,
) -> dict[str, Any]:
    """Measure whether an extension replays its predecessor's aligned trajectory.

    This is deliberately a human-review signal, not an automatic retry trigger:
    locked cameras and very subtle action can legitimately look similar.
    """
    previous = _sample_video_frames(previous_video).astype(np.float32) / 255.0
    following = _sample_video_frames(following_video).astype(np.float32) / 255.0
    aligned_count = min(len(previous), len(following))
    previous = previous[:aligned_count]
    following = following[:aligned_count]
    previous_motion = np.diff(previous, axis=0)
    following_motion = np.diff(following, axis=0)
    denominator = float(
        np.linalg.norm(previous_motion) * np.linalg.norm(following_motion)
    )
    motion_cosine = (
        float(np.sum(previous_motion * following_motion) / denominator)
        if denominator > 1e-9
        else 0.0
    )
    # Identical decoded trajectories can land a few ulps below 1.0 depending
    # on the NumPy reduction implementation. Preserve the mathematical
    # identity instead of exposing a dependency-version artifact.
    if abs(1.0 - motion_cosine) <= 1e-5:
        motion_cosine = 1.0
    frame_similarity = 1.0 - float(np.abs(previous - following).mean())
    previous_energy = float(np.abs(previous_motion).mean())
    following_energy = float(np.abs(following_motion).mean())
    likely_replay = (
        aligned_count >= 8
        and frame_similarity >= 0.95
        and motion_cosine >= 0.75
        and min(previous_energy, following_energy) >= 0.003
    )
    return {
        "aligned_frame_count": aligned_count,
        "aligned_frame_similarity": round(frame_similarity, 6),
        "motion_cosine_similarity": round(motion_cosine, 6),
        "previous_motion_energy": round(previous_energy, 6),
        "following_motion_energy": round(following_energy, 6),
        "likely_replay": likely_replay,
        "policy": "human_review_only",
    }


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    output_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if (
        completed.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        detail = completed.stderr.strip().splitlines()
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cannot extract seam frame from {video_path}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )


def extract_video_tail_frame(video_path: Path, output_path: Path) -> Path:
    """Extract a deterministic near-final frame for provider continuation anchoring."""
    duration = _video_duration(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: RuntimeError | None = None
    for distance_from_end in (0.08, 0.16, 0.25):
        try:
            _extract_frame(
                video_path,
                max(0.0, duration - distance_from_end),
                output_path,
            )
            return output_path
        except RuntimeError as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover - the fixed candidate list is non-empty
        raise RuntimeError(f"cannot extract tail frame from {video_path}")
    raise last_error


def extract_video_head_frame(video_path: Path, output_path: Path) -> Path:
    """Extract the actual first decodable frame of a completed primary video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _extract_frame(video_path, 0.0, output_path)
    return output_path


def extract_video_tail_window(
    video_path: Path,
    output_path: Path,
    *,
    window_s: float = 1.5,
) -> Path:
    """Render an exact, video-only tail window for continuation conditioning."""
    if window_s <= 0:
        raise ValueError("tail window duration must be positive")
    source_duration = _video_duration(video_path)
    actual_window = min(window_s, source_duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{max(0.0, source_duration - actual_window):.6f}",
            "-t",
            f"{actual_window:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if (
        completed.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        detail = completed.stderr.strip().splitlines()
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cannot extract tail video window from {video_path}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    return output_path


def extract_ordered_video_frames(
    video_path: Path,
    output_paths: Sequence[Path],
    *,
    fractions: Sequence[float] = (0.2, 0.6, 0.95),
) -> tuple[Path, ...]:
    """Extract ordered state anchors across a short reference video."""
    if len(output_paths) != len(fractions) or not output_paths:
        raise ValueError("output paths and frame fractions must have equal non-zero length")
    if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
        raise ValueError("frame fractions must be between 0 and 1")
    duration = _video_duration(video_path)
    extracted = []
    for path, fraction in zip(output_paths, fractions, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        _extract_frame(video_path, max(0.0, duration * fraction - 0.001), path)
        extracted.append(path)
    return tuple(extracted)


def measure_video_seam(
    previous_video: Path,
    following_video: Path,
    boundary_id: str,
    *,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Extract bounded tail/head samples and measure one real video boundary."""
    duration = _video_duration(previous_video)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    tail_times = [max(0.0, duration - 0.25), max(0.0, duration - 0.08)]
    head_times = [0.02, 0.18]
    tail_paths = [evidence_dir / f"{boundary_id}_tail_{index}.jpg" for index in range(2)]
    head_paths = [evidence_dir / f"{boundary_id}_head_{index}.jpg" for index in range(2)]
    for timestamp, path in zip(tail_times, tail_paths, strict=True):
        _extract_frame(previous_video, timestamp, path)
    for timestamp, path in zip(head_times, head_paths, strict=True):
        _extract_frame(following_video, timestamp, path)
    return {
        "boundary_id": boundary_id,
        "previous_video": str(previous_video),
        "following_video": str(following_video),
        "tail_frames": [str(path) for path in tail_paths],
        "head_frames": [str(path) for path in head_paths],
        "metrics": compare_frame_sequences(tail_paths, head_paths),
        "replay": measure_video_replay_similarity(previous_video, following_video),
    }
