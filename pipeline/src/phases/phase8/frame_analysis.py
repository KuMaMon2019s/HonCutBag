"""Phase 8 per-shot visual QA and actionable edit/reshoot decisions."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


SemanticReviewer = Callable[[list[Path], dict[str, Any]], dict[str, Any]]


def _read_pixels(path: Path) -> np.ndarray:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {path}")
        return image.astype(np.float32)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def is_black_frame(path: Path, threshold: float = 20.0) -> bool:
    return float(_read_pixels(Path(path)).mean()) < threshold


def is_static_frame(first: Path, last: Path, threshold: float = 3.0) -> bool:
    first_pixels = _read_pixels(Path(first))
    last_pixels = _read_pixels(Path(last))
    if first_pixels.shape != last_pixels.shape:
        return False
    return float(np.abs(first_pixels - last_pixels).mean()) < threshold


def probe_duration(video_path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return float(completed.stdout.strip().splitlines()[0])


def _sample_timestamps(duration: float, max_frames: int, interval_s: float) -> list[float]:
    if duration <= 0:
        return []
    usable_end = max(0.0, duration - min(0.1, duration / 10))
    count = max(3, min(max_frames, int(math.ceil(duration / interval_s)) + 1))
    return [round(float(value), 4) for value in np.linspace(0.05, usable_end, count)]


def extract_review_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    max_frames: int = 12,
    interval_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract dense, bounded samples across a complete generated shot."""
    duration = probe_duration(video_path)
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)

    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(_sample_timestamps(duration, max_frames, interval_s)):
        output = frames_dir / f"frame_{index:03d}.jpg"
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.6f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(output),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = completed.stderr.strip().splitlines()
            raise RuntimeError(
                f"ffmpeg frame extraction failed for {video_path} at {timestamp:.3f}s: "
                f"{detail[-1] if detail else 'unknown error'}"
            )
        frames.append({"path": output, "timestamp_s": timestamp})
    return frames


def extract_three_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    """Compatibility wrapper returning representative first/middle/tail frames."""
    frames = extract_review_frames(video_path, frames_dir, max_frames=3, interval_s=9999)
    return [item["path"] for item in frames]


def _frame_metrics(path: Path) -> dict[str, Any]:
    pixels = _read_pixels(path)
    gray = pixels.mean(axis=2)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    blur_score = float(laplacian.var())
    issues: list[str] = []
    if brightness < 20:
        issues.append("black")
    elif brightness < 35:
        issues.append("underexposed")
    elif brightness > 225:
        issues.append("overexposed")
    if blur_score < 45:
        issues.append("blurry")
    if contrast < 18:
        issues.append("low_contrast")
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "issues": issues,
    }


def _detect_black_segments(video_path: Path) -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path), "-vf", "blackdetect=d=0.08:pix_th=0.10",
            "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    segments: list[dict[str, float]] = []
    pattern = re.compile(
        r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)\s+"
        r"black_duration:(?P<duration>[\d.]+)"
    )
    for match in pattern.finditer(completed.stderr):
        segments.append({key: round(float(value), 4) for key, value in match.groupdict().items()})
    return segments


def _detect_freeze_segments(video_path: Path) -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path), "-vf", "freezedetect=n=-50dB:d=1.5",
            "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    starts: list[float] = []
    durations: list[float] = []
    ends: list[float] = []
    for line in completed.stderr.splitlines():
        start = re.search(r"freeze_start:\s*([\d.]+)", line)
        duration = re.search(r"freeze_duration:\s*([\d.]+)", line)
        end = re.search(r"freeze_end:\s*([\d.]+)", line)
        if start:
            starts.append(float(start.group(1)))
        if duration:
            durations.append(float(duration.group(1)))
        if end:
            ends.append(float(end.group(1)))
    segments: list[dict[str, float]] = []
    for index, start in enumerate(starts):
        duration = durations[index] if index < len(durations) else 0.0
        end = ends[index] if index < len(ends) else start + duration
        segments.append({"start": round(start, 4), "end": round(end, 4), "duration": round(duration, 4)})
    return segments


def decide_shot_action(
    duration: float,
    frames: list[dict[str, Any]],
    black_segments: list[dict[str, float]],
    freeze_segments: list[dict[str, float]],
    semantic_review: dict[str, Any] | None = None,
    shot_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn technical/semantic evidence into one keep, trim, or reshoot action."""
    boundary_tolerance = min(0.35, max(0.12, duration * 0.04))
    trim_start = 0.0
    trim_end = duration
    reasons: list[str] = []
    interior_black: list[dict[str, float]] = []

    for segment in black_segments:
        if segment["start"] <= boundary_tolerance:
            trim_start = max(trim_start, segment["end"])
        elif segment["end"] >= duration - boundary_tolerance:
            trim_end = min(trim_end, segment["start"])
        else:
            interior_black.append(segment)

    semantic_review = semantic_review or {}
    if semantic_review.get("verdict") in {"fail", "reshoot"}:
        reasons.extend(str(issue) for issue in semantic_review.get("issues", []) or ["semantic visual review failed"])
        return {"action": "reshoot", "reasons": reasons, "trim_start_s": 0.0, "trim_end_s": duration}

    if interior_black:
        reasons.append(f"interior black segment detected: {interior_black}")

    severe_samples = [
        frame for frame in frames
        if "black" in frame.get("metrics", {}).get("issues", [])
        or (
            "blurry" in frame.get("metrics", {}).get("issues", [])
            and any(
                issue in frame.get("metrics", {}).get("issues", [])
                for issue in ("underexposed", "overexposed")
            )
        )
    ]
    if len(severe_samples) >= max(2, math.ceil(len(frames) * 0.4)):
        reasons.append(f"{len(severe_samples)}/{len(frames)} sampled frames have severe quality issues")

    meta = shot_meta or {}
    expected_motion = " ".join(
        str(meta.get(key, "")) for key in ("action", "camera", "camera_movement", "motion")
    ).strip().lower()
    intentionally_static = not expected_motion or any(word in expected_motion for word in ("static", "locked", "固定"))
    long_freezes = [
        segment for segment in freeze_segments
        if segment.get("duration", 0) >= max(1.5, duration * 0.3)
    ]
    if long_freezes and not intentionally_static:
        reasons.append(f"unexpected long freeze detected: {long_freezes}")

    remaining = trim_end - trim_start
    if remaining < max(1.0, duration * 0.5):
        reasons.append(f"boundary trimming would leave only {remaining:.2f}s of {duration:.2f}s")

    if reasons:
        return {"action": "reshoot", "reasons": reasons, "trim_start_s": 0.0, "trim_end_s": duration}
    if trim_start > 0.05 or trim_end < duration - 0.05:
        return {
            "action": "trim",
            "reasons": [f"trim boundary defects to {trim_start:.3f}s-{trim_end:.3f}s"],
            "trim_start_s": round(trim_start, 4),
            "trim_end_s": round(trim_end, 4),
        }
    return {"action": "keep", "reasons": [], "trim_start_s": 0.0, "trim_end_s": round(duration, 4)}


def _automatic_semantic_reviewer() -> SemanticReviewer | None:
    setting = os.environ.get("HONCUT_SHOT_VLM_REVIEW", "auto").strip().lower()
    if setting in {"0", "false", "off", "no"}:
        return None
    try:
        from clients.ark_multimodal_client import ArkMultimodalClient

        client = ArkMultimodalClient()
    except Exception:
        return None

    def review(frame_paths: list[Path], shot_meta: dict[str, Any]) -> dict[str, Any]:
        expected = {
            key: shot_meta.get(key)
            for key in ("shot_id", "visual", "action", "who", "where", "characters")
            if shot_meta.get(key) not in (None, "", [])
        }
        prompt = (
            "Review these ordered frames from one generated video shot. Compare them with the expected "
            f"shot metadata: {json.dumps(expected, ensure_ascii=False)}. Detect character identity drift, "
            "extra or missing limbs/objects, broken anatomy, impossible geometry, continuity jumps, text or "
            "watermark artifacts, and content that contradicts the shot. Return JSON only: "
            '{"verdict":"pass|reshoot","issues":["..."],"confidence":0.0}.'
        )
        raw = client.review(frame_paths, prompt).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"verdict": "reshoot", "issues": ["invalid semantic review"]}

    return review


def analyze_shot_frames(
    shots_dir: Path,
    output_path: Path,
    *,
    semantic_reviewer: SemanticReviewer | bool | None = None,
    max_frames: int = 12,
    interval_s: float = 1.0,
) -> dict[str, Any]:
    """Analyze every generated shot and persist actionable assembly decisions."""
    if semantic_reviewer is None or semantic_reviewer is True:
        reviewer = _automatic_semantic_reviewer()
    elif semantic_reviewer is False:
        reviewer = None
    else:
        reviewer = semantic_reviewer

    report: dict[str, Any] = {
        "shots": {},
        "has_issues": False,
        "summary": {"keep": [], "trim": [], "reshoot": []},
        "semantic_review": "enabled" if reviewer else "unavailable",
    }
    for shot_dir in sorted(Path(shots_dir).iterdir()) if Path(shots_dir).is_dir() else []:
        video = shot_dir / "output.mp4"
        if not shot_dir.is_dir() or not shot_dir.name.startswith("S") or not video.is_file():
            continue
        meta_path = shot_dir / "SHOT_META.json"
        try:
            shot_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            shot_meta = {}

        try:
            duration = probe_duration(video)
            extracted = extract_review_frames(
                video, shot_dir / "frames", max_frames=max_frames, interval_s=interval_s
            )
            frames: list[dict[str, Any]] = []
            for item in extracted:
                frames.append({
                    "path": str(item["path"].relative_to(Path(shots_dir).parent)),
                    "timestamp_s": item["timestamp_s"],
                    "metrics": _frame_metrics(item["path"]),
                })
            black_segments = _detect_black_segments(video)
            freeze_segments = _detect_freeze_segments(video)
            semantic: dict[str, Any] | None = None
            if reviewer:
                try:
                    representative = [item["path"] for item in extracted]
                    if len(representative) > 5:
                        positions = np.linspace(0, len(representative) - 1, 5).astype(int)
                        representative = [representative[index] for index in positions]
                    semantic = reviewer(representative, shot_meta)
                except Exception as exc:
                    semantic = {"verdict": "unavailable", "issues": [], "error": str(exc)}
            decision = decide_shot_action(
                duration, frames, black_segments, freeze_segments, semantic, shot_meta
            )
            entry = {
                "duration_s": round(duration, 4),
                "frames": frames,
                "black_segments": black_segments,
                "freeze_segments": freeze_segments,
                "semantic_review": semantic,
                **decision,
            }
        except Exception as exc:
            entry = {
                "duration_s": 0.0,
                "frames": [],
                "black_segments": [],
                "freeze_segments": [],
                "semantic_review": None,
                "action": "reshoot",
                "reasons": [f"frame analysis failed: {exc}"],
                "trim_start_s": 0.0,
                "trim_end_s": 0.0,
            }

        action = entry["action"]
        report["summary"][action].append(shot_dir.name)
        if action != "keep":
            report["has_issues"] = True
            print(f"  ⚠ [8.2] {shot_dir.name}: {action} — {'; '.join(entry['reasons'])}", flush=True)
        else:
            print(f"  ✓ [8.2] {shot_dir.name}: {len(entry['frames'])} 帧通过", flush=True)
        report["shots"][shot_dir.name] = entry

    report["summary"].update({
        "reviewed_shots": len(report["shots"]),
        "sampled_frames": sum(len(item["frames"]) for item in report["shots"].values()),
    })
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
