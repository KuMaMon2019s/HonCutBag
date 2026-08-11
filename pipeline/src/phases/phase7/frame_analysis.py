"""Phase 7.2 deterministic frame sampling and local heuristics."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


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
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return float(completed.stdout.strip().splitlines()[0])


def extract_three_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    duration = probe_duration(video_path)
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Seeking to the mathematical 99% point can land after the final decodable
    # frame in short/CFR clips. 95% remains representative of the tail while
    # leaving enough room for one frame interval.
    timestamps = (("00", 0.0), ("50", duration * 0.5), ("99", max(0.0, duration * 0.95)))
    outputs: list[Path] = []
    for label, timestamp in timestamps:
        output = frames_dir / f"frame_{label}.jpg"
        completed = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{timestamp:.6f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(output)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = completed.stderr.strip().splitlines()
            raise RuntimeError(f"ffmpeg frame extraction failed for {video_path} at {timestamp:.3f}s: {detail[-1] if detail else 'unknown error'}")
        outputs.append(output)
    return outputs


def analyze_shot_frames(shots_dir: Path, output_path: Path) -> dict:
    report: dict = {"shots": {}, "has_issues": False}
    for shot_dir in sorted(Path(shots_dir).iterdir()) if Path(shots_dir).is_dir() else []:
        video = shot_dir / "output.mp4"
        if not shot_dir.is_dir() or not shot_dir.name.startswith("S") or not video.is_file():
            continue
        issues: list[str] = []
        frame_paths: list[Path] = []
        black = False
        static = False
        try:
            frame_paths = extract_three_frames(video, shot_dir / "frames")
            black = any(is_black_frame(path) for path in frame_paths)
            static = is_static_frame(frame_paths[0], frame_paths[-1])
            if black:
                issues.append("检测到黑帧（平均像素值 < 20）")
            if static:
                issues.append("首尾帧平均像素差低于 3，镜头可能无运动")
        except Exception as exc:
            issues.append(f"抽帧分析失败: {exc}")
            print(f"  ⚠ [7.2] {shot_dir.name}: {issues[-1]}", flush=True)
        if issues:
            report["has_issues"] = True
            print(f"  ⚠ [7.2] {shot_dir.name}: {'; '.join(issues)}", flush=True)
        report["shots"][shot_dir.name] = {
            "frames": [str(path.relative_to(Path(shots_dir).parent)) for path in frame_paths],
            "black_frame": black,
            "static_frame": static,
            "issues": issues,
        }
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
