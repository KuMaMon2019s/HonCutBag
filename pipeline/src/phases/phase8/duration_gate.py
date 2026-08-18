"""Phase 8 duration measurement and reshoot planning."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .frame_analysis import probe_duration


def trim_excess_to_target(
    output_dir: Path,
    target_duration: float | None,
    *,
    timeline_fps: int = 30,
    rounding_tolerance_frames: int = 2,
) -> dict | None:
    """Close codec rounding only; never delete a material surplus from the tail."""
    if target_duration is None:
        return None
    video = Path(output_dir) / "raw_assembly.mp4"
    actual = probe_duration(video)
    target = float(target_duration)
    target_frames = round(target * timeline_fps)
    actual_frames = round(actual * timeline_fps)
    excess_frames = actual_frames - target_frames
    if excess_frames <= 0:
        return None
    if excess_frames > rounding_tolerance_frames:
        raise RuntimeError(
            "refusing destructive tail trim: reviewed edit is "
            f"{excess_frames} frames over target, above the "
            f"{rounding_tolerance_frames}-frame codec-rounding allowance"
        )
    temporary = video.with_name("raw_assembly.duration_trim.mp4")
    completed = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-filter:v", f"trim=end_frame={target_frames},setpts=PTS-STARTPTS",
            "-filter:a", f"atrim=duration={target_frames / timeline_fps:.9f},asetpts=PTS-STARTPTS",
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            "failed to trim overlong Phase 8 assembly: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    trimmed = probe_duration(temporary)
    if abs(round(trimmed * timeline_fps) - target_frames) > 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"trimmed assembly duration invalid: target={target:.3f}s actual={trimmed:.3f}s"
        )
    temporary.replace(video)
    receipt = {
        "original_s": round(actual, 3),
        "target_s": round(target, 3),
        "trimmed_s": round(trimmed, 3),
        "target_frames": target_frames,
        "trimmed_frames": round(trimmed * timeline_fps),
        "method": "frame_exact_reencode",
        "reason": "codec_rounding_only",
        "discarded_rounding_frames": excess_frames,
    }
    (Path(output_dir) / "duration_trim.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt


def build_reshoot_list(shots_dir: Path, required_gap_s: float, round_number: int) -> dict:
    deficits: list[dict] = []
    for shot_dir in sorted(Path(shots_dir).iterdir()) if Path(shots_dir).is_dir() else []:
        video = shot_dir / "output.mp4"
        meta_path = shot_dir / "SHOT_META.json"
        if not video.is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            requested = float(meta.get("duration") or meta.get("requested_duration") or 0)
            actual = probe_duration(video)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ⚠ [8.3] 无法评估 {shot_dir.name} 时长短板: {exc}", flush=True)
            continue
        gap = max(0.0, requested - actual)
        if gap > 0:
            deficits.append({
                "shot_id": shot_dir.name,
                "requested_s": round(requested, 3),
                "actual_s": round(actual, 3),
                "gap_s": round(gap, 3),
            })
    deficits.sort(key=lambda item: item["gap_s"], reverse=True)
    selected: list[dict] = []
    covered = 0.0
    for item in deficits:
        selected.append(item)
        covered += item["gap_s"]
        if covered >= required_gap_s:
            break
    return {"shots": selected, "round": round_number}


def evaluate_duration_gate(
    output_dir: Path,
    target_duration: float | None,
    round_number: int = 0,
    reshoots: list[dict] | None = None,
    *,
    timeline_fps: int = 30,
    tolerance_frames: int = 2,
) -> tuple[dict, dict | None]:
    output_dir = Path(output_dir)
    actual = probe_duration(output_dir / "raw_assembly.mp4")
    history = list(reshoots or [])
    if target_duration is None:
        gate = {
            "target_s": None,
            "actual_s": round(actual, 3),
            "gap_s": None,
            "passed": True,
            "reshoots": history,
            "skipped_reason": "target_duration is None",
        }
        reshoot_plan = None
    else:
        target = float(target_duration)
        target_frames = round(target * timeline_fps)
        actual_frames = round(actual * timeline_fps)
        delta_frames = actual_frames - target_frames
        gap = max(0.0, -delta_frames / timeline_fps)
        passed = abs(delta_frames) <= tolerance_frames
        gate = {
            "target_s": round(target, 3),
            "actual_s": round(actual, 3),
            "gap_s": round(gap, 3),
            "passed": passed,
            "timeline_fps": timeline_fps,
            "target_frames": target_frames,
            "actual_frames": actual_frames,
            "delta_frames": delta_frames,
            "tolerance_frames": tolerance_frames,
            "reshoots": history,
        }
        reshoot_plan = None if passed else build_reshoot_list(output_dir / "shots", gap, round_number + 1)
        if reshoot_plan is not None:
            (output_dir / "reshoot_list.json").write_text(
                json.dumps(reshoot_plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (output_dir / "duration_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return gate, reshoot_plan
