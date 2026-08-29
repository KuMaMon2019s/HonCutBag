"""Phase 9.5: Video QA — 抽帧分析硬性质检

Combines HonCut frame sampling, visual QA, video understanding, and final review.
Runs after Phase 9 (polished.mp4) to verify final output quality.

Checks:
  1. ffprobe validation — resolution, duration, codec, pixel format, file size
  2. Scene detection — ffmpeg scene change filter
  3. Frame sampling — first/middle/last per shot + transition boundary frames
  4. Black frame / frozen frame / duplicate frame detection (frame differencing)
  5. STORYBOARD / SHOT_META cross-reference — shot durations match storyboard
  6. pass / revise / fail output — fail blocks and downgrades grade

VLM semantic check is optional (uses ARK vision model if available, else leaves interface).

Usage:
    from quality.video_qa import run_video_qa
    report = run_video_qa(output_dir, storyboard_data=...)
    if report["verdict"] == "fail":
        # Block pipeline
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.privacy_visual_policy import (
    SYNTHETIC_QA_CONTRACT,
    synthetic_character_review_evidence,
    synthetic_makeup_qa_requirements,
)
from utils.body_action_contracts import body_action_qa_instruction
from utils.temporal_visual_contracts import apply_temporal_visual_contract

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
    """A single extracted frame."""
    path: str
    timestamp: float
    label: str  # e.g. "S01_first", "S01_mid", "S01_last", "trans_S01_S02_before"


@dataclass
class QAIssue:
    """A single QA issue found."""
    severity: str  # "critical", "warning", "info"
    check: str     # which check found it
    message: str
    suggestion: str = ""


@dataclass
class VideoQAReport:
    """Full QA report for Phase 9.5."""
    verdict: str  # "pass", "revise", "fail"
    grade: str    # A/B/C/D
    issues: List[QAIssue] = field(default_factory=list)
    probe_data: Dict[str, Any] = field(default_factory=dict)
    scene_boundaries: List[float] = field(default_factory=list)
    frames_extracted: List[FrameSample] = field(default_factory=list)
    black_frames: List[dict] = field(default_factory=list)
    frozen_frames: List[dict] = field(default_factory=list)
    duplicate_frames: List[dict] = field(default_factory=list)
    storyboard_crossref: List[dict] = field(default_factory=list)
    vlm_check_available: bool = False
    vlm_result: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "verdict": self.verdict,
            "grade": self.grade,
            "issues": [
                {"severity": i.severity, "check": i.check,
                 "message": i.message, "suggestion": i.suggestion}
                for i in self.issues
            ],
            "probe_data": self.probe_data,
            "scene_boundaries": self.scene_boundaries,
            "frames_extracted_count": len(self.frames_extracted),
            "black_frames_count": len(self.black_frames),
            "frozen_frames_count": len(self.frozen_frames),
            "duplicate_frames_count": len(self.duplicate_frames),
            "storyboard_crossref": self.storyboard_crossref,
            "vlm_check_available": self.vlm_check_available,
            "vlm_result": self.vlm_result,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_video_qa(
    output_dir: Path,
    storyboard_data: Optional[dict] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_min_duration: Optional[float] = None,
    expected_max_duration: Optional[float] = None,
    vlm_client: Optional[Any] = None,
) -> VideoQAReport:
    """Run Phase 9.5 Video QA on polished.mp4.

    Args:
        output_dir: Project output directory (contains polished.mp4)
        storyboard_data: Parsed STORYBOARD.json content (optional)
        expected_width: Expected video width (from media profile)
        expected_height: Expected video height
        expected_min_duration: Minimum expected duration in seconds
        expected_max_duration: Maximum expected duration in seconds
        vlm_client: Optional VLM client for semantic checks (ARK vision model)

    Returns:
        VideoQAReport with verdict, grade, and detailed issues
    """
    output_dir = Path(output_dir)
    video_path = output_dir / "polished.mp4"
    report = VideoQAReport(verdict="pass", grade="A")

    # Check file exists
    if not video_path.exists():
        report.issues.append(QAIssue(
            severity="critical", check="file_exists",
            message="polished.mp4 not found",
            suggestion="Phase 9 may have failed; check pipeline logs",
        ))
        report.verdict = "fail"
        report.grade = "D"
        return report

    # 1. ffprobe validation
    probe = _ffprobe_validate(video_path, report, expected_width, expected_height,
                               expected_min_duration, expected_max_duration)

    # 2. Scene detection
    scene_bounds = _detect_scenes(video_path, report)
    report.scene_boundaries = scene_bounds

    # 3. Frame sampling
    frames = _sample_frames(video_path, output_dir, scene_bounds, storyboard_data, report)
    report.frames_extracted = frames

    # 4. Black / frozen / duplicate frame detection
    _detect_black_frames(video_path, report)
    _detect_frozen_frames(video_path, report)
    _detect_duplicate_frames(video_path, report)

    # 5. STORYBOARD / SHOT_META cross-reference
    if storyboard_data:
        _crossref_storyboard(output_dir, storyboard_data, report)

    # 6. VLM semantic check. A technical-only delivery may still pass, but it
    # is explicitly downgraded so grade A always means semantic review passed.
    if vlm_client is None and os.environ.get("HONCUT_FINAL_VLM_REVIEW", "auto").lower() not in {
        "0", "false", "off", "no",
    }:
        try:
            from clients.ark_multimodal_client import ArkMultimodalClient

            vlm_client = ArkMultimodalClient()
        except Exception:
            vlm_client = None
    if vlm_client is not None:
        report.vlm_check_available = True
        try:
            report.vlm_result = _vlm_semantic_check(
                vlm_client,
                frames,
                storyboard_data,
                output_dir=output_dir,
            )
        except Exception as e:
            report.vlm_result = {"error": str(e)}
    else:
        report.vlm_result = {"status": "unavailable", "reason": "no usable VLM client"}

    semantic = report.vlm_result or {}
    semantic_verdict = str(semantic.get("verdict", "")).lower()
    if semantic_verdict in {"fail", "reshoot", "revise"}:
        report.issues.append(QAIssue(
            severity="critical",
            check="semantic_review",
            message="Final semantic review found storyboard or visual defects: "
            + "; ".join(str(item) for item in semantic.get("issues", [])[:5]),
            suggestion="Correct or reshoot the affected shots before delivery",
        ))
    elif semantic_verdict != "pass":
        report.issues.append(QAIssue(
            severity="warning",
            check="semantic_review_unavailable",
            message="Final semantic review did not complete; grade A is unavailable",
            suggestion="Configure a working multimodal reviewer or inspect frames manually",
        ))

    # Compute verdict and grade
    _compute_verdict(report)

    # Write report to disk
    report_path = output_dir / "video_qa_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"\n  📋 [Phase 9.5] Video QA Report: {report.verdict} ({report.grade})")
    for issue in report.issues:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
        print(f"    {icon} [{issue.check}] {issue.message}")
    if not report.issues:
        print(f"    ✅ No issues found")

    return report


# ---------------------------------------------------------------------------
# 1. ffprobe validation
# ---------------------------------------------------------------------------

def _ffprobe_validate(
    video_path: Path,
    report: VideoQAReport,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_min_duration: Optional[float] = None,
    expected_max_duration: Optional[float] = None,
) -> dict:
    """Run ffprobe and validate video metadata."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration,size:stream=width,height,codec_name,pix_fmt,"
            "r_frame_rate,sample_rate,channels,codec_type,nb_frames",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        probe_data = json.loads(result.stdout)
    except Exception as e:
        report.issues.append(QAIssue(
            severity="critical", check="ffprobe",
            message=f"ffprobe failed: {e}",
            suggestion="Ensure ffprobe is installed and video is not corrupted",
        ))
        return {}

    report.probe_data = probe_data

    # Extract stream info
    video_stream = None
    audio_stream = None
    for s in probe_data.get("streams", []):
        if s.get("codec_type") == "video" and not video_stream:
            video_stream = s
        elif s.get("codec_type") == "audio" and not audio_stream:
            audio_stream = s

    duration = float(probe_data.get("format", {}).get("duration", 0))
    file_size = int(probe_data.get("format", {}).get("size", 0))

    # File size sanity check (> 500KB for a real video)
    if file_size < 512_000:
        report.issues.append(QAIssue(
            severity="critical", check="file_size",
            message=f"File too small: {file_size} bytes (< 512KB)",
            suggestion="Video may be corrupted or incomplete",
        ))

    if video_stream:
        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)

        # Resolution checks
        if expected_width and width != expected_width:
            report.issues.append(QAIssue(
                severity="warning", check="resolution",
                message=f"Width mismatch: expected {expected_width}, got {width}",
            ))
        if expected_height and height != expected_height:
            report.issues.append(QAIssue(
                severity="warning", check="resolution",
                message=f"Height mismatch: expected {expected_height}, got {height}",
            ))

        # Duration checks
        if expected_min_duration and duration < expected_min_duration:
            report.issues.append(QAIssue(
                severity="warning", check="duration",
                message=f"Duration too short: {duration:.1f}s < {expected_min_duration}s",
            ))
        if expected_max_duration and duration > expected_max_duration:
            report.issues.append(QAIssue(
                severity="warning", check="duration",
                message=f"Duration too long: {duration:.1f}s > {expected_max_duration}s",
            ))

        # Codec check
        codec = video_stream.get("codec_name", "")
        if codec and codec not in ("h264", "hevc", "h265", "vp9", "av1", "mpeg4"):
            report.issues.append(QAIssue(
                severity="info", check="codec",
                message=f"Unusual video codec: {codec}",
            ))
    else:
        report.issues.append(QAIssue(
            severity="critical", check="video_stream",
            message="No video stream found in polished.mp4",
            suggestion="Phase 9 may have stripped the video track",
        ))

    if not audio_stream:
        report.issues.append(QAIssue(
            severity="critical",
            check="audio_stream",
            message="No audio stream found in polished.mp4",
            suggestion="Phase 9 must deliver an audible or explicitly designed silent audio track",
        ))

    return probe_data


# ---------------------------------------------------------------------------
# 2. Scene detection
# ---------------------------------------------------------------------------

def _detect_scenes(video_path: Path, report: VideoQAReport, threshold: float = 0.3) -> List[float]:
    """Detect scene change timestamps using ffmpeg scene filter."""
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # Parse showinfo output for pts_time values
        boundaries = [0.0]  # Always include start
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                try:
                    pts_str = line.split("pts_time:")[1].split()[0]
                    boundaries.append(float(pts_str))
                except (ValueError, IndexError):
                    pass
        return sorted(set(round(b, 3) for b in boundaries))
    except Exception as e:
        report.issues.append(QAIssue(
            severity="info", check="scene_detection",
            message=f"Scene detection failed (non-blocking): {e}",
        ))
        return [0.0]


# ---------------------------------------------------------------------------
# 3. Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(
    video_path: Path,
    output_dir: Path,
    scene_boundaries: List[float],
    storyboard_data: Optional[dict],
    report: VideoQAReport,
) -> List[FrameSample]:
    """Extract representative frames: first/mid/last per shot + transition boundary frames."""
    frames_dir = output_dir / "qa_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: List[FrameSample] = []

    # Get video duration
    duration = _get_duration(video_path)
    if duration <= 0:
        return frames

    # Prefer Phase 9's delivery timeline because rhythm changes can move shot
    # boundaries. Fall back to Phase 8's canonical assembly timebase.
    timeline_path = output_dir / "delivery_timeline.json"
    if not timeline_path.is_file():
        timeline_path = output_dir / "edit_timeline.json"
    timeline_shots: list[dict] = []
    if timeline_path.is_file():
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            timeline_shots = timeline.get("shots", [])
        except (OSError, json.JSONDecodeError):
            timeline_shots = []

    if timeline_shots:
        for i, item in enumerate(timeline_shots):
            start = float(item.get("output_start_s", 0.0))
            end = min(float(item.get("output_end_s", start)), duration)
            shot_id = str(item.get("shot_id") or f"S{i+1:02d}")
            if end <= start:
                continue
            for suffix, timestamp in (
                ("first", min(start + 0.1, end - 0.01)),
                ("mid", start + (end - start) / 2),
                ("last", max(start + 0.01, end - 0.1)),
            ):
                frame = _extract_frame(
                    video_path, frames_dir, max(0.0, timestamp), f"{shot_id}_{suffix}"
                )
                if frame:
                    frames.append(frame)
            if i > 0:
                for suffix, timestamp in (
                    ("trans_before", max(start - 0.2, 0.0)),
                    ("trans_after", min(start + 0.2, duration - 0.1)),
                ):
                    frame = _extract_frame(video_path, frames_dir, timestamp, f"{shot_id}_{suffix}")
                    if frame:
                        frames.append(frame)
    # If we have storyboard data but no EDL, use authored shot boundaries.
    elif storyboard_data:
        shots = storyboard_data.get("shots", [])
        cumulative = 0.0
        for i, shot in enumerate(shots):
            shot_dur = shot.get("suggested_duration", shot.get("duration", 5))
            shot_id = shot.get("shot_id", f"S{i+1:02d}")

            # First frame of shot
            ts_first = min(cumulative + 0.1, duration - 0.1)
            f = _extract_frame(video_path, frames_dir, ts_first, f"{shot_id}_first")
            if f:
                frames.append(f)

            # Middle frame of shot
            ts_mid = min(cumulative + shot_dur / 2, duration - 0.1)
            f = _extract_frame(video_path, frames_dir, ts_mid, f"{shot_id}_mid")
            if f:
                frames.append(f)

            # Last frame of shot
            ts_last = min(cumulative + shot_dur - 0.1, duration - 0.1)
            f = _extract_frame(video_path, frames_dir, max(ts_last, 0.1), f"{shot_id}_last")
            if f:
                frames.append(f)

            # Transition boundary frames (before/after shot boundary)
            if i > 0:
                ts_before = max(cumulative - 0.2, 0.0)
                f = _extract_frame(video_path, frames_dir, ts_before, f"{shot_id}_trans_before")
                if f:
                    frames.append(f)

                ts_after = min(cumulative + 0.2, duration - 0.1)
                f = _extract_frame(video_path, frames_dir, ts_after, f"{shot_id}_trans_after")
                if f:
                    frames.append(f)

            cumulative += shot_dur
    else:
        # No storyboard: use scene boundaries or uniform sampling
        if len(scene_boundaries) > 1:
            for i, sb in enumerate(scene_boundaries):
                f = _extract_frame(video_path, frames_dir, sb, f"scene_{i:02d}")
                if f:
                    frames.append(f)
        else:
            # Uniform: 5 frames across the video
            for i in range(5):
                ts = duration * (i + 0.5) / 5
                f = _extract_frame(video_path, frames_dir, ts, f"uniform_{i:02d}")
                if f:
                    frames.append(f)

    return frames


def _extract_frame(video_path: Path, output_dir: Path, timestamp: float, label: str) -> Optional[FrameSample]:
    """Extract a single frame at the given timestamp."""
    out_path = output_dir / f"{label}.jpg"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(max(timestamp, 0)),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out_path.exists() and out_path.stat().st_size > 100:
            return FrameSample(path=str(out_path), timestamp=timestamp, label=label)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 4. Black / frozen / duplicate frame detection
# ---------------------------------------------------------------------------

def _detect_black_frames(video_path: Path, report: VideoQAReport,
                         threshold: float = 5.0, max_ratio: float = 0.1) -> None:
    """Detect black frames using ffmpeg blackdetect filter."""
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"blackdetect=d=0.05:pix_th=0.10",
            "-an", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        black_segments = []
        for line in result.stderr.split("\n"):
            if "black_start:" in line:
                try:
                    start = float(line.split("black_start:")[1].split()[0])
                    end_str = line.split("black_end:")[1].split()[0]
                    end = float(end_str)
                    dur = float(line.split("black_duration:")[1].split()[0])
                    black_segments.append({"start": start, "end": end, "duration": dur})
                except (ValueError, IndexError):
                    pass

        report.black_frames = black_segments

        # Check total black duration ratio
        duration = _get_duration(video_path)
        if duration > 0:
            total_black = sum(s["duration"] for s in black_segments)
            ratio = total_black / duration
            if ratio > max_ratio:
                report.issues.append(QAIssue(
                    severity="critical" if ratio > 0.3 else "warning",
                    check="black_frames",
                    message=f"Black frames occupy {ratio*100:.1f}% of video ({total_black:.1f}s / {duration:.1f}s)",
                    suggestion="Check for rendering failures or missing source media",
                ))
            elif black_segments:
                report.issues.append(QAIssue(
                    severity="info", check="black_frames",
                    message=f"Found {len(black_segments)} black segment(s) ({total_black:.1f}s total)",
                ))
    except Exception as e:
        report.issues.append(QAIssue(
            severity="info", check="black_detect",
            message=f"Black frame detection failed (non-blocking): {e}",
        ))


def _detect_frozen_frames(video_path: Path, report: VideoQAReport,
                          threshold: float = 0.01, max_ratio: float = 0.15) -> None:
    """Detect frozen (static) frames using ffmpeg freezedetect filter."""
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"freezedetect=n={threshold}:d=2",
            "-an", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        frozen_segments = []
        for line in result.stderr.split("\n"):
            if "lavfi.freezedetect" in line:
                try:
                    if "freeze_start:" in line:
                        ts = float(line.split("freeze_start:")[1].split()[0])
                        frozen_segments.append({"start": ts, "type": "start"})
                    elif "freeze_duration:" in line:
                        dur = float(line.split("freeze_duration:")[1].split()[0])
                        if frozen_segments and frozen_segments[-1]["type"] == "start":
                            frozen_segments[-1]["duration"] = dur
                            frozen_segments[-1]["type"] = "complete"
                except (ValueError, IndexError):
                    pass

        completed = [s for s in frozen_segments if s.get("type") == "complete"]
        report.frozen_frames = completed

        duration = _get_duration(video_path)
        if duration > 0:
            total_frozen = sum(s.get("duration", 0) for s in completed)
            ratio = total_frozen / duration
            if ratio > max_ratio:
                report.issues.append(QAIssue(
                    severity="critical" if ratio > 0.3 else "warning",
                    check="frozen_frames",
                    message=f"Frozen frames occupy {ratio*100:.1f}% of video ({total_frozen:.1f}s / {duration:.1f}s)",
                    suggestion="Video may have rendering stalls or duplicate frame issues",
                ))
            elif completed:
                report.issues.append(QAIssue(
                    severity="info", check="frozen_frames",
                    message=f"Found {len(completed)} frozen segment(s) ({total_frozen:.1f}s total)",
                ))
    except Exception as e:
        report.issues.append(QAIssue(
            severity="info", check="frozen_detect",
            message=f"Frozen frame detection failed (non-blocking): {e}",
        ))


def _detect_duplicate_frames(video_path: Path, report: VideoQAReport,
                             sample_interval: float = 1.0, diff_threshold: float = 2.0) -> None:
    """Detect consecutive duplicate frames via frame differencing.

    Samples frames at intervals and checks if consecutive samples are identical
    (very low pixel difference), indicating frozen/duplicate content.
    """
    try:
        duration = _get_duration(video_path)
        if duration <= 0:
            return

        with tempfile.TemporaryDirectory(prefix="honcut_qa_dup_") as tmpdir:
            tmpdir = Path(tmpdir)
            # Extract frames at sample_interval
            pattern = str(tmpdir / "frame_%04d.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps=1/{sample_interval}",
                "-q:v", "5",
                pattern,
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            frame_files = sorted(tmpdir.glob("frame_*.jpg"))
            if len(frame_files) < 3:
                return

            # Compare consecutive frames by file size (quick heuristic)
            dup_count = 0
            dup_segments = []
            for i in range(1, len(frame_files)):
                prev_size = frame_files[i-1].stat().st_size
                curr_size = frame_files[i].stat().st_size
                # If sizes are within 1% they're likely duplicates
                if prev_size > 0 and abs(curr_size - prev_size) / prev_size < 0.01:
                    dup_count += 1
                    dup_segments.append({
                        "frame_index": i,
                        "timestamp": round(i * sample_interval, 2),
                        "size_diff_ratio": abs(curr_size - prev_size) / prev_size,
                    })

            report.duplicate_frames = dup_segments

            # If > 50% of sampled frames are duplicates, that's a problem
            if len(frame_files) > 2:
                dup_ratio = dup_count / (len(frame_files) - 1)
                if dup_ratio > 0.5:
                    report.issues.append(QAIssue(
                        severity="warning",
                        check="duplicate_frames",
                        message=f"{dup_ratio*100:.0f}% of sampled frames appear duplicated ({dup_count}/{len(frame_files)-1})",
                        suggestion="Video may contain extended static sections",
                    ))
    except Exception as e:
        report.issues.append(QAIssue(
            severity="info", check="duplicate_detect",
            message=f"Duplicate frame detection failed (non-blocking): {e}",
        ))


# ---------------------------------------------------------------------------
# 5. STORYBOARD / SHOT_META cross-reference
# ---------------------------------------------------------------------------

def _crossref_storyboard(
    output_dir: Path,
    storyboard_data: dict,
    report: VideoQAReport,
) -> None:
    """Cross-reference actual video with STORYBOARD.json and SHOT_META.json."""
    shots = storyboard_data.get("shots", [])
    if not shots:
        return

    video_path = output_dir / "polished.mp4"
    actual_duration = _get_duration(video_path)

    # Compare against the authoritative delivery/edit timeline when available.
    authored_total = sum(
        s.get("suggested_duration", s.get("duration", 5)) for s in shots
    )
    timeline_path = output_dir / "delivery_timeline.json"
    if not timeline_path.is_file():
        timeline_path = output_dir / "edit_timeline.json"
    timeline = {}
    if timeline_path.is_file():
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timeline = {}
    storyboard_total = float(timeline.get("duration_s") or authored_total)

    crossref_entry = {
        "storyboard_shot_count": len(shots),
        "storyboard_total_duration": round(storyboard_total, 2),
        "authored_total_duration": round(authored_total, 2),
        "duration_basis": timeline_path.name if timeline else "storyboard",
        "actual_video_duration": round(actual_duration, 2),
        "duration_diff": round(abs(actual_duration - storyboard_total), 2),
    }

    # A gross duration mismatch is blocking: it indicates dropped segments or
    # timestamp inflation, not merely a cosmetic QA concern.
    if storyboard_total > 0:
        diff_ratio = abs(actual_duration - storyboard_total) / storyboard_total
        if diff_ratio > 0.2:
            report.issues.append(QAIssue(
                severity="critical",
                check="storyboard_duration",
                message=f"Video duration ({actual_duration:.1f}s) differs from storyboard total ({storyboard_total:.1f}s) by {diff_ratio*100:.0f}%",
                suggestion="Phase 7 assembly may have dropped or added segments",
            ))
            crossref_entry["duration_match"] = False
        else:
            crossref_entry["duration_match"] = True
    else:
        crossref_entry["duration_match"] = None

    # Check individual shot directories for SHOT_META consistency
    shots_dir = output_dir / "shots"
    if shots_dir.exists():
        shot_dirs = sorted([d for d in shots_dir.iterdir() if d.is_dir()])
        missing_meta = []
        missing_video = []
        for sd in shot_dirs:
            meta_path = sd / "SHOT_META.json"
            video_path_shot = sd / "output.mp4"
            if not meta_path.exists():
                missing_meta.append(sd.name)
            if not video_path_shot.exists():
                missing_video.append(sd.name)

        crossref_entry["shots_with_missing_meta"] = missing_meta
        crossref_entry["shots_with_missing_video"] = missing_video

        if missing_video and len(missing_video) > len(shot_dirs) * 0.3:
            report.issues.append(QAIssue(
                severity="warning",
                check="shot_videos",
                message=f"{len(missing_video)}/{len(shot_dirs)} shot directories missing output.mp4",
                suggestion="Phase 5 video generation may have failed for some shots",
            ))

    report.storyboard_crossref = [crossref_entry]


# ---------------------------------------------------------------------------
# 6. VLM semantic check (optional interface)
# ---------------------------------------------------------------------------

def _batch_character_evidence(
    batch_shot_ids: list[str],
    shot_by_id: dict[str, dict],
    characters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select only identities declared by shots in the current VLM batch."""
    requested: set[str] = set()
    for shot_id in batch_shot_ids:
        shot = shot_by_id.get(shot_id, {})
        for field_name in ("who", "characters"):
            raw_cast = shot.get(field_name) or []
            if isinstance(raw_cast, str):
                raw_cast = [raw_cast]
            requested.update(
                str(value).strip().casefold() for value in raw_cast if value
            )
        requested.update(
            str(asset)[5:].split(":", 1)[0].strip().casefold()
            for asset in (shot.get("associate_assets") or [])
            if isinstance(asset, str) and asset.startswith("char:")
        )
    if not requested:
        return []
    selected = []
    for character in characters:
        keys = {
            str(character.get("id") or "").strip().casefold(),
            str(character.get("name") or "").strip().casefold(),
            *(
                str(alias).strip().casefold()
                for alias in (character.get("aliases") or [])
                if alias
            ),
        }
        if not requested.isdisjoint(keys):
            selected.append(character)
    return selected


def _batch_prop_evidence(
    batch_shot_ids: list[str],
    shot_by_id: dict[str, dict],
    characters: list[dict[str, Any]],
) -> list[Any]:
    """Collect shot-active props plus persistent identity props for selected cast."""
    props: list[Any] = []
    for shot_id in batch_shot_ids:
        shot = shot_by_id.get(shot_id, {})
        for field_name in ("interaction_props", "props"):
            value = shot.get(field_name) or []
            if isinstance(value, (str, dict)):
                value = [value]
            props.extend(value if isinstance(value, list) else [])
    for character in characters:
        props.extend(character.get("identity_props") or [])
    unique: dict[str, Any] = {}
    for prop in props:
        key = json.dumps(prop, ensure_ascii=False, sort_keys=True) if isinstance(prop, dict) else str(prop)
        if key.strip():
            unique.setdefault(key, prop)
    return list(unique.values())


def _vlm_semantic_check(
    vlm_client: Any,
    frames: List[FrameSample],
    storyboard_data: Optional[dict],
    *,
    output_dir: Path | None = None,
) -> dict:
    """Run VLM-based semantic quality check on extracted frames.

    This is an optional interface — if a VLM client (e.g. ARK vision model)
    is provided, it can analyze frames for:
    - Character presence consistency
    - Scene content matching storyboard descriptions
    - Visual quality issues (blur, artifacts, etc.)

    Returns:
        dict with semantic check results
    """
    if not frames:
        return {"status": "skipped", "reason": "no frames to analyze"}

    if not hasattr(vlm_client, "review"):
        return {"status": "error", "reason": "vlm_client missing review() method"}

    shots = [
        shot
        for shot in (storyboard_data or {}).get("shots", [])
        if isinstance(shot, dict)
    ]
    shot_by_id: dict[str, dict] = {}
    for index, shot in enumerate(shots, 1):
        apply_temporal_visual_contract(shot)
        shot_id = str(shot.get("shot_id") or shot.get("id") or f"S{index:02d}")
        if shot_id.isdigit():
            shot_id = f"S{int(shot_id):02d}"
        shot_by_id[shot_id] = shot

    suffixes = ("trans_before", "trans_after", "first", "mid", "last")

    def frame_shot_id(frame: FrameSample) -> str | None:
        for suffix in suffixes:
            marker = f"_{suffix}"
            if frame.label.endswith(marker):
                return frame.label[: -len(marker)]
        return None

    frames_by_shot: dict[str, list[FrameSample]] = {}
    unmatched: list[FrameSample] = []
    for frame in frames:
        shot_id = frame_shot_id(frame)
        if shot_id is None:
            unmatched.append(frame)
        else:
            frames_by_shot.setdefault(shot_id, []).append(frame)

    selected: list[FrameSample] = []
    selected_ids: set[int] = set()

    def add(frame: FrameSample | None) -> None:
        if frame is not None and id(frame) not in selected_ids:
            selected.append(frame)
            selected_ids.add(id(frame))

    def labelled(candidates: list[FrameSample], suffix: str) -> FrameSample | None:
        return next(
            (frame for frame in candidates if frame.label.endswith(f"_{suffix}")),
            None,
        )

    # Preserve one semantic sample for every delivered Sxx. Character and
    # high-risk motion shots receive first/middle/last coverage. This list is
    # intentionally not globally capped: provider limits are handled by the
    # bounded review batches below.
    for shot_id, candidates in frames_by_shot.items():
        shot = shot_by_id.get(shot_id, {})
        has_characters = bool(
            shot.get("who")
            or shot.get("characters")
            or any(
                isinstance(asset, str) and asset.startswith("char:")
                for asset in (shot.get("associate_assets") or [])
            )
        )
        high_risk = bool(
            shot.get("generation_actions")
            or str(shot.get("gen_strategy") or "").lower() == "flf2v"
            or str(shot.get("shot_intent") or "").lower() == "action"
        )
        if has_characters or high_risk:
            for suffix in ("first", "mid", "last"):
                add(labelled(candidates, suffix))
            if high_risk:
                add(labelled(candidates, "trans_before"))
                add(labelled(candidates, "trans_after"))
        else:
            add(labelled(candidates, "mid") or candidates[len(candidates) // 2])

    if not selected:
        # Without shot-labelled frames, retain the legacy bounded uniform
        # behavior instead of pretending per-shot coverage is available.
        sample_count = min(12, len(frames))
        positions = {
            round(index * (len(frames) - 1) / max(1, sample_count - 1))
            for index in range(sample_count)
        }
        for index in sorted(positions):
            add(frames[index])
    else:
        for frame in unmatched:
            add(frame)

    review_batch_size = 12
    batch_results: list[dict] = []
    issues: list[str] = []
    verdict = "pass"
    confidence_values: list[float] = []
    verdict_rank = {"pass": 0, "revise": 1, "fail": 2}
    review_evidence = synthetic_character_review_evidence(output_dir)
    synthetic_review = bool(review_evidence["enabled"])
    qa_contract = (
        SYNTHETIC_QA_CONTRACT
        if synthetic_review
        else "human_visual_anatomy_v1"
    )
    if (
        synthetic_review
        and output_dir is not None
        and not review_evidence["identity_contract_complete"]
    ):
        return {
            "status": "error",
            "verdict": "revise",
            "reason": "synthetic character identity evidence is incomplete",
            "issues": [
                "Synthetic QA requires every character to have an ID, synthetic policy/gender, "
                "canonical face styling, clothing/identity marker, non-human material, and at least two visible styling anchors"
            ],
            "qa_contract": qa_contract,
            "qa_contract_evidence": review_evidence,
            "sampled_frames": [item.label for item in selected],
            "review_batches": 0,
        }
    structure_contract = (
        (
            "All characters in this project are intentionally synthetic stylized CGI characters. "
            "Their only allowed facial treatment is the declared synthetic porcelain makeup: a beautiful "
            "pearl bio-ceramic complexion, one narrow iridescent circuit stripe from temple to cheekbone, "
            "and a soft luminous iris ring around a clear pupil, layered iris and bright catchlights. The "
            "complexion must remain warm, healthy and elegant with coordinated living cheek and lip color, "
            "never gray, blue-gray, bloodless, waxy, corpse-like, haunted or uncanny. The complete face must "
            "stay visible, harmonious, clean, and "
            "recognizably synthetic; veils, face masks, coarse mechanical plates, cracks, scars, and horror "
            "effects are failures. Never flag the declared pearl bio-ceramic material merely for not matching "
            "natural human skin. Judge structural and styling consistency instead: each visible character "
            "must preserve the declared makeup design, colors, iris ring, clothing, and identity markers; an "
            "untreated natural human face is a failure. Require visible "
            "positive evidence of an unintended break, detachment, merge, extra/missing component, "
            "impossible self-intersection, or reference-inconsistent deformation. Continue to detect "
            "face-styling/material color drift, loss of living eye/cheek/lip color, corpse-like or uncanny "
            "styling, identity-marker drift, grotesque damage, action discontinuity, "
            "and wrong spatial order. Storyboard or performance-board grids, labels, arrows, split panels, "
            "or cloned copies of one character must not appear in the final video. "
            "Structured aesthetic QA requirements: "
            f"{json.dumps(synthetic_makeup_qa_requirements(), ensure_ascii=False)}. "
        )
        if synthetic_review
        else (
            "Detect broken human anatomy as well as impossible geometry, but require visible positive "
            "evidence rather than treating occlusion, costume, or camera angle as a defect. "
        )
    )

    for batch_index in range(0, len(selected), review_batch_size):
        sample_frames = selected[batch_index : batch_index + review_batch_size]
        batch_shot_ids = list(
            dict.fromkeys(filter(None, map(frame_shot_id, sample_frames)))
        )
        batch_characters = _batch_character_evidence(
            batch_shot_ids,
            shot_by_id,
            review_evidence["characters"],
        )
        batch_props = _batch_prop_evidence(
            batch_shot_ids,
            shot_by_id,
            batch_characters,
        )
        identity_contract = ""
        if batch_characters:
            identity_label = (
                "Canonical synthetic identities"
                if synthetic_review
                else "Canonical character identities"
            )
            identity_contract = (
                f"{identity_label} for this batch only (treat every ID as distinct and preserve "
                "the listed face styling, materials, clothing, colors and markers exactly): "
                f"{json.dumps(batch_characters, ensure_ascii=False)}. "
                "Shared styling families do not permit identity merging: keep each role's complete "
                "anchor combination separate and never transfer styling or identity props. "
            )
        if batch_props:
            identity_contract += (
                "Props involved in this batch only (preserve owner, count, color, material, geometry, "
                "markings and attachment mode; never substitute or transfer): "
                f"{json.dumps(batch_props, ensure_ascii=False)}. "
            )
        storyboard = {
            "shots": [
                {
                    key: shot_by_id[shot_id].get(key)
                    for key in (
                        "shot_id", "id", "visual", "action", "generation_actions",
                        "body_action_choreography",
                        "who", "where", "time", "time_of_day", "time_window",
                        "temporal_visual_contract", "lighting", "lighting_description",
                        "interaction_props", "props", "associate_assets", "gen_strategy",
                    )
                    if shot_by_id[shot_id].get(key) not in (None, "", [])
                }
                for shot_id in batch_shot_ids
                if shot_id in shot_by_id
            ]
        }
        body_action_qa = " ".join(
            instruction
            for shot_id in batch_shot_ids
            if shot_id in shot_by_id
            for instruction in [body_action_qa_instruction(shot_by_id[shot_id])]
            if instruction
        )
        prompt = (
            f"QA contract: {qa_contract}. "
            "Review these chronologically sampled frames from the final edited film against the storyboard. "
            f"This is semantic review batch {len(batch_results) + 1}; frame labels in order are "
            f"{json.dumps([item.label for item in sample_frames], ensure_ascii=False)}. "
            "Detect wrong subjects or locations, missing key actions, identity drift, structural defects, "
            "modern/watermark/text artifacts, material continuity errors, and any time-of-day mismatch. "
            "For every temporal_visual_contract, enforce its local_clock_window using the listed visible-light "
            "requirements and forbidden cues; rain, neon, cold color grading, or dramatic mood never excuses a "
            "day/night mismatch. A mismatch or first-to-last temporal drift requires revise or fail. Return JSON only with "
            '{"verdict":"pass|revise|fail","issues":["..."],"confidence":0.0}. '
            f"{structure_contract}"
            f"{identity_contract}"
            f"{body_action_qa} "
            f"Storyboard: {json.dumps(storyboard, ensure_ascii=False)}"
        )
        from clients.ark_multimodal_client import review_as
        from schemas.understanding import ShotSemanticReview

        try:
            parsed = review_as(
                vlm_client,
                [Path(item.path) for item in sample_frames],
                prompt,
                ShotSemanticReview,
            ).model_dump()
        except (json.JSONDecodeError, ValueError):
            return {
                "status": "error",
                "reason": "invalid semantic review response",
                "qa_contract": qa_contract,
                "qa_contract_evidence": review_evidence,
                "sampled_frames": [item.label for item in selected],
                "review_batches": len(batch_results) + 1,
            }
        batch_results.append(parsed)
        if verdict_rank[parsed["verdict"]] > verdict_rank[verdict]:
            verdict = parsed["verdict"]
        issues.extend(str(issue) for issue in parsed.get("issues", []) if issue)
        try:
            confidence_values.append(float(parsed["confidence"]))
        except (KeyError, TypeError, ValueError):
            pass

    return {
        "status": "completed",
        "verdict": verdict,
        "issues": issues,
        "qa_contract": qa_contract,
        "qa_contract_evidence": review_evidence,
        "confidence": min(confidence_values) if confidence_values else None,
        "sampled_frames": [item.label for item in selected],
        "covered_shots": list(frames_by_shot),
        "review_batches": len(batch_results),
        "batch_results": batch_results,
    }


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def _compute_verdict(report: VideoQAReport) -> None:
    """Compute final verdict and grade from collected issues."""
    critical_count = sum(1 for i in report.issues if i.severity == "critical")
    warning_count = sum(1 for i in report.issues if i.severity == "warning")

    # Verdict
    if any(i.check == "storyboard_duration" and i.severity == "critical"
           for i in report.issues):
        report.verdict = "fail"
    elif critical_count >= 2:
        report.verdict = "fail"
    elif critical_count >= 1:
        report.verdict = "revise"
    elif warning_count >= 3:
        report.verdict = "revise"
    else:
        report.verdict = "pass"

    # Grade
    if critical_count >= 3:
        report.grade = "D"
    elif critical_count >= 1:
        report.grade = "C"
    elif warning_count > 2 or any(
        issue.check == "semantic_review_unavailable" for issue in report.issues
    ):
        report.grade = "B"
    else:
        report.grade = "A"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0
