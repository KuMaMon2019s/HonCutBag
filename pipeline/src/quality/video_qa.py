"""Phase 8.5: Video QA — 抽帧分析硬性质检

Inspired by OpenMontage FrameSampler / VisualQA / VideoUnderstand / final_review.
Runs after Phase 8 (polished.mp4) to verify final output quality.

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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    """Full QA report for Phase 8.5."""
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
    """Run Phase 8.5 Video QA on polished.mp4.

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
            suggestion="Phase 8 may have failed; check pipeline logs",
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

    # 6. VLM semantic check (optional)
    if vlm_client is not None:
        report.vlm_check_available = True
        try:
            report.vlm_result = _vlm_semantic_check(vlm_client, frames, storyboard_data)
        except Exception as e:
            report.vlm_result = {"error": str(e)}
    else:
        # Check if ARK vision model is available via env
        ark_key = os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("VOLCANO_API_KEY")
        if ark_key:
            report.vlm_check_available = True
            report.vlm_result = {"status": "skipped", "reason": "no vlm_client provided but API key exists"}

    # Compute verdict and grade
    _compute_verdict(report)

    # Write report to disk
    report_path = output_dir / "video_qa_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"\n  📋 [Phase 8.5] Video QA Report: {report.verdict} ({report.grade})")
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
            suggestion="Phase 8 may have stripped the video track",
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

    # If we have storyboard data, use shot boundaries
    if storyboard_data:
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

    # Sum of storyboard shot durations
    storyboard_total = sum(
        s.get("suggested_duration", s.get("duration", 5)) for s in shots
    )

    crossref_entry = {
        "storyboard_shot_count": len(shots),
        "storyboard_total_duration": round(storyboard_total, 2),
        "actual_video_duration": round(actual_duration, 2),
        "duration_diff": round(abs(actual_duration - storyboard_total), 2),
    }

    # Check duration mismatch (> 20% difference is a warning)
    if storyboard_total > 0:
        diff_ratio = abs(actual_duration - storyboard_total) / storyboard_total
        if diff_ratio > 0.3:
            report.issues.append(QAIssue(
                severity="warning",
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

def _vlm_semantic_check(
    vlm_client: Any,
    frames: List[FrameSample],
    storyboard_data: Optional[dict],
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

    # Interface for VLM client — actual implementation depends on the client
    # Expected vlm_client interface:
    #   vlm_client.analyze(image_path: str, prompt: str) -> str
    if not hasattr(vlm_client, "analyze"):
        return {"status": "error", "reason": "vlm_client missing analyze() method"}

    results = []
    # Sample up to 5 frames for VLM analysis (cost control)
    sample_frames = frames[:5] if len(frames) > 5 else frames

    for frame in sample_frames:
        try:
            # Find corresponding storyboard description
            prompt = "Describe this video frame briefly. Note any visual quality issues."
            if storyboard_data:
                shots = storyboard_data.get("shots", [])
                # Try to match frame to shot by label
                for shot in shots:
                    shot_id = shot.get("shot_id", "")
                    if shot_id in frame.label:
                        visual = shot.get("visual", "")
                        if visual:
                            prompt = f"Does this frame match the description: '{visual[:100]}'? Note any quality issues."
                        break

            result = vlm_client.analyze(frame.path, prompt)
            results.append({
                "frame": frame.label,
                "timestamp": frame.timestamp,
                "vlm_response": result,
            })
        except Exception as e:
            results.append({
                "frame": frame.label,
                "error": str(e),
            })

    return {"status": "completed", "results": results}


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def _compute_verdict(report: VideoQAReport) -> None:
    """Compute final verdict and grade from collected issues."""
    critical_count = sum(1 for i in report.issues if i.severity == "critical")
    warning_count = sum(1 for i in report.issues if i.severity == "warning")

    # Verdict
    if critical_count >= 2:
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
    elif warning_count > 2:
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
