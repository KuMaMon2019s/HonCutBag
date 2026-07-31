#!/usr/bin/env python3
"""Assembly Engine — Phase 4 video assembly (stitch + subtitle burn + OCC transitions).

Importable by orchestrator.py. Provides:
    stitch_clips()       — OM video_stitch for concatenating shot clips
    burn_subtitles()     — OM remotion_caption_burn for overlaying captions
    add_occ_transitions() — OCC MCP stub for transition insertion
    assemble()           — main entry: scan shots dir → stitch → burn → final.mp4

Usage (from orchestrator):
    from assembly_engine import assemble
    final = assemble(shots_dir, output_final)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── OM tool paths ────────────────────────────────────────────────────────────
OM_VIDEO_DIR = Path(__file__).resolve().parent.parent / "openmontage" / "tools" / "video"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# OCC MCP endpoint (Phase 4 context)
OCC_MCP_URL = "http://localhost:5173/api/external-mcp/mcp"
OCC_BEARER_TOKEN = "occ-panda-secret"


# ─── 0. preprocess helpers (silence removal + waste trimming) ────────────────

def _om_available() -> bool:
    """Check if OM video tools directory exists and is importable."""
    return OM_VIDEO_DIR.exists() and (OM_VIDEO_DIR / "silence_cutter.py").exists()


def _trim_silence(clip_path: str, output_path: str) -> str:
    """Remove head/tail silence from a clip using OM SilenceCutter.

    Returns the output path on success, or the original path on failure.
    """
    try:
        if str(OM_VIDEO_DIR) not in sys.path:
            sys.path.insert(0, str(OM_VIDEO_DIR))
        from silence_cutter import SilenceCutter

        cutter = SilenceCutter()
        result = cutter.execute({
            "input_path": clip_path,
            "output_path": output_path,
            "mode": "remove",
            "silence_threshold_db": -35,
            "min_silence_duration": 0.4,
            "padding_seconds": 0.05,
        })
        if result.success and Path(output_path).exists():
            logger.info("Silence trimmed: %s → %s", clip_path, output_path)
            return output_path
        else:
            logger.warning("SilenceCutter returned no output for %s: %s", clip_path, result.error)
            return clip_path
    except ImportError:
        logger.debug("SilenceCutter not available, skipping silence trim")
        return clip_path
    except Exception as e:
        logger.warning("Silence trim failed for %s: %s", clip_path, e)
        return clip_path


def _trim_waste(clip_path: str, output_path: str) -> str:
    """Trim black/static frames from head/tail of a clip using FFmpeg blackdetect.

    Detects leading/trailing black segments and cuts them.
    Returns the output path on success, or the original path on failure.
    """
    import subprocess
    import re

    try:
        # Detect black segments using blackdetect filter
        cmd = [
            "ffmpeg", "-i", clip_path,
            "-vf", "blackdetect=d=0.3:pix_th=0.10",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr

        # Parse blackdetect output: black_start: X black_end: Y
        blacks = re.findall(
            r"black_start:\s*([\d.]+)\s*black_end:\s*([\d.]+)", stderr
        )
        if not blacks:
            logger.debug("No black segments detected in %s", clip_path)
            return clip_path

        # Get total duration
        total_dur = _probe_duration(clip_path)

        # Determine cut points: trim leading black, trim trailing black
        cut_start = 0.0
        cut_end = total_dur

        first_black = (float(blacks[0][0]), float(blacks[0][1]))
        last_black = (float(blacks[-1][0]), float(blacks[-1][1]))

        # If first black segment starts at ~0, trim it
        if first_black[0] < 0.3:
            cut_start = first_black[1]
            logger.info("Trimming leading black: 0 → %.2fs", cut_start)

        # If last black segment ends near total duration, trim it
        if abs(last_black[1] - total_dur) < 0.5:
            cut_end = last_black[0]
            logger.info("Trimming trailing black: %.2fs → %.2fs", cut_end, total_dur)

        # If nothing to trim, return original
        if cut_start < 0.05 and cut_end >= total_dur - 0.05:
            return clip_path

        # Ensure we have a valid segment
        if cut_end - cut_start < 0.5:
            logger.warning("Trimmed segment too short (%.2fs), skipping waste trim", cut_end - cut_start)
            return clip_path

        # Cut the clip
        cmd = [
            "ffmpeg", "-y", "-i", clip_path,
            "-ss", f"{cut_start:.3f}",
            "-to", f"{cut_end:.3f}",
            "-c", "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)

        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            logger.info("Waste trimmed: %s → %s (%.2fs → %.2fs)",
                        clip_path, output_path, total_dur, cut_end - cut_start)
            return output_path
        return clip_path

    except Exception as e:
        logger.warning("Waste trim failed for %s: %s", clip_path, e)
        return clip_path


def preprocess_clips(clip_paths: list[str]) -> list[str]:
    """Preprocess shot clips: remove silence and trim waste (black/static frames).

    For each clip:
        1. Remove head/tail silence via OM SilenceCutter
        2. Trim black/static frames via FFmpeg blackdetect

    If OM tools are not available, returns original paths unchanged (graceful fallback).

    Args:
        clip_paths: List of absolute paths to clip .mp4 files.

    Returns:
        List of paths to preprocessed clips (may be same as input if tools unavailable).
    """
    if not _om_available():
        logger.info("OM tools not available — skipping clip preprocessing")
        return clip_paths

    processed = []
    for clip_path in clip_paths:
        if not Path(clip_path).exists():
            logger.warning("Clip not found, skipping preprocess: %s", clip_path)
            processed.append(clip_path)
            continue

        clip_p = Path(clip_path)
        work_dir = clip_p.parent / ".preprocess_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        current = clip_path

        # Step 1: Remove silence
        silence_out = str(work_dir / f"{clip_p.stem}_nosilence{clip_p.suffix}")
        current = _trim_silence(current, silence_out)

        # Step 2: Trim waste (black/static frames)
        waste_out = str(work_dir / f"{clip_p.stem}_clean{clip_p.suffix}")
        current = _trim_waste(current, waste_out)

        processed.append(current)

    n_changed = sum(1 for a, b in zip(clip_paths, processed) if a != b)
    logger.info("Preprocessed %d/%d clips", n_changed, len(clip_paths))
    return processed


def global_silence_cut(video_path: str, output_path: str) -> str:
    """Apply global silence removal to a stitched video.

    Uses OM SilenceCutter with slightly more aggressive settings to remove
    any remaining long silences after stitching.

    Returns output_path on success, or the original video_path on failure.
    """
    if not _om_available():
        logger.info("OM tools not available — skipping global silence cut")
        return video_path

    try:
        if str(OM_VIDEO_DIR) not in sys.path:
            sys.path.insert(0, str(OM_VIDEO_DIR))
        from silence_cutter import SilenceCutter

        cutter = SilenceCutter()
        result = cutter.execute({
            "input_path": video_path,
            "output_path": output_path,
            "mode": "remove",
            "silence_threshold_db": -30,
            "min_silence_duration": 0.8,
            "padding_seconds": 0.1,
        })
        if result.success and Path(output_path).exists():
            logger.info("Global silence cut: %s → %s", video_path, output_path)
            return output_path
        else:
            logger.warning("Global silence cut failed: %s", result.error)
            return video_path
    except ImportError:
        logger.debug("SilenceCutter not available, skipping global silence cut")
        return video_path
    except Exception as e:
        logger.warning("Global silence cut failed: %s", e)
        return video_path


# ─── 1. stitch_clips ─────────────────────────────────────────────────────────

def stitch_clips(
    clip_paths: list[str],
    output_path: str,
    transition: str = "crossfade",
    duration: float = 0.5,
) -> str:
    """Stitch multiple video clips into one using OM VideoStitch.

    Args:
        clip_paths: Ordered list of absolute paths to .mp4 clips.
        output_path: Destination path for the stitched video.
        transition: Transition type between clips (e.g. "crossfade", "cut", "dissolve").
        duration: Transition duration in seconds.

    Returns:
        Absolute path to the output video.

    Raises:
        FileNotFoundError: If any clip path doesn't exist.
        RuntimeError: If VideoStitch execution fails.
    """
    # Validate inputs
    for p in clip_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Clip not found: {p}")

    # Import OM VideoStitch
    if str(OM_VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(OM_VIDEO_DIR))
    try:
        from video_stitch import VideoStitch
    except ImportError as e:
        raise ImportError(
            f"Cannot import VideoStitch from {OM_VIDEO_DIR}. "
            "Ensure OpenMontage is available."
        ) from e

    # Build operation payload
    params = {
        "operation": "stitch",
        "clips": [str(p) for p in clip_paths],
        "output_path": str(output_path),
        "transition": transition,
        "transition_duration": duration,
    }

    logger.info("Stitching %d clips → %s (transition=%s, dur=%.2fs)",
                len(clip_paths), output_path, transition, duration)

    stitcher = VideoStitch()
    result = stitcher.execute(params)

    if not Path(output_path).exists():
        raise RuntimeError(f"VideoStitch completed but output not found: {output_path}")

    logger.info("Stitch complete: %s", output_path)
    return str(output_path)


# ─── 2. burn_subtitles ───────────────────────────────────────────────────────

def burn_subtitles(
    video_path: str,
    captions: list[dict],
    output_path: str,
) -> str:
    """Burn subtitle captions onto a video using OM RemotionCaptionBurn.

    Args:
        video_path: Path to the input video.
        captions: List of caption dicts, each with keys:
            - "text" (str): The subtitle text.
            - "start" (float): Start time in seconds.
            - "end" (float): End time in seconds.
            - "style" (dict, optional): Font size, color, position, etc.
        output_path: Destination path for the captioned video.

    Returns:
        Absolute path to the output video with burned subtitles.

    Raises:
        FileNotFoundError: If input video doesn't exist.
        RuntimeError: If caption burn fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not captions:
        logger.warning("No captions to burn; copying input → output")
        import shutil
        shutil.copy2(video_path, output_path)
        return str(output_path)

    # Set Chrome path for Remotion rendering
    os.environ["CHROME_PATH"] = CHROME_PATH

    # Import OM RemotionCaptionBurn
    if str(OM_VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(OM_VIDEO_DIR))
    try:
        from remotion_caption_burn import RemotionCaptionBurn
    except ImportError as e:
        raise ImportError(
            f"Cannot import RemotionCaptionBurn from {OM_VIDEO_DIR}. "
            "Ensure OpenMontage is available."
        ) from e

    params = {
        "video_path": str(video_path),
        "captions": captions,
        "output_path": str(output_path),
    }

    logger.info("Burning %d captions onto %s → %s",
                len(captions), video_path, output_path)

    burner = RemotionCaptionBurn()
    result = burner.execute(params)

    if not Path(output_path).exists():
        raise RuntimeError(f"RemotionCaptionBurn completed but output not found: {output_path}")

    logger.info("Subtitle burn complete: %s", output_path)
    return str(output_path)


# ─── 3. add_occ_transitions (OCC MCP stub) ───────────────────────────────────

def add_occ_transitions(
    project_id: str,
    clip_list: list[dict],
    transition_type: str = "dissolve",
    transition_duration: float = 0.5,
) -> dict:
    """DEPRECATED: OCC 已归档，此函数保留为参考。

    Add transitions between clips via OpenChatCut MCP (stub).

    This function demonstrates the OCC MCP tool-call pattern:
        1. begin_edit_session(project_id) → session_id
        2. add_clip(session_id, clip_path, track, position) for each clip
        3. add_transition(session_id, clip_a_id, clip_b_id, params)
           where params = {"type": "dissolve", "duration": 0.5}
        4. submit_export(session_id, output_path) → export_job_id

    Args:
        project_id: OCC project identifier.
        clip_list: Ordered list of clip dicts, each with:
            - "path" (str): Absolute path to the clip.
            - "track" (int, optional): Timeline track index (default 0).
            - "position" (float, optional): Start time on track (default auto).
        transition_type: OCC transition type (e.g. "dissolve", "wipe", "cut").
        transition_duration: Transition duration in seconds.

    Returns:
        Dict with keys: session_id, clip_ids, transition_ids, export_job_id.

    Note:
        This is a STUB — actual HTTP calls to OCC MCP are not yet implemented.
        The function documents the exact MCP tool sequence for future integration.
        Endpoint: POST {OCC_MCP_URL} with Authorization: Bearer {OCC_BEARER_TOKEN}
    """
    import requests

    headers = {
        "Authorization": f"Bearer {OCC_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }

    def _mcp_call(tool_name: str, arguments: dict) -> dict:
        """Make a single MCP tool call to OCC."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": 1,
        }
        resp = requests.post(OCC_MCP_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("result", {})

    # Step 1: Begin edit session
    session_result = _mcp_call("begin_edit_session", {"project_id": project_id})
    session_id = session_result.get("session_id", "")
    logger.info("OCC session started: %s", session_id)

    # Step 2: Add clips to timeline
    clip_ids = []
    for i, clip in enumerate(clip_list):
        clip_result = _mcp_call("add_clip", {
            "session_id": session_id,
            "clip_path": clip["path"],
            "track": clip.get("track", 0),
            "position": clip.get("position", None),
        })
        clip_ids.append(clip_result.get("clip_id", f"clip_{i}"))

    # Step 3: Add transitions between consecutive clips
    transition_ids = []
    for i in range(len(clip_ids) - 1):
        trans_result = _mcp_call("add_transition", {
            "session_id": session_id,
            "clip_a_id": clip_ids[i],
            "clip_b_id": clip_ids[i + 1],
            "params": {
                "type": transition_type,
                "duration": transition_duration,
            },
        })
        transition_ids.append(trans_result.get("transition_id", f"trans_{i}"))

    # Step 4: Submit export
    export_result = _mcp_call("submit_export", {
        "session_id": session_id,
        "output_path": f"/tmp/occ_export_{project_id}.mp4",
    })
    export_job_id = export_result.get("job_id", "")

    logger.info("OCC export submitted: job=%s", export_job_id)

    return {
        "session_id": session_id,
        "clip_ids": clip_ids,
        "transition_ids": transition_ids,
        "export_job_id": export_job_id,
    }


# ─── 4. assemble (main entry point) ─────────────────────────────────────────

def assemble(shot_dir_path: str, output_final: str) -> str:
    """Main assembly entry point: scan shots, preprocess, stitch, burn subtitles.

    Workflow:
        1. Scan shot_dir_path/shots/S*/*.mp4 for generated shot clips.
        2. Read SHOT_META.json from each shot dir for caption data.
        3. Preprocess clips: remove silence and trim waste (black/static frames).
        4. Call stitch_clips() to concatenate all shots into one video.
        5. Apply global silence cut to remove any remaining long silences.
        6. Call burn_subtitles() to overlay captions on the final video.

    Args:
        shot_dir_path: Path to the shots directory (contains S01/, S02/, etc.).
        output_final: Path for the final assembled video (e.g. output/final.mp4).

    Returns:
        Absolute path to the final assembled video.

    Raises:
        FileNotFoundError: If shot directory doesn't exist or contains no clips.
        RuntimeError: If assembly steps fail.
    """
    shots_dir = Path(shot_dir_path)
    if not shots_dir.exists():
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")

    # Discover shot clips in sorted order
    shot_dirs = sorted(shots_dir.glob("S*"))
    if not shot_dirs:
        raise FileNotFoundError(f"No shot directories (S*) found in {shots_dir}")

    clip_paths = []
    all_captions = []
    current_time = 0.0

    for sd in shot_dirs:
        # Find the .mp4 clip in this shot dir
        clips = sorted(sd.glob("*.mp4"))
        if not clips:
            logger.warning("No .mp4 found in %s — skipping", sd)
            continue

        clip_path = clips[0]
        clip_paths.append(str(clip_path))

        # Read caption data from SHOT_META.json
        meta_path = sd / "SHOT_META.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            captions = meta.get("captions", [])
            for cap in captions:
                # Offset caption times by current clip position in timeline
                offset_cap = {
                    "text": cap.get("text", ""),
                    "start": cap.get("start", 0.0) + current_time,
                    "end": cap.get("end", 0.0) + current_time,
                }
                if cap.get("style"):
                    offset_cap["style"] = cap["style"]
                all_captions.append(offset_cap)
        else:
            logger.warning("No SHOT_META.json in %s — no captions for this shot", sd)

        # Estimate clip duration (probe with ffprobe if available)
        clip_duration = _probe_duration(str(clip_path))
        current_time += clip_duration

    if not clip_paths:
        raise FileNotFoundError("No .mp4 clips found in any shot directory")

    logger.info("Assembling %d clips, %d total captions", len(clip_paths), len(all_captions))

    # Step 1: Preprocess clips (remove silence + trim waste)
    logger.info("Step 1/4: Preprocessing clips (silence removal + waste trimming)")
    preprocessed_clips = preprocess_clips(clip_paths)

    # Step 2: Stitch all clips
    logger.info("Step 2/4: Stitching clips")
    stitched_path = str(Path(output_final).parent / "stitched.mp4")
    stitch_clips(preprocessed_clips, stitched_path, transition="crossfade", duration=0.5)

    # Step 3: Global silence cut on stitched video
    logger.info("Step 3/4: Applying global silence cut")
    stitched_clean_path = str(Path(output_final).parent / "stitched_clean.mp4")
    stitched_clean_path = global_silence_cut(stitched_path, stitched_clean_path)

    # Step 4: Burn subtitles
    logger.info("Step 4/4: Burning subtitles")
    burn_subtitles(stitched_clean_path, all_captions, output_final)

    # Step 3: Extract frames from final video for QA verification (1 fps)
    final_frames_dir = Path(output_final).parent / "final_frames"
    final_frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_final, "-vf", "fps=1",
             str(final_frames_dir / "frame_%04d.jpg")],
            capture_output=True, timeout=60, check=True,
        )
        frame_count = len(list(final_frames_dir.glob("frame_*.jpg")))
        logger.info("Final frame extraction: %d frames → %s", frame_count, final_frames_dir)
        print(f"[assemble] Extracted {frame_count} frames from final video → {final_frames_dir}")
    except Exception as e:
        logger.warning("Final frame extraction failed: %s", e)

    logger.info("Assembly complete → %s", output_final)
    return str(output_final)


def _probe_duration(video_path: str) -> float:
    """Probe video duration in seconds using ffprobe. Returns 5.0 on failure."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        logger.warning("Could not probe duration for %s, defaulting to 5.0s", video_path)
        return 5.0


# ─── CLI entry (for standalone testing) ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Assembly Engine — stitch + caption burn")
    parser.add_argument("shot_dir", help="Path to shots/ directory")
    parser.add_argument("output", help="Output path for final video")
    args = parser.parse_args()

    final = assemble(args.shot_dir, args.output)
    print(f"Final video: {final}")
