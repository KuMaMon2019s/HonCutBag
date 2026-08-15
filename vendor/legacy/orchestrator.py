#!/usr/bin/env python3
"""Phase 4 Orchestrator — Storyboard-driven intelligent video generation engine.

Reads STORYBOARD.json, creates per-shot directories with SHOT_META.json,
routes each shot to the appropriate generation mode (img2vid / txt2vid),
and orchestrates Seedance generation + frame extraction.

Usage:
    python orchestrator.py --dry-run          # parse + route only, no API calls
    python orchestrator.py                    # full generation (requires ARK_API_KEY)
    python orchestrator.py --shots S01,S03    # generate specific shots only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORYBOARD_PATH = Path(os.environ.get("STORYBOARD_PATH", "STORYBOARD.json"))
SHOTS_DIR = PROJECT_ROOT / "shots"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Add scripts dir to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import ToolRouter for enhanced routing analysis
try:
    from tool_router import ToolRouter
    _tool_router = ToolRouter()
except ImportError:
    _tool_router = None


# ─── Storyboard Parser ───────────────────────────────────────────────────────

def load_storyboard(path: Path) -> dict:
    """Load and validate storyboard JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Storyboard not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if "shots" not in data:
        raise ValueError("Storyboard missing 'shots' array")
    return data


def parse_shots(storyboard: dict) -> list:
    """Extract shot metadata from storyboard."""
    shots = []
    project_aspect_ratio = storyboard.get("aspect_ratio")
    project_width = storyboard.get("width")
    project_height = storyboard.get("height")
    for s in storyboard["shots"]:
        shot = {
            "id": s["id"],
            "shot_id": f"S{s['id']:02d}",
            "name": s["name"],
            "duration": s.get("duration", 7),
            "prompt": s["prompt"],
            "first_frame": s.get("first_frame"),
            "caption": s.get("caption", ""),
            "caption_frames": s.get("caption_frames", ""),
            "who": s.get("who", []),
            "associate_assets": s.get("associate_assets", []),
            "gen_strategy": s.get("gen_strategy", "i2v"),
            "visual": s.get("visual", ""),
            "what": s.get("what", ""),
            "action_description": s.get("action_description", s.get("what", "")),
            "generation_actions": s.get("generation_actions", []),
            "generation_load": s.get("generation_load"),
            "source_action_unit_ids": s.get("source_action_unit_ids", []),
            "start_state": s.get("start_state", ""),
            "end_state": s.get("end_state", ""),
            "causal_link": s.get("causal_link", ""),
            "shot_type": s.get("shot_type"),
            "shot_size": s.get("shot_size"),
            "camera_movement": s.get("camera_movement"),
            "where": s.get("where", ""),
            "emotion": s.get("emotion", ""),
            "dialogue": s.get("dialogue"),
            "speech_duration_s": s.get("speech_duration_s", 0),
            "audio": s.get("audio", s.get("sound")),
            "aspect_ratio": s.get("aspect_ratio", project_aspect_ratio),
            "width": s.get("width", project_width),
            "height": s.get("height", project_height),
        }
        shots.append(shot)
    return shots


# ─── Tool Router ─────────────────────────────────────────────────────────────

def route_shot(shot: dict, static_refs: dict) -> dict:
    """Determine generation route for a shot.

    Rules:
    - Has first_frame reference → img2vid (use reference image)
    - No first_frame → txt2vid (text-only generation)

    Returns shot dict augmented with 'route' and 'route_reason'.
    """
    first_frame = shot.get("first_frame")

    if first_frame:
        # Determine if it's character-locked or scene-locked
        if "characters/" in first_frame:
            route = "img2vid"
            reason = f"Character-locked: reference image '{first_frame}' provides character consistency"
            ref_type = "character"
        elif "scenes/" in first_frame:
            route = "img2vid"
            reason = f"Scene-locked: reference image '{first_frame}' provides visual continuity"
            ref_type = "scene"
        elif "props/" in first_frame:
            route = "img2vid"
            reason = f"Prop-locked: reference image '{first_frame}' anchors the shot"
            ref_type = "prop"
        else:
            route = "img2vid"
            reason = f"Reference image '{first_frame}' available"
            ref_type = "unknown"

        # Resolve full path for the reference image
        ref_path = STORYBOARD_PATH.parent / first_frame
        shot["route"] = route
        shot["route_reason"] = reason
        shot["ref_type"] = ref_type
        shot["first_frame_path"] = str(ref_path)
        shot["first_frame_exists"] = ref_path.exists()
    else:
        shot["route"] = "txt2vid"
        shot["route_reason"] = "No reference image — pure text-to-video generation"
        shot["ref_type"] = None
        shot["first_frame_path"] = None
        shot["first_frame_exists"] = False

    # Enhance routing with ToolRouter analysis if available
    if _tool_router is not None:
        try:
            analysis = _tool_router.analyze_shot(shot)
            shot["tool_routing"] = analysis
        except Exception as e:
            logger.warning("ToolRouter analysis failed for %s: %s", shot.get("shot_id"), e)

    return shot


# ─── Shot Directory Setup ────────────────────────────────────────────────────

def parse_captions(caption_str: str, caption_frames_str: str, fps: int = 30) -> list:
    """Parse caption string + frame range into list of caption dicts.

    Args:
        caption_str: Caption text, e.g. '雪狼来了/快躲呀' ('/' separates segments).
        caption_frames_str: Frame range, e.g. '30-210' or '30-210,220-390'.
        fps: Frames per second for time conversion (default 30).

    Returns:
        List of dicts: [{"text": str, "start": float, "end": float}, ...]
    """
    if not caption_str or not caption_frames_str:
        return []

    # Parse frame ranges: support comma-separated ranges like '30-210,220-390'
    ranges = []
    for part in caption_frames_str.split(","):
        part = part.strip()
        if "-" in part:
            start_frame, end_frame = part.split("-", 1)
            ranges.append((int(start_frame.strip()), int(end_frame.strip())))

    if not ranges:
        return []

    # Split caption text by '/' into segments
    text_segments = [t.strip() for t in caption_str.split("/") if t.strip()]

    captions = []

    if len(ranges) == len(text_segments):
        # One range per text segment — direct mapping
        for text, (start_f, end_f) in zip(text_segments, ranges):
            captions.append({
                "text": text,
                "start": round(start_f / fps, 3),
                "end": round(end_f / fps, 3),
            })
    elif len(ranges) == 1:
        # Single range for multiple text segments — distribute evenly
        start_f, end_f = ranges[0]
        total_duration = (end_f - start_f) / fps
        seg_duration = total_duration / len(text_segments)
        for i, text in enumerate(text_segments):
            seg_start = start_f / fps + i * seg_duration
            seg_end = seg_start + seg_duration
            captions.append({
                "text": text,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
            })
    else:
        # Mismatched counts — use first range, distribute text evenly
        start_f, end_f = ranges[0]
        total_duration = (end_f - start_f) / fps
        seg_duration = total_duration / len(text_segments)
        for i, text in enumerate(text_segments):
            seg_start = start_f / fps + i * seg_duration
            seg_end = seg_start + seg_duration
            captions.append({
                "text": text,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
            })

    return captions


def setup_shot_dirs(shots: list) -> list:
    """Create per-shot directories and write SHOT_META.json."""
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_paths = []

    for shot in shots:
        shot_dir = SHOTS_DIR / shot["shot_id"]
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "frames").mkdir(exist_ok=True)

        # Parse caption string + frames into structured caption list
        captions = parse_captions(shot["caption"], shot["caption_frames"])

        # Write SHOT_META.json
        meta = {
            "shot_id": shot["shot_id"],
            "name": shot["name"],
            "duration": shot["duration"],
            "prompt": shot["prompt"],
            "route": shot["route"],
            "route_reason": shot["route_reason"],
            "ref_type": shot.get("ref_type"),
            "first_frame_path": shot.get("first_frame_path"),
            "first_frame_exists": shot.get("first_frame_exists", False),
            "caption": shot["caption"],          # legacy: original string
            "caption_frames": shot["caption_frames"],  # legacy: original frame range
            "captions": captions,                # structured list for assembly_engine
            "who": shot.get("who", []),
            "associate_assets": shot.get("associate_assets", []),
            "gen_strategy": shot.get("gen_strategy", "i2v"),
            "visual": shot.get("visual", ""),
            "what": shot.get("what", ""),
            "action_description": shot.get("action_description", shot.get("what", "")),
            "generation_actions": shot.get("generation_actions", []),
            "generation_load": shot.get("generation_load"),
            "source_action_unit_ids": shot.get("source_action_unit_ids", []),
            "start_state": shot.get("start_state", ""),
            "end_state": shot.get("end_state", ""),
            "causal_link": shot.get("causal_link", ""),
            "shot_type": shot.get("shot_type"),
            "shot_size": shot.get("shot_size"),
            "camera_movement": shot.get("camera_movement"),
            "where": shot.get("where", ""),
            "emotion": shot.get("emotion", ""),
            "dialogue": shot.get("dialogue"),
            "speech_duration_s": shot.get("speech_duration_s", 0),
            "audio": shot.get("audio"),
            "aspect_ratio": shot.get("aspect_ratio"),
            "width": shot.get("width"),
            "height": shot.get("height"),
            "status": "pending",
            "task_id": None,
            "video_path": None,
        }
        meta_path = shot_dir / "SHOT_META.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        meta_paths.append(meta_path)
        print(f"  [setup] {shot['shot_id']}/ — {shot['name']} → {shot['route']}")

    return meta_paths


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_shot(shot: dict, api_key: str, dry_run: bool = False) -> dict:
    """Generate a single shot via Seedance. Returns updated shot dict."""
    shot_dir = SHOTS_DIR / shot["shot_id"]
    meta_path = shot_dir / "SHOT_META.json"

    # Load current meta
    with open(meta_path) as f:
        meta = json.load(f)

    if dry_run:
        print(f"  [DRY-RUN] Would generate {shot['shot_id']} via {shot['route']}")
        print(f"           Prompt: {shot['prompt'][:80]}...")
        if shot["route"] == "img2vid":
            print(f"           First frame: {shot.get('first_frame_path', 'N/A')}")
        meta["status"] = "dry_run"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta

    # Real generation
    from clients import seedance_client

    print(f"  [gen] {shot['shot_id']} — submitting via {shot['route']}...")
    meta["status"] = "generating"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    try:
        # Prepare first frame if img2vid
        first_frame_b64 = None
        if shot["route"] == "img2vid" and shot.get("first_frame_exists"):
            import base64
            with open(shot["first_frame_path"], "rb") as img_f:
                first_frame_b64 = base64.b64encode(img_f.read()).decode()

        # Submit
        task_id = seedance_client.submit(
            prompt=shot["prompt"],
            api_key=api_key,
            duration=shot["duration"],
            ratio="16:9",
            first_frame_base64=first_frame_b64,
        )
        meta["task_id"] = task_id
        print(f"  [gen] {shot['shot_id']} — task_id={task_id}, polling...")

        # Poll
        video_url = seedance_client.poll(task_id, api_key)

        # Download
        video_path = str(shot_dir / "output.mp4")
        seedance_client.download(video_url, video_path)
        meta["video_path"] = video_path
        meta["status"] = "done"

        # Extract frames
        extract_frames(video_path, shot_dir / "frames")

    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
        print(f"  [ERROR] {shot['shot_id']} — {e}")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def extract_frames(video_path: str, frames_dir: Path, fps: int = 1):
    """Extract frames from video using ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        frame_count = len(list(frames_dir.glob("frame_*.png")))
        print(f"  [frames] Extracted {frame_count} frames → {frames_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [frames] Warning: frame extraction failed: {e}")


# ─── Progress Summary ────────────────────────────────────────────────────────

def print_summary(shots: list):
    """Print generation progress summary."""
    print("\n" + "=" * 60)
    print("  ORCHESTRATION SUMMARY")
    print("=" * 60)

    status_counts = {}
    for shot in shots:
        shot_dir = SHOTS_DIR / shot["shot_id"]
        meta_path = shot_dir / "SHOT_META.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            status = meta.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            icon = {"done": "✅", "failed": "❌", "dry_run": "🔍", "pending": "⏳", "generating": "⚙️"}.get(status, "?")
            print(f"  {icon} {shot['shot_id']} ({shot['name']}) — {status} [{shot['route']}]")

    print(f"\n  Total: {len(shots)} shots")
    for status, count in sorted(status_counts.items()):
        print(f"    {status}: {count}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global SHOTS_DIR, STORYBOARD_PATH

    parser = argparse.ArgumentParser(description="Phase 4 Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Parse + route only, no API calls")
    parser.add_argument("--shots", type=str, help="Comma-separated shot IDs (e.g. S01,S03)")
    parser.add_argument("--storyboard", type=str, default=str(STORYBOARD_PATH), help="Path to storyboard JSON")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ARK_API_KEY"), help="ARK API key")
    parser.add_argument("--skip-assembly", action="store_true", help="Skip consistency guard and assembly (dry-run mode)")
    parser.add_argument("--shots-dir", type=str, default=None, help="Override shots output directory (default: PROJECT_ROOT/shots)")
    args = parser.parse_args()

    # Allow external override of SHOTS_DIR (e.g. when called by pipeline_runner.py)
    if args.shots_dir:
        SHOTS_DIR = Path(args.shots_dir).resolve()

    # Fix: update STORYBOARD_PATH so route_shot() resolves relative first_frame paths correctly
    STORYBOARD_PATH = Path(args.storyboard).resolve()

    print(f"🎬 Phase 4 Orchestrator — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"   Storyboard: {args.storyboard}")
    print(f"   Project: {PROJECT_ROOT}")
    print(f"   Shots dir: {SHOTS_DIR}")
    print()

    # 1. Load storyboard
    print("[1/4] Loading storyboard...")
    storyboard = load_storyboard(Path(args.storyboard))
    shots = parse_shots(storyboard)
    print(f"   Found {len(shots)} shots")

    # 2. Route shots
    print("\n[2/4] Routing shots...")
    static_refs = storyboard.get("static_reference_images", {})
    for i, shot in enumerate(shots):
        shots[i] = route_shot(shot, static_refs)
        print(f"   {shot['shot_id']} → {shot['route']} ({shot['route_reason'][:60]})")

    # 3. Setup directories
    print("\n[3/4] Setting up shot directories...")
    setup_shot_dirs(shots)

    # 4. Generate (or dry-run)
    print(f"\n[4/4] {'Simulating' if args.dry_run else 'Generating'} shots...")

    # Filter shots if --shots specified
    if args.shots:
        selected = [s.strip().upper() for s in args.shots.split(",")]
        shots = [s for s in shots if s["shot_id"] in selected]
        print(f"   Filtered to {len(shots)} shots: {[s['shot_id'] for s in shots]}")

    if not args.dry_run and not args.api_key:
        print("   ⚠️  No ARK_API_KEY set. Falling back to dry-run mode.")
        args.dry_run = True

    results = []
    for shot in shots:
        result = generate_shot(shot, args.api_key or "", dry_run=args.dry_run)
        results.append(result)

    # Summary
    print_summary(shots)

    if args.dry_run:
        print("\n✅ Dry run complete. No API calls made.")
        print(f"   Shot directories created in: {SHOTS_DIR}")
        print("   Review SHOT_META.json files for routing decisions.")
        return

    # Skip assembly if requested
    if args.skip_assembly:
        print("\n⏭️  --skip-assembly set. Skipping consistency guard and assembly.")
        return

    # 5. Check results — support partial assembly (P1-4 fix)
    failed_shots = [r for r in results if r.get("status") == "failed"]
    done_shots = [r for r in results if r.get("status") == "done"]

    if failed_shots:
        print(f"\n⚠️  {len(failed_shots)} shot(s) failed:")
        for fs in failed_shots:
            print(f"   ❌ {fs.get('shot_id', '?')} — {fs.get('error', 'unknown')}")
        if not done_shots:
            print("   No shots completed. Aborting.")
            return
        print(f"   Proceeding with {len(done_shots)} successful shot(s) (partial assembly).")

    if not done_shots:
        print("\n⚠️  No shots completed successfully. Skipping assembly.")
        return

    # 6. Run consistency guard + re-gen loop (P1-2 fix)
    print("\n[5/6] Running consistency guard...")
    guard_script = PROJECT_ROOT / "scripts" / "consistency_guard.py"
    report_path = SHOTS_DIR / "consistency_report.json"
    if guard_script.exists():
        guard_cmd = [sys.executable, str(guard_script)]
        if args.api_key:
            guard_cmd.extend(["--api-key", args.api_key])
        guard_result = subprocess.run(
            guard_cmd,
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if guard_result.stdout:
            print(guard_result.stdout)
        if guard_result.returncode != 0:
            print(f"⚠️  Consistency guard exited with code {guard_result.returncode}")
            if guard_result.stderr:
                print(f"   stderr: {guard_result.stderr[:500]}")

        # Parse report and re-generate 🔴 shots (P1-2: close the loop)
        # consistency_guard report format:
        #   report["cross_shot"]["S01_vs_S03"] = {"shots": ["S01","S03"], "color": "red", ...}
        #   report["intra_shot"]["S01"] = {"color": "red", ...}
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            regen_ids = set()
            # Collect red shots from cross_shot pairs
            for key, data in report.get("cross_shot", {}).items():
                if data.get("color") == "red":
                    for sid in data.get("shots", []):
                        regen_ids.add(sid)
            # Collect red shots from intra_shot
            for shot_id, data in report.get("intra_shot", {}).items():
                if data.get("color") == "red":
                    regen_ids.add(shot_id)
            if regen_ids:
                print(f"\n   🔴 {len(regen_ids)} inconsistent shot(s) detected. Re-generating...")
                for shot in shots:
                    if shot["shot_id"] in regen_ids and shot.get("route"):
                        print(f"   🔄 Re-generating {shot['shot_id']}...")
                        generate_shot(shot, args.api_key or "", dry_run=False)
            else:
                print("   ✅ All pairs within consistency threshold.")
    else:
        print(f"   ⚠️  consistency_guard.py not found at {guard_script}, skipping.")

    # 7. Run assembly engine
    print("\n[6/6] Assembling final video...")
    try:
        import assembly_engine
        shot_dir_path = str(SHOTS_DIR)
        output_final = str(OUTPUT_DIR / "final.mp4")
        assembly_engine.assemble(shot_dir_path, output_final)
        print(f"\n🎉 Assembly complete! Output: {output_final}")
    except ImportError:
        print("   ⚠️  assembly_engine.py not importable, trying subprocess fallback...")
        assembly_script = PROJECT_ROOT / "scripts" / "assembly_engine.py"
        if assembly_script.exists():
            output_final = str(OUTPUT_DIR / "final.mp4")
            subprocess.run(
                [sys.executable, str(assembly_script), str(SHOTS_DIR), output_final],
                cwd=str(PROJECT_ROOT)
            )
        else:
            print(f"   ❌ assembly_engine.py not found at {assembly_script}")
    except Exception as e:
        print(f"   ❌ Assembly failed: {e}")


if __name__ == "__main__":
    main()
