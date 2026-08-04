#!/usr/bin/env python3
"""
audio_pipeline.py — Enhanced audio post-processing with OM AudioMixer capabilities

Integrates FFmpeg-based audio processing inspired by OpenMontage AudioMixer:
- Loudness normalization (loudnorm filter, target -14 LUFS)
- Audio fade in/out (afade filter)
- Background music ducking (sidechaincompress)
- Audio track mixing (amix)

Usage:
    from audio_pipeline import process_audio
    process_audio(video_path, storyboard_path, output_path, bgm_path=None)
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Optional


# ─── Silent Audio Detection ───────────────────────────────────────────────────

def is_silent_audio(video_path: str, threshold_db: float = -60.0) -> bool:
    """Detect whether the audio track of a video is effectively silent.

    Uses ffmpeg volumedetect filter.  mean_volume < threshold_db (default -60 dB)
    is considered silent — this covers both truly empty tracks and the anullsrc
    silence injected by edit_decisions normalisation.

    Returns True when the track is silent (or when no audio stream exists, or
    when detection fails — callers should treat failure as "assume silent" so
    the ambient fallback always produces audible output).
    """
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-af", "volumedetect",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # volumedetect writes to stderr
        match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", result.stderr)
        if match:
            mean_vol = float(match.group(1))
            return mean_vol < threshold_db
        # No mean_volume line → probably no audio stream → silent
        return True
    except Exception:
        # Detection failure → assume silent so ambient fallback kicks in
        return True


# ─── Ambient Audio Generation (local fallback) ────────────────────────────────

def generate_ambient_audio(
    duration: float,
    output_path: str,
    scene_hint: str = "lake_evening",
    target_db: float = -20.0,
) -> bool:
    """Generate lightweight ambient audio using ffmpeg filters only.

    Produces pink-noise-based ambience filtered to match a scene mood.
    No external model or API required.

    Scene presets:
      lake_evening  — low-pass filtered pink noise (warm lake breeze)
      forest        — band-pass pink noise (rustling leaves)
      city          — wider band-pass with slight rumble
      generic       — gentle pink noise (safe default)

    Args:
        duration:    target duration in seconds
        output_path: where to write the generated audio (mp4/m4a)
        scene_hint:  one of the presets above
        target_db:   target volume in dB (default -20 dB, moderate)

    Returns:
        True on success.
    """
    if duration <= 0:
        return False

    # Preset filter chains (applied to anullsrc white noise → shaped to pink-ish)
    presets = {
        "lake_evening": (
            "highpass=f=40,"
            "lowpass=f=800,"
            "equalizer=f=200:t=q:w=1.5:g=4,"
            "equalizer=f=500:t=q:w=2:g=-3,"
            "volume={vol_db}dB"
        ),
        "forest": (
            "highpass=f=200,"
            "lowpass=f=4000,"
            "equalizer=f=1000:t=q:w=1:g=3,"
            "equalizer=f=3000:t=q:w=1.5:g=-2,"
            "volume={vol_db}dB"
        ),
        "city": (
            "highpass=f=60,"
            "lowpass=f=3000,"
            "equalizer=f=120:t=q:w=2:g=5,"
            "volume={vol_db}dB"
        ),
        "generic": (
            "highpass=f=80,"
            "lowpass=f=2000,"
            "volume={vol_db}dB"
        ),
    }

    af = presets.get(scene_hint, presets["generic"]).format(vol_db=target_db)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-t", str(duration),
        "-i", "anoisesrc=color=pink:amplitude=1.0:sample_rate=48000",
        "-af", af,
        "-c:a", "aac", "-b:a", "128k",
        "-ar", "48000", "-ac", "2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ Ambient generation failed: {e.stderr[:200] if e.stderr else e}")
        return False


def get_audio_duration(file_path: str) -> float:
    """Get audio/video duration using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip().split("\n")[0])
    except Exception as e:
        print(f"  ⚠ Failed to get duration: {e}")
        return 0.0


def apply_loudnorm(input_path: str, output_path: str, target_lufs: float = -14.0) -> bool:
    """
    Apply loudness normalization using FFmpeg loudnorm filter.
    Target: -14 LUFS (YouTube/TikTok/Instagram standard)
    
    Args:
        input_path: Input audio/video file
        output_path: Output file path
        target_lufs: Target integrated loudness in LUFS (default: -14)
    
    Returns:
        True if successful, False otherwise
    """
    # Clamp to sane range
    target_lufs = max(-40.0, min(0.0, target_lufs))
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5",
        "-c:v", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ loudnorm failed: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def apply_fades(input_path: str, output_path: str, 
                fade_in_sec: float = 1.0, fade_out_sec: float = 2.0) -> bool:
    """
    Apply fade in at start and fade out at end.
    
    Args:
        input_path: Input audio/video file
        output_path: Output file path
        fade_in_sec: Fade in duration in seconds (default: 1.0)
        fade_out_sec: Fade out duration in seconds (default: 2.0)
    
    Returns:
        True if successful, False otherwise
    """
    duration = get_audio_duration(input_path)
    if duration <= 0:
        print(f"  ⚠ Cannot apply fades: invalid duration {duration}")
        return False
    
    fade_out_start = max(0.0, duration - fade_out_sec)
    
    # Build filter chain
    filters = []
    if fade_in_sec > 0:
        filters.append(f"afade=t=in:d={fade_in_sec}")
    if fade_out_sec > 0:
        filters.append(f"afade=t=out:st={fade_out_start}:d={fade_out_sec}")
    
    if not filters:
        # No fades needed, just copy
        cmd = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]
    else:
        filter_chain = ",".join(filters)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", filter_chain,
            "-c:v", "copy",
            output_path
        ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ fade failed: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def apply_bgm_ducking(video_path: str, bgm_path: str, output_path: str,
                     music_volume: float = 0.15, 
                     attack_ms: float = 200, release_ms: float = 500) -> bool:
    """
    Apply background music ducking using sidechaincompress.
    Lowers music volume when speech/narration is present.
    
    Args:
        video_path: Input video with speech/narration audio
        bgm_path: Background music file
        output_path: Output file path
        music_volume: Music volume level during speech (0.0-1.0, default: 0.15)
        attack_ms: Ducking attack time in ms (default: 200)
        release_ms: Ducking release time in ms (default: 500)
    
    Returns:
        True if successful, False otherwise
    """
    attack = attack_ms / 1000.0
    release = release_ms / 1000.0
    
    # Sidechain compress: video audio is the key signal to duck music
    filter_complex = (
        f"[1:a]sidechaincompress="
        f"threshold=0.02:ratio=9:attack={attack}:release={release}:"
        f"level_sc=1:mix=0.9[ducked];"
        f"[ducked]volume={music_volume * 3}[music_out];"
        f"[0:a][music_out]amix=inputs=2:duration=first[out]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1",  # Loop music if shorter than video
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ BGM ducking failed: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def process_audio(video_path: str, 
                 storyboard_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 bgm_path: Optional[str] = None,
                 target_lufs: float = -14.0,
                 fade_in: float = 1.0,
                 fade_out: float = 2.0) -> bool:
    """
    Main audio processing pipeline with OM AudioMixer capabilities.
    
    Processing steps:
    1. BGM ducking (if bgm_path provided)
    2. Loudness normalization (loudnorm, target -14 LUFS)
    3. Fade in/out (1s in, 2s out by default)
    
    Args:
        video_path: Input video file
        storyboard_path: Storyboard JSON path (unused, for API compatibility)
        output_path: Output file path (default: video_path with _audio suffix)
        bgm_path: Optional background music file for ducking
        target_lufs: Target loudness in LUFS (default: -14)
        fade_in: Fade in duration in seconds (default: 1.0)
        fade_out: Fade out duration in seconds (default: 2.0)
    
    Returns:
        True if successful, False otherwise
    """
    if not Path(video_path).exists():
        print(f"  ✗ Input video not found: {video_path}")
        return False
    
    if output_path is None:
        output_path = str(Path(video_path).with_stem(Path(video_path).stem + "_audio"))
    
    print(f"  → Audio pipeline: {Path(video_path).name}")
    
    current_input = video_path
    temp_files = []
    
    # Step 1: BGM ducking (if background music provided)
    if bgm_path and Path(bgm_path).exists():
        print(f"    → Applying BGM ducking...")
        ducked_output = str(Path(output_path).with_stem(Path(output_path).stem + "_ducked"))
        if apply_bgm_ducking(current_input, bgm_path, ducked_output):
            current_input = ducked_output
            temp_files.append(ducked_output)
            print(f"      ✓ BGM ducking applied")
        else:
            print(f"      ⚠ BGM ducking failed, continuing without it")
    elif bgm_path:
        print(f"    ⚠ BGM file not found: {bgm_path}")
    
    # Step 2: Loudness normalization
    print(f"    → Applying loudness normalization (target: {target_lufs} LUFS)...")
    normalized_output = str(Path(output_path).with_stem(Path(output_path).stem + "_normalized"))
    if apply_loudnorm(current_input, normalized_output, target_lufs):
        current_input = normalized_output
        temp_files.append(normalized_output)
        print(f"      ✓ Loudness normalized")
    else:
        print(f"      ⚠ Loudness normalization failed, using previous output")
    
    # Step 3: Fade in/out
    print(f"    → Applying fades (in: {fade_in}s, out: {fade_out}s)...")
    if apply_fades(current_input, output_path, fade_in, fade_out):
        print(f"      ✓ Fades applied")
    else:
        print(f"      ⚠ Fade failed, copying previous output")
        # Fallback: just copy the current input to output
        cmd = ["ffmpeg", "-y", "-i", current_input, "-c", "copy", output_path]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Final copy failed: {e}")
            return False
    
    # Cleanup temp files
    for temp_file in temp_files:
        try:
            Path(temp_file).unlink()
        except Exception:
            pass
    
    print(f"  ✓ Audio processing complete: {Path(output_path).name}")
    return True


if __name__ == "__main__":
    # Test with dry-run
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_pipeline.py <video_path> [bgm_path] [output_path]")
        sys.exit(1)
    
    video = sys.argv[1]
    bgm = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = process_audio(video, bgm_path=bgm, output_path=output)
    sys.exit(0 if success else 1)
