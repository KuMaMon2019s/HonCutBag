"""Background-score discovery, continuity, and audio-mix request construction."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

from phases.phase9.captions import clean_subtitle_text


def _detect_bgm(output_dir: Path, storyboard_path: Optional[Path] = None) -> Optional[str]:
    """
    Detect background music file for Phase 9 audio processing.

    Search order:
    1. BGM referenced in STORYBOARD.json (metadata.bgm_path)
    2. Common BGM filenames in output_dir (bgm.mp3, bg_music.mp3, etc.)
    3. Any .mp3/.wav/.aac file in output_dir/audio/ subdirectory

    Returns:
        Path to BGM file as string, or None if not found.
    """
    # 1. Check storyboard metadata
    if storyboard_path and storyboard_path.exists():
        try:
            sb_data = json.loads(storyboard_path.read_text())
            bgm = sb_data.get("metadata", {}).get("bgm_path")
            if bgm and Path(bgm).exists():
                return str(bgm)
            # Also check top-level bgm field
            bgm = sb_data.get("bgm_path") or sb_data.get("bgm")
            if bgm and Path(bgm).exists():
                return str(bgm)
        except (json.JSONDecodeError, KeyError):
            pass

    # 2. Check common BGM filenames in output_dir
    common_bgm_names = ["bgm.mp3", "bgm.wav", "bg_music.mp3", "background_music.mp3",
                        "music.mp3", "soundtrack.mp3", "ost.mp3"]
    for name in common_bgm_names:
        candidate = output_dir / name
        if candidate.exists():
            return str(candidate)

    # 3. Check audio subdirectory
    audio_dir = output_dir / "audio"
    if audio_dir.exists():
        for ext in ("*.mp3", "*.wav", "*.aac", "*.m4a"):
            matches = list(audio_dir.glob(ext))
            if matches:
                return str(matches[0])

    return None


def _prepare_continuous_bgm(
    bgm_path: str,
    target_duration: float,
    output_path: Path,
    *,
    crossfade_s: float = 2.0,
) -> str:
    """Extend one score across the film with equal-power loop crossfades."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", bgm_path],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    source_duration = float(probe.stdout.strip().splitlines()[0])
    if source_duration <= 0 or target_duration <= 0:
        raise ValueError("BGM and target durations must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fade = min(crossfade_s, max(0.1, source_duration / 4))
    count = max(1, math.ceil((target_duration - fade) / max(0.1, source_duration - fade)))
    inputs = [item for _ in range(count) for item in ("-i", bgm_path)]
    filters = [f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}]" for index in range(count)]
    current = "a0"
    for index in range(1, count):
        output = f"mix{index}"
        filters.append(f"[{current}][a{index}]acrossfade=d={fade}:c1=qsin:c2=qsin[{output}]")
        current = output
    filters.append(
        f"[{current}]atrim=duration={target_duration},"
        f"afade=t=in:d=1,afade=t=out:st={max(0.0, target_duration - 2.0)}:d=2[out]"
    )
    command = [
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", "[out]", "-c:a", "aac", "-b:a", "192k", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=600)
    if completed.returncode != 0 or not output_path.is_file():
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"Failed to prepare continuous BGM: {detail}")
    return str(output_path)


def _phase9_real_audio_tracks(
    output_dir: Path,
    storyboard_data: Optional[dict],
    transcript_data: Optional[dict],
    raw_video: Path,
    bgm_path: Optional[str] = None,
) -> tuple[list[dict], int]:
    """Build the real-audio base and narration tracks for Phase 9."""
    tracks = [{"path": str(raw_video), "role": "music"}]
    if bgm_path:
        tracks.append({"path": bgm_path, "role": "music", "volume": 0.18})
    audio_options = storyboard_data.get("audio", {}) if storyboard_data else {}
    if not storyboard_data or not audio_options.get("enabled", False) or not audio_options.get("tts", True):
        return tracks, 0

    from phases.phase9.audio_mixer import AudioMixer as Phase7AudioMixer

    transcript_shots = (transcript_data or {}).get("shots", [])
    skipped = 0
    for index, shot in enumerate(storyboard_data.get("shots", []), 1):
        narration_path = output_dir / "audio_layer" / f"narration_{index:03d}.mp3"
        spoken_text = Phase7AudioMixer._spoken_text(
            shot.get("narration") or shot.get("voiceover") or shot.get("dialogue")
        )
        if not spoken_text or not narration_path.is_file():
            continue

        transcript_shot = transcript_shots[index - 1] if index <= len(transcript_shots) else {}
        # Only ASR text proves the line exists in source audio. Script fallback
        # text is not evidence and must never suppress an overlay.
        asr_text = transcript_shot.get("text", "") if transcript_shot.get("source") == "asr" else ""
        normalized_line = "".join(clean_subtitle_text(spoken_text).casefold().split())
        normalized_asr = "".join(clean_subtitle_text(asr_text).casefold().split())
        if normalized_line and normalized_asr and normalized_line in normalized_asr:
            skipped += 1
            shot_id = shot.get("shot_id") or f"S{index:02d}"
            print(f"    ⊘ [P0-D3] {shot_id}: TTS skipped (dialogue already in source audio)")
            continue

        tracks.append({
            "path": str(narration_path),
            "role": "speech",
            "start_seconds": float(transcript_shot.get("start_ms", 0)) / 1000,
        })
    return tracks, skipped


def _phase9_real_audio_mix_request(tracks: list[dict], audio_out: Path) -> dict:
    """Return the AudioMixer request for a preserved real-audio base track."""
    has_tts = any(track.get("role") == "speech" for track in tracks)
    return {
        "operation": "full_mix" if has_tts else "mix",
        "tracks": tracks,
        "ducking": {
            "enabled": True,
            "music_volume_during_speech": 0.15,
        } if has_tts else None,
        "normalize": True,
        "loudnorm_target": -14,
        "output_path": str(audio_out),
    }
