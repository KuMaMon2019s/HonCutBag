"""Phase 7 audio-material orchestration and multi-track mixing."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tools.audio.ducking import AudioDucking
from tools.audio.music_library import MusicLibrary, MusicTrack
from tools.audio.tts_narration import TTSNarration
from utils.config import AUDIO_CONFIG


class AudioMixer:
    """Select music, synthesize narration, duck it, and attach it to video."""

    def __init__(
        self,
        music_library: MusicLibrary | None = None,
        narrator: TTSNarration | None = None,
        ducking: AudioDucking | None = None,
    ) -> None:
        self.music_library = music_library or MusicLibrary(AUDIO_CONFIG["music_dir"])
        self.narrator = narrator
        self.ducking = ducking or AudioDucking(
            threshold=AUDIO_CONFIG["ducking_threshold"],
            ratio=AUDIO_CONFIG["ducking_ratio"],
        )

    def select_bgm(
        self, mood: str | None, duration: float | None = None
    ) -> MusicTrack | None:
        """Choose the closest-duration local track for a supported mood."""
        duration_range = None if duration is None else (max(0.0, duration * 0.5), None)
        tracks = self.music_library.search(mood=mood, duration_range=duration_range)
        if not tracks and mood is not None:
            tracks = self.music_library.search(duration_range=duration_range)
        if not tracks:
            return None
        if duration is None:
            return tracks[0]
        return min(tracks, key=lambda track: abs(track.duration - duration))

    def generate_narration(
        self,
        segments: Iterable[dict[str, Any]],
        output_dir: str | Path,
        *,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> list[str]:
        """Synthesize each non-empty narration segment as a separate track."""
        narrator = self.narrator or TTSNarration(api_key=AUDIO_CONFIG["tts_api_key"])
        directory = Path(output_dir)
        outputs: list[str] = []
        for index, segment in enumerate(segments, 1):
            raw_text = (
                segment.get("narration")
                or segment.get("voiceover")
                or segment.get("dialogue")
            )
            text = self._spoken_text(raw_text)
            if text:
                outputs.append(narrator.generate(
                    text, str(directory / f"narration_{index:03d}.mp3"), speed, pitch
                ))
        return outputs

    @classmethod
    def _spoken_text(cls, value: Any) -> str:
        """Return speakable text from supported storyboard dialogue shapes."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return cls._spoken_text(value.get("line") or value.get("text"))
        if isinstance(value, list):
            return "\n".join(filter(None, (cls._spoken_text(item) for item in value)))
        return ""

    @staticmethod
    def mix_tracks(tracks: list[str], output_path: str, volumes: list[float] | None = None) -> str:
        """Mix one or more audio tracks without normalizing away volume choices."""
        if not tracks:
            raise ValueError("At least one audio track is required")
        for track in tracks:
            if not Path(track).is_file():
                raise FileNotFoundError(f"Audio track not found: {track}")
        volumes = volumes or [1.0] * len(tracks)
        if len(volumes) != len(tracks) or any(volume < 0 for volume in volumes):
            raise ValueError("volumes must contain one non-negative value per track")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        inputs = [item for track in tracks for item in ("-i", track)]
        filters = ";".join(
            f"[{index}:a]aresample=48000,volume={volume:g}[a{index}]"
            for index, volume in enumerate(volumes)
        )
        labels = "".join(f"[a{index}]" for index in range(len(tracks)))
        filters += f";{labels}amix=inputs={len(tracks)}:duration=longest:normalize=0[out]"
        command = [
            "ffmpeg", "-y", *inputs, "-filter_complex", filters, "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k", str(destination),
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"FFmpeg multi-track mix failed: {error}") from exc
        return str(destination)

    @staticmethod
    def attach_to_video(video_path: str, audio_path: str, output_path: str) -> str:
        """Mux the mixed audio into the original video stream."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(destination),
        ]
        subprocess.run(command, capture_output=True, check=True, timeout=600)
        return str(destination)

    def process(
        self,
        video_path: str,
        storyboard: dict[str, Any],
        output_path: str,
        *,
        sound_effects: list[str] | None = None,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Run the full Phase 7 audio-material workflow."""
        work_dir = Path(output_path).parent / "audio_layer"
        work_dir.mkdir(parents=True, exist_ok=True)
        audio_options = storyboard.get("audio", {})
        mood = audio_options.get("mood") or storyboard.get("metadata", {}).get("mood")
        bgm_path = audio_options.get("bgm_path")
        selected = None
        if not bgm_path:
            selected = self.select_bgm(mood, duration)
            bgm_path = selected.path if selected else None

        narration = self.generate_narration(
            storyboard.get("shots", []), work_dir,
            speed=float(audio_options.get("speed", 1.0)),
            pitch=float(audio_options.get("pitch", 1.0)),
        ) if audio_options.get("tts", True) else []

        effects = sound_effects or audio_options.get("sound_effects", [])
        voice_track = None
        if narration:
            voice_track = self.mix_tracks(narration, str(work_dir / "narration.m4a"))
        base_track = None
        if bgm_path and voice_track:
            base_track = self.ducking.apply(bgm_path, voice_track, str(work_dir / "ducked.m4a"))
        elif bgm_path or voice_track:
            base_track = bgm_path or voice_track
        all_tracks = ([base_track] if base_track else []) + list(effects)
        if not all_tracks:
            shutil.copy2(video_path, output_path)
            return {"output": output_path, "bgm": None, "narration": [], "effects": []}
        mixed = self.mix_tracks(all_tracks, str(work_dir / "final_mix.m4a"))
        self.attach_to_video(video_path, mixed, output_path)
        return {
            "output": output_path, "bgm": bgm_path, "bgm_track_id": selected.id if selected else None,
            "narration": narration, "effects": list(effects),
        }


def apply_phase7_audio(output_dir: str | Path) -> dict[str, Any] | None:
    """Apply configured audio to ``raw_assembly.mp4`` and replace it safely."""
    directory = Path(output_dir)
    storyboard_path = directory / "STORYBOARD.json"
    raw_video = directory / "raw_assembly.mp4"
    if not storyboard_path.is_file() or not raw_video.is_file():
        return None
    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    if not storyboard.get("audio", {}).get("enabled", False):
        return None
    temporary = directory / "raw_assembly_audio.mp4"
    receipt = AudioMixer().process(str(raw_video), storyboard, str(temporary))
    temporary.replace(raw_video)
    receipt["output"] = str(raw_video)
    return receipt
