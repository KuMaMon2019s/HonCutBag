"""FFmpeg side-chain compression for narration-aware background music."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


class AudioDucking:
    """Duck a BGM signal whenever a narration side-chain becomes active."""

    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 10.0,
        release: float = 100.0,
    ) -> None:
        if not -80.0 <= threshold <= 0.0:
            raise ValueError("threshold must be between -80 and 0 dB")
        if not 1.0 <= ratio <= 20.0:
            raise ValueError("ratio must be between 1 and 20")
        if not 0.01 <= attack <= 2000.0:
            raise ValueError("attack must be between 0.01 and 2000 ms")
        if not 0.01 <= release <= 9000.0:
            raise ValueError("release must be between 0.01 and 9000 ms")
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release

    @property
    def threshold_amplitude(self) -> float:
        """Convert the user-facing dB threshold to FFmpeg's linear value."""
        return math.pow(10.0, self.threshold / 20.0)

    def build_filter(self) -> str:
        """Build a labelled FFmpeg graph that outputs narration plus ducked BGM."""
        return (
            "[0:a]aresample=48000,asetpts=PTS-STARTPTS[bgm];"
            "[1:a]aresample=48000,asetpts=PTS-STARTPTS[voice];"
            f"[bgm][voice]sidechaincompress=threshold={self.threshold_amplitude:.8f}:"
            f"ratio={self.ratio:g}:attack={self.attack:g}:release={self.release:g}"
            "[ducked];[ducked][voice]amix=inputs=2:duration=shortest:normalize=0[out]"
        )

    def apply(self, bgm_path: str, tts_path: str, output_path: str) -> str:
        """Apply side-chain compression and return the created audio path."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for audio ducking")
        for role, raw_path in (("BGM", bgm_path), ("TTS", tts_path)):
            if not Path(raw_path).is_file():
                raise FileNotFoundError(f"{role} audio not found: {raw_path}")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", bgm_path, "-i", tts_path,
            "-filter_complex", self.build_filter(), "-map", "[out]", "-shortest",
            "-c:a", "aac" if destination.suffix.lower() in {".m4a", ".mp4", ".aac"} else "libmp3lame",
            "-b:a", "192k", str(destination),
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"ffmpeg sidechain compression failed: {message}") from exc
        return str(destination)
