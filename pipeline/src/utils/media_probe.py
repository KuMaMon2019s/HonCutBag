"""FFprobe-backed audio/video duration contracts."""

import json
import subprocess
from pathlib import Path


def probe_av_durations(path: Path) -> dict[str, float | None]:
    """Return independent stream durations; probing failures block delivery."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    durations: dict[str, float | None] = {"video": None, "audio": None}
    for stream in streams:
        kind = stream.get("codec_type")
        if (
            kind in durations
            and durations[kind] is None
            and stream.get("duration") not in (None, "N/A")
        ):
            durations[kind] = float(stream["duration"])
    if durations["video"] is None:
        raise RuntimeError(f"No measurable video stream duration: {path}")
    return durations


def assert_duration_conserved(
    before: dict[str, float | None],
    after: dict[str, float | None],
    tolerance_s: float = 1.0,
    *,
    audio_tolerance_s: float | None = None,
) -> None:
    """Assert final encoding conserved video and audio durations independently."""
    comparison_epsilon_s = 1e-6
    for kind in ("video", "audio"):
        expected, actual = before.get(kind), after.get(kind)
        tolerance = (
            audio_tolerance_s
            if kind == "audio" and audio_tolerance_s is not None
            else tolerance_s
        )
        if expected is None:
            continue
        if actual is None or abs(actual - expected) > tolerance + comparison_epsilon_s:
            raise RuntimeError(
                f"Final {kind} duration changed from {expected:.3f}s to "
                f"{actual if actual is not None else 'missing'} "
                f"(tolerance ±{tolerance:.3f}s)"
            )


_assert_duration_conserved = assert_duration_conserved
_probe_av_durations = probe_av_durations


__all__ = [
    "_assert_duration_conserved",
    "_probe_av_durations",
    "assert_duration_conserved",
    "probe_av_durations",
]
