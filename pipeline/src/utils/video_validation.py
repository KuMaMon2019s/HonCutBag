"""Strict technical validation for reusable generated video artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def is_valid_video(path: str | Path) -> bool:
    """Return true only when ffprobe finds a non-empty, decodable video stream."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        return False
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of",
                "json",
                str(candidate),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return False
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            return False
        streams = payload.get("streams") or []
        duration = float((payload.get("format") or {}).get("duration") or 0)
        return bool(
            duration > 0
            and any(
                stream.get("codec_type") == "video"
                and int(stream.get("width") or 0) > 0
                and int(stream.get("height") or 0) > 0
                for stream in streams
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return False
