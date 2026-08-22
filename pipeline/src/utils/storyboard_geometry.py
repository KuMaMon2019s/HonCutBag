"""Shared storyboard canvas and provider image-size contracts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional


SEEDREAM_MIN_PIXELS = 3686400


def _storyboard_canvas(storyboard: dict) -> tuple[int, int, str]:
    """Resolve the project canvas from storyboard metadata without forcing 16:9."""
    first_shot = next(
        (shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)),
        {},
    )
    width = int(storyboard.get("width") or first_shot.get("width") or 0)
    height = int(storyboard.get("height") or first_shot.get("height") or 0)
    ratio = str(
        storyboard.get("aspect_ratio")
        or first_shot.get("aspect_ratio")
        or ""
    ).strip()
    if width > 0 and height > 0:
        divisor = math.gcd(width, height)
        return width, height, ratio or f"{width // divisor}:{height // divisor}"
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*", ratio)
    if match:
        left, right = float(match.group(1)), float(match.group(2))
        if left > 0 and right > 0:
            return round(left * 1000), round(right * 1000), f"{match.group(1)}:{match.group(2)}"
    return 16, 9, "16:9"


def _storyboard_image_size(image_path: Optional[Path] = None, video_width: int = 1280, video_height: int = 720) -> str:
    """Return Seedream's WxH size string for storyboard/end-frame generation.

    M7 fix: ensure returned size meets Seedream's minimum pixel requirement
    (SEEDREAM_MIN_PIXELS = 3686400). The old 1920x1080 (2,073,600 px) was
    rejected with HTTP 400 InvalidParameter.

    Formula: for aspect a=w/h, width = ceil(sqrt(min_pixels * a)), then
    round to even. Height = width / a, also rounded to even.

    For 16:9: 2560x1440 = 3,686,400 px (exactly at minimum).
    """
    import math

    aspect = video_width / video_height  # a = w/h

    # Compute smallest width >= sqrt(min_pixels * aspect) at correct aspect
    raw_w = math.sqrt(SEEDREAM_MIN_PIXELS * aspect)
    w = math.ceil(raw_w)
    # Round to even
    w = w if w % 2 == 0 else w + 1

    # Compute height from width and aspect, round to even
    h = math.ceil(w / aspect)
    h = h if h % 2 == 0 else h + 1

    # Safety check: if rounding pushed us below minimum, bump width
    if w * h < SEEDREAM_MIN_PIXELS:
        w += 2  # next even number
        h = math.ceil(w / aspect)
        h = h if h % 2 == 0 else h + 1

    return f"{w}x{h}"
