"""Shared aspect-ratio and raster resolution resolution for video providers."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_RATIO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1344, 576),
}


def _ratio_from_dimensions(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    if width * 9 == height * 16:
        return "16:9"
    if width * 16 == height * 9:
        return "9:16"
    if width * 3 == height * 4:
        return "4:3"
    if width * 4 == height * 3:
        return "3:4"
    return f"{width}:{height}"


def resolve_video_geometry(
    metadata: Mapping[str, Any] | None,
    *,
    default_ratio: str = "16:9",
) -> tuple[str, int, int]:
    """Return ``(ratio, width, height)`` from one authoritative metadata map."""

    data = metadata or {}
    raw_width = data.get("width") or data.get("video_width")
    raw_height = data.get("height") or data.get("video_height")
    try:
        width = int(raw_width) if raw_width else 0
        height = int(raw_height) if raw_height else 0
    except (TypeError, ValueError):
        width = height = 0

    ratio = str(data.get("aspect_ratio") or data.get("ratio") or "").strip()
    if width > 0 and height > 0:
        return ratio or _ratio_from_dimensions(width, height), width, height

    if ratio in _RATIO_DIMENSIONS:
        width, height = _RATIO_DIMENSIONS[ratio]
        return ratio, width, height

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*",
        ratio,
    )
    if match:
        left, right = float(match.group(1)), float(match.group(2))
        if left > 0 and right > 0:
            if left >= right:
                width = 1280
                height = max(2, round(1280 * right / left / 2) * 2)
            else:
                height = 1280
                width = max(2, round(1280 * left / right / 2) * 2)
            return f"{match.group(1)}:{match.group(2)}", width, height

    ratio = default_ratio
    width, height = _RATIO_DIMENSIONS.get(ratio, _RATIO_DIMENSIONS["16:9"])
    return ratio, width, height


__all__ = ["resolve_video_geometry"]
