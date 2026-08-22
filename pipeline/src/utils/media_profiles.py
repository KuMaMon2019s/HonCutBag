"""Internal media profile constants and render-profile helpers.

Defines platform-specific media profiles (resolution, aspect ratio, codec, etc.)
so the composer and publisher agents can format output correctly.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


DEFAULT_MEDIA_PROFILE = "480p"


class AspectRatio(str, Enum):
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    CINEMATIC_21_9 = "21:9"
    STANDARD_4_3 = "4:3"


@dataclass(frozen=True)
class MediaProfile:
    """A named render profile for a target platform/format."""
    name: str
    width: int
    height: int
    aspect_ratio: AspectRatio
    fps: int
    codec: str
    audio_codec: str
    crf: int
    pixel_format: str = "yuv420p"
    max_file_size_mb: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    caption_format: str = "srt"
    notes: str = ""


# ---- Platform profiles ----

YOUTUBE_LANDSCAPE = MediaProfile(
    name="youtube_landscape",
    width=1920, height=1080,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=30, codec="libx264", audio_codec="aac", crf=18,
    caption_format="srt",
    notes="YouTube standard HD upload",
)

YOUTUBE_4K = MediaProfile(
    name="youtube_4k",
    width=3840, height=2160,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=30, codec="libx264", audio_codec="aac", crf=18,
    caption_format="srt",
    notes="YouTube 4K upload",
)

YOUTUBE_SHORTS = MediaProfile(
    name="youtube_shorts",
    width=1080, height=1920,
    aspect_ratio=AspectRatio.PORTRAIT_9_16,
    fps=30, codec="libx264", audio_codec="aac", crf=20,
    max_duration_seconds=60,
    caption_format="srt",
    notes="YouTube Shorts (max 60s, vertical)",
)

INSTAGRAM_REELS = MediaProfile(
    name="instagram_reels",
    width=1080, height=1920,
    aspect_ratio=AspectRatio.PORTRAIT_9_16,
    fps=30, codec="libx264", audio_codec="aac", crf=20,
    max_file_size_mb=250,
    max_duration_seconds=90,
    caption_format="srt",
    notes="Instagram Reels (max 90s, vertical)",
)

INSTAGRAM_FEED = MediaProfile(
    name="instagram_feed",
    width=1080, height=1080,
    aspect_ratio=AspectRatio.SQUARE_1_1,
    fps=30, codec="libx264", audio_codec="aac", crf=20,
    max_file_size_mb=250,
    max_duration_seconds=60,
    notes="Instagram feed video (square)",
)

TIKTOK = MediaProfile(
    name="tiktok",
    width=1080, height=1920,
    aspect_ratio=AspectRatio.PORTRAIT_9_16,
    fps=30, codec="libx264", audio_codec="aac", crf=20,
    max_file_size_mb=287,
    max_duration_seconds=600,
    caption_format="srt",
    notes="TikTok (max 10min, vertical preferred)",
)

LINKEDIN = MediaProfile(
    name="linkedin",
    width=1920, height=1080,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=30, codec="libx264", audio_codec="aac", crf=20,
    max_file_size_mb=5120,
    max_duration_seconds=600,
    caption_format="srt",
    notes="LinkedIn video (landscape preferred, max 10min)",
)

CINEMATIC = MediaProfile(
    name="cinematic",
    width=2560, height=1080,
    aspect_ratio=AspectRatio.CINEMATIC_21_9,
    fps=24, codec="libx264", audio_codec="aac", crf=16,
    notes="Cinematic ultra-wide format",
)

GENERIC_HD = MediaProfile(
    name="generic_hd",
    width=1920, height=1080,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=30, codec="libx264", audio_codec="aac", crf=23,
    caption_format="srt",
    notes="Generic HD output (no platform-specific constraints)",
)


# ---- Profile registry ----

ALL_PROFILES: dict[str, MediaProfile] = {
    p.name: p for p in [
        YOUTUBE_LANDSCAPE, YOUTUBE_4K, YOUTUBE_SHORTS,
        INSTAGRAM_REELS, INSTAGRAM_FEED,
        TIKTOK, LINKEDIN, CINEMATIC, GENERIC_HD,
    ]
}


def get_profile(name: str) -> MediaProfile:
    """Get a media profile by name."""
    if name not in ALL_PROFILES:
        available = ", ".join(ALL_PROFILES.keys())
        raise ValueError(f"Unknown profile {name!r}. Available: {available}")
    return ALL_PROFILES[name]


def get_profiles_for_platform(platform: str) -> list[MediaProfile]:
    """Get all profiles matching a platform prefix."""
    return [p for name, p in ALL_PROFILES.items() if name.startswith(platform)]


def ffmpeg_output_args(profile: MediaProfile) -> list[str]:
    """Generate FFmpeg output arguments for a media profile."""
    args = [
        "-c:v", profile.codec,
        "-c:a", profile.audio_codec,
        "-crf", str(profile.crf),
        "-pix_fmt", profile.pixel_format,
        "-r", str(profile.fps),
        "-vf", f"scale={profile.width}:{profile.height}",
    ]
    return args


_PROFILE_NAME_MAP = {
    "480p": None,
    "720p": None,
    "1080p": "generic_hd",
    "youtube": "youtube_landscape",
    "youtube_shorts": "youtube_shorts",
    "tiktok": "tiktok",
    "instagram_reels": "instagram_reels",
    "cinematic": "cinematic",
}

_LEGACY_FALLBACKS = {
    "480p": {
        "name": "480p",
        "width": 854,
        "height": 480,
        "fps": 30,
        "codec": "libx264",
        "audio_codec": "aac",
        "crf": 23,
        "pixel_format": "yuv420p",
        "aspect_ratio": "16:9",
    },
    "720p": {
        "name": "720p",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "codec": "libx264",
        "audio_codec": "aac",
        "crf": 23,
        "pixel_format": "yuv420p",
        "aspect_ratio": "16:9",
    },
}

AVAILABLE_PROFILES = sorted(
    {
        "480p",
        "720p",
        "1080p",
        "youtube",
        "youtube_shorts",
        "tiktok",
        "instagram_reels",
        "cinematic",
        *ALL_PROFILES,
    }
)


def get_profile_dict(profile_name: str = "1080p") -> dict[str, Any]:
    """Resolve a media profile into the legacy dictionary contract."""

    if profile_name in _LEGACY_FALLBACKS:
        return dict(_LEGACY_FALLBACKS[profile_name])
    resolved_name = _PROFILE_NAME_MAP.get(profile_name, profile_name)
    if resolved_name is not None:
        try:
            return dataclasses.asdict(get_profile(resolved_name))
        except ValueError:
            pass
    try:
        return dataclasses.asdict(get_profile(profile_name))
    except ValueError:
        return dataclasses.asdict(GENERIC_HD)


def project_video_spec(media_profile: str) -> dict[str, Any]:
    """Resolve the geometry and delivery contract shared by every phase."""

    profile = get_profile_dict(media_profile)
    width = int(profile["width"])
    height = int(profile["height"])
    semantic_ratio = profile.get("aspect_ratio")
    if hasattr(semantic_ratio, "value"):
        semantic_ratio = semantic_ratio.value
    if not semantic_ratio:
        divisor = math.gcd(width, height)
        semantic_ratio = f"{width // divisor}:{height // divisor}"
    return {
        "aspect_ratio": str(semantic_ratio),
        "width": width,
        "height": height,
        "fps": float(profile.get("fps") or 30),
        "delivery_profile": media_profile,
    }


_get_profile_dict = get_profile_dict
_project_video_spec = project_video_spec
