"""Provider/model-owned limits used by director planning and generation QA.

These values describe a video model, not HonCut's storytelling policy.  Callers
must resolve a profile from the requested model/provider instead of embedding a
Seedance probe result in a global director rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

MAX_CONTENT_BEATS_PER_PRIMARY_SHOT = 2


@dataclass(frozen=True)
class VideoModelCapabilities:
    """Executable limits for one video-model family."""

    name: str
    min_shot_duration_s: float
    max_shot_duration_s: float
    min_unique_beat_s: float
    max_unique_beat_s: float
    max_action_units_per_beat: int
    max_micro_actions_per_beat: int
    action_budget_steps: tuple[tuple[float, int], ...]
    duration_quantum_s: float = 0.001
    tail_reference_window_s: float | None = None
    tail_reference_frame_fractions: tuple[float, ...] = ()
    max_reference_images: int | None = None
    continuity_anchor_frame_count: int = 0

    def action_limit(self, duration_seconds: float | int | None) -> int:
        if duration_seconds is None:
            return max(limit for _duration, limit in self.action_budget_steps)
        duration = max(0.0, float(duration_seconds))
        for upper_bound, limit in self.action_budget_steps:
            if duration <= upper_bound:
                return limit
        return self.action_budget_steps[-1][1]


# Values below are the conservative results of the project's Seedance 2.x
# probes.  Keeping them in a named profile makes the provenance explicit and
# prevents them from silently governing Kling, Wan, or a future provider.
SEEDANCE_2_CAPABILITIES = VideoModelCapabilities(
    name="seedance-2.x",
    min_shot_duration_s=4,
    max_shot_duration_s=15,
    min_unique_beat_s=3,
    max_unique_beat_s=7,
    max_action_units_per_beat=1,
    max_micro_actions_per_beat=2,
    action_budget_steps=((5, 1), (6, 2), (8, 3), (15, 4)),
    duration_quantum_s=1,
    tail_reference_window_s=2,
    tail_reference_frame_fractions=(0.2, 0.6, 0.95),
    max_reference_images=9,
    continuity_anchor_frame_count=3,
)


# Unknown providers use a neutral, permissive planning profile.  Production
# integrations should add an explicit profile once their capabilities are
# measured rather than inheriting Seedance's choreography limits by accident.
GENERIC_VIDEO_CAPABILITIES = VideoModelCapabilities(
    name="generic-video",
    min_shot_duration_s=1,
    max_shot_duration_s=60,
    min_unique_beat_s=1,
    max_unique_beat_s=15,
    max_action_units_per_beat=2,
    max_micro_actions_per_beat=4,
    action_budget_steps=((5, 2), (10, 4), (60, 6)),
)


def max_primary_story_duration(
    capabilities: VideoModelCapabilities,
    *,
    max_content_beats: int = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
) -> float:
    """Return the largest primary-shot duration one base clip plus extensions can carry."""
    if max_content_beats < 1:
        raise ValueError("max_content_beats must be positive")
    return min(
        capabilities.max_shot_duration_s,
        capabilities.max_unique_beat_s * max_content_beats,
    )


def get_video_capabilities(
    model: str | None = None,
    provider: str | None = None,
) -> VideoModelCapabilities:
    """Resolve limits from explicit metadata, then environment configuration."""

    explicit_provider = str(provider or "").strip().lower()
    explicit_model = str(model or "").strip().lower()

    # HonCut's shipped/default provider is Seedance.  Preserve that behaviour
    # when metadata is absent, while ensuring an explicitly different provider
    # never inherits Seedance merely because SEEDANCE_MODEL exists in the env.
    if explicit_provider and "seedance" not in explicit_provider:
        return GENERIC_VIDEO_CAPABILITIES
    if explicit_model and "seedance" not in explicit_model:
        return GENERIC_VIDEO_CAPABILITIES

    selected_provider = explicit_provider or str(
        os.environ.get("VIDEO_PROVIDER") or "seedance"
    ).lower()
    selected_model = explicit_model or str(
        os.environ.get("SEEDANCE_MODEL") or "doubao-seedance-2.0-mini"
    ).lower()
    identity = f"{selected_provider} {selected_model}"
    if "seedance" in identity:
        return SEEDANCE_2_CAPABILITIES
    return GENERIC_VIDEO_CAPABILITIES


def capabilities_for(mapping: dict[str, Any] | None) -> VideoModelCapabilities:
    """Resolve a profile from storyboard/shot metadata without mutating it."""

    data = mapping or {}
    return get_video_capabilities(
        model=str(data.get("video_model") or data.get("model") or ""),
        provider=str(data.get("video_provider") or data.get("provider") or ""),
    )


__all__ = [
    "GENERIC_VIDEO_CAPABILITIES",
    "MAX_CONTENT_BEATS_PER_PRIMARY_SHOT",
    "SEEDANCE_2_CAPABILITIES",
    "VideoModelCapabilities",
    "capabilities_for",
    "get_video_capabilities",
    "max_primary_story_duration",
]
