"""Provider/model-owned limits used by director planning and generation QA.

These values describe a video model, not HonCut's storytelling policy.  Callers
must resolve a profile from the requested model/provider instead of embedding a
Seedance probe result in a global director rule.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

MAX_CONTENT_BEATS_PER_PRIMARY_SHOT = 3


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
    min_first_last_frame_duration_s: float | None = None
    max_first_last_frame_duration_s: float | None = None
    min_multi_image_duration_s: float | None = None
    max_multi_image_duration_s: float | None = None
    min_tail_extend_duration_s: float | None = None
    max_tail_extend_duration_s: float | None = None
    min_primary_story_duration_s: float | None = None
    max_primary_story_duration_s: float | None = None

    def action_limit(self, duration_seconds: float | int | None) -> int:
        if duration_seconds is None:
            return max(limit for _duration, limit in self.action_budget_steps)
        duration = max(0.0, float(duration_seconds))
        for upper_bound, limit in self.action_budget_steps:
            if duration <= upper_bound:
                return limit
        return self.action_budget_steps[-1][1]

    def request_duration_bounds(self, execution_strategy: str) -> tuple[float, float]:
        """Return provider-request limits for one execution strategy.

        Narrative time and provider request time are deliberately different:
        a tail extension may request replay context in addition to its new story
        time, while FLF2V has its own API minimum even when a shorter narrative
        bridge would otherwise be valid.
        """
        if execution_strategy == "multi_image":
            return (
                self.min_multi_image_duration_s
                if self.min_multi_image_duration_s is not None
                else self.min_shot_duration_s,
                self.max_multi_image_duration_s
                if self.max_multi_image_duration_s is not None
                else self.max_shot_duration_s,
            )
        if execution_strategy == "tail_video_extend":
            return (
                self.min_tail_extend_duration_s
                if self.min_tail_extend_duration_s is not None
                else self.min_shot_duration_s,
                self.max_tail_extend_duration_s
                if self.max_tail_extend_duration_s is not None
                else self.max_shot_duration_s,
            )
        if execution_strategy == "first_last_frame_bridge":
            return (
                self.min_first_last_frame_duration_s
                if self.min_first_last_frame_duration_s is not None
                else self.min_shot_duration_s,
                self.max_first_last_frame_duration_s
                if self.max_first_last_frame_duration_s is not None
                else self.max_shot_duration_s,
            )
        return self.min_shot_duration_s, self.max_shot_duration_s

    def effective_duration_bounds(self, execution_strategy: str) -> tuple[float, float]:
        """Return visible story time, excluding Provider padding/context.

        A Provider may require an 8-second request for a 3-second usable beat.
        Treating that request minimum as narrative time makes cost accounting
        block otherwise executable stories. Only an FLF bridge consumes its
        complete generated clip as visible transition time.
        """
        if execution_strategy in {"multi_image", "tail_video_extend"}:
            _request_minimum, request_maximum = self.request_duration_bounds(
                execution_strategy
            )
            return self.min_unique_beat_s, request_maximum
        if execution_strategy == "first_last_frame_bridge":
            return self.request_duration_bounds(execution_strategy)
        return self.min_unique_beat_s, self.max_unique_beat_s

    def request_duration_for_effective_story(
        self,
        effective_story_duration_s: float,
        execution_strategy: str,
    ) -> float:
        """Select the smallest valid request that can carry visible story time."""

        effective = float(effective_story_duration_s)
        effective_minimum, effective_maximum = self.effective_duration_bounds(
            execution_strategy
        )
        if not effective_minimum - 1e-6 <= effective <= effective_maximum + 1e-6:
            raise ValueError(
                f"effective story duration {effective:g}s is outside {self.name}'s "
                f"{effective_minimum:g}-{effective_maximum:g}s "
                f"{execution_strategy} range"
            )
        request_minimum, request_maximum = self.request_duration_bounds(
            execution_strategy
        )
        quantum = self.duration_quantum_s
        requested = max(request_minimum, effective)
        requested = math.ceil(requested / quantum - 1e-9) * quantum
        if requested > request_maximum + 1e-6:
            raise ValueError(
                f"{effective:g}s effective story time requires a {requested:g}s "
                f"request, above {self.name}'s {request_maximum:g}s maximum"
            )
        return round(requested, 6)

    def validate_chunk_durations(
        self,
        request_duration_s: float,
        unique_duration_s: float,
        execution_strategy: str,
        *,
        resource_id: str = "chunk",
    ) -> tuple[float, float]:
        """Validate provider-request and effective-story clocks independently."""
        request_duration = float(request_duration_s)
        unique_duration = float(unique_duration_s)
        quantum = self.duration_quantum_s
        if quantum <= 0:
            raise ValueError(f"{self.name} duration quantum must be positive")
        units = round(request_duration / quantum)
        quantized = units * quantum
        if not math.isclose(request_duration, quantized, abs_tol=1e-6):
            raise ValueError(
                f"{resource_id} provider request duration {request_duration:g}s cannot be "
                f"represented by {self.name}'s {quantum:g}s duration quantum"
            )
        minimum, maximum = self.request_duration_bounds(execution_strategy)
        if not minimum - 1e-6 <= quantized <= maximum + 1e-6:
            raise ValueError(
                f"{resource_id} {execution_strategy} provider request duration "
                f"{quantized:g}s is outside {self.name}'s {minimum:g}-{maximum:g}s "
                "request range"
            )
        effective_minimum, effective_maximum = self.effective_duration_bounds(
            execution_strategy
        )
        if execution_strategy != "legacy" and not (
            effective_minimum - 1e-6
            <= unique_duration
            <= effective_maximum + 1e-6
        ):
            raise ValueError(
                f"{resource_id} effective story duration {unique_duration:g}s is outside "
                f"{self.name}'s {effective_minimum:g}-"
                f"{effective_maximum:g}s {execution_strategy} range"
            )
        if unique_duration > quantized + 1e-6:
            raise ValueError(
                f"{resource_id} effective story duration {unique_duration:g}s exceeds "
                f"its {quantized:g}s provider request"
            )
        return round(quantized, 6), round(unique_duration, 6)


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
    min_first_last_frame_duration_s=4,
    max_first_last_frame_duration_s=15,
    min_multi_image_duration_s=8,
    max_multi_image_duration_s=15,
    min_tail_extend_duration_s=6,
    max_tail_extend_duration_s=10,
    min_primary_story_duration_s=3,
    max_primary_story_duration_s=35,
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
    if capabilities.max_primary_story_duration_s is not None:
        return capabilities.max_primary_story_duration_s
    return capabilities.max_unique_beat_s * max_content_beats


def min_primary_story_duration(
    capabilities: VideoModelCapabilities,
) -> float:
    """Return the minimum assembled duration of one primary story shot."""
    if capabilities.min_primary_story_duration_s is not None:
        return capabilities.min_primary_story_duration_s
    return capabilities.min_shot_duration_s


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
        os.environ.get("SEEDANCE_MODEL") or "doubao-seedance-2.0-fast"
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
    "min_primary_story_duration",
]
