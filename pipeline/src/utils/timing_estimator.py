"""Compatibility facade for the Runtime-owned structured estimator."""

from runtime.phase_estimates import (
    estimate_phase_duration,
    estimate_remaining,
    estimate_total,
    format_duration,
)


_format_duration = format_duration


__all__ = [
    "_format_duration",
    "estimate_phase_duration",
    "estimate_remaining",
    "estimate_total",
]
