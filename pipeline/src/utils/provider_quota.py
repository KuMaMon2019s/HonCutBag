"""Provider-independent fixed-window quota classification."""

from __future__ import annotations

from typing import Any


class FixedWindowQuotaExceededError(RuntimeError):
    """A provider quota cannot recover before its declared reset window."""


_FIXED_WINDOW_QUOTA_MARKERS = (
    "monthly usage quota",
    "will reset at",
    "waiting for the reset",
    "wait for the reset",
    "quota resets at",
    "quota will reset",
)


def _quota_error_text(value: Any) -> str:
    """Collect stable public error fields without depending on one SDK."""
    parts = [str(value)]
    for name in ("code", "provider_code", "message", "provider_message", "body"):
        field = getattr(value, name, None)
        if field not in (None, ""):
            parts.append(str(field))
    response = getattr(value, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text not in (None, ""):
            parts.append(str(text))
    return " ".join(parts).casefold()


def is_fixed_window_quota_exhaustion(value: Any) -> bool:
    """Return whether retrying cannot help before a provider-declared reset."""
    text = _quota_error_text(value)
    return any(marker in text for marker in _FIXED_WINDOW_QUOTA_MARKERS)
