"""Pure Phase 5 policy for one bounded screenplay rewrite."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from schemas.replanning import (
    PADDING_LOSS_ERROR_CODE,
    SCREENPLAY_REWRITE_REQUEST_SCHEMA,
)

MAX_PADDING_SCREENPLAY_REWRITES = 1


def build_padding_screenplay_rewrite_request(
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build the only Phase 5 request allowed to restart the screenwriter."""

    issue = next(
        (
            item
            for item in issues
            if isinstance(item, Mapping)
            and item.get("code") == PADDING_LOSS_ERROR_CODE
        ),
        None,
    )
    if issue is None:
        return None
    details = issue.get("details")
    values = details if isinstance(details, Mapping) else issue
    return {
        "schema": SCREENPLAY_REWRITE_REQUEST_SCHEMA,
        "reason_code": PADDING_LOSS_ERROR_CODE,
        "attempt": 1,
        "maximum_padding_loss_rate": float(
            values.get("maximum_padding_loss_rate") or 0
        ),
        "observed_padding_loss_rate": float(
            values.get("padding_loss_rate") or 0
        ),
        "content_provider_request_duration_s": float(
            values.get("content_provider_request_duration_s") or 0
        ),
        "content_provider_padding_duration_s": float(
            values.get("content_provider_padding_duration_s") or 0
        ),
    }


def rewrite_request_from_receipt(
    phase_receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a validated padding rewrite request from a Phase 5 receipt."""

    if not isinstance(phase_receipt, Mapping):
        return None
    correction = phase_receipt.get("correction")
    if not isinstance(correction, Mapping):
        return None
    request = correction.get("screenplay_rewrite_request")
    if not isinstance(request, Mapping):
        return None
    if (
        request.get("schema") != SCREENPLAY_REWRITE_REQUEST_SCHEMA
        or request.get("reason_code") != PADDING_LOSS_ERROR_CODE
        or request.get("attempt") != 1
    ):
        return None
    maximum = request.get("maximum_padding_loss_rate")
    if isinstance(maximum, bool):
        return None
    try:
        maximum_value = float(maximum)
    except (TypeError, ValueError):
        return None
    if not 0 < maximum_value < 1:
        return None
    return dict(request)


def rewrite_attempt_from_receipt(
    phase_receipt: Mapping[str, Any] | None,
) -> int:
    """Read the persisted attempt count without accepting malformed values."""

    if not isinstance(phase_receipt, Mapping):
        return 0
    correction = phase_receipt.get("correction")
    if not isinstance(correction, Mapping):
        return 0
    raw = correction.get("screenplay_rewrite_attempt", 0)
    if isinstance(raw, bool):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


__all__ = [
    "MAX_PADDING_SCREENPLAY_REWRITES",
    "PADDING_LOSS_ERROR_CODE",
    "SCREENPLAY_REWRITE_REQUEST_SCHEMA",
    "build_padding_screenplay_rewrite_request",
    "rewrite_attempt_from_receipt",
    "rewrite_request_from_receipt",
]
