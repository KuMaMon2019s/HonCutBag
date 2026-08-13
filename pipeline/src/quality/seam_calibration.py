"""Auditable calibration and fail-closed decisions for continuity seams."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CALIBRATION_KIND = "honcut.seam_calibration.v1"
SCORE_KIND = "honcut.provisional_seam_risk.v1"
SeamLabel = Literal["acceptable", "defective"]


class SeamObservation(BaseModel):
    """One human-labelled boundary and its raw metric evidence."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1)
    label: SeamLabel
    source: str = Field(min_length=1)
    metrics: dict[str, Any]

    @field_validator("metrics")
    @classmethod
    def validate_risk_score(cls, value: dict[str, Any]) -> dict[str, Any]:
        _risk_score(value)
        return value


class SeamCalibration(BaseModel):
    """Persisted threshold contract derived only from labelled observations."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["honcut.seam_calibration.v1"] = CALIBRATION_KIND
    score_kind: Literal["honcut.provisional_seam_risk.v1"] = SCORE_KIND
    status: Literal["certified", "insufficient", "overlap"]
    dataset_fingerprint: str
    sample_counts: dict[str, int]
    minimum_samples_per_label: int
    minimum_separation: float
    accept_threshold: float | None = None
    regenerate_threshold: float | None = None
    observed_ranges: dict[str, list[float]]
    reason: str


def _risk_score(metrics: dict[str, Any]) -> float:
    value = metrics.get("provisional_risk_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("seam metrics require a numeric provisional_risk_score")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("provisional_risk_score must be finite and between 0 and 1")
    return score


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_seam_observation(
    evidence: dict[str, Any],
    *,
    label: SeamLabel,
    source: str,
    observation_id: str | None = None,
) -> SeamObservation:
    """Convert measured evidence into an explicitly labelled calibration sample."""
    metrics = evidence.get("metrics", evidence)
    if not isinstance(metrics, dict):
        raise ValueError("seam evidence must contain a metrics object")
    boundary_id = evidence.get("boundary_id")
    resolved_id = observation_id or (str(boundary_id) if boundary_id else "")
    return SeamObservation(
        observation_id=resolved_id,
        label=label,
        source=source,
        metrics=metrics,
    )


def calibrate_seam_policy(
    observations: Sequence[SeamObservation | dict[str, Any]],
    *,
    minimum_samples_per_label: int = 3,
    minimum_separation: float = 0.05,
) -> SeamCalibration:
    """Derive a conservative three-way policy from labelled observations.

    Scores at or below the worst known acceptable example may pass. Scores at or
    above the best known defective example may be repaired. The gap remains a
    human-review band. Overlapping classes never certify automation.
    """
    if minimum_samples_per_label < 1:
        raise ValueError("minimum_samples_per_label must be at least 1")
    if not 0.0 <= minimum_separation <= 1.0:
        raise ValueError("minimum_separation must be between 0 and 1")

    samples = [
        item if isinstance(item, SeamObservation) else SeamObservation.model_validate(item)
        for item in observations
    ]
    ids = [item.observation_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("seam observation ids must be unique")

    scores = {
        label: sorted(_risk_score(item.metrics) for item in samples if item.label == label)
        for label in ("acceptable", "defective")
    }
    counts = {label: len(values) for label, values in scores.items()}
    fingerprint = _canonical_hash(
        [item.model_dump(mode="json") for item in sorted(samples, key=lambda item: item.observation_id)]
    )
    ranges = {
        label: ([round(values[0], 6), round(values[-1], 6)] if values else [])
        for label, values in scores.items()
    }

    if any(count < minimum_samples_per_label for count in counts.values()):
        return SeamCalibration(
            status="insufficient",
            dataset_fingerprint=fingerprint,
            sample_counts=counts,
            minimum_samples_per_label=minimum_samples_per_label,
            minimum_separation=minimum_separation,
            observed_ranges=ranges,
            reason="each label needs more human-reviewed samples before automation",
        )

    accept_threshold = scores["acceptable"][-1]
    regenerate_threshold = scores["defective"][0]
    separation = regenerate_threshold - accept_threshold
    if separation < minimum_separation:
        return SeamCalibration(
            status="overlap",
            dataset_fingerprint=fingerprint,
            sample_counts=counts,
            minimum_samples_per_label=minimum_samples_per_label,
            minimum_separation=minimum_separation,
            observed_ranges=ranges,
            reason=(
                "labelled score ranges overlap or do not leave the required review margin"
            ),
        )

    return SeamCalibration(
        status="certified",
        dataset_fingerprint=fingerprint,
        sample_counts=counts,
        minimum_samples_per_label=minimum_samples_per_label,
        minimum_separation=minimum_separation,
        accept_threshold=round(accept_threshold, 6),
        regenerate_threshold=round(regenerate_threshold, 6),
        observed_ranges=ranges,
        reason="labelled classes are separated; ambiguous scores remain human-reviewed",
    )


def decide_seam(
    metrics: dict[str, Any],
    calibration: SeamCalibration | dict[str, Any],
    *,
    repair_attempts: int = 0,
    max_repairs: int = 1,
) -> dict[str, Any]:
    """Return accept, regenerate, review, or observe without hiding uncertainty."""
    if repair_attempts < 0 or max_repairs < 0:
        raise ValueError("repair counts must not be negative")
    policy = (
        calibration
        if isinstance(calibration, SeamCalibration)
        else SeamCalibration.model_validate(calibration)
    )
    score = _risk_score(metrics)
    base = {
        "risk_score": round(score, 6),
        "calibration_fingerprint": policy.dataset_fingerprint,
        "repair_attempts": repair_attempts,
        "max_repairs": max_repairs,
    }
    if policy.status != "certified":
        return {
            **base,
            "action": "observe_only",
            "reason": f"calibration is {policy.status}; automatic decisions are disabled",
        }
    if policy.accept_threshold is None or policy.regenerate_threshold is None:
        raise ValueError("certified calibration requires both thresholds")
    if score <= policy.accept_threshold:
        return {**base, "action": "accept", "reason": "score is in the calibrated accept range"}
    if score < policy.regenerate_threshold:
        return {
            **base,
            "action": "human_review",
            "reason": "score is in the calibrated uncertainty band",
        }
    if repair_attempts < max_repairs:
        return {
            **base,
            "action": "regenerate",
            "reason": "score is in the calibrated defective range",
        }
    return {
        **base,
        "action": "human_review",
        "reason": "defective-range score remains after the repair budget was exhausted",
    }


def write_seam_calibration(
    path: str | Path,
    observations: Sequence[SeamObservation | dict[str, Any]],
    *,
    minimum_samples_per_label: int = 3,
    minimum_separation: float = 0.05,
) -> SeamCalibration:
    policy = calibrate_seam_policy(
        observations,
        minimum_samples_per_label=minimum_samples_per_label,
        minimum_separation=minimum_separation,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return policy


def load_seam_calibration(path: str | Path) -> SeamCalibration:
    return SeamCalibration.model_validate_json(Path(path).read_text(encoding="utf-8"))
