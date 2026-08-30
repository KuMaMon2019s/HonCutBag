"""Deterministic tolerance policy for probabilistic visual QA observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal


SEMANTIC_PASS_THRESHOLD = 0.65
NEGATIVE_BLOCK_THRESHOLD = 0.85
POLICY_ID = "honcut.visual-qa-policy.v1"
DETERMINISTIC_BLOCKING_CATEGORIES = frozenset({
    "artifact_hash",
    "budget",
    "canonical_contract",
    "lineage",
    "media_authority",
    "media_count",
    "people_count",
    "schema",
})


def policy_sha256() -> str:
    value = {
        "deterministic_blocking_categories": sorted(DETERMINISTIC_BLOCKING_CATEGORIES),
        "negative_block_threshold": NEGATIVE_BLOCK_THRESHOLD,
        "policy_id": POLICY_ID,
        "semantic_pass_threshold": SEMANTIC_PASS_THRESHOLD,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VisualQADecision:
    verdict: Literal["pass", "acceptable_deviation", "block", "manual_review"]
    semantic_score: float | None
    blocking_categories: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocking_categories": list(self.blocking_categories),
            "diagnostics": list(self.diagnostics),
            "negative_block_threshold": NEGATIVE_BLOCK_THRESHOLD,
            "rationale": self.rationale,
            "semantic_pass_threshold": SEMANTIC_PASS_THRESHOLD,
            "semantic_score": self.semantic_score,
            "verdict": self.verdict,
        }


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def decide_visual_qa(
    *,
    semantic_score: float | None,
    findings: list[dict[str, Any]],
    deterministic_errors: list[dict[str, Any]] | None = None,
) -> VisualQADecision:
    """Convert model observations into one deterministic, tolerant decision."""
    strict_errors = list(deterministic_errors or [])
    strict_categories = sorted({
        str(value.get("category") or "schema")
        for value in strict_errors
    })
    if strict_errors:
        return VisualQADecision(
            verdict="block",
            semantic_score=semantic_score,
            blocking_categories=tuple(strict_categories),
            diagnostics=tuple([*strict_errors, *findings]),
            rationale="deterministic contract error",
        )

    blocking: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for finding in findings:
        category = str(finding.get("blocking_category") or "").strip()
        evidence = finding.get("evidence") or finding.get("panel_evidence")
        confidence = _confidence(finding.get("confidence"))
        if category and evidence and confidence >= NEGATIVE_BLOCK_THRESHOLD:
            blocking.append(finding)
        else:
            diagnostics.append(finding)
    if blocking:
        return VisualQADecision(
            verdict="block",
            semantic_score=semantic_score,
            blocking_categories=tuple(sorted({
                str(value["blocking_category"]) for value in blocking
            })),
            diagnostics=tuple(diagnostics),
            rationale="high-confidence negative finding with concrete evidence",
        )

    if semantic_score is None:
        return VisualQADecision(
            verdict="manual_review",
            semantic_score=None,
            blocking_categories=(),
            diagnostics=tuple(diagnostics),
            rationale="semantic score missing",
        )
    score = _confidence(semantic_score)
    if score >= SEMANTIC_PASS_THRESHOLD and not findings:
        verdict = "pass"
        rationale = "semantic score meets tolerance threshold"
    elif score >= SEMANTIC_PASS_THRESHOLD:
        verdict = "acceptable_deviation"
        rationale = "score passes and lower-confidence findings remain diagnostic"
    else:
        verdict = "manual_review"
        rationale = "score is below tolerance threshold without blocking evidence"
    return VisualQADecision(
        verdict=verdict,
        semantic_score=score,
        blocking_categories=(),
        diagnostics=tuple(diagnostics),
        rationale=rationale,
    )
