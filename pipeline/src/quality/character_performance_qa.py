"""Blocking pixel QA for the run-local Phase 3 performance reference board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from clients.ark_multimodal_client import review_as
from schemas.understanding import CharacterPerformanceBoardUnderstanding
from schemas.understanding import CharacterPerformanceCellUnderstanding


CHARACTER_PERFORMANCE_QA_SCHEMA = "honcut.character-performance-board-qa.v6"
CHARACTER_PERFORMANCE_CELL_QA_SCHEMA = "honcut.character-performance-cell-qa.v4"
PERFORMANCE_CELL_IDS = tuple(f"A{index:02d}" for index in range(1, 7))
SEMANTIC_ACCEPTANCE_CONFIDENCE = 0.65
SEMANTIC_REJECTION_CONFIDENCE = 0.85
CONFIDENCE_QA_POLICY = "honcut.confidence-tolerant-visual-qa.v1"


class CharacterPerformanceReviewer(Protocol):
    def review(self, image_paths: list[Path], prompt: str) -> str: ...


class CharacterPerformanceQAError(RuntimeError):
    """Raised when a performance board cannot satisfy its blocking contract."""


_HARD_CELL_FIELDS = (
    "same_character",
    "clothing_consistent",
    "makeup_consistent",
    "healthy_beautiful_synthetic_styling",
    "no_uncanny_or_corpse_like_styling",
    "prop_ownership_correct",
    "no_extra_character",
    "no_text_or_layout_marks",
)

_HARD_BOARD_FIELDS = (
    "same_single_character",
    "clothing_makeup_consistent",
    "healthy_beautiful_synthetic_styling",
    "props_correct",
    "no_extra_characters",
    "no_text_or_layout_marks",
)

_BOARD_OUTPUT_FIELDS = (
    "same_single_character",
    "six_distinct_poses",
    "clothing_makeup_consistent",
    "healthy_beautiful_synthetic_styling",
    "props_correct",
    "no_extra_characters",
    "no_text_or_layout_marks",
)


def _normalized_cell_verdict(
    payload: dict[str, Any],
    *,
    expected_cell_id: str | None = None,
) -> dict[str, Any]:
    """Apply deterministic policy to one probabilistic semantic review."""
    blocking_fields = [
        field for field in _HARD_CELL_FIELDS if payload.get(field) is not True
    ]
    confidence = float(payload.get("action_semantics_confidence", 0.0))
    evidence = [
        str(item).strip()
        for item in payload.get("action_semantics_evidence") or []
        if str(item).strip()
    ]
    if payload.get("action_semantics_match") is True:
        action_status = (
            "passed"
            if confidence >= SEMANTIC_ACCEPTANCE_CONFIDENCE
            else "diagnostic_uncertain"
        )
    elif confidence >= SEMANTIC_REJECTION_CONFIDENCE and evidence:
        action_status = "blocking"
        blocking_fields.append("action_semantics_match")
    else:
        action_status = "diagnostic_uncertain"
    if expected_cell_id is not None and payload.get("cell_id") != expected_cell_id:
        blocking_fields.append("cell_id")
    return {
        **payload,
        "action_semantics_confidence": confidence,
        "action_semantics_evidence": evidence,
        "action_semantics_status": action_status,
        "blocking_fields": list(dict.fromkeys(blocking_fields)),
        "confidence_qa_policy": CONFIDENCE_QA_POLICY,
        "semantic_acceptance_confidence": SEMANTIC_ACCEPTANCE_CONFIDENCE,
        "semantic_rejection_confidence": SEMANTIC_REJECTION_CONFIDENCE,
        "passed": not blocking_fields,
    }


def _normalized_board_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep deterministic board invariants hard and calibrate pose diversity."""
    blocking_fields = [
        field for field in _HARD_BOARD_FIELDS if payload.get(field) is not True
    ]
    confidence = float(payload.get("pose_diversity_confidence", 0.0))
    evidence = [
        str(item).strip()
        for item in payload.get("pose_diversity_evidence") or []
        if str(item).strip()
    ]
    if payload.get("six_distinct_poses") is True:
        diversity_status = (
            "passed"
            if confidence >= SEMANTIC_ACCEPTANCE_CONFIDENCE
            else "diagnostic_uncertain"
        )
    elif confidence >= SEMANTIC_REJECTION_CONFIDENCE and evidence:
        diversity_status = "blocking"
        blocking_fields.append("six_distinct_poses")
    else:
        diversity_status = "diagnostic_uncertain"
    return {
        **payload,
        "pose_diversity_confidence": confidence,
        "pose_diversity_evidence": evidence,
        "pose_diversity_status": diversity_status,
        "board_blocking_fields": list(dict.fromkeys(blocking_fields)),
        "confidence_qa_policy": CONFIDENCE_QA_POLICY,
        "semantic_acceptance_confidence": SEMANTIC_ACCEPTANCE_CONFIDENCE,
        "semantic_rejection_confidence": SEMANTIC_REJECTION_CONFIDENCE,
        "passed": not blocking_fields,
    }


def build_character_performance_cell_qa_prompt(
    *,
    character_id: str,
    cell: dict[str, Any],
    synthetic_styling: dict[str, Any] | None,
) -> str:
    from utils.privacy_visual_policy import synthetic_makeup_qa_requirements

    return f"""You are the blocking Phase 3 single performance-cell inspector.
Image 1 is one generated action pose. Image 2 is the canonical character identity board.
Image 3, when present, is the canonical prop-detail board. Compare identity and prop against those
references, but judge the action only from Image 1. The internal Axx ID must not appear in pixels.

Character ID: {character_id}
Authored cell binding: {json.dumps(cell, ensure_ascii=False, sort_keys=True)}
Synthetic styling: {json.dumps(synthetic_styling, ensure_ascii=False, sort_keys=True)}
Structured aesthetic QA requirements:
{json.dumps(synthetic_makeup_qa_requirements(), ensure_ascii=False)}

Set action_semantics_match from the stable action family and major physical relationship: the
declared ready/attack/evade/block/hold/use action, its visible weight shift or torso state, and the
declared prop relationship must be recognizable. A neutral portrait or wrong action family fails.
Set fine_direction_match separately for exact anatomical left/right foot, screen-sensitive
diagonal endpoint and minor orientation. Camera mirroring or an ambiguous 3/4 view must not by
itself turn a recognizable correct action into action_semantics_match=false; report it through
fine_direction_match and issues instead. Also require the same one character, consistent outfit
and warm beautiful synthetic porcelain makeup, correct declared prop, no extra character and no
text/labels/arrows/borders/grid/UI.

Set action_semantics_confidence to your calibrated confidence in action_semantics_match from the
visible pixels, from 0.0 to 1.0. A recognizable match at 0.65 or above is accepted. Use 0.90 or
above only when the action family and major prop/body
relationship are plainly visible enough to support the verdict; use 0.50-0.80 for an ambiguous
camera, hidden joint, transitional pose or plausible alternative reading. Put only concrete visible
facts in action_semantics_evidence, not a repetition of the authored contract. A low-confidence
negative is a diagnostic uncertainty, not proof that a paid redraw is needed. pose_distinct is
diagnostic in this isolated single-cell review because six-pose diversity belongs to whole-board QA.

Return exactly one JSON object matching this schema:
{{
  "cell_id": "{cell['cell_id']}",
  "same_character": true,
  "action_semantics_match": true,
  "action_semantics_confidence": 0.95,
  "action_semantics_evidence": ["weight is shifted and the prop crosses the torso defensively"],
  "fine_direction_match": true,
  "pose_distinct": true,
  "clothing_consistent": true,
  "makeup_consistent": true,
  "healthy_beautiful_synthetic_styling": true,
  "no_uncanny_or_corpse_like_styling": true,
  "prop_ownership_correct": true,
  "no_extra_character": true,
  "no_text_or_layout_marks": true,
  "issues": []
}}
"""


def review_character_performance_cell(
    reviewer: CharacterPerformanceReviewer,
    image_path: Path,
    *,
    identity_path: Path,
    prop_path: Path | None,
    character_id: str,
    cell: dict[str, Any],
    synthetic_styling: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = build_character_performance_cell_qa_prompt(
        character_id=character_id,
        cell=cell,
        synthetic_styling=synthetic_styling,
    )
    image_paths = [image_path, identity_path]
    if prop_path is not None:
        image_paths.append(prop_path)
    result = review_as(
        reviewer,
        image_paths,
        prompt,
        CharacterPerformanceCellUnderstanding,
    )
    payload = _normalized_cell_verdict(
        result.model_dump(mode="json"),
        expected_cell_id=str(cell.get("cell_id") or ""),
    )
    return {
        "schema": CHARACTER_PERFORMANCE_CELL_QA_SCHEMA,
        **payload,
        "issues": [str(item) for item in payload.get("issues") or []],
    }


def combine_character_performance_qa(
    board_result: dict[str, Any],
    cell_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze isolated action verdicts while retaining board-global inspection."""
    ids = [str(item.get("cell_id") or "") for item in cell_results]
    cells_passed = (
        ids == list(PERFORMANCE_CELL_IDS)
        and all(item.get("passed") is True for item in cell_results)
    )
    # Once isolated cell verdicts exist, whole-board action-cell verdicts are
    # deliberately superseded.  Retain only true board-global blockers.
    board_blocking_fields = [
        str(field)
        for field in board_result.get("board_blocking_fields") or []
        if str(field) != "cells"
    ]
    board_passed = not board_blocking_fields
    issues = (
        [str(item) for item in board_result.get("issues") or []]
        if not board_passed
        else []
    )
    for item in cell_results:
        issues.extend(str(issue) for issue in item.get("issues") or [])
    return {
        "schema": CHARACTER_PERFORMANCE_QA_SCHEMA,
        "passed": cells_passed and board_passed,
        "cells": [
            {key: value for key, value in item.items() if key not in {"schema", "passed"}}
            for item in cell_results
        ],
        **{field: board_result.get(field) is True for field in _BOARD_OUTPUT_FIELDS},
        "pose_diversity_confidence": board_result.get("pose_diversity_confidence"),
        "pose_diversity_evidence": board_result.get("pose_diversity_evidence") or [],
        "pose_diversity_status": board_result.get("pose_diversity_status"),
        "board_blocking_fields": board_blocking_fields,
        "confidence_qa_policy": CONFIDENCE_QA_POLICY,
        "semantic_acceptance_confidence": SEMANTIC_ACCEPTANCE_CONFIDENCE,
        "semantic_rejection_confidence": SEMANTIC_REJECTION_CONFIDENCE,
        "issues": list(dict.fromkeys(issues)),
        "action_verdict_source": "isolated_persisted_cells",
        "board_verdict_source": "whole_board_global_fields_only",
    }


def build_character_performance_qa_prompt(
    *,
    character_id: str,
    cells: list[dict[str, Any]],
    synthetic_styling: dict[str, Any] | None,
) -> str:
    from utils.privacy_visual_policy import synthetic_makeup_qa_requirements

    return f"""You are the blocking Phase 3 performance-reference-board inspector.
The single input is a clean 2x3 image read left-to-right, top-to-bottom as A01-A06.
The Axx identifiers exist only in this inspection contract and must NOT be printed in pixels.

Character ID: {character_id}
Synthetic styling: {json.dumps(synthetic_styling, ensure_ascii=False, sort_keys=True)}
Structured aesthetic QA requirements:
{json.dumps(synthetic_makeup_qa_requirements(), ensure_ascii=False)}
Authored cell bindings (do not invent or extend story action):
{json.dumps(cells, ensure_ascii=False, sort_keys=True)}

Blocking requirements:
- all six cells depict the same one character, never six people or clones;
- six clearly different practical body poses are visible;
- action_semantics_match judges the declared action family and major body/prop relationship;
  fine_direction_match records exact anatomical-side or diagonal details but is diagnostic because
  a 3/4 camera or mirrored screen direction cannot reliably block a correct action family;
- action_semantics_confidence is confidence in that semantic verdict, not image quality. A positive
  semantic match at 0.65 or above is accepted. Use 0.90+ only for a plainly visible action
  match/mismatch, and 0.50-0.80 for occlusion, camera ambiguity,
  a transitional pose or a plausible alternative reading. action_semantics_evidence contains only
  concrete visible facts. A low-confidence negative is diagnostic, not redraw evidence;
- outfit, face identity, synthetic porcelain makeup, colors and proportions stay identical;
- the complexion stays warm, healthy and elegant; eyes have pupils, iris detail and catchlights;
  lips and cheeks retain coordinated living color; no cell looks gray, bloodless, waxy,
  corpse-like, haunted, uncanny or horror-styled;
- only the declared prop belongs to this character and its geometry/material are correct;
- no extra character appears;
- pixels contain no text, Axx labels, numbers, arrows, panel borders, grid lines, captions or UI.

Set pose_diversity_confidence with the same calibration rule and list concrete visual comparisons in
pose_diversity_evidence. Use a high-confidence negative only when cells visibly repeat essentially
the same silhouette/weight/prop relationship; small stylistic similarity is allowed.

Return exactly one JSON object with this schema:
{{
  "passed": true,
  "cells": [
    {{
      "cell_id": "A01",
      "same_character": true,
      "action_semantics_match": true,
      "action_semantics_confidence": 0.95,
      "action_semantics_evidence": ["the prop and body form the declared defensive relationship"],
      "fine_direction_match": true,
      "pose_distinct": true,
      "clothing_consistent": true,
      "makeup_consistent": true,
      "healthy_beautiful_synthetic_styling": true,
      "no_uncanny_or_corpse_like_styling": true,
      "prop_ownership_correct": true,
      "no_extra_character": true,
      "no_text_or_layout_marks": true,
      "issues": []
    }}
  ],
  "same_single_character": true,
  "six_distinct_poses": true,
  "pose_diversity_confidence": 0.95,
  "pose_diversity_evidence": ["the six silhouettes use visibly different weight and prop states"],
  "clothing_makeup_consistent": true,
  "healthy_beautiful_synthetic_styling": true,
  "props_correct": true,
  "no_extra_characters": true,
  "no_text_or_layout_marks": true,
  "issues": []
}}
The cells array must contain A01 through A06 exactly once and in order.
"""


def review_character_performance_board(
    reviewer: CharacterPerformanceReviewer,
    image_path: Path,
    *,
    character_id: str,
    cells: list[dict[str, Any]],
    synthetic_styling: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = build_character_performance_qa_prompt(
        character_id=character_id,
        cells=cells,
        synthetic_styling=synthetic_styling,
    )
    result = review_as(
        reviewer,
        [image_path],
        prompt,
        CharacterPerformanceBoardUnderstanding,
    )
    payload = result.model_dump(mode="json")
    reviewed_cells = [
        _normalized_cell_verdict(dict(cell))
        for cell in payload.get("cells") or []
    ]
    ids = [str(cell.get("cell_id") or "") for cell in reviewed_cells]
    normalized = _normalized_board_verdict({
        **payload,
        "cells": reviewed_cells,
    })
    if ids != list(PERFORMANCE_CELL_IDS) or any(
        cell.get("passed") is not True for cell in reviewed_cells
    ):
        normalized["board_blocking_fields"] = list(dict.fromkeys([
            *normalized["board_blocking_fields"],
            "cells",
        ]))
        normalized["passed"] = False
    return {
        "schema": CHARACTER_PERFORMANCE_QA_SCHEMA,
        "cells": reviewed_cells,
        **{field: normalized.get(field) is True for field in _BOARD_OUTPUT_FIELDS},
        "pose_diversity_confidence": normalized["pose_diversity_confidence"],
        "pose_diversity_evidence": normalized["pose_diversity_evidence"],
        "pose_diversity_status": normalized["pose_diversity_status"],
        "board_blocking_fields": normalized["board_blocking_fields"],
        "confidence_qa_policy": CONFIDENCE_QA_POLICY,
        "semantic_acceptance_confidence": SEMANTIC_ACCEPTANCE_CONFIDENCE,
        "semantic_rejection_confidence": SEMANTIC_REJECTION_CONFIDENCE,
        "passed": normalized["passed"],
        "issues": [str(item) for item in payload.get("issues") or []],
    }
