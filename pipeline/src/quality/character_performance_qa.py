"""Blocking pixel QA for the run-local Phase 3 performance reference board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from clients.ark_multimodal_client import review_as
from schemas.understanding import CharacterPerformanceBoardUnderstanding
from schemas.understanding import CharacterPerformanceCellUnderstanding


CHARACTER_PERFORMANCE_QA_SCHEMA = "honcut.character-performance-board-qa.v4"
CHARACTER_PERFORMANCE_CELL_QA_SCHEMA = "honcut.character-performance-cell-qa.v2"
PERFORMANCE_CELL_IDS = tuple(f"A{index:02d}" for index in range(1, 7))


class CharacterPerformanceReviewer(Protocol):
    def review(self, image_paths: list[Path], prompt: str) -> str: ...


class CharacterPerformanceQAError(RuntimeError):
    """Raised when a performance board cannot satisfy its blocking contract."""


_CELL_FIELDS = (
    "same_character",
    "action_semantics_match",
    "pose_distinct",
    "clothing_consistent",
    "makeup_consistent",
    "healthy_beautiful_synthetic_styling",
    "no_uncanny_or_corpse_like_styling",
    "prop_ownership_correct",
    "no_extra_character",
    "no_text_or_layout_marks",
)

_BOARD_FIELDS = (
    "same_single_character",
    "six_distinct_poses",
    "clothing_makeup_consistent",
    "healthy_beautiful_synthetic_styling",
    "props_correct",
    "no_extra_characters",
    "no_text_or_layout_marks",
)


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

Return exactly one JSON object matching this schema:
{{
  "cell_id": "{cell['cell_id']}",
  "same_character": true,
  "action_semantics_match": true,
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
    payload = result.model_dump(mode="json")
    passed = (
        payload.get("cell_id") == cell.get("cell_id")
        and all(payload.get(field) is True for field in _CELL_FIELDS)
    )
    return {
        "schema": CHARACTER_PERFORMANCE_CELL_QA_SCHEMA,
        "passed": passed,
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
        and all(
            item.get("passed") is True
            and all(item.get(field) is True for field in _CELL_FIELDS)
            for item in cell_results
        )
    )
    board_passed = all(board_result.get(field) is True for field in _BOARD_FIELDS)
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
        **{field: board_result.get(field) is True for field in _BOARD_FIELDS},
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
- outfit, face identity, synthetic porcelain makeup, colors and proportions stay identical;
- the complexion stays warm, healthy and elegant; eyes have pupils, iris detail and catchlights;
  lips and cheeks retain coordinated living color; no cell looks gray, bloodless, waxy,
  corpse-like, haunted, uncanny or horror-styled;
- only the declared prop belongs to this character and its geometry/material are correct;
- no extra character appears;
- pixels contain no text, Axx labels, numbers, arrows, panel borders, grid lines, captions or UI.

Return exactly one JSON object with this schema:
{{
  "passed": true,
  "cells": [
    {{
      "cell_id": "A01",
      "same_character": true,
      "action_semantics_match": true,
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
    reviewed_cells = payload.get("cells") or []
    ids = [str(cell.get("cell_id") or "") for cell in reviewed_cells]
    cells_passed = (
        ids == list(PERFORMANCE_CELL_IDS)
        and all(
            all(cell.get(field) is True for field in _CELL_FIELDS)
            for cell in reviewed_cells
        )
    )
    passed = cells_passed and all(payload.get(field) is True for field in _BOARD_FIELDS)
    return {
        "schema": CHARACTER_PERFORMANCE_QA_SCHEMA,
        "passed": passed,
        "cells": reviewed_cells,
        **{field: payload.get(field) is True for field in _BOARD_FIELDS},
        "issues": [str(item) for item in payload.get("issues") or []],
    }
