"""Blocking pixel QA for the run-local Phase 3 performance reference board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from clients.ark_multimodal_client import review_as
from schemas.understanding import CharacterPerformanceBoardUnderstanding


CHARACTER_PERFORMANCE_QA_SCHEMA = "honcut.character-performance-board-qa.v1"
PERFORMANCE_CELL_IDS = tuple(f"A{index:02d}" for index in range(1, 7))


class CharacterPerformanceReviewer(Protocol):
    def review(self, image_paths: list[Path], prompt: str) -> str: ...


class CharacterPerformanceQAError(RuntimeError):
    """Raised when a performance board cannot satisfy its blocking contract."""


def build_character_performance_qa_prompt(
    *,
    character_id: str,
    cells: list[dict[str, Any]],
    synthetic_styling: dict[str, Any] | None,
) -> str:
    return f"""You are the blocking Phase 3 performance-reference-board inspector.
The single input is a clean 2x3 image read left-to-right, top-to-bottom as A01-A06.
The Axx identifiers exist only in this inspection contract and must NOT be printed in pixels.

Character ID: {character_id}
Synthetic styling: {json.dumps(synthetic_styling, ensure_ascii=False, sort_keys=True)}
Authored cell bindings (do not invent or extend story action):
{json.dumps(cells, ensure_ascii=False, sort_keys=True)}

Blocking requirements:
- all six cells depict the same one character, never six people or clones;
- six clearly different practical body poses are visible;
- outfit, face identity, synthetic porcelain makeup, colors and proportions stay identical;
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
      "pose_matches_action": true,
      "pose_distinct": true,
      "clothing_consistent": true,
      "makeup_consistent": true,
      "prop_ownership_correct": true,
      "no_extra_character": true,
      "no_text_or_layout_marks": true,
      "issues": []
    }}
  ],
  "same_single_character": true,
  "six_distinct_poses": true,
  "clothing_makeup_consistent": true,
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
    cell_fields = (
        "same_character",
        "pose_matches_action",
        "pose_distinct",
        "clothing_consistent",
        "makeup_consistent",
        "prop_ownership_correct",
        "no_extra_character",
        "no_text_or_layout_marks",
    )
    cells_passed = (
        ids == list(PERFORMANCE_CELL_IDS)
        and all(
            all(cell.get(field) is True for field in cell_fields)
            for cell in reviewed_cells
        )
    )
    board_fields = (
        "same_single_character",
        "six_distinct_poses",
        "clothing_makeup_consistent",
        "props_correct",
        "no_extra_characters",
        "no_text_or_layout_marks",
    )
    passed = cells_passed and all(payload.get(field) is True for field in board_fields)
    return {
        "schema": CHARACTER_PERFORMANCE_QA_SCHEMA,
        "passed": passed,
        "cells": reviewed_cells,
        **{field: payload.get(field) is True for field in board_fields},
        "issues": [str(item) for item in payload.get("issues") or []],
    }
