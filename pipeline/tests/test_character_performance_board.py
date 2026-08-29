"""Phase 3 run-local performance-board contracts."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from phases.phase3.performance_reference_board import (
    CHARACTER_PERFORMANCE_BOARD_SCHEMA,
    PERFORMANCE_CELL_IDS,
    attach_performance_guides_to_storyboard,
    build_character_performance_plan,
    generate_character_performance_board,
    validate_character_performance_board,
    validate_character_performance_guide,
)


def _character() -> dict:
    return {
        "id": "lead",
        "name": "领队",
        "appearance": {
            "clothing": "深蓝短外套与黑色长裤",
            "identity_props": [{
                "id": "energy_baton",
                "name": "蓝色能量短棍",
                "owner": "lead",
            }],
            "synthetic_styling": {
                "schema": "honcut.synthetic-styling.v3",
                "mode": "synthetic_porcelain_makeup",
                "makeup_design_id": "porcelain-lead",
                "non_human_material": "pearl bio-ceramic complexion",
                "visible_anchors": ["blue circuit stripe", "silver iris ring"],
            },
        },
    }


def _unit(unit_id: str, source_id: str, action: str) -> dict:
    return {
        "unit_id": unit_id,
        "source_action_unit_id": source_id,
        "source_fact_echoes": [action],
        "actions": [action],
    }


def _storyboard() -> dict:
    return {
        "shots": [{
            "id": "S01",
            "shot_intent": "action",
            "storyboard_beats": [
                {
                    "beat_id": "S01_P01",
                    "character_ids": ["lead"],
                    "action": "领队握住蓝色能量短棍格挡",
                    "source_action_unit_ids": ["AU001"],
                    "generation_action_units": [
                        _unit("GAU001", "AU001", "领队握住蓝色能量短棍格挡")
                    ],
                },
                {
                    "beat_id": "S01_P02",
                    "character_ids": ["lead"],
                    "action": "领队侧身闪避后挥动蓝色能量短棍",
                    "source_action_unit_ids": ["AU002"],
                    "generation_action_units": [
                        _unit("GAU002", "AU002", "领队侧身闪避后挥动蓝色能量短棍")
                    ],
                },
            ],
        }],
    }


def _write_reference_assets(output_dir: Path) -> None:
    character_dir = output_dir / "characters" / "lead"
    character_dir.mkdir(parents=True)
    Image.new("RGB", (1536, 1536), (190, 200, 220)).save(
        character_dir / "reference_board.png"
    )
    Image.new("RGB", (1536, 1024), (80, 120, 210)).save(
        character_dir / "prop_detail_board.png"
    )


def _qa_payload() -> str:
    return json.dumps({
        "passed": True,
        "cells": [
            {
                "cell_id": cell_id,
                "same_character": True,
                "pose_matches_action": True,
                "pose_distinct": True,
                "clothing_consistent": True,
                "makeup_consistent": True,
                "prop_ownership_correct": True,
                "no_extra_character": True,
                "no_text_or_layout_marks": True,
                "issues": [],
            }
            for cell_id in PERFORMANCE_CELL_IDS
        ],
        "same_single_character": True,
        "six_distinct_poses": True,
        "clothing_makeup_consistent": True,
        "props_correct": True,
        "no_extra_characters": True,
        "no_text_or_layout_marks": True,
        "issues": [],
    })


class _ImageClient:
    model = "doubao-seedream-5.0-lite"

    def __init__(self) -> None:
        self.calls = []

    def image_to_image(self, *, prompt, ref_image, output_path, size):
        self.calls.append({
            "prompt": prompt,
            "ref_image": list(ref_image),
            "output_path": output_path,
            "size": size,
        })
        image = Image.new("RGB", (3072, 2048), (225, 225, 225))
        colors = [(20, 50, 90), (30, 70, 100), (40, 90, 110), (50, 110, 120), (60, 130, 130), (70, 150, 140)]
        for index, color in enumerate(colors):
            left = (index % 3) * 1024 + 160
            upper = (index // 3) * 1024 + 100
            patch = Image.new("RGB", (704, 824), color)
            image.paste(patch, (left, upper))
        image.save(output_path, format="PNG")
        return "https://provider.invalid/performance.png"


class _ReviewClient:
    def __init__(self) -> None:
        self.calls = []

    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        return _qa_payload()


def test_plan_has_six_ordered_cells_bound_to_real_pxx_action_and_prop():
    plan = build_character_performance_plan(_storyboard(), _character())

    assert plan is not None
    assert plan["schema"] == CHARACTER_PERFORMANCE_BOARD_SCHEMA
    assert [cell["cell_id"] for cell in plan["cells"]] == list(PERFORMANCE_CELL_IDS)
    assert [cell["beat_id"] for cell in plan["cells"]] == [
        "S01_P01", "S01_P02", "S01_P01", "S01_P02", "S01_P01", "S01_P02",
    ]
    assert {cell["source_action_unit_id"] for cell in plan["cells"]} == {"AU001", "AU002"}
    assert all(cell["prop_ids"] == ["energy_baton"] for cell in plan["cells"])


def test_generate_board_and_current_pxx_guides_are_exactly_cached(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _ImageClient()
    review_client = _ReviewClient()

    first = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )
    second = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )

    assert first is not None and first["provider_requests"] == 1
    assert second is not None and second["provider_requests"] == 0
    assert len(image_client.calls) == 1
    assert len(review_client.calls) == 1
    assert "no text" in image_client.calls[0]["prompt"]
    assert "wet clothing" in image_client.calls[0]["prompt"]
    assert image_client.calls[0]["size"] == "3072x2048"
    assert validate_character_performance_board(tmp_path, "lead")

    p01 = validate_character_performance_guide(tmp_path, "lead", "S01_P01")
    p02 = validate_character_performance_guide(tmp_path, "lead", "S01_P02")
    assert p01 is not None and p01["cell_ids"] == ["A01", "A03", "A05"]
    assert p02 is not None and p02["cell_ids"] == ["A02", "A04", "A06"]
    assert p01["provider_requests"] == p02["provider_requests"] == 0
    assert set(p01["source_action_unit_ids"]) == {"AU001"}
    assert set(p02["source_action_unit_ids"]) == {"AU002"}

    storyboard = _storyboard()
    attach_performance_guides_to_storyboard(storyboard, [first])
    beats = storyboard["shots"][0]["storyboard_beats"]
    assert all(beat["character_performance_required"] is True for beat in beats)
    assert beats[0]["character_performance_guides"][0]["cell_ids"] == [
        "A01", "A03", "A05"
    ]
    assert beats[1]["character_performance_guides"][0]["cell_ids"] == [
        "A02", "A04", "A06"
    ]


def test_wet_or_damaged_state_alone_does_not_create_performance_board():
    storyboard = {
        "shots": [{
            "id": "S01",
            "shot_intent": "atmosphere",
            "storyboard_beats": [{
                "beat_id": "S01_P01",
                "character_ids": ["lead"],
                "action": "领队站在雨中，风衣已经淋湿并有泥污",
                "source_action_unit_ids": ["AU001"],
                "generation_action_units": [
                    _unit("GAU001", "AU001", "领队站在雨中，风衣已经淋湿并有泥污")
                ],
            }],
        }],
    }
    character = _character()
    character["appearance"]["identity_props"] = []

    assert build_character_performance_plan(storyboard, character) is None


def test_corrupt_or_future_board_receipt_fails_closed(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_ImageClient(),
        review_client=_ReviewClient(),
    )
    receipt_path = tmp_path / "characters/lead/performance_reference_board.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "honcut.character-performance-board.v999"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert not validate_character_performance_board(tmp_path, "lead")
    assert validate_character_performance_guide(tmp_path, "lead", "S01_P01") is None
