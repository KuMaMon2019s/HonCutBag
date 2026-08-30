"""Shared fixtures that build production-valid test run boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.character_reference_board import ensure_character_reference_board
from utils.canonical_visual_contracts import (
    CANONICAL_VISUAL_CONTRACT_FILENAME,
    persist_canonical_visual_contract,
)


@pytest.fixture
def canonical_run_contract():
    """Persist the same Phase 1 visual contract consumed by later phases."""

    def persist(
        output_dir: str | Path,
        characters_data: dict,
        *,
        requested_policy: str = "source_derived",
    ) -> tuple[dict, dict]:
        root = Path(output_dir)
        projected, contract = persist_canonical_visual_contract(
            root,
            copy.deepcopy(characters_data),
            requested_policy=requested_policy,
        )
        (root / "CHARACTERS.json").write_text(
            json.dumps(projected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        storyboard_path = root / "STORYBOARD.json"
        if storyboard_path.is_file():
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            storyboard["canonical_visual_contract"] = (
                CANONICAL_VISUAL_CONTRACT_FILENAME
            )
            storyboard["canonical_visual_contract_sha256"] = contract[
                "contract_sha256"
            ]
            storyboard_path.write_text(
                json.dumps(storyboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return projected, contract

    return persist


@pytest.fixture
def character_reference_board():
    """Create one production-valid four-view identity board without a Provider."""

    def write(output_dir: str | Path, character_id: str, *, color_seed: int = 1) -> Path:
        character_dir = Path(output_dir) / "characters" / character_id
        character_dir.mkdir(parents=True, exist_ok=True)
        for index, view in enumerate(
            ("face_closeup", "full_body", "side", "back"),
            start=1,
        ):
            value = (color_seed * 37 + index * 41) % 256
            Image.new(
                "RGB",
                (512, 512),
                (value, (value + 61) % 256, (value + 127) % 256),
            ).save(character_dir / f"{view}.png", format="PNG")
        return ensure_character_reference_board(
            character_dir,
            character_id=character_id,
        )

    return write
