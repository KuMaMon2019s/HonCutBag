"""Deterministically pack canonical character views into one reference image."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


CHARACTER_REFERENCE_BOARD_SCHEMA = "honcut.character-reference-board.v1"
CHARACTER_REFERENCE_BOARD_FILENAME = "reference_board.png"
CHARACTER_REFERENCE_BOARD_RECEIPT = "reference_board.json"
CHARACTER_REFERENCE_BOARD_SIZE = (1536, 1536)
CHARACTER_REFERENCE_BOARD_BACKGROUND = (232, 232, 232)
CHARACTER_REFERENCE_VIEWS = (
    "face_closeup",
    "full_body",
    "side",
    "back",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_records(character_dir: Path) -> list[dict[str, str]]:
    paths = {
        view: character_dir / f"{view}.png"
        for view in CHARACTER_REFERENCE_VIEWS
    }
    missing = [
        path.name
        for path in paths.values()
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(
            "incomplete canonical character reference pack: " + ", ".join(missing)
        )

    records: list[dict[str, str]] = []
    for view, path in paths.items():
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid character reference view {path}: {exc}") from exc
        records.append({"view": view, "path": path.name, "sha256": _sha256(path)})
    return records


def _layout() -> list[str]:
    return list(CHARACTER_REFERENCE_VIEWS)


def _cells() -> list[dict[str, Any]]:
    width, height = CHARACTER_REFERENCE_BOARD_SIZE
    cell_width, cell_height = width // 2, height // 2
    return [
        {
            "view": view,
            "row": index // 2,
            "column": index % 2,
            "box": [
                (index % 2) * cell_width,
                (index // 2) * cell_height,
                ((index % 2) + 1) * cell_width,
                ((index // 2) + 1) * cell_height,
            ],
        }
        for index, view in enumerate(CHARACTER_REFERENCE_VIEWS)
    ]


def _valid_existing_board(
    board_path: Path,
    receipt_path: Path,
    *,
    character_id: str,
    sources: list[dict[str, str]],
) -> bool:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            return False
        if receipt.get("schema") != CHARACTER_REFERENCE_BOARD_SCHEMA:
            return False
        if receipt.get("status") != "done":
            return False
        if receipt.get("character_id") != character_id:
            return False
        if receipt.get("image") != board_path.name:
            return False
        if receipt.get("sources") != sources or receipt.get("layout") != _layout():
            return False
        if receipt.get("cells") != _cells():
            return False
        if receipt.get("canvas") != {
            "width": CHARACTER_REFERENCE_BOARD_SIZE[0],
            "height": CHARACTER_REFERENCE_BOARD_SIZE[1],
            "background": "#E8E8E8",
        }:
            return False
        if not board_path.is_file() or receipt.get("image_sha256") != _sha256(board_path):
            return False
        with Image.open(board_path) as image:
            image.verify()
            return image.size == CHARACTER_REFERENCE_BOARD_SIZE
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _atomic_save_png(image: Image.Image, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".png",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.stem}.",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_character_reference_board(
    character_dir: Path,
    *,
    character_id: str | None = None,
) -> Path:
    """Return one audited 2x2 board derived from four canonical views.

    This operation is local and deterministic. It never calls a model or a
    Provider. The four source images remain canonical QA evidence; the board is
    the compact transport asset used by downstream multimodal requests.
    """
    character_dir = Path(character_dir)
    character_id = str(character_id or character_dir.name).strip()
    if not character_id:
        raise ValueError("character_id must not be empty")
    sources = _source_records(character_dir)
    board_path = character_dir / CHARACTER_REFERENCE_BOARD_FILENAME
    receipt_path = character_dir / CHARACTER_REFERENCE_BOARD_RECEIPT
    if _valid_existing_board(
        board_path,
        receipt_path,
        character_id=character_id,
        sources=sources,
    ):
        return board_path

    width, height = CHARACTER_REFERENCE_BOARD_SIZE
    cell_size = (width // 2, height // 2)
    board = Image.new("RGB", CHARACTER_REFERENCE_BOARD_SIZE, CHARACTER_REFERENCE_BOARD_BACKGROUND)
    for index, source in enumerate(sources):
        source_path = character_dir / source["path"]
        with Image.open(source_path) as raw_image:
            contained = ImageOps.contain(
                raw_image.convert("RGB"),
                cell_size,
                method=Image.Resampling.LANCZOS,
            )
        cell_x = (index % 2) * cell_size[0]
        cell_y = (index // 2) * cell_size[1]
        paste_x = cell_x + (cell_size[0] - contained.width) // 2
        paste_y = cell_y + (cell_size[1] - contained.height) // 2
        board.paste(contained, (paste_x, paste_y))

    _atomic_save_png(board, board_path)
    receipt = {
        "schema": CHARACTER_REFERENCE_BOARD_SCHEMA,
        "status": "done",
        "character_id": character_id,
        "image": board_path.name,
        "image_sha256": _sha256(board_path),
        "canvas": {
            "width": width,
            "height": height,
            "background": "#E8E8E8",
        },
        "layout": _layout(),
        "cells": _cells(),
        "sources": sources,
    }
    _atomic_write_json(receipt, receipt_path)
    return board_path


def validate_character_reference_board(
    character_dir: Path,
    *,
    character_id: str | None = None,
) -> bool:
    """Validate an existing board and receipt without regenerating either file."""
    character_dir = Path(character_dir)
    character_id = str(character_id or character_dir.name).strip()
    if not character_id:
        return False
    try:
        sources = _source_records(character_dir)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return _valid_existing_board(
        character_dir / CHARACTER_REFERENCE_BOARD_FILENAME,
        character_dir / CHARACTER_REFERENCE_BOARD_RECEIPT,
        character_id=character_id,
        sources=sources,
    )


def resolve_character_reference_board(
    output_dir: Path,
    character_id: str,
) -> Path | None:
    """Resolve or derive a board from either supported character directory."""
    output_dir = Path(output_dir)
    for character_dir in (
        output_dir / "characters" / character_id,
        output_dir / "characters" / "characters" / character_id,
    ):
        if not character_dir.is_dir():
            continue
        try:
            return ensure_character_reference_board(
                character_dir,
                character_id=character_id,
            )
        except FileNotFoundError:
            continue
    return None


def character_reference_role(path: Path) -> str:
    """Return the Seedream role matching one canonical identity asset."""
    return (
        "character_identity_board_only"
        if Path(path).name == CHARACTER_REFERENCE_BOARD_FILENAME
        else "character_identity_only"
    )
