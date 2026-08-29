"""Run-local Phase 3 performance boards derived from canonical Pxx action lineage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from prompt.seedream_image_prompt import (
    bind_reference_roles,
    image_request_fingerprint,
)
from quality.character_performance_qa import (
    CHARACTER_PERFORMANCE_QA_SCHEMA,
    CharacterPerformanceQAError,
    review_character_performance_board,
)
from quality.character_reference_qa import file_sha256


CHARACTER_PERFORMANCE_BOARD_SCHEMA = "honcut.character-performance-board.v1"
CHARACTER_PERFORMANCE_GUIDE_SCHEMA = "honcut.character-performance-guide.v1"
PERFORMANCE_BOARD_FILENAME = "performance_reference_board.png"
PERFORMANCE_BOARD_RECEIPT = "performance_reference_board.json"
PERFORMANCE_BOARD_QA_RECEIPT = "performance_reference_board_qa.json"
PERFORMANCE_BOARD_SIZE = "3072x2048"
PERFORMANCE_BOARD_PIXEL_SIZE = (3072, 2048)
PERFORMANCE_CELL_IDS = tuple(f"A{index:02d}" for index in range(1, 7))
PERFORMANCE_POSE_VOCABULARY = (
    "combat_ready",
    "attack",
    "evade",
    "block",
    "prop_hold",
    "prop_use",
)
PERFORMANCE_KEY_POSE_PHASES = (
    "起势：保持当前编剧动作开始时的重心和道具关系",
    "执行峰值：清楚展示当前编剧动作的发力、位移或接触关系",
    "落位：只到达当前编剧动作写明的结束姿态，不追加结果",
)
_BEAT_ID_RE = re.compile(r"^S\d+_P\d+$")
_SOURCE_ACTION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_ACTION_MARKERS = (
    "战斗", "攻击", "挥", "砍", "刺", "踢", "打", "冲", "闪避", "躲", "后仰",
    "格挡", "防御", "抵挡", "持", "握", "拿", "举", "使用", "操作", "投掷",
    "fight", "attack", "strike", "kick", "dodge", "evade", "block", "guard",
    "hold", "wield", "use", "operate", "throw",
)


class PerformanceBoardImageClient(Protocol):
    model: str

    def image_to_image(
        self,
        prompt: str,
        ref_image: str | list[str],
        output_path: str,
        size: str,
    ) -> str: ...


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _character_props(character: Mapping[str, Any]) -> list[dict[str, Any]]:
    appearance = character.get("appearance")
    props = appearance.get("identity_props") if isinstance(appearance, Mapping) else []
    return [dict(item) for item in props or [] if isinstance(item, Mapping)]


def _prop_ids_for_action(
    props: list[dict[str, Any]],
    action_text: str,
) -> list[str]:
    folded = action_text.casefold()
    matched = []
    for index, prop in enumerate(props, 1):
        prop_id = str(prop.get("id") or f"prop_{index}").strip()
        prop_name = str(prop.get("name") or "").strip()
        if (
            prop_id.casefold() in folded
            or (prop_name and prop_name.casefold() in folded)
            or prop.get("persistence") == "always"
            or prop.get("attachment_mode") == "body_attached"
        ):
            matched.append(prop_id)
    if not matched and len(props) == 1 and any(marker in folded for marker in _ACTION_MARKERS):
        matched.append(str(props[0].get("id") or "prop_1"))
    return matched


def _generation_units_for_source(
    beat: Mapping[str, Any], source_action_unit_id: str
) -> list[dict[str, Any]]:
    return [
        dict(unit)
        for unit in beat.get("generation_action_units") or []
        if isinstance(unit, Mapping)
        and str(unit.get("source_action_unit_id") or "").strip()
        == source_action_unit_id
    ]


def _unit_action_text(units: list[dict[str, Any]], beat: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for unit in units:
        for field in ("source_fact_echoes", "actions"):
            values = unit.get(field) or []
            if isinstance(values, str):
                values = [values]
            parts.extend(str(value).strip() for value in values if str(value).strip())
    if not parts:
        parts.append(str(beat.get("action") or "").strip())
    seen: set[str] = set()
    return "；".join(part for part in parts if part and not (part in seen or seen.add(part)))


def _pose_category(action_text: str, prop_ids: list[str]) -> str:
    folded = action_text.casefold()
    classifications = (
        ("prop_use", ("使用", "操作", "启动", "发射", "use", "operate", "activate", "fire")),
        ("block", ("格挡", "抵挡", "防御", "架住", "block", "guard", "parry")),
        ("evade", ("闪避", "躲", "后仰", "侧身", "evade", "dodge", "avoid")),
        ("attack", ("攻击", "挥", "砍", "刺", "踢", "击", "attack", "strike", "kick", "swing")),
        ("prop_hold", ("持", "握", "拿", "举", "hold", "wield", "carry")),
        ("combat_ready", ("戒备", "准备", "对峙", "ready", "stance")),
    )
    for category, markers in classifications:
        if any(marker in folded for marker in markers):
            return category
    return "prop_hold" if prop_ids else "combat_ready"


def _eligible_beat(shot: Mapping[str, Any], beat: Mapping[str, Any], props: list[dict[str, Any]]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            beat.get("action"), beat.get("start_state"), beat.get("end_state"),
            shot.get("what"), shot.get("visual"),
        )
    ).casefold()
    body_contract = beat.get("body_action_contract")
    return bool(
        str(shot.get("shot_intent") or "").casefold() == "action"
        or (isinstance(body_contract, Mapping) and body_contract.get("required") is True)
        or any(marker in text for marker in _ACTION_MARKERS)
        or _prop_ids_for_action(props, text)
    )


def build_character_performance_plan(
    storyboard: Mapping[str, Any],
    character: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind six pose cells to existing canonical Pxx/action-unit facts only."""
    character_id = str(character.get("id") or "").strip()
    if not character_id:
        raise ValueError("performance board character ID is missing")
    props = _character_props(character)
    beat_bindings: list[dict[str, Any]] = []
    for shot in storyboard.get("shots") or []:
        if not isinstance(shot, Mapping):
            continue
        shot_id = str(shot.get("id") or "").strip()
        for beat in shot.get("storyboard_beats") or []:
            if not isinstance(beat, Mapping):
                continue
            if character_id not in [str(value) for value in beat.get("character_ids") or []]:
                continue
            if not _eligible_beat(shot, beat, props):
                continue
            beat_id = str(beat.get("beat_id") or "").strip()
            if not _BEAT_ID_RE.fullmatch(beat_id) or not beat_id.startswith(f"{shot_id}_"):
                raise ValueError(f"invalid canonical performance beat ID: {beat_id!r}")
            source_ids = [
                str(value).strip()
                for value in beat.get("source_action_unit_ids") or []
                if str(value).strip()
            ]
            if not source_ids or any(not _SOURCE_ACTION_ID_RE.fullmatch(value) for value in source_ids):
                raise ValueError(f"{beat_id} lacks canonical source action-unit lineage")
            source_bindings = []
            for source_id in source_ids:
                units = _generation_units_for_source(beat, source_id)
                if not units:
                    raise ValueError(
                        f"{beat_id} source action unit {source_id} has no generation lineage"
                    )
                action_text = _unit_action_text(units, beat)
                if not action_text:
                    raise ValueError(f"{beat_id} has an empty authored action")
                source_bindings.append({
                    "source_action_unit_id": source_id,
                    "generation_action_unit_ids": [
                        str(unit.get("unit_id") or "").strip()
                        for unit in units
                        if str(unit.get("unit_id") or "").strip()
                    ],
                    "action_description": action_text,
                    "prop_ids": _prop_ids_for_action(props, action_text),
                })
            beat_bindings.append({
                "beat_id": beat_id,
                "parent_shot_id": shot_id,
                "sources": source_bindings,
            })
    if not beat_bindings:
        return None
    if len(beat_bindings) > len(PERFORMANCE_CELL_IDS):
        raise ValueError(
            f"{character_id} needs {len(beat_bindings)} Pxx performance bindings; "
            "one v1 board can carry at most six"
        )

    candidates = [
        {**source, "beat_id": beat["beat_id"], "parent_shot_id": beat["parent_shot_id"]}
        for beat in beat_bindings
        for source in beat["sources"]
    ]
    selected: list[dict[str, Any]] = []
    for beat in beat_bindings:
        selected.append({
            **beat["sources"][0],
            "beat_id": beat["beat_id"],
            "parent_shot_id": beat["parent_shot_id"],
        })
    cursor = 0
    while len(selected) < len(PERFORMANCE_CELL_IDS):
        selected.append(dict(candidates[cursor % len(candidates)]))
        cursor += 1

    cells = []
    occurrence_by_binding: dict[tuple[str, str], int] = {}
    for index, (cell_id, binding) in enumerate(
        zip(PERFORMANCE_CELL_IDS, selected, strict=True)
    ):
        binding_key = (binding["beat_id"], binding["source_action_unit_id"])
        occurrence = occurrence_by_binding.get(binding_key, 0)
        occurrence_by_binding[binding_key] = occurrence + 1
        category = _pose_category(
            binding["action_description"],
            binding["prop_ids"],
        )
        cells.append({
            "cell_id": cell_id,
            "grid_position": {"row": index // 3 + 1, "column": index % 3 + 1},
            "character_id": character_id,
            "parent_shot_id": binding["parent_shot_id"],
            "beat_id": binding["beat_id"],
            "source_action_unit_id": binding["source_action_unit_id"],
            "generation_action_unit_ids": binding["generation_action_unit_ids"],
            "prop_ids": binding["prop_ids"],
            "pose_category": category,
            "pose_focus": PERFORMANCE_KEY_POSE_PHASES[
                occurrence % len(PERFORMANCE_KEY_POSE_PHASES)
            ],
            "action_description": binding["action_description"],
        })
    return {
        "schema": CHARACTER_PERFORMANCE_BOARD_SCHEMA,
        "character_id": character_id,
        "usage": "run_local_video_motion_reference_only",
        "pose_vocabulary": list(PERFORMANCE_POSE_VOCABULARY),
        "layout": {"rows": 2, "columns": 3, "cell_order": list(PERFORMANCE_CELL_IDS)},
        "cells": cells,
    }


def build_character_performance_prompt(
    character: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    appearance = character.get("appearance")
    styling = appearance.get("synthetic_styling") if isinstance(appearance, Mapping) else None
    return f"""Create one clean 3:2 character performance reference image containing six evenly
spaced full-body poses of the same single character, arranged conceptually as 2 rows x 3 columns.
There are no visible cell dividers: use one seamless neutral light-gray studio background.

Identity is locked by Image 1. Preserve the exact face, pearl bio-ceramic synthetic porcelain
makeup, narrow temple-to-cheek iridescent circuit stripe, luminous iris ring, hair, body
proportions, outfit, colors and character-specific makeup design across all six poses.
Image 2, when present, supplies declared prop geometry/material/color only.

Internal authoring order (never print these IDs in pixels):
{json.dumps(plan['cells'], ensure_ascii=False, sort_keys=True)}

For each position, express only its bound authored action and pose focus. Do not invent an attack,
outcome, injury, wet clothing, torn clothing, dirt or later story state. A declared prop may appear
only in cells whose prop_ids include it and must remain owned by this character.

Pixel prohibitions: no text, no letters, no numbers, no Axx labels, no arrows, no captions, no UI,
no panel borders, no grid lines. Do not add another character. This is a pose reference sheet, not a
storyboard and not a finished cinematic frame.
Synthetic styling contract: {json.dumps(styling, ensure_ascii=False, sort_keys=True)}
"""


def _reference_records(output_dir: Path, character: Mapping[str, Any]) -> list[dict[str, str]]:
    character_id = str(character.get("id") or "").strip()
    character_dir = output_dir / "characters" / character_id
    paths = [character_dir / "reference_board.png"]
    prop_board = character_dir / "prop_detail_board.png"
    if prop_board.is_file():
        paths.append(prop_board)
    records = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"performance board reference missing: {path}")
        records.append({
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(path),
            "kind": (
                "character_identity_board" if path.name == "reference_board.png"
                else "prop_detail_board"
            ),
        })
    return records


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_character_performance_board(
    output_dir: Path,
    character_id: str,
    *,
    expected_plan: Mapping[str, Any] | None = None,
    expected_prompt_sha256: str | None = None,
    expected_model: str | None = None,
    expected_references: list[dict[str, str]] | None = None,
) -> bool:
    output_dir = Path(output_dir)
    character_dir = output_dir / "characters" / character_id
    image_path = character_dir / PERFORMANCE_BOARD_FILENAME
    receipt = _load_json(character_dir / PERFORMANCE_BOARD_RECEIPT)
    qa = _load_json(character_dir / PERFORMANCE_BOARD_QA_RECEIPT)
    if receipt is None or qa is None:
        return False
    if (
        receipt.get("schema") != CHARACTER_PERFORMANCE_BOARD_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("character_id") != character_id
        or receipt.get("image") != PERFORMANCE_BOARD_FILENAME
        or qa.get("schema") != CHARACTER_PERFORMANCE_QA_SCHEMA
        or qa.get("status") != "passed"
        or qa.get("character_id") != character_id
        or qa.get("passed") is not True
    ):
        return False
    plan = receipt.get("plan")
    if not isinstance(plan, dict) or plan.get("schema") != CHARACTER_PERFORMANCE_BOARD_SCHEMA:
        return False
    cells = plan.get("cells")
    if (
        not isinstance(cells, list)
        or any(not isinstance(cell, dict) for cell in cells)
        or [cell.get("cell_id") for cell in cells]
        != list(PERFORMANCE_CELL_IDS)
        or any(cell.get("character_id") != character_id for cell in cells)
    ):
        return False
    plan_hash = _canonical_hash(plan)
    if receipt.get("plan_sha256") != plan_hash or qa.get("plan_sha256") != plan_hash:
        return False
    if expected_plan is not None and dict(expected_plan) != plan:
        return False
    if expected_prompt_sha256 is not None and receipt.get("prompt_sha256") != expected_prompt_sha256:
        return False
    if expected_model is not None and receipt.get("model") != expected_model:
        return False
    references = receipt.get("references")
    if expected_references is not None and references != expected_references:
        return False
    if not isinstance(references, list):
        return False
    for record in references:
        if not isinstance(record, dict):
            return False
        source = output_dir / str(record.get("path") or "")
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            return False
    if not image_path.is_file():
        return False
    image_hash = file_sha256(image_path)
    if receipt.get("image_sha256") != image_hash or qa.get("image_sha256") != image_hash:
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
            if image.size != PERFORMANCE_BOARD_PIXEL_SIZE:
                return False
    except (OSError, ValueError):
        return False
    return True


def _materialize_performance_guides(
    output_dir: Path,
    character_id: str,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    character_dir = output_dir / "characters" / character_id
    board_path = character_dir / PERFORMANCE_BOARD_FILENAME
    board_receipt_path = character_dir / PERFORMANCE_BOARD_RECEIPT
    board_hash = file_sha256(board_path)
    board_receipt_hash = file_sha256(board_receipt_path)
    cells = [dict(cell) for cell in plan["cells"]]
    beat_ids = list(dict.fromkeys(cell["beat_id"] for cell in cells))
    results = []
    with Image.open(board_path) as raw:
        board = raw.convert("RGB")
        cell_width = board.width // 3
        cell_height = board.height // 2
        for beat_id in beat_ids:
            selected = [cell for cell in cells if cell["beat_id"] == beat_id]
            destination_dir = output_dir / "performance_guides" / beat_id
            destination = destination_dir / f"{character_id}.png"
            receipt_path = destination.with_suffix(".json")
            guide = Image.new("RGB", (cell_width * len(selected), cell_height))
            for offset, cell in enumerate(selected):
                position = cell["grid_position"]
                left = (int(position["column"]) - 1) * cell_width
                upper = (int(position["row"]) - 1) * cell_height
                crop = board.crop((left, upper, left + cell_width, upper + cell_height))
                guide.paste(crop, (offset * cell_width, 0))
            _atomic_png(destination, guide)
            receipt = {
                "schema": CHARACTER_PERFORMANCE_GUIDE_SCHEMA,
                "status": "done",
                "usage": "current_pxx_motion_reference_only",
                "character_id": character_id,
                "beat_id": beat_id,
                "image": destination.relative_to(output_dir).as_posix(),
                "image_sha256": file_sha256(destination),
                "source_board": board_path.relative_to(output_dir).as_posix(),
                "source_board_sha256": board_hash,
                "source_board_receipt": board_receipt_path.relative_to(output_dir).as_posix(),
                "source_board_receipt_sha256": board_receipt_hash,
                "cell_ids": [cell["cell_id"] for cell in selected],
                "source_action_unit_ids": [cell["source_action_unit_id"] for cell in selected],
                "prop_ids": list(dict.fromkeys(
                    prop_id for cell in selected for prop_id in cell["prop_ids"]
                )),
                "provider_requests": 0,
            }
            _atomic_json(receipt_path, receipt)
            results.append(receipt)
    return results


def validate_character_performance_guide(
    output_dir: Path,
    character_id: str,
    beat_id: str,
) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    receipt_path = output_dir / "performance_guides" / beat_id / f"{character_id}.json"
    receipt = _load_json(receipt_path)
    if receipt is None:
        return None
    image_path = output_dir / str(receipt.get("image") or "")
    board_path = output_dir / str(receipt.get("source_board") or "")
    board_receipt_path = output_dir / str(receipt.get("source_board_receipt") or "")
    if (
        receipt.get("schema") != CHARACTER_PERFORMANCE_GUIDE_SCHEMA
        or receipt.get("status") != "done"
        or receipt.get("usage") != "current_pxx_motion_reference_only"
        or receipt.get("character_id") != character_id
        or receipt.get("beat_id") != beat_id
        or receipt.get("provider_requests") != 0
        or not isinstance(receipt.get("cell_ids"), list)
        or not receipt["cell_ids"]
        or any(cell_id not in PERFORMANCE_CELL_IDS for cell_id in receipt["cell_ids"])
        or not image_path.is_file()
        or not board_path.is_file()
        or not board_receipt_path.is_file()
        or receipt.get("image_sha256") != file_sha256(image_path)
        or receipt.get("source_board_sha256") != file_sha256(board_path)
        or receipt.get("source_board_receipt_sha256") != file_sha256(board_receipt_path)
        or not validate_character_performance_board(output_dir, character_id)
    ):
        return None
    return receipt


def generate_character_performance_board(
    output_dir: Path,
    storyboard: Mapping[str, Any],
    character: Mapping[str, Any],
    *,
    image_client: PerformanceBoardImageClient,
    review_client: Any,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Generate or exactly reuse one performance board, then derive Pxx guides locally."""
    output_dir = Path(output_dir)
    plan = build_character_performance_plan(storyboard, character)
    if plan is None:
        return None
    character_id = str(character.get("id") or "").strip()
    character_dir = output_dir / "characters" / character_id
    character_dir.mkdir(parents=True, exist_ok=True)
    references = _reference_records(output_dir, character)
    prompt = build_character_performance_prompt(character, plan)
    roles = [
        "character_identity_board_only",
        *(["character_prop_detail_only"] if len(references) > 1 else []),
    ]
    bound_prompt = bind_reference_roles(prompt, roles)
    prompt_hash = hashlib.sha256(bound_prompt.encode("utf-8")).hexdigest()
    resolved_model = str(model or getattr(image_client, "model", "") or "doubao-seedream-5.0-lite")
    request_fingerprint = image_request_fingerprint(
        prompt=bound_prompt,
        model=resolved_model,
        size=PERFORMANCE_BOARD_SIZE,
        reference_image_sha256=[record["sha256"] for record in references],
    )
    if validate_character_performance_board(
        output_dir,
        character_id,
        expected_plan=plan,
        expected_prompt_sha256=prompt_hash,
        expected_model=resolved_model,
        expected_references=references,
    ):
        guides = _materialize_performance_guides(output_dir, character_id, plan)
        return {
            "character_id": character_id,
            "reused": True,
            "board": (character_dir / PERFORMANCE_BOARD_FILENAME).relative_to(output_dir).as_posix(),
            "guides": guides,
            "provider_requests": 0,
        }

    image_path = character_dir / PERFORMANCE_BOARD_FILENAME
    receipt_path = character_dir / PERFORMANCE_BOARD_RECEIPT
    qa_path = character_dir / PERFORMANCE_BOARD_QA_RECEIPT
    pending = {
        "schema": CHARACTER_PERFORMANCE_BOARD_SCHEMA,
        "status": "pending",
        "character_id": character_id,
        "plan": plan,
        "plan_sha256": _canonical_hash(plan),
        "model": resolved_model,
        "size": PERFORMANCE_BOARD_SIZE,
        "prompt_sha256": prompt_hash,
        "request_fingerprint": request_fingerprint,
        "references": references,
        "image": PERFORMANCE_BOARD_FILENAME,
    }
    _atomic_json(receipt_path, pending)
    image_client.image_to_image(
        prompt=bound_prompt,
        ref_image=[str(output_dir / record["path"]) for record in references],
        output_path=str(image_path),
        size=PERFORMANCE_BOARD_SIZE,
    )
    try:
        with Image.open(image_path) as generated:
            generated.verify()
            if generated.size != PERFORMANCE_BOARD_PIXEL_SIZE:
                raise CharacterPerformanceQAError(
                    f"{character_id} performance board has invalid size {generated.size}"
                )
    except (OSError, ValueError) as exc:
        raise CharacterPerformanceQAError(
            f"{character_id} performance board is not a valid image"
        ) from exc
    appearance = character.get("appearance")
    styling = appearance.get("synthetic_styling") if isinstance(appearance, Mapping) else None
    qa_result = review_character_performance_board(
        review_client,
        image_path,
        character_id=character_id,
        cells=plan["cells"],
        synthetic_styling=styling if isinstance(styling, dict) else None,
    )
    image_hash = file_sha256(image_path)
    qa_receipt = {
        **qa_result,
        "status": "passed" if qa_result["passed"] else "failed",
        "character_id": character_id,
        "image": PERFORMANCE_BOARD_FILENAME,
        "image_sha256": image_hash,
        "plan_sha256": _canonical_hash(plan),
    }
    _atomic_json(qa_path, qa_receipt)
    if not qa_result["passed"]:
        _atomic_json(receipt_path, {**pending, "status": "failed", "image_sha256": image_hash})
        raise CharacterPerformanceQAError(
            f"{character_id} performance board failed blocking pixel QA"
        )
    complete = {
        **pending,
        "status": "passed",
        "image_sha256": image_hash,
        "qa_receipt": PERFORMANCE_BOARD_QA_RECEIPT,
        "qa_receipt_sha256": file_sha256(qa_path),
    }
    _atomic_json(receipt_path, complete)
    guides = _materialize_performance_guides(output_dir, character_id, plan)
    return {
        "character_id": character_id,
        "reused": False,
        "board": image_path.relative_to(output_dir).as_posix(),
        "guides": guides,
        "provider_requests": 1,
    }


def generate_performance_reference_boards(
    output_dir: Path,
    storyboard: Mapping[str, Any],
    characters: list[Mapping[str, Any]],
    *,
    image_client: PerformanceBoardImageClient,
    review_client: Any,
) -> list[dict[str, Any]]:
    results = []
    for character in characters:
        result = generate_character_performance_board(
            output_dir,
            storyboard,
            character,
            image_client=image_client,
            review_client=review_client,
        )
        if result is not None:
            results.append(result)
    return results


def attach_performance_guides_to_storyboard(
    storyboard: dict[str, Any],
    generated_boards: list[dict[str, Any]],
) -> None:
    """Persist canonical run-local guide provenance on its owning Pxx beat."""
    by_beat: dict[str, list[dict[str, Any]]] = {}
    for board in generated_boards:
        for receipt in board.get("guides") or []:
            if not isinstance(receipt, dict):
                raise ValueError("performance guide receipt must be a mapping")
            beat_id = str(receipt.get("beat_id") or "").strip()
            by_beat.setdefault(beat_id, []).append({
                "kind": CHARACTER_PERFORMANCE_GUIDE_SCHEMA,
                "usage": "current_pxx_motion_reference_only",
                "character_id": receipt["character_id"],
                "beat_id": beat_id,
                "image": receipt["image"],
                "image_sha256": receipt["image_sha256"],
                "receipt": str(Path(receipt["image"]).with_suffix(".json")),
                "cell_ids": list(receipt["cell_ids"]),
                "source_action_unit_ids": list(
                    receipt["source_action_unit_ids"]
                ),
                "prop_ids": list(receipt["prop_ids"]),
                "source_board": receipt["source_board"],
                "source_board_sha256": receipt["source_board_sha256"],
                "source_board_receipt": receipt["source_board_receipt"],
                "source_board_receipt_sha256": (
                    receipt["source_board_receipt_sha256"]
                ),
            })
    observed_beats: set[str] = set()
    for shot in storyboard.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for beat in shot.get("storyboard_beats") or []:
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or "").strip()
            guides = sorted(
                by_beat.get(beat_id, []),
                key=lambda item: item["character_id"],
            )
            beat["character_performance_required"] = bool(guides)
            beat["character_performance_guides"] = guides
            if guides:
                observed_beats.add(beat_id)
    unknown = sorted(set(by_beat) - observed_beats)
    if unknown:
        raise ValueError(
            "performance guides reference unknown Pxx beats: " + ", ".join(unknown)
        )
