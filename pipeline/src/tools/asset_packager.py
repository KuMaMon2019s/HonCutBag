#!/usr/bin/env python3
"""
Asset packager for Phase 5 video generation.

Packages character references, storyboard images, and metadata into a zip file
for upload to Bridge API. Falls back to base64 list if zip upload fails.
"""

import base64
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

from utils.storyboard_motion_policy import apply_storyboard_motion_policy
from utils.video_generation_contracts import ensure_video_generation_contract
from utils.video_capabilities import SEEDANCE_2_CAPABILITIES
from tools.character_reference_board import (
    ensure_character_reference_board,
    resolve_character_reference_board,
)

CINEMATIC_FIRST_FRAME_SCHEMA = "honcut.cinematic-first-frame.v1"
STORYBOARD_NARRATIVE_GUIDE_SCHEMA = "honcut.storyboard-narrative-guide.v3"
STORYBOARD_NARRATIVE_GUIDE_USAGE = "phase6_story_narrative_guide_not_output_pixels"
STORYBOARD_NARRATIVE_GUIDE_RENDERER = (
    "honcut.identity-neutral-story-guide-renderer.v2"
)
STORYBOARD_GUIDE_POSE_CONTRACT_SCHEMA = (
    "honcut.storyboard-guide-pose-contract.v2"
)
CHARACTER_PERFORMANCE_GUIDE_SCHEMA = "honcut.character-performance-guide.v2"
CHARACTER_PERFORMANCE_GUIDE_USAGE = "current_pxx_motion_reference_only"
PREVIS_FRAME_PATH_PARTS = frozenset({
    "director_panels",
    "storyboard_beats",
    "shot_storyboards",
    "storyboard_groups",
    "storyboard_bridges",
    "phase5_reference_boards",
})


def _frame_receipt(path: Path) -> dict[str, Any] | None:
    receipt_path = path.with_suffix(".json")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _assert_video_frame_provenance(path: Path, declared_kind: Any = None) -> None:
    """Reject director/PREVIS pixels at the final video transport boundary."""
    lowered_parts = {part.casefold() for part in path.parts}
    if lowered_parts & PREVIS_FRAME_PATH_PARTS or path.name.casefold() in {
        "director_storyboard.png",
        "storyboard.png",
    }:
        raise ValueError(
            f"PREVIS/director board cannot be used as a video frame: {path}"
        )
    kind = str(declared_kind or "").strip()
    if kind and kind != CINEMATIC_FIRST_FRAME_SCHEMA:
        raise ValueError(
            f"video frame has non-cinematic provenance kind {kind}: {path}"
        )
    receipt = _frame_receipt(path)
    # A path named ``storyboard_images/Sxx.png`` is not trustworthy by itself:
    # Phase 2 deliberately writes a PREVIS compatibility placeholder there and
    # Phase 4 replaces it.  Require the Phase 4 receipt at every video transport
    # boundary so a missing/mutated sidecar fails closed instead of silently
    # reviving the historical contamination route.
    if not receipt or receipt.get("status") != "done":
        raise ValueError(f"cinematic video frame has no completed receipt: {path}")
    if receipt.get("kind") != CINEMATIC_FIRST_FRAME_SCHEMA:
        raise ValueError(
            f"video frame receipt marks the asset as non-cinematic: {path}"
        )
    expected = str(receipt.get("image_sha256") or "")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != observed:
        raise ValueError(f"cinematic video frame receipt hash mismatch: {path}")
    if receipt.get("previs_reference_images") != []:
        raise ValueError(f"cinematic video frame contains PREVIS lineage: {path}")


def _assert_narrative_guide_provenance(
    path: Path,
    *,
    declared_kind: Any,
    declared_usage: Any,
    declared_cell_ids: Any,
    declared_sha256: Any,
    declared_renderer: Any,
    declared_pose_contract_schema: Any,
    declared_pose_policy_sha256: Any,
    declared_pose_contracts_sha256: Any,
    declared_pose_fingerprints: Any,
    declared_source_pixel_usage: Any,
    declared_semantic_payload_sha256: Any,
    declared_source_board: Any,
    declared_source_board_sha256: Any,
    declared_receipt: Any,
    declared_authority_roles: Any,
    declared_non_authority_roles: Any,
    declared_beat_id: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate an annotated story guide without treating it as a video frame."""
    if str(declared_kind or "") != STORYBOARD_NARRATIVE_GUIDE_SCHEMA:
        raise ValueError("storyboard narrative guide has an unknown provenance kind")
    if str(declared_usage or "") != STORYBOARD_NARRATIVE_GUIDE_USAGE:
        raise ValueError("storyboard narrative guide has an unsafe usage contract")
    receipt_path = Path(str(declared_receipt or ""))
    if not receipt_path.is_absolute():
        receipt_path = output_dir / receipt_path
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("storyboard narrative guide receipt is unreadable") from exc
    source_board = Path(str(declared_source_board or ""))
    if not source_board.is_absolute():
        source_board = output_dir / source_board
    if not path.is_file() or not source_board.is_file():
        raise ValueError("storyboard narrative guide or source board is missing")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    source_observed = hashlib.sha256(source_board.read_bytes()).hexdigest()
    cell_ids = [str(value) for value in (declared_cell_ids or [])]
    beat_id = str(declared_beat_id or "").strip()
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("storyboard narrative guide needs ordered unique Gxx cells")
    semantic_payload = receipt.get("semantic_payload")
    if not isinstance(semantic_payload, dict):
        raise ValueError("storyboard narrative guide semantic payload is missing")
    semantic_observed = hashlib.sha256(
        json.dumps(
            semantic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    authority_roles = [str(value) for value in (declared_authority_roles or [])]
    non_authority_roles = [
        str(value) for value in (declared_non_authority_roles or [])
    ]
    pose_fingerprints = [
        str(value) for value in (declared_pose_fingerprints or [])
    ]
    semantic_cells = semantic_payload.get("cells")
    if not isinstance(semantic_cells, list):
        raise ValueError("storyboard narrative guide semantic cells are missing")
    semantic_pose_fingerprints = [
        str((cell.get("pose_contract") or {}).get("pose_fingerprint") or "")
        for cell in semantic_cells
        if isinstance(cell, dict)
    ]
    semantic_pose_contracts = [
        cell.get("pose_contract")
        for cell in semantic_cells
        if isinstance(cell, dict)
    ]
    semantic_pose_contracts_sha256 = hashlib.sha256(
        json.dumps(
            semantic_pose_contracts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if len(semantic_pose_contracts) != len(cell_ids):
        raise ValueError("storyboard narrative guide pose coverage is incomplete")
    for cell_id, contract in zip(
        cell_ids, semantic_pose_contracts, strict=True
    ):
        if not isinstance(contract, dict):
            raise ValueError("storyboard narrative guide pose contract is missing")
        unhashed = dict(contract)
        stored_contract_sha256 = str(unhashed.pop("contract_sha256", ""))
        observed_contract_sha256 = hashlib.sha256(
            json.dumps(
                unhashed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            contract.get("schema") != STORYBOARD_GUIDE_POSE_CONTRACT_SCHEMA
            or contract.get("pose_policy_sha256")
            != str(declared_pose_policy_sha256 or "")
            or contract.get("cell_id") != cell_id
            or contract.get("secondary_beat_id") != beat_id
            or stored_contract_sha256 != observed_contract_sha256
        ):
            raise ValueError(
                "storyboard narrative guide pose contract binding is invalid"
            )
    if (
        str(declared_sha256 or "") != observed
        or str(declared_renderer or "") != STORYBOARD_NARRATIVE_GUIDE_RENDERER
        or str(declared_pose_contract_schema or "")
        != STORYBOARD_GUIDE_POSE_CONTRACT_SCHEMA
        or re.fullmatch(r"[0-9a-f]{64}", str(declared_pose_policy_sha256 or ""))
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(declared_pose_contracts_sha256 or "")
        )
        is None
        or str(declared_pose_contracts_sha256 or "")
        != semantic_pose_contracts_sha256
        or len(pose_fingerprints) != len(cell_ids)
        or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pose_fingerprints)
        or str(declared_source_pixel_usage or "") != "none"
        or str(declared_semantic_payload_sha256 or "") != semantic_observed
        or str(declared_source_board_sha256 or "") != source_observed
        or receipt.get("kind") != STORYBOARD_NARRATIVE_GUIDE_SCHEMA
        or receipt.get("version") != 3
        or receipt.get("usage") != STORYBOARD_NARRATIVE_GUIDE_USAGE
        or receipt.get("renderer") != STORYBOARD_NARRATIVE_GUIDE_RENDERER
        or receipt.get("pose_contract_schema")
        != STORYBOARD_GUIDE_POSE_CONTRACT_SCHEMA
        or receipt.get("pose_contract_schema") != declared_pose_contract_schema
        or receipt.get("pose_policy_sha256") != declared_pose_policy_sha256
        or receipt.get("pose_contracts_sha256")
        != declared_pose_contracts_sha256
        or receipt.get("pose_fingerprints") != pose_fingerprints
        or semantic_payload.get("pose_contract_schema")
        != STORYBOARD_GUIDE_POSE_CONTRACT_SCHEMA
        or semantic_payload.get("pose_policy_sha256")
        != declared_pose_policy_sha256
        or semantic_payload.get("pose_contracts_sha256")
        != declared_pose_contracts_sha256
        or semantic_payload.get("pose_fingerprints") != pose_fingerprints
        or semantic_pose_fingerprints != pose_fingerprints
        or receipt.get("source_pixel_usage") != "none"
        or receipt.get("semantic_payload_sha256") != semantic_observed
        or receipt.get("status") != "done"
        or receipt.get("beat_id") != beat_id
        or receipt.get("primary_shot_id") != beat_id.split("_P", 1)[0]
        or receipt.get("image") != str(path.relative_to(output_dir))
        or receipt.get("source_board") != str(declared_source_board or "")
        or receipt.get("image_sha256") != observed
        or receipt.get("source_board_sha256") != source_observed
        or receipt.get("cell_ids") != cell_ids
        or receipt.get("authority_roles") != authority_roles
        or receipt.get("non_authority_roles") != non_authority_roles
        or authority_roles != [
            "narrative_order",
            "action_direction",
            "camera_motion",
            "spatial_relationship",
        ]
        or "character_identity" not in non_authority_roles
        or int(receipt.get("provider_request_count") or 0) != 0
    ):
        raise ValueError("storyboard narrative guide receipt/hash/cell binding mismatch")
    return receipt


def _assert_performance_guide_provenance(
    path: Path,
    *,
    declared: Any,
    declared_beat_id: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate one current-Pxx pose guide and every source-board hash."""
    if not isinstance(declared, dict):
        raise ValueError("character performance guide declaration is invalid")
    character_id = str(declared.get("character_id") or "").strip()
    beat_id = str(declared_beat_id or "").strip()
    if (
        declared.get("kind") != CHARACTER_PERFORMANCE_GUIDE_SCHEMA
        or declared.get("usage") != CHARACTER_PERFORMANCE_GUIDE_USAGE
        or declared.get("beat_id") != beat_id
        or not character_id
    ):
        raise ValueError("character performance guide has an unsafe contract")
    from phases.phase3.performance_reference_board import (
        validate_character_performance_guide,
    )

    receipt = validate_character_performance_guide(
        output_dir,
        character_id,
        beat_id,
    )
    if receipt is None:
        raise ValueError("character performance guide receipt is missing or stale")
    expected = {
        "image": receipt["image"],
        "image_sha256": receipt["image_sha256"],
        "cell_ids": receipt["cell_ids"],
        "source_action_unit_ids": receipt["source_action_unit_ids"],
        "prop_ids": receipt["prop_ids"],
        "source_board": receipt["source_board"],
        "source_board_sha256": receipt["source_board_sha256"],
        "source_board_receipt": receipt["source_board_receipt"],
        "source_board_receipt_sha256": receipt[
            "source_board_receipt_sha256"
        ],
    }
    if any(declared.get(key) != value for key, value in expected.items()):
        raise ValueError("character performance guide declaration does not match receipt")
    receipt_path = output_dir / str(declared.get("receipt") or "")
    if (
        path != output_dir / receipt["image"]
        or receipt_path != path.with_suffix(".json")
        or not receipt_path.is_file()
    ):
        raise ValueError("character performance guide path binding is invalid")
    return receipt


def package_shot_assets(
    output_dir: Path,
    shot_id: str,
    shot_meta: dict,
) -> Tuple[Optional[str], List[str]]:
    """
    Package shot assets into a zip file with metadata.
    
    Args:
        output_dir: Project output directory
        shot_id: Shot identifier (e.g., "S01")
        shot_meta: Shot metadata dict with prompt, duration, seed, width, height
    
    Returns:
        Tuple of (zip_path_or_None, base64_list)
        - zip_path: Path to zip file if assets found, None otherwise
        - base64_list: List of base64-encoded images (sorted by priority, max 9)
    
    Zip structure:
        assets.zip
        ├─ meta.json
        ├─ character_refs/      (static reference_board or canonical views)
        └─ shot_frames/         (Phase 4 cinematic storyboard_images/{shot_id}.png)
    
    meta.json structure:
        {
            "shot_id": "S01",
            "prompt": "...",
            "duration": 5,
            "seed": 12345,
            "width": 1280,
            "height": 720,
            "images": [
                {"path": "character_refs/char_id_face_closeup.png", "role": "identity", "priority": 1},
                ...
            ]
        }
    """
    output_dir = Path(output_dir)
    
    # Collect assets with metadata
    assets = []
    
    # 1. Character reference images (priority 1, role=identity)
    # Try both directory structures: characters/{char_id}/ and characters/characters/{char_id}/
    char_bases = [output_dir / "characters", output_dir / "characters" / "characters"]
    seen_char_dirs = set()
    for char_base in char_bases:
        if char_base.exists():
            for char_dir in char_base.iterdir():
                if char_dir.is_dir() and char_dir.name not in seen_char_dirs:
                    seen_char_dirs.add(char_dir.name)
                    try:
                        identity_reference = ensure_character_reference_board(
                            char_dir,
                            character_id=char_dir.name,
                        )
                    except FileNotFoundError:
                        identity_reference = None
                    reference_paths = (
                        [identity_reference]
                        if identity_reference is not None
                        else [
                            char_dir / "face_closeup.png",
                            char_dir / "full_body.png",
                        ]
                    )
                    for reference_path in reference_paths:
                        if reference_path.exists():
                            assets.append({
                                "src_path": reference_path,
                                "zip_path": f"character_refs/{char_dir.name}_{reference_path.name}",
                                "role": "identity",
                                "priority": 1,
                            })
    
    # 2. Shot frame from storyboard_images (priority 1, role=composition)
    shot_frame_path = output_dir / "storyboard_images" / f"{shot_id}.png"
    if shot_frame_path.exists():
        _assert_video_frame_provenance(shot_frame_path)
        assets.append({
            "src_path": shot_frame_path,
            "zip_path": f"shot_frames/{shot_id}.png",
            "role": "composition",
            "priority": 1,
        })
    
    if not assets:
        return None, []
    
    # Build meta.json
    meta = {
        "shot_id": shot_id,
        "prompt": shot_meta.get("prompt", ""),
        "duration": shot_meta.get("duration"),
        "seed": shot_meta.get("seed"),
        "width": shot_meta.get("width"),
        "height": shot_meta.get("height"),
        "images": [
            {
                "path": asset["zip_path"],
                "role": asset["role"],
                "priority": asset["priority"],
            }
            for asset in assets
        ],
    }
    
    # Create zip file
    zip_path = output_dir / "shots" / shot_id / "assets.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write meta.json
        zf.writestr("meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
        
        # Write image files
        for asset in assets:
            zf.write(asset["src_path"], asset["zip_path"])
    
    # Build base64 fallback list (sorted by priority, then by role, max 9)
    # Priority order: identity (character) > cinematic composition frame.
    sorted_assets = sorted(assets, key=lambda a: (a["priority"], a["role"]))
    base64_list = []
    
    for asset in sorted_assets[:9]:
        try:
            img_data = asset["src_path"].read_bytes()
            b64 = base64.b64encode(img_data).decode('utf-8')
            base64_list.append(b64)
        except Exception as e:
            print(f"  ⚠ Failed to encode {asset['src_path']}: {e}")
    
    return str(zip_path), base64_list


def _resolve_char_ids(output_dir: Path, raw_ids: List[str]) -> List[str]:
    """Resolve source/entity labels to every canonical instance ID."""
    characters_json = Path(output_dir) / "CHARACTERS.json"
    try:
        chars_data = json.loads(characters_json.read_text(encoding="utf-8"))
    except Exception:
        return list(raw_ids)

    characters = chars_data.get("characters", [])
    valid_ids = {char.get("id") for char in characters if char.get("id")}
    name_to_ids: dict[str, list[str]] = {}
    for char in characters:
        char_id = char.get("id")
        if not char_id:
            continue
        for value in (
            char.get("entity_id"),
            char.get("name"),
            *char.get("aliases", []),
        ):
            if value:
                bucket = name_to_ids.setdefault(str(value).casefold(), [])
                if char_id not in bucket:
                    bucket.append(char_id)
    resolved: list[str] = []
    for raw_id in raw_ids:
        candidates = (
            [raw_id]
            if raw_id in valid_ids
            else name_to_ids.get(str(raw_id).casefold(), [raw_id])
        )
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def _detect_shot_characters(
    output_dir: Path,
    shot_meta: dict,
) -> List[str]:
    """
    Detect which characters appear in a shot.

    Resolution order:
        1. Structured who/characters names from the storyboard
        2. Explicit char_ids in shot_meta (from pipeline_runner)
        3. associate_assets in shot_meta (e.g. ["char:lin_xiao", "char:chen_yang"])
        4. Name matching against CHARACTERS.json + prompt text

    Returns:
        List of character ids (e.g. ["lin_xiao", "chen_yang"])
    """
    # 1. Structured storyboard identity is authoritative. In particular, an
    # explicit who=[] must not fall through to stale inferred char assets, and
    # a canonical display name must beat an obsolete associate_assets alias.
    if "who" in shot_meta or "characters" in shot_meta:
        structured = shot_meta.get("who") or shot_meta.get("characters") or []
        if not isinstance(structured, list):
            structured = [structured] if structured else []
        if not structured:
            return []
        resolved = _resolve_char_ids(output_dir, list(structured))
        if resolved:
            return resolved

    # 2. Explicit char_ids
    explicit = shot_meta.get("_char_ids")
    if explicit:
        return _resolve_char_ids(output_dir, list(explicit))

    # 3. associate_assets
    associate = shot_meta.get("associate_assets", [])
    if associate:
        char_ids = [
            aid[5:].split(":")[0]
            for aid in associate
            if isinstance(aid, str) and aid.startswith("char:")
        ]
        if char_ids:
            char_ids = _resolve_char_ids(output_dir, char_ids)
            return char_ids

    # 4. Prompt name matching against CHARACTERS.json
    prompt_text = shot_meta.get("prompt", "")
    characters_json = output_dir / "CHARACTERS.json"
    if not prompt_text or not characters_json.exists():
        return []

    try:
        import json as _json
        chars_data = _json.loads(characters_json.read_text())
    except Exception:
        return []

    prompt_lower = prompt_text.lower()
    matched = []
    for char in chars_data.get("characters", []):
        char_id = char.get("id", "")
        char_name = char.get("name", "")
        # Match by display name (case-insensitive) or char_id
        if char_name and char_name.lower() in prompt_lower:
            matched.append(char_id)
        elif char_id and char_id.lower().replace("_", " ") in prompt_lower:
            matched.append(char_id)
        elif char_id and char_id.lower() in prompt_lower:
            matched.append(char_id)
    return matched


def collect_character_reference_assets(
    output_dir: Path,
    shot_meta: dict,
) -> List[dict]:
    """Return Phantom references in the exact order used for 图片N numbering."""
    output_dir = Path(output_dir)
    character_names = {}
    characters_json = output_dir / "CHARACTERS.json"
    if characters_json.exists():
        characters_data = json.loads(characters_json.read_text(encoding="utf-8"))
        character_names = {
            character.get("id"): character.get("name") or character.get("id")
            for character in characters_data.get("characters", [])
            if character.get("id")
        }
        character_definitions = {
            character.get("id"): character.get("prompt_definition", "")
            for character in characters_data.get("characters", [])
            if character.get("id")
        }
        character_identity_props = {
            character.get("id"): (
                character.get("appearance", {}).get("identity_props", [])
                if isinstance(character.get("appearance"), dict)
                else []
            )
            for character in characters_data.get("characters", [])
            if character.get("id")
        }
    else:
        character_definitions = {}
        character_identity_props = {}

    references = []
    for char_id in _detect_shot_characters(output_dir, shot_meta):
        character_name = character_names.get(char_id, char_id)
        identity_props = character_identity_props.get(char_id, [])
        char_dir = output_dir / "characters" / char_id
        if not char_dir.exists():
            char_dir = output_dir / "characters" / "characters" / char_id
        identity_reference = resolve_character_reference_board(output_dir, char_id)
        reference_paths = [identity_reference] if identity_reference is not None else []
        for reference_path in reference_paths:
            if reference_path.exists() and reference_path.stat().st_size > 1024:
                references.append({
                    "path": reference_path,
                    "char_id": char_id,
                    "character_name": character_name,
                    "prompt_definition": character_definitions.get(char_id, ""),
                    "role": "reference_image",
                    "priority": "high",
                    "reference_kind": (
                        "character_identity_board"
                        if reference_path.name == "reference_board.png"
                        else "character_identity"
                    ),
                    "bind_subject": True,
                    "identity_props": identity_props,
                    "reference_description": (
                        f"{character_name}的四视图身份参考板（面部特写、正面全身、侧面全身、背面全身）"
                        if reference_path.name == "reference_board.png"
                        else f"{character_name}的面部特写"
                        if reference_path.name == "face_closeup.png"
                        else f"{character_name}的全身照"
                        if reference_path.name == "full_body.png"
                        else f"{character_name}的静态身份参考"
                    ),
                })
    return references


_CANONICAL_PROMPT_MARKER = (
    "[CANONICAL_VISUAL_CONTRACT — HIGHEST IDENTITY AUTHORITY]"
)
_CANONICAL_PROMPT_SUFFIX = "其他图片只承担其声明职责，不得改写这些事实。"


def _insert_after_canonical_authority(prompt_text: str, instruction: str) -> str:
    """Keep the Phase 1 authority first while adding downstream instructions."""
    if prompt_text.startswith(_CANONICAL_PROMPT_MARKER):
        boundary = prompt_text.find(_CANONICAL_PROMPT_SUFFIX)
        if boundary >= 0:
            boundary += len(_CANONICAL_PROMPT_SUFFIX)
            return (
                f"{prompt_text[:boundary]}\n{instruction}"
                f"{prompt_text[boundary:]}"
            )
    return f"{instruction}{prompt_text}"


def inject_reference_instruction(prompt_text: str, descriptions: List[Any]) -> str:
    """Explain every reference image and bind only identity images as subjects."""
    if not descriptions:
        return prompt_text
    normalized = [
        item if isinstance(item, dict) else {"reference_description": str(item), "char_id": str(index)}
        for index, item in enumerate(descriptions, start=1)
    ]
    references = "，".join(
        f"图片{index}为{item['reference_description']}"
        for index, item in enumerate(normalized, start=1)
    )
    subject_bindings = []
    named_subject_bindings = []
    subject_numbers = {}
    references_by_subject = {}
    for image_number, item in enumerate(normalized, start=1):
        if item.get("bind_subject", True) is False:
            continue
        char_id = item.get("char_id") or str(image_number)
        references_by_subject.setdefault(char_id, []).append((image_number, item))
        if char_id in subject_numbers:
            continue
        subject_number = len(subject_numbers) + 1
        subject_numbers[char_id] = (image_number, subject_number)
        character_name = str(item.get("character_name") or char_id).strip()
        if character_name:
            named_subject_bindings.append(
                f"{character_name}=<主体{subject_number}>（图片{image_number}）"
            )
        definition = item.get("prompt_definition") or (
            f"将{{图片N}}中的[{item.get('reference_description', char_id)}]定义为{{主体N}}"
        )
        subject_bindings.append(
            definition.replace("{图片N}", f"图片{image_number}").replace("{主体N}", f"<主体{subject_number}>")
        )

    reference_role_bindings = []
    for char_id, (first_image_number, subject_number) in subject_numbers.items():
        subject_references = references_by_subject.get(char_id, [])
        board_number = next(
            (
                image_number
                for image_number, item in subject_references
                if item.get("reference_kind") == "character_identity_board"
                or "reference_board" in str(item.get("path") or "")
                or "四视图" in str(item.get("reference_description") or "")
            ),
            None,
        )
        face_number = next(
            (
                image_number
                for image_number, item in subject_references
                if any(
                    token in str(item.get("reference_description") or "")
                    for token in ("面部特写", "脸部特写", "大头照")
                )
                or "face_closeup" in str(item.get("path") or "")
            ),
            None,
        )
        body_number = next(
            (
                image_number
                for image_number, item in subject_references
                if any(
                    token in str(item.get("reference_description") or "")
                    for token in ("全身照", "全身参考")
                )
                or "full_body" in str(item.get("path") or "")
            ),
            None,
        )
        role_parts = []
        if board_number is not None:
            role_parts.append(
                f"<主体{subject_number}>的面部、妆造、身体比例及侧背轮廓统一参考图片{board_number}（同一张四视图身份板）"
            )
        elif face_number is not None:
            role_parts.append(f"<主体{subject_number}>的面部特征参考图片{face_number}（大头照）")
        if board_number is None and body_number is not None:
            role_parts.append(
                f"<主体{subject_number}>的妆造和身体比例参考图片{body_number}（全身照）"
            )
        if not role_parts:
            role_parts.append(f"<主体{subject_number}>参考图片{first_image_number}")
        reference_role_bindings.append("，".join(role_parts))

    image_replacements = iter(subject_numbers.values())
    def replace_image_placeholder(match: re.Match) -> str:
        try:
            image_number, _ = next(image_replacements)
        except StopIteration:
            return match.group(0)
        return f"图片{image_number}"

    prompt_text = re.sub(r"\{图片N\}", replace_image_placeholder, prompt_text)
    subject_replacements = iter(subject_numbers.values())
    def replace_subject_placeholder(match: re.Match) -> str:
        try:
            _, subject_number = next(subject_replacements)
        except StopIteration:
            return match.group(0)
        return f"<主体{subject_number}>"
    prompt_text = re.sub(r"\{主体N\}", replace_subject_placeholder, prompt_text)
    binding_text = "；".join([*reference_role_bindings, *subject_bindings])
    instruction = references + "。"
    if binding_text:
        instruction += f"{binding_text}。生成时严格保持参考图中角色的外观一致。"
    if named_subject_bindings:
        instruction += (
            "角色名与主体编号硬绑定（全镜不得互换身份、造型、服装、颜色、动作或空间角色）："
            + "；".join(named_subject_bindings)
            + "。"
        )
    if "元素参考" in prompt_text:
        return re.sub(
            r"元素参考(?:声明)?\s*[：:]?\s*",
            f"元素参考：{instruction}",
            prompt_text,
            count=1,
        )
    return _insert_after_canonical_authority(
        prompt_text,
        f"元素参考：{instruction}",
    )


def inject_omni_reference_instruction(
    prompt_text: str,
    descriptions: List[Any],
) -> str:
    """Bind numbered references and request the cinematic image as frame one."""
    prompt_text = inject_reference_instruction(prompt_text, descriptions)
    cinematic_number = next(
        (
            index
            for index, asset in enumerate(descriptions, start=1)
            if isinstance(asset, dict)
            and asset.get("reference_kind") == "cinematic_composition"
        ),
        None,
    )
    if cinematic_number is None:
        return prompt_text
    instruction = (
        "全模态参考首帧合同："
        f"首帧为图片{cinematic_number}。"
        f"图片{cinematic_number}锁定开场构图、角色站位、场景结构、"
        "项目美术风格、时间天气和光影；其余图片只按各自编号锁定"
        "角色身份、服装、身体比例、道具或明确声明的参考职责。"
        f"不得把图片{cinematic_number}中的单一姿态冻结为全片动作。"
    )
    return _insert_after_canonical_authority(prompt_text, instruction)


def inject_flf2v_identity_lock(
    output_dir: Path,
    shot_meta: dict,
    prompt_text: str,
) -> str:
    """Add canonical character traits to FLF2V prompts without reference media."""
    from utils.pixel_text_policy import strip_pixel_text_identity_markers

    char_ids = _detect_shot_characters(output_dir, shot_meta)
    if not char_ids:
        return prompt_text

    characters_json = Path(output_dir) / "CHARACTERS.json"
    try:
        characters_data = json.loads(characters_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"  [assets] ⚠ cannot load FLF2V identity traits: {error}")
        return prompt_text

    characters_by_id = {
        character.get("id"): character
        for character in characters_data.get("characters", [])
        if character.get("id")
    }
    identity_lines = []
    trait_labels = (
        ("hair", "hair"),
        ("face", "face"),
        ("clothing", "clothing"),
        ("build", "body build"),
        ("distinguishing", "distinguishing features"),
    )
    for char_id in char_ids:
        character = characters_by_id.get(char_id)
        if not character:
            continue
        appearance = character.get("appearance") or {}
        if not isinstance(appearance, dict):
            continue
        traits = []
        for field, label in trait_labels:
            if not appearance.get(field):
                continue
            sanitized = strip_pixel_text_identity_markers(appearance[field])
            if sanitized:
                traits.append(f"{label}: {sanitized}")
        if traits:
            name = character.get("name") or char_id
            identity_lines.append(f"{name} — " + "; ".join(traits))

    if not identity_lines:
        return prompt_text
    lock = (
        "[identity-lock: text-only; no reference media]\n"
        + "\n".join(identity_lines)
        + "\nKeep each named character's sex, age, facial structure, hairstyle, "
        "clothing, body proportions, and distinguishing features unchanged in every frame; "
        "do not masculinize, feminize, age-shift, body-swap, or change build."
    )
    return f"{lock}\n{prompt_text}" if prompt_text else lock


def build_content_for_shot(
    output_dir: Path,
    shot_id: str,
    shot_meta: dict,
) -> List[dict]:
    """
    Build Bridge content[] for the shot's deterministic generation strategy.

    Args:
        output_dir: Project output directory
        shot_id: Shot identifier (e.g., "S01")
        shot_meta: Shot metadata dict with prompt, duration, seed, width, height

    Returns:
        List of content items for Bridge /generate API:
        [
            {"type": "text", "text": "shot prompt"},
            {"type": "image_url", "image_url": {"url": "https://..."}, "role": "reference_image", "priority": "high"}
        ]

    Standard Seedance generation uses numbered ``reference_image`` inputs.
    Canonical character boards precede the Phase 4 cinematic frame, and the
    prompt binds the frame's final media index as the requested first frame.
    FLF2V alone keeps explicit ``first_frame`` / ``last_frame`` endpoint control.

    Legacy paths (zip/base64) are preserved for backward compatibility but not
    used in the primary content[] workflow.

    Each image is uploaded to TOS and replaced with a signed URL.
    """
    output_dir = Path(output_dir)
    content = []

    # 1. Text prompt (always first)
    prompt_text = apply_storyboard_motion_policy(shot_meta.get("prompt", ""))
    requested_strategy = shot_meta.get("gen_strategy", "i2v")
    if requested_strategy not in {"flf2v", "phantom", "i2v"}:
        requested_strategy = "i2v"
    strategy = requested_strategy
    if strategy == "flf2v":
        prompt_text = inject_flf2v_identity_lock(output_dir, shot_meta, prompt_text)
    generation_actions = shot_meta.get("generation_actions") or []
    if generation_actions:
        prompt_text = (
            f"{prompt_text}\n[motion-priority] Reference images constrain identity, costume, "
            "environment, and boundary composition only. They do not constrain the reference pose. "
            "Move the subjects through the ordered action contract with clear body displacement; "
            "do not hold or gently animate the input pose."
        ).strip()
    from utils.body_action_contracts import body_action_prompt

    choreography_prompt = body_action_prompt(shot_meta)
    if choreography_prompt and choreography_prompt not in prompt_text:
        prompt_text = f"{prompt_text}\n{choreography_prompt}".strip()
    try:
        characters_data = json.loads(
            (output_dir / "CHARACTERS.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        characters_data = {}
    prompt_text = ensure_video_generation_contract(
        prompt_text,
        shot_meta,
        characters_data,
    )
    from utils.canonical_visual_contracts import (
        load_canonical_visual_contract,
        render_canonical_visual_prompt_contract,
    )

    canonical_contract = load_canonical_visual_contract(
        output_dir,
        characters_data=characters_data,
    )
    canonical_prompt = render_canonical_visual_prompt_contract(
        canonical_contract,
        character_ids=_detect_shot_characters(output_dir, shot_meta),
    )
    selected_character_ids = set(_detect_shot_characters(output_dir, shot_meta))
    selected_contract_characters = [
        character
        for character in canonical_contract.get("characters", [])
        if character.get("character_id") in selected_character_ids
        or character.get("entity_id") in selected_character_ids
        or any(
            instance.get("instance_id") in selected_character_ids
            for instance in (character.get("instances") or [])
            if isinstance(instance, dict)
        )
    ]
    synthetic_contract = ""
    if selected_contract_characters and all(
        character.get("visual_identity_policy")
        == "synthetic_stylized_character_v3"
        for character in selected_contract_characters
    ):
        from utils.privacy_visual_policy import synthetic_stylized_prompt_contract

        candidate_contract = synthetic_stylized_prompt_contract()
        if candidate_contract not in prompt_text:
            synthetic_contract = candidate_contract
    prompt_text = "\n".join(
        part for part in (canonical_prompt, synthetic_contract, prompt_text) if part
    ).strip()
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})

    # The Phase 4 cinematic frame is a numbered all-modal reference for ordinary
    # generation. Identity boards precede it; the prompt binds its final index
    # as the requested first frame.
    # Strict frame roles are reserved for FLF2V endpoints. Director/PREVIS
    # pixels are rejected at this boundary even when an old continuity plan
    # points at them.
    storyboard_images_dir = output_dir / "storyboard_images"
    if not storyboard_images_dir.exists():
        print(
            f"  [assets] ⚠ storyboard_images directory missing: {storyboard_images_dir}; "
            "run Phase 4 to generate cinematic first frames"
        )
    frame_override = shot_meta.get("_storyboard_frame_path")
    shot_frame_path = (
        Path(str(frame_override))
        if frame_override and Path(str(frame_override)).is_absolute()
        else output_dir / str(frame_override)
        if frame_override
        else storyboard_images_dir / f"{shot_id}.png"
    )
    image_assets = []
    cinematic_assets = []
    include_cinematic_frame = shot_meta.get("_include_cinematic_frame", True) is not False
    if (
        include_cinematic_frame
        and shot_frame_path.exists()
        and shot_frame_path.stat().st_size > 1024
    ):
        _assert_video_frame_provenance(
            shot_frame_path,
            shot_meta.get("_storyboard_frame_kind"),
        )
        beat_label = shot_meta.get("_storyboard_beat_id")
        frame_label = (
            f"{beat_label}成片质感第一帧"
            if beat_label
            else f"{shot_id}成片质感第一帧"
        )
        cinematic_assets.append({
            "path": shot_frame_path,
            "role": (
                "first_frame"
                if strategy == "flf2v"
                else "reference_image"
            ),
            "priority": "high",
            "bind_subject": False,
            "mandatory": True,
            "reference_kind": "cinematic_composition",
            "reference_description": (
                f"{frame_label}，用于锁定本生成片段的构图、角色站位、"
                "场景结构、项目美术风格、时间天气和光影；该资产已经过"
                "无文字、无箭头、无分格的像素洁净检查"
            ),
        })

    if strategy == "flf2v":
        image_assets.extend(cinematic_assets)
        end_frame_path = storyboard_images_dir / f"{shot_id}_end.png"
        if end_frame_path.exists() and end_frame_path.stat().st_size > 1024:
            image_assets.append({
                "path": end_frame_path,
                "role": "last_frame",
                "priority": "high",
                "bind_subject": False,
                "reference_description": (
                    f"{shot_id}分镜尾帧，用于锁定镜头结束时的动作、构图和光影"
                ),
            })
        else:
            raise FileNotFoundError(
                f"FLF2V end frame missing or too small: {end_frame_path}"
            )
    elif strategy == "i2v":
        image_assets.extend(cinematic_assets)
        if not image_assets:
            raise FileNotFoundError(
                f"I2V cinematic first frame missing or too small: {shot_frame_path}"
            )
    elif strategy == "phantom":
        character_assets = collect_character_reference_assets(output_dir, shot_meta)
        expected_characters = _detect_shot_characters(output_dir, shot_meta)
        by_character: dict[str, list[dict]] = {}
        for asset in character_assets:
            by_character.setdefault(str(asset.get("char_id") or ""), []).append(asset)
        missing_characters = [
            character_id
            for character_id in expected_characters
            if not by_character.get(character_id)
        ]
        if missing_characters:
            raise FileNotFoundError(
                "Phantom character references missing for shot "
                f"{shot_id}: {', '.join(missing_characters)}"
            )
        guide_value = str(
            shot_meta.get("_storyboard_narrative_guide_path") or ""
        ).strip()
        if not guide_value:
            # Compatibility for direct/legacy callers that do not declare a
            # current authored beat. Preserve their historical reference order;
            # for action, keep only one body-capable identity image per actor.
            if generation_actions:
                selected_characters = []
                for character_id in expected_characters:
                    candidates = by_character.get(character_id, [])
                    selected = next(
                        (
                            asset
                            for asset in candidates
                            if asset.get("reference_kind")
                            == "character_identity_board"
                        ),
                        next(
                            (
                                asset
                                for asset in candidates
                                if Path(asset.get("path", "")).name
                                == "full_body.png"
                            ),
                            candidates[0] if candidates else None,
                        ),
                    )
                    if selected is not None:
                        selected_characters.append({**selected, "mandatory": True})
                image_assets.extend(selected_characters)
            else:
                image_assets.extend(character_assets)
            image_assets.extend(cinematic_assets)
        else:
            primary_character_assets: list[dict] = []
            for character_id in expected_characters:
                candidates = by_character.get(character_id, [])
                if not candidates:
                    continue
                primary = next(
                    (
                        asset
                        for asset in candidates
                        if asset.get("reference_kind")
                        == "character_identity_board"
                    ),
                    candidates[0],
                )
                primary_character_assets.append({**primary, "mandatory": True})
            image_assets.extend(primary_character_assets)
            image_assets.extend(cinematic_assets)
            guide_path = Path(guide_value)
            if not guide_path.is_absolute():
                guide_path = output_dir / guide_path
            guide_receipt = _assert_narrative_guide_provenance(
                guide_path,
                declared_kind=shot_meta.get("_storyboard_narrative_guide_kind"),
                declared_usage=shot_meta.get("_storyboard_narrative_guide_usage"),
                declared_cell_ids=shot_meta.get(
                    "_storyboard_narrative_guide_cell_ids"
                ),
                declared_sha256=shot_meta.get(
                    "_storyboard_narrative_guide_sha256"
                ),
                declared_renderer=shot_meta.get(
                    "_storyboard_narrative_guide_renderer"
                ),
                declared_pose_contract_schema=shot_meta.get(
                    "_storyboard_narrative_guide_pose_contract_schema"
                ),
                declared_pose_policy_sha256=shot_meta.get(
                    "_storyboard_narrative_guide_pose_policy_sha256"
                ),
                declared_pose_contracts_sha256=shot_meta.get(
                    "_storyboard_narrative_guide_pose_contracts_sha256"
                ),
                declared_pose_fingerprints=shot_meta.get(
                    "_storyboard_narrative_guide_pose_fingerprints"
                ),
                declared_source_pixel_usage=shot_meta.get(
                    "_storyboard_narrative_guide_source_pixel_usage"
                ),
                declared_semantic_payload_sha256=shot_meta.get(
                    "_storyboard_narrative_guide_semantic_payload_sha256"
                ),
                declared_source_board=shot_meta.get(
                    "_storyboard_narrative_guide_source_board"
                ),
                declared_source_board_sha256=shot_meta.get(
                    "_storyboard_narrative_guide_source_board_sha256"
                ),
                declared_receipt=shot_meta.get(
                    "_storyboard_narrative_guide_receipt"
                ),
                declared_authority_roles=shot_meta.get(
                    "_storyboard_narrative_guide_authority_roles"
                ),
                declared_non_authority_roles=shot_meta.get(
                    "_storyboard_narrative_guide_non_authority_roles"
                ),
                declared_beat_id=shot_meta.get("_storyboard_beat_id"),
                output_dir=output_dir,
            )
            cell_ids = list(guide_receipt["cell_ids"])
            narrative_asset = {
                "path": guide_path,
                "role": "reference_image",
                "priority": "high",
                "bind_subject": False,
                "mandatory": True,
                "reference_kind": "storyboard_narrative_guide",
                "narrative_cell_ids": cell_ids,
                "narrative_beat_id": guide_receipt["beat_id"],
                "authority_roles": list(guide_receipt["authority_roles"]),
                "non_authority_roles": list(
                    guide_receipt["non_authority_roles"]
                ),
                "semantic_payload_sha256": guide_receipt[
                    "semantic_payload_sha256"
                ],
                "pose_contract_schema": guide_receipt["pose_contract_schema"],
                "pose_policy_sha256": guide_receipt["pose_policy_sha256"],
                "pose_contracts_sha256": guide_receipt[
                    "pose_contracts_sha256"
                ],
                "pose_fingerprints": list(guide_receipt["pose_fingerprints"]),
                "reference_description": (
                    f"{guide_receipt['beat_id']}剧情导航图，只按"
                    + "→".join(cell_ids)
                    + "理解剧情、动作方向、运镜和空间关系；图中序号、"
                    "文字、箭头、边框、网格和指示标识不得渲染进视频"
                ),
            }
            performance_guides = shot_meta.get("_character_performance_guides") or []
            performance_required = bool(
                shot_meta.get("_character_performance_required")
            )
            if performance_required != bool(performance_guides):
                raise ValueError(
                    f"{shot_id} performance-guide requirement is incomplete"
                )
            for declared in performance_guides:
                guide_value = str(
                    declared.get("image") if isinstance(declared, dict) else ""
                ).strip()
                performance_path = Path(guide_value)
                if not performance_path.is_absolute():
                    performance_path = output_dir / performance_path
                performance_receipt = _assert_performance_guide_provenance(
                    performance_path,
                    declared=declared,
                    declared_beat_id=shot_meta.get("_storyboard_beat_id"),
                    output_dir=output_dir,
                )
                performance_cells = list(performance_receipt["cell_ids"])
                image_assets.append({
                    "path": performance_path,
                    "role": "reference_image",
                    "priority": "high",
                    "bind_subject": False,
                    "mandatory": True,
                    "reference_kind": "character_performance_guide",
                    "char_id": performance_receipt["character_id"],
                    "performance_beat_id": performance_receipt["beat_id"],
                    "performance_cell_ids": performance_cells,
                    "performance_source_action_unit_ids": list(
                        performance_receipt["source_action_unit_ids"]
                    ),
                    "performance_prop_ids": list(
                        performance_receipt["prop_ids"]
                    ),
                    "performance_source_board_sha256": (
                        performance_receipt["source_board_sha256"]
                    ),
                    "reference_description": (
                        f"{performance_receipt['beat_id']}中"
                        f"{performance_receipt['character_id']}的当前动作姿态图，仅按"
                        + "→".join(performance_cells)
                        + "理解同一角色的姿态与道具握持；不得生成克隆、分栏、"
                        "网格或板中其他动作"
                    ),
                })
            image_assets.append(narrative_asset)
        if not image_assets:
            raise FileNotFoundError(
                "Phantom references missing for shot "
                f"{shot_id}; expected cinematic, character, or narrative-guide media"
            )
        print(
            "  [assets] phantom all-modal reference: "
            f"{sum(asset.get('mandatory') is True for asset in image_assets)} "
            "mandatory image(s)"
        )

    max_reference_images = shot_meta.get("_max_reference_images")
    provider_budget = int(SEEDANCE_2_CAPABILITIES.max_reference_images or 9)
    effective_budget = (
        provider_budget
        if max_reference_images is None
        else min(provider_budget, max(0, int(max_reference_images)))
    )
    mandatory_assets = [asset for asset in image_assets if asset.get("mandatory")]
    optional_assets = [asset for asset in image_assets if not asset.get("mandatory")]
    if len(mandatory_assets) > effective_budget:
        raise ValueError(
            f"{shot_id} requires {len(mandatory_assets)} mandatory reference images, "
            f"above the available Seedance budget {effective_budget}"
        )
    optional_capacity = effective_budget - len(mandatory_assets)
    selected_optional_ids = {
        id(asset) for asset in optional_assets[:optional_capacity]
    }
    image_assets = [
        asset
        for asset in image_assets
        if asset.get("mandatory") or id(asset) in selected_optional_ids
    ]
    if len(optional_assets) > optional_capacity:
        print(
            "  [assets] omitted optional supplemental references: "
            f"{len(optional_assets) - optional_capacity}"
        )

    route_label = (
        f"{requested_strategy}->{strategy}"
        if requested_strategy != strategy
        else strategy
    )
    print(f"  [assets] {route_label}: images_used={len(image_assets)}")
    
    # Upload each image to TOS and add to content
    uploaded_count = 0
    uploaded_reference_descriptions = []
    try:
        from clients import tos_uploader
    except ImportError as exc:
        raise RuntimeError(
            "TOS uploader is required for every Seedance image input"
        ) from exc
    
    for asset in image_assets:
        # M5: fit first_frame/last_frame to video aspect ratio before upload (no stretching)
        video_w = shot_meta.get("width", 1280)
        video_h = shot_meta.get("height", 720)

        if asset["role"] in {"first_frame", "last_frame"}:
            try:
                import tempfile
                from pipeline_runner import fit_to_aspect
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                fit_to_aspect(asset["path"], video_w, video_h, tmp_path)
                img_data = tmp_path.read_bytes()
                tmp_path.unlink()
            except Exception:
                # Preserve the original validated image when deterministic fitting
                # is unavailable; the upload itself must still succeed.
                img_data = asset["path"].read_bytes()
        else:
            img_data = asset["path"].read_bytes()

        tos_url = tos_uploader.upload_image_required(img_data, "image/png")
        content_item = {
            "type": "image_url",
            "image_url": {"url": tos_url},
            "role": asset["role"],
            "priority": asset["priority"],
            "_reference_kind": asset.get("reference_kind"),
            "_reference_description": asset.get("reference_description"),
            "_character_id": asset.get("char_id"),
            "_narrative_cell_ids": asset.get("narrative_cell_ids"),
            "_narrative_beat_id": asset.get("narrative_beat_id"),
            "_authority_roles": asset.get("authority_roles"),
            "_non_authority_roles": asset.get("non_authority_roles"),
            "_semantic_payload_sha256": asset.get(
                "semantic_payload_sha256"
            ),
            "_performance_beat_id": asset.get("performance_beat_id"),
            "_performance_cell_ids": asset.get("performance_cell_ids"),
            "_performance_source_action_unit_ids": asset.get(
                "performance_source_action_unit_ids"
            ),
            "_performance_prop_ids": asset.get("performance_prop_ids"),
            "_performance_source_board_sha256": asset.get(
                "performance_source_board_sha256"
            ),
            "_mandatory_reference": asset.get("mandatory") is True,
            "_reference_path": (
                str(asset["path"].relative_to(output_dir))
                if asset["path"].is_relative_to(output_dir)
                else str(asset["path"])
            ),
            "_reference_sha256": hashlib.sha256(
                asset["path"].read_bytes()
            ).hexdigest(),
        }
        content.append(content_item)
        uploaded_count += 1
        if asset["role"] == "reference_image":
            uploaded_reference_descriptions.append(asset)
    
    if strategy in {"phantom", "i2v"} and uploaded_reference_descriptions and content:
        text_item = next(
            (item for item in content if item.get("type") == "text"), None
        )
        if text_item is not None:
            text_item["text"] = inject_omni_reference_instruction(
                text_item["text"], uploaded_reference_descriptions
            )

    if strategy in {"i2v", "flf2v"} and content:
        frame_descriptions = [
            asset for asset in image_assets
            if asset.get("role") in {"first_frame", "last_frame"}
        ]
        text_item = next(
            (item for item in content if item.get("type") == "text"), None
        )
        if text_item is not None and frame_descriptions:
            frame_contract = "；".join(
                f"图片{index}为{asset['reference_description']}"
                for index, asset in enumerate(frame_descriptions, start=1)
            )
            text_item["text"] = f"成片首帧参考说明：{frame_contract}。{text_item['text']}"

    print(f"  [assets] 上传 {uploaded_count} 张参考图到 TOS")
    return content


if __name__ == "__main__":
    # Quick test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create test structure
        char_dir = output_dir / "characters" / "characters" / "char_001"
        char_dir.mkdir(parents=True)
        (char_dir / "front.png").write_bytes(b"fake_front")
        (char_dir / "side.png").write_bytes(b"fake_side")
        
        storyboard_img_dir = output_dir / "storyboard_images"
        storyboard_img_dir.mkdir()
        (storyboard_img_dir / "S01.png").write_bytes(b"fake_shot_frame")
        
        (output_dir / "storyboard.png").write_bytes(b"fake_storyboard")
        
        shot_meta = {
            "prompt": "Test shot",
            "duration": 5,
            "seed": 12345,
            "width": 1280,
            "height": 720,
        }
        
        # Test old package_shot_assets
        zip_path, base64_list = package_shot_assets(output_dir, "S01", shot_meta)
        
        if zip_path is None:
            print("✗ No zip created (no assets found)")
            exit(1)
        
        print(f"✓ Created zip: {zip_path}")
        print(f"✓ Base64 list length: {len(base64_list)}")
        
        # Verify zip contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            print(f"✓ Zip contents: {zf.namelist()}")
            meta = json.loads(zf.read("meta.json"))
            print(f"✓ Meta: {json.dumps(meta, indent=2)}")
        
        # Test new build_content_for_shot (will fail TOS upload without credentials, but structure should be correct)
        print("\n--- Testing build_content_for_shot ---")
        content = build_content_for_shot(output_dir, "S01", shot_meta)
        print(f"✓ Content items: {len(content)}")
        for i, item in enumerate(content):
            if item["type"] == "text":
                print(f"  [{i}] text: {item['text'][:50]}...")
            else:
                print(f"  [{i}] image_url: role={item.get('role')}, priority={item.get('priority')}")
