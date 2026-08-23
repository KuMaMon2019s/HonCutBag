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

CINEMATIC_FIRST_FRAME_SCHEMA = "honcut.cinematic-first-frame.v1"
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
        ├─ character_refs/      (face_closeup/full_body/identity_detail/variant_*.png)
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
                    reference_paths = [
                        char_dir / "face_closeup.png",
                        char_dir / "full_body.png",
                        char_dir / "identity_detail.png",
                        *sorted(char_dir.glob("variant_*.png")),
                    ]
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
    """Resolve character display names and aliases to ids, preserving unknown values."""
    characters_json = Path(output_dir) / "CHARACTERS.json"
    try:
        chars_data = json.loads(characters_json.read_text(encoding="utf-8"))
    except Exception:
        return list(raw_ids)

    characters = chars_data.get("characters", [])
    valid_ids = {char.get("id") for char in characters if char.get("id")}
    name_to_id = {}
    for char in characters:
        char_id = char.get("id")
        if not char_id:
            continue
        for value in (char.get("name"), *char.get("aliases", [])):
            if value:
                name_to_id[str(value).casefold()] = char_id
    return [
        raw_id
        if raw_id in valid_ids
        else name_to_id.get(str(raw_id).casefold(), raw_id)
        for raw_id in raw_ids
    ]


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
    action_text = " ".join(
        str(shot_meta.get(key) or "")
        for key in ("visual", "what", "action_description", "generation_actions", "prompt")
    ).casefold()
    for char_id in _detect_shot_characters(output_dir, shot_meta):
        character_name = character_names.get(char_id, char_id)
        identity_props = character_identity_props.get(char_id, [])
        identity_detail_active = any(
            isinstance(item, dict)
            and (
                item.get("persistence") == "always"
                or item.get("attachment_mode") == "body_attached"
                or bool(str(item.get("name") or "").strip())
                and str(item.get("name") or "").casefold() in action_text
                or (
                    str(character_name).casefold() in action_text
                    and any(
                        marker in action_text
                        for marker in ("拍摄", "跟拍", "手持", "使用", "操作", "记录")
                    )
                )
            )
            for item in identity_props
        )
        char_dir = output_dir / "characters" / char_id
        if not char_dir.exists():
            char_dir = output_dir / "characters" / "characters" / char_id
        reference_paths = [
            char_dir / "face_closeup.png",
            char_dir / "full_body.png",
            char_dir / "identity_detail.png",
            *sorted(char_dir.glob("variant_*.png")),
        ]
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
                        "identity_detail"
                        if reference_path.name == "identity_detail.png"
                        else "character_identity"
                    ),
                    "bind_subject": reference_path.name != "identity_detail.png",
                    "identity_props": identity_props,
                    "identity_detail_active": (
                        identity_detail_active
                        if reference_path.name == "identity_detail.png"
                        else False
                    ),
                    "reference_description": (
                        f"{character_name}的面部特写"
                        if reference_path.name == "face_closeup.png"
                        else f"{character_name}的全身照"
                        if reference_path.name == "full_body.png"
                        else (
                            f"{character_name}的身份道具与材质细节板；仅锁定声明道具的几何、"
                            "颜色、材质、标记和佩挂关系，不把板上孤立道具当成新主体"
                        )
                        if reference_path.name == "identity_detail.png"
                        else f"{character_name}的变体图（{reference_path.stem}）"
                    ),
                })
    return references


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
        if face_number is not None:
            role_parts.append(f"<主体{subject_number}>的面部特征参考图片{face_number}（大头照）")
        if body_number is not None:
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
    return f"元素参考：{instruction}{prompt_text}"


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
            {"type": "image_url", "image_url": {"url": "https://..."}, "role": "first_frame", "priority": "high"}
        ]

    i2v uses one first frame; FLF2V uses an explicit first and last frame.
    A legacy Phantom request is upgraded to strict i2v transport whenever a
    Phase 4 cinematic frame exists.  Phantom's separate character references
    remain only as a compatibility fallback for projects without that frame.

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
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})

    # Every route uses the Phase 4 cinematic frame as a strict frame.  In
    # particular, a legacy Phantom request must not downgrade it to a generic
    # reference_image or mix in separate character media: either behavior lets
    # the provider redraw the approved start state and can reintroduce solid
    # actors into a flat shadow-puppet shot. Director/PREVIS pixels are rejected
    # at this boundary even when an old continuity plan points at them.
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
    if shot_frame_path.exists() and shot_frame_path.stat().st_size > 1024:
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
        image_assets.append({
            "path": shot_frame_path,
            "role": "first_frame",
            "priority": "high",
            "bind_subject": False,
            "reference_description": (
                f"{frame_label}，用于锁定本生成片段的构图、角色站位、"
                "场景结构、项目美术风格、时间天气和光影；该资产已经过"
                "无文字、无箭头、无分格的像素洁净检查"
            ),
        })

    if strategy == "flf2v":
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
    elif strategy == "phantom":
        character_assets = collect_character_reference_assets(output_dir, shot_meta)
        expected_characters = _detect_shot_characters(output_dir, shot_meta)
        if expected_characters and not character_assets:
            raise FileNotFoundError(
                "Phantom character references missing for shot "
                f"{shot_id}; expected face_closeup.png, full_body.png, identity_detail.png, or variant_*.png"
            )
        if image_assets:
            # The approved Phase 4 render already resolves identity, costume,
            # medium, composition, and continuity. Seedance cannot legally mix
            # first_frame with reference_image, and doing so would weaken the
            # exact-start-frame contract, so keep identity text-only here.
            strategy = "i2v"
            text_item = next(
                (item for item in content if item.get("type") == "text"), None
            )
            if text_item is not None:
                text_item["text"] = inject_flf2v_identity_lock(
                    output_dir,
                    shot_meta,
                    text_item["text"],
                )
            print(
                "  [assets] phantom -> i2v: Phase 4 cinematic frame promoted "
                "to strict first_frame; character references kept text-only"
            )
        elif not character_assets:
            raise FileNotFoundError(
                "Phantom references missing for shot "
                f"{shot_id}; expected a cinematic first frame or character reference"
            )
        else:
            # Compatibility only: legacy projects that have not generated a
            # Phase 4 cinematic frame still use Phantom identity references.
            image_assets = character_assets

    max_reference_images = shot_meta.get("_max_reference_images")
    if strategy == "phantom" and generation_actions:
        # Dense identity packs over-constrain motion. For action, retain one
        # identity image per character plus the composition frame; native
        # continuation adds its ordered tail anchors outside this budget.
        active_detail_count = sum(
            asset.get("reference_kind") == "identity_detail"
            and asset.get("identity_detail_active") is True
            for asset in image_assets
        )
        motion_budget = (
            len(_detect_shot_characters(output_dir, shot_meta))
            + 1
            + min(active_detail_count, 1)
        )
        motion_budget = max(2, min(4, motion_budget))
        max_reference_images = (
            motion_budget
            if max_reference_images is None
            else min(int(max_reference_images), motion_budget)
        )
    if max_reference_images is not None:
        max_reference_images = max(0, int(max_reference_images))
        if len(image_assets) > max_reference_images:
            # Continuation reserves three provider image slots for ordered tail
            # anchors. Keep the composition frame and distribute the remaining
            # identity budget across characters in rounds (face, body, detail, variant)
            # so one character cannot consume every slot.
            composition_assets = [
                asset
                for asset in image_assets
                if asset.get("reference_kind") == "cinematic_composition"
            ][:1]
            character_assets = [
                asset
                for asset in image_assets
                if asset.get("reference_kind") != "cinematic_composition"
            ]
            character_budget = max(0, max_reference_images - len(composition_assets))
            grouped: dict[str, list[dict]] = {}
            for asset in character_assets:
                grouped.setdefault(str(asset.get("char_id") or ""), []).append(asset)
            if strategy == "phantom" and generation_actions:
                # Face close-ups bias an action generation toward portrait
                # framing and can push the second fighter out of frame. Prefer
                # a full-body identity anchor for choreography, then a state
                # variant, and keep the face crop only as a last fallback.
                def action_reference_rank(asset: dict) -> int:
                    name = Path(asset.get("path", "")).name
                    if name == "full_body.png":
                        return 0
                    if name == "identity_detail.png":
                        return 1
                    if name.startswith("variant_"):
                        return 2
                    if name == "face_closeup.png":
                        return 3
                    return 4

                for assets in grouped.values():
                    assets.sort(key=action_reference_rank)
            selected_characters = []
            round_index = 0
            while len(selected_characters) < character_budget:
                added = False
                for assets in grouped.values():
                    if round_index < len(assets):
                        selected_characters.append(assets[round_index])
                        added = True
                        if len(selected_characters) >= character_budget:
                            break
                if not added:
                    break
                round_index += 1
            image_assets = [*selected_characters, *composition_assets]
            print(
                "  [assets] continuation reference budget: "
                f"trimmed to {len(image_assets)}/{max_reference_images} images"
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
    except ImportError:
        print(f"  [assets] ⚠ tos_uploader not available, skipping image upload")
        return content
    
    for asset in image_assets:
        try:
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
                    # Fallback: use raw bytes if fit_to_aspect fails (e.g. non-image file)
                    img_data = asset["path"].read_bytes()
            else:
                img_data = asset["path"].read_bytes()

            tos_url = tos_uploader.upload_image(img_data, "image/png")
            if tos_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": tos_url},
                    "role": asset["role"],
                    "priority": asset["priority"],
                })
                uploaded_count += 1
                if asset["role"] == "reference_image":
                    uploaded_reference_descriptions.append(asset)
            else:
                print(f"  [assets] ⚠ TOS upload failed for {asset['path'].name}, skipping")
        except Exception as e:
            print(f"  [assets] ⚠ Failed to upload {asset['path'].name}: {e}")
    
    if strategy == "phantom" and uploaded_reference_descriptions and content:
        text_item = next(
            (item for item in content if item.get("type") == "text"), None
        )
        if text_item is not None:
            text_item["text"] = inject_reference_instruction(
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
