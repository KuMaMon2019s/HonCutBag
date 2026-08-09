#!/usr/bin/env python3
"""
Asset packager for Phase 5 video generation.

Packages character references, storyboard images, and metadata into a zip file
for upload to Bridge API. Falls back to base64 list if zip upload fails.
"""

import base64
import json
import re
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple


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
        ├─ character_refs/      (face_closeup/full_body/variant_*.png)
        ├─ shot_frames/         (storyboard_images/{shot_id}.png)
        └─ storyboard/          (storyboard.png)
    
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
        assets.append({
            "src_path": shot_frame_path,
            "zip_path": f"shot_frames/{shot_id}.png",
            "role": "composition",
            "priority": 1,
        })
    
    # 3. Storyboard.png (priority 2, role=style)
    storyboard_path = output_dir / "storyboard.png"
    if storyboard_path.exists():
        assets.append({
            "src_path": storyboard_path,
            "zip_path": "storyboard/storyboard.png",
            "role": "style",
            "priority": 2,
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
    # Priority order: identity (character) > composition (shot frame) > style (storyboard)
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
    """Resolve character display names to ids, preserving unknown values."""
    characters_json = Path(output_dir) / "CHARACTERS.json"
    try:
        chars_data = json.loads(characters_json.read_text(encoding="utf-8"))
    except Exception:
        return list(raw_ids)

    characters = chars_data.get("characters", [])
    valid_ids = {char.get("id") for char in characters if char.get("id")}
    name_to_id = {
        char.get("name"): char.get("id")
        for char in characters
        if char.get("name") and char.get("id")
    }
    return [raw_id if raw_id in valid_ids else name_to_id.get(raw_id, raw_id) for raw_id in raw_ids]


def _detect_shot_characters(
    output_dir: Path,
    shot_meta: dict,
) -> List[str]:
    """
    Detect which characters appear in a shot.

    Resolution order:
        1. Explicit char_ids in shot_meta (from pipeline_runner)
        2. associate_assets in shot_meta (e.g. ["char:lin_xiao", "char:chen_yang"])
        3. Name matching against CHARACTERS.json + prompt text

    Returns:
        List of character ids (e.g. ["lin_xiao", "chen_yang"])
    """
    # 1. Explicit char_ids
    explicit = shot_meta.get("_char_ids")
    if explicit:
        return list(explicit)

    # 2. associate_assets
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

    # 3. Prompt name matching against CHARACTERS.json
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
    else:
        character_definitions = {}

    references = []
    for char_id in _detect_shot_characters(output_dir, shot_meta):
        character_name = character_names.get(char_id, char_id)
        char_dir = output_dir / "characters" / char_id
        if not char_dir.exists():
            char_dir = output_dir / "characters" / "characters" / char_id
        reference_paths = [
            char_dir / "face_closeup.png",
            char_dir / "full_body.png",
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
                    "reference_description": (
                        f"{character_name}的面部特写"
                        if reference_path.name == "face_closeup.png"
                        else f"{character_name}的全身照"
                        if reference_path.name == "full_body.png"
                        else f"{character_name}的变体图（{reference_path.stem}）"
                    ),
                })
    return references


def inject_reference_instruction(prompt_text: str, descriptions: List[Any]) -> str:
    """Inject 图片N descriptions using the shared manifest/content order."""
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
    subject_numbers = {}
    for image_number, item in enumerate(normalized, start=1):
        char_id = item.get("char_id") or str(image_number)
        if char_id in subject_numbers:
            continue
        subject_number = len(subject_numbers) + 1
        subject_numbers[char_id] = (image_number, subject_number)
        definition = item.get("prompt_definition") or (
            f"将{{图片N}}中的[{item.get('reference_description', char_id)}]定义为{{主体N}}"
        )
        subject_bindings.append(
            definition.replace("{图片N}", f"图片{image_number}").replace("{主体N}", f"<主体{subject_number}>")
        )

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
    instruction = f"{references}。{'；'.join(subject_bindings)}。生成时严格保持参考图中角色的外观一致。"
    if "元素参考" in prompt_text:
        return re.sub(
            r"元素参考(?:声明)?\s*[：:]?\s*",
            f"元素参考：{instruction}",
            prompt_text,
            count=1,
        )
    return f"元素参考：{instruction}{prompt_text}"


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

    i2v uses one first frame; Phantom adds the shot characters' face, body, and
    optional variant references; FLF2V uses an explicit first and last frame.

    Legacy paths (zip/base64) are preserved for backward compatibility but not
    used in the primary content[] workflow.

    Each image is uploaded to TOS and replaced with a signed URL.
    """
    output_dir = Path(output_dir)
    content = []

    # 1. Text prompt (always first)
    prompt_text = shot_meta.get("prompt", "")
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})

    strategy = shot_meta.get("gen_strategy", "i2v")
    if strategy not in {"flf2v", "phantom", "i2v"}:
        strategy = "i2v"

    # Every route starts from the per-shot storyboard image.
    storyboard_images_dir = output_dir / "storyboard_images"
    if not storyboard_images_dir.exists():
        print(
            f"  [assets] ⚠ storyboard_images directory missing: {storyboard_images_dir}; "
            "run Phase 2.5 to generate per-shot first frames"
        )
    shot_frame_path = storyboard_images_dir / f"{shot_id}.png"
    image_assets = []
    if shot_frame_path.exists() and shot_frame_path.stat().st_size > 1024:
        image_assets.append({
            "path": shot_frame_path,
            "role": "first_frame",
            "priority": "high",
        })

    if strategy == "flf2v":
        end_frame_path = storyboard_images_dir / f"{shot_id}_end.png"
        if end_frame_path.exists() and end_frame_path.stat().st_size > 1024:
            image_assets.append({
                "path": end_frame_path,
                "role": "last_frame",
                "priority": "high",
            })
        else:
            raise FileNotFoundError(
                f"FLF2V end frame missing or too small: {end_frame_path}"
            )
    elif strategy == "phantom":
        image_assets.extend(collect_character_reference_assets(output_dir, shot_meta))
        if not any(asset["role"] == "reference_image" for asset in image_assets):
            raise FileNotFoundError(
                "Phantom character references missing for shot "
                f"{shot_id}; expected face_closeup.png, full_body.png, or variant_*.png"
            )

    print(f"  [assets] {strategy}: images_used={len(image_assets)}")
    
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
