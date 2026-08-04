#!/usr/bin/env python3
"""
Asset packager for Phase 5 video generation.

Packages character references, storyboard images, and metadata into a zip file
for upload to Bridge API. Falls back to base64 list if zip upload fails.
"""

import base64
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple


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
        - base64_list: List of base64-encoded images (sorted by priority, max 3)
    
    Zip structure:
        assets.zip
        ├─ meta.json
        ├─ three_view/          (character front/side/back.png)
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
                {"path": "three_view/front.png", "role": "identity", "priority": 1},
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
                    for view in ["front.png", "side.png", "back.png"]:
                        view_path = char_dir / view
                        if view_path.exists():
                            assets.append({
                                "src_path": view_path,
                                "zip_path": f"three_view/{char_dir.name}_{view}",
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
    
    # Build base64 fallback list (sorted by priority, then by role, max 3)
    # Priority order: identity (character) > composition (shot frame) > style (storyboard)
    sorted_assets = sorted(assets, key=lambda a: (a["priority"], a["role"]))
    base64_list = []
    
    for asset in sorted_assets[:3]:
        try:
            img_data = asset["src_path"].read_bytes()
            b64 = base64.b64encode(img_data).decode('utf-8')
            base64_list.append(b64)
        except Exception as e:
            print(f"  ⚠ Failed to encode {asset['src_path']}: {e}")
    
    return str(zip_path), base64_list


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


def build_content_for_shot(
    output_dir: Path,
    shot_id: str,
    shot_meta: dict,
) -> List[dict]:
    """
    Build Bridge content[] list for a shot with TOS-uploaded reference images.

    Args:
        output_dir: Project output directory
        shot_id: Shot identifier (e.g., "S01")
        shot_meta: Shot metadata dict with prompt, duration, seed, width, height

    Returns:
        List of content items for Bridge /generate API:
        [
            {"type": "text", "text": "shot prompt"},
            {"type": "image_url", "image_url": {"url": "https://..."}, "role": "first_frame", "priority": "high"},
            {"type": "image_url", "image_url": {"url": "https://..."}, "role": "reference_image", "priority": "high"},
            ...
        ]

    Smart image selection strategy (max 9 total, Bridge picks ~3):
        - Single-character shot: front + side + back of that character (all high)
          → Bridge gets 3 views of the same character for maximum consistency
        - Multi-character shot: front of each character (all high)
          → Bridge gets identity reference for each character
        - Shot frame always high priority
        - Storyboard.png always medium priority (style reference)

    Each image is uploaded to TOS and replaced with a signed URL.
    """
    output_dir = Path(output_dir)
    content = []

    # 1. Text prompt (always first)
    prompt_text = shot_meta.get("prompt", "")
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})

    # Detect which characters appear in this shot
    shot_char_ids = _detect_shot_characters(output_dir, shot_meta)

    # Collect image assets with metadata
    image_assets = []

    # 2. Shot frame (storyboard_images/{shot_id}.png) — highest priority
    shot_frame_path = output_dir / "storyboard_images" / f"{shot_id}.png"
    if shot_frame_path.exists() and shot_frame_path.stat().st_size > 1024:
        image_assets.append({
            "path": shot_frame_path,
            "role": "first_frame",
            "priority": "high",
        })

    # 3. Character reference images — smart selection based on shot characters
    char_bases = [output_dir / "characters", output_dir / "characters" / "characters"]
    seen_char_dirs = set()

    # Determine which characters to include and their view priority
    if shot_char_ids:
        # We know exactly which characters are in this shot
        for char_base in char_bases:
            if char_base.exists():
                for char_dir in char_base.iterdir():
                    if char_dir.is_dir() and char_dir.name not in seen_char_dirs:
                        seen_char_dirs.add(char_dir.name)
                        if char_dir.name in shot_char_ids:
                            # This character is IN the shot
                            is_single_char_shot = len(shot_char_ids) == 1
                            # Front view always high
                            front_path = char_dir / "front.png"
                            if front_path.exists() and front_path.stat().st_size > 1024:
                                image_assets.append({
                                    "path": front_path,
                                    "role": "reference_image",
                                    "priority": "high",
                                })
                            # Side/back: high for single-char shots, medium for multi-char
                            side_back_priority = "high" if is_single_char_shot else "medium"
                            for view in ["side.png", "back.png"]:
                                view_path = char_dir / view
                                if view_path.exists() and view_path.stat().st_size > 1024:
                                    image_assets.append({
                                        "path": view_path,
                                        "role": "reference_image",
                                        "priority": side_back_priority,
                                    })
                        # Characters NOT in shot: skip entirely (don't waste slots)
    else:
        # Fallback: no character detection — include all characters (legacy behavior)
        for char_base in char_bases:
            if char_base.exists():
                for char_dir in char_base.iterdir():
                    if char_dir.is_dir() and char_dir.name not in seen_char_dirs:
                        seen_char_dirs.add(char_dir.name)
                        front_path = char_dir / "front.png"
                        if front_path.exists() and front_path.stat().st_size > 1024:
                            image_assets.append({
                                "path": front_path,
                                "role": "reference_image",
                                "priority": "high",
                            })
                        for view in ["side.png", "back.png"]:
                            view_path = char_dir / view
                            if view_path.exists() and view_path.stat().st_size > 1024:
                                image_assets.append({
                                    "path": view_path,
                                    "role": "reference_image",
                                    "priority": "medium",
                                })

    # 4. Storyboard.png (medium priority)
    storyboard_path = output_dir / "storyboard.png"
    if storyboard_path.exists() and storyboard_path.stat().st_size > 1024:
        image_assets.append({
            "path": storyboard_path,
            "role": "reference_image",
            "priority": "medium",
        })

    # Sort by priority (high first), then upload to TOS
    image_assets.sort(key=lambda a: (0 if a["priority"] == "high" else 1, a["path"].name))

    # Log the smart selection
    high_count = sum(1 for a in image_assets if a["priority"] == "high")
    print(f"  [assets] 智能选图: {len(shot_char_ids)} 个出场角色 {shot_char_ids}, "
          f"{high_count} high + {len(image_assets) - high_count} medium = {len(image_assets)} 张")

    # Limit to 9 images max (Bridge limit)
    image_assets = image_assets[:9]
    
    # Upload each image to TOS and add to content
    uploaded_count = 0
    try:
        import tos_uploader
    except ImportError:
        print(f"  [assets] ⚠ tos_uploader not available, skipping image upload")
        return content
    
    for asset in image_assets:
        try:
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
            else:
                print(f"  [assets] ⚠ TOS upload failed for {asset['path'].name}, skipping")
        except Exception as e:
            print(f"  [assets] ⚠ Failed to upload {asset['path'].name}: {e}")
    
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
