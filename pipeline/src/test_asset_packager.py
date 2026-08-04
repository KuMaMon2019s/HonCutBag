#!/usr/bin/env python3
"""Unit tests for asset_packager module."""

import json
import tempfile
import zipfile
from pathlib import Path
import pytest

from asset_packager import package_shot_assets


def test_package_shot_assets_with_all_assets():
    """Test packaging when all asset types exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create character directory structure
        char_dir = output_dir / "characters" / "characters" / "char_001"
        char_dir.mkdir(parents=True)
        
        # Create character images
        (char_dir / "front.png").write_bytes(b"front_image_data")
        (char_dir / "side.png").write_bytes(b"side_image_data")
        (char_dir / "back.png").write_bytes(b"back_image_data")
        
        # Create storyboard image
        storyboard_img_dir = output_dir / "storyboard_images"
        storyboard_img_dir.mkdir()
        (storyboard_img_dir / "S01.png").write_bytes(b"storyboard_image_data")
        
        # Create storyboard.png
        (output_dir / "storyboard.png").write_bytes(b"storyboard_data")
        
        # Package assets
        shot_meta = {
            "prompt": "A test shot",
            "duration": 5,
            "seed": 12345,
            "width": 1280,
            "height": 720,
        }
        
        zip_path, base64_list = package_shot_assets(
            output_dir=output_dir,
            shot_id="S01",
            shot_meta=shot_meta,
        )
        
        # Verify zip was created
        assert zip_path is not None
        assert Path(zip_path).exists()
        
        # Verify zip contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            
            # Check meta.json exists
            assert "meta.json" in namelist
            
            # Check character images (three_view/{char_id}_{view} structure)
            assert "three_view/char_001_front.png" in namelist
            assert "three_view/char_001_side.png" in namelist
            assert "three_view/char_001_back.png" in namelist
            
            # Check storyboard image
            assert "shot_frames/S01.png" in namelist
            
            # Check storyboard.png
            assert "storyboard/storyboard.png" in namelist
            
            # Verify meta.json content
            meta_data = json.loads(zf.read("meta.json"))
            assert meta_data["shot_id"] == "S01"
            assert meta_data["prompt"] == "A test shot"
            assert meta_data["duration"] == 5
            assert meta_data["seed"] == 12345
            assert meta_data["width"] == 1280
            assert meta_data["height"] == 720
            
            # Verify images metadata
            assert len(meta_data["images"]) == 5
            image_paths = [img["path"] for img in meta_data["images"]]
            assert "three_view/char_001_front.png" in image_paths
            assert "shot_frames/S01.png" in image_paths
            assert "storyboard/storyboard.png" in image_paths
        
        # Verify base64 list (implementation takes sorted[:3])
        assert len(base64_list) == 3
        assert all(isinstance(b64, str) for b64 in base64_list)
        
        print("✓ Test passed: package_shot_assets with all assets")


def test_package_shot_assets_missing_assets():
    """Test packaging when some assets are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Only create character directory (no character images)
        char_dir = output_dir / "characters" / "characters" / "char_001"
        char_dir.mkdir(parents=True)
        
        # Create only front.png
        (char_dir / "front.png").write_bytes(b"front_image_data")
        
        # No storyboard image, no storyboard.png
        
        shot_meta = {"prompt": "Test"}
        
        zip_path, base64_list = package_shot_assets(
            output_dir=output_dir,
            shot_id="S01",
            shot_meta=shot_meta,
        )
        
        # Verify zip was created with only available assets
        assert zip_path is not None
        assert Path(zip_path).exists()
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            assert "meta.json" in namelist
            assert "three_view/char_001_front.png" in namelist
            assert "shot_frames/S01.png" not in namelist
            assert "storyboard/storyboard.png" not in namelist
            
            meta_data = json.loads(zf.read("meta.json"))
            assert len(meta_data["images"]) == 1
        
        assert len(base64_list) == 1
        
        print("✓ Test passed: package_shot_assets with missing assets")


def test_package_shot_assets_no_assets():
    """Test packaging when no assets exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        shot_meta = {"prompt": "Test"}
        
        zip_path, base64_list = package_shot_assets(
            output_dir=output_dir,
            shot_id="S01",
            shot_meta=shot_meta,
        )
        
        # Should return None for zip_path when no assets
        assert zip_path is None
        assert len(base64_list) == 0
        
        print("✓ Test passed: package_shot_assets with no assets")


def test_package_shot_assets_multiple_characters():
    """Test packaging with multiple characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create multiple character directories
        for char_id in ["char_001", "char_002"]:
            char_dir = output_dir / "characters" / "characters" / char_id
            char_dir.mkdir(parents=True)
            (char_dir / "front.png").write_bytes(f"{char_id}_front".encode())
        
        shot_meta = {"prompt": "Test"}
        
        zip_path, base64_list = package_shot_assets(
            output_dir=output_dir,
            shot_id="S01",
            shot_meta=shot_meta,
        )
        
        assert zip_path is not None
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            assert "three_view/char_001_front.png" in namelist
            assert "three_view/char_002_front.png" in namelist
            
            meta_data = json.loads(zf.read("meta.json"))
            assert len(meta_data["images"]) == 2
        
        assert len(base64_list) == 2
        
        print("✓ Test passed: package_shot_assets with multiple characters")


# ─── Tests for build_content_for_shot smart selection ───

from asset_packager import build_content_for_shot, _detect_shot_characters


def _make_char_images(output_dir: Path, char_id: str, views=("front", "side", "back")):
    """Helper: create character images with realistic size (>1024 bytes)."""
    char_dir = output_dir / "characters" / char_id
    char_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        (char_dir / f"{view}.png").write_bytes(b"x" * 2048)
    return char_dir


def _make_shot_frame(output_dir: Path, shot_id: str):
    img_dir = output_dir / "storyboard_images"
    img_dir.mkdir(exist_ok=True)
    (img_dir / f"{shot_id}.png").write_bytes(b"x" * 2048)


def _make_storyboard(output_dir: Path):
    (output_dir / "storyboard.png").write_bytes(b"x" * 2048)


def _make_characters_json(output_dir: Path, characters: list):
    """Create CHARACTERS.json with list of {id, name}."""
    import json
    data = {"characters": [{"id": c["id"], "name": c["name"]} for c in characters]}
    (output_dir / "CHARACTERS.json").write_text(json.dumps(data, ensure_ascii=False))


def test_detect_shot_characters_explicit():
    """_detect_shot_characters returns explicit _char_ids when present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        result = _detect_shot_characters(od, {"_char_ids": ["char_a"]})
        assert result == ["char_a"]


def test_detect_shot_characters_associate_assets():
    """_detect_shot_characters extracts from associate_assets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        result = _detect_shot_characters(od, {
            "associate_assets": ["char:lin_xiao", "char:chen_yang", "scene:park"]
        })
        assert result == ["lin_xiao", "chen_yang"]


def test_detect_shot_characters_prompt_matching():
    """_detect_shot_characters matches character names in prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_characters_json(od, [
            {"id": "lin_xiao", "name": "林晓"},
            {"id": "chen_yang", "name": "陈阳"},
        ])
        result = _detect_shot_characters(od, {
            "prompt": "林晓 sits by the lake, looking at the sunset."
        })
        assert result == ["lin_xiao"]


def test_detect_shot_characters_multi_char_prompt():
    """_detect_shot_characters matches multiple characters in prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_characters_json(od, [
            {"id": "lin_xiao", "name": "林晓"},
            {"id": "chen_yang", "name": "陈阳"},
        ])
        result = _detect_shot_characters(od, {
            "prompt": "林晓 and 陈阳 are sitting by the lake together."
        })
        assert set(result) == {"lin_xiao", "chen_yang"}


def test_detect_shot_characters_no_match():
    """_detect_shot_characters returns empty when no characters match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_characters_json(od, [
            {"id": "lin_xiao", "name": "林晓"},
        ])
        result = _detect_shot_characters(od, {
            "prompt": "A beautiful sunset over the lake."
        })
        assert result == []


def test_build_content_single_char_gets_three_views():
    """Single-character shot: front + side + back all high priority."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_shot_frame(od, "S01")
        _make_storyboard(od)
        _make_characters_json(od, [{"id": "lin_xiao", "name": "林晓"}])

        content = build_content_for_shot(od, "S01", {
            "prompt": "林晓 sits by the lake.",
            "_char_ids": ["lin_xiao"],
        })

        # Extract image items
        images = [c for c in content if c.get("type") == "image_url"]
        high_images = [c for c in images if c.get("priority") == "high"]

        # Should have: shot_frame + front + side + back = 4 high
        assert len(high_images) == 4, f"Expected 4 high, got {len(high_images)}: {high_images}"

        # Verify no medium character images leaked in for non-shot characters
        medium_images = [c for c in images if c.get("priority") == "medium"]
        # Only storyboard should be medium
        assert len(medium_images) == 1  # storyboard.png


def test_build_content_multi_char_gets_fronts_only():
    """Multi-character shot: each character's front is high, side/back are medium."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_char_images(od, "chen_yang")
        _make_shot_frame(od, "S03")
        _make_storyboard(od)
        _make_characters_json(od, [
            {"id": "lin_xiao", "name": "林晓"},
            {"id": "chen_yang", "name": "陈阳"},
        ])

        content = build_content_for_shot(od, "S03", {
            "prompt": "林晓 and 陈阳 sit by the lake.",
            "_char_ids": ["lin_xiao", "chen_yang"],
        })

        images = [c for c in content if c.get("type") == "image_url"]
        high_images = [c for c in images if c.get("priority") == "high"]
        medium_images = [c for c in images if c.get("priority") == "medium"]

        # High: shot_frame + lin_xiao.front + chen_yang.front = 3
        assert len(high_images) == 3, f"Expected 3 high, got {len(high_images)}"

        # Medium: lin_xiao.side + lin_xiao.back + chen_yang.side + chen_yang.back + storyboard = 5
        assert len(medium_images) == 5, f"Expected 5 medium, got {len(medium_images)}"


def test_build_content_skips_non_shot_characters():
    """Characters not in the shot should NOT be included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_char_images(od, "chen_yang")
        _make_shot_frame(od, "S02")
        _make_storyboard(od)
        _make_characters_json(od, [
            {"id": "lin_xiao", "name": "林晓"},
            {"id": "chen_yang", "name": "陈阳"},
        ])

        content = build_content_for_shot(od, "S02", {
            "prompt": "林晓 walks alone in the park.",
            "_char_ids": ["lin_xiao"],
        })

        images = [c for c in content if c.get("type") == "image_url"]

        # chen_yang should NOT appear at all
        # We can't check by path (TOS URLs replace them), but we can count:
        # High: shot_frame + lin_xiao.front + lin_xiao.side + lin_xiao.back = 4
        # Medium: storyboard = 1
        # Total = 5 (not 8 which would include chen_yang)
        assert len(images) == 5, f"Expected 5 total images, got {len(images)}"


def test_build_content_fallback_no_detection():
    """When no character detection available, legacy behavior (all chars)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_char_images(od, "chen_yang")
        _make_shot_frame(od, "S01")
        _make_storyboard(od)
        # No CHARACTERS.json, no _char_ids, no associate_assets

        content = build_content_for_shot(od, "S01", {
            "prompt": "A beautiful sunset.",
        })

        images = [c for c in content if c.get("type") == "image_url"]
        high_images = [c for c in images if c.get("priority") == "high"]

        # Legacy: shot_frame + lin_xiao.front + chen_yang.front = 3 high
        assert len(high_images) == 3
        # Medium: lin_xiao.side + lin_xiao.back + chen_yang.side + chen_yang.back + storyboard = 5
        medium_images = [c for c in images if c.get("priority") == "medium"]
        assert len(medium_images) == 5


if __name__ == "__main__":
    test_package_shot_assets_with_all_assets()
    test_package_shot_assets_missing_assets()
    test_package_shot_assets_no_assets()
    test_package_shot_assets_multiple_characters()
    test_detect_shot_characters_explicit()
    test_detect_shot_characters_associate_assets()
    test_detect_shot_characters_prompt_matching()
    test_detect_shot_characters_multi_char_prompt()
    test_detect_shot_characters_no_match()
    test_build_content_single_char_gets_three_views()
    test_build_content_multi_char_gets_fronts_only()
    test_build_content_skips_non_shot_characters()
    test_build_content_fallback_no_detection()
    print("\n✓ All asset_packager tests passed!")
