#!/usr/bin/env python3
"""Unit tests for asset_packager module."""

import json
import tempfile
import zipfile
from pathlib import Path
import pytest

from tools.asset_packager import package_shot_assets


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
        
        # Verify base64 list (implementation takes sorted[:9], all 5 fit)
        assert len(base64_list) == 5
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

from tools.asset_packager import build_content_for_shot, _detect_shot_characters


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


def test_build_content_single_image_anti_contamination(monkeypatch):
    """Single-image strategy: only storyboard image, no three-view contamination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_shot_frame(od, "S01")
        _make_storyboard(od)
        _make_characters_json(od, [{"id": "lin_xiao", "name": "林晓"}])

        monkeypatch.setattr(
            "clients.tos_uploader.upload_image",
            lambda *_args, **_kwargs: "https://example.invalid/S01.png",
        )
        content = build_content_for_shot(od, "S01", {
            "prompt": "林晓 sits by the lake.",
            "_char_ids": ["lin_xiao"],
        })

        # Extract image items — should be exactly 1 (storyboard image only)
        images = [c for c in content if c.get("type") == "image_url"]
        assert len(images) == 1, f"Expected 1 image (anti-contamination), got {len(images)}"
        assert images[0]["role"] == "first_frame"
        assert images[0]["priority"] == "high"

        # No reference_image role should exist
        ref_items = [c for c in content if c.get("role") == "reference_image"]
        assert len(ref_items) == 0, "No reference_image (three-view) should exist"


def test_build_content_multi_char_still_single_image(monkeypatch):
    """Multi-character shot: still only 1 image (storyboard), no three-view injection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        _make_char_images(od, "lin_xiao")
        _make_char_images(od, "chen_yang")
        _make_shot_frame(od, "S03")
        _make_storyboard(od)

        monkeypatch.setattr(
            "clients.tos_uploader.upload_image",
            lambda *_args, **_kwargs: "https://example.invalid/S03.png",
        )
        content = build_content_for_shot(od, "S03", {
            "prompt": "林晓 and 陈阳 sit by the lake.",
            "_char_ids": ["lin_xiao", "chen_yang"],
        })

        images = [c for c in content if c.get("type") == "image_url"]
        assert len(images) == 1, f"Expected 1 image (anti-contamination), got {len(images)}"
        assert images[0]["role"] == "first_frame"


def test_build_content_no_storyboard_image():
    """When storyboard image is missing, only text prompt in content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        od = Path(tmpdir)
        # No storyboard image created

        content = build_content_for_shot(od, "S01", {
            "prompt": "A beautiful sunset.",
        })

        images = [c for c in content if c.get("type") == "image_url"]
        assert len(images) == 0, "No images when storyboard image missing"
        assert len(content) == 1  # text only
        assert content[0]["type"] == "text"


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
    test_build_content_single_image_anti_contamination()
    test_build_content_multi_char_still_single_image()
    test_build_content_no_storyboard_image()
    print("\n✓ All asset_packager tests passed!")
