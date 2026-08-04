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


if __name__ == "__main__":
    test_package_shot_assets_with_all_assets()
    test_package_shot_assets_missing_assets()
    test_package_shot_assets_no_assets()
    test_package_shot_assets_multiple_characters()
    print("\n✓ All asset_packager tests passed!")
