#!/usr/bin/env python3
"""Tests for Fix C: storyboard structured fields + M2 character reference fixes.

Covers:
1. adaptation_engine._parse_response: normalizes shot_size/camera_movement/lighting_key/shot_intent
2. storyboard_generator.generate_storyboard: passes through who/shot_size/camera_movement/lighting_key/shot_intent/associate_assets
3. pipeline_runner._generate_shot_images: no_character (empty who=[]) skips ref injection; multi-char uses first match
4. orchestrator.route_shot: first_frame_exists resolves relative to storyboard dir (not CWD)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Mock openai before importing modules that depend on it
sys.modules['openai'] = MagicMock()

# Ensure pipeline/src is importable
PIPELINE_SRC = Path(__file__).resolve().parent.parent / "src"
VENDOR_LEGACY = Path(__file__).resolve().parent.parent.parent / "vendor" / "legacy"
sys.path.insert(0, str(PIPELINE_SRC))
sys.path.insert(0, str(VENDOR_LEGACY))


# ─── 1. adaptation_engine: _parse_response normalization ────────────────────

class TestParseResponseNormalization:
    """Test that _parse_response normalizes structured fields."""

    def _make_response(self, shots_override=None):
        """Build a minimal valid LLM response JSON string."""
        shots = shots_override or [{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": ["林晓"],
            "where": "西湖边",
            "what": "看夕阳",
            "visual": "林晓坐在湖边看夕阳",
            "suggested_duration": 6,
        }]
        return json.dumps({"strategy": "test", "shots": shots})

    def test_valid_structured_fields_preserved(self):
        """When LLM returns valid shot_size etc., they are preserved."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response([{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": ["林晓"],
            "where": "西湖边",
            "what": "看夕阳",
            "visual": "林晓坐在湖边看夕阳",
            "suggested_duration": 6,
            "shot_size": "close_up",
            "camera_movement": "dolly_in",
            "lighting_key": "golden_hour",
            "shot_intent": "reveal",
        }])
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["shot_size"] == "close_up"
        assert shot["camera_movement"] == "dolly_in"
        assert shot["lighting_key"] == "golden_hour"
        assert shot["shot_intent"] == "reveal"

    def test_missing_structured_fields_get_defaults(self):
        """When LLM omits structured fields, sensible defaults are filled."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response()  # no shot_size etc.
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["shot_size"] == "wide"
        assert shot["camera_movement"] == "static"
        assert shot["lighting_key"] == "natural"
        assert shot["shot_intent"] == "atmosphere"

    def test_invalid_structured_fields_get_defaults(self):
        """When LLM returns garbage values, defaults are used."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response([{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": ["林晓"],
            "where": "西湖边",
            "what": "看夕阳",
            "visual": "test",
            "suggested_duration": 6,
            "shot_size": "bogus_value",
            "camera_movement": "spin_around",
            "lighting_key": "rainbow",
            "shot_intent": "confused",
        }])
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["shot_size"] == "wide"
        assert shot["camera_movement"] == "static"
        assert shot["lighting_key"] == "natural"
        assert shot["shot_intent"] == "atmosphere"

    def test_who_string_converted_to_list(self):
        """When LLM returns who as a string instead of list, it's normalized."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response([{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": "林晓",  # string, not list
            "where": "西湖边",
            "what": "看夕阳",
            "visual": "test",
            "suggested_duration": 6,
        }])
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["who"] == ["林晓"]

    def test_empty_who_preserved_as_empty_list(self):
        """Pure landscape shots: who=[] stays as empty list."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response([{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": [],
            "where": "西湖边",
            "what": "夕阳西下",
            "visual": "夕阳照在湖面上",
            "suggested_duration": 6,
        }])
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["who"] == []

    def test_associate_assets_normalized(self):
        """associate_assets is ensured to be a list."""
        from phases.adaptation_engine import _parse_response
        resp = self._make_response([{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": ["林晓"],
            "where": "西湖边",
            "what": "看夕阳",
            "visual": "test",
            "suggested_duration": 6,
            "associate_assets": ["char:lin_xiao", "scene:西湖"],
        }])
        parsed = _parse_response(resp)
        shot = parsed["shots"][0]
        assert shot["associate_assets"] == ["char:lin_xiao", "scene:西湖"]


# ─── 2. storyboard_generator: pass-through of structured fields ─────────────

class TestStoryboardGeneratorPassthrough:
    """Test that generate_storyboard passes structured fields to STORYBOARD.json."""

    @patch("phases.storyboard_generator._call_llm")
    def test_structured_fields_passed_through(self, mock_llm):
        """who/shot_size/camera_movement/lighting_key/shot_intent/associate_assets appear in output."""
        from phases.storyboard_generator import generate_storyboard

        mock_llm.return_value = json.dumps({
            "prompt": "Cinematic test shot",
            "caption": "测试镜头",
        })

        shots = [{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": ["林晓", "陈阳"],
            "where": "西湖边",
            "what": "一起看夕阳",
            "visual": "两人坐在湖边",
            "suggested_duration": 6,
            "shot_size": "medium_wide",
            "camera_movement": "static",
            "lighting_key": "golden_hour",
            "shot_intent": "atmosphere",
            "associate_assets": ["char:lin_xiao", "char:chen_yang"],
        }]

        result = generate_storyboard(shots)
        sb_shot = result["shots"][0]

        assert sb_shot["who"] == ["林晓", "陈阳"]
        assert sb_shot["shot_size"] == "medium_wide"
        assert sb_shot["camera_movement"] == "static"
        assert sb_shot["lighting_key"] == "golden_hour"
        assert sb_shot["shot_intent"] == "atmosphere"
        assert sb_shot["associate_assets"] == ["char:lin_xiao", "char:chen_yang"]

    @patch("phases.storyboard_generator._call_llm")
    def test_empty_who_no_first_frame(self, mock_llm):
        """Pure landscape shot (who=[]) → no first_frame set."""
        from phases.storyboard_generator import generate_storyboard

        mock_llm.return_value = json.dumps({
            "prompt": "Sunset over lake",
            "caption": "夕阳湖面",
        })

        shots = [{
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "who": [],
            "where": "西湖",
            "what": "夕阳",
            "visual": "夕阳照在湖面上",
            "suggested_duration": 6,
            "shot_size": "extreme_wide",
            "camera_movement": "static",
            "lighting_key": "golden_hour",
            "shot_intent": "establishing",
        }]

        result = generate_storyboard(shots)
        sb_shot = result["shots"][0]

        assert sb_shot["who"] == []
        assert "first_frame" not in sb_shot  # no characters → no first_frame
        assert sb_shot["shot_size"] == "extreme_wide"


# ─── 3. _generate_shot_images: no_character + multi-character ───────────────

class TestGenerateShotImagesCharacterRef:
    """Test M2 character reference matching logic."""

    def test_empty_who_skips_character_ref(self):
        """Pure landscape shot (who=[]) should NOT inject any character reference."""
        # Import the function directly
        from pipeline_runner import _generate_shot_images

        storyboard_data = {"shots": [
            {"id": 1, "prompt": "Sunset over lake", "who": []},
        ]}

        # Track what ref_image is passed to image_to_image
        captured_calls = []

        def mock_image_to_image(prompt, ref_image, output_path, size=None):
            captured_calls.append({"prompt": prompt, "ref_image": ref_image, "output_path": output_path})
            # Create a fake output file
            Path(output_path).write_bytes(b"fake_png")

        def mock_text_to_image(prompt, output_path, size=None):
            captured_calls.append({"prompt": prompt, "ref_image": None, "output_path": output_path})
            Path(output_path).write_bytes(b"fake_png")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            storyboard_images_dir = output_dir / "storyboard_images"
            storyboard_images_dir.mkdir()

            # Mock the seedream_client module
            mock_seedream = MagicMock()
            mock_seedream.text_to_image = mock_text_to_image
            sys.modules['clients.seedream_client'] = mock_seedream

            try:
                _generate_shot_images(output_dir, storyboard_data)
            finally:
                del sys.modules['clients.seedream_client']

            # For who=[], text_to_image should be called (no ref image)
            assert len(captured_calls) == 1
            assert captured_calls[0]["ref_image"] is None

    def test_multi_char_uses_first_match(self):
        """When who has multiple characters, first matching character's front.png is used."""
        from pipeline_runner import _generate_shot_images

        storyboard_data = {"shots": [
            {"id": 1, "prompt": "Two people sitting", "who": ["林晓", "陈阳"]},
        ]}

        captured_calls = []

        def mock_image_to_image(prompt, ref_image, output_path, size=None):
            captured_calls.append({"prompt": prompt, "ref_image": ref_image, "output_path": output_path})
            Path(output_path).write_bytes(b"fake_png")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            storyboard_images_dir = output_dir / "storyboard_images"
            storyboard_images_dir.mkdir()

            # Create CHARACTERS.json
            chars_data = {
                "characters": [
                    {"id": "lin_xiao", "name": "林晓"},
                    {"id": "chen_yang", "name": "陈阳"},
                ]
            }
            (output_dir / "CHARACTERS.json").write_text(json.dumps(chars_data))

            # Create front.png for both characters
            for char_id in ["lin_xiao", "chen_yang"]:
                char_dir = output_dir / "characters" / char_id
                char_dir.mkdir(parents=True)
                (char_dir / "front.png").write_bytes(b"fake_png_data")

            # Mock SeedreamClient as a class with image_to_image method
            MockSeedreamClient = MagicMock()
            MockSeedreamClient.return_value.image_to_image = mock_image_to_image

            mock_seedream = MagicMock()
            mock_seedream.SeedreamClient = MockSeedreamClient
            sys.modules['clients.seedream_client'] = mock_seedream

            try:
                _generate_shot_images(output_dir, storyboard_data)
            finally:
                del sys.modules['clients.seedream_client']

            # Should use 林晓's front.png (first in who list)
            assert len(captured_calls) == 1
            assert captured_calls[0]["ref_image"] is not None
            assert "lin_xiao" in captured_calls[0]["ref_image"]


# ─── 4. orchestrator: first_frame_exists path resolution ────────────────────

def _load_orchestrator():
    """Load orchestrator.py as a fresh module for testing."""
    import importlib.util
    orch_path = VENDOR_LEGACY / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("orchestrator_under_test", orch_path)
    orch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orch)
    return orch


class TestOrchestratorFirstFrameExists:
    """Test that route_shot resolves first_frame relative to storyboard dir."""

    def test_first_frame_exists_with_correct_storyboard_path(self):
        """When STORYBOARD_PATH is set correctly, first_frame_exists resolves properly."""
        orch = _load_orchestrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create storyboard with a shot that references a character
            char_dir = tmpdir / "characters" / "lin_xiao"
            char_dir.mkdir(parents=True)
            front_png = char_dir / "front.png"
            front_png.write_bytes(b"fake_character_image")

            storyboard = {
                "shots": [{
                    "id": 1,
                    "name": "test shot",
                    "duration": 6,
                    "prompt": "test prompt",
                    "first_frame": "characters/lin_xiao/front.png",
                    "caption": "test",
                    "caption_frames": "1-100",
                }]
            }
            sb_path = tmpdir / "STORYBOARD.json"
            sb_path.write_text(json.dumps(storyboard))

            # Set the global STORYBOARD_PATH to the correct location
            orch.STORYBOARD_PATH = sb_path.resolve()
            orch.SHOTS_DIR = tmpdir / "shots"

            shots = orch.parse_shots(storyboard)
            routed = orch.route_shot(shots[0], {})

            assert routed["first_frame_exists"] is True
            assert routed["route"] == "img2vid"
            assert "lin_xiao" in routed["first_frame_path"]

    def test_first_frame_not_exists(self):
        """When the referenced file doesn't exist, first_frame_exists is False."""
        orch = _load_orchestrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            storyboard = {
                "shots": [{
                    "id": 1,
                    "name": "test shot",
                    "duration": 6,
                    "prompt": "test prompt",
                    "first_frame": "characters/nonexistent/front.png",
                    "caption": "test",
                    "caption_frames": "1-100",
                }]
            }
            sb_path = tmpdir / "STORYBOARD.json"
            sb_path.write_text(json.dumps(storyboard))

            orch.STORYBOARD_PATH = sb_path.resolve()
            orch.SHOTS_DIR = tmpdir / "shots"

            shots = orch.parse_shots(storyboard)
            routed = orch.route_shot(shots[0], {})

            assert routed["first_frame_exists"] is False

    def test_no_first_frame_means_txt2vid(self):
        """Shot without first_frame → txt2vid route."""
        orch = _load_orchestrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sb_path = tmpdir / "STORYBOARD.json"
            sb_path.write_text("{}")
            orch.STORYBOARD_PATH = sb_path.resolve()
            orch.SHOTS_DIR = tmpdir / "shots"

            storyboard = {
                "shots": [{
                    "id": 1,
                    "name": "landscape",
                    "duration": 6,
                    "prompt": "sunset",
                    "caption": "",
                    "caption_frames": "",
                }]
            }
            shots = orch.parse_shots(storyboard)
            routed = orch.route_shot(shots[0], {})

            assert routed["route"] == "txt2vid"
            assert routed["first_frame_exists"] is False
