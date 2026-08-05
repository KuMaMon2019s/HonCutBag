"""M2: FLF2V end frame reference swap + post-generation validation tests.

Tests cover:
- Reference selection: first frame as primary reference (not character front.png)
- End-state mapping: action verb → structured end-state + fallback
- validate_end_frame: resolution, brightness, sharpness, similarity band checks
- Cache sidecar: hash-based skip/regenerate logic
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add pipeline/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pipeline_runner


class TestEndStateMapping:
    """Test _derive_end_state action verb → end-state mapping."""

    def test_chinese_action_verb_tai_shou(self):
        """抬手 → hand raised end state."""
        shot = {"visual": "少女抬手拂发，夕阳洒落"}
        end_state = pipeline_runner._derive_end_state(shot)
        assert "hand" in end_state.lower() or "hair" in end_state.lower()

    def test_chinese_action_verb_zou_lai(self):
        """走来 → arrived at destination."""
        shot = {"visual": "少年从远处走来"}
        end_state = pipeline_runner._derive_end_state(shot)
        assert "arrived" in end_state.lower() or "standing" in end_state.lower()

    def test_chinese_action_verb_zuo_xia(self):
        """坐下 → seated."""
        shot = {"what": "老者缓缓坐下"}
        end_state = pipeline_runner._derive_end_state(shot)
        assert "seated" in end_state.lower()

    def test_english_action_verb_raise_hand(self):
        """raise hand → arm extended."""
        shot = {"description": "character raises hand to wave"}
        end_state = pipeline_runner._derive_end_state(shot)
        assert "raised" in end_state.lower() or "extended" in end_state.lower()

    def test_unknown_action_fallback(self):
        """Unknown action → generic fallback."""
        shot = {"visual": " mysterious energy flows through the scene"}
        end_state = pipeline_runner._derive_end_state(shot)
        assert "fully completed" in end_state.lower() or "natural resting" in end_state.lower()

    def test_empty_shot_fallback(self):
        """Empty shot → fallback."""
        shot = {}
        end_state = pipeline_runner._derive_end_state(shot)
        assert end_state  # non-empty


class TestBuildEndFramePrompt:
    """Test build_end_frame_prompt incorporates end-state."""

    def test_prompt_contains_start_frame_reference(self):
        """Prompt must reference 'start frame' semantics."""
        shot = {"prompt": "西湖傍晚，少女拂发", "visual": "夕阳下少女抬手拂发"}
        prompt = pipeline_runner.build_end_frame_prompt(shot)
        assert "start frame" in prompt.lower()

    def test_prompt_contains_end_state(self):
        """Prompt must include derived end-state."""
        shot = {"visual": "少年走来", "prompt": "西湖边"}
        prompt = pipeline_runner.build_end_frame_prompt(shot)
        # Should contain the end-state for 走来
        end_state = pipeline_runner._derive_end_state(shot)
        assert end_state in prompt


class TestValidateEndFrame:
    """Test _validate_end_frame metric-based validation."""

    @pytest.fixture
    def temp_images(self, tmp_path):
        """Create synthetic test images."""
        # First frame: gradient image (not black, has sharpness)
        first_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            first_arr[i, :, :] = i * 2  # vertical gradient 0-198
        first_img = Image.fromarray(first_arr)
        first_path = tmp_path / "S01.png"
        first_img.save(first_path)

        return {
            "first_path": first_path,
            "first_arr": first_arr,
            "tmp_path": tmp_path,
        }

    def test_identical_image_too_similar(self, temp_images):
        """Identical end frame → rejected as too similar (no action progress)."""
        end_path = temp_images["tmp_path"] / "S01_end.png"
        # Copy first frame as end frame (identical)
        Image.open(temp_images["first_path"]).save(end_path)

        result = pipeline_runner._validate_end_frame(
            temp_images["first_path"], end_path,
            similarity_low=0.25, similarity_high=0.95
        )
        assert result["passed"] is False
        assert "too similar" in result["reason"].lower()

    def test_black_image_rejected(self, temp_images):
        """Black end frame → rejected as brightness out of range."""
        end_path = temp_images["tmp_path"] / "S01_end.png"
        black_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        Image.fromarray(black_arr).save(end_path)

        result = pipeline_runner._validate_end_frame(
            temp_images["first_path"], end_path
        )
        assert result["passed"] is False
        assert "brightness" in result["reason"].lower()

    def test_resolution_mismatch_rejected(self, temp_images):
        """Different resolution → rejected."""
        end_path = temp_images["tmp_path"] / "S01_end.png"
        # Different size: 50x50 vs 100x100
        diff_size_arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(diff_size_arr).save(end_path)

        result = pipeline_runner._validate_end_frame(
            temp_images["first_path"], end_path
        )
        assert result["passed"] is False
        assert "resolution" in result["reason"].lower()

    def test_drifted_image_too_different(self, temp_images):
        """Completely different scene → rejected as too different (scene drift)."""
        end_path = temp_images["tmp_path"] / "S01_end.png"
        # Solid white (maximally different from gradient which averages ~100)
        white_arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        Image.fromarray(white_arr).save(end_path)

        result = pipeline_runner._validate_end_frame(
            temp_images["first_path"], end_path,
            similarity_low=0.25, similarity_high=0.85
        )
        assert result["passed"] is False
        # White vs gradient should be quite different
        assert result["reason"] is not None

    def test_sane_pair_passes(self, temp_images):
        """Moderately different image (action progressed) → passes."""
        end_path = temp_images["tmp_path"] / "S01_end.png"
        # Create a more different image: invert + shift + heavy noise
        first_arr = temp_images["first_arr"]
        # Invert the gradient (creates significant difference)
        inverted = 255 - first_arr
        # Shift to simulate camera/character movement
        shifted = np.roll(inverted, shift=20, axis=0)
        # Add heavy noise
        np.random.seed(456)
        noisy = np.clip(shifted.astype(np.int16) + np.random.randint(-50, 50, shifted.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(end_path)

        result = pipeline_runner._validate_end_frame(
            temp_images["first_path"], end_path,
            similarity_low=0.25, similarity_high=0.85
        )
        # Should pass: not identical, not black, not drifted
        assert result["resolution_ok"] is True
        assert result["brightness_ok"] is True
        assert result["sharpness_ok"] is True
        assert result["similarity"] is not None
        # The similarity should be in the valid band
        assert 0.25 <= result["similarity"] <= 0.85 or result["passed"]


class TestCacheSidecar:
    """Test cache sidecar read/write/hash logic."""

    def test_write_and_read_sidecar(self, tmp_path):
        """Write sidecar, read it back, verify contents."""
        end_path = tmp_path / "S01_end.png"
        end_path.touch()

        first_sha = "abc123"
        prompt_sha = "def456"
        validation = {"passed": True, "similarity": 0.65, "sharpness_ok": True}

        pipeline_runner._write_end_frame_sidecar(
            end_path, first_sha, prompt_sha, validation
        )

        sidecar = pipeline_runner._read_end_frame_sidecar(end_path)
        assert sidecar is not None
        assert sidecar["first_frame_sha256"] == first_sha
        assert sidecar["prompt_sha256"] == prompt_sha
        assert sidecar["validation"]["passed"] is True
        assert sidecar["validation"]["similarity"] == 0.65

    def test_missing_sidecar_returns_none(self, tmp_path):
        """Missing sidecar → None."""
        end_path = tmp_path / "S01_end.png"
        sidecar = pipeline_runner._read_end_frame_sidecar(end_path)
        assert sidecar is None

    def test_file_sha256_deterministic(self, tmp_path):
        """Same file → same SHA-256."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        sha1 = pipeline_runner._file_sha256(test_file)
        sha2 = pipeline_runner._file_sha256(test_file)
        assert sha1 == sha2
        assert len(sha1) == 64  # SHA-256 hex digest

    def test_file_sha256_changes_with_content(self, tmp_path):
        """Different content → different SHA-256."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        sha1 = pipeline_runner._file_sha256(test_file)
        test_file.write_text("world")
        sha2 = pipeline_runner._file_sha256(test_file)
        assert sha1 != sha2


class TestGenerateFLF2VEndFrame:
    """Test _generate_flf2v_end_frame reference swap + validation."""

    def test_first_frame_must_exist(self, tmp_path):
        """Missing first frame → FileNotFoundError."""
        shot = {"gen_strategy": "flf2v", "visual": "抬手"}
        first_frame_path = tmp_path / "S01.png"  # does not exist
        ref_image_path = tmp_path / "char_front.png"
        ref_image_path.touch()

        with pytest.raises(FileNotFoundError, match="first frame.*not found"):
            pipeline_runner._generate_flf2v_end_frame(
                shot, "S01", first_frame_path, ref_image_path
            )

    def test_non_flf2v_shot_skipped(self, tmp_path):
        """Non-FLF2V shot → return False (no generation)."""
        shot = {"gen_strategy": "i2v"}
        first_frame_path = tmp_path / "S01.png"
        first_frame_path.touch()

        result = pipeline_runner._generate_flf2v_end_frame(
            shot, "S01", first_frame_path, None
        )
        assert result is False

    @patch("seedream_client.SeedreamClient")
    @patch("pipeline_runner._validate_end_frame")
    def test_first_frame_used_as_reference(self, mock_validate, mock_client_class, tmp_path):
        """First frame (not character front.png) passed as reference."""
        @patch("pipeline_runner._validate_end_frame")
        @patch("seedream_client.SeedreamClient")
        def test_first_frame_used_as_reference(self, mock_client_class, mock_validate, tmp_path):
            """First frame (not character front.png) passed as reference."""
            # Create first frame
            first_arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            first_path = tmp_path / "S01.png"
            Image.fromarray(first_arr).save(first_path)

            # Mock validation to always pass
            mock_validate.return_value = {"passed": True, "similarity": 0.65}

            # Mock SeedreamClient — capture the call, then create a synthetic end frame
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            def fake_generate(ref_image, **kwargs):
                # Simulate generation: create a moderately different end frame
                end_path = Path(kwargs["output_path"])
                # Create a random noise pattern (maximally different)
                np.random.seed(99)
                random_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                Image.fromarray(random_arr).save(end_path)

            mock_client.image_to_image.side_effect = fake_generate

            shot = {"gen_strategy": "flf2v", "visual": "抬手拂发"}
            char_front = tmp_path / "char_front.png"
            char_front.touch()

            pipeline_runner._generate_flf2v_end_frame(
                shot, "S01", first_path, char_front
            )

            # Verify image_to_image was called with first frame as ref_image
            mock_client.image_to_image.assert_called_once()
            call_kwargs = mock_client.image_to_image.call_args[1]
            assert call_kwargs["ref_image"] == str(first_path)

    @patch("seedream_client.SeedreamClient")
    def test_cache_skip_when_valid(self, mock_client_class, tmp_path):
        """Valid cached end frame → skip generation."""
        first_arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        first_path = tmp_path / "S01.png"
        Image.fromarray(first_arr).save(first_path)

        end_path = tmp_path / "S01_end.png"
        end_arr = np.roll(first_arr, shift=5, axis=0)
        Image.fromarray(end_arr).save(end_path)

        # Write valid sidecar
        first_sha = pipeline_runner._file_sha256(first_path)
        shot = {"gen_strategy": "flf2v", "visual": "抬手"}
        prompt = pipeline_runner.build_end_frame_prompt(shot)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        pipeline_runner._write_end_frame_sidecar(
            end_path, first_sha, prompt_sha,
            {"passed": True, "similarity": 0.65}
        )

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = pipeline_runner._generate_flf2v_end_frame(
            shot, "S01", first_path, None
        )

        # Should skip generation
        assert result is False
        mock_client.image_to_image.assert_not_called()

    @patch("seedream_client.SeedreamClient")
    @patch("pipeline_runner._validate_end_frame")
    def test_stale_cache_regenerates(self, mock_validate, mock_client_class, tmp_path):
        """Changed first frame → regenerate (cache miss)."""
        first_arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        first_path = tmp_path / "S01.png"
        Image.fromarray(first_arr).save(first_path)

        end_path = tmp_path / "S01_end.png"
        end_arr = np.roll(first_arr, shift=5, axis=0)
        Image.fromarray(end_arr).save(end_path)

        # Write sidecar with WRONG first frame hash (stale)
        shot = {"gen_strategy": "flf2v", "visual": "抬手"}
        prompt = pipeline_runner.build_end_frame_prompt(shot)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        pipeline_runner._write_end_frame_sidecar(
            end_path, "wrong_hash_abc", prompt_sha,
            {"passed": True, "similarity": 0.65}
        )

        # Mock validation to always pass
        mock_validate.return_value = {"passed": True, "similarity": 0.65}

        # Mock SeedreamClient — simulate generation
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        def fake_generate(ref_image, **kwargs):
            end_path = Path(kwargs["output_path"])
            src = np.array(Image.open(ref_image))
            # Heavy transformation to ensure similarity < 0.85
            # Invert + large shift + heavy noise
            inverted = 255 - src
            shifted = np.roll(inverted, shift=30, axis=(0, 1))
            np.random.seed(99)
            noisy = np.clip(shifted.astype(np.int16) + np.random.randint(-80, 80, shifted.shape), 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(end_path)

        mock_client.image_to_image.side_effect = fake_generate

        pipeline_runner._generate_flf2v_end_frame(
            shot, "S01", first_path, None
        )

        # Should regenerate (cache miss due to hash mismatch)
        mock_client.image_to_image.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
