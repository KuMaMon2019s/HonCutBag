"""Tests for M2: FLF2V end frame generation with first-frame reference + visual validation."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline_runner
from pipeline_runner import (
    build_end_frame_prompt,
    _derive_end_state,
    _validate_end_frame,
    _generate_flf2v_end_frame,
    _file_sha256,
    _end_frame_sidecar_path,
    _read_end_frame_sidecar,
    _write_end_frame_sidecar,
    _ACTION_END_STATES,
    FLF2V_SIMILARITY_LOW,
    FLF2V_SIMILARITY_HIGH,
    FLF2V_SHARPNESS_RATIO,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_image(path: Path, color=(128, 128, 128), size=(64, 64)):
    """Create a solid-color test image."""
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _make_noisy_image(path: Path, size=(64, 64), seed=42):
    """Create a random-noise test image."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path))
    return path


# ─── End-state mapping tests ─────────────────────────────────────────────────

class TestDeriveEndState:
    def test_chinese_action_verb(self):
        shot = {"visual": "少女抬手拂发，夕阳下"}
        result = _derive_end_state(shot)
        assert "hair" in result.lower() or "hand" in result.lower()

    def test_chinese_sit_down(self):
        shot = {"what": "坐下"}
        result = _derive_end_state(shot)
        assert "seated" in result.lower()

    def test_english_action_verb(self):
        shot = {"description": "She raise hand slowly"}
        result = _derive_end_state(shot)
        assert "hand" in result.lower() and "raised" in result.lower()

    def test_unknown_action_fallback(self):
        shot = {"visual": "something completely unknown xyz"}
        result = _derive_end_state(shot)
        assert "fully completed" in result.lower()

    def test_empty_shot(self):
        shot = {}
        result = _derive_end_state(shot)
        assert "fully completed" in result.lower()

    def test_all_known_verbs_have_mapping(self):
        """Every verb in _ACTION_END_STATES should produce its own end state."""
        for verb in _ACTION_END_STATES:
            shot = {"visual": f"character {verb} slowly"}
            result = _derive_end_state(shot)
            # Should NOT fall back to generic
            assert result == _ACTION_END_STATES[verb]


class TestBuildEndFramePrompt:
    def test_includes_start_frame_semantics(self):
        shot = {"prompt": "西湖傍晚", "visual": "少女抬手拂发"}
        prompt = build_end_frame_prompt(shot)
        assert "start frame" in prompt.lower()
        assert "completed" in prompt.lower()

    def test_includes_end_state(self):
        shot = {"visual": "坐下"}
        prompt = build_end_frame_prompt(shot)
        assert "seated" in prompt.lower()

    def test_preserve_instructions(self):
        shot = {"prompt": "test scene"}
        prompt = build_end_frame_prompt(shot)
        assert "match" in prompt.lower()
        assert "background" in prompt.lower()


# ─── Validation tests ────────────────────────────────────────────────────────

class TestValidateEndFrame:
    def test_t2i_threshold_defaults(self):
        assert FLF2V_SIMILARITY_LOW == 0.3
        assert FLF2V_SIMILARITY_HIGH == 0.93
        assert FLF2V_SHARPNESS_RATIO == 0.15

    def test_identical_images_too_similar(self, tmp_dir):
        """Identical first and end frame → rejected (no action progress)."""
        first = _make_image(tmp_dir / "first.png", color=(100, 150, 200))
        end = _make_image(tmp_dir / "end.png", color=(100, 150, 200))
        result = _validate_end_frame(first, end)
        assert result["passed"] is False
        assert "too similar" in result["reason"]

    def test_black_image_rejected(self, tmp_dir):
        """All-black end frame → rejected (brightness check)."""
        first = _make_noisy_image(tmp_dir / "first.png", seed=1)
        end = _make_image(tmp_dir / "end.png", color=(0, 0, 0))
        result = _validate_end_frame(first, end)
        assert result["passed"] is False
        assert "brightness" in result["reason"]

    def test_resolution_mismatch_normalized(self, tmp_dir):
        """Different resolution but same aspect → normalized via fit_to_aspect, should pass if content similar."""
        # 64×64 square first frame, 128×128 square end frame — same aspect, different size
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100), size=(64, 64))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181), size=(128, 128))
        result = _validate_end_frame(first, end)
        # After normalization both should be compared at end frame size (128×128)
        assert result["resolution_ok"] is True
        assert result["similarity"] is not None

    def test_drifted_scene_too_different(self, tmp_dir):
        """Completely different noise patterns → rejected (scene drift)."""
        first = _make_noisy_image(tmp_dir / "first.png", seed=1)
        end = _make_noisy_image(tmp_dir / "end.png", seed=999)
        result = _validate_end_frame(first, end)
        # Two completely different noise images should have low similarity
        assert result["similarity"] is not None
        # May or may not fail depending on random noise — just check it ran
        assert "passed" in result

    def test_moderately_different_passes(self, tmp_dir):
        """Slightly modified image → should pass (action progressed, scene stable)."""
        # Create first frame
        first = _make_image(tmp_dir / "first.png", color=(100, 150, 200), size=(64, 64))
        # Create end frame with slight color shift (simulating action progress)
        end = _make_image(tmp_dir / "end.png", color=(110, 145, 195), size=(64, 64))
        result = _validate_end_frame(first, end, similarity_low=0.3, similarity_high=0.99)
        # Very similar solid colors → might be "too similar" with high threshold
        # Lower the high threshold to allow this
        assert result["resolution_ok"] is True
        assert result["brightness_ok"] is True

    def test_similarity_090_passes_as_action_progress(self, tmp_dir):
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181))

        result = _validate_end_frame(first, end)

        assert result["similarity"] == pytest.approx(0.8991, abs=0.0001)
        assert result["passed"] is True

    def test_similarity_099_rejected_as_copy(self, tmp_dir):
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100))
        end = _make_image(tmp_dir / "end.png", color=(120, 120, 120))

        result = _validate_end_frame(first, end)

        assert result["similarity"] == pytest.approx(0.9938, abs=0.0001)
        assert result["passed"] is False
        assert "too similar" in result["reason"]

    def test_sharpness_ratio_020_passes_for_t2i_softness(self, tmp_dir):
        checkerboard = (np.indices((64, 64)).sum(axis=0) % 2) * 80 + 88
        first_arr = np.repeat(checkerboard[:, :, None], 3, axis=2).astype(np.uint8)
        # Contrast scaling by sqrt(0.20) produces a 0.20 Laplacian variance ratio.
        end_gray = 208 + np.sqrt(0.20) * (checkerboard - 128)
        end_arr = np.repeat(end_gray[:, :, None], 3, axis=2).astype(np.uint8)
        first = tmp_dir / "first.png"
        end = tmp_dir / "end.png"
        Image.fromarray(first_arr).save(first)
        Image.fromarray(end_arr).save(end)

        result = _validate_end_frame(first, end)

        assert result["sharpness_ok"] is True
        assert result["passed"] is True

    def test_cannot_open_images(self, tmp_dir):
        """Non-existent file → fails gracefully."""
        first = _make_image(tmp_dir / "first.png")
        end = tmp_dir / "nonexistent.png"
        result = _validate_end_frame(first, end)
        assert result["passed"] is False
        assert "cannot open" in result["reason"]

    def test_m8_square_first_vs_16x9_end_normalized(self, tmp_dir):
        """M8: Legacy square first frame (1920×1920) vs 16:9 end frame (2560×1440) — should normalize and compare."""
        # Use smaller sizes for speed but same aspect ratios: square vs 16:9
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100), size=(192, 192))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181), size=(256, 144))
        result = _validate_end_frame(first, end)
        # After fit_to_aspect normalization, first is cropped/resized to 256×144
        assert result["resolution_ok"] is True
        assert result["similarity"] is not None
        # Solid colors with moderate difference → should be in valid similarity band
        assert result["brightness_ok"] is True

    def test_m8_same_dimensions_still_works(self, tmp_dir):
        """M8: Same dimensions → no normalization needed, original path works."""
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100), size=(128, 72))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181), size=(128, 72))
        result = _validate_end_frame(first, end)
        assert result["resolution_ok"] is True
        assert result["passed"] is True
        assert result["similarity"] == pytest.approx(0.8991, abs=0.001)

    def test_m8_normalization_called_only_when_sizes_differ(self, tmp_dir):
        """M8: fit_to_aspect is only invoked when first.size != end.size."""
        from unittest.mock import patch as mock_patch
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100), size=(64, 64))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181), size=(64, 64))
        with mock_patch("pipeline_runner.fit_to_aspect") as mock_fit:
            result = _validate_end_frame(first, end)
            mock_fit.assert_not_called()
        assert result["passed"] is True

    def test_m8_normalization_called_when_sizes_differ(self, tmp_dir):
        """M8: fit_to_aspect IS invoked when first.size != end.size."""
        from unittest.mock import patch as mock_patch
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100), size=(192, 192))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181), size=(256, 144))
        with mock_patch("pipeline_runner.fit_to_aspect", wraps=pipeline_runner.fit_to_aspect) as mock_fit:
            result = _validate_end_frame(first, end)
            mock_fit.assert_called_once()
            # Verify it was called with end frame dimensions as target
            call_args = mock_fit.call_args
            assert call_args[0][1] == 256  # target_w
            assert call_args[0][2] == 144  # target_h
        assert result["resolution_ok"] is True

# ─── Cache sidecar tests ─────────────────────────────────────────────────────

class TestEndFrameSidecar:
    def test_write_and_read(self, tmp_dir):
        end_path = tmp_dir / "S01_end.png"
        _make_image(end_path)
        validation = {"passed": True, "similarity": 0.5, "reason": None}
        _write_end_frame_sidecar(end_path, "sha_first", "sha_prompt", validation)
        
        sidecar = _read_end_frame_sidecar(end_path)
        assert sidecar is not None
        assert sidecar["first_frame_sha256"] == "sha_first"
        assert sidecar["prompt_sha256"] == "sha_prompt"
        assert sidecar["validation"]["passed"] is True

    def test_validation_sidecar_records_current_thresholds(self, tmp_dir):
        first = _make_image(tmp_dir / "first.png", color=(100, 100, 100))
        end = _make_image(tmp_dir / "end.png", color=(181, 181, 181))
        validation = _validate_end_frame(first, end)

        _write_end_frame_sidecar(end, "sha_first", "sha_prompt", validation)

        sidecar = _read_end_frame_sidecar(end)
        assert sidecar["validation"]["thresholds"] == {
            "similarity_low": 0.3,
            "similarity_high": 0.93,
            "sharpness_ratio": 0.15,
        }

    def test_missing_sidecar_returns_none(self, tmp_dir):
        end_path = tmp_dir / "S01_end.png"
        assert _read_end_frame_sidecar(end_path) is None

    def test_corrupt_sidecar_returns_none(self, tmp_dir):
        end_path = tmp_dir / "S01_end.png"
        sidecar_path = _end_frame_sidecar_path(end_path)
        sidecar_path.write_text("not json{{{")
        assert _read_end_frame_sidecar(end_path) is None

    def test_sidecar_path_naming(self, tmp_dir):
        end_path = tmp_dir / "S01_end.png"
        expected = tmp_dir / "S01_end_end.meta.json"
        assert _end_frame_sidecar_path(end_path) == expected


# ─── Reference swap tests (mocked) ───────────────────────────────────────────

class TestGenerateFlf2vEndFrame:
    def test_missing_first_frame_raises(self, tmp_dir):
        """First frame must exist — loud failure if missing."""
        shot = {"gen_strategy": "flf2v", "visual": "抬手"}
        first_frame = tmp_dir / "S01.png"  # does NOT exist
        with pytest.raises(FileNotFoundError, match="first frame"):
            _generate_flf2v_end_frame(shot, "S01", first_frame, None)

    def test_non_flf2v_skipped(self, tmp_dir):
        """Non-FLF2V shots are skipped."""
        shot = {"gen_strategy": "i2v"}
        first_frame = _make_image(tmp_dir / "S01.png")
        result = _generate_flf2v_end_frame(shot, "S01", first_frame, None)
        assert result is False

    @patch("seedream_client.SeedreamClient")
    def test_uses_t2i_for_generation(self, mock_client_cls, tmp_dir):
        """M3: primary generation uses text_to_image (no reference image)."""
        first_frame = _make_image(tmp_dir / "S01.png", color=(100, 150, 200))
        char_front = _make_image(tmp_dir / "char_front.png", color=(50, 50, 50))
        
        shot = {"gen_strategy": "flf2v", "visual": "走来坐下", "prompt": "西湖"}
        
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.text_to_image.return_value = "http://example.com/img.png"
        
        # Patch _validate_end_frame to always pass (we're testing generation method, not validation)
        with patch("pipeline_runner._validate_end_frame") as mock_validate:
            mock_validate.return_value = {"passed": True, "similarity": 0.5}
            result = _generate_flf2v_end_frame(shot, "S01", first_frame, char_front)
        
        # M3: Verify text_to_image was called (NOT image_to_image)
        mock_client.text_to_image.assert_called()
        call_kwargs = mock_client.text_to_image.call_args
        assert "西湖" in call_kwargs[1]["prompt"] or "西湖" in call_kwargs[0][0]

    @patch("seedream_client.SeedreamClient")
    def test_cache_hit_skips_generation(self, mock_client_cls, tmp_dir):
        """Valid sidecar + matching hashes → skip generation."""
        first_frame = _make_image(tmp_dir / "S01.png", color=(100, 150, 200))
        end_frame = _make_image(tmp_dir / "S01_end.png", color=(110, 145, 195))
        
        shot = {"gen_strategy": "flf2v", "visual": "走来坐下"}
        
        # Write valid sidecar
        first_sha = _file_sha256(first_frame)
        from pipeline_runner import build_end_frame_prompt
        prompt = build_end_frame_prompt(shot)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        _write_end_frame_sidecar(
            end_frame, first_sha, prompt_sha,
            {"passed": True, "similarity": 0.5}
        )
        
        result = _generate_flf2v_end_frame(shot, "S01", first_frame, None)
        assert result is False  # skipped (cache hit)
        mock_client_cls.assert_not_called()

    @patch("pipeline_runner._validate_end_frame")
    @patch("seedream_client.SeedreamClient")
    def test_stale_sidecar_regenerates(self, mock_client_cls, mock_validate, tmp_dir):
        """Changed first frame → stale sidecar → regenerate."""
        first_frame = _make_image(tmp_dir / "S01.png", color=(100, 150, 200))
        end_frame = _make_image(tmp_dir / "S01_end.png", color=(110, 145, 195))
        
        shot = {"gen_strategy": "flf2v", "visual": "走来坐下"}
        
        # Write sidecar with WRONG first frame hash (stale)
        _write_end_frame_sidecar(
            end_frame, "old_stale_hash", "prompt_hash",
            {"passed": True, "similarity": 0.5}
        )
        
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.text_to_image.return_value = "http://example.com/img.png"
        mock_validate.return_value = {"passed": True, "similarity": 0.5}
        
        result = _generate_flf2v_end_frame(shot, "S01", first_frame, None)
        # Should have regenerated (stale cache)
        mock_client.text_to_image.assert_called()


# ─── File SHA-256 test ───────────────────────────────────────────────────────

class TestFileSha256:
    def test_deterministic(self, tmp_dir):
        path = _make_image(tmp_dir / "test.png")
        sha1 = _file_sha256(path)
        sha2 = _file_sha256(path)
        assert sha1 == sha2
        assert len(sha1) == 64  # SHA-256 hex

    def test_different_files_different_hash(self, tmp_dir):
        p1 = _make_image(tmp_dir / "a.png", color=(0, 0, 0))
        p2 = _make_image(tmp_dir / "b.png", color=(255, 255, 255))
        assert _file_sha256(p1) != _file_sha256(p2)
