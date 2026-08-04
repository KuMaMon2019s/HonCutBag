"""Tests for audio_pipeline.py - silent detection and ambient generation."""
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.audio_pipeline import is_silent_audio, generate_ambient_audio


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_test_video(path: Path, duration: float = 2.0, with_audio: bool = True, silent: bool = False):
    """Helper to create test video files."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={duration}",
    ]
    if with_audio:
        if silent:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        else:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, capture_output=True, check=True, timeout=30)


class TestIsSilentAudio:
    """Test silent audio detection."""

    def test_detects_silent_audio(self, temp_dir):
        """Silent audio track should be detected as silent."""
        video_path = temp_dir / "silent.mp4"
        _create_test_video(video_path, silent=True)
        
        assert is_silent_audio(str(video_path)) is True

    def test_detects_audible_audio(self, temp_dir):
        """Audible audio track should not be detected as silent."""
        video_path = temp_dir / "audible.mp4"
        _create_test_video(video_path, silent=False)
        
        assert is_silent_audio(str(video_path)) is False

    def test_no_audio_stream_is_silent(self, temp_dir):
        """Video without audio stream should be treated as silent."""
        video_path = temp_dir / "no_audio.mp4"
        _create_test_video(video_path, with_audio=False)
        
        assert is_silent_audio(str(video_path)) is True

    def test_nonexistent_file_is_silent(self):
        """Non-existent file should be treated as silent (fail-safe)."""
        assert is_silent_audio("/nonexistent/path.mp4") is True

    def test_custom_threshold(self, temp_dir):
        """Custom threshold should affect detection sensitivity."""
        video_path = temp_dir / "quiet.mp4"
        _create_test_video(video_path, silent=False)
        
        # Very high threshold should detect as silent
        assert is_silent_audio(str(video_path), threshold_db=100.0) is True
        # Very low threshold should detect as audible
        assert is_silent_audio(str(video_path), threshold_db=-100.0) is False


class TestGenerateAmbientAudio:
    """Test ambient audio generation."""

    def test_generates_audio_file(self, temp_dir):
        """Should generate an audio file with correct duration."""
        output_path = temp_dir / "ambient.m4a"
        
        success = generate_ambient_audio(
            duration=3.0,
            output_path=str(output_path),
            scene_hint="lake_evening"
        )
        
        assert success is True
        assert output_path.exists()
        
        # Check duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        assert 2.5 <= duration <= 3.5  # Allow small variance

    def test_different_scenes(self, temp_dir):
        """Different scene hints should all generate valid audio."""
        scenes = ["lake_evening", "forest", "city", "generic", "unknown_scene"]
        
        for scene in scenes:
            output_path = temp_dir / f"ambient_{scene}.m4a"
            success = generate_ambient_audio(
                duration=1.0,
                output_path=str(output_path),
                scene_hint=scene
            )
            assert success is True, f"Failed for scene: {scene}"
            assert output_path.exists(), f"File not created for scene: {scene}"

    def test_invalid_duration_fails(self, temp_dir):
        """Zero or negative duration should fail gracefully."""
        output_path = temp_dir / "invalid.m4a"
        
        assert generate_ambient_audio(0.0, str(output_path)) is False
        assert generate_ambient_audio(-1.0, str(output_path)) is False

    def test_generated_audio_is_audible(self, temp_dir):
        """Generated ambient audio should not be silent."""
        output_path = temp_dir / "audible_ambient.m4a"
        
        success = generate_ambient_audio(
            duration=2.0,
            output_path=str(output_path),
            scene_hint="lake_evening",
            target_db=-20.0
        )
        
        assert success is True
        # The generated audio should NOT be detected as silent
        assert is_silent_audio(str(output_path)) is False
