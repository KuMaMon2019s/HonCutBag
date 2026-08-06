"""Tests for Phase 8 subtitle burn — captions→segments adapter.

Fix E: HonCut passed 'captions' key but RemotionCaptionBurn expects
'segments' or 'srt_path'. Also, storyboard has no start_time/end_time
so all times were 0. Fix: probe real shot durations via ffprobe,
build cumulative timeline, generate proper segments + SRT fallback.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure pipeline src is importable
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline_runner import _probe_shot_duration, _write_srt, _fmt_srt_time

# ---------------------------------------------------------------------------
# Helper function tests (no external deps)
# ---------------------------------------------------------------------------


class TestFmtSrtTime:
    def test_zero(self):
        from pipeline_runner import _fmt_srt_time
        assert _fmt_srt_time(0) == "00:00:00,000"

    def test_fractional_seconds(self):
        from pipeline_runner import _fmt_srt_time
        assert _fmt_srt_time(1.5) == "00:00:01,500"

    def test_minutes(self):
        from pipeline_runner import _fmt_srt_time
        assert _fmt_srt_time(65.123) == "00:01:05,123"

    def test_hours(self):
        from pipeline_runner import _fmt_srt_time
        assert _fmt_srt_time(3661.5) == "01:01:01,500"


class TestWriteSrt:
    def test_basic_segments(self, tmp_path):
        from pipeline_runner import _write_srt
        srt_path = str(tmp_path / "test.srt")
        segments = [
            {"text": "你好", "start": 0.0, "end": 2.0},
            {"text": "世界", "start": 2.0, "end": 4.0},
        ]
        _write_srt(segments, srt_path)
        content = Path(srt_path).read_text(encoding="utf-8")
        assert "1" in content
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "你好" in content
        assert "2" in content
        assert "00:00:02,000 --> 00:00:04,000" in content
        assert "世界" in content

    def test_empty_segments(self, tmp_path):
        from pipeline_runner import _write_srt
        srt_path = str(tmp_path / "empty.srt")
        _write_srt([], srt_path)
        content = Path(srt_path).read_text(encoding="utf-8")
        assert content.strip() == ""


class TestProbeShotDuration:
    def test_missing_file_returns_fallback(self, tmp_path):
        from pipeline_runner import _probe_shot_duration
        shots_dir = tmp_path / "shots"
        shots_dir.mkdir()
        dur = _probe_shot_duration(shots_dir, 1)
        assert dur == 2.0

    def test_real_file_with_ffprobe(self, tmp_path):
        """Test with a real ffprobe call on a tiny generated video."""
        from pipeline_runner import _probe_shot_duration
        import subprocess

        shots_dir = tmp_path / "shots" / "S01"
        shots_dir.mkdir(parents=True)
        video_path = shots_dir / "output.mp4"

        # Generate a 1-second black video with ffmpeg
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "color=c=black:s=64x64:d=1:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available")

        dur = _probe_shot_duration(tmp_path / "shots", 1)
        assert 0.9 < dur < 1.2  # ~1 second


# ---------------------------------------------------------------------------
# Integration: segment building from storyboard
# ---------------------------------------------------------------------------


class TestSegmentBuilding:
    """Test that the Phase 8 subtitle code builds correct segments."""

    def test_segments_have_cumulative_timing(self, tmp_path):
        """Verify cumulative timeline: each segment starts after previous ends."""
        import subprocess

        # Create fake shot videos (1s each)
        shots_dir = tmp_path / "shots"
        for i in range(1, 4):
            shot_dir = shots_dir / f"S{i:02d}"
            shot_dir.mkdir(parents=True)
            video_path = shot_dir / "output.mp4"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=64x64:d=1:r=24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Create storyboard
        storyboard = {
            "shots": [
                {"id": 1, "caption": "第一段"},
                {"id": 2, "caption": "第二段"},
                {"id": 3, "caption": "第三段"},
            ]
        }
        sb_path = tmp_path / "STORYBOARD.json"
        sb_path.write_text(json.dumps(storyboard), encoding="utf-8")

        # Simulate the segment-building logic from Phase 8
        from pipeline_runner import _probe_shot_duration

        with open(sb_path, 'r', encoding='utf-8') as f:
            storyboard_data = json.load(f)

        sb_shots = storyboard_data.get('shots', [])
        segments = []
        cumulative_start = 0.0

        for i, shot in enumerate(sb_shots):
            caption_text = shot.get('caption', '')
            if not caption_text:
                shot_dur = _probe_shot_duration(shots_dir, i + 1)
                cumulative_start += shot_dur
                continue

            shot_dur = _probe_shot_duration(shots_dir, i + 1)
            seg_start = cumulative_start
            seg_end = cumulative_start + shot_dur

            chars = list(caption_text)
            per_char = shot_dur / max(len(chars), 1)
            words = []
            for ci, ch in enumerate(chars):
                words.append({
                    "word": ch,
                    "start": seg_start + ci * per_char,
                    "end": seg_start + (ci + 1) * per_char,
                })

            segments.append({
                "text": caption_text,
                "start": seg_start,
                "end": seg_end,
                "words": words,
            })
            cumulative_start = seg_end

        # Verify
        assert len(segments) == 3

        # Cumulative timing: each starts where previous ended
        assert segments[0]["start"] == pytest.approx(0.0, abs=0.1)
        assert segments[0]["end"] == pytest.approx(1.0, abs=0.2)
        assert segments[1]["start"] == pytest.approx(1.0, abs=0.2)
        assert segments[1]["end"] == pytest.approx(2.0, abs=0.2)
        assert segments[2]["start"] == pytest.approx(2.0, abs=0.2)
        assert segments[2]["end"] == pytest.approx(3.0, abs=0.2)

        # Word-level entries exist
        for seg in segments:
            assert len(seg["words"]) == len(seg["text"])
            for w in seg["words"]:
                assert "word" in w
                assert "start" in w
                assert "end" in w
                assert w["end"] > w["start"]

    def test_shots_without_caption_still_advance_timeline(self, tmp_path):
        """Shots without captions still advance the cumulative clock."""
        import subprocess

        shots_dir = tmp_path / "shots"
        for i in range(1, 3):
            shot_dir = shots_dir / f"S{i:02d}"
            shot_dir.mkdir(parents=True)
            video_path = shot_dir / "output.mp4"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=64x64:d=1:r=24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        storyboard = {
            "shots": [
                {"id": 1},  # no caption
                {"id": 2, "caption": "有字幕"},
            ]
        }
        sb_path = tmp_path / "STORYBOARD.json"
        sb_path.write_text(json.dumps(storyboard), encoding="utf-8")

        from pipeline_runner import _probe_shot_duration

        with open(sb_path, 'r', encoding='utf-8') as f:
            storyboard_data = json.load(f)

        sb_shots = storyboard_data.get('shots', [])
        segments = []
        cumulative_start = 0.0

        for i, shot in enumerate(sb_shots):
            caption_text = shot.get('caption', '')
            if not caption_text:
                shot_dur = _probe_shot_duration(shots_dir, i + 1)
                cumulative_start += shot_dur
                continue

            shot_dur = _probe_shot_duration(shots_dir, i + 1)
            segments.append({
                "text": caption_text,
                "start": cumulative_start,
                "end": cumulative_start + shot_dur,
            })
            cumulative_start += shot_dur

        # Only 1 segment (shot 2), but it starts after shot 1's duration
        assert len(segments) == 1
        assert segments[0]["start"] == pytest.approx(1.0, abs=0.2)
        assert segments[0]["end"] == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# RemotionCaptionBurn interface compatibility
# ---------------------------------------------------------------------------


class TestRemotionCaptionBurnInterface:
    """Verify the adapter passes correct keys to RemotionCaptionBurn."""

    def test_segments_key_accepted(self):
        """RemotionCaptionBurn.execute() must accept 'segments' key."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        burner = RemotionCaptionBurn()
        schema = burner.input_schema
        # 'segments' must be in the schema
        assert "segments" in schema["properties"]

    def test_srt_path_key_accepted(self):
        """RemotionCaptionBurn.execute() must accept 'srt_path' key."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        burner = RemotionCaptionBurn()
        schema = burner.input_schema
        assert "srt_path" in schema["properties"]

    def test_captions_key_NOT_accepted(self):
        """RemotionCaptionBurn does NOT accept 'captions' key — this was the bug."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        burner = RemotionCaptionBurn()
        schema = burner.input_schema
        assert "captions" not in schema["properties"]

    def test_style_key_NOT_accepted(self):
        """RemotionCaptionBurn does NOT accept 'style' dict — this was also wrong."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        burner = RemotionCaptionBurn()
        schema = burner.input_schema
        assert "style" not in schema["properties"]

    def test_caption_pages_preserve_srt_cue_boundaries(self, tmp_path):
        """Separate SRT cues must not be merged into one timed overlay."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        srt_path = tmp_path / "two_cues.srt"
        srt_path.write_text(
            "1\n00:00:00,500 --> 00:00:02,500\n第一条\n\n"
            "2\n00:00:03,000 --> 00:00:05,500\n第二条\n",
            encoding="utf-8",
        )
        burner = RemotionCaptionBurn()
        captions = burner._srt_to_word_captions(str(srt_path))

        assert burner._caption_pages(captions) == [
            ("第一条", 0.5, 2.5),
            ("第二条", 3.0, 5.5),
        ]

    def test_filter_detection_finds_overlay(self):
        """The minimal Homebrew FFmpeg path can feature-detect overlay."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vendor.openmontage.tools.video.remotion_caption_burn import RemotionCaptionBurn

        filters_output = "Filters:\n TS overlay VV->V Overlay a video source.\n"
        completed = MagicMock(stdout=filters_output)
        with patch(
            "vendor.openmontage.tools.video.remotion_caption_burn.subprocess.run",
            return_value=completed,
        ):
            assert "overlay" in RemotionCaptionBurn._ffmpeg_filters()
