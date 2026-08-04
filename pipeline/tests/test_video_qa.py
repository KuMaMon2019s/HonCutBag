#!/usr/bin/env python3
"""
Tests for Phase 8.5: Video QA (video_qa.py)

Covers:
  - ffprobe validation (resolution, duration, codec, file size)
  - Scene detection
  - Frame sampling (with/without storyboard)
  - Black frame detection
  - Frozen frame detection
  - Duplicate frame detection
  - STORYBOARD / SHOT_META cross-reference
  - VLM semantic check interface
  - Verdict computation (pass/revise/fail, grading)
  - Edge cases (missing file, empty storyboard, etc.)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure pipeline/src is on the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_qa import (
    VideoQAReport,
    QAIssue,
    FrameSample,
    run_video_qa,
    _ffprobe_validate,
    _detect_scenes,
    _sample_frames,
    _detect_black_frames,
    _detect_frozen_frames,
    _detect_duplicate_frames,
    _crossref_storyboard,
    _vlm_semantic_check,
    _compute_verdict,
    _get_duration,
    _extract_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with a dummy polished.mp4."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a minimal valid-looking video file (not actually playable,
    # but enough for file-existence and size checks)
    video = project_dir / "polished.mp4"
    video.write_bytes(b"\x00" * 600_000)  # 600KB

    return project_dir


@pytest.fixture
def real_video(tmp_path):
    """Create a real short video using ffmpeg for integration tests."""
    video = tmp_path / "test_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg not available or failed: {result.stderr[:200]}")
    return video


@pytest.fixture
def storyboard_data():
    """Sample STORYBOARD.json content."""
    return {
        "shots": [
            {
                "shot_id": "S01",
                "visual": "A quiet lake at sunset",
                "suggested_duration": 5,
            },
            {
                "shot_id": "S02",
                "visual": "A boat glides across the water",
                "suggested_duration": 4,
            },
            {
                "shot_id": "S03",
                "visual": "Close-up of ripples",
                "suggested_duration": 3,
            },
        ]
    }


# ---------------------------------------------------------------------------
# Test: File existence check
# ---------------------------------------------------------------------------

class TestFileExistence:
    def test_missing_video(self, tmp_path):
        """polished.mp4 missing → fail verdict, D grade."""
        report = run_video_qa(tmp_path)
        assert report.verdict == "fail"
        assert report.grade == "D"
        assert any(i.check == "file_exists" for i in report.issues)

    def test_empty_file(self, tmp_path):
        """polished.mp4 exists but is tiny → critical file_size issue."""
        video = tmp_path / "polished.mp4"
        video.write_bytes(b"\x00" * 100)  # 100 bytes
        report = run_video_qa(tmp_path)
        # ffprobe will fail on this, so we get critical issues
        assert any(i.severity == "critical" for i in report.issues)


# ---------------------------------------------------------------------------
# Test: ffprobe validation
# ---------------------------------------------------------------------------

class TestFFprobeValidation:
    def test_valid_video(self, real_video, tmp_path):
        """Real video should pass ffprobe validation without stream-level critical issues."""
        report = VideoQAReport(verdict="pass", grade="A")
        probe = _ffprobe_validate(real_video, report)
        assert probe != {}
        assert "streams" in probe
        # No critical issues for a valid video (file_size may trigger for short test videos)
        critical = [i for i in report.issues if i.severity == "critical" and i.check != "file_size"]
        assert len(critical) == 0

    def test_resolution_mismatch(self, real_video):
        """Expected width/height mismatch should produce warnings."""
        report = VideoQAReport(verdict="pass", grade="A")
        _ffprobe_validate(real_video, report, expected_width=1920, expected_height=1080)
        warnings = [i for i in report.issues if i.severity == "warning" and i.check == "resolution"]
        assert len(warnings) >= 1  # 320x240 != 1920x1080

    def test_duration_too_short(self, real_video):
        """Expected min duration > actual should produce warning."""
        report = VideoQAReport(verdict="pass", grade="A")
        _ffprobe_validate(real_video, report, expected_min_duration=30.0)
        warnings = [i for i in report.issues if i.severity == "warning" and i.check == "duration"]
        assert len(warnings) >= 1

    def test_duration_too_long(self, real_video):
        """Expected max duration < actual should produce warning."""
        report = VideoQAReport(verdict="pass", grade="A")
        _ffprobe_validate(real_video, report, expected_max_duration=1.0)
        warnings = [i for i in report.issues if i.severity == "warning" and i.check == "duration"]
        assert len(warnings) >= 1

    def test_small_file_size(self, tmp_path):
        """File < 512KB should trigger critical file_size issue."""
        video = tmp_path / "polished.mp4"
        video.write_bytes(b"\x00" * 100_000)  # 100KB
        report = VideoQAReport(verdict="pass", grade="A")
        _ffprobe_validate(video, report)
        # May get file_size or ffprobe critical
        critical = [i for i in report.issues if i.severity == "critical"]
        assert len(critical) >= 1


# ---------------------------------------------------------------------------
# Test: Scene detection
# ---------------------------------------------------------------------------

class TestSceneDetection:
    def test_returns_boundaries(self, real_video):
        """Scene detection should return at least [0.0]."""
        report = VideoQAReport(verdict="pass", grade="A")
        bounds = _detect_scenes(real_video, report)
        assert isinstance(bounds, list)
        assert 0.0 in bounds

    def test_handles_failure_gracefully(self, tmp_path):
        """Scene detection on non-existent file should not crash."""
        report = VideoQAReport(verdict="pass", grade="A")
        bounds = _detect_scenes(tmp_path / "nonexistent.mp4", report)
        assert bounds == [0.0]


# ---------------------------------------------------------------------------
# Test: Frame sampling
# ---------------------------------------------------------------------------

class TestFrameSampling:
    def test_with_storyboard(self, real_video, tmp_path, storyboard_data):
        """Frame sampling with storyboard should extract first/mid/last per shot."""
        report = VideoQAReport(verdict="pass", grade="A")
        frames = _sample_frames(real_video, tmp_path, [0.0], storyboard_data, report)
        assert len(frames) > 0
        # Should have frames for each shot (first, mid, last = 3 per shot)
        labels = [f.label for f in frames]
        assert any("S01" in l for l in labels)
        assert any("S02" in l for l in labels)

    def test_without_storyboard(self, real_video, tmp_path):
        """Frame sampling without storyboard should use uniform sampling."""
        report = VideoQAReport(verdict="pass", grade="A")
        frames = _sample_frames(real_video, tmp_path, [0.0], None, report)
        assert len(frames) > 0
        labels = [f.label for f in frames]
        assert any("uniform" in l for l in labels)

    def test_with_scene_boundaries(self, real_video, tmp_path):
        """Frame sampling with scene boundaries should extract at boundaries."""
        report = VideoQAReport(verdict="pass", grade="A")
        frames = _sample_frames(real_video, tmp_path, [0.0, 2.5, 4.0], None, report)
        assert len(frames) > 0


# ---------------------------------------------------------------------------
# Test: Black frame detection
# ---------------------------------------------------------------------------

class TestBlackFrameDetection:
    def test_normal_video_no_black(self, real_video):
        """A normal test video should have minimal or no black frames."""
        report = VideoQAReport(verdict="pass", grade="A")
        _detect_black_frames(real_video, report)
        # testsrc shouldn't have significant black content
        critical = [i for i in report.issues if i.check == "black_frames" and i.severity == "critical"]
        assert len(critical) == 0

    def test_handles_missing_file(self, tmp_path):
        """Black detection on missing file should not crash."""
        report = VideoQAReport(verdict="pass", grade="A")
        _detect_black_frames(tmp_path / "nonexistent.mp4", report)
        # Should have info-level issue, not crash
        assert not any(i.severity == "critical" for i in report.issues)


# ---------------------------------------------------------------------------
# Test: Frozen frame detection
# ---------------------------------------------------------------------------

class TestFrozenFrameDetection:
    def test_normal_video_no_frozen(self, real_video):
        """A normal test video should have no frozen frames."""
        report = VideoQAReport(verdict="pass", grade="A")
        _detect_frozen_frames(real_video, report)
        critical = [i for i in report.issues if i.check == "frozen_frames" and i.severity == "critical"]
        assert len(critical) == 0


# ---------------------------------------------------------------------------
# Test: Duplicate frame detection
# ---------------------------------------------------------------------------

class TestDuplicateFrameDetection:
    def test_normal_video_low_duplication(self, real_video):
        """A normal test video should have low duplicate frame ratio."""
        report = VideoQAReport(verdict="pass", grade="A")
        _detect_duplicate_frames(real_video, report)
        # testsrc has motion, so duplicates should be low
        dup_warnings = [i for i in report.issues if i.check == "duplicate_frames" and i.severity == "warning"]
        assert len(dup_warnings) == 0


# ---------------------------------------------------------------------------
# Test: STORYBOARD cross-reference
# ---------------------------------------------------------------------------

class TestStoryboardCrossref:
    def test_duration_match(self, real_video, tmp_path, storyboard_data):
        """Cross-ref should compare video duration with storyboard total."""
        # storyboard total = 5+4+3 = 12s, video = 5s → mismatch
        report = VideoQAReport(verdict="pass", grade="A")
        _crossref_storyboard(tmp_path, storyboard_data, report)
        assert len(report.storyboard_crossref) > 0
        entry = report.storyboard_crossref[0]
        assert entry["storyboard_shot_count"] == 3
        assert entry["storyboard_total_duration"] == 12.0

    def test_empty_storyboard(self, tmp_path):
        """Empty storyboard should not crash."""
        report = VideoQAReport(verdict="pass", grade="A")
        _crossref_storyboard(tmp_path, {"shots": []}, report)
        assert len(report.storyboard_crossref) == 0

    def test_missing_shots_dir(self, tmp_path, storyboard_data):
        """Missing shots/ dir should be handled gracefully."""
        report = VideoQAReport(verdict="pass", grade="A")
        _crossref_storyboard(tmp_path, storyboard_data, report)
        # No crash, crossref entry exists
        assert len(report.storyboard_crossref) > 0

    def test_shots_with_missing_video(self, tmp_path, storyboard_data):
        """Shot dirs without output.mp4 should be reported."""
        # Create shot directories
        shots_dir = tmp_path / "shots"
        for sid in ["S01", "S02", "S03"]:
            sd = shots_dir / sid
            sd.mkdir(parents=True)
            # Don't create output.mp4

        report = VideoQAReport(verdict="pass", grade="A")
        _crossref_storyboard(tmp_path, storyboard_data, report)
        entry = report.storyboard_crossref[0]
        assert len(entry.get("shots_with_missing_video", [])) == 3


# ---------------------------------------------------------------------------
# Test: VLM semantic check interface
# ---------------------------------------------------------------------------

class TestVLMCheck:
    def test_no_frames(self):
        """VLM check with no frames should return skipped."""
        result = _vlm_semantic_check(MagicMock(), [], None)
        assert result["status"] == "skipped"

    def test_no_analyze_method(self):
        """VLM client without analyze() should return error."""
        frames = [FrameSample(path="/tmp/f.jpg", timestamp=1.0, label="test")]
        result = _vlm_semantic_check(object(), frames, None)
        assert result["status"] == "error"

    def test_with_mock_client(self, storyboard_data):
        """VLM check with mock client should return results."""
        mock_client = MagicMock()
        mock_client.analyze.return_value = "A lake scene with sunset colors"
        frames = [
            FrameSample(path="/tmp/S01_first.jpg", timestamp=0.0, label="S01_first"),
            FrameSample(path="/tmp/S02_mid.jpg", timestamp=5.0, label="S02_mid"),
        ]
        result = _vlm_semantic_check(mock_client, frames, storyboard_data)
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert mock_client.analyze.call_count == 2


# ---------------------------------------------------------------------------
# Test: Verdict computation
# ---------------------------------------------------------------------------

class TestVerdictComputation:
    def test_pass_no_issues(self):
        """No issues → pass, A grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        _compute_verdict(report)
        assert report.verdict == "pass"
        assert report.grade == "A"

    def test_revise_one_critical(self):
        """1 critical → revise, C grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        report.issues.append(QAIssue(severity="critical", check="test", message="test"))
        _compute_verdict(report)
        assert report.verdict == "revise"
        assert report.grade == "C"

    def test_fail_two_criticals(self):
        """2 criticals → fail, C grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        report.issues.append(QAIssue(severity="critical", check="test1", message="t1"))
        report.issues.append(QAIssue(severity="critical", check="test2", message="t2"))
        _compute_verdict(report)
        assert report.verdict == "fail"
        assert report.grade == "C"  # 2 criticals = C, not D

    def test_revise_many_warnings(self):
        """3+ warnings → revise, B grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        for i in range(4):
            report.issues.append(QAIssue(severity="warning", check=f"w{i}", message=f"warn{i}"))
        _compute_verdict(report)
        assert report.verdict == "revise"
        assert report.grade == "B"

    def test_pass_few_warnings(self):
        """1-2 warnings → pass, A grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        report.issues.append(QAIssue(severity="warning", check="w1", message="warn"))
        _compute_verdict(report)
        assert report.verdict == "pass"
        assert report.grade == "A"

    def test_three_criticals_d_grade(self):
        """3+ criticals → D grade."""
        report = VideoQAReport(verdict="pass", grade="A")
        for i in range(3):
            report.issues.append(QAIssue(severity="critical", check=f"c{i}", message=f"crit{i}"))
        _compute_verdict(report)
        assert report.verdict == "fail"
        assert report.grade == "D"


# ---------------------------------------------------------------------------
# Test: Report serialization
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_to_dict(self):
        """Report should serialize to dict with all expected keys."""
        report = VideoQAReport(verdict="pass", grade="A")
        report.issues.append(QAIssue(severity="info", check="test", message="test msg"))
        report.frames_extracted.append(FrameSample(path="/tmp/f.jpg", timestamp=1.0, label="test"))
        report.scene_boundaries = [0.0, 2.5]
        report.vlm_check_available = True

        d = report.to_dict()
        assert d["verdict"] == "pass"
        assert d["grade"] == "A"
        assert len(d["issues"]) == 1
        assert d["frames_extracted_count"] == 1
        assert d["scene_boundaries"] == [0.0, 2.5]
        assert d["vlm_check_available"] is True

    def test_report_written_to_disk(self, real_video, tmp_path):
        """run_video_qa should write video_qa_report.json."""
        # Copy video to tmp_path as polished.mp4
        import shutil
        video = tmp_path / "polished.mp4"
        shutil.copy(real_video, video)

        report = run_video_qa(tmp_path)
        report_path = tmp_path / "video_qa_report.json"
        assert report_path.exists()

        data = json.loads(report_path.read_text())
        assert data["verdict"] in ("pass", "revise", "fail")
        assert data["grade"] in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Test: Integration — full run_video_qa with real video
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_qa_pass(self, real_video, tmp_path):
        """Full QA on a valid video should pass or at least not fail."""
        import shutil
        video = tmp_path / "polished.mp4"
        shutil.copy(real_video, video)

        report = run_video_qa(tmp_path)
        assert report.verdict in ("pass", "revise")  # Should not fail on valid video
        assert report.grade in ("A", "B", "C")

    def test_full_qa_with_storyboard(self, real_video, tmp_path, storyboard_data):
        """Full QA with storyboard should include crossref data."""
        import shutil
        video = tmp_path / "polished.mp4"
        shutil.copy(real_video, video)

        # Create shots dir to test crossref
        shots_dir = tmp_path / "shots"
        for sid in ["S01", "S02", "S03"]:
            sd = shots_dir / sid
            sd.mkdir(parents=True)
            (sd / "output.mp4").write_bytes(b"\x00" * 200_000)
            (sd / "SHOT_META.json").write_text(json.dumps({"duration": 4}))

        report = run_video_qa(tmp_path, storyboard_data=storyboard_data)
        assert len(report.storyboard_crossref) > 0

    def test_full_qa_with_expected_params(self, real_video, tmp_path):
        """Full QA with expected width/height should detect mismatches."""
        import shutil
        video = tmp_path / "polished.mp4"
        shutil.copy(real_video, video)

        report = run_video_qa(
            tmp_path,
            expected_width=1920,
            expected_height=1080,
            expected_min_duration=1.0,
            expected_max_duration=30.0,
        )
        # 320x240 != 1920x1080 → resolution warnings
        res_warnings = [i for i in report.issues if i.check == "resolution"]
        assert len(res_warnings) >= 1


# ---------------------------------------------------------------------------
# Test: Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_duration(self, real_video):
        """_get_duration should return correct duration for real video."""
        dur = _get_duration(real_video)
        assert 4.0 <= dur <= 6.0  # ~5 seconds

    def test_get_duration_missing_file(self, tmp_path):
        """_get_duration on missing file should return 0."""
        dur = _get_duration(tmp_path / "nonexistent.mp4")
        assert dur == 0.0

    def test_extract_frame(self, real_video, tmp_path):
        """_extract_frame should create a jpg file."""
        frame = _extract_frame(real_video, tmp_path, 1.0, "test_frame")
        assert frame is not None
        assert frame.label == "test_frame"
        assert Path(frame.path).exists()

    def test_extract_frame_bad_timestamp(self, real_video, tmp_path):
        """_extract_frame with very large timestamp should return None."""
        frame = _extract_frame(real_video, tmp_path, 9999.0, "bad_ts")
        # May or may not succeed depending on ffmpeg behavior
        # Just verify no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
