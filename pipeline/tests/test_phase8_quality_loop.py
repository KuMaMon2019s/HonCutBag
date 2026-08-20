"""Regression coverage for Phase 8 visual-QA → trim/reshoot → assembly."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from phases import pipeline_core
from phases.phase8 import edit_decisions, frame_analysis, story_order_reviewer
from phases.phase8.frame_analysis import decide_shot_action
from utils import shot_embedder
from tools.audio_pipeline import is_silent_audio


def _frame(timestamp: float, *issues: str) -> dict:
    return {"timestamp_s": timestamp, "metrics": {"issues": list(issues)}}


def _write_black_head_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=black:s=320x180:d=0.35:r=24",
            "-f", "lavfi", "-i", "color=red:s=320x180:d=1.65:r=24",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
            "-map", "[v]", str(path),
        ],
        capture_output=True,
        check=True,
    )


def _write_color_video(path: Path, color: str, duration: float = 2.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color={color}:s=320x180:d={duration}:r=24",
            "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration}",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True,
        check=True,
    )


def test_boundary_black_frames_become_trim_decision():
    decision = decide_shot_action(
        5.0,
        [_frame(0.05, "black"), _frame(2.5), _frame(4.9)],
        [{"start": 0.0, "end": 0.4, "duration": 0.4}],
        [],
    )

    assert decision["action"] == "trim"
    assert decision["trim_start_s"] == 0.4
    assert decision["trim_end_s"] == 5.0


def test_interior_black_or_semantic_continuity_failure_requires_reshoot():
    technical = decide_shot_action(
        5.0,
        [_frame(0.1), _frame(2.5, "black"), _frame(4.9)],
        [{"start": 2.0, "end": 2.5, "duration": 0.5}],
        [],
    )
    semantic = decide_shot_action(
        5.0,
        [_frame(0.1), _frame(2.5), _frame(4.9)],
        [],
        [],
        {"verdict": "reshoot", "issues": ["character identity drift"]},
    )

    assert technical["action"] == "reshoot"
    assert semantic["action"] == "reshoot"
    assert "character identity drift" in semantic["reasons"]


def test_semantic_reviewer_receives_time_and_lighting_contract(monkeypatch):
    observed = {}

    class FakeClient:
        def review(self, _frames, prompt):
            observed["prompt"] = prompt
            return '{"verdict":"pass","issues":[],"confidence":0.99}'

    monkeypatch.setattr(
        "clients.ark_multimodal_client.ArkMultimodalClient", FakeClient
    )
    reviewer = frame_analysis._automatic_semantic_reviewer()

    result = reviewer(
        [Path("first.jpg"), Path("last.jpg")],
        {
            "shot_id": "S06",
            "time_of_day": "夜间，雨天",
            "lighting_description": "冷蓝雨夜光，全程无日光",
        },
    )

    assert result["verdict"] == "pass"
    assert '"time_of_day": "夜间，雨天"' in observed["prompt"]
    assert '"lighting_description": "冷蓝雨夜光，全程无日光"' in observed["prompt"]
    assert "daylight/night drift" in observed["prompt"]


def test_dense_analyzer_extracts_frames_and_persists_actionable_report(tmp_path):
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg unavailable")
    shot = tmp_path / "shots" / "S01"
    shot.mkdir(parents=True)
    (shot / "SHOT_META.json").write_text(json.dumps({"shot_id": "S01"}))
    _write_black_head_video(shot / "output.mp4")

    report = frame_analysis.analyze_shot_frames(
        tmp_path / "shots",
        tmp_path / "frame_analysis.json",
        semantic_reviewer=False,
        max_frames=6,
        interval_s=0.4,
    )

    assert report["summary"]["reviewed_shots"] == 1
    assert report["summary"]["sampled_frames"] >= 3
    assert report["shots"]["S01"]["action"] == "trim"
    assert report["shots"]["S01"]["trim_start_s"] > 0.2
    assert (shot / "frames" / "frame_000.jpg").is_file()
    assert json.loads((tmp_path / "frame_analysis.json").read_text())["summary"]["trim"] == ["S01"]


def test_analyzed_black_head_is_physically_removed_from_assembly(tmp_path):
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg unavailable")
    shot = tmp_path / "shots" / "S01"
    shot.mkdir(parents=True)
    (shot / "SHOT_META.json").write_text(json.dumps({"shot_id": "S01"}))
    _write_black_head_video(shot / "output.mp4")

    report = frame_analysis.analyze_shot_frames(
        tmp_path / "shots",
        tmp_path / "frame_analysis.json",
        semantic_reviewer=False,
        max_frames=6,
        interval_s=0.4,
    )
    decisions = edit_decisions.build_edit_decisions(
        tmp_path / "shots",
        target_width=320,
        target_height=180,
        quality_report=report,
        shot_order=["S01"],
    )
    result = edit_decisions.execute_edit_decisions(
        decisions,
        str(tmp_path / "raw_assembly.mp4"),
    )

    assert result["success"] is True
    assert result["segments"] == 1
    assert 1.3 < result["duration"] < 1.8
    assert decisions["cuts"][0]["in_seconds"] >= 0.3


def test_edit_decisions_can_explicitly_assemble_unresolved_best_take(tmp_path):
    shot = tmp_path / "shots" / "S01"
    shot.mkdir(parents=True)
    _write_color_video(shot / "output.mp4", "navy")
    quality = {
        "shots": {
            "S01": {
                "action": "reshoot",
                "reasons": ["action fidelity remains below delivery threshold"],
            }
        }
    }

    with pytest.raises(ValueError, match="still requires reshoot"):
        edit_decisions.build_edit_decisions(
            tmp_path / "shots",
            quality_report=quality,
        )

    decisions = edit_decisions.build_edit_decisions(
        tmp_path / "shots",
        quality_report=quality,
        allow_unresolved_reshoots=True,
    )

    assert decisions["cuts"][0]["quality_action"] == "reshoot"
    assert decisions["metadata"]["allow_unresolved_reshoots"] is True
    assert decisions["metadata"]["unresolved_reshoots"] == ["S01"]


def test_video_only_generation_is_treated_as_silent_for_asr(tmp_path):
    video = tmp_path / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:d=0.2:r=24",
            "-an",
            str(video),
        ],
        capture_output=True,
        check=True,
    )

    assert is_silent_audio(str(video)) is True


def test_xfade_resets_input_timestamps_and_preserves_timeline(tmp_path):
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        pytest.skip("ffmpeg unavailable")
    shots = tmp_path / "shots"
    for name, color in (("S01", "red"), ("S02", "blue"), ("S03", "green")):
        directory = shots / name
        directory.mkdir(parents=True)
        _write_color_video(directory / "output.mp4", color)

    decisions = edit_decisions.build_edit_decisions(
        shots,
        target_width=320,
        target_height=180,
        transition_decisions=[{"decision": "dissolve"}, {"decision": "dissolve"}],
        transition_duration=0.5,
    )
    result = edit_decisions.execute_edit_decisions(
        decisions, str(tmp_path / "xfade.mp4")
    )

    assert result["success"] is True
    assert 4.0 < result["duration"] < 5.2


def test_target_duration_compensates_for_transition_overlap(monkeypatch, tmp_path):
    shots = tmp_path / "shots"
    for index in range(1, 8):
        directory = shots / f"S{index:02d}"
        directory.mkdir(parents=True)
        (directory / "output.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(
        edit_decisions,
        "probe_video",
        lambda _path: {
            "duration": 5.088,
            "width": 1280,
            "height": 720,
            "fps": 24,
            "has_audio": True,
            "has_video": True,
        },
    )
    monkeypatch.setattr(
        edit_decisions,
        "detect_black_frames",
        lambda _path: {"trim_start": 0.1, "trim_end": 0.1},
    )

    decisions = edit_decisions.build_edit_decisions(
        shots,
        transition_decisions=[{"decision": "dissolve"}] * 6,
        target_duration=35,
        transition_duration=0.5,
    )

    speeds = {cut["speed"] for cut in decisions["cuts"]}
    assert len(speeds) == 1
    assert 0.89 < speeds.pop() < 0.91


def test_visual_cut_gets_audio_edge_fade_without_timeline_overlap(monkeypatch, tmp_path):
    shots = tmp_path / "shots"
    for name in ("S01", "S02"):
        directory = shots / name
        directory.mkdir(parents=True)
        (directory / "output.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(
        edit_decisions,
        "probe_video",
        lambda _path: {
            "duration": 5.0,
            "width": 1280,
            "height": 720,
            "fps": 24,
            "has_audio": True,
            "has_video": True,
        },
    )
    monkeypatch.setattr(
        edit_decisions,
        "detect_black_frames",
        lambda _path: {"trim_start": 0.1, "trim_end": 0.1},
    )

    decisions = edit_decisions.build_edit_decisions(
        shots,
        transition_decisions=[{"decision": "cut"}],
        transition_duration=0.5,
    )

    assert decisions["transitions"] == [{
        "index": 0,
        "type": "cut",
        "duration": 0.0,
        "duration_frames": 0,
        "audio_transition": "edge_fade",
        "audio_duration": 0.35,
    }]
    assert decisions["metadata"]["audio_transition_policy"]["visual_cut"] == (
        "equal_power_edge_fade"
    )


def test_all_visual_cuts_still_use_audio_transition_renderer(monkeypatch, tmp_path):
    segments = [str(tmp_path / "one.mp4"), str(tmp_path / "two.mp4")]
    for path in segments:
        Path(path).write_bytes(b"fixture")
    rendered = []
    copied = []
    monkeypatch.setattr(
        edit_decisions,
        "_xfade_chain",
        lambda *_args: rendered.append(True),
    )
    monkeypatch.setattr(
        edit_decisions,
        "_concat_copy",
        lambda *_args: copied.append(True),
    )
    monkeypatch.setattr(
        edit_decisions,
        "probe_video",
        lambda _path: {"duration": 10.0},
    )
    monkeypatch.setattr(
        edit_decisions.subprocess,
        "run",
        lambda *_args, **_kwargs: None,
    )

    decisions = {
        "cuts": [
            {
                "source": path,
                "in_seconds": 0,
                "out_seconds": 1,
                "has_audio": True,
            }
            for path in segments
        ],
        "transitions": [{"index": 0, "type": "cut", "duration": 0.5}],
        "metadata": {"compose_target": {}, "target_fps": 30},
    }
    result = edit_decisions.execute_edit_decisions(
        decisions, str(tmp_path / "assembled.mp4")
    )

    assert result["success"] is True
    assert rendered == [True]
    assert copied == []


def test_edit_decisions_apply_quality_trim_and_story_order(monkeypatch, tmp_path):
    shots = tmp_path / "shots"
    for name in ("S01", "S02"):
        directory = shots / name
        directory.mkdir(parents=True)
        (directory / "output.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        edit_decisions,
        "probe_video",
        lambda _path: {"duration": 5.0, "width": 320, "height": 180, "fps": 24, "has_audio": False, "has_video": True},
    )
    monkeypatch.setattr(edit_decisions, "detect_black_frames", lambda _path: {"trim_start": 0.1, "trim_end": 0.1})
    quality = {
        "shots": {
            "S01": {"action": "keep", "trim_start_s": 0.0, "trim_end_s": 5.0, "reasons": []},
            "S02": {"action": "trim", "trim_start_s": 0.6, "trim_end_s": 4.4, "reasons": ["boundary black"]},
        }
    }

    decisions = edit_decisions.build_edit_decisions(
        shots,
        quality_report=quality,
        shot_order=["S02", "S01"],
    )

    assert [cut["shot_id"] for cut in decisions["cuts"]] == ["S02", "S01"]
    assert decisions["cuts"][0]["in_seconds"] == 0.6
    assert decisions["cuts"][0]["out_seconds"] == 4.4
    assert decisions["metadata"]["quality_reviewed"] is True


def test_visual_reshoot_is_bounded_to_two_rounds(monkeypatch, tmp_path):
    shots = tmp_path / "shots"
    shot = shots / "S01"
    shot.mkdir(parents=True)
    (shot / "output.mp4").write_bytes(b"video")
    (shot / "SHOT_META.json").write_text(
        json.dumps({"shot_id": "S01"}),
        encoding="utf-8",
    )
    (tmp_path / "STORYBOARD.json").write_text(json.dumps({"shots": [{"shot_id": "S01"}]}))
    monkeypatch.setattr(
        story_order_reviewer,
        "review_story_order",
        lambda *_args: {"matches_current_order": True, "narrative_consistent": True, "issues": [], "suggested_order": ["S01"]},
    )
    monkeypatch.setattr(
        frame_analysis,
        "analyze_shot_frames",
        lambda *_args, **_kwargs: {
            "shots": {"S01": {"action": "reshoot", "reasons": ["identity drift"]}},
            "summary": {"keep": [], "trim": [], "reshoot": ["S01"]},
            "has_issues": True,
        },
    )
    calls: list[bool] = []

    def regenerate(*_args, **kwargs):
        calls.append(kwargs.get("chain_mode"))
        (shot / "output.mp4").write_bytes(b"regenerated")
        return {"status": "done"}

    monkeypatch.setattr(pipeline_core, "run_phase6", regenerate)

    result = pipeline_core.run_phase8(
        tmp_path,
        dry_run=False,
        enable_reshoot=True,
        chain_mode=True,
    )

    assert result["status"] == "error"
    assert "after 2 reshoot rounds" in result["error"]
    assert calls == [True, True]


def test_required_trim_never_falls_back_to_raw_concat(monkeypatch, tmp_path):
    shot = tmp_path / "shots" / "S01"
    shot.mkdir(parents=True)
    (shot / "output.mp4").write_bytes(b"video")
    (shot / "SHOT_META.json").write_text(
        json.dumps({"shot_id": "S01"}),
        encoding="utf-8",
    )
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"shot_id": "S01"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        story_order_reviewer,
        "review_story_order",
        lambda *_args: {"matches_current_order": True, "narrative_consistent": True, "issues": [], "suggested_order": ["S01"]},
    )
    monkeypatch.setattr(
        frame_analysis,
        "analyze_shot_frames",
        lambda *_args, **_kwargs: {
            "shots": {"S01": {"action": "trim", "trim_start_s": 0.4, "trim_end_s": 4.0, "reasons": ["black head"]}},
            "summary": {"keep": [], "trim": ["S01"], "reshoot": []},
            "has_issues": True,
        },
    )
    monkeypatch.setattr(shot_embedder, "embed_all_shots", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(edit_decisions, "build_edit_decisions", lambda *_args, **_kwargs: {"cuts": [{}]})
    monkeypatch.setattr(edit_decisions, "execute_edit_decisions", lambda *_args, **_kwargs: {"success": False, "error": "encoder failed"})

    result = pipeline_core.run_phase8(tmp_path, dry_run=False, enable_reshoot=True)

    assert result["status"] == "error"
    assert "reviewed edit execution failed" in result["error"]
    assert not (tmp_path / "raw_assembly.mp4").exists()
def test_phase8_rejects_unknown_exhausted_reshoot_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY", "unknown")

    with pytest.raises(ValueError, match="EXHAUSTED_RESHOOT_POLICY"):
        pipeline_core.run_phase8(tmp_path, dry_run=True)
