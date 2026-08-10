import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.pipeline_core import (
    _phase8_real_audio_mix_request,
    _phase8_real_audio_tracks,
)


def _project(tmp_path, *, enabled=True, tts=True, dialogue="Generic spoken line"):
    raw_video = tmp_path / "raw_assembly.mp4"
    raw_video.touch()
    narration = tmp_path / "audio_layer" / "narration_001.mp3"
    narration.parent.mkdir()
    narration.touch()
    storyboard = {
        "audio": {"enabled": enabled, "tts": tts},
        "shots": [{"shot_id": "S01", "dialogue": dialogue}],
    }
    return raw_video, narration, storyboard


def test_real_audio_with_tts_uses_ducked_normalized_mix(tmp_path):
    raw_video, narration, storyboard = _project(tmp_path)
    transcript = {"shots": [{"source": "asr", "text": "different words", "start_ms": 1750}]}

    tracks, skipped = _phase8_real_audio_tracks(tmp_path, storyboard, transcript, raw_video)
    request = _phase8_real_audio_mix_request(tracks, tmp_path / "audio_processed.mp4")

    assert tracks == [
        {"path": str(raw_video), "role": "music"},
        {"path": str(narration), "role": "speech", "start_seconds": 1.75},
    ]
    assert skipped == 0
    assert request["operation"] == "full_mix"
    assert request["ducking"]["enabled"] is True
    assert request["normalize"] is True
    assert request["loudnorm_target"] == -14
    assert request["output_path"] != request["tracks"][0]["path"]


def test_real_audio_without_tts_is_base_loudnorm_only(tmp_path):
    raw_video, _, storyboard = _project(tmp_path, dialogue=None)

    tracks, skipped = _phase8_real_audio_tracks(tmp_path, storyboard, None, raw_video)
    request = _phase8_real_audio_mix_request(tracks, tmp_path / "audio_processed.mp4")

    assert tracks == [{"path": str(raw_video), "role": "music"}]
    assert skipped == 0
    assert request["operation"] == "mix"
    assert request["ducking"] is None
    assert request["loudnorm_target"] == -14


def test_real_audio_duplicate_asr_line_skips_tts(tmp_path, capsys):
    raw_video, _, storyboard = _project(tmp_path, dialogue="Hello,   world!")
    transcript = {"shots": [{
        "source": "asr", "text": "The source says: hello world.", "start_ms": 0,
    }]}

    tracks, skipped = _phase8_real_audio_tracks(tmp_path, storyboard, transcript, raw_video)

    assert tracks == [{"path": str(raw_video), "role": "music"}]
    assert skipped == 1
    assert "dialogue already in source audio" in capsys.readouterr().out


def test_script_fallback_transcript_does_not_suppress_tts(tmp_path):
    raw_video, _, storyboard = _project(tmp_path)
    transcript = {"shots": [{
        "source": "dialogue_script", "text": "Generic spoken line", "start_ms": 0,
    }]}

    tracks, skipped = _phase8_real_audio_tracks(tmp_path, storyboard, transcript, raw_video)

    assert len(tracks) == 2
    assert skipped == 0


def test_audio_disabled_prevents_overlay_even_when_track_exists(tmp_path):
    raw_video, _, storyboard = _project(tmp_path, enabled=False)

    tracks, skipped = _phase8_real_audio_tracks(tmp_path, storyboard, None, raw_video)

    assert tracks == [{"path": str(raw_video), "role": "music"}]
    assert skipped == 0


def test_silent_audio_branch_remains_separate_from_real_audio_helpers():
    """The existing no-real-audio branch is not routed through these helpers."""
    source = (SRC / "phases" / "pipeline_core.py").read_text(encoding="utf-8")
    branch_start = source.index("        if has_real_audio:", source.index("# Step 8.1: Audio processing"))
    silent_start = source.index("        else:", branch_start)
    branch_end = source.index("        audio_out = str(audio_out)", silent_start)
    assert "_phase8_real_audio_tracks" in source[branch_start:silent_start]
    assert "generate_ambient_audio" in source[silent_start:branch_end]
