"""Unit and integration tests for the Phase 7 audio-material layer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phases.audio_mixer import AudioMixer, apply_phase7_audio
from tools.audio.ducking import AudioDucking
from tools.audio.music_library import MusicLibrary
from tools.audio.tts_narration import TTSNarration


def make_tone(path: Path, duration: float = 1.0, frequency: int = 440) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
        str(path),
    ], capture_output=True, check=True)


def make_video(path: Path, duration: float = 1.0) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"color=c=black:s=160x90:d={duration}", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True, check=True)


def probe_duration(path: Path) -> float:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], capture_output=True, text=True, check=True)
    return float(completed.stdout)


class TestMusicLibrary:
    def test_scan_search_metadata_and_get_track(self, tmp_path: Path) -> None:
        calm = tmp_path / "calm_ambient.wav"
        epic = tmp_path / "epic_trailer.mp3"
        make_tone(calm, 1.0)
        make_tone(epic, 2.0)
        (tmp_path / "notes.txt").write_text("not audio")

        library = MusicLibrary(str(tmp_path))
        tracks = library.scan()

        assert len(tracks) == 2
        assert {track.format for track in tracks} == {"wav", "mp3"}
        assert all(track.metadata["bitrate"] > 0 for track in tracks)
        assert library.search("epic", (1.5, 2.5))[0].path == str(epic)
        assert library.get_track(tracks[0].id) == tracks[0]

    def test_search_validation_and_missing_track(self, tmp_path: Path) -> None:
        library = MusicLibrary(str(tmp_path))
        with pytest.raises(ValueError, match="Unsupported mood"):
            library.search("angry")
        with pytest.raises(ValueError, match="minimum"):
            library.search(duration_range=(2, 1))
        with pytest.raises(KeyError, match="not found"):
            library.get_track("missing")

    def test_base_tool_execute_remains_compatible(self, tmp_path: Path, monkeypatch) -> None:
        audio = tmp_path / "happy.wav"
        audio.write_bytes(b"audio")
        library = MusicLibrary(str(tmp_path))
        monkeypatch.setattr(library, "_probe_duration", lambda path: 3.5)

        response = library.execute({"library_dir": str(tmp_path)})

        assert response.success is True
        assert response.data["track_count"] == 1
        assert response.data["total_duration_seconds"] == 3.5

    def test_external_provider_is_normalized(self, tmp_path: Path, monkeypatch) -> None:
        downloaded = tmp_path / "calm-result.mp3"
        provider = Mock()
        provider.execute.return_value = SimpleNamespace(
            success=True, error=None,
            data={"output": str(downloaded), "duration_seconds": 42, "format": "mp3"},
        )
        fake_module = SimpleNamespace(PixabayMusic=lambda: provider)
        monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

        tracks = MusicLibrary(str(tmp_path)).search_external("calm", "pixabay")

        assert len(tracks) == 1
        assert tracks[0].source == "pixabay"
        assert tracks[0].duration == 42

    def test_external_provider_errors_are_explained(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="source"):
            MusicLibrary(str(tmp_path)).search_external("calm", "unknown")


class TestAudioDucking:
    def test_parameters_and_ffmpeg_filter(self) -> None:
        ducking = AudioDucking(-20, 4, 10, 100)
        graph = ducking.build_filter()
        assert "sidechaincompress=threshold=0.10000000" in graph
        assert "ratio=4:attack=10:release=100" in graph
        with pytest.raises(ValueError):
            AudioDucking(ratio=0.5)

    def test_apply_creates_ducked_mix(self, tmp_path: Path) -> None:
        bgm = tmp_path / "bgm.wav"
        voice = tmp_path / "voice.wav"
        output = tmp_path / "ducked.m4a"
        make_tone(bgm, 2.0, 220)
        make_tone(voice, 1.0, 880)

        assert AudioDucking().apply(str(bgm), str(voice), str(output)) == str(output)
        assert output.is_file()
        assert 0.8 <= probe_duration(output) <= 1.2

    def test_missing_input_is_clear(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BGM"):
            AudioDucking().apply("missing.wav", "also-missing.wav", str(tmp_path / "x.m4a"))


class TestTTSNarration:
    def test_generate_posts_openai_compatible_payload(self, tmp_path: Path) -> None:
        response = Mock()
        response.headers = {"Content-Type": "audio/mpeg"}
        response.content = b"fake-mp3"
        response.raise_for_status.return_value = None
        session = Mock()
        session.post.return_value = response
        output = tmp_path / "voice.mp3"
        narrator = TTSNarration(
            "secret", base_url="https://ark.example/v3", model="doubao-tts",
            voice="voice-a", session=session,
        )

        assert narrator.generate("你好", str(output), 1.25, 0.9) == str(output)
        assert output.read_bytes() == b"fake-mp3"
        request = session.post.call_args
        assert request.args[0] == "https://ark.example/v3/audio/speech"
        assert request.kwargs["json"]["speed"] == 1.25
        assert request.kwargs["json"]["pitch"] == 0.9
        assert request.kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_rejects_bad_input_without_network(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("ARK_AGENT_API_KEY", raising=False)
        with pytest.raises(ValueError, match="required"):
            TTSNarration(api_key=None).generate("hello", str(tmp_path / "x.mp3"))
        with pytest.raises(ValueError, match="empty"):
            TTSNarration("key").generate(" ", str(tmp_path / "x.mp3"))


class TestPhase7AudioMixer:
    def test_full_process_with_injected_collaborators(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        bgm = tmp_path / "epic.wav"
        make_video(video, 1.0)
        make_tone(bgm, 1.0)
        library = MusicLibrary(str(tmp_path))
        narrator = Mock()

        def synthesize(text: str, path: str, speed: float, pitch: float) -> str:
            make_tone(Path(path), 1.0, 900)
            return path

        narrator.generate.side_effect = synthesize
        mixer = AudioMixer(music_library=library, narrator=narrator)
        output = tmp_path / "mixed.mp4"
        receipt = mixer.process(
            str(video),
            {"metadata": {"mood": "epic"}, "audio": {"tts": True},
             "shots": [{"narration": "第一幕"}]},
            str(output), duration=1.0,
        )

        assert output.is_file()
        assert receipt["bgm"] == str(bgm)
        assert len(receipt["narration"]) == 1

    def test_phase7_entrypoint_is_opt_in(self, tmp_path: Path) -> None:
        make_video(tmp_path / "raw_assembly.mp4")
        (tmp_path / "STORYBOARD.json").write_text('{"audio": {"enabled": false}}')
        assert apply_phase7_audio(tmp_path) is None
