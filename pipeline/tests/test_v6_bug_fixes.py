import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clients import local_video_client
from phases.character_discoverer import _filter_descriptive_phrases


class _DownloadResponse:
    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"downloaded-video"


def test_download_keeps_file_when_ffprobe_metadata_is_unreadable(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "output.mp4"
    session = SimpleNamespace(get=lambda *args, **kwargs: _DownloadResponse())
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    monkeypatch.setattr(
        local_video_client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid data"
        ),
    )

    local_video_client.download(
        "task-123",
        str(output_path),
        expected_duration=10.0,
        expected_width=1280,
        expected_height=720,
    )

    assert output_path.read_bytes() == b"downloaded-video"
    assert "WARNING: metadata unavailable" in capsys.readouterr().err


def test_compound_robot_name_passes_character_filter():
    name = "白色金属AI巡检机器人"
    stats = {name: {"count": 3, "events": ["E01", "E02", "E03"]}}

    filtered = _filter_descriptive_phrases(stats)

    assert name in filtered
