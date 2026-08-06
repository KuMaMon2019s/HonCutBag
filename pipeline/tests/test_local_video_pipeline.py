import json
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clients import local_video_client
import pipeline_runner


class _Response:
    text = ""

    def __init__(self, payload, status_code=200, content=b"video"):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "video/mp4", "content-length": str(len(content))}
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=8192):
        yield self._content

    def close(self):
        return None


class _Session:
    def __init__(self, statuses=None):
        self.statuses = iter(statuses or [])
        self.posts = []

    def get(self, *args, **kwargs):
        return _Response(next(self.statuses))

    def post(self, *args, **kwargs):
        self.posts.append(kwargs["json"])
        return _Response({"task_id": "task-1"})


def test_poll_zero_progress_uses_queue_timeout_not_stall_timeout(monkeypatch):
    statuses = [{"status": "running", "progress": 0}] * 31
    statuses.append({"status": "completed", "progress": 100})
    monkeypatch.setattr(local_video_client, "_request_session", lambda: _Session(statuses))
    monkeypatch.setenv("LOCAL_VIDEO_QUEUE_TIMEOUT", "7200")

    assert local_video_client.poll("queued-task", max_attempts=30, interval=0)["status"] == "completed"

    monkeypatch.setattr(
        local_video_client,
        "_request_session",
        lambda: _Session([{"status": "queued", "progress": 0}]),
    )
    monkeypatch.setenv("LOCAL_VIDEO_QUEUE_TIMEOUT", "0")
    with pytest.raises(TimeoutError, match="queue wait exceeded"):
        local_video_client.poll("queued-task", max_attempts=30, interval=0)


def test_poll_started_task_still_stalls(monkeypatch):
    statuses = [{"status": "running", "progress": 10}] * 3
    monkeypatch.setattr(local_video_client, "_request_session", lambda: _Session(statuses))

    with pytest.raises(TimeoutError, match="stalled: progress stuck at 10%"):
        local_video_client.poll("started-task", max_attempts=2, interval=0)


def test_poll_stall_polls_env_override(monkeypatch):
    statuses = [{"status": "running", "progress": 10}] * 4
    monkeypatch.setattr(local_video_client, "_request_session", lambda: _Session(statuses))
    monkeypatch.setenv("LOCAL_VIDEO_STALL_POLLS", "3")

    with pytest.raises(TimeoutError, match="for 3 polls"):
        local_video_client.poll("env-stall-task", interval=0)


def test_poll_high_progress_stall_download_probe_completes(monkeypatch):
    class ProbeSession(_Session):
        def get(self, url, **kwargs):
            if "/download/" in url:
                return _Response({}, content=b"ready")
            return super().get(url, **kwargs)

    statuses = [{"status": "running", "progress": 80}] * 3
    monkeypatch.setattr(local_video_client, "_request_session", lambda: ProbeSession(statuses))

    assert local_video_client.poll("slow-task", max_attempts=2, interval=0) == {
        "status": "completed",
        "progress": 100,
    }


def test_poll_high_progress_genuine_stall_raises(monkeypatch):
    class FailedProbeSession(_Session):
        def get(self, url, **kwargs):
            if "/download/" in url:
                return _Response({}, status_code=404, content=b"")
            return super().get(url, **kwargs)

    statuses = [{"status": "running", "progress": 80}] * 3
    monkeypatch.setattr(local_video_client, "_request_session", lambda: FailedProbeSession(statuses))

    with pytest.raises(TimeoutError, match="stalled: progress stuck at 80%"):
        local_video_client.poll("genuine-stall", max_attempts=2, interval=0)


def test_generate_video_uses_num_frames_override_and_real_duration(monkeypatch, tmp_path):
    submitted = {}
    downloaded = {}
    monkeypatch.setenv("LOCAL_VIDEO_NUM_FRAMES", "73")
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: submitted.update(kwargs) or "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: {"status": "completed"})
    monkeypatch.setattr(local_video_client, "download", lambda *args, **kwargs: downloaded.update(kwargs))

    local_video_client.generate_video("prompt", str(tmp_path / "output.mp4"), duration=6, fps=24)

    assert submitted["num_frames"] == 73
    assert downloaded["expected_duration"] == pytest.approx(73 / 24)


@pytest.mark.parametrize(
    ("duration", "expected_frames"),
    [(2, 49), (3.5, 97), (5, 145), (6, 145)],
)
def test_snap_duration_to_verified_wan22_frames(duration, expected_frames):
    frames, actual_duration, reason = local_video_client.snap_duration_to_frames(
        duration, 24, [49, 97, 145]
    )

    assert frames == expected_frames
    assert actual_duration == pytest.approx(expected_frames / 24)
    assert "ties prefer larger" in reason


def test_snap_duration_tie_breaks_to_larger_frame_count():
    frames, _, _ = local_video_client.snap_duration_to_frames(3, 24, [49, 97])

    assert frames == 97


def test_generate_video_uses_duration_snap_and_snapped_validation_duration(monkeypatch, tmp_path):
    submitted = {}
    downloaded = {}
    monkeypatch.delenv("LOCAL_VIDEO_NUM_FRAMES", raising=False)
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: submitted.update(kwargs) or "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: {"status": "completed"})
    monkeypatch.setattr(local_video_client, "download", lambda *args, **kwargs: downloaded.update(kwargs))

    local_video_client.generate_video("prompt", str(tmp_path / "S02" / "output.mp4"), duration=5)

    assert submitted["num_frames"] == 145
    assert downloaded["expected_duration"] == pytest.approx(145 / 24)


def test_generate_video_respects_custom_valid_frames(monkeypatch, tmp_path):
    submitted = {}
    monkeypatch.delenv("LOCAL_VIDEO_NUM_FRAMES", raising=False)
    monkeypatch.setenv("LOCAL_VIDEO_VALID_FRAMES", "25,73,121")
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: submitted.update(kwargs) or "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: {"status": "completed"})
    monkeypatch.setattr(local_video_client, "download", lambda *args, **kwargs: None)

    local_video_client.generate_video("prompt", str(tmp_path / "output.mp4"), duration=3)

    assert submitted["num_frames"] == 73


def test_generate_video_missing_duration_uses_middle_valid_frame_count(monkeypatch, tmp_path):
    submitted = {}
    monkeypatch.delenv("LOCAL_VIDEO_NUM_FRAMES", raising=False)
    monkeypatch.delenv("LOCAL_VIDEO_VALID_FRAMES", raising=False)
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: submitted.update(kwargs) or "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: {"status": "completed"})
    monkeypatch.setattr(local_video_client, "download", lambda *args, **kwargs: None)

    local_video_client.generate_video("prompt", str(tmp_path / "output.mp4"), duration=None)

    assert submitted["num_frames"] == 97


def test_generate_video_records_probe_provenance_and_duration_drift(monkeypatch, tmp_path, capsys):
    shot_dir = tmp_path / "S02"
    shot_dir.mkdir()
    meta_path = shot_dir / "SHOT_META.json"
    meta_path.write_text(json.dumps({"prompt": "test", "duration": 5}))
    monkeypatch.delenv("LOCAL_VIDEO_NUM_FRAMES", raising=False)
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: "task-1")
    monkeypatch.setattr(local_video_client, "poll", lambda task_id: {"status": "completed"})

    def fake_download(*args, **kwargs):
        kwargs["verification_out"].update({"duration": 6.04, "num_frames": 145})

    monkeypatch.setattr(local_video_client, "download", fake_download)

    local_video_client.generate_video("prompt", str(shot_dir / "output.mp4"), duration=5)

    meta = json.loads(meta_path.read_text())
    assert meta["requested_duration"] == 5
    assert meta["requested_num_frames"] == 145
    assert meta["actual_duration"] == pytest.approx(6.04)
    assert meta["actual_num_frames"] == 145
    assert "duration drift" in capsys.readouterr().out


def test_submit_includes_batch_id_in_payload(monkeypatch):
    session = _Session()
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)

    assert local_video_client.submit("prompt", batch_id="westlake_evening_v8") == "task-1"
    assert session.posts[0]["batch_id"] == "westlake_evening_v8"

    content_session = _Session()
    monkeypatch.setattr(local_video_client, "_request_session", lambda: content_session)
    assert local_video_client.submit(
        "prompt",
        content=[{"type": "text", "text": "prompt"}],
        batch_id="westlake_evening_v8",
    ) == "task-1"
    assert content_session.posts[0]["batch_id"] == "westlake_evening_v8"
    assert content_session.posts[0]["num_frames"] == 73


def test_phase5_skips_existing_valid_output(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots" / "S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(json.dumps({"prompt": "test", "duration": 6}))
    (shot_dir / "output.mp4").write_bytes(b"x" * (10 * 1024 + 1))

    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    monkeypatch.setattr(
        local_video_client,
        "generate_video",
        lambda **kwargs: pytest.fail("existing output must not be regenerated"),
    )

    result = pipeline_runner._run_phase5_fallback(tmp_path)

    assert result["status"] == "done"
    assert result["outputs"] == ["shots/S01/output.mp4"]
