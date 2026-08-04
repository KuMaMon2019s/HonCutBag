import json
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import local_video_client
import pipeline_runner


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


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
