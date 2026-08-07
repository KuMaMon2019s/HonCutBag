"""Bridge download-probe timeout and compatibility-attempt coverage."""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clients import local_video_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Mock session: returns canned statuses for /status, and download_ok for /download."""

    def __init__(self, statuses, download_ok=False):
        self.statuses = iter(statuses)
        self.download_ok = download_ok
        self.download_calls = 0

    def get(self, url, **kwargs):
        if "/download/" in url:
            self.download_calls += 1
            if self.download_ok:
                return _Response({}, content=b"ready")
            return _Response({}, status_code=404, content=b"")
        return _Response(next(self.statuses))

    def post(self, *args, **kwargs):
        return _Response({"task_id": "task-1"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_probe_timeout_is_60_seconds(monkeypatch):
    monkeypatch.delenv("VIDEO_DOWNLOAD_PROBE_TIMEOUT", raising=False)
    monkeypatch.delenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", raising=False)
    clock = {"now": 0.0}
    monkeypatch.setattr(local_video_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(local_video_client.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    statuses = [{"status": "running", "progress": 100}] * 20
    sess = _Session(statuses, download_ok=False)
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    with pytest.raises(TimeoutError, match=r"timeout=60s"):
        local_video_client.poll("probe-task", interval=10)
    assert clock["now"] == 60
    assert sess.download_calls == 12


def test_env_override_respected(monkeypatch):
    """LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS=5 → give up after 5 failures."""
    monkeypatch.setenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", "5")
    statuses = [{"status": "running", "progress": 100}] * 10  # more than enough
    sess = _Session(statuses, download_ok=False)
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    with pytest.raises(TimeoutError, match=r"download probe failed 5/5"):
        local_video_client.poll("probe-task", interval=0)
    assert sess.download_calls == 5


def test_probe_success_after_failures_completes_immediately(monkeypatch):
    """A ready download is sufficient even while Bridge still reports running/100."""
    monkeypatch.setenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", "40")

    call_count = {"n": 0}

    class LateSuccessSession:
        download_calls = 0

        def get(self, url, **kwargs):
            if "/download/" in url:
                self.download_calls += 1
                call_count["n"] += 1
                # First 28 calls fail, then the first OK completes immediately.
                if call_count["n"] > 28:
                    return _Response({}, content=b"ready")
                return _Response({}, status_code=404, content=b"")
            return _Response({"status": "running", "progress": 100})

        def post(self, *args, **kwargs):
            return _Response({"task_id": "task-1"})

    sess = LateSuccessSession()
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    result = local_video_client.poll("probe-task", interval=0)
    assert result == {"status": "completed", "progress": 100}
    assert call_count["n"] == 29


def test_timeout_error_includes_wait_duration(monkeypatch):
    """TimeoutError message must mention how long it waited."""
    monkeypatch.setenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", "3")
    statuses = [{"status": "running", "progress": 100}] * 5
    sess = _Session(statuses, download_ok=False)
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    with pytest.raises(TimeoutError) as exc_info:
        local_video_client.poll("probe-task", interval=0)
    msg = str(exc_info.value)
    # Must contain the fraction and a duration
    assert "3/3" in msg
    assert "over" in msg and "s" in msg


def test_invalid_env_value_raises(monkeypatch):
    """LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS=0 must raise ValueError."""
    monkeypatch.setenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", "0")
    statuses = [{"status": "running", "progress": 100}]
    sess = _Session(statuses, download_ok=False)
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    with pytest.raises(ValueError, match="must be a positive integer"):
        local_video_client.poll("probe-task", interval=0)


def test_probe_timeout_env_and_diagnostic_status(monkeypatch, capsys):
    monkeypatch.setenv("VIDEO_DOWNLOAD_PROBE_TIMEOUT", "10")
    monkeypatch.delenv("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", raising=False)
    clock = {"now": 0.0}
    monkeypatch.setattr(local_video_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(local_video_client.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    sess = _Session([{"status": "running", "progress": 100, "bridge_detail": "archiving"}] * 4)
    monkeypatch.setattr(local_video_client, "_request_session", lambda: sess)

    with pytest.raises(TimeoutError, match=r"timeout=10s"):
        local_video_client.poll("probe-task", interval=10)

    assert clock["now"] == 10
    output = capsys.readouterr().out
    assert "Bridge status:" in output
    assert "bridge_detail" in output


def test_invalid_probe_timeout_raises(monkeypatch):
    monkeypatch.setenv("VIDEO_DOWNLOAD_PROBE_TIMEOUT", "0")
    with pytest.raises(ValueError, match="VIDEO_DOWNLOAD_PROBE_TIMEOUT must be a positive integer"):
        local_video_client.poll("probe-task", interval=0)
