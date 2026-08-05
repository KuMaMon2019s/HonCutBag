import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import rescue_stalled_shots


class Response:
    def __init__(self, payload=None, status_code=200, content=b"video-data"):
        self.payload = payload or {}
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise rescue_stalled_shots.requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=8192):
        yield self.content


class Session:
    def __init__(self):
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        if "/status/" in url:
            return Response({"status": "completed"})
        return Response(content=b"x" * 12000)


def test_parse_log_matches_submissions_and_failures(tmp_path):
    log = tmp_path / "phase5.log"
    log.write_text(
        "  → S01: 提交本地 API 视频生成...\n"
        "  [local_submit] ✓ task_id=abc123, images_used=1\n"
        "    ✗ S01: 本地 API 失败 — task stalled\n"
        "  → S02: 提交本地 API 视频生成... task_id=def456\n",
        encoding="utf-8",
    )

    assert rescue_stalled_shots.parse_log(log) == {"S01": "abc123", "S02": "def456"}


def test_rescue_download_is_idempotent_with_mocked_http(monkeypatch, tmp_path):
    log = tmp_path / "phase5.log"
    log.write_text("→ S01: 提交\n[local_submit] task_id=abc123, images_used=1\n")
    output_dir = tmp_path / "output"
    session = Session()

    def fake_verify(path):
        if path.exists() and path.stat().st_size >= rescue_stalled_shots.MIN_BYTES:
            return True, "width=1280, nb_frames=145"
        return False, "missing or smaller than 10KB"

    monkeypatch.setattr(rescue_stalled_shots, "verify_video", fake_verify)

    first = rescue_stalled_shots.rescue(output_dir, log, session=session, api_url="http://bridge")
    assert first["rescued"] == ["S01"]
    assert (output_dir / "shots" / "S01" / "output.mp4").stat().st_size == 12000
    calls_after_first_run = list(session.urls)

    second = rescue_stalled_shots.rescue(output_dir, log, session=session, api_url="http://bridge")
    assert second["skipped"] == ["S01"]
    assert session.urls == calls_after_first_run
