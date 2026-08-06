import base64
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clients import seedream_client


class _FakeResponse:
    status_code = 200
    headers = {}
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"b64_json": base64.b64encode(b"image").decode()}]}


def test_seedream_requests_are_serialized_and_spaced(monkeypatch, tmp_path):
    min_interval = 0.05
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", str(min_interval))
    monkeypatch.setattr(
        seedream_client,
        "_SEEDREAM_RATE_LIMITER",
        seedream_client._SeedreamRateLimiter(),
    )

    request_starts = []
    active_requests = 0
    max_active_requests = 0
    state_lock = threading.Lock()

    def fake_post(*args, **kwargs):
        nonlocal active_requests, max_active_requests
        with state_lock:
            request_starts.append(time.monotonic())
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
        time.sleep(0.01)
        with state_lock:
            active_requests -= 1
        return _FakeResponse()

    monkeypatch.setattr(seedream_client.requests, "post", fake_post)
    client = seedream_client.SeedreamClient(api_key="test-key")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                client._call_and_save,
                {"prompt": f"request-{index}"},
                str(tmp_path / f"request-{index}.png"),
            )
            for index in range(3)
        ]
        for future in futures:
            future.result()

    intervals = [
        later - earlier
        for earlier, later in zip(request_starts, request_starts[1:])
    ]
    assert max_active_requests == 1
    assert len(request_starts) == 3
    assert all(interval >= min_interval * 0.95 for interval in intervals)


class _QuotaExceededResponse:
    status_code = 429
    headers = {"x-request-id": "test-request"}
    text = '{"error":{"code":"AccountQuotaExceeded"}}'

    def raise_for_status(self):
        raise AssertionError("AccountQuotaExceeded should be handled before raise_for_status")


def test_account_quota_exceeded_retries_three_times_then_succeeds(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "0")
    monkeypatch.setattr(
        seedream_client,
        "_SEEDREAM_RATE_LIMITER",
        seedream_client._SeedreamRateLimiter(),
    )
    responses = [_QuotaExceededResponse()] * 3 + [_FakeResponse()]
    sleep_calls = []

    monkeypatch.setattr(
        seedream_client.requests,
        "post",
        lambda *args, **kwargs: responses.pop(0),
    )
    monkeypatch.setattr(seedream_client.time, "sleep", sleep_calls.append)

    client = seedream_client.SeedreamClient(api_key="test-key")
    client._call_and_save({"prompt": "test"}, str(tmp_path / "image.png"))

    assert responses == []
    assert sleep_calls == [60, 60, 60]


def test_account_quota_exceeded_raises_after_three_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "0")
    monkeypatch.setattr(
        seedream_client,
        "_SEEDREAM_RATE_LIMITER",
        seedream_client._SeedreamRateLimiter(),
    )
    post_calls = 0

    def fake_post(*args, **kwargs):
        nonlocal post_calls
        post_calls += 1
        return _QuotaExceededResponse()

    monkeypatch.setattr(seedream_client.requests, "post", fake_post)
    monkeypatch.setattr(seedream_client.time, "sleep", lambda seconds: None)

    client = seedream_client.SeedreamClient(api_key="test-key")
    try:
        client._call_and_save({"prompt": "test"}, str(tmp_path / "image.png"))
    except seedream_client.AgentPlanQuotaExceededError as exc:
        assert "HTTP 429 AccountQuotaExceeded" in str(exc)
    else:
        raise AssertionError("expected AgentPlanQuotaExceededError")

    assert post_calls == 4
