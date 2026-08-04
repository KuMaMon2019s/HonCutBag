import base64
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import seedream_client


class _FakeResponse:
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
