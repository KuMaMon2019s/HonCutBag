"""Tests for Batch A video dual-track routing."""

from clients import local_video_client, seedance_client
from clients.video_client import VideoClient
from utils.config import get_bridge_api_url, get_video_route


def test_video_route_defaults_to_bridge(monkeypatch):
    monkeypatch.delenv("VIDEO_GENERATION_MODE", raising=False)
    monkeypatch.delenv("VIDEO_PROVIDER_WAN", raising=False)
    assert get_video_route("wan") == "bridge"


def test_provider_route_overrides_global(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "direct")
    monkeypatch.setenv("VIDEO_PROVIDER_WAN", "local")
    assert get_video_route("wan") == "local"


def test_bridge_url_from_environment(monkeypatch):
    monkeypatch.setenv("BRIDGE_API_URL", "http://bridge.test:9100/")
    assert get_bridge_api_url() == "http://bridge.test:9100"


def test_video_client_bridge_uses_local_client(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "bridge")
    submitted = {}
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: submitted.update(kwargs) or "task-7")

    result = VideoClient("seedance", direct_generator=lambda **kwargs: "external").generate(
        "pan across lake", api_key="secret", duration=7
    )

    assert result.route == "bridge"
    assert result.task_id == "task-7"
    assert submitted == {"prompt": "pan across lake", "model": "seedance"}


def test_seedance_direct_uses_ark_fallback(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_MODE", "direct")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"id": "ark-task"}

    called = {}
    monkeypatch.setattr(
        seedance_client.requests,
        "post",
        lambda url, **kwargs: called.update(url=url, **kwargs) or Response(),
    )
    assert seedance_client.submit("a quiet street", "ark-key") == "ark-task"
    assert called["url"] == seedance_client.SUBMIT_ENDPOINT


def test_video_client_direct_never_calls_bridge(monkeypatch):
    monkeypatch.setenv("VIDEO_PROVIDER_KLING", "direct")
    monkeypatch.setattr(local_video_client, "submit", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))
    result = VideoClient("kling", direct_generator=lambda **kwargs: "kling-task").generate("prompt")
    assert result.route == "direct"
    assert result.value == "kling-task"
