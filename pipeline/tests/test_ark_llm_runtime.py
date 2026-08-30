from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import ark_llm
from runtime.provider_attempt_policy import provider_attempt_scope


def _client_with_create(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def test_fixed_window_llm_quota_fails_immediately_without_backoff(monkeypatch):
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(
            "Error code: 429 - {'error': {'code': 'AccountQuotaExceeded', "
            "'message': 'You have exceeded the monthly usage quota. It will "
            "reset at 2026-09-02 23:59:59 +0800 CST.'}}"
        )

    monkeypatch.setattr(
        ark_llm.time,
        "sleep",
        lambda _seconds: pytest.fail("fixed-window quota must not back off"),
    )

    with pytest.raises(ark_llm.LLMQuotaExceededError) as captured:
        ark_llm.call_llm_stream(
            [{"role": "user", "content": "bounded test"}],
            _client=_client_with_create(create),
            launch_stagger=0,
            rate_limit_retries=3,
        )

    assert calls == 1
    assert "2026-09-02 23:59:59 +0800 CST" in str(captured.value)


def test_transient_429_still_uses_bounded_runtime_backoff(monkeypatch):
    calls = 0
    sleeps = []

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Error code: 429 - too many requests")
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
            )
        ]

    monkeypatch.setattr(ark_llm.time, "sleep", sleeps.append)

    result = ark_llm.call_llm_stream(
        [{"role": "user", "content": "bounded test"}],
        _client=_client_with_create(create),
        launch_stagger=0,
        rate_limit_retries=1,
        rate_limit_base_wait=2,
        rate_limit_jitter=0,
    )

    assert result == "ok"
    assert calls == 2
    assert sleeps == [2]


def test_live_acceptance_scope_disables_transport_retry_and_records_attempt(
    monkeypatch,
):
    calls = 0
    started = []
    completed = []
    failed = []

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Error code: 429 - too many requests")

    monkeypatch.setattr(
        ark_llm.time,
        "sleep",
        lambda _seconds: pytest.fail("zero-retry scope must not back off"),
    )

    with provider_attempt_scope(
        max_retries=0,
        before_provider_request=lambda payload: (
            started.append(payload) or "request-1"
        ),
        after_provider_request=lambda token, outcome: completed.append(
            (token, outcome)
        ),
        failed_provider_request=lambda token, outcome: failed.append(
            (token, outcome)
        ),
    ):
        with pytest.raises(ark_llm.LLMRateLimitedError):
            ark_llm.call_llm_stream(
                [{"role": "user", "content": "bounded test"}],
                _client=_client_with_create(create),
                launch_stagger=0,
                rate_limit_retries=3,
            )

    assert calls == 1
    assert len(started) == 1
    assert started[0]["provider_family"] == "ark_text"
    assert started[0]["messages_sha256"]
    assert completed == []
    assert failed == [(
        "request-1",
        {
            "submission_outcome": "unknown",
            "error_type": "RuntimeError",
        },
    )]


def test_stream_interruption_never_marks_request_completed_or_retries(
    monkeypatch,
):
    calls = 0
    started = []
    completed = []
    failed = []

    class BrokenStream:
        def __iter__(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))]
            )
            raise ark_llm.httpx.RemoteProtocolError("stream interrupted")

        @staticmethod
        def close():
            return None

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        return BrokenStream()

    monkeypatch.setattr(
        ark_llm.time,
        "sleep",
        lambda _seconds: pytest.fail("interrupted acceptance stream must not retry"),
    )
    with provider_attempt_scope(
        max_retries=0,
        before_provider_request=lambda payload: (
            started.append(payload) or "request-1"
        ),
        after_provider_request=lambda token, outcome: completed.append(
            (token, outcome)
        ),
        failed_provider_request=lambda token, outcome: failed.append(
            (token, outcome)
        ),
    ):
        with pytest.raises(ark_llm.LLMStreamError):
            ark_llm.call_llm_stream(
                [{"role": "user", "content": "bounded test"}],
                _client=_client_with_create(create),
                launch_stagger=0,
                rate_limit_retries=3,
            )

    assert calls == 1
    assert len(started) == 1
    assert completed == []
    assert failed == [(
        "request-1",
        {
            "submission_outcome": "unknown",
            "error_type": "RemoteProtocolError",
        },
    )]
