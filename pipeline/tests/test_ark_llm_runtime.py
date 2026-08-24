from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import ark_llm


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
