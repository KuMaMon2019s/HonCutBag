from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase1 import adaptation_engine
from runtime.llm_policy import LLMStreamPolicy
from runtime.provider_attempt_policy import provider_attempt_scope
from utils import ark_llm


def _client_with_create(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def test_ark_client_uses_openai_transport_timeout_type(monkeypatch):
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")

    client = ark_llm.create_ark_client(
        connect_timeout=11,
        read_timeout=321,
    )
    timeout = client._client.timeout

    assert timeout.connect == 11
    assert timeout.read == 321
    assert timeout.write == 30
    assert timeout.pool == 10


def test_httpx2_read_timeout_is_classified_and_never_completed():
    started = []
    completed = []
    failed = []

    class TimedOutStream:
        def __iter__(self):
            raise httpx2.ReadTimeout("provider stream read timed out")
            yield  # pragma: no cover

        @staticmethod
        def close():
            return None

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
        with pytest.raises(ark_llm.LLMReadTimeout):
            ark_llm.call_llm_stream(
                [{"role": "user", "content": "bounded test"}],
                _client=_client_with_create(lambda **_kwargs: TimedOutStream()),
                launch_stagger=0,
                rate_limit_retries=3,
            )

    assert len(started) == 1
    assert completed == []
    assert failed == [(
        "request-1",
        {
            "submission_outcome": "unknown",
            "error_type": "ReadTimeout",
        },
    )]


def test_adaptation_selects_runtime_long_stream_profile(monkeypatch):
    observed = {}
    client = object()

    def fake_client(**kwargs):
        observed["client"] = kwargs
        return client

    def fake_stream(**kwargs):
        observed["stream"] = kwargs
        return "{}"

    monkeypatch.setattr(adaptation_engine, "create_ark_client", fake_client)
    monkeypatch.setattr(adaptation_engine, "call_llm_stream", fake_stream)

    assert adaptation_engine._call_llm("bounded prompt", max_tokens=16000) == "{}"

    policy = LLMStreamPolicy.adaptation_structured_output(max_tokens=16000)
    assert adaptation_engine.ADAPTATION_LLM_POLICY == (
        LLMStreamPolicy.adaptation_structured_output(max_tokens=32000)
    )
    assert observed["client"] == {
        "read_timeout": policy.transport_read_timeout_seconds,
    }
    assert observed["stream"]["max_tokens"] == policy.max_tokens
    assert observed["stream"]["wall_timeout"] == policy.wall_timeout_seconds
    assert observed["stream"]["idle_timeout"] == policy.idle_timeout_seconds
    assert observed["stream"]["read_timeout"] == (
        policy.transport_read_timeout_seconds
    )
    assert policy.idle_timeout_seconds >= 240
    assert policy.transport_read_timeout_seconds > policy.wall_timeout_seconds


def test_adaptation_acceptance_scope_keeps_timeout_to_one_attempt(monkeypatch):
    calls = 0

    def timed_out(_prompt, *, max_tokens):
        nonlocal calls
        calls += 1
        raise ark_llm.LLMReadTimeout(f"timed out at {max_tokens}")

    monkeypatch.setattr(adaptation_engine, "_call_llm", timed_out)
    monkeypatch.setattr(
        adaptation_engine.time,
        "sleep",
        lambda _seconds: pytest.fail("zero-retry scope must not sleep"),
    )

    with provider_attempt_scope(max_retries=0):
        with pytest.raises(RuntimeError, match="连续 0 次网络超时"):
            adaptation_engine._call_llm_with_timeout_retry(
                "bounded prompt",
                max_tokens=16000,
            )

    assert calls == 1
