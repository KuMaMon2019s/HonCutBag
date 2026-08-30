from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.provider_attempt_policy import (
    effective_provider_attempts,
    effective_provider_retries,
    provider_attempt_scope,
)
from prompt import event_extractor


def test_provider_attempt_scope_is_visible_to_phase_worker_threads():
    with provider_attempt_scope(max_retries=0):
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed = list(executor.map(effective_provider_retries, [1, 3]))

    assert observed == [0, 0]
    assert effective_provider_retries(3) == 3
    assert effective_provider_attempts(3) == 3


def test_provider_attempt_scope_rejects_nested_runtime_policy():
    with provider_attempt_scope(max_retries=0):
        with pytest.raises(RuntimeError, match="already active"):
            with provider_attempt_scope(max_retries=0):
                pass


def test_provider_attempt_scope_disables_event_schema_resubmission(monkeypatch):
    calls = 0

    def invalid_response(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "not-json"

    monkeypatch.setattr(event_extractor, "_call_llm", invalid_response)
    with provider_attempt_scope(max_retries=0):
        with pytest.raises(event_extractor.EventExtractionError):
            event_extractor._extract_events_from_segment({
                "id": 1,
                "content": "A fictional character enters a room.",
                "format_hint": "general_prose",
            })

    assert calls == 1


def test_provider_transports_do_not_reverse_import_runtime_policy():
    repository = Path(__file__).resolve().parents[2]
    for relative in (
        "pipeline/src/utils/ark_llm.py",
        "pipeline/src/clients/seedream_client.py",
    ):
        source = (repository / relative).read_text(encoding="utf-8")
        assert "runtime.provider_attempt_policy" not in source
        assert "utils.provider_request_guard" in source
