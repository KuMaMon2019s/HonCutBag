import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality import supervision_agent


def _storyboard():
    return {
        "target_duration": 10,
        "shots": [
            {"shot_order": 1, "duration": 5, "description": "Opening"},
            {"shot_order": 2, "duration": 5, "description": "Response"},
        ],
    }


def test_happy_path_writes_report_atomically(tmp_path, monkeypatch):
    expected = {
        "grade": "A",
        "issues": [],
        "verdict": "pass",
        "summary": "Ready for production.",
    }
    monkeypatch.setattr(
        supervision_agent,
        "_call_llm",
        lambda prompt, config: f"```json\n{json.dumps(expected)}\n```",
    )

    result = supervision_agent.run_supervision(
        _storyboard(), "restrained cinematic style", tmp_path, {}
    )

    assert result == expected
    assert json.loads((tmp_path / "supervision_report.json").read_text()) == expected
    assert not list(tmp_path.glob(".supervision_report.json.*.tmp"))


def test_parse_failure_degrades_to_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision_agent, "_call_llm", lambda prompt, config: "not json")

    result = supervision_agent.run_supervision(_storyboard(), "", tmp_path, {})

    assert result["grade"] == "B"
    assert result["verdict"] == "warn"
    assert (tmp_path / "supervision_report.json").is_file()


def test_llm_call_uses_idle_and_wall_timeouts(monkeypatch):
    observed = {}
    fake_client = object()
    monkeypatch.setattr(supervision_agent, "_get_llm_client", lambda _config: fake_client)

    def fake_stream(messages, **kwargs):
        observed.update(messages=messages, **kwargs)
        return "review"

    monkeypatch.setattr(supervision_agent, "call_llm_stream", fake_stream)

    result = supervision_agent._call_llm(
        "storyboard",
        {"supervision_wall_timeout": 90, "supervision_idle_timeout": 30},
    )

    assert result == "review"
    assert observed["wall_timeout"] == 90.0
    assert observed["idle_timeout"] == 30.0
    assert observed["_client"] is fake_client


def test_client_ignores_ambient_socks_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")

    client = supervision_agent._get_llm_client({
        "api_key": "test-key",
        "base_url": "https://ark.invalid/api/v3",
    })
    try:
        assert client._client._trust_env is False
    finally:
        client.close()


def test_review_prompt_uses_bounded_semantic_projection():
    storyboard = _storyboard()
    storyboard.update({
        "title": "Future Station",
        "provider_prompt": "LEAKED_PROVIDER_PROMPT" * 10_000,
        "material_budget": {"receipt_body": "LEAKED_RECEIPT" * 10_000},
    })
    storyboard["shots"][0].update({
        "where": "rain-soaked platform",
        "what": "the train arrives",
        "start_state": "platform empty",
        "end_state": "doors open",
        "prompt": "LEAKED_SHOT_PROMPT" * 10_000,
        "storyboard_beats": [{
            "beat_id": "S01_P01",
            "duration_s": 5,
            "action": "train glides into view",
            "provider_receipt": "LEAKED_BEAT_RECEIPT" * 10_000,
        }],
    })

    prompt = supervision_agent._review_prompt(
        storyboard,
        "restrained cinematic style",
    )

    assert len(prompt) < 10_000
    assert "honcut.supervision-storyboard-projection.v1" in prompt
    assert "rain-soaked platform" in prompt
    assert "train glides into view" in prompt
    assert "LEAKED_PROVIDER_PROMPT" not in prompt
    assert "LEAKED_RECEIPT" not in prompt
    assert "LEAKED_SHOT_PROMPT" not in prompt
    assert "LEAKED_BEAT_RECEIPT" not in prompt


def test_blocking_mode_aborts_and_lists_issues(tmp_path, monkeypatch):
    response = {
        "grade": "D",
        "issues": [
            {
                "shot_order": 2,
                "category": "continuity",
                "severity": "critical",
                "description": "Character position reverses without a transition.",
            }
        ],
        "verdict": "block",
        "summary": "Continuity must be repaired.",
    }
    monkeypatch.setattr(
        supervision_agent, "_call_llm", lambda prompt, config: json.dumps(response)
    )

    with pytest.raises(supervision_agent.SupervisionBlockedError) as exc_info:
        supervision_agent.run_supervision(
            _storyboard(), "", tmp_path, {"supervision_blocking": True}
        )

    assert "shot 2" in str(exc_info.value)
    assert "Character position reverses" in str(exc_info.value)
    assert json.loads((tmp_path / "supervision_report.json").read_text()) == response


def test_disabled_mode_skips_without_call_or_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        supervision_agent,
        "_call_llm",
        lambda *args, **kwargs: pytest.fail("disabled supervision called the LLM"),
    )

    result = supervision_agent.run_supervision(
        _storyboard(), "", tmp_path, {"supervision": False}
    )

    assert result == {"status": "skipped", "reason": "supervision disabled"}
    assert not (tmp_path / "supervision_report.json").exists()
