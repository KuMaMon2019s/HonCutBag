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
