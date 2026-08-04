import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline_runner


def test_phase3_error_stops_sequential_pipeline(monkeypatch, tmp_path):
    storyboard = {"shots": []}
    characters = {"characters": [{"name": "test character"}]}
    calls = []

    monkeypatch.setattr(
        pipeline_runner,
        "run_phase2",
        lambda *args, **kwargs: {
            "status": "done",
            "_storyboard": storyboard,
            "_characters": characters,
        },
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_phase3",
        lambda *args, **kwargs: {"status": "error", "error": "missing front.png"},
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_phase4",
        lambda *args, **kwargs: calls.append("phase4") or {"status": "done"},
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_phase5",
        lambda *args, **kwargs: calls.append("phase5") or {"status": "done"},
    )

    report = pipeline_runner.run_pipeline(
        text="test",
        output_dir=str(tmp_path),
        dry_run=True,
        # A non-empty skip list selects the sequential runner; Phase 2.5 is
        # irrelevant to this regression and would otherwise need image mocks.
        skip_phase=[1, 2.5],
    )

    assert report["status"] == "failed"
    assert report["phases"]["3"]["status"] == "error"
    assert "4" not in report["phases"]
    assert "5" not in report["phases"]
    assert calls == []


def test_phase3_error_routes_langgraph_to_end(monkeypatch, tmp_path):
    if not pipeline_runner.LANGGRAPH_AVAILABLE:
        return

    monkeypatch.setattr(
        pipeline_runner,
        "run_phase3",
        lambda *args, **kwargs: {"status": "error", "error": "missing front.png"},
    )

    command = pipeline_runner.node_phase3(
        {
            "output_dir": str(tmp_path),
            "characters": [{"name": "test character"}],
            "dry_run": True,
            "phase_results": {},
            "completed_phases": [],
            "skip_phase": [],
        }
    )

    assert command.goto == pipeline_runner.END
    assert command.update["status"] == "failed"
    assert command.update["phase_results"]["phase3"]["status"] == "error"
    assert "phase3" not in command.update["completed_phases"]
