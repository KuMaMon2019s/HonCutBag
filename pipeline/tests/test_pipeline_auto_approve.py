import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pipeline_runner


class _FakeApp:
    def __init__(self, final_state):
        self.final_state = final_state

    def invoke(self, initial_state, config):
        return {**initial_state, **self.final_state}


class _FakeGraph:
    def __init__(self, final_state):
        self.final_state = final_state

    def compile(self, checkpointer=None):
        return _FakeApp(self.final_state)


class _FakeSaver:
    def __enter__(self):
        return object()


def _run_with_final_state(monkeypatch, tmp_path, final_state, *, auto_approve):
    monkeypatch.setattr(pipeline_runner, "LANGGRAPH_AVAILABLE", True)
    monkeypatch.setattr(
        pipeline_runner,
        "build_pipeline_graph",
        lambda auto_approve=False: _FakeGraph(final_state),
    )
    monkeypatch.setattr(pipeline_runner, "get_sqlite_checkpointer", lambda output_dir: _FakeSaver())
    return pipeline_runner.run_pipeline(
        text="test",
        output_dir=str(tmp_path),
        dry_run=True,
        auto_approve=auto_approve,
    )


def test_returned_interrupt_is_persisted_as_interrupted(monkeypatch, tmp_path):
    report = _run_with_final_state(
        monkeypatch,
        tmp_path,
        {"__interrupt__": ("review_storyboard",), "status": "running"},
        auto_approve=False,
    )

    persisted = (tmp_path / "pipeline_report.json").read_text(encoding="utf-8")
    assert report["status"] == "interrupted"
    assert '"status": "interrupted"' in persisted
    assert '"status": "running"' not in persisted


def test_auto_approve_removes_human_review_node():
    if not pipeline_runner.LANGGRAPH_AVAILABLE:
        return

    manual_graph = pipeline_runner.build_pipeline_graph(auto_approve=False)
    automatic_graph = pipeline_runner.build_pipeline_graph(auto_approve=True)

    assert "review_storyboard" in manual_graph.nodes
    assert "review_storyboard" not in automatic_graph.nodes


def test_auto_approve_logs_skip_and_never_persists_running(monkeypatch, tmp_path, capsys):
    report = _run_with_final_state(
        monkeypatch,
        tmp_path,
        {"status": "running"},
        auto_approve=True,
    )

    output = capsys.readouterr().out
    persisted = (tmp_path / "pipeline_report.json").read_text(encoding="utf-8")
    assert "自动跳过人工审核节点 (--auto-approve)" in output
    assert report["status"] == "failed"
    assert '"status": "running"' not in persisted
