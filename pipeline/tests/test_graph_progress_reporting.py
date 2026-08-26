from __future__ import annotations

from runtime import pipeline_execution


class _FakeGraph:
    def __init__(self, events):
        self._events = events
        self.stream_modes = None

    def stream(self, invocation_input, *, config, stream_mode):
        assert invocation_input == {"status": "running"}
        assert config == {"configurable": {"thread_id": "run-1"}}
        self.stream_modes = stream_mode
        yield from self._events


class _RecordingReporter:
    def __init__(self):
        self.calls = []

    def phase_start(self, phase_id, phase_name):
        self.calls.append(("start", phase_id, phase_name))

    def phase_done(self, phase_id, message, duration_s=None):
        self.calls.append(("done", phase_id, message, duration_s))


def test_stream_graph_reports_canonical_phase_start_and_completion():
    final_state = {
        "status": "completed",
        "phase_results": {"phase6": {"status": "done", "duration_s": 12.5}},
    }
    graph = _FakeGraph(
        [
            ("values", {"status": "running"}),
            ("tasks", {"id": "task-1", "name": "phase6_img2vid", "input": {}}),
            (
                "tasks",
                {
                    "id": "task-1",
                    "name": "phase6_img2vid",
                    "error": None,
                    "result": {
                        "phase_results": {
                            "phase6": {"status": "done", "duration_s": 12.5}
                        }
                    },
                    "interrupts": [],
                },
            ),
            ("values", final_state),
        ]
    )
    reporter = _RecordingReporter()

    result = pipeline_execution._stream_graph_with_progress(
        graph,
        {"status": "running"},
        config={"configurable": {"thread_id": "run-1"}},
        reporter=reporter,
    )

    assert result == final_state
    assert graph.stream_modes == ["tasks", "values"]
    assert reporter.calls == [
        ("start", "phase6", "视频生成"),
        ("done", "phase6", "视频生成完成", 12.5),
    ]


def test_stream_graph_leaves_failed_phase_current_without_marking_it_done():
    final_state = {
        "status": "failed",
        "phase_results": {
            "phase5": {
                "status": "error",
                "error": "Storyboard QA blocked Phase 6",
            }
        },
    }
    graph = _FakeGraph(
        [
            ("values", {"status": "running"}),
            ("tasks", {"id": "task-5", "name": "phase5", "input": {}}),
            (
                "tasks",
                {
                    "id": "task-5",
                    "name": "phase5",
                    "error": None,
                    "result": {
                        "status": "failed",
                        "phase_results": final_state["phase_results"],
                    },
                    "interrupts": [],
                },
            ),
            ("values", final_state),
        ]
    )
    reporter = _RecordingReporter()

    result = pipeline_execution._stream_graph_with_progress(
        graph,
        {"status": "running"},
        config={"configurable": {"thread_id": "run-1"}},
        reporter=reporter,
    )

    assert result == final_state
    assert reporter.calls == [("start", "phase5", "分镜质检闸门")]
