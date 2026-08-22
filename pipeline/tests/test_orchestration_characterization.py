"""Behavior locks for the orchestration surface before the v2 refactor."""

from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.graph import END, START

import pipeline_runner
from graph.composition import build_pipeline_graph as build_composed_graph
from graph.context import initial_state_from_config
from graph.migrations import (
    CURRENT_STATE_SCHEMA_VERSION,
    LEGACY_STATE_ALIASES,
    StateMigrationError,
    migrate_state,
)
from graph.state import HonCutState
from graph.workflow import PHASE_NODE_IDS, build_workflow
from phases import pipeline_core
from runtime.generation_tasks import GenerationTaskStore
from runtime.checkpoint_resolution import (
    ResumeResolutionError,
    STAGE_CHECKPOINT_SCHEMA_VERSION,
    resolve_resume_snapshot,
)
from schemas.workflow import GraphRunConfig
from utils.artifact_chain import (
    can_resume_from,
    invalidate_checkpoints_from,
    save_checkpoint,
)


def _identity_node(state: HonCutState) -> dict[str, Any]:
    return dict(state)


def test_live_graph_config_seeds_complete_json_safe_compatibility_state():
    config = GraphRunConfig(
        run_id="fingerprint-1",
        project_id="studio-a",
        input_text="story",
        output_dir="/tmp/honcut-characterization",
        target_duration_s=12,
        shot_duration_s=4,
        transition_duration_s=0.25,
        project_video_spec={"width": 1920, "height": 1080},
        resume_from="phase5",
    )

    state = initial_state_from_config(config)

    assert state["run_id"] == state["run_fingerprint"] == "fingerprint-1"
    assert state["project_id"] == "studio-a"
    assert state["state_schema_version"] == CURRENT_STATE_SCHEMA_VERSION
    assert state["input_text"] == state["text"] == "story"
    assert state["target_duration_s"] == state["duration"] == 12
    assert state["shot_duration_s"] == state["shot_duration"] == 4
    assert state["project_video_spec"] == {"width": 1920, "height": 1080}
    assert state["resume_from"] == "phase5"
    assert json.loads(json.dumps(state))["phase_results"] == {}

    canonical = initial_state_from_config(config, include_legacy_aliases=False)
    assert LEGACY_STATE_ALIASES.isdisjoint(canonical)


def test_legacy_state_migrates_deterministically_and_future_versions_fail():
    migrated = migrate_state(
        {
            "text": "legacy story",
            "duration": 30,
            "shot_duration": 5,
            "transition_duration": 0.5,
            "shots": [{"shot_id": "S01"}],
            "videos": ["shots/S01/output.mp4"],
            "quality_report": {"failed_shots": []},
            "error": "legacy failure",
            "run_fingerprint": "run-old",
        }
    )

    assert migrated["state_schema_version"] == CURRENT_STATE_SCHEMA_VERSION
    assert migrated["project_id"] == "local"
    assert migrated["run_id"] == "run-old"
    assert migrated["input_text"] == "legacy story"
    assert migrated["target_duration_s"] == 30
    assert migrated["shot_ids"] == ["S01"]
    assert migrated["generated_shots"] == ["shots/S01/output.mp4"]
    assert migrated["errors"][-1]["message"] == "legacy failure"
    assert LEGACY_STATE_ALIASES.isdisjoint(migrated)

    with pytest.raises(StateMigrationError, match="newer than supported"):
        migrate_state({"state_schema_version": CURRENT_STATE_SCHEMA_VERSION + 1})


def test_production_composition_strips_legacy_node_patch_aliases(tmp_path):
    owner = SimpleNamespace(
        AVG_SHOT_DURATION=5,
        run_phase1=lambda **_kwargs: {
            "status": "error",
            "error": "screenwriter failed",
        },
    )
    graph = build_composed_graph(phase_owner=owner)
    state = initial_state_from_config(
        GraphRunConfig(input_text="story", output_dir=str(tmp_path)),
        include_legacy_aliases=False,
    )

    result = graph.nodes["phase1"].runnable.invoke(state)

    assert "error" not in result.update
    assert result.update["errors"][-1]["message"] == (
        "Phase 1 failed: screenwriter failed"
    )


def test_production_composition_uses_canonical_state_and_core_is_only_a_facade():
    assert pipeline_core.HonCutState is HonCutState
    assert "StateGraph(" not in inspect.getsource(build_composed_graph)
    core_facade = inspect.getsource(pipeline_core.build_pipeline_graph)
    assert "graph.composition" in core_facade
    assert "nodes={" not in core_facade


def test_cli_dispatch_preserves_public_arguments_and_exit_contract(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    report = {
        "status": "completed",
        "phases": {"phase1": {"status": "done"}},
    }

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(pipeline_runner._core, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline_runner, "_record_report_checkpoints", lambda *_: None)
    monkeypatch.setattr(pipeline_runner, "_record_run_memory", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "honcut",
            "--text",
            "story",
            "--output-dir",
            str(tmp_path),
            "--project-id",
            "studio-a",
            "--duration",
            "12",
            "--phase",
            "phase1",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        pipeline_runner.main()

    assert exit_info.value.code == 0
    assert captured["text"] == "story"
    assert captured["duration"] == 12
    assert captured["dry_run"] is True
    assert captured["output_dir"] == str(tmp_path)
    assert captured["project_id"] == "studio-a"
    assert captured["skip_phase"] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]


def test_workflow_topology_and_terminal_routes_are_stable():
    graph = build_workflow(
        state_schema=HonCutState,
        nodes=dict.fromkeys(PHASE_NODE_IDS, _identity_node),
        route_phase5=lambda _state: "txt2vid",
        quality_gate_router=lambda _state: "pass",
        route_after_phase8=lambda _state: "continue",
        route_after_phase9=lambda _state: "continue",
    )

    assert set(graph.nodes) == set(PHASE_NODE_IDS)
    assert graph.edges == {
        (START, "phase1"),
        ("phase1", "phase2"),
        ("phase2", "phase3"),
        ("phase3", "phase4"),
        ("phase4", "phase5"),
        ("phase6_txt2vid", "phase7"),
        ("phase6_img2vid", "phase7"),
        ("phase6_reference", "phase7"),
        ("phase9_5", END),
    }
    phase5_branch = next(iter(graph.branches["phase5"].values()))
    phase7_branch = next(iter(graph.branches["phase7"].values()))
    assert phase5_branch.ends == {
        "txt2vid": "phase6_txt2vid",
        "img2vid": "phase6_img2vid",
        "reference": "phase6_reference",
    }
    assert phase7_branch.ends == {
        "pass": "phase8",
        "block": END,
    }
    graph.compile()


def test_cli_records_receipts_only_for_successful_reported_phases(monkeypatch, tmp_path):
    recorded: list[tuple[Path, str, dict[str, Any]]] = []
    monkeypatch.setattr(
        pipeline_runner._core,
        "_record_stage_checkpoint",
        lambda output_dir, phase, result: recorded.append((output_dir, phase, result)),
    )

    pipeline_runner._record_report_checkpoints(
        {
            "phases": {
                "phase1": {"status": "done", "outputs": ["STORYBOARD.json"]},
                "2": {"status": "done", "outputs": ["SHOT_STORYBOARDS.json"]},
                "phase3": {"status": "skipped"},
                "phase4": {"status": "error", "error": "boom"},
            }
        },
        tmp_path,
    )

    assert [(phase, result["status"]) for _, phase, result in recorded] == [
        ("phase1", "done"),
        ("phase2", "done"),
    ]
    assert all(output_dir == tmp_path for output_dir, _, _ in recorded)


def test_resume_prerequisites_and_downstream_receipt_invalidation(tmp_path):
    assert can_resume_from("phase6", tmp_path) is False
    (tmp_path / "shots").mkdir()
    (tmp_path / "storyboard_qa_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "STORYBOARD.json").write_text('{"shots": []}', encoding="utf-8")
    assert can_resume_from("phase6", tmp_path) is True

    for phase in ("phase1", "phase2", "phase3"):
        save_checkpoint(phase, tmp_path, {"exists": True})

    assert invalidate_checkpoints_from("phase2", tmp_path) == ["phase2", "phase3"]
    phase1 = json.loads((tmp_path / "checkpoint_phase1.json").read_text())
    phase2 = json.loads((tmp_path / "checkpoint_phase2.json").read_text())
    phase3 = json.loads((tmp_path / "checkpoint_phase3.json").read_text())
    assert phase1["status"] == "done"
    assert phase2["status"] == phase3["status"] == "stale"
    assert phase2["invalidated_by_resume_from"] == "phase2"


def test_resume_resolver_prefers_graph_and_honors_stale_artifact_boundary(tmp_path):
    (tmp_path / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_schema_version": STAGE_CHECKPOINT_SCHEMA_VERSION,
                "run_fingerprint": "run-1",
                "completed": ["phase1", "phase2", "phase3"],
                "results": {
                    "phase1": {"status": "done"},
                    "phase2": {"status": "done"},
                    "phase3": {"status": "done"},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_phase2.json").write_text(
        json.dumps({"phase": "phase2", "status": "stale"}),
        encoding="utf-8",
    )
    graph_state = {
        "text": "legacy",
        "duration": 30,
        "run_fingerprint": "run-1",
        "completed_phases": ["phase1", "phase2"],
        "phase_results": {
            "phase1": {"status": "done"},
            "phase2": {"status": "done"},
        },
    }

    snapshot = resolve_resume_snapshot(
        tmp_path,
        run_fingerprint="run-1",
        project_id="local",
        graph_states=[("graph", graph_state)],
    )

    assert snapshot.source == "graph"
    assert snapshot.completed_phases == ("phase1",)
    assert snapshot.state["input_text"] == "legacy"
    assert snapshot.state["state_schema_version"] == CURRENT_STATE_SCHEMA_VERSION


def test_resume_resolver_rejects_future_and_cross_project_state(tmp_path):
    with pytest.raises(ResumeResolutionError, match="newer than supported"):
        resolve_resume_snapshot(
            tmp_path,
            run_fingerprint="run-1",
            project_id="local",
            graph_states=[
                (
                    "graph",
                    {
                        "state_schema_version": CURRENT_STATE_SCHEMA_VERSION + 1,
                        "run_fingerprint": "run-1",
                    },
                )
            ],
        )

    with pytest.raises(ResumeResolutionError, match="project_id"):
        resolve_resume_snapshot(
            tmp_path,
            run_fingerprint="run-1",
            project_id="project-b",
            graph_states=[
                (
                    "graph",
                    {
                        "state_schema_version": CURRENT_STATE_SCHEMA_VERSION,
                        "run_id": "run-1",
                        "run_fingerprint": "run-1",
                        "project_id": "project-a",
                    },
                )
            ],
        )


def test_sqlite_checkpointer_initialization_fails_open_to_uncheckpointed_graph(
    monkeypatch, tmp_path
):
    if not pipeline_core.LANGGRAPH_AVAILABLE:
        pytest.skip("LangGraph is not installed")

    monkeypatch.setattr(pipeline_core, "_sqlite_saver_instance", None)
    monkeypatch.setattr(pipeline_core, "_sqlite_saver_path", None)

    def fail_to_open(_path):
        raise OSError("database unavailable")

    monkeypatch.setattr(pipeline_core.SqliteSaver, "from_conn_string", fail_to_open)
    assert pipeline_core.get_sqlite_checkpointer(tmp_path) is None


def test_generation_task_dedupe_is_payload_exact_and_provider_scoped(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    seedance = store.enqueue(
        run_id="run-1",
        task_type="video",
        media_type="video",
        resource_id="shot-1",
        payload={"prompt": "same", "duration": 5},
        provider_id="seedance",
    )
    duplicate = store.enqueue(
        run_id="run-1",
        task_type="video",
        media_type="video",
        resource_id="shot-1",
        payload={"duration": 5, "prompt": "same"},
        provider_id="seedance",
    )
    bridge = store.enqueue(
        run_id="run-1",
        task_type="video",
        media_type="video",
        resource_id="shot-1",
        payload={"prompt": "same", "duration": 5},
        provider_id="bridge",
    )

    assert seedance.deduped is False
    assert duplicate.deduped is True
    assert duplicate.task.task_id == seedance.task.task_id
    assert bridge.deduped is False
    assert bridge.task.task_id != seedance.task.task_id

    with pytest.raises(RuntimeError, match="payload changed"):
        store.enqueue(
            run_id="run-1",
            task_type="video",
            media_type="video",
            resource_id="shot-1",
            payload={"prompt": "changed", "duration": 5},
            provider_id="seedance",
        )


def test_pipeline_report_shape_is_json_safe_and_completed_reports_drop_error(tmp_path):
    pipeline_core._write_report(
        {
            "status": "completed",
            "error": "stale error",
            "input_text_length": 5,
            "duration_target_s": 10,
            "dry_run": True,
            "output_dir": tmp_path,
            "resumed": False,
            "phases": {"phase1": {"status": "done", "duration_s": 0.1}},
            "total_duration_s": 0.2,
            "final_video": "",
        },
        tmp_path,
    )

    report = json.loads((tmp_path / "pipeline_report.json").read_text())
    assert set(report) == {
        "status",
        "input_text_length",
        "duration_target_s",
        "dry_run",
        "output_dir",
        "resumed",
        "phases",
        "total_duration_s",
        "final_video",
    }
    assert report["status"] == "completed"
    assert report["phases"]["phase1"]["status"] == "done"
    assert report["output_dir"] == str(tmp_path)
