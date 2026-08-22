"""Regression tests for the canonical HonCut Phase 1-9.5 contract."""

import inspect
from pathlib import Path

import phases
from phases import pipeline_core
from phases.phase1 import phase1_pipeline, phase1_screenwriter
from phases.phase1.phase1_director import run_phase1_director
from phases.phase2 import phase2_storyboard
from phases.phase3.character_factory import _PROMPTS_DIR, load_template
from phases.phase9 import rhythm_editor, visual_post
from utils.artifact_chain import ARTIFACT_CHAIN, PHASE_SEQUENCE, phase_numbers_before
from utils.config import ToolPaths
from utils.progress_reporter import ProgressReporter


EXPECTED_PHASES = [
    "phase1", "phase2", "phase3", "phase4", "phase5",
    "phase6", "phase7", "phase8", "phase9", "phase9_5",
]


def test_phase_order_is_unique_and_shared():
    assert PHASE_SEQUENCE == EXPECTED_PHASES
    assert pipeline_core.PHASE_ORDER == EXPECTED_PHASES
    assert list(ARTIFACT_CHAIN) == EXPECTED_PHASES


def test_langgraph_uses_canonical_phase_node_ids():
    graph = pipeline_core.build_pipeline_graph(auto_approve=True)
    assert set(graph.nodes) == {
        "phase1", "phase2", "phase3", "phase4", "phase5",
        "phase6_txt2vid", "phase6_img2vid", "phase6_reference",
        "phase7", "phase8", "phase9", "phase9_5",
    }


def test_public_entrypoints_match_their_phase_numbers():
    assert phases.run_phase1 is phase1_pipeline.run_phase1
    assert phases.run_phase2 is phase2_storyboard.run_phase2
    assert phases.run_phase1_director is run_phase1_director
    assert phases.run_phase1_screenwriter is phase1_screenwriter.run_phase1_screenwriter
    assert pipeline_core.run_phase1 is not phases.run_phase1
    assert pipeline_core.run_phase2 is not phases.run_phase2


def test_resume_from_converts_phase_names_to_live_skip_numbers():
    assert phase_numbers_before("phase1") == []
    assert phase_numbers_before("phase6") == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert phase_numbers_before("phase9_5") == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_progress_reporter_covers_every_phase_without_gaps():
    reporter = object.__new__(ProgressReporter)
    assert [reporter._phase_index(name) for name in EXPECTED_PHASES] == list(range(10))


def test_moved_modules_resolve_pipeline_and_vendor_resources():
    repo_root = Path(__file__).resolve().parents[2]
    assert _PROMPTS_DIR == repo_root / "pipeline" / "prompts"
    assert ToolPaths.PROMPTS_DIR == _PROMPTS_DIR
    assert load_template()
    expected_om = repo_root / "vendor" / "video_tools" / "tools"
    assert Path(rhythm_editor.OM_TOOLS_DIR) == expected_om
    assert visual_post.OM_TOOLS_DIR == expected_om
    assert expected_om.is_dir()


def test_video_qa_only_runs_in_phase9_5_node():
    assert "run_video_qa" not in inspect.getsource(pipeline_core.run_phase9)
    assert ARTIFACT_CHAIN["phase9_5"]["produces"] == "video_qa_report.json"
