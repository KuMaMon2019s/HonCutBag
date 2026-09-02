"""Bounded guard against new Phase 1-5 raw probabilistic visual gates."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase3.character_factory import _quality_control_identity_detail
from quality.character_reference_qa import parse_identity_detail_qa


INVENTORY_PATH = (
    ROOT / "pipeline" / "tests" / "fixtures" / "phase1_5_visual_gate_inventory.json"
)
BASELINE_PIPELINE_CORE_SHA256 = (
    "2cd41fb3ea7c77d1b29d41488e9608fa0abe2f20403c975a4131d2a5d6cbdd5a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase1_5_visual_gate_inventory_is_complete_and_bounded():
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["schema"] == "honcut.phase1-5-visual-gate-inventory.v1"
    entries = {value["owner"]: value for value in inventory["entries"]}
    assert set(entries) == {
        "phase2.storyboard_assets",
        "phase3.character_reference_qa",
        "phase3.character_performance_qa",
        "phase5.storyboard_qa_gate",
        "phase5.optional_supervision",
    }
    classifications = {value["classification"] for value in entries.values()}
    assert classifications == {
        "deterministic_validation",
        "ledgered_policy",
        "out_of_contract_raw_probabilistic_gate",
    }
    raw_gate = entries["phase5.optional_supervision"]
    raw_path = ROOT / raw_gate["path"]
    assert _sha256(raw_path) == raw_gate["source_sha256"]
    assert raw_gate["follow_up_change"] == (
        "stabilize-phase5-optional-supervision-gate"
    )
    raw_source = raw_path.read_text(encoding="utf-8")
    assert 'config.get("supervision_blocking", False)' in raw_source


def test_vlm_call_sites_are_ledgered_or_exactly_inventoried():
    phase_roots = [SRC / "phases" / f"phase{value}" for value in range(1, 6)]
    phase_review_as = {
        path.relative_to(ROOT).as_posix()
        for phase_root in phase_roots
        for path in phase_root.rglob("*.py")
        if "review_as(" in path.read_text(encoding="utf-8")
    }
    assert phase_review_as == {
        "pipeline/src/phases/phase5/storyboard_qa_gate.py"
    }

    ledgered_quality = {
        "pipeline/src/quality/character_reference_qa.py",
        "pipeline/src/quality/character_performance_qa.py",
    }
    for relative in ledgered_quality:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "QALedger" in source
        assert "decide_visual_qa" in source


def test_prop_detail_model_passed_is_diagnostic_only():
    tree = ast.parse(inspect.getsource(parse_identity_detail_qa))
    conditional_passed_reads = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            continue
        for child in ast.walk(node.test):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "payload"
                and isinstance(child.slice, ast.Constant)
                and child.slice.value == "passed"
            ):
                conditional_passed_reads.append(child)
    assert conditional_passed_reads == []


def test_prop_detail_owner_has_one_review_and_no_correction_loop():
    source = inspect.getsource(_quality_control_identity_detail)
    assert source.count("review_identity_detail_reference(") == 1
    assert "for attempt in range" not in source
    assert "correction=" not in source
    assert "automatic re-review and redraw are disabled" in source


def test_graph_and_sequential_paths_share_phase3_owner():
    composition = (
        SRC / "graph" / "composition.py"
    ).read_text(encoding="utf-8")
    sequential = (
        SRC / "runtime" / "pipeline_execution.py"
    ).read_text(encoding="utf-8")
    assert "phase3_node(state, runner=_resolve_phase_owner(phase_owner).run_phase3)" in (
        composition
    )
    assert "run_phase3 = phase_owner.run_phase3" in sequential
    assert "p3 = run_phase3(output_dir, characters_data, dry_run)" in sequential


def test_pipeline_core_remains_byte_identical_to_run16_baseline():
    pipeline_core = SRC / "phases" / "pipeline_core.py"
    assert _sha256(pipeline_core) == BASELINE_PIPELINE_CORE_SHA256
