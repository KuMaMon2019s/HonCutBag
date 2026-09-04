from __future__ import annotations

import copy

import pytest

from phases.phase2.storyboard_pose_atlas import build_pose_atlas_plan
from phases.phase6.action_execution_prompt import (
    ACTION_EXECUTION_BRIEF_SCHEMA,
    compile_action_execution_brief,
    render_action_execution_brief,
)


def _plan() -> dict:
    return build_pose_atlas_plan(
        {
            "beat_id": "S01_P01",
            "duration_s": 7,
            "planner_version": "honcut.secondary-storyboard.v16",
            "generation_action_units": [
                {
                    "unit_id": "GAU001",
                    "source_action_unit_id": "AU001",
                    "source_event_id": 1,
                    "source_generation_unit_indexes": [1],
                    "source_micro_action_indexes": [1],
                    "ledger_indexes": [0],
                    "actions": ["actor shifts weight, evades, then blocks contact"],
                    "performers": ["actor"],
                    "targets": ["attacker"],
                }
            ],
            "character_ids": ["actor"],
        }
    )


def _compile(plan: dict) -> dict:
    return compile_action_execution_brief(
        beat_id="S01_P01",
        action_prompt="actor shifts weight, evades, then blocks contact",
        start_state="already moving",
        end_state="balanced block",
        target_duration_s=7,
        action_groups=plan["action_groups"],
        pose_samples=plan["pose_samples"],
        timing_contract=plan["timing_contract"],
        media_manifest=[
            {
                "prompt_index": "图片1",
                "responsibility": "storyboard_pose_atlas",
                "narrative_cell_ids": [sample["sample_id"] for sample in plan["pose_samples"]],
                "sha256": "a" * 64,
            }
        ],
        prompt_context={"camera_movement": "track right"},
        canonical_visual_contract_sha256="b" * 64,
    )


def test_action_execution_brief_is_deterministic_and_action_complete() -> None:
    plan = _plan()
    first = _compile(plan)
    second = _compile(plan)

    assert first == second
    assert first["schema"] == ACTION_EXECUTION_BRIEF_SCHEMA
    assert first["ordered_action_group_ids"] == ["S01_P01_A01"]
    assert first["initial_anchor_sample_ids"] == ["G01"]
    rendered = render_action_execution_brief(first)
    assert rendered.count("S01_P01_A01") == 1
    assert "零时长初始锚点=G01" in rendered
    assert "weight_transfer_profile=" in rendered
    assert "contact_profile=" in rendered


def test_action_execution_brief_rejects_foreign_pose_lineage() -> None:
    plan = _plan()
    samples = copy.deepcopy(plan["pose_samples"])
    samples[0]["pose_contract"]["secondary_beat_id"] = "S01_P02"

    with pytest.raises(ValueError, match="future or foreign"):
        compile_action_execution_brief(
            beat_id="S01_P01",
            action_prompt="actor shifts weight, evades, then blocks contact",
            start_state="already moving",
            end_state="balanced block",
            target_duration_s=7,
            action_groups=plan["action_groups"],
            pose_samples=samples,
            timing_contract=plan["timing_contract"],
            media_manifest=[
                {
                    "prompt_index": "图片1",
                    "responsibility": "storyboard_pose_atlas",
                    "sha256": "a" * 64,
                }
            ],
            canonical_visual_contract_sha256="b" * 64,
        )
