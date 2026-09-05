from __future__ import annotations

import copy

import pytest

from phases.phase2.storyboard_pose_atlas import build_pose_atlas_plan
from phases.phase6.action_execution_prompt import (
    ACTION_EXECUTION_BRIEF_SCHEMA,
    compile_action_execution_brief,
    render_action_execution_brief,
)
from utils.action_kinematics import (
    apply_generation_kinematics_projection,
    compile_source_kinematics,
)


def _plan() -> dict:
    return build_pose_atlas_plan(
        {
            "beat_id": "S01_P01",
            "duration_s": 7,
            "planner_version": "honcut.secondary-storyboard.v17",
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


@pytest.mark.parametrize(
    ("action", "targets", "expected_contact"),
    [
        ("actor sidesteps and lowers weight", [], False),
        ("actor grips blue baton and strikes target", ["blue baton"], True),
    ],
)
def test_action_execution_brief_preserves_contact_and_prop_semantics(
    action,
    targets,
    expected_contact,
) -> None:
    plan = build_pose_atlas_plan(
        {
            "beat_id": "S01_P01",
            "duration_s": 7,
            "planner_version": "honcut.secondary-storyboard.v17",
            "generation_action_units": [
                {
                    "unit_id": "GAU001",
                    "source_action_unit_id": "AU001",
                    "source_event_id": 1,
                    "source_generation_unit_indexes": [1],
                    "source_micro_action_indexes": [1],
                    "ledger_indexes": [0],
                    "actions": [action],
                    "performers": ["actor"],
                    "targets": targets,
                }
            ],
            "character_ids": ["actor"],
        }
    )
    brief = compile_action_execution_brief(
        beat_id="S01_P01",
        action_prompt=action,
        start_state="already moving",
        end_state="balanced completion",
        target_duration_s=7,
        action_groups=plan["action_groups"],
        pose_samples=plan["pose_samples"],
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

    contact = brief["action_groups"][0]["observable_mechanics"]["contact_profile"]
    assert bool(contact["required_targets"]) is expected_contact
    if expected_contact:
        assert contact["required_targets"] == targets
    else:
        assert contact["mode"] == "no invented target contact"


def test_action_execution_brief_projects_canonical_phase_and_channels() -> None:
    mechanics = {
        "micro_action_index": 1,
        "micro_action": "actor lunges and strikes",
        "performer": "actor",
        "technique": "right-hand forward strike",
        "side": "right",
        "limbs": ["right arm", "right hand", "left leg", "left foot", "waist", "head"],
        "footwork": "left foot supports while right foot advances",
        "torso": "waist leans forward",
        "weight_shift": "weight shifts forward",
        "direction": "forward",
        "contact": "right hand reaches contact",
        "end_pose": "right foot forward",
    }
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v17",
        "character_ids": ["actor"],
        "body_action_contract": {
            "schema": "honcut.body-action-choreography.v2",
            "required": True,
            "valid": True,
            "beats": [{**mechanics, "kinematics": compile_source_kinematics(mechanics)}],
        },
        "generation_action_units": [
            {
                "unit_id": "GAU001",
                "source_action_unit_id": "AU001",
                "source_event_id": 1,
                "source_generation_unit_indexes": [1],
                "source_micro_action_indexes": [1],
                "ledger_indexes": [0],
                "actions": ["actor lunges and strikes"],
                "performers": ["actor"],
                "targets": ["target"],
            }
        ],
    }
    apply_generation_kinematics_projection(beat)
    plan = build_pose_atlas_plan(beat, known_actor_roles=("actor",))

    brief = _compile(plan)
    group = brief["action_groups"][0]["canonical_kinematics"]
    rendered = render_action_execution_brief(brief)

    assert group["projection_sha256s"] == brief["kinematics_projection_sha256s"]
    assert group["ordered_phase_ids"]
    assert "right_arm" in group["performer_active_channels"]["actor"]
    assert "运动学执行=" in rendered
    assert "不得增添翻转或旋转" in rendered
    assert brief["media_roles"] == _compile(plan)["media_roles"]


def test_ten_recoveries_keep_pose_and_phase6_fingerprints_stable() -> None:
    mechanics = {
        "micro_action_index": 1,
        "micro_action": "actor lunges and strikes",
        "performer": "actor",
        "technique": "right-hand forward strike",
        "side": "right",
        "limbs": ["right arm", "right hand", "left leg", "left foot", "waist", "head"],
        "footwork": "left foot supports while right foot advances",
        "torso": "waist leans forward",
        "weight_shift": "weight shifts forward",
        "direction": "forward",
        "contact": "right hand reaches contact",
        "end_pose": "right foot forward",
    }
    source = {
        "beat_id": "S01_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v17",
        "character_ids": ["actor"],
        "body_action_contract": {
            "schema": "honcut.body-action-choreography.v2",
            "required": True,
            "valid": True,
            "beats": [{**mechanics, "kinematics": compile_source_kinematics(mechanics)}],
        },
        "generation_action_units": [
            {
                "unit_id": "GAU001",
                "source_action_unit_id": "AU001",
                "source_event_id": 1,
                "source_generation_unit_indexes": [1],
                "source_micro_action_indexes": [1],
                "ledger_indexes": [0],
                "actions": ["actor lunges and strikes"],
                "performers": ["actor"],
                "targets": ["target"],
            }
        ],
    }
    fingerprints = []

    for _ in range(10):
        beat = copy.deepcopy(source)
        projection = apply_generation_kinematics_projection(beat)
        plan = build_pose_atlas_plan(beat, known_actor_roles=("actor",))
        brief = _compile(plan)
        fingerprints.append(
            (
                projection["projection_sha256"],
                plan["plan_sha256"],
                tuple(
                    sample["pose_contract"]["pose_fingerprint"]
                    for sample in plan["pose_samples"]
                ),
                brief["brief_sha256"],
            )
        )

    assert len(set(fingerprints)) == 1
