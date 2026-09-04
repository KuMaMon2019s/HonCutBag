from __future__ import annotations

import copy
import hashlib
import inspect
from io import BytesIO

import pytest
from PIL import ImageFont

from phases.phase2 import storyboard_guide_pose as pose_owner
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import _task_payload
from schemas.continuity import GenerationChunk


def _cell(label: str = "S01_G01", stage: str = "action_progress") -> dict:
    return {
        "cell": int(label.rsplit("G", 1)[-1]),
        "label": label,
        "primary_shot_id": "S01",
        "secondary_beat_id": "S01_P01",
        "stage": stage,
        "camera_movement": "camera moves right",
    }


def _unit(
    unit_id: str,
    action: str,
    *,
    performers: tuple[str, ...] = ("actor-alpha",),
    targets: tuple[str, ...] = (),
    ledger_index: int = 0,
) -> dict:
    return {
        "unit_id": unit_id,
        "actions": [action],
        "performers": list(performers),
        "targets": list(targets),
        "ledger_indexes": [ledger_index],
        "source_event_id": ledger_index + 1,
        "source_generation_unit_indexes": [ledger_index + 1],
    }


def _beat(*units: dict, character_ids: tuple[str, ...] = ()) -> dict:
    return {
        "beat_id": "S01_P01",
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": list(units),
        "character_ids": list(character_ids),
        "action": " then ".join(unit["actions"][0] for unit in units),
    }


@pytest.fixture
def anonymized_action_catalog() -> tuple[dict, ...]:
    return (
        _unit(
            "GAU001",
            "automatic door moves left while light shifts",
            performers=("automatic door",),
            targets=("light",),
            ledger_index=0,
        ),
        _unit(
            "GAU002",
            "actor-alpha appears as three guards emerge",
            performers=("actor-alpha", "guard-1", "guard-2", "guard-3"),
            ledger_index=1,
        ),
        _unit("GAU003", "actor-alpha runs forward", ledger_index=2),
        _unit("GAU004", "actor-alpha dodges backward", ledger_index=3),
        _unit(
            "GAU005",
            "actor-alpha blocks guard-1",
            targets=("guard-1",),
            ledger_index=4,
        ),
        _unit(
            "GAU006",
            "actor-alpha strikes guard-1",
            targets=("guard-1",),
            ledger_index=5,
        ),
        _unit(
            "GAU007",
            "actor-alpha grabs guard-1 and throws left",
            targets=("guard-1",),
            ledger_index=6,
        ),
        _unit(
            "GAU008",
            "actor-alpha activates handheld device",
            targets=("handheld device",),
            ledger_index=7,
        ),
    )


def test_anonymized_catalog_covers_required_motion_families(
    anonymized_action_catalog,
):
    observed = []
    for unit in anonymized_action_catalog:
        contract = pose_owner.compile_pose_contracts(_beat(unit), [_cell()])[0]["pose_contract"]
        observed.append(contract["pose_family"])
        assert contract["action_bindings"][0]["unit_id"] == unit["unit_id"]
        assert contract["lineage_status"] == "canonical"
    assert observed == [
        "locomotion",
        "reveal",
        "locomotion",
        "evade",
        "block",
        "strike",
        "throw",
        "prop_use",
    ]


def test_pose_progress_changes_joint_geometry_and_fingerprint():
    unit = _unit("GAU001", "actor-alpha kicks right")
    cells = pose_owner.compile_pose_contracts(
        _beat(unit),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "action_progress"),
            _cell("S01_G03", "end"),
        ],
    )
    fingerprints = pose_owner.pose_fingerprints(cells)
    assert len(set(fingerprints)) == 3
    feet = [cell["pose_contract"]["geometry"]["actors"][0]["joints"]["left_foot"] for cell in cells]
    assert len({tuple(point) for point in feet}) == 3
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


@pytest.mark.parametrize(
    ("contact", "action"),
    [
        ("双手保持握持短棍，无实际格挡或击中接触", "actor-alpha 向右滑步后仰闪避"),
        ("keeps both hands on the prop without blocking or striking", "actor-alpha dodges right and leans back"),
    ],
)
def test_negated_contact_cannot_override_positive_pose_evidence(contact, action):
    unit = _unit("GAU001", action)
    beat = _beat(unit)
    beat["body_action_contract"] = {
        "beats": [
            {
                "micro_action_index": 1,
                "performer": "actor-alpha",
                "technique": action,
                "footwork": "右脚向右侧滑步，左脚支撑" if "闪避" in action else "right foot side-steps while the left foot supports",
                "torso": "降低重心并向后仰" if "闪避" in action else "lower the center of gravity and lean back",
                "weight_shift": "向右、向后转移" if "闪避" in action else "shift weight right and backward",
                "contact": contact,
                "end_pose": "低重心后倾闪避姿态" if "闪避" in action else "low backward-leaning evade pose",
            }
        ]
    }

    contract = pose_owner.compile_pose_contracts(beat, [_cell()])[0]["pose_contract"]

    assert contract["pose_family"] == "evade"
    assert contract["classification_evidence"]["field"] == "technique"
    assert contract["classification_evidence"]["polarity"] == "positive"
    assert any(
        item["field"] == "contact" and item["family"] in {"block", "strike"}
        for item in contract["classification_evidence"]["rejected_negated_matches"]
    )


def test_repeated_action_cells_form_monotonic_distinct_pose_samples():
    unit = _unit("GAU001", "actor-alpha dodges right and leans back")
    cells = pose_owner.compile_pose_contracts(
        _beat(unit),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "action_progress"),
            _cell("S01_G03", "action_progress"),
            _cell("S01_G04", "action_progress"),
            _cell("S01_G05", "end"),
        ],
    )

    progress = [cell["pose_contract"]["pose_progress"] for cell in cells]
    fingerprints = pose_owner.pose_fingerprints(cells)
    hips = [
        tuple(cell["pose_contract"]["geometry"]["actors"][0]["joints"]["left_hip"])
        for cell in cells
    ]

    assert progress == sorted(progress)
    assert progress[0] < progress[-1]
    assert len(set(progress)) == len(progress)
    assert len(set(fingerprints)) == len(fingerprints)
    assert len(set(hips)) >= 3
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_initial_ready_is_one_zero_time_anchor_before_dynamic_action():
    ready = _unit("GAU001", "actor-alpha takes a ready guard stance", ledger_index=0)
    evade = _unit("GAU002", "actor-alpha side-steps and dodges right", ledger_index=1)
    cells = pose_owner.compile_pose_contracts(
        _beat(ready, evade),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "action_progress"),
            _cell("S01_G03", "action_progress"),
            _cell("S01_G04", "action_progress"),
            _cell("S01_G05", "end"),
        ],
    )

    contracts = [cell["pose_contract"] for cell in cells]
    assert [
        [binding["unit_id"] for binding in contract["action_bindings"]]
        for contract in contracts
    ] == [["GAU001"], ["GAU002"], ["GAU002"], ["GAU002"], ["GAU002"]]
    assert contracts[0]["timing_role"] == "initial_anchor"
    assert contracts[0]["story_time_weight"] == 0
    assert contracts[0]["pose_progress"] == 1.0
    assert all(contract["timing_role"] == "story_action" for contract in contracts[1:])
    assert all(contract["story_time_weight"] == 1 for contract in contracts[1:])
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


@pytest.mark.parametrize(
    "actions",
    [
        ("actor-alpha takes a ready guard stance",),
        ("actor-alpha holds a staff", "actor-alpha strikes right"),
        ("light remains fixed", "actor-alpha strikes right"),
    ],
)
def test_non_eligible_initial_state_remains_timed(actions):
    units = tuple(
        _unit(f"GAU{index:03d}", action, ledger_index=index - 1)
        for index, action in enumerate(actions, 1)
    )
    cells = pose_owner.compile_pose_contracts(
        _beat(*units),
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )

    assert all(
        cell["pose_contract"]["timing_role"] == "story_action" for cell in cells
    )
    assert all(cell["pose_contract"]["story_time_weight"] == 1 for cell in cells)


def test_p02_ready_is_not_zero_time_without_a_cinematic_first_frame():
    beat = _beat(
        _unit("GAU001", "actor-alpha takes a ready guard stance", ledger_index=0),
        _unit("GAU002", "actor-alpha dodges right", ledger_index=1),
    )
    beat["beat_id"] = "S01_P02"
    cells = [_cell("S01_G06", "start"), _cell("S01_G07", "end")]
    for cell in cells:
        cell["secondary_beat_id"] = "S01_P02"

    compiled = pose_owner.compile_pose_contracts(beat, cells)

    assert all(
        cell["pose_contract"]["timing_role"] == "story_action"
        for cell in compiled
    )


def test_initial_anchor_weight_tamper_fails_closed_after_rehash():
    cells = pose_owner.compile_pose_contracts(
        _beat(
            _unit("GAU001", "actor-alpha takes a ready stance", ledger_index=0),
            _unit("GAU002", "actor-alpha runs right", ledger_index=1),
        ),
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )
    contract = cells[0]["pose_contract"]
    contract["story_time_weight"] = 1.0
    fingerprint_payload = {
        "family": contract["pose_family"],
        "stage": contract["stage"],
        "pose_progress": contract["pose_progress"],
        "direction": contract["direction"],
        "mechanics_modifiers": contract["mechanics_modifiers"],
        "transition_origin": contract["transition_origin"],
        "actors": contract["geometry"]["actors"],
        "objects": contract["geometry"]["objects"],
        "action_vector": contract["action_vector"],
        "camera_vector": contract["camera_vector"],
        "static_spatial_state": contract["static_spatial_state"],
        "timing_role": contract["timing_role"],
        "story_time_weight": contract["story_time_weight"],
    }
    contract["pose_fingerprint"] = pose_owner._canonical_sha256(fingerprint_payload)
    unhashed = dict(contract)
    unhashed.pop("contract_sha256")
    contract["contract_sha256"] = pose_owner._canonical_sha256(unhashed)

    with pytest.raises(ValueError, match="initial pose anchor is invalid"):
        pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_single_action_can_progress_across_all_nine_cells_without_false_block():
    unit = _unit("GAU001", "actor-alpha pivots and swings a staff right")
    cells = pose_owner.compile_pose_contracts(
        _beat(unit),
        [
            _cell(f"S01_G{index:02d}", "action_progress")
            for index in range(1, 10)
        ],
    )

    assert [cell["pose_contract"]["pose_progress"] for cell in cells] == [
        0.0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
    ]
    assert len(set(pose_owner.pose_fingerprints(cells))) == 9
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_unclassified_actor_action_gets_generic_weight_shift_not_arrow_only_motion():
    cells = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha makes an unfamiliar transition")),
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )

    assert [cell["pose_contract"]["pose_family"] for cell in cells] == [
        "spatial",
        "spatial",
    ]
    roots = [
        tuple(cell["pose_contract"]["geometry"]["actors"][0]["root_translation"])
        for cell in cells
    ]
    assert roots[0] != roots[1]
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_next_action_starts_from_previous_canonical_end_pose_without_neutral_reset():
    ready = _unit("GAU001", "actor-alpha takes a ready stance", ledger_index=0)
    evade = _unit("GAU002", "actor-alpha dodges right", ledger_index=1)
    cells = pose_owner.compile_pose_contracts(
        _beat(ready, evade),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "end"),
            _cell("S01_G03", "start"),
            _cell("S01_G04", "end"),
        ],
    )

    previous_end = cells[0]["pose_contract"]
    next_start = cells[1]["pose_contract"]
    assert next_start["transition_origin"] == {
        "source": "previous_canonical_action",
        "unit_ids": ["GAU001"],
        "pose_family": "ready",
        "direction": "right",
    }
    assert previous_end["geometry"]["actors"][0]["joints"] == (
        next_start["geometry"]["actors"][0]["joints"]
    )
    assert previous_end["geometry"]["actors"][0]["root_translation"] == (
        next_start["geometry"]["actors"][0]["root_translation"]
    )
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_new_actor_does_not_inherit_previous_actors_pose_or_root_motion():
    first = _unit(
        "GAU001",
        "actor-alpha runs right",
        performers=("actor-alpha",),
        ledger_index=0,
    )
    second = _unit(
        "GAU002",
        "actor-beta blocks right",
        performers=("actor-beta",),
        ledger_index=1,
    )
    cells = pose_owner.compile_pose_contracts(
        _beat(first, second),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "end"),
            _cell("S01_G03", "start"),
            _cell("S01_G04", "end"),
        ],
    )

    assert cells[1]["pose_contract"]["geometry"]["actors"][0]["root_translation"] == [
        110,
        0,
    ]
    second_start = cells[2]["pose_contract"]["geometry"]["actors"][0]
    assert second_start["role_ref"] == "actor-beta"
    assert second_start["root_origin"] == [0, 0]
    assert second_start["root_translation"] == [0, 0]
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_body_mechanics_modify_torso_weight_and_foot_geometry():
    unit = _unit("GAU001", "actor-alpha dodges right")
    plain = pose_owner.compile_pose_contracts(_beat(unit), [_cell()])[0]["pose_contract"]
    detailed_beat = _beat(unit)
    detailed_beat["body_action_contract"] = {
        "beats": [
            {
                "micro_action_index": 1,
                "performer": "actor-alpha",
                "technique": "向右滑步后倾闪避",
                "footwork": "右脚向右侧滑步，左脚支撑并维持稳定",
                "torso": "躯干降低重心并向后仰倾",
                "weight_shift": "重心下沉，并向右、向后转移",
                "contact": "没有格挡或击中接触",
                "end_pose": "低重心、躯干后倾的闪避姿态",
            }
        ]
    }
    detailed = pose_owner.compile_pose_contracts(detailed_beat, [_cell()])[0]["pose_contract"]

    assert detailed["pose_family"] == "evade"
    assert detailed["mechanics_modifiers"]["center_drop"] > 0
    assert detailed["mechanics_modifiers"]["torso_lean"] < 0
    assert detailed["mechanics_modifiers"]["stance_width"] > 0
    plain_joints = plain["geometry"]["actors"][0]["joints"]
    detailed_joints = detailed["geometry"]["actors"][0]["joints"]
    assert detailed_joints["neck"][1] > plain_joints["neck"][1]
    assert detailed_joints["head"][0] < plain_joints["head"][0]
    assert (
        detailed_joints["right_foot"][0] - detailed_joints["left_foot"][0]
        > plain_joints["right_foot"][0] - plain_joints["left_foot"][0]
    )


def test_body_mechanics_do_not_drift_to_an_unmatched_action_unit():
    ready = _unit("GAU001", "actor-alpha takes a ready stance", ledger_index=0)
    ready["source_micro_action_indexes"] = [1]
    evade = _unit("GAU002", "actor-alpha dodges right", ledger_index=1)
    evade["source_micro_action_indexes"] = [2]
    beat = _beat(ready, evade)
    beat["body_action_contract"] = {
        "beats": [
            {
                "micro_action_index": 2,
                "performer": "actor-alpha",
                "technique": "right side-step evade",
                "footwork": "right foot side-steps",
                "torso": "lower and lean back",
                "weight_shift": "right and backward",
                "contact": "without blocking",
                "end_pose": "low evade pose",
            }
        ]
    }

    cells = pose_owner.compile_pose_contracts(
        beat,
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )

    assert [cell["pose_contract"]["pose_family"] for cell in cells] == [
        "ready",
        "evade",
    ]
    assert cells[0]["pose_contract"]["matched_body_action_beats"] == []
    assert cells[1]["pose_contract"]["matched_body_action_beats"] == [
        {
            "micro_action_index": 2,
            "body_action_sha256": cells[1]["pose_contract"]["matched_body_action_beats"][0][
                "body_action_sha256"
            ],
        }
    ]


@pytest.mark.parametrize(
    ("action", "expected_family"),
    [
        ("actor-alpha 双手握住道具并进入战斗戒备", "ready"),
        ("actor-alpha 走进训练区域", "locomotion"),
        ("actor-alpha 上步斜挥短棍", "strike"),
        ("actor-alpha plants the rear foot and swings a staff", "strike"),
    ],
)
def test_specific_action_semantics_beat_ambiguous_motion_words(action, expected_family):
    contract = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", action)),
        [_cell()],
    )[0]["pose_contract"]
    assert contract["pose_family"] == expected_family


def test_action_units_partition_in_source_order_without_loss():
    units = (
        _unit("GAU001", "actor-alpha runs forward", ledger_index=0),
        _unit("GAU002", "actor-alpha blocks right", ledger_index=1),
        _unit("GAU003", "actor-alpha strikes right", ledger_index=2),
    )
    cells = pose_owner.compile_pose_contracts(
        _beat(*units),
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )
    ordered_ids = [
        binding["unit_id"]
        for cell in cells
        for binding in cell["pose_contract"]["action_bindings"]
    ]
    assert ordered_ids == ["GAU001", "GAU002", "GAU003"]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("actor-alpha dodges left", "evade"),
        ("actor-alpha blocks forward", "block"),
        ("actor-alpha strikes right", "strike"),
        ("actor-alpha kicks left", "kick"),
        ("actor-alpha grabs guard-1", "grab_control"),
        ("actor-alpha throws guard-1 right", "throw"),
        ("actor-alpha uses handheld device", "prop_use"),
        ("actor-alpha makes an unfamiliar transition", "spatial"),
    ],
)
def test_multilingual_fallback_is_controlled_and_auditable(action, expected):
    targets = ("guard-1",) if "guard" in action else ()
    contract = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", action, targets=targets)), [_cell()]
    )[0]["pose_contract"]
    assert contract["pose_family"] == expected
    assert contract["classification_evidence"]
    assert contract["pose_policy_sha256"] == pose_owner.POSE_POLICY_SHA256


def test_multi_subject_interaction_uses_distinct_role_slots():
    cell = pose_owner.compile_pose_contracts(
        _beat(
            _unit(
                "GAU001",
                "actor-alpha strikes guard-1 right",
                targets=("guard-1",),
            )
        ),
        [_cell()],
    )[0]
    contract = cell["pose_contract"]
    assert contract["actor_roles"] == ["actor-alpha", "guard-1"]
    assert [actor["slot"] for actor in contract["geometry"]["actors"]] == [1, 2]
    assert contract["geometry"]["actors"][0]["pose_family"] == "strike"
    assert contract["geometry"]["actors"][1]["pose_family"] == "evade"
    assert all(
        len(actor["pose_fingerprint"]) == 64
        for actor in contract["geometry"]["actors"]
    )
    assert (
        contract["geometry"]["actors"][0]["pose_fingerprint"]
        != contract["geometry"]["actors"][1]["pose_fingerprint"]
    )


def test_environment_only_motion_uses_object_glyph_without_fabricated_actor():
    cell = pose_owner.compile_pose_contracts(
        _beat(
            _unit(
                "GAU001",
                "automatic door moves left",
                performers=("automatic door",),
            ),
            character_ids=("actor-not-performing",),
        ),
        [_cell()],
    )[0]
    contract = cell["pose_contract"]
    assert contract["actor_roles"] == []
    assert contract["geometry"]["actors"] == []
    assert contract["object_roles"] == ["automatic door"]
    assert contract["geometry"]["objects"]


def test_action_and_camera_arrows_follow_resolved_vectors(monkeypatch):
    rendered_arrows = []
    original = pose_owner._draw_arrow

    def record(draw, start, end, *, fill, width):
        rendered_arrows.append((start, end, fill))
        return original(draw, start, end, fill=fill, width=width)

    monkeypatch.setattr(pose_owner, "_draw_arrow", record)
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha runs left")), [_cell()]
    )[0]
    pose_owner.render_pose_cell(
        cell,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    action_arrow = next(item for item in rendered_arrows if item[2] == (205, 48, 54))
    camera_arrow = next(item for item in rendered_arrows if item[2] == (42, 104, 190))
    assert action_arrow[1][0] < action_arrow[0][0]
    assert camera_arrow[1][0] > camera_arrow[0][0]


def test_identical_inputs_render_identical_bytes():
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha blocks right")), [_cell()]
    )[0]

    def png_sha256() -> str:
        image = pose_owner.render_pose_cell(
            copy.deepcopy(cell),
            font_factory=lambda _size: ImageFont.load_default(),
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    assert png_sha256() == png_sha256()


def test_tampered_geometry_and_missing_lineage_fail_closed():
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha blocks right")), [_cell()]
    )[0]
    tampered = copy.deepcopy(cell)
    tampered["pose_contract"]["geometry"]["actors"][0]["joints"]["head"] = [0, 0]
    with pytest.raises(ValueError, match="contract hash mismatch"):
        pose_owner.validate_pose_sequence([tampered], beat_id="S01_P01")

    no_lineage = _unit("GAU001", "actor-alpha blocks right")
    no_lineage.pop("ledger_indexes")
    no_lineage.pop("source_event_id")
    no_lineage.pop("source_generation_unit_indexes")
    with pytest.raises(ValueError, match="no source action/event lineage"):
        pose_owner.compile_pose_contracts(_beat(no_lineage), [_cell()])


@pytest.mark.parametrize(
    "planner_version",
    ["honcut.secondary-storyboard.v16", None],
)
def test_storyboard_beat_without_canonical_action_units_fails_closed(planner_version):
    beat = {
        "beat_id": "S01_P01",
        "action": "actor-alpha moves",
    }
    if planner_version is not None:
        beat["planner_version"] = planner_version
    with pytest.raises(ValueError, match="missing canonical generation action units"):
        pose_owner.compile_pose_contracts(beat, [_cell()])


def test_pose_owner_has_no_test_only_lineage_fallback():
    source = inspect.getsource(pose_owner)
    assert "TEST-COMPAT" not in source
    assert "unversioned_test_compatibility" not in source


def test_pose_owner_has_no_provider_phase3_or_pipeline_core_dependency():
    source = inspect.getsource(pose_owner)
    assert "clients." not in source
    assert "phase3" not in source.casefold()
    assert "pipeline_core" not in source


def test_phase6_task_fingerprint_binds_pose_fingerprints_and_zero_time_anchor(tmp_path):
    guide_fields = {
        "storyboard_beat_id": "S01_P01",
        "storyboard_narrative_guide": "storyboard_guides/S01_P01.png",
        "storyboard_narrative_guide_kind": "honcut.storyboard-narrative-guide.v4",
        "storyboard_narrative_guide_usage": ("phase6_story_narrative_guide_not_output_pixels"),
        "storyboard_narrative_guide_cell_ids": ["S01_G01", "S01_G02"],
        "storyboard_narrative_guide_zero_time_anchor_cell_ids": ["S01_G01"],
        "storyboard_narrative_guide_sha256": "a" * 64,
        "storyboard_narrative_guide_renderer": ("honcut.identity-neutral-story-guide-renderer.v2"),
        "storyboard_narrative_guide_pose_contract_schema": (
            "honcut.storyboard-guide-pose-contract.v3"
        ),
        "storyboard_narrative_guide_pose_policy_sha256": "b" * 64,
        "storyboard_narrative_guide_pose_contracts_sha256": "c" * 64,
        "storyboard_narrative_guide_pose_fingerprints": ["d" * 64, "e" * 64],
        "storyboard_narrative_guide_source_pixel_usage": "none",
        "storyboard_narrative_guide_semantic_payload_sha256": "f" * 64,
        "storyboard_narrative_guide_source_board": "shot_storyboards/S01.png",
        "storyboard_narrative_guide_source_board_sha256": "1" * 64,
        "storyboard_narrative_guide_receipt": "storyboard_guides/S01_P01.json",
        "storyboard_narrative_guide_authority_roles": [
            "narrative_order",
            "action_direction",
            "camera_motion",
            "spatial_relationship",
        ],
        "storyboard_narrative_guide_non_authority_roles": ["character_identity"],
    }

    def payload(fingerprints: list[str], anchor_cells: list[str]) -> dict:
        chunk = GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=5,
            mode="fresh",
            execution_strategy="multi_image",
            action_prompt="actor-alpha moves",
            **{
                **guide_fields,
                "storyboard_narrative_guide_pose_fingerprints": fingerprints,
                "storyboard_narrative_guide_zero_time_anchor_cell_ids": anchor_cells,
            },
        )
        request = ChunkExecutionRequest(
            resource_id="S01_C01",
            shot_id="S01",
            chunk=chunk,
            anchors={},
            output_path=tmp_path / "S01_C01.mp4",
            previous_output_path=None,
            input_fingerprint="2" * 64,
            memory_context="",
        )
        return _task_payload(
            request,
            model="test-model",
            provider_id="test-provider",
            provider_version="1",
            project_id="project",
            run_id="run",
            duration=5,
            seed=7,
        )

    first = payload(["d" * 64, "e" * 64], ["S01_G01"])
    second = payload(["d" * 64, "9" * 64], ["S01_G01"])
    no_anchor = payload(["d" * 64, "e" * 64], [])
    assert first["storyboard_narrative_guide_pose_fingerprints"] == [
        "d" * 64,
        "e" * 64,
    ]
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert first["storyboard_narrative_guide_zero_time_anchor_cell_ids"] == [
        "S01_G01"
    ]
    assert first["input_fingerprint"] != no_anchor["input_fingerprint"]


def test_continuity_chunk_rejects_incomplete_pose_fingerprint_binding():
    with pytest.raises(ValueError, match="pose fingerprints must bind every"):
        GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=5,
            mode="fresh",
            execution_strategy="multi_image",
            storyboard_beat_id="S01_P01",
            storyboard_narrative_guide="storyboard_guides/S01_P01.png",
            storyboard_narrative_guide_kind="honcut.storyboard-narrative-guide.v4",
            storyboard_narrative_guide_usage=("phase6_story_narrative_guide_not_output_pixels"),
            storyboard_narrative_guide_cell_ids=["S01_G01", "S01_G02"],
            storyboard_narrative_guide_sha256="a" * 64,
            storyboard_narrative_guide_renderer=("honcut.identity-neutral-story-guide-renderer.v2"),
            storyboard_narrative_guide_pose_contract_schema=(
                "honcut.storyboard-guide-pose-contract.v3"
            ),
            storyboard_narrative_guide_pose_policy_sha256="b" * 64,
            storyboard_narrative_guide_pose_contracts_sha256="c" * 64,
            storyboard_narrative_guide_pose_fingerprints=["d" * 64],
            storyboard_narrative_guide_source_pixel_usage="none",
            storyboard_narrative_guide_semantic_payload_sha256="e" * 64,
            storyboard_narrative_guide_source_board="shot_storyboards/S01.png",
            storyboard_narrative_guide_source_board_sha256="f" * 64,
            storyboard_narrative_guide_receipt="storyboard_guides/S01_P01.json",
            storyboard_narrative_guide_authority_roles=[
                "narrative_order",
                "action_direction",
                "camera_motion",
                "spatial_relationship",
            ],
            storyboard_narrative_guide_non_authority_roles=["character_identity"],
        )
