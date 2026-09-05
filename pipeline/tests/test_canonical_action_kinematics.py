from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest
from PIL import ImageChops, ImageFont

from phases.phase2.storyboard_guide_pose import (
    compile_pose_contracts,
    render_pose_cell,
    validate_pose_contract,
)
from phases.phase2.storyboard_pose_atlas import build_pose_atlas_plan
from schemas.understanding import BodyActionUnderstanding
from utils.action_kinematics import (
    CHANNEL_ORDER,
    KINEMATICS_PROJECTION_SCHEMA,
    SOURCE_KINEMATICS_SCHEMA,
    apply_generation_kinematics_projection,
    compile_source_kinematics,
    sample_projection,
    validate_generation_kinematics_projection,
    validate_source_kinematics,
)
from utils import action_kinematics


def _beat(index: int, performer: str, *, side: str = "右侧") -> dict:
    left = "左" in side
    return {
        "beat": index,
        "micro_action_index": index,
        "micro_action": f"{performer}向前攻击",
        "performer": performer,
        "technique": "向前跨步直拳",
        "side": side,
        "limbs": [
            "左臂" if left else "右臂",
            "左手" if left else "右手",
            "右腿" if left else "左腿",
            "右脚" if left else "左脚",
            "腰",
            "头",
        ],
        "footwork": "右脚支撑，左脚向前跨步" if left else "左脚支撑，右脚向前跨步",
        "torso": "腰部向右旋转并前倾" if left else "腰部向左旋转并前倾",
        "weight_shift": "重心从右脚转移至左脚" if left else "重心从左脚转移至右脚",
        "direction": "向前",
        "contact": "左拳接触目标后收回" if left else "右拳接触目标后收回",
        "end_pose": "左脚在前，双膝弯曲，保持平衡"
        if left
        else "右脚在前，双膝弯曲，保持平衡",
    }


def _rehash(payload: dict, field: str) -> None:
    unhashed = copy.deepcopy(payload)
    unhashed.pop(field, None)
    payload[field] = hashlib.sha256(
        json.dumps(
            unhashed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _record() -> dict:
    first = _beat(1, "actor-alpha")
    second = _beat(2, "actor-beta", side="左侧")
    second.update(
        micro_action="actor-beta向后闪避并格挡",
        technique="后撤闪避后抬起左臂格挡",
        direction="向后",
        contact="左前臂承接攻击",
    )
    return {
        "beat_id": "S01_P01",
        "body_action_contract": {
            "schema": "honcut.body-action-choreography.v2",
            "beats": [
                {**first, "kinematics": compile_source_kinematics(first)},
                {**second, "kinematics": compile_source_kinematics(second)},
            ],
            "valid": True,
        },
        "generation_action_units": [
            {
                "unit_id": "GAU001",
                "source_action_unit_id": "AU001",
                "source_event_id": 1,
                "source_micro_action_indexes": [1, 2],
                "actions": [first["micro_action"], second["micro_action"]],
                "performers": ["actor-alpha", "actor-beta"],
                "targets": ["actor-beta", "actor-alpha"],
            }
        ],
    }


def _single_actor_record(beat: dict) -> dict:
    return {
        "beat_id": "S01_P01",
        "duration_s": 4,
        "planner_version": "honcut.secondary-storyboard.v17",
        "character_ids": [str(beat["performer"])],
        "body_action_contract": {
            "schema": "honcut.body-action-choreography.v2",
            "beats": [{**beat, "kinematics": compile_source_kinematics(beat)}],
            "valid": True,
        },
        "generation_action_units": [
            {
                "unit_id": "GAU001",
                "source_action_unit_id": "AU001",
                "source_event_id": 1,
                "source_micro_action_indexes": [int(beat["micro_action_index"])],
                "actions": [str(beat["micro_action"])],
                "performers": [str(beat["performer"])],
                "targets": [],
            }
        ],
    }


def _rendered_body(sample: dict):
    contract = sample["pose_contract"]
    rendered = render_pose_cell(
        {
            "label": contract["cell_id"],
            "secondary_beat_id": contract["secondary_beat_id"],
            "pose_contract": contract,
        },
        width=600,
        height=400,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    return rendered.crop((120, 55, 480, 390))


def test_source_kinematics_has_complete_numeric_bilateral_channels() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))

    assert payload["schema"] == SOURCE_KINEMATICS_SCHEMA
    assert validate_source_kinematics(payload) == payload
    track = payload["actor_tracks"][0]
    assert track["performer_id"] == "actor-alpha"
    assert all(tuple(phase["channels"]) == CHANNEL_ORDER for phase in track["phases"])
    assert all(
        isinstance(channel["translation_milli"], list)
        and isinstance(channel["rotation_mdeg"], list)
        for phase in track["phases"]
        for channel in phase["channels"].values()
    )
    apex = track["phases"][2]["channels"]
    assert abs(apex["root"]["translation_milli"][2]) >= 450
    assert abs(apex["waist_torso"]["rotation_mdeg"][1]) >= 25000
    assert apex["right_arm"]["amplitude"] == "large"
    assert apex["right_hand"]["translation_milli"] != apex["right_arm"]["translation_milli"]
    assert apex["left_leg"]["translation_milli"] != apex["left_foot"]["translation_milli"]


def test_left_and_right_source_actions_mirror_without_swapping_distal_channels() -> None:
    right = compile_source_kinematics(_beat(1, "actor", side="右侧"))
    left = compile_source_kinematics(_beat(1, "actor", side="左侧"))
    right_apex = right["actor_tracks"][0]["phases"][2]["channels"]
    left_apex = left["actor_tracks"][0]["phases"][2]["channels"]

    assert right_apex["right_arm"]["translation_milli"][0] == -left_apex["left_arm"]["translation_milli"][0]
    assert right_apex["right_hand"]["role"] == left_apex["left_hand"]["role"] == "active"


def test_projection_is_created_after_gau_and_keeps_multi_actor_tracks() -> None:
    record = _record()
    projection = apply_generation_kinematics_projection(record)

    assert projection["schema"] == KINEMATICS_PROJECTION_SCHEMA
    assert projection["beat_id"] == "S01_P01"
    assert projection["source_micro_action_indexes"] == [1, 2]
    assert [item["performer_id"] for item in projection["actor_tracks"]] == [
        "actor-alpha",
        "actor-beta",
    ]
    assert record["generation_action_units"][0]["kinematics_projection_sha256"] == record[
        "generation_action_units"
    ][0]["kinematics_projection"]["projection_sha256"]
    assert validate_generation_kinematics_projection(projection) == projection


def test_projection_rejects_overlapping_source_indexes() -> None:
    record = _record()
    record["generation_action_units"].append(
        {
            **copy.deepcopy(record["generation_action_units"][0]),
            "unit_id": "GAU002",
            "source_micro_action_indexes": [2],
        }
    )

    with pytest.raises(ValueError, match="overlap"):
        apply_generation_kinematics_projection(record)


def test_camera_projection_never_changes_relative_actor_yaw() -> None:
    record = _record()
    projection = apply_generation_kinematics_projection(record)
    front = sample_projection(projection, 0.55, camera_yaw_mdeg=0)
    back = sample_projection(projection, 0.55, camera_yaw_mdeg=180_000)

    assert front["actor_tracks"][0]["relative_yaw_mdeg"] == back["actor_tracks"][0][
        "relative_yaw_mdeg"
    ]
    assert front["actor_tracks"][0]["camera_relation"] != back["actor_tracks"][0][
        "camera_relation"
    ]


def test_source_schema_rejects_unknown_fields_and_hash_drift() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    payload["future"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_source_kinematics(payload)

    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    payload["actor_tracks"][0]["phases"][1]["end_milli"] = 999
    with pytest.raises(ValueError, match="hash|window"):
        validate_source_kinematics(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda phase: phase.__setitem__("source_micro_action_index", 99), "source index"),
        (lambda phase: phase.__setitem__("phase_id", ""), "phase ID"),
        (lambda phase: phase.__setitem__("relative_yaw_mdeg", "0"), "yaw"),
        (lambda phase: phase.__setitem__("camera_relation", "diagonal"), "camera relation"),
    ),
)
def test_source_schema_rejects_rehashed_internal_lineage_drift(mutation, message) -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    mutation(payload["actor_tracks"][0]["phases"][0])
    _rehash(payload, "kinematics_sha256")

    with pytest.raises(ValueError, match=message):
        validate_source_kinematics(payload)


def test_projection_rejects_duplicate_generation_unit_identity_after_rehash() -> None:
    record = _record()
    template = record["generation_action_units"][0]
    record["generation_action_units"] = [
        {**copy.deepcopy(template), "unit_id": "GAU001", "source_micro_action_indexes": [1]},
        {**copy.deepcopy(template), "unit_id": "GAU002", "source_micro_action_indexes": [2]},
    ]
    projection = apply_generation_kinematics_projection(record)
    projection["action_units"][1]["generation_action_unit_id"] = "GAU001"
    _rehash(projection, "projection_sha256")

    with pytest.raises(ValueError, match="unique"):
        validate_generation_kinematics_projection(projection)


def test_projection_rejects_phase_source_index_outside_projection_after_rehash() -> None:
    projection = apply_generation_kinematics_projection(_record())
    projection["actor_tracks"][0]["phases"][0]["source_micro_action_index"] = 99
    _rehash(projection, "projection_sha256")

    with pytest.raises(ValueError, match="source index"):
        validate_generation_kinematics_projection(projection)


def test_projection_rejects_source_kinematics_detached_from_body_evidence() -> None:
    record = _record()
    record["body_action_contract"]["beats"][0]["technique"] = "篡改后的动作"

    with pytest.raises(ValueError, match="evidence"):
        apply_generation_kinematics_projection(record)


def test_phase2_rejects_partial_kinematics_coverage_within_one_cell_group() -> None:
    record = _record()
    template = record["generation_action_units"][0]
    record["generation_action_units"] = [
        {**copy.deepcopy(template), "unit_id": "GAU001", "source_micro_action_indexes": [1]},
        {**copy.deepcopy(template), "unit_id": "GAU002", "source_micro_action_indexes": [2]},
    ]
    apply_generation_kinematics_projection(record)
    record["generation_action_units"][1].pop("kinematics_projection")
    record["generation_action_units"][1].pop("kinematics_projection_sha256")

    with pytest.raises(ValueError, match="coverage"):
        compile_pose_contracts(
            record,
            [
                {
                    "label": "S01_G01",
                    "primary_shot_id": "S01",
                    "secondary_beat_id": "S01_P01",
                    "stage": "action_progress",
                    "camera_movement": "static",
                }
            ],
            known_actor_roles=("actor-alpha", "actor-beta"),
        )


def test_phase2_rejects_pxx_child_projection_hash_drift() -> None:
    record = _record()
    record.update(
        duration_s=4,
        planner_version="honcut.secondary-storyboard.v17",
        character_ids=["actor-alpha", "actor-beta"],
    )
    apply_generation_kinematics_projection(record)
    record["kinematics_projection"]["action_units"][0]["projection_sha256"] = "f" * 64
    _rehash(record["kinematics_projection"], "projection_sha256")

    with pytest.raises(ValueError, match="child lineage"):
        build_pose_atlas_plan(
            record,
            known_actor_roles=("actor-alpha", "actor-beta"),
        )


def test_pose_contract_rejects_action_binding_projection_hash_drift() -> None:
    record = _single_actor_record(_beat(1, "actor-alpha"))
    apply_generation_kinematics_projection(record)
    contract = copy.deepcopy(
        build_pose_atlas_plan(
            record,
            known_actor_roles=("actor-alpha",),
        )["pose_samples"][0]["pose_contract"]
    )
    contract["action_bindings"][0]["kinematics_projection_sha256"] = "f" * 64
    _rehash(contract, "contract_sha256")

    with pytest.raises(ValueError, match="binding lineage"):
        validate_pose_contract(
            contract,
            cell_id=contract["cell_id"],
            beat_id=contract["secondary_beat_id"],
        )

def test_body_action_provider_dto_remains_unchanged() -> None:
    assert tuple(BodyActionUnderstanding.model_fields) == (
        "micro_action_index",
        "performer",
        "technique",
        "side",
        "limbs",
        "footwork",
        "torso",
        "weight_shift",
        "direction",
        "contact",
        "end_pose",
    )


def test_non_body_unit_receives_no_invented_kinematics() -> None:
    record = _record()
    record["generation_action_units"].insert(
        0,
        {
            "unit_id": "GAU000",
            "source_action_unit_id": "AU000",
            "source_event_id": 1,
            "source_micro_action_indexes": [99],
            "actions": ["雨水沿玻璃滑落"],
            "performers": [],
            "targets": [],
        },
    )

    projection = apply_generation_kinematics_projection(record)

    assert "kinematics_projection" not in record["generation_action_units"][0]
    assert projection["source_micro_action_indexes"] == [1, 2]


def test_waist_rotation_does_not_invent_a_whole_body_spin() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))

    assert {
        phase["transform"]["kind"]
        for phase in payload["actor_tracks"][0]["phases"]
    } == {"none"}


def test_explicit_flip_has_airborne_support_release_and_landing() -> None:
    beat = _beat(1, "actor-alpha")
    beat["micro_action"] = "actor-alpha前翻越过障碍后落地"
    beat["technique"] = "前翻"
    payload = compile_source_kinematics(beat)
    phases = payload["actor_tracks"][0]["phases"]

    assert [phase["transform"]["kind"] for phase in phases] == ["flip"] * 5
    assert [phase["transform"]["axis"] for phase in phases] == ["pitch"] * 5
    assert [phase["transform"]["airborne"] for phase in phases] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert [phase["transform"]["support_release"] for phase in phases] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert phases[-1]["transform"]["landing_state"] == "landed"


def test_transform_schema_rejects_illegal_enum_before_hash_check() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    payload["actor_tracks"][0]["phases"][0]["transform"]["kind"] = "teleport"

    with pytest.raises(ValueError, match="transform enum"):
        validate_source_kinematics(payload)


def test_transform_schema_rejects_incomplete_rotation_definition() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    transform = payload["actor_tracks"][0]["phases"][0]["transform"]
    transform.update(kind="turn", axis="yaw", direction="none")

    with pytest.raises(ValueError, match="definition is incomplete"):
        validate_source_kinematics(payload)


def test_contact_is_phase_local_not_a_sustained_hold() -> None:
    payload = compile_source_kinematics(_beat(1, "actor-alpha"))
    contacts = [
        any(channel["contact"] for channel in phase["channels"].values())
        for phase in payload["actor_tracks"][0]["phases"]
    ]

    assert contacts == [False, False, True, False, False]

    no_contact = _beat(1, "actor-alpha")
    no_contact["contact"] = "无目标接触；身体保持既有支撑接触"
    no_contact_payload = compile_source_kinematics(no_contact)
    assert no_contact_payload["actor_tracks"][0]["phases"][2]["phase_id"].endswith(
        "_apex"
    )
    assert not any(
        channel["contact"]
        for phase in no_contact_payload["actor_tracks"][0]["phases"]
        for channel in phase["channels"].values()
    )


def test_phase2_raster_uses_bilateral_distal_and_proximal_kinematics() -> None:
    right_record = _single_actor_record(_beat(1, "actor-alpha", side="右侧"))
    left_record = _single_actor_record(_beat(1, "actor-alpha", side="左侧"))
    apply_generation_kinematics_projection(right_record)
    apply_generation_kinematics_projection(left_record)
    right_plan = build_pose_atlas_plan(right_record, known_actor_roles=("actor-alpha",))
    left_plan = build_pose_atlas_plan(left_record, known_actor_roles=("actor-alpha",))
    right_sample = right_plan["pose_samples"][4]
    left_sample = left_plan["pose_samples"][4]
    right_actor = right_sample["pose_contract"]["geometry"]["actors"][0]
    left_actor = left_sample["pose_contract"]["geometry"]["actors"][0]

    assert right_actor["joints"]["right_hand"] != right_actor["joints"]["right_elbow"]
    assert right_actor["joints"]["left_foot"] != right_actor["joints"]["left_knee"]
    assert right_actor["joints"]["head"] != right_actor["joints"]["neck"]
    assert right_actor["joints"]["right_hand"] != left_actor["joints"]["right_hand"]
    difference = ImageChops.difference(
        _rendered_body(right_sample),
        _rendered_body(left_sample),
    )
    assert difference.getbbox() is not None


@pytest.mark.parametrize(
    ("move", "technique", "expected_kind"),
    (
        ("actor-alpha前翻后落地", "前翻", "flip"),
        ("actor-alpha旋转攻击", "旋转攻击", "spin"),
    ),
)
def test_phase2_raster_changes_for_flip_and_spin_sequences(
    move: str,
    technique: str,
    expected_kind: str,
) -> None:
    beat = _beat(1, "actor-alpha")
    beat.update(micro_action=move, technique=technique)
    record = _single_actor_record(beat)
    apply_generation_kinematics_projection(record)
    plan = build_pose_atlas_plan(record, known_actor_roles=("actor-alpha",))
    samples = plan["pose_samples"]
    kinds = {
        sample["pose_contract"]["kinematics_sample"]["actor_tracks"][0][
            "transform"
        ]["kind"]
        for sample in samples
    }
    relations = {
        sample["pose_contract"]["geometry"]["actors"][0]["facing"]
        for sample in samples
    }

    assert kinds == {expected_kind}
    if expected_kind == "spin":
        assert len(relations.intersection({"front", "left_profile", "right_profile", "back"})) >= 2
    difference = ImageChops.difference(
        _rendered_body(samples[0]),
        _rendered_body(samples[-1]),
    )
    assert difference.getbbox() is not None


def test_phase2_keeps_simultaneous_performers_as_separate_raster_skeletons() -> None:
    record = _record()
    record["duration_s"] = 4
    record["planner_version"] = "honcut.secondary-storyboard.v17"
    record["character_ids"] = ["actor-alpha", "actor-beta"]
    apply_generation_kinematics_projection(record)
    plan = build_pose_atlas_plan(
        record,
        known_actor_roles=("actor-alpha", "actor-beta"),
    )
    sample = plan["pose_samples"][4]
    actors = sample["pose_contract"]["geometry"]["actors"]

    assert [actor["role_ref"] for actor in actors] == ["actor-alpha", "actor-beta"]
    assert max(point[0] for point in actors[0]["joints"].values()) < min(
        point[0] for point in actors[1]["joints"].values()
    )
    assert _rendered_body(sample).getbbox() is not None


def test_kinematic_decomposition_preserves_action_count_and_recovers_stably() -> None:
    original = _record()
    expected_unit_ids = [unit["unit_id"] for unit in original["generation_action_units"]]
    fingerprints = []
    for _ in range(10):
        record = copy.deepcopy(original)
        projection = apply_generation_kinematics_projection(record)
        fingerprints.append(
            (
                projection["projection_sha256"],
                tuple(unit["unit_id"] for unit in record["generation_action_units"]),
                tuple(
                    unit.get("kinematics_projection_sha256", "")
                    for unit in record["generation_action_units"]
                ),
            )
        )

    assert len(set(fingerprints)) == 1
    assert list(fingerprints[0][1]) == expected_unit_ids


def test_kinematics_compiler_has_no_provider_or_model_dependency() -> None:
    source = inspect.getsource(action_kinematics)

    assert "schemas.understanding" not in source
    assert "providers." not in source
    assert "requests." not in source
    assert "ark_client" not in source
    assert "seedream" not in source


def test_adjacent_generation_units_inherit_terminal_channel_state() -> None:
    # Reuse one performer in both source beats to exercise direct terminal
    # inheritance rather than a neutral reset between moves.
    record = _record()
    source = record["body_action_contract"]["beats"][1]
    source["performer"] = "actor-alpha"
    source["kinematics"] = compile_source_kinematics(source)
    record["generation_action_units"] = [
        {**record["generation_action_units"][0], "unit_id": "GAU001", "source_micro_action_indexes": [1]},
        {**record["generation_action_units"][0], "unit_id": "GAU002", "source_micro_action_indexes": [2]},
    ]
    apply_generation_kinematics_projection(record)
    first_track = record["generation_action_units"][0]["kinematics_projection"]["actor_tracks"][0]
    second_track = record["generation_action_units"][1]["kinematics_projection"]["actor_tracks"][0]

    for name in CHANNEL_ORDER:
        terminal = first_track["phases"][-1]["channels"][name]
        initial = second_track["phases"][0]["channels"][name]
        assert terminal["translation_milli"] == initial["translation_milli"]
        assert terminal["rotation_mdeg"] == initial["rotation_mdeg"]
    assert not any(
        channel["contact"]
        for channel in second_track["phases"][0]["channels"].values()
    )
    assert first_track["phases"][-1]["relative_yaw_mdeg"] == second_track["phases"][0][
        "relative_yaw_mdeg"
    ]
