from __future__ import annotations

import copy

from runtime.action_kinematics_migration import migrate_parent_action_kinematics


def _legacy_parent() -> dict:
    beat = {
        "micro_action_index": 1,
        "micro_action": "actor向前踢击",
        "performer": "actor",
        "technique": "右腿前踢",
        "side": "右侧",
        "limbs": ["右腿", "右脚", "左腿", "左脚", "腰", "头"],
        "footwork": "左脚支撑",
        "torso": "腰部前倾",
        "weight_shift": "重心移至左脚",
        "direction": "向前",
        "contact": "右脚接触目标",
        "end_pose": "右脚落地保持平衡",
    }
    return {
        "shots": [
            {
                "storyboard_beats": [
                    {
                        "beat_id": "S01_P01",
                        "body_action_contract": {
                            "schema": "honcut.body-action-choreography.v1",
                            "required": True,
                            "valid": True,
                            "beats": [beat],
                        },
                        "generation_action_units": [
                            {
                                "unit_id": "GAU001",
                                "source_action_unit_id": "AU001",
                                "source_micro_action_indexes": [1],
                                "actions": [beat["micro_action"]],
                            }
                        ],
                    }
                ]
            }
        ]
    }


def test_evidence_complete_parent_migrates_to_immutable_sidecar() -> None:
    parent = _legacy_parent()
    original = copy.deepcopy(parent)

    receipt = migrate_parent_action_kinematics(
        parent,
        parent_artifact_id="artifact-parent",
        parent_content_sha256="a" * 64,
        downstream_artifact_ids=("artifact-pose", "artifact-atlas"),
    )

    assert parent == original
    assert receipt["status"] == "migrated_sidecar"
    assert receipt["provider_request_count"] == 0
    record = receipt["sidecar"]["records"][0]
    assert record["body_action_contract"]["schema"] == "honcut.body-action-choreography.v2"
    assert record["kinematics_projection"]["beat_id"] == "S01_P01"
    assert receipt["downstream_disposition"] == [
        {"artifact_id": "artifact-pose", "status": "stale", "usage": "audit_only"},
        {"artifact_id": "artifact-atlas", "status": "stale", "usage": "audit_only"},
    ]


def test_incomplete_or_future_parent_is_audit_only_without_guessing() -> None:
    incomplete = _legacy_parent()
    del incomplete["shots"][0]["storyboard_beats"][0]["body_action_contract"][
        "beats"
    ][0]["footwork"]
    future = _legacy_parent()
    future["shots"][0]["storyboard_beats"][0]["body_action_contract"][
        "schema"
    ] = "honcut.body-action-choreography.v99"

    incomplete_receipt = migrate_parent_action_kinematics(
        incomplete,
        parent_artifact_id="artifact-parent",
        parent_content_sha256="b" * 64,
    )
    future_receipt = migrate_parent_action_kinematics(
        future,
        parent_artifact_id="artifact-parent",
        parent_content_sha256="c" * 64,
    )

    assert incomplete_receipt["status"] == "audit_only"
    assert incomplete_receipt["sidecar"] is None
    assert future_receipt["status"] == "audit_only"
    assert future_receipt["reason"] == "future_body_action_schema"
    assert incomplete_receipt["provider_request_count"] == 0
    assert future_receipt["provider_request_count"] == 0
