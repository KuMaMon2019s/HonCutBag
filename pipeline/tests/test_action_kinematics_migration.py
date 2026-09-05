from __future__ import annotations

import copy
import json

import pytest

from runtime.action_kinematics_migration import (
    migrate_action_kinematics_artifact,
    migrate_parent_action_kinematics,
)
from runtime.artifact_manifest import ArtifactManifestStore


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


def test_artifact_migration_registers_immutable_sidecar_and_receipt(tmp_path) -> None:
    (tmp_path / "CANONICAL_VISUAL_CONTRACT.json").write_text(
        json.dumps({"contract_sha256": "d" * 64}),
        encoding="utf-8",
    )
    parent_path = tmp_path / "legacy_action.json"
    parent_path.write_text(json.dumps(_legacy_parent()), encoding="utf-8")
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    parent = store.register_file(
        parent_path,
        artifact_type="legacy_action_contract",
        producer_node="phase1.action_contract",
        authority_roles=("story_action",),
    )
    downstream_path = tmp_path / "legacy_pose.json"
    downstream_path.write_text("{}", encoding="utf-8")
    downstream = store.register_file(
        downstream_path,
        artifact_type="legacy_pose_contract",
        producer_node="phase2.storyboard_pose",
        parent_artifact_ids=(parent.artifact_id,),
        non_authority_roles=("story_action",),
    )
    parent_before = parent_path.read_bytes()

    first = migrate_action_kinematics_artifact(
        store,
        parent_artifact_id=parent.artifact_id,
        downstream_artifact_ids=(downstream.artifact_id,),
    )
    second = migrate_action_kinematics_artifact(
        store,
        parent_artifact_id=parent.artifact_id,
        downstream_artifact_ids=(downstream.artifact_id,),
    )

    assert parent_path.read_bytes() == parent_before
    assert first == second
    assert first["receipt"]["status"] == "migrated_sidecar"
    assert first["provider_request_count"] == 0
    sidecar = store.resolve(first["sidecar_artifact_id"])
    receipt = store.resolve(first["receipt_artifact_id"])
    assert sidecar.parent_artifact_ids == (parent.artifact_id,)
    assert receipt.parent_artifact_ids == (parent.artifact_id, sidecar.artifact_id)
    assert sidecar.authority_roles == ("story_action",)


def test_artifact_migration_rejects_unrelated_downstream_lineage(tmp_path) -> None:
    (tmp_path / "CANONICAL_VISUAL_CONTRACT.json").write_text(
        json.dumps({"contract_sha256": "e" * 64}),
        encoding="utf-8",
    )
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    parent_path = tmp_path / "legacy_action.json"
    parent_path.write_text(json.dumps(_legacy_parent()), encoding="utf-8")
    parent = store.register_file(
        parent_path,
        artifact_type="legacy_action_contract",
        producer_node="phase1.action_contract",
        authority_roles=("story_action",),
    )
    unrelated_path = tmp_path / "unrelated.json"
    unrelated_path.write_text("{}", encoding="utf-8")
    unrelated = store.register_file(
        unrelated_path,
        artifact_type="unrelated",
        producer_node="test",
    )

    with pytest.raises(RuntimeError, match="does not descend"):
        migrate_action_kinematics_artifact(
            store,
            parent_artifact_id=parent.artifact_id,
            downstream_artifact_ids=(unrelated.artifact_id,),
        )
