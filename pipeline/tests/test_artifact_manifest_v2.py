from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from runtime.artifact_manifest import ArtifactManifestStore
from runtime.artifact_migrations import migrate_artifact_manifest
from schemas.artifact import ArtifactRef


def _legacy_artifact(artifact_type: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_id": "legacy-1",
        "run_id": "run-1",
        "project_id": "project-1",
        "type": artifact_type,
        "content_sha256": "a" * 64,
        "semantic_fingerprint": None,
        "relative_path": "legacy.bin",
        "producer_node": "legacy",
        "producer_task_id": None,
        "parent_artifact_ids": [],
        "created_at": datetime.now(UTC).isoformat(),
    }


def test_known_v1_video_receives_fixed_authority_mapping():
    migrated = migrate_artifact_manifest({
        "schema_version": 1,
        "run_id": "run-1",
        "project_id": "project-1",
        "artifacts": [_legacy_artifact("video")],
    })
    artifact = migrated["artifacts"][0]
    assert artifact["migration_status"] == "mapped_known_v1"
    assert artifact["authority_roles"] == [
        "current_visual_state",
        "continuity_anchor",
    ]


def test_unknown_v1_type_is_audit_only_and_cannot_supply_authority(tmp_path):
    asset = tmp_path / "legacy.bin"
    asset.write_bytes(b"legacy")
    raw = {
        "schema_version": 1,
        "run_id": "run-1",
        "project_id": "project-1",
        "artifacts": [{
            **_legacy_artifact("unclassified_board"),
            "content_sha256": __import__("hashlib").sha256(b"legacy").hexdigest(),
        }],
    }
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(json.dumps(raw), encoding="utf-8")
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    artifact = store.load().artifacts[0]
    assert artifact.migration_status == "audit_only_unknown_v1"
    with pytest.raises(RuntimeError, match="audit-only"):
        store.resolve(artifact.artifact_id, required_authority_roles=["story_action"])


def test_v2_rejects_overlapping_authority_roles():
    with pytest.raises(ValidationError, match="overlap"):
        ArtifactRef(
            artifact_id="artifact-1",
            run_id="run-1",
            project_id="project-1",
            type="guide",
            content_sha256="a" * 64,
            authority_roles=("story_action",),
            non_authority_roles=("story_action",),
            relative_path="guide.png",
            producer_node="phase2",
            created_at=datetime.now(UTC),
        )


def test_registration_binds_contract_prompt_roles_and_lineage(tmp_path):
    contract = {
        "schema_version": "honcut.canonical-visual-contract.v1",
        "contract_sha256": "b" * 64,
    }
    (tmp_path / "CANONICAL_VISUAL_CONTRACT.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    parent_path = tmp_path / "parent.png"
    parent_path.write_bytes(b"parent")
    child_path = tmp_path / "child.png"
    child_path.write_bytes(b"child")
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    parent = store.register_file(
        parent_path,
        artifact_type="identity_board",
        producer_node="phase3",
        authority_roles=["character_identity", "hair_geometry"],
        prompt_sha256="c" * 64,
    )
    child = store.register_file(
        child_path,
        artifact_type="first_frame",
        producer_node="phase4",
        parent_artifact_ids=[parent.artifact_id],
        authority_roles=["current_visual_state"],
        non_authority_roles=["story_action"],
        prompt_sha256="d" * 64,
    )
    assert child.canonical_contract_sha256 == "b" * 64
    assert child.parent_artifact_ids == (parent.artifact_id,)
    store.resolve(child.artifact_id, required_authority_roles=["current_visual_state"])
    with pytest.raises(RuntimeError, match="missing required authority"):
        store.resolve(child.artifact_id, required_authority_roles=["prop_geometry"])


def test_authority_resolution_rejects_contract_and_parent_hash_drift(tmp_path):
    contract = {
        "schema": "honcut.canonical-visual-contract.v1",
        "contract_sha256": "b" * 64,
    }
    (tmp_path / "CANONICAL_VISUAL_CONTRACT.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    parent_path = tmp_path / "parent.bin"
    parent_path.write_bytes(b"parent")
    child_path = tmp_path / "child.bin"
    child_path.write_bytes(b"child")
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    parent = store.register_file(
        parent_path,
        artifact_type="identity_board",
        producer_node="phase3",
        authority_roles=["character_identity"],
    )
    child = store.register_file(
        child_path,
        artifact_type="first_frame",
        producer_node="phase4",
        parent_artifact_ids=[parent.artifact_id],
        authority_roles=["current_visual_state"],
    )

    parent_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        store.resolve(
            child.artifact_id,
            required_authority_roles=["current_visual_state"],
        )

    parent_path.write_bytes(b"parent")
    contract["contract_sha256"] = "c" * 64
    (tmp_path / "CANONICAL_VISUAL_CONTRACT.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="canonical contract"):
        store.resolve(
            child.artifact_id,
            required_authority_roles=["current_visual_state"],
        )
