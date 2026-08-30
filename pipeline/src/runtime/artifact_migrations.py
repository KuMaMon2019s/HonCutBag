"""Version-by-version migrations for artifact manifests."""

from __future__ import annotations

from typing import Any

from runtime.migration_registry import apply_migration_registry
from schemas.artifact import ARTIFACT_MANIFEST_SCHEMA_VERSION, ARTIFACT_SCHEMA_VERSION


def _manifest_v0_to_v1(document: dict[str, Any]) -> dict[str, Any]:
    artifacts = document.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest v0 artifacts must be a list")
    migrated = []
    aliases = {
        "id": "artifact_id",
        "run": "run_id",
        "project": "project_id",
        "kind": "type",
        "sha256": "content_sha256",
        "path": "relative_path",
        "producer": "producer_node",
        "task_id": "producer_task_id",
        "parents": "parent_artifact_ids",
    }
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, dict):
            raise ValueError("artifact manifest v0 entries must be objects")
        artifact = dict(raw_artifact)
        for old_name, new_name in aliases.items():
            if old_name in artifact and new_name not in artifact:
                artifact[new_name] = artifact.pop(old_name)
        artifact["schema_version"] = 1
        artifact.setdefault("parent_artifact_ids", [])
        artifact.setdefault("semantic_fingerprint", None)
        migrated.append(artifact)
    return {**document, "schema_version": 1, "artifacts": migrated}


def _manifest_v1_to_v2(document: dict[str, Any]) -> dict[str, Any]:
    artifacts = document.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest v1 artifacts must be a list")
    known_roles = {
        "video": (["current_visual_state", "continuity_anchor"], []),
        "character_registry_receipt": ([], []),
    }
    migrated = []
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, dict):
            raise ValueError("artifact manifest v1 entries must be objects")
        artifact = dict(raw_artifact)
        artifact_type = str(artifact.get("type") or "")
        roles = known_roles.get(artifact_type)
        if roles is None:
            authority_roles: list[str] = []
            non_authority_roles: list[str] = []
            migration_status = "audit_only_unknown_v1"
        else:
            authority_roles, non_authority_roles = roles
            migration_status = "mapped_known_v1"
        artifact.update({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "canonical_contract_sha256": None,
            "prompt_sha256": None,
            "authority_roles": authority_roles,
            "non_authority_roles": non_authority_roles,
            "migration_status": migration_status,
        })
        migrated.append(artifact)
    return {**document, "schema_version": 2, "artifacts": migrated}


ARTIFACT_MANIFEST_MIGRATIONS = {
    0: _manifest_v0_to_v1,
    1: _manifest_v1_to_v2,
}


def migrate_artifact_manifest(document: dict[str, Any]) -> dict[str, Any]:
    return apply_migration_registry(
        document,
        current_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        migrations=ARTIFACT_MANIFEST_MIGRATIONS,
        document_name="artifact manifest",
    )


__all__ = ["ARTIFACT_MANIFEST_MIGRATIONS", "migrate_artifact_manifest"]
