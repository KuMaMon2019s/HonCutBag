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
        artifact["schema_version"] = ARTIFACT_SCHEMA_VERSION
        artifact.setdefault("parent_artifact_ids", [])
        artifact.setdefault("semantic_fingerprint", None)
        migrated.append(artifact)
    return {**document, "schema_version": 1, "artifacts": migrated}


ARTIFACT_MANIFEST_MIGRATIONS = {0: _manifest_v0_to_v1}


def migrate_artifact_manifest(document: dict[str, Any]) -> dict[str, Any]:
    return apply_migration_registry(
        document,
        current_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        migrations=ARTIFACT_MANIFEST_MIGRATIONS,
        document_name="artifact manifest",
    )


__all__ = ["ARTIFACT_MANIFEST_MIGRATIONS", "migrate_artifact_manifest"]
