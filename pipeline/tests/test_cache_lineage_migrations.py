from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from graph.migrations import STATE_MIGRATIONS, StateMigrationError, migrate_state
from runtime.artifact_manifest import ArtifactManifestStore
from runtime.artifact_migrations import ARTIFACT_MANIFEST_MIGRATIONS
from runtime.cache_lineage import build_cache_key, validate_cache_identity


def _cache_key(**overrides):
    values = {
        "project_id": "project-1",
        "run_id": "run-1",
        "input_lineage": ["artifact:a", "artifact:b"],
        "semantic_fingerprint": "a" * 64,
    }
    values.update(overrides)
    return build_cache_key(**values)


def test_cache_key_is_namespaced_by_project_run_lineage_and_semantics():
    baseline = _cache_key()
    assert baseline.value == _cache_key(input_lineage=reversed(["artifact:a", "artifact:b"])).value
    variants = [
        _cache_key(project_id="project-2"),
        _cache_key(run_id="run-2"),
        _cache_key(input_lineage=["artifact:a", "artifact:c"]),
        _cache_key(semantic_fingerprint="b" * 64),
    ]
    assert all(candidate.value != baseline.value for candidate in variants)
    validate_cache_identity(baseline.task_metadata(), baseline)


def test_cache_identity_fails_closed_instead_of_using_a_weaker_key():
    expected = _cache_key()
    with pytest.raises(RuntimeError, match="project/run lineage"):
        validate_cache_identity(
            {**expected.task_metadata(), "cache_key": "legacy-weak-key"},
            expected,
        )
    with pytest.raises(RuntimeError, match="material"):
        validate_cache_identity(
            {
                **expected.task_metadata(),
                "cache_identity": {
                    **expected.material,
                    "project_id": "different-project",
                },
            },
            expected,
        )


def test_state_migration_registry_upgrades_v0_and_rejects_future_versions():
    assert set(STATE_MIGRATIONS) == {0, 1}
    migrated = migrate_state(
        {
            "text": "story",
            "duration": 30,
            "shots": [{"id": "S01"}],
            "error": "legacy failure",
        }
    )
    assert migrated["state_schema_version"] == 2
    assert migrated["shot_policy"] == "cut-driven"
    assert migrated["input_text"] == "story"
    assert migrated["target_duration_s"] == 30
    assert migrated["shot_ids"] == ["S01"]
    assert "text" not in migrated
    with pytest.raises(StateMigrationError, match="newer than supported"):
        migrate_state({"state_schema_version": 3})


def test_artifact_migration_registry_upgrades_known_v0_manifest(tmp_path):
    assert set(ARTIFACT_MANIFEST_MIGRATIONS) == {0}
    artifact = tmp_path / "story.txt"
    artifact.write_text("story", encoding="utf-8")
    import hashlib

    content_hash = hashlib.sha256(b"story").hexdigest()
    legacy = {
        "run_id": "run-1",
        "project_id": "project-1",
        "artifacts": [
            {
                "id": "artifact-legacy",
                "run": "run-1",
                "project": "project-1",
                "kind": "text",
                "sha256": content_hash,
                "path": "story.txt",
                "producer": "phase1.story",
                "task_id": None,
                "parents": [],
                "created_at": datetime.now(UTC).isoformat(),
            }
        ],
    }
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")

    migrated = store.load()
    assert migrated.schema_version == 1
    assert migrated.artifacts[0].schema_version == 1
    assert store.resolve("artifact-legacy").relative_path == "story.txt"


def test_artifact_migration_registry_rejects_unknown_future_version(tmp_path):
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "project_id": "project-1",
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    with pytest.raises(RuntimeError, match="newer than supported"):
        store.load()
