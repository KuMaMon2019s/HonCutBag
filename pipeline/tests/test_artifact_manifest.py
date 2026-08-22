from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.artifact_manifest import (
    ARTIFACT_MANIFEST_FILENAME,
    ArtifactManifestStore,
)
from runtime.generation_tasks import GenerationTaskStore
from runtime.seedance_execution import execute_seedance_video_task
from schemas.artifact import ArtifactRef


def test_artifact_ref_is_strict_and_rejects_escaping_paths():
    fields = {
        "artifact_id": "artifact-1",
        "run_id": "run-1",
        "project_id": "project-1",
        "type": "video",
        "content_sha256": "a" * 64,
        "relative_path": "shots/S01/output.mp4",
        "producer_node": "phase6.video_generation",
        "created_at": datetime.now(UTC),
    }
    artifact = ArtifactRef(**fields)
    assert artifact.artifact_type == "video"
    with pytest.raises(ValidationError, match="relative"):
        ArtifactRef(**{**fields, "relative_path": "../outside.mp4"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        ArtifactRef(**{**fields, "secret": "must-not-be-stored"})


def test_manifest_registers_lineage_atomically_and_resolves_content(tmp_path):
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    source = tmp_path / "story.json"
    source.write_text("{}", encoding="utf-8")
    story = store.register_file(
        source,
        artifact_type="story",
        producer_node="phase1.story",
    )
    video = tmp_path / "shots/S01/output.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    output = store.register_file(
        video,
        artifact_type="video",
        producer_node="phase6.video_generation",
        producer_task_id="task-1",
        parent_artifact_ids=[story.artifact_id],
    )

    manifest = store.load()
    assert [item.artifact_id for item in manifest.artifacts] == [
        story.artifact_id,
        output.artifact_id,
    ]
    assert store.resolve(output.artifact_id) == output
    persisted = json.loads(
        (tmp_path / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted["artifacts"][1]["type"] == "video"
    assert not list(tmp_path.glob(f".{ARTIFACT_MANIFEST_FILENAME}.*.tmp"))


def test_manifest_fails_closed_for_parent_hash_and_project_mismatch(tmp_path):
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    source = tmp_path / "artifact.txt"
    source.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing parents"):
        store.register_file(
            source,
            artifact_type="text",
            producer_node="phase1.story",
            parent_artifact_ids=["artifact-missing"],
        )
    artifact = store.register_file(
        source,
        artifact_type="text",
        producer_node="phase1.story",
    )
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        store.resolve(artifact.artifact_id)
    with pytest.raises(RuntimeError, match="different project or run"):
        ArtifactManifestStore(
            tmp_path,
            run_id="run-1",
            project_id="project-2",
        ).load()


def test_failed_atomic_replace_preserves_previous_manifest(tmp_path, monkeypatch):
    store = ArtifactManifestStore(tmp_path, run_id="run-1", project_id="project-1")
    first = tmp_path / "first.txt"
    first.write_text("first", encoding="utf-8")
    store.register_file(first, artifact_type="text", producer_node="phase1.story")
    before = store.path.read_bytes()
    second = tmp_path / "second.txt"
    second.write_text("second", encoding="utf-8")

    monkeypatch.setattr(
        "runtime.artifact_manifest.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated crash")),
    )
    with pytest.raises(OSError, match="simulated crash"):
        store.register_file(
            second,
            artifact_type="text",
            producer_node="phase1.story",
        )

    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob(f".{ARTIFACT_MANIFEST_FILENAME}.*.tmp"))


def test_seedance_success_binds_task_to_registered_output(tmp_path):
    artifacts = ArtifactManifestStore(
        tmp_path,
        run_id="run-1",
        project_id="project-1",
    )
    tasks = GenerationTaskStore(tmp_path / "runtime.db")
    output = tmp_path / "shots/S01/output.mp4"

    def download(_url: str, destination: str) -> str:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"video")
        return destination

    execution = execute_seedance_video_task(
        tasks,
        run_id="run-1",
        resource_id="S01",
        payload={"prompt": "canonical"},
        provider_endpoint="https://provider.test/jobs",
        output_path=output,
        submit=lambda: "provider-job-1",
        poll=lambda _task_id: "https://video.test/output.mp4",
        download=download,
        artifact_store=artifacts,
    )

    task = tasks.get(execution.task_id)
    assert task is not None
    assert task.output_artifact_id is not None
    artifact = artifacts.resolve(task.output_artifact_id)
    assert artifact.producer_task_id == task.task_id
    assert artifact.relative_path == "shots/S01/output.mp4"
