from __future__ import annotations

import sqlite3

import pytest

from runtime.execution_errors import SubmissionUncertainError
from runtime.generation_tasks import GenerationTaskStore
from runtime.seedance_execution import execute_seedance_video_task


def _enqueue(store: GenerationTaskStore):
    return store.enqueue(
        run_id="run-1",
        task_type="video.generate",
        media_type="video",
        resource_id="S01_P01",
        payload={"input_fingerprint": "a" * 64},
        provider_id="seedance",
    ).task


def test_submission_attempt_and_uncertain_transition_are_atomic(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    task = _enqueue(store)
    claimed = store.claim(task.task_id)
    assert claimed is not None
    reserved = store.reserve_submission_attempt(
        task.task_id,
        provider_endpoint="https://provider.invalid/tasks",
    )
    assert reserved.status == "submission_uncertain"
    assert store.submission_attempt_count(run_id="run-1") == 1
    assert [event.event_type for event in store.events(task.task_id)] == [
        "TaskQueued",
        "TaskClaimed",
        "SubmissionAttempted",
    ]
    with pytest.raises(RuntimeError, match="cannot reserve"):
        store.reserve_submission_attempt(
            task.task_id,
            provider_endpoint="https://provider.invalid/tasks",
        )
    assert store.submission_attempt_count(run_id="run-1") == 1


def test_provider_acceptance_resumes_running_without_second_attempt(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    task = _enqueue(store)
    store.claim(task.task_id)
    store.reserve_submission_attempt(
        task.task_id,
        provider_endpoint="https://provider.invalid/tasks",
    )
    accepted = store.confirm_provider_job(
        task.task_id,
        provider_job_id="job-1",
        provider_endpoint="https://provider.invalid/tasks",
    )
    assert accepted.status == "running"
    assert accepted.provider_job_id == "job-1"
    assert store.submission_attempt_count(task_id=task.task_id) == 1
    assert store.events(task.task_id)[-1].event_type == "ProviderAccepted"


def test_direct_execution_never_retries_uncertain_submission(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    submits = 0

    def uncertain_submit() -> str:
        nonlocal submits
        submits += 1
        raise TimeoutError("connection lost after request write")

    arguments = dict(
        run_id="run-1",
        resource_id="S01_P01",
        payload={"input_fingerprint": "a" * 64},
        provider_endpoint="https://provider.invalid/tasks",
        output_path=tmp_path / "video.mp4",
        submit=uncertain_submit,
        poll=lambda _job: "https://provider.invalid/video.mp4",
        download=lambda _url, _path: _path,
    )
    with pytest.raises(TimeoutError):
        execute_seedance_video_task(store, **arguments)
    assert submits == 1
    assert store.submission_attempt_count(run_id="run-1") == 1
    with pytest.raises(SubmissionUncertainError):
        execute_seedance_video_task(store, **arguments)
    assert submits == 1
    assert store.submission_attempt_count(run_id="run-1") == 1


def test_v2_database_imports_snapshot_without_fabricating_transitions(tmp_path):
    database = tmp_path / "runtime.db"
    legacy = GenerationTaskStore(database)
    task = _enqueue(legacy)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER generation_task_events_no_delete")
        connection.execute("DELETE FROM generation_task_events")
        connection.execute("DROP TABLE generation_task_events")
        connection.execute("PRAGMA user_version=2")
    migrated = GenerationTaskStore(database)
    events = migrated.events(task.task_id)
    assert [event.event_type for event in events] == ["LegacySnapshotImported"]
    assert migrated.submission_attempt_count(run_id="run-1") == 0
