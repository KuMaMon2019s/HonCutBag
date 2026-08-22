from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from runtime.execution_errors import SubmissionUncertainError
from runtime.generation_tasks import (
    GENERATION_TASK_SCHEMA_VERSION,
    GenerationTaskStore,
)
from runtime.provider_policy import ProviderExecutionPolicy
from runtime.video_provider import (
    ProviderErrorKind,
    VideoGenerationRequest,
    VideoJobState,
    VideoJobStatus,
    VideoProvider,
    VideoProviderCapabilities,
    VideoSubmission,
    classify_provider_error,
)


class FakeProvider:
    provider_id = "fake"
    capabilities = VideoProviderCapabilities(
        provider_id="fake",
        generation_modes=frozenset({"txt2vid", "img2vid"}),
        supports_cancel=True,
    )

    def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        assert request.resource_id == "S01"
        return VideoSubmission("job-1", "https://provider.example/jobs")

    def status(self, provider_job_id: str) -> VideoJobStatus:
        return VideoJobStatus(
            VideoJobState.SUCCEEDED,
            output_uri=f"https://provider.example/{provider_job_id}.mp4",
        )

    def cancel(self, provider_job_id: str) -> VideoJobStatus:
        return VideoJobStatus(VideoJobState.CANCELLED)


def test_request_is_canonical_and_detached_from_mutable_input():
    payload = {"prompt": "雨夜", "params": {"duration": 5}}
    fingerprint = hashlib.sha256(b"semantic-input").hexdigest()
    request = VideoGenerationRequest.from_payload(
        resource_id="S01",
        input_fingerprint=fingerprint,
        payload=payload,
    )
    payload["params"]["duration"] = 99

    assert request.payload == {"params": {"duration": 5}, "prompt": "雨夜"}
    assert request.payload_json == '{"params":{"duration":5},"prompt":"雨夜"}'


def test_provider_protocol_normalizes_submit_status_and_cancel():
    provider = FakeProvider()
    assert isinstance(provider, VideoProvider)
    request = VideoGenerationRequest.from_payload(
        resource_id="S01",
        input_fingerprint="0" * 64,
        payload={"prompt": "test"},
    )

    submission = provider.submit(request)
    status = provider.status(submission.provider_job_id)
    cancelled = provider.cancel(submission.provider_job_id)

    assert provider.capabilities.supports_cancel is True
    assert status.state is VideoJobState.SUCCEEDED and status.state.terminal
    assert cancelled.state is VideoJobState.CANCELLED and cancelled.state.terminal


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SubmissionUncertainError("accepted, id lost"), ProviderErrorKind.SUBMISSION_UNCERTAIN),
        (RuntimeError("Seedance API 429: too many requests"), ProviderErrorKind.RATE_LIMIT),
        (RuntimeError("QuotaExceeded"), ProviderErrorKind.QUOTA),
        (RuntimeError("PrivacyInformation"), ProviderErrorKind.MODERATION),
        (TimeoutError("poll timed out"), ProviderErrorKind.TIMEOUT),
        (RuntimeError("HTTP 503 temporarily unavailable"), ProviderErrorKind.TRANSIENT),
    ],
)
def test_provider_error_classification(error, expected):
    assert classify_provider_error(error) is expected


def test_status_contract_fails_closed_for_missing_terminal_details():
    with pytest.raises(ValueError, match="output_uri"):
        VideoJobStatus(VideoJobState.SUCCEEDED)
    with pytest.raises(ValueError, match="error_message"):
        VideoJobStatus(VideoJobState.FAILED)


def test_generation_task_store_migrates_legacy_rows_without_loss(tmp_path):
    database = tmp_path / "runtime.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE generation_tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                media_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                outcome_json TEXT,
                error_message TEXT,
                provider_id TEXT,
                provider_job_id TEXT,
                provider_endpoint TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                queued_at TEXT NOT NULL,
                started_at TEXT,
                updated_at TEXT NOT NULL,
                finished_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO generation_tasks (
                task_id, run_id, task_type, media_type, resource_id, status,
                payload_json, outcome_json, provider_id, provider_job_id,
                provider_endpoint, queued_at, updated_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-task",
                "run-1",
                "video.generate",
                "video",
                "S01",
                "succeeded",
                json.dumps({"input_fingerprint": "legacy-fingerprint", "prompt": "x"}),
                json.dumps({"output_artifact_id": "artifact-1"}),
                "seedance",
                "job-1",
                "https://provider.example/jobs",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:01:00+00:00",
            ),
        )

    store = GenerationTaskStore(database)
    migrated = store.get("legacy-task")
    assert migrated is not None
    assert migrated.payload["prompt"] == "x"
    assert migrated.input_fingerprint == "legacy-fingerprint"
    assert migrated.output_artifact_id == "artifact-1"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == (
            GENERATION_TASK_SCHEMA_VERSION
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(generation_tasks)")
        }
    assert {"input_fingerprint", "output_artifact_id"} <= columns


def test_generation_task_store_persists_fingerprint_and_output_artifact(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    enqueued = store.enqueue(
        run_id="run-1",
        task_type="video.generate",
        media_type="video",
        resource_id="S01",
        provider_id="seedance",
        payload={"prompt": "canonical"},
    )
    assert len(enqueued.task.input_fingerprint) == 64
    claimed = store.claim(enqueued.task.task_id)
    assert claimed is not None
    store.persist_provider_job(
        claimed.task_id,
        provider_job_id="job-1",
        provider_endpoint="https://provider.example/jobs",
    )
    succeeded = store.mark_succeeded(
        claimed.task_id,
        {"output_artifact_id": "artifact-1", "output_path": "S01.mp4"},
    )
    assert succeeded.output_artifact_id == "artifact-1"


def test_generation_task_store_rejects_unknown_future_schema(tmp_path):
    database = tmp_path / "runtime.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"PRAGMA user_version={GENERATION_TASK_SCHEMA_VERSION + 1}"
        )
    with pytest.raises(RuntimeError, match="newer than this runtime"):
        GenerationTaskStore(database)


def test_runtime_policy_owns_rate_limit_backoff_and_call_count():
    policy = ProviderExecutionPolicy(
        provider_id="seedance",
        max_rate_limit_retries=2,
        initial_backoff_seconds=3,
        max_backoff_seconds=5,
    )
    attempts = 0
    waits = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Seedance API 429: too many requests")
        return "job-1"

    assert policy.execute_rate_limited(operation, sleeper=waits.append) == "job-1"
    assert attempts == 3
    assert waits == [3, 5]


def test_runtime_policy_never_retries_uncertain_submission():
    policy = ProviderExecutionPolicy(provider_id="seedance")
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise SubmissionUncertainError("provider may have accepted request")

    with pytest.raises(SubmissionUncertainError):
        policy.execute_rate_limited(operation, sleeper=lambda _seconds: None)
    assert attempts == 1


def test_runtime_policy_owns_seedance_capacity(monkeypatch):
    monkeypatch.setenv("HONCUT_SEEDANCE_VIDEO_CONCURRENCY", "2")
    policy = ProviderExecutionPolicy(provider_id="seedance")
    assert policy.capacity(5) == 2
    assert ProviderExecutionPolicy(provider_id="bridge").capacity(5) == 5


def test_runtime_policy_binds_poll_deadline_and_request_timeout():
    calls = []

    def poll(
        task_id,
        *,
        api_key,
        max_attempts,
        interval,
        request_timeout,
    ):
        calls.append(
            (task_id, api_key, max_attempts, interval, request_timeout)
        )
        return "https://video.test/output.mp4"

    policy = ProviderExecutionPolicy(
        provider_id="seedance",
        status_timeout_seconds=7,
        poll_deadline_seconds=90,
    )
    bound = policy.bind_poll(poll, interval_seconds=15, api_key="secret")

    assert bound("job-1") == "https://video.test/output.mp4"
    assert calls == [("job-1", "secret", 6, 15, 7)]
