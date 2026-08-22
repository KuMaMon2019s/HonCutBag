"""Crash-safe execution of one direct Seedance video task."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.artifact_manifest import ArtifactManifestStore
from runtime.execution_errors import (
    ProviderEndpointChangedError,
    ProviderJobFailedError,
    ProviderPreparationError,
    SubmissionUncertainError,
)
from runtime.generation_tasks import GenerationTaskStore
from runtime.security_boundaries import CorrelationContext, emit_runtime_event


@dataclass(frozen=True)
class SeedanceExecution:
    task_id: str
    provider_job_id: str
    video_url: str
    output_path: str
    resumed: bool


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_succeeded_output(path: Path, expected_hash: Any) -> bool:
    return bool(
        path.is_file()
        and path.stat().st_size > 0
        and (not expected_hash or _file_hash(path) == expected_hash)
    )


def _provider_rejected_submission(error: Exception) -> bool:
    if isinstance(error, ProviderPreparationError):
        return True
    message = str(error)
    rejection_markers = (
        "Seedance API 400",
        "InvalidParameter",
        "401",
        "403",
        "429",
        "QuotaExceeded",
        "PolicyViolation",
        "PrivacyInformation",
    )
    return any(marker in message for marker in rejection_markers)


def execute_seedance_video_task(
    task_store: GenerationTaskStore,
    *,
    run_id: str,
    resource_id: str,
    payload: dict[str, Any],
    provider_endpoint: str,
    output_path: str | Path,
    submit: Callable[[], str],
    poll: Callable[[str], str],
    download: Callable[[str, str], str],
    validate_output: Callable[[Path], bool] | None = None,
    artifact_store: ArtifactManifestStore | None = None,
) -> SeedanceExecution:
    """Submit once, persist the job id, then resume polling after failures."""
    destination = Path(output_path)
    succeeded = task_store.find_succeeded(
        run_id=run_id,
        task_type="video.generate",
        resource_id=resource_id,
        payload=payload,
        provider_id="seedance",
    )
    if succeeded is not None:
        provider_job_id = succeeded.provider_job_id
        video_url = succeeded.outcome.get("video_url")
        if not provider_job_id or not video_url:
            raise RuntimeError(
                f"successful Seedance ledger entry for {resource_id} is missing recovery data"
            )
        output_matches = _matches_succeeded_output(
            destination, succeeded.outcome.get("output_sha256")
        ) and (validate_output is None or validate_output(destination))
        if not output_matches:
            download(str(video_url), str(destination))
        if not _matches_succeeded_output(
            destination, succeeded.outcome.get("output_sha256")
        ) or (
            validate_output is not None and not validate_output(destination)
        ):
            raise RuntimeError(f"recovered Seedance output is invalid for {resource_id}")
        if artifact_store is not None:
            artifact_id = succeeded.output_artifact_id
            if artifact_id:
                artifact_store.resolve(artifact_id)
            else:
                artifact = artifact_store.register_file(
                    destination,
                    artifact_type="video",
                    producer_node="phase6.video_generation",
                    producer_task_id=succeeded.task_id,
                    semantic_fingerprint=succeeded.input_fingerprint,
                )
                task_store.mark_succeeded(
                    succeeded.task_id,
                    {**succeeded.outcome, "output_artifact_id": artifact.artifact_id},
                )
        return SeedanceExecution(
            task_id=succeeded.task_id,
            provider_job_id=provider_job_id,
            video_url=str(video_url),
            output_path=str(destination),
            resumed=True,
        )

    enqueued = task_store.enqueue(
        run_id=run_id,
        task_type="video.generate",
        media_type="video",
        resource_id=resource_id,
        payload=payload,
        provider_id="seedance",
    )
    task = enqueued.task
    resumed = bool(task.provider_job_id)

    if task.status == "submission_uncertain":
        raise SubmissionUncertainError(
            f"Seedance submission for {resource_id} is uncertain; refusing to resubmit"
        )

    if task.status == "queued":
        claimed = task_store.claim(task.task_id)
        if claimed is None:
            current = task_store.get(task.task_id)
            if current is None or not current.provider_job_id:
                raise RuntimeError(f"generation task {task.task_id} was claimed elsewhere")
            task = current
            resumed = True
        else:
            task = claimed
    elif enqueued.deduped and task.status == "running" and not task.provider_job_id:
        task_store.mark_submission_uncertain(
            task.task_id,
            "process stopped after claim and before provider job id persistence",
        )
        raise SubmissionUncertainError(
            f"Seedance submission for {resource_id} may be in flight; refusing to resubmit"
        )

    emit_runtime_event(
        "provider_task_active",
        CorrelationContext(
            project_id=(artifact_store.project_id if artifact_store else "local"),
            run_id=run_id,
            node_id="phase6.video_generation",
            task_id=task.task_id,
        ),
        provider_id="seedance",
        status=task.status,
        resource_id=resource_id,
    )

    provider_job_id = task.provider_job_id
    if provider_job_id and task.provider_endpoint != provider_endpoint:
        message = (
            f"Seedance endpoint changed for {resource_id}: "
            f"stored={task.provider_endpoint!r}, current={provider_endpoint!r}"
        )
        task_store.note_resumable_error(task.task_id, message)
        raise ProviderEndpointChangedError(message)
    if not provider_job_id:
        try:
            provider_job_id = submit()
        except Exception as error:
            if _provider_rejected_submission(error):
                task_store.mark_failed(task.task_id, str(error))
            else:
                task_store.mark_submission_uncertain(task.task_id, str(error))
            raise
        task = task_store.persist_provider_job(
            task.task_id,
            provider_job_id=provider_job_id,
            provider_endpoint=provider_endpoint,
        )

    destination_text = str(destination)
    try:
        video_url = poll(provider_job_id)
        download(video_url, destination_text)
    except Exception as error:
        if isinstance(error, ProviderJobFailedError):
            task_store.mark_failed(
                task.task_id,
                str(error),
                provider_terminal=True,
            )
        else:
            task_store.note_resumable_error(task.task_id, str(error))
        raise

    if not destination.is_file() or destination.stat().st_size <= 0 or (
        validate_output is not None and not validate_output(destination)
    ):
        message = f"downloaded Seedance output is not a valid video for {resource_id}"
        task_store.note_resumable_error(task.task_id, message)
        raise RuntimeError(message)

    outcome = {
        "provider_job_id": provider_job_id,
        "video_url": video_url,
        "output_path": destination_text,
        "output_sha256": _file_hash(destination),
    }
    if artifact_store is not None:
        artifact = artifact_store.register_file(
            destination,
            artifact_type="video",
            producer_node="phase6.video_generation",
            producer_task_id=task.task_id,
            expected_sha256=outcome["output_sha256"],
            semantic_fingerprint=task.input_fingerprint,
        )
        outcome["output_artifact_id"] = artifact.artifact_id
    task_store.mark_succeeded(task.task_id, outcome)
    return SeedanceExecution(
        task_id=task.task_id,
        provider_job_id=provider_job_id,
        video_url=video_url,
        output_path=destination_text,
        resumed=resumed,
    )
