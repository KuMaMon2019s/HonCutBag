"""Crash-safe execution of one direct Seedance video task."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.execution_errors import (
    ProviderEndpointChangedError,
    ProviderPreparationError,
    SubmissionUncertainError,
)
from runtime.generation_tasks import GenerationTaskStore


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
        if not _matches_succeeded_output(
            destination, succeeded.outcome.get("output_sha256")
        ):
            download(str(video_url), str(destination))
        if not _matches_succeeded_output(
            destination, succeeded.outcome.get("output_sha256")
        ):
            raise RuntimeError(f"recovered Seedance output is invalid for {resource_id}")
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
        if _provider_rejected_submission(error):
            task_store.mark_failed(task.task_id, str(error))
        else:
            task_store.note_resumable_error(task.task_id, str(error))
        raise

    task_store.mark_succeeded(
        task.task_id,
        {
            "provider_job_id": provider_job_id,
            "video_url": video_url,
            "output_path": destination_text,
            "output_sha256": _file_hash(destination),
        },
    )
    return SeedanceExecution(
        task_id=task.task_id,
        provider_job_id=provider_job_id,
        video_url=video_url,
        output_path=destination_text,
        resumed=resumed,
    )
