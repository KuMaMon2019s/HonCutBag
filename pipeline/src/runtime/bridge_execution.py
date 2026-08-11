"""Crash-safe execution of one local Bridge video task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.execution_errors import (
    ProviderEndpointChangedError,
    SubmissionUncertainError,
)
from runtime.generation_tasks import GenerationTaskStore


@dataclass(frozen=True)
class BridgeExecution:
    task_id: str
    provider_job_id: str
    output_path: str
    generation_result: str | dict[str, Any]
    resumed: bool


def execute_bridge_video_task(
    task_store: GenerationTaskStore,
    *,
    run_id: str,
    resource_id: str,
    payload: dict[str, Any],
    provider_endpoint: str,
    output_path: str | Path,
    generate: Callable[..., str | dict[str, Any]],
) -> BridgeExecution:
    """Persist a Bridge job ID before polling and resume it after interruption."""

    enqueued = task_store.enqueue(
        run_id=run_id,
        task_type="video.generate",
        media_type="video",
        resource_id=resource_id,
        payload=payload,
        provider_id="bridge",
    )
    task = enqueued.task
    resumed = bool(task.provider_job_id)

    if task.status == "submission_uncertain":
        raise SubmissionUncertainError(
            f"Bridge submission for {resource_id} is uncertain; refusing to resubmit"
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
            "process stopped after claim and before Bridge job ID persistence",
        )
        raise SubmissionUncertainError(
            f"Bridge submission for {resource_id} may be in flight; refusing to resubmit"
        )

    provider_job_id = task.provider_job_id
    if provider_job_id and task.provider_endpoint != provider_endpoint:
        message = (
            f"Bridge endpoint changed for {resource_id}: "
            f"stored={task.provider_endpoint!r}, current={provider_endpoint!r}"
        )
        task_store.note_resumable_error(task.task_id, message)
        raise ProviderEndpointChangedError(message)

    latest_provider_job_id = provider_job_id
    submission_without_job_id = False

    def remember_submission_start() -> None:
        nonlocal submission_without_job_id
        submission_without_job_id = True

    def remember_submission(submitted_job_id: str) -> None:
        nonlocal latest_provider_job_id, submission_without_job_id
        task_store.persist_provider_job(
            task.task_id,
            provider_job_id=submitted_job_id,
            provider_endpoint=provider_endpoint,
        )
        latest_provider_job_id = submitted_job_id
        submission_without_job_id = False

    try:
        generation_result = generate(
            resume_task_id=provider_job_id,
            on_submit_start=remember_submission_start,
            on_submitted=remember_submission,
        )
    except Exception as error:
        if submission_without_job_id:
            task_store.mark_submission_uncertain(task.task_id, str(error))
        elif latest_provider_job_id:
            task_store.note_resumable_error(task.task_id, str(error))
        elif isinstance(error, (ValueError, FileNotFoundError)):
            task_store.mark_failed(task.task_id, str(error))
        else:
            task_store.mark_submission_uncertain(task.task_id, str(error))
        raise

    if not latest_provider_job_id:
        message = "Bridge generation completed without reporting its provider job ID"
        task_store.mark_submission_uncertain(task.task_id, message)
        raise SubmissionUncertainError(message)

    destination = str(output_path)
    outcome: dict[str, Any] = {
        "provider_job_id": latest_provider_job_id,
        "output_path": destination,
    }
    if isinstance(generation_result, dict):
        outcome.update(
            {
                key: generation_result[key]
                for key in ("actual_model", "last_frame_path")
                if key in generation_result
            }
        )
    task_store.mark_succeeded(task.task_id, outcome)
    return BridgeExecution(
        task_id=task.task_id,
        provider_job_id=latest_provider_job_id,
        output_path=destination,
        generation_result=generation_result,
        resumed=resumed,
    )
