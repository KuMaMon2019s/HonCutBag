"""Persistent arq queue for per-shot storyboard generation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from arq import Retry, create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from arq.worker import func


REDIS_URL = os.environ.get("HONCUT_REDIS_URL", "redis://127.0.0.1:6380")
QUEUE_NAME = "honcut:storyboard_shots"
JOB_TIMEOUT = 600
MAX_TRIES = 3
RETRY_BACKOFF = True
CHECKPOINT_KIND = "storyboard_shot_queue_v1"


def redis_settings() -> RedisSettings:
    """Build settings at call time so tests and operators can override the URL."""
    return RedisSettings.from_dsn(
        os.environ.get("HONCUT_REDIS_URL", "redis://127.0.0.1:6380")
    )


def shot_concurrency() -> int:
    raw = os.environ.get("HONCUT_SHOT_CONCURRENCY", "5")
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def serialize_payload(payload: dict[str, Any]) -> str:
    """Validate and serialize a job payload without Python-only objects."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def deserialize_payload(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("shot payload must be a JSON object")
    return payload


def make_payload(
    shot: dict[str, Any],
    index: int,
    total: int,
    *,
    characters: list[dict[str, Any]] | None,
    visual_style_text: str | None,
    scene_style_map: dict[str, str],
    previous_shot: dict[str, Any] | None,
    visual_style_path: str | None,
) -> dict[str, Any]:
    payload = {
        "shot": shot,
        "index": index,
        "total": total,
        "characters": characters or [],
        "visual_style_text": visual_style_text,
        "scene_style_map": scene_style_map,
        "previous_shot": previous_shot,
        "visual_style_path": visual_style_path,
    }
    # Fail at the producer rather than after a job reaches a worker.
    serialize_payload(payload)
    return payload


async def generate_shot_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Generate one storyboard shot, retrying failures with exponential deferral."""
    try:
        from pipeline.src.phases.phase2.storyboard_generator import _generate_single_shot

        return await asyncio.to_thread(_generate_single_shot, **payload)
    except Exception as exc:
        job_try = int(ctx.get("job_try", 1))
        if job_try < MAX_TRIES and RETRY_BACKOFF:
            raise Retry(defer=2 ** (job_try - 1)) from exc
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    """Mission 12 atomic checkpoint pattern: temp file followed by os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_completed(path: Path, run_tag: str) -> dict[int, dict[str, Any]]:
    """Load valid results for this run; ignore unrelated/old checkpoint shapes."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("kind") != CHECKPOINT_KIND or value.get("run_tag") != run_tag:
            return {}
        entries = value.get("shots", [])
        return {
            int(entry["index"]): entry["result"]
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("index"), int)
            and isinstance(entry.get("result"), dict)
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return {}


def missing_payloads(
    payloads: Iterable[dict[str, Any]], completed: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Crash recovery: enqueue only shot indexes absent from the checkpoint."""
    return [payload for payload in payloads if int(payload["index"]) not in completed]


def sorted_results(completed: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [completed[index] for index in sorted(completed)]


def write_completed(path: Path, run_tag: str, completed: dict[int, dict[str, Any]]) -> None:
    _atomic_write_json(
        path,
        {
            "kind": CHECKPOINT_KIND,
            "run_tag": run_tag,
            "shots": [
                {"index": index, "result": completed[index]}
                for index in sorted(completed)
            ],
        },
    )


async def enqueue_and_collect(
    payloads: list[dict[str, Any]],
    *,
    run_tag: str,
    partial_path: Path,
    pool_factory: Callable[..., Awaitable[Any]] = create_pool,
) -> list[dict[str, Any]]:
    """Enqueue missing jobs, await results, and checkpoint each completed shot."""
    completed = load_completed(partial_path, run_tag)
    pending = missing_payloads(payloads, completed)
    if not pending:
        return sorted_results(completed)

    redis = await pool_factory(redis_settings())
    try:
        jobs: list[tuple[int, Job]] = []
        for payload in pending:
            index = int(payload["index"])
            job = await redis.enqueue_job(
                "generate_shot_job",
                payload,
                _job_id=f"{run_tag}:shot:{index}",
                _queue_name=QUEUE_NAME,
            )
            # A duplicate id may already be queued/running after producer restart.
            jobs.append((index, job or Job(f"{run_tag}:shot:{index}", redis=redis)))

        async def collect(index: int, job: Job) -> None:
            result = await job.result(timeout=JOB_TIMEOUT * MAX_TRIES + 60)
            if not isinstance(result, dict):
                raise TypeError(f"shot {index} returned a non-object result")
            completed[index] = result
            write_completed(partial_path, run_tag, completed)

        await asyncio.gather(*(collect(index, job) for index, job in jobs))
        return sorted_results(completed)
    finally:
        await redis.aclose()


def run_shot_queue(
    payloads: list[dict[str, Any]], *, run_tag: str, partial_path: Path
) -> list[dict[str, Any]]:
    return asyncio.run(
        enqueue_and_collect(payloads, run_tag=run_tag, partial_path=partial_path)
    )


class WorkerSettings:
    functions = [func(generate_shot_job, timeout=JOB_TIMEOUT, max_tries=MAX_TRIES)]
    redis_settings = redis_settings()
    queue_name = QUEUE_NAME
    max_jobs = shot_concurrency()
    job_timeout = JOB_TIMEOUT
    max_tries = MAX_TRIES
    # arq retries are deferred exponentially by generate_shot_job.
    retry_backoff = True
    keep_result = 3600
