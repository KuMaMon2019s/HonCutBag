#!/usr/bin/env python3
"""Run the Future Station fixture through offline real-media Phase 6-9 acceptance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
FIXTURE_PATH = REPO_ROOT / "pipeline" / "tests" / "fixtures" / "future_station_cron.txt"
FIXTURE_SHA256 = "5c578e0ad4a3d81efd245abbbd09f4460e3daa6f853acfaf90ba1b26ccd91d3f"
RECEIPT_SCHEMA = "honcut.phase6-9-offline-acceptance.v1"
RECEIPT_NAME = "phase6_9_offline_acceptance.json"
OFFLINE_PROVIDER_ID = "offline_fixture"
OFFLINE_PROVIDER_ENDPOINT = "offline://ffmpeg-lavfi"
MEDIA_PROFILE = "480p"
TARGET_DURATION_S = 60

for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


class OfflineProviderRequestError(RuntimeError):
    """Raised if an acceptance run reaches a network/provider boundary."""


@dataclass
class OfflineExecutionStats:
    generated_chunks: int = 0
    reused_chunks: int = 0
    provider_requests: int = 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


@contextlib.contextmanager
def _temporary_environment(values: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _deny_provider_requests(stats: OfflineExecutionStats) -> Iterator[None]:
    from clients import local_video_client, seedance_client, tos_uploader
    from clients.ark_multimodal_client import ArkMultimodalClient

    def blocked(*_args, **_kwargs):
        stats.provider_requests += 1
        raise OfflineProviderRequestError(
            "offline Phase 6-9 acceptance attempted a Provider/network request"
        )

    with contextlib.ExitStack() as stack:
        for owner, name in (
            (seedance_client, "submit_content"),
            (seedance_client, "submit_video_extension"),
            (seedance_client, "submit"),
            (local_video_client, "submit"),
            (tos_uploader, "upload_image"),
            (tos_uploader, "upload_media_file"),
            (ArkMultimodalClient, "review"),
        ):
            stack.enter_context(mock.patch.object(owner, name, side_effect=blocked))
        yield


def _offline_payload(request: Any) -> dict[str, Any]:
    return {
        "schema": "honcut.offline-video-fixture-request.v1",
        "test_only": True,
        "resource_id": request.resource_id,
        "shot_id": request.shot_id,
        "chunk": request.chunk.model_dump(mode="json"),
        "input_fingerprint": request.input_fingerprint,
        "relative_output": request.output_path.as_posix(),
        "media": {
            "width": 854,
            "height": 480,
            "fps": 30,
            "video_codec": "libx264",
            "audio_codec": "aac",
        },
    }


def _valid_offline_output(path: Path) -> bool:
    from utils.video_validation import is_valid_video

    return path.is_file() and path.stat().st_size > 0 and is_valid_video(path)


def _generate_offline_video(path: Path, *, resource_id: str, duration_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{resource_id}.tmp.mp4")
    token = int(hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:8], 16)
    frequency = 220 + token % 440
    hue = token % 360
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=854x480:rate=30:duration={duration_s:g}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={frequency}:sample_rate=48000:duration={duration_s:g}",
        "-vf",
        f"hue=h={hue}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(120, round(duration_s * 15)),
        )
        if not _valid_offline_output(temporary):
            raise RuntimeError(f"FFmpeg produced an invalid fixture video: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _offline_executor_factory(
    output_dir: Path,
    task_store: Any,
    *,
    stats: OfflineExecutionStats,
):
    from runtime.continuity_chunks import ChunkExecutionResult

    run_id = str(output_dir.resolve())

    def execute(request: Any) -> ChunkExecutionResult:
        payload = _offline_payload(request)
        succeeded = task_store.find_succeeded(
            run_id=run_id,
            task_type="video.generate",
            resource_id=request.resource_id,
            payload=payload,
            provider_id=OFFLINE_PROVIDER_ID,
        )
        if succeeded is not None:
            expected_hash = str(succeeded.outcome.get("output_sha256") or "")
            if (
                not _valid_offline_output(request.output_path)
                or not expected_hash
                or _sha256(request.output_path) != expected_hash
            ):
                raise RuntimeError(
                    f"offline fixture output no longer matches its ledger: {request.resource_id}"
                )
            stats.reused_chunks += 1
            return ChunkExecutionResult(
                request.output_path,
                succeeded.provider_job_id,
            )

        enqueued = task_store.enqueue(
            run_id=run_id,
            task_type="video.generate",
            media_type="video",
            resource_id=request.resource_id,
            payload=payload,
            provider_id=OFFLINE_PROVIDER_ID,
            input_fingerprint=request.input_fingerprint,
        )
        task = enqueued.task
        if task.status == "queued":
            task = task_store.claim(task.task_id)
            if task is None:
                raise RuntimeError(f"offline task was claimed elsewhere: {request.resource_id}")
        if task.status != "running":
            raise RuntimeError(
                f"offline task is not resumable: {request.resource_id} status={task.status}"
            )
        provider_job_id = task.provider_job_id or (
            f"offline-{request.resource_id}-{task.task_id[:12]}"
        )
        if not task.provider_job_id:
            task_store.persist_provider_job(
                task.task_id,
                provider_job_id=provider_job_id,
                provider_endpoint=OFFLINE_PROVIDER_ENDPOINT,
            )
        try:
            _generate_offline_video(
                request.output_path,
                resource_id=request.resource_id,
                duration_s=float(request.chunk.target_duration_s),
            )
        except Exception as exc:
            task_store.mark_failed(task.task_id, str(exc), provider_terminal=True)
            raise
        outcome = {
            "test_only": True,
            "provider_job_id": provider_job_id,
            "output_path": str(request.output_path),
            "output_sha256": _sha256(request.output_path),
            "input_fingerprint": request.input_fingerprint,
        }
        task_store.mark_succeeded(task.task_id, outcome)
        stats.generated_chunks += 1
        return ChunkExecutionResult(request.output_path, provider_job_id)

    return execute


def _task_rows(database: Path) -> list[dict[str, Any]]:
    if not database.is_file():
        return []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT task_id, resource_id, status, provider_id, provider_job_id,
                   provider_endpoint, input_fingerprint, outcome_json
            FROM generation_tasks
            ORDER BY queued_at, task_id
            """
        ).fetchall()
    values = []
    for row in rows:
        outcome = json.loads(row["outcome_json"] or "{}")
        values.append(
            {
                "task_id": row["task_id"],
                "resource_id": row["resource_id"],
                "status": row["status"],
                "provider_id": row["provider_id"],
                "provider_job_id": row["provider_job_id"],
                "provider_endpoint": row["provider_endpoint"],
                "input_fingerprint": row["input_fingerprint"],
                "output_sha256": outcome.get("output_sha256"),
                "test_only": outcome.get("test_only"),
            }
        )
    return values


def _media_summary(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "duration_s": round(float(payload.get("format", {}).get("duration") or 0), 4),
        "streams": payload.get("streams", []),
    }


def _phase_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "error",
        "duration_s",
        "outputs",
        "provider",
        "mode",
        "method",
        "media_profile",
        "clip_count",
        "video_quality_owner",
        "step_status",
    )
    return {key: receipt[key] for key in keys if key in receipt}


def _offline_transition_embedding_runner(*_args: Any, **_kwargs: Any) -> dict:
    """Disable remote smart-transition embeddings for this offline acceptance."""
    return {}


def _initial_receipt(output_dir: Path) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "test_only": True,
        "status": "running",
        "repo_commit": _repo_commit(),
        "fixture": {
            "path": str(FIXTURE_PATH),
            "sha256": FIXTURE_SHA256,
        },
        "output_dir": str(output_dir),
        "remote_checks_skipped": [
            "paid_video_provider_submission",
            "provider_media_upload",
            "story_order_multimodal_review",
            "per_shot_multimodal_review",
            "smart_transition_remote_embeddings",
            "remote_asr",
            "sam3_object_tracking",
        ],
        "invocations": [],
    }


def _prepare_phase1_to_phase5(output_dir: Path) -> dict[str, Any]:
    from runtime.pipeline_execution import run_pipeline

    result = run_pipeline(
        input_file=str(FIXTURE_PATH),
        duration=TARGET_DURATION_S,
        shot_duration=6,
        dry_run=True,
        skip_phase=[6, 7, 8, 9, 9.5],
        output_dir=str(output_dir),
        project_id="cron-future-station-offline-media",
        transition="cut",
        transition_duration=0.0,
        media_profile=MEDIA_PROFILE,
        enable_reshoot=False,
        no_real_person=False,
        auto_approve=True,
    )
    if result.get("status") != "completed":
        raise RuntimeError(
            "Phase 1-5 dry-run preparation failed: "
            + str(result.get("error") or result.get("status"))
        )
    return result


def _validate_resume_inputs(output_dir: Path, receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError("cannot resume an unknown offline acceptance receipt")
    if receipt.get("fixture", {}).get("sha256") != FIXTURE_SHA256:
        raise RuntimeError("offline acceptance fixture lineage changed")
    required = (
        "STORYBOARD.json",
        "CONTINUITY_PLAN.json",
        "storyboard_qa_report.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("offline acceptance resume inputs missing: " + ", ".join(missing))


def _validate_completed_media(
    output_dir: Path, completed_invocation: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    recorded_media = completed_invocation.get("media", {})
    names = ("raw_assembly.mp4", "polished.mp4")
    missing_receipts = [name for name in names if name not in recorded_media]
    if missing_receipts:
        raise RuntimeError(
            "completed acceptance is missing media lineage: "
            + ", ".join(missing_receipts)
        )
    current_media = {name: _media_summary(output_dir / name) for name in names}
    changed_media = [
        name
        for name, summary in current_media.items()
        if recorded_media[name].get("sha256") != summary["sha256"]
    ]
    if changed_media:
        raise RuntimeError(
            "completed acceptance media no longer matches its lineage: "
            + ", ".join(changed_media)
        )
    return current_media


def run_acceptance(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if not FIXTURE_PATH.is_file() or _sha256(FIXTURE_PATH) != FIXTURE_SHA256:
        raise RuntimeError("Future Station fixture is missing or has the wrong SHA-256")
    output_dir = output_dir.resolve()
    receipt_path = output_dir / RECEIPT_NAME
    if resume:
        if not receipt_path.is_file():
            raise RuntimeError("--resume requires an existing offline acceptance receipt")
        receipt = _read_json(receipt_path)
        _validate_resume_inputs(output_dir, receipt)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("cold acceptance output directory must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        receipt = _initial_receipt(output_dir)

    stats = OfflineExecutionStats()
    tasks_before = _task_rows(output_dir / "runtime.db")
    prior_completed = next(
        (
            item
            for item in reversed(receipt.get("invocations", []))
            if item.get("status") == "completed"
        ),
        None,
    )
    completed_media = (
        _validate_completed_media(output_dir, prior_completed)
        if resume and prior_completed is not None
        else None
    )
    invocation: dict[str, Any] = {
        "mode": "resume" if resume else "cold",
        "started_at": _utc_now(),
        "status": "running",
        "task_count_before": len(tasks_before),
        "phase_results": {},
    }
    receipt.setdefault("invocations", []).append(invocation)
    receipt["status"] = "running"
    _atomic_write_json(receipt_path, receipt)

    environment = {
        "ARK_AGENT_API_KEY": None,
        "ARK_API_KEY": None,
        "TOS_ACCESS_KEY": None,
        "TOS_SECRET_KEY": None,
        "TOS_BUCKET": None,
        "HONCUT_CONTINUITY_MODE": "auto",
        "HONCUT_CONTINUITY_BRIDGE": "off",
        "HONCUT_STORYBOARD_REVIEW": "mock",
        "HONCUT_SHOT_VLM_REVIEW": "off",
        "HONCUT_FINAL_VLM_REVIEW": "off",
        "HONCUT_ASR_MOCK": "1",
        "HONCUT_SAM3_MODE": "off",
        "HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY": "fail",
        "VIDEO_GEN_CONCURRENCY": "1",
    }

    try:
        with _temporary_environment(environment), _deny_provider_requests(stats):
            if not resume:
                prepared = _prepare_phase1_to_phase5(output_dir)
                invocation["phase1_5"] = {"status": prepared["status"]}

            storyboard = _read_json(output_dir / "STORYBOARD.json")
            from phases.phase6.phase6_video_gen import run_phase6
            from phases.phase7.phase7_consistency import run_phase7
            from phases.phase8.phase8_assembly import run_phase8
            from phases.phase9.phase9_post import run_phase9

            phase6 = run_phase6(
                storyboard,
                output_dir,
                dry_run=False,
                chain_mode=False,
                _test_continuity_executor_factory=lambda root, store: (
                    _offline_executor_factory(root, store, stats=stats)
                ),
            )
            invocation["phase_results"]["phase6"] = _phase_summary(phase6)
            _atomic_write_json(receipt_path, receipt)
            if phase6.get("status") != "done":
                raise RuntimeError(f"Phase 6 failed: {phase6.get('error') or phase6}")

            phase7 = run_phase7(output_dir, dry_run=False, storyboard_data=storyboard)
            invocation["phase_results"]["phase7"] = _phase_summary(phase7)
            _atomic_write_json(receipt_path, receipt)
            if phase7.get("status") != "done":
                raise RuntimeError(f"Phase 7 failed: {phase7.get('error') or phase7}")

            if completed_media is None:
                phase8 = run_phase8(
                    output_dir,
                    dry_run=False,
                    transition="cut",
                    transition_duration=0.0,
                    media_profile=MEDIA_PROFILE,
                    target_duration=TARGET_DURATION_S,
                    enable_reshoot=False,
                    chain_mode=False,
                    _transition_embedding_runner=(
                        _offline_transition_embedding_runner
                    ),
                )
                invocation["phase_results"]["phase8"] = _phase_summary(phase8)
                _atomic_write_json(receipt_path, receipt)
                if phase8.get("status") != "done":
                    raise RuntimeError(
                        f"Phase 8 failed: {phase8.get('error') or phase8}"
                    )

                phase9 = run_phase9(
                    output_dir,
                    dry_run=False,
                    media_profile=MEDIA_PROFILE,
                    target_duration=TARGET_DURATION_S,
                )
                invocation["phase_results"]["phase9"] = _phase_summary(phase9)
                _atomic_write_json(receipt_path, receipt)
                if phase9.get("status") != "done":
                    raise RuntimeError(
                        f"Phase 9 failed: {phase9.get('error') or phase9}"
                    )
            else:
                source_started_at = prior_completed.get("started_at")
                invocation["phase_results"]["phase8"] = {
                    "status": "done",
                    "mode": "reused_completed_media",
                    "outputs": ["raw_assembly.mp4"],
                    "source_started_at": source_started_at,
                }
                invocation["phase_results"]["phase9"] = {
                    "status": "done",
                    "mode": "reused_completed_media",
                    "outputs": ["polished.mp4"],
                    "source_started_at": source_started_at,
                }
                _atomic_write_json(receipt_path, receipt)

        tasks_after = _task_rows(output_dir / "runtime.db")
        invalid_tasks = [
            task
            for task in tasks_after
            if task.get("provider_id") != OFFLINE_PROVIDER_ID
            or task.get("provider_endpoint") != OFFLINE_PROVIDER_ENDPOINT
            or task.get("status") != "succeeded"
            or task.get("test_only") is not True
        ]
        if not tasks_after or invalid_tasks:
            raise RuntimeError(
                "offline acceptance task ledger contains missing or non-fixture tasks: "
                + json.dumps(invalid_tasks, ensure_ascii=False)
            )
        if stats.provider_requests:
            raise RuntimeError(
                f"offline acceptance attempted {stats.provider_requests} Provider requests"
            )
        if resume:
            task_ids_before = [task["task_id"] for task in tasks_before]
            task_ids_after = [task["task_id"] for task in tasks_after]
            if task_ids_after != task_ids_before:
                raise RuntimeError(
                    "resume changed generation task lineage for unchanged inputs"
                )

        media = {
            name: _media_summary(output_dir / name)
            for name in ("raw_assembly.mp4", "polished.mp4")
        }
        if resume and prior_completed is not None:
            prior_media = prior_completed.get("media", {})
            changed_media = [
                name
                for name, summary in media.items()
                if prior_media.get(name, {}).get("sha256") != summary["sha256"]
            ]
            if changed_media:
                raise RuntimeError(
                    "resume changed final media hashes for unchanged inputs: "
                    + ", ".join(changed_media)
                )
        invocation.update(
            status="completed",
            finished_at=_utc_now(),
            task_count_after=len(tasks_after),
            generated_chunks=stats.generated_chunks,
            reused_chunks=stats.reused_chunks,
            provider_requests=stats.provider_requests,
            task_ids=[task["task_id"] for task in tasks_after],
            media=media,
        )
        if resume:
            invocation["resume_lineage"] = {
                "task_ids_preserved": True,
                "media_hashes_preserved": prior_completed is not None,
            }
        receipt.update(
            status="completed",
            completed_at=_utc_now(),
            provider_requests=0,
            tasks=tasks_after,
            media=media,
        )
        _atomic_write_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        invocation.update(
            status="failed",
            finished_at=_utc_now(),
            error=str(exc),
            generated_chunks=stats.generated_chunks,
            reused_chunks=stats.reused_chunks,
            provider_requests=stats.provider_requests,
        )
        receipt.update(status="failed", error=str(exc), failed_at=_utc_now())
        _atomic_write_json(receipt_path, receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        receipt = run_acceptance(args.output_dir, resume=args.resume)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    latest = receipt["invocations"][-1]
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "mode": latest["mode"],
                "output_dir": receipt["output_dir"],
                "task_count": latest["task_count_after"],
                "provider_requests": latest["provider_requests"],
                "polished_video": latest["media"]["polished.mp4"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
