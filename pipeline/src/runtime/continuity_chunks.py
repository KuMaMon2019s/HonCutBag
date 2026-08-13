"""Dependency-aware execution for provider-sized chunks inside editorial shots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from schemas.continuity import ContinuityPlan, ContinuityShot, GenerationChunk

CONTINUITY_MODE_ENV = "HONCUT_CONTINUITY_MODE"
CONTINUITY_MODES = {"off", "shadow", "auto"}
LINEAGE_KIND = "honcut.continuity_lineage.v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class ChunkExecutionRequest:
    """One chunk invocation passed to an existing crash-safe provider boundary."""

    resource_id: str
    shot_id: str
    chunk: GenerationChunk
    anchors: dict[str, Any]
    output_path: Path
    previous_output_path: Path | None
    input_fingerprint: str
    memory_context: str
    repair_attempt: int = 0


@dataclass(frozen=True)
class ChunkExecutionResult:
    """Minimum provider result needed for lineage and restart recovery."""

    output_path: Path
    provider_task_id: str | None = None


class ContinuityLineageStore:
    """Atomically persist chunk fingerprints while independent shots finish."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._document = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"kind": LINEAGE_KIND, "chunks": {}, "seams": {}, "shots": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("kind") != LINEAGE_KIND:
            raise ValueError(f"unsupported continuity lineage in {self.path}")
        value.setdefault("seams", {})
        if any(not isinstance(value.get(key), dict) for key in ("chunks", "seams", "shots")):
            raise ValueError(f"invalid continuity lineage shape in {self.path}")
        return value

    def get_chunk(self, resource_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._document["chunks"].get(resource_id)
            return dict(record) if isinstance(record, dict) else None

    def get_shot(self, shot_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._document["shots"].get(shot_id)
            return dict(record) if isinstance(record, dict) else None

    def get_seam(self, boundary_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._document["seams"].get(boundary_id)
            return dict(record) if isinstance(record, dict) else None

    def put_chunk(self, resource_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._document["chunks"][resource_id] = record
            _atomic_write_json(self.path, self._document)

    def put_shot(self, shot_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._document["shots"][shot_id] = record
            _atomic_write_json(self.path, self._document)

    def put_seam(self, boundary_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._document["seams"][boundary_id] = record
            _atomic_write_json(self.path, self._document)


def continuity_mode() -> str:
    """Read and validate the rollout mode; shadow is the safe default."""
    mode = os.environ.get(CONTINUITY_MODE_ENV, "shadow").strip().lower()
    if mode not in CONTINUITY_MODES:
        expected = ", ".join(sorted(CONTINUITY_MODES))
        raise ValueError(f"{CONTINUITY_MODE_ENV} must be one of {expected}, got {mode!r}")
    return mode


def load_continuity_plan(path: str | Path) -> ContinuityPlan:
    return ContinuityPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_shadow_runtime_report(output_dir: str | Path) -> dict[str, Any]:
    """Validate rollout inputs and persist a report without invoking a provider."""
    root = Path(output_dir)
    mode = continuity_mode()
    report: dict[str, Any] = {
        "version": 1,
        "mode": mode,
        "execution_enabled": False,
    }
    if mode == "off":
        report["reason"] = "continuity runtime disabled"
        return report

    plan_path = root / "CONTINUITY_PLAN.json"
    if not plan_path.is_file():
        if mode == "auto":
            raise RuntimeError("continuity seam guard requires CONTINUITY_PLAN.json")
        report["reason"] = "CONTINUITY_PLAN.json not found"
    else:
        plan = load_continuity_plan(plan_path)
        report.update(
            {
                "reason": "shadow mode records intent without provider submissions",
                "shot_count": len(plan.shots),
                "chunk_count": sum(len(shot.chunks) for shot in plan.shots),
                "extension_count": sum(
                    chunk.mode == "native_extend" for shot in plan.shots for chunk in shot.chunks
                ),
                "plan_path": "CONTINUITY_PLAN.json",
            }
        )
        if mode == "auto":
            from quality.seam_calibration import load_seam_calibration

            calibration_path = root / "CONTINUITY_CALIBRATION.json"
            if not calibration_path.is_file():
                raise RuntimeError(
                    "continuity seam guard requires CONTINUITY_CALIBRATION.json"
                )
            calibration = load_seam_calibration(calibration_path)
            if calibration.status != "certified":
                raise RuntimeError(
                    "continuity seam guard requires certified calibration, "
                    f"got {calibration.status}"
                )
            report.update(
                {
                    "execution_enabled": True,
                    "reason": "certified auto mode routes provider work through chunk lineage",
                    "calibration_path": "CONTINUITY_CALIBRATION.json",
                    "calibration_fingerprint": calibration.dataset_fingerprint,
                }
            )
    _atomic_write_json(root / "CONTINUITY_RUNTIME.json", report)
    return report


def _chunk_fingerprint(
    shot: ContinuityShot,
    chunk: GenerationChunk,
    parent_fingerprint: str | None,
    execution_context: Any = None,
) -> str:
    return _canonical_hash(
        {
            "shot_id": shot.shot_id,
            "anchors": shot.anchors.model_dump(mode="json"),
            "chunk": chunk.model_dump(mode="json"),
            "parent_fingerprint": parent_fingerprint,
            "execution_context": execution_context,
        }
    )


def _valid_record(record: dict[str, Any] | None, fingerprint: str, path: Path) -> bool:
    return bool(
        record
        and record.get("status") == "succeeded"
        and record.get("input_fingerprint") == fingerprint
        and path.is_file()
        and path.stat().st_size > 0
        and record.get("output_sha256") == _file_hash(path)
    )


def _default_materialize(chunk_paths: Sequence[Path], output_path: Path) -> None:
    if len(chunk_paths) != 1:
        raise RuntimeError("multi-chunk shots require an overlap-aware materialize_shot callback")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    shutil.copy2(chunk_paths[0], temporary)
    os.replace(temporary, output_path)


def execute_continuity_plan(
    plan: ContinuityPlan,
    output_dir: str | Path,
    *,
    execute_chunk: Callable[[ChunkExecutionRequest], ChunkExecutionResult],
    inspect_seam: Callable[[Path, Path, str], dict[str, Any]] | None = None,
    materialize_shot: Callable[[Sequence[Path], Path], None] = _default_materialize,
    chunk_context: Callable[[ContinuityShot, GenerationChunk], Any] | None = None,
    seam_calibration: dict[str, Any] | None = None,
    max_seam_repairs: int = 1,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run each shot's chunks serially while independent shots run concurrently."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_seam_repairs < 0:
        raise ValueError("max_seam_repairs must not be negative")

    root = Path(output_dir)
    from runtime.continuity_memory import (
        initialize_continuity_memory,
        render_continuity_memory_context,
    )

    initialize_continuity_memory(root, plan)
    calibration = None
    if seam_calibration is not None:
        from quality.seam_calibration import SeamCalibration

        calibration = SeamCalibration.model_validate(seam_calibration)
        if calibration.status != "certified":
            raise RuntimeError(
                f"continuity repair requires certified seam calibration, got {calibration.status}"
            )
    lineage = ContinuityLineageStore(root / "CONTINUITY_LINEAGE.json")
    outputs: list[str] = []
    errors: list[dict[str, str]] = []
    skipped_chunks = 0
    executed_chunks = 0
    measured_seams = 0
    repair_attempts = 0
    totals_lock = threading.Lock()

    def run_shot(shot: ContinuityShot) -> str:
        nonlocal executed_chunks, measured_seams, repair_attempts, skipped_chunks
        chunk_paths: list[Path] = []
        seam_fingerprints: list[str] = []
        parent_fingerprint: str | None = None
        previous_output: Path | None = None

        def execute_one(
            chunk: GenerationChunk,
            chunk_path: Path,
            fingerprint: str,
            *,
            attempt: int,
        ) -> None:
            nonlocal executed_chunks, repair_attempts
            resource_id = chunk.chunk_id if attempt == 0 else f"{chunk.chunk_id}_R{attempt:02d}"
            request = ChunkExecutionRequest(
                resource_id=resource_id,
                shot_id=shot.shot_id,
                chunk=chunk,
                anchors=shot.anchors.model_dump(mode="json"),
                output_path=chunk_path,
                previous_output_path=previous_output,
                input_fingerprint=fingerprint,
                memory_context=render_continuity_memory_context(root, plan, shot.shot_id),
                repair_attempt=attempt,
            )
            try:
                result = execute_chunk(request)
                produced_path = Path(result.output_path)
                if produced_path.resolve() != chunk_path.resolve():
                    raise RuntimeError(
                        f"{resource_id} wrote {produced_path}, expected {chunk_path}"
                    )
                if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                    raise RuntimeError(f"{resource_id} produced no video bytes")
            except Exception as exc:
                lineage.put_chunk(
                    chunk.chunk_id,
                    {
                        "status": "failed",
                        "shot_id": shot.shot_id,
                        "resource_id": resource_id,
                        "input_fingerprint": fingerprint,
                        "repair_attempts": attempt,
                        "error": str(exc),
                        "updated_at": _utc_now(),
                    },
                )
                raise
            lineage.put_chunk(
                chunk.chunk_id,
                {
                    "status": "succeeded",
                    "shot_id": shot.shot_id,
                    "resource_id": resource_id,
                    "input_fingerprint": fingerprint,
                    "output_path": str(chunk_path.relative_to(root)),
                    "output_sha256": _file_hash(chunk_path),
                    "provider_task_id": result.provider_task_id,
                    "repair_attempts": attempt,
                    "updated_at": _utc_now(),
                },
            )
            with totals_lock:
                executed_chunks += 1
                if attempt > 0:
                    repair_attempts += 1

        def inspect_boundary(
            chunk: GenerationChunk,
            following_path: Path,
            fingerprint: str,
        ) -> str:
            nonlocal measured_seams
            if previous_output is None:
                raise RuntimeError("cannot inspect a seam without a predecessor")
            boundary_id = f"{previous_output.stem}__{following_path.stem}"
            attempts = int((lineage.get_chunk(chunk.chunk_id) or {}).get("repair_attempts", 0))
            while True:
                seam_fingerprint = _canonical_hash(
                    {
                        "previous_sha256": _file_hash(previous_output),
                        "following_sha256": _file_hash(following_path),
                        "calibration_fingerprint": (
                            calibration.dataset_fingerprint if calibration else None
                        ),
                        "repair_attempts": attempts,
                        "max_repairs": max_seam_repairs,
                    }
                )
                seam_record = lineage.get_seam(boundary_id)
                if (
                    seam_record
                    and seam_record.get("status") == "observed"
                    and seam_record.get("input_fingerprint") == seam_fingerprint
                ):
                    evidence = seam_record["evidence"]
                    decision = seam_record.get(
                        "decision",
                        {"action": "observe_only", "reason": "no calibrated policy supplied"},
                    )
                else:
                    if inspect_seam is None:
                        from quality.continuity_seam import measure_video_seam

                        evidence = measure_video_seam(
                            previous_output,
                            following_path,
                            boundary_id,
                            evidence_dir=root / "continuity_seams",
                        )
                    else:
                        evidence = inspect_seam(previous_output, following_path, boundary_id)
                    if calibration is None:
                        decision = {
                            "action": "observe_only",
                            "reason": "no calibrated policy supplied",
                            "repair_attempts": attempts,
                            "max_repairs": max_seam_repairs,
                        }
                    else:
                        from quality.seam_calibration import decide_seam

                        metrics = evidence.get("metrics")
                        if not isinstance(metrics, dict):
                            raise RuntimeError(
                                f"{boundary_id} seam inspector returned no metrics object"
                            )
                        decision = decide_seam(
                            metrics,
                            calibration,
                            repair_attempts=attempts,
                            max_repairs=max_seam_repairs,
                        )
                    candidate_frames = [
                        *evidence.get("tail_frames", []),
                        *evidence.get("head_frames", []),
                    ]
                    if candidate_frames:
                        from runtime.continuity_memory import record_recent_motion

                        with totals_lock:
                            record_recent_motion(
                                root,
                                plan,
                                shot_id=shot.shot_id,
                                chunk_id=following_path.stem,
                                candidate_frames=candidate_frames,
                                screen_direction=shot.anchors.screen_direction,
                                camera_motion=shot.anchors.camera_motion,
                            )
                    prior_attempts = []
                    if seam_record and isinstance(seam_record.get("attempt_history"), list):
                        prior_attempts = seam_record["attempt_history"]
                    current_attempt = {
                        "input_fingerprint": seam_fingerprint,
                        "evidence": evidence,
                        "decision": decision,
                        "updated_at": _utc_now(),
                    }
                    lineage.put_seam(
                        boundary_id,
                        {
                            "status": "observed",
                            "policy": (
                                "calibrated_bounded_repair" if calibration else "observe_only"
                            ),
                            "input_fingerprint": seam_fingerprint,
                            "evidence": evidence,
                            "decision": decision,
                            "attempt_history": [*prior_attempts, current_attempt],
                            "updated_at": _utc_now(),
                        },
                    )
                    with totals_lock:
                        measured_seams += 1

                action = decision.get("action")
                if action == "regenerate":
                    attempts += 1
                    execute_one(chunk, following_path, fingerprint, attempt=attempts)
                    continue
                if action == "human_review":
                    raise RuntimeError(
                        f"{boundary_id} requires human review: {decision.get('reason', '')}"
                    )
                if action not in {"accept", "observe_only"}:
                    raise RuntimeError(f"{boundary_id} returned unsupported action {action!r}")
                return seam_fingerprint

        for chunk in shot.chunks:
            chunk_path = root / "shots" / shot.shot_id / "chunks" / f"{chunk.chunk_id}.mp4"
            execution_context = chunk_context(shot, chunk) if chunk_context else None
            fingerprint = _chunk_fingerprint(
                shot,
                chunk,
                parent_fingerprint,
                execution_context,
            )
            record = lineage.get_chunk(chunk.chunk_id)
            if _valid_record(record, fingerprint, chunk_path):
                with totals_lock:
                    skipped_chunks += 1
            else:
                if chunk.mode == "native_extend" and previous_output is None:
                    raise RuntimeError(
                        f"{chunk.chunk_id} cannot run without its predecessor output"
                    )
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                resume_attempt = 0
                if record and record.get("input_fingerprint") == fingerprint:
                    resume_attempt = int(record.get("repair_attempts", 0))
                execute_one(chunk, chunk_path, fingerprint, attempt=resume_attempt)
            if previous_output is not None:
                seam_fingerprints.append(inspect_boundary(chunk, chunk_path, fingerprint))
            chunk_paths.append(chunk_path)
            previous_output = chunk_path
            parent_fingerprint = _canonical_hash(
                {
                    "input_fingerprint": fingerprint,
                    "output_sha256": _file_hash(chunk_path),
                }
            )

        shot_output = root / "shots" / shot.shot_id / "output.mp4"
        materialization_fingerprint = _canonical_hash(
            {
                "shot_id": shot.shot_id,
                "chunks": [
                    {
                        "input_fingerprint": lineage.get_chunk(chunk.chunk_id)["input_fingerprint"],
                        "output_sha256": _file_hash(path),
                    }
                    for chunk, path in zip(shot.chunks, chunk_paths, strict=True)
                ],
                "seams": seam_fingerprints,
            }
        )
        shot_record = lineage.get_shot(shot.shot_id)
        if not _valid_record(shot_record, materialization_fingerprint, shot_output):
            materialize_shot(tuple(chunk_paths), shot_output)
            if not shot_output.is_file() or shot_output.stat().st_size == 0:
                raise RuntimeError(f"{shot.shot_id} materialization produced no video bytes")
            lineage.put_shot(
                shot.shot_id,
                {
                    "status": "succeeded",
                    "input_fingerprint": materialization_fingerprint,
                    "output_path": str(shot_output.relative_to(root)),
                    "output_sha256": _file_hash(shot_output),
                    "updated_at": _utc_now(),
                },
            )
        return str(shot_output.relative_to(root))

    worker_count = min(max_workers, max(1, len(plan.shots)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_shot, shot): shot for shot in plan.shots}
        for future in as_completed(futures):
            shot = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                errors.append({"shot_id": shot.shot_id, "error": str(exc)})

    outputs.sort()
    errors.sort(key=lambda item: item["shot_id"])
    return {
        "status": "error" if errors else "done",
        "outputs": outputs,
        "errors": errors,
        "executed_chunks": executed_chunks,
        "measured_seams": measured_seams,
        "repair_attempts": repair_attempts,
        "skipped_chunks": skipped_chunks,
        "lineage_path": "CONTINUITY_LINEAGE.json",
    }
