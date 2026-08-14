"""Dependency-aware execution for provider-sized chunks inside editorial shots."""

from __future__ import annotations

import hashlib
import json
import math
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
REVIEW_DECISIONS_KIND = "honcut.continuity_review_decisions.v1"
SEAM_DECISIONS_KIND = "honcut.continuity_seam_decisions.v1"


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


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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


def _load_review_decisions(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "CONTINUITY_REVIEW_DECISIONS.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != REVIEW_DECISIONS_KIND:
        raise ValueError(f"unsupported continuity review decisions in {path}")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"{path} must contain a decisions object")
    for boundary_id, decision in decisions.items():
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError(f"{path} contains an invalid boundary id")
        if not isinstance(decision, dict):
            raise ValueError(f"{boundary_id} review decision must be an object")
        if decision.get("action") not in {"accept", "regenerate"}:
            raise ValueError(f"{boundary_id} review action must be accept or regenerate")
        if not str(decision.get("approved_input_fingerprint") or "").strip():
            raise ValueError(f"{boundary_id} review decision requires an input fingerprint")
    return decisions


def _load_phase8_seam_decisions(root: Path) -> dict[str, dict[str, Any]]:
    """Load machine decisions made after the full chunk trajectory was visible."""
    path = root / "CONTINUITY_SEAM_DECISIONS.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != SEAM_DECISIONS_KIND:
        raise ValueError(f"unsupported Phase 8 continuity decisions in {path}")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"{path} must contain a decisions object")
    return {
        str(boundary_id): decision
        for boundary_id, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") == "hard_trim"
    }


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
                raise RuntimeError("continuity seam guard requires CONTINUITY_CALIBRATION.json")
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
    prepare_seam: Callable[[Path, Path, str], dict[str, Any]] | None = None,
    materialize_shot: Callable[[Sequence[Path], Path], None] = _default_materialize,
    probe_frames: Callable[[Path, int], dict[str, Any]] | None = None,
    finalize_shot: Callable[[Path, int, int], dict[str, Any]] | None = None,
    chunk_context: Callable[[ContinuityShot, GenerationChunk], Any] | None = None,
    seam_calibration: dict[str, Any] | None = None,
    max_seam_repairs: int = 1,
    max_duration_topups: int = 3,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run each shot's chunks serially while independent shots run concurrently."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_seam_repairs < 0:
        raise ValueError("max_seam_repairs must not be negative")
    if max_duration_topups < 0:
        raise ValueError("max_duration_topups must not be negative")

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
    review_decisions = _load_review_decisions(root)
    phase8_seam_decisions = _load_phase8_seam_decisions(root)
    outputs: list[str] = []
    errors: list[dict[str, str]] = []
    skipped_chunks = 0
    executed_chunks = 0
    measured_seams = 0
    prepared_boundaries: set[str] = set()
    timing_manifests = 0
    duration_topups = 0
    repair_attempts = 0
    totals_lock = threading.Lock()

    def run_shot(shot: ContinuityShot) -> str:
        nonlocal executed_chunks, measured_seams, repair_attempts, skipped_chunks
        nonlocal timing_manifests, duration_topups
        chunk_paths: list[Path] = []
        executed_chunk_models: list[GenerationChunk] = []
        timing_rows: list[dict[str, Any]] = []
        seam_fingerprints: list[str] = []
        parent_fingerprint: str | None = None
        previous_output: Path | None = None
        previous_chunk_id: str | None = None

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
        ) -> tuple[str, Path, dict[str, Any] | None]:
            nonlocal measured_seams
            if previous_output is None:
                raise RuntimeError("cannot inspect a seam without a predecessor")
            boundary_id = f"{previous_chunk_id or previous_output.stem}__{chunk.chunk_id}"
            attempts = int((lineage.get_chunk(chunk.chunk_id) or {}).get("repair_attempts", 0))
            while True:
                preparation: dict[str, Any] | None = None
                effective_following = following_path
                if prepare_seam is not None:
                    preparation = prepare_seam(previous_output, following_path, boundary_id)
                    prepared_path = Path(str(preparation.get("output_path") or following_path))
                    if prepared_path.resolve() != following_path.resolve():
                        if not prepared_path.is_file() or prepared_path.stat().st_size == 0:
                            raise RuntimeError(
                                f"{boundary_id} seam preparation produced no video bytes"
                            )
                        effective_following = prepared_path
                        with totals_lock:
                            prepared_boundaries.add(boundary_id)
                seam_fingerprint = _canonical_hash(
                    {
                        "previous_sha256": _file_hash(previous_output),
                        "following_sha256": _file_hash(following_path),
                        "effective_following_sha256": _file_hash(effective_following),
                        "seam_preparation": preparation,
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
                    preparation = seam_record.get("preparation")
                else:
                    if inspect_seam is None:
                        from quality.continuity_seam import measure_video_seam

                        evidence = measure_video_seam(
                            previous_output,
                            effective_following,
                            boundary_id,
                            evidence_dir=root / "continuity_seams",
                        )
                    else:
                        evidence = inspect_seam(previous_output, effective_following, boundary_id)
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
                    replay = evidence.get("replay")
                    if isinstance(replay, dict) and replay.get("likely_replay"):
                        replay_reason = (
                            "the following chunk appears to replay the predecessor's "
                            "aligned motion trajectory"
                        )
                        if decision.get("action") in {"accept", "observe_only"}:
                            decision = {
                                **decision,
                                "action": "human_review",
                                "reason": replay_reason,
                                "replay_policy": "human_review_only",
                            }
                        elif decision.get("action") == "human_review":
                            decision = {
                                **decision,
                                "reason": f"{decision.get('reason', '')}; {replay_reason}",
                                "replay_policy": "human_review_only",
                            }
                    temporal_decision = phase8_seam_decisions.get(boundary_id)
                    if (
                        temporal_decision is not None
                        and decision.get("action") in {"accept", "observe_only", "human_review"}
                    ):
                        decision = {
                            **decision,
                            "action": "accept",
                            "reason": (
                                "Phase 8 inspected the complete temporal trajectory and "
                                "authorized an interpolation-free hard trim"
                            ),
                            "phase8_temporal_adjudication": temporal_decision,
                        }
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
                                chunk_id=chunk.chunk_id,
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
                        "preparation": preparation,
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
                            "preparation": preparation,
                            "decision": decision,
                            "attempt_history": [*prior_attempts, current_attempt],
                            "updated_at": _utc_now(),
                        },
                    )
                    with totals_lock:
                        measured_seams += 1

                action = decision.get("action")
                review = review_decisions.get(boundary_id)
                if (
                    action == "human_review"
                    and review
                    and review.get("approved_input_fingerprint") == seam_fingerprint
                ):
                    action = str(review["action"])
                    if action == "regenerate" and attempts >= max_seam_repairs:
                        raise RuntimeError(
                            f"{boundary_id} human-approved repair exceeds the repair budget"
                        )
                    decision = {
                        **decision,
                        "action": action,
                        "reason": f"human approved {action} for this exact seam evidence",
                        "human_review": review,
                    }
                    current_seam = lineage.get_seam(boundary_id) or {}
                    lineage.put_seam(
                        boundary_id,
                        {
                            **current_seam,
                            "decision": decision,
                            "human_review": review,
                            "updated_at": _utc_now(),
                        },
                    )
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
                return seam_fingerprint, effective_following, preparation

        scheduled_chunks = list(shot.chunks)
        planned_overlap_frames = max(
            (chunk.expected_overlap_frames for chunk in shot.chunks),
            default=0,
        )
        target_frames = shot.target_frames or round(
            shot.target_duration_s * plan.timeline_fps
        )
        chunk_index = 0
        while chunk_index < len(scheduled_chunks):
            chunk = scheduled_chunks[chunk_index]
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
                seam_fingerprint, effective_path, preparation = inspect_boundary(
                    chunk, chunk_path, fingerprint
                )
                seam_fingerprints.append(seam_fingerprint)
            else:
                effective_path = chunk_path
                preparation = None
            chunk_paths.append(effective_path)
            executed_chunk_models.append(chunk)
            if probe_frames is not None:
                raw_timing = probe_frames(chunk_path, plan.timeline_fps)
                effective_timing = probe_frames(effective_path, plan.timeline_fps)
                overlap = (preparation or {}).get("overlap")
                overlap_seconds = (
                    float(overlap.get("overlap_seconds", 0.0))
                    if isinstance(overlap, dict)
                    else 0.0
                )
                timing_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "requested_frames": chunk.requested_frames,
                        "expected_overlap_frames": chunk.expected_overlap_frames,
                        "expected_unique_frames": chunk.expected_unique_frames,
                        "decoded_raw_frames": int(raw_timing["frames"]),
                        "detected_overlap_frames": round(overlap_seconds * plan.timeline_fps),
                        "bridge_frames": int(
                            (preparation or {}).get("selected_bridge_frames") or 0
                        ),
                        "effective_unique_frames": int(effective_timing["frames"]),
                        "effective_path": _portable_path(effective_path, root),
                    }
                )
            previous_output = effective_path
            previous_chunk_id = chunk.chunk_id
            parent_fingerprint = _canonical_hash(
                {
                    "input_fingerprint": fingerprint,
                    "output_sha256": _file_hash(effective_path),
                }
            )
            chunk_index += 1

            if (
                chunk_index == len(scheduled_chunks)
                and probe_frames is not None
                and planned_overlap_frames > 0
            ):
                effective_frames = sum(
                    int(row["effective_unique_frames"]) for row in timing_rows
                )
                deficit_frames = target_frames - effective_frames
                small_deficit_limit = max(2, math.ceil(target_frames * 0.02))
                if deficit_frames > small_deficit_limit:
                    topup_number = len(scheduled_chunks) - len(shot.chunks) + 1
                    if topup_number > max_duration_topups:
                        raise RuntimeError(
                            f"{shot.shot_id} remains short by {deficit_frames} frames after "
                            f"{max_duration_topups} continuation top-ups"
                        )
                    provider_limit_frames = round(
                        plan.provider_chunk_limit_s * plan.timeline_fps
                    )
                    usable_limit = provider_limit_frames - planned_overlap_frames
                    unique_request_frames = min(deficit_frames, usable_limit)
                    requested_seconds = max(
                        2,
                        math.ceil(
                            (unique_request_frames + planned_overlap_frames)
                            / plan.timeline_fps
                        ),
                    )
                    requested_frames = requested_seconds * plan.timeline_fps
                    if requested_frames > provider_limit_frames:
                        raise RuntimeError(
                            f"{shot.shot_id} duration top-up exceeds provider chunk limit"
                        )
                    topup_id = f"{shot.shot_id}_T{topup_number:02d}"
                    scheduled_chunks.append(
                        GenerationChunk(
                            chunk_id=topup_id,
                            sequence=chunk.sequence + 1,
                            target_duration_s=float(requested_seconds),
                            requested_frames=requested_frames,
                            expected_overlap_frames=planned_overlap_frames,
                            expected_unique_frames=requested_frames - planned_overlap_frames,
                            mode="native_extend",
                            depends_on=chunk.chunk_id,
                        )
                    )
                    with totals_lock:
                        duration_topups += 1

        shot_output = root / "shots" / shot.shot_id / "output.mp4"
        materialization_fingerprint = _canonical_hash(
            {
                "shot_id": shot.shot_id,
                "chunks": [
                    {
                        "input_fingerprint": lineage.get_chunk(chunk.chunk_id)["input_fingerprint"],
                        "output_sha256": _file_hash(path),
                    }
                    for chunk, path in zip(executed_chunk_models, chunk_paths, strict=True)
                ],
                "seams": seam_fingerprints,
                "duration_closure": {
                    "target_frames": shot.target_frames,
                    "timeline_fps": plan.timeline_fps,
                    "enabled": finalize_shot is not None,
                },
            }
        )
        shot_record = lineage.get_shot(shot.shot_id)
        if not _valid_record(shot_record, materialization_fingerprint, shot_output):
            materialize_shot(tuple(chunk_paths), shot_output)
            if not shot_output.is_file() or shot_output.stat().st_size == 0:
                raise RuntimeError(f"{shot.shot_id} materialization produced no video bytes")
            closure = None
            if finalize_shot is not None:
                closure = finalize_shot(shot_output, target_frames, plan.timeline_fps)
            lineage.put_shot(
                shot.shot_id,
                {
                    "status": "succeeded",
                    "input_fingerprint": materialization_fingerprint,
                    "output_path": str(shot_output.relative_to(root)),
                    "output_sha256": _file_hash(shot_output),
                    "duration_closure": closure,
                    "updated_at": _utc_now(),
                },
            )
        else:
            closure = (shot_record or {}).get("duration_closure")
        if probe_frames is not None:
            final_timing = probe_frames(shot_output, plan.timeline_fps)
            cumulative_unique = 0
            for row in timing_rows:
                cumulative_unique += int(row["effective_unique_frames"])
                row["remaining_target_frames"] = target_frames - cumulative_unique
            timing_receipt = {
                "kind": "honcut.continuity_timing.v1",
                "shot_id": shot.shot_id,
                "timeline_fps": plan.timeline_fps,
                "target_frames": target_frames,
                "target_duration_s": round(target_frames / plan.timeline_fps, 6),
                "chunks": timing_rows,
                "materialized_frames_before_closure": sum(
                    int(row["effective_unique_frames"]) for row in timing_rows
                ),
                "final_frames": int(final_timing["frames"]),
                "delta_frames": int(final_timing["frames"]) - target_frames,
                "duration_closure": closure,
                "internal_seams_finalized": True,
            }
            _atomic_write_json(
                root / "shots" / shot.shot_id / "CONTINUITY_TIMING.json",
                timing_receipt,
            )
            with totals_lock:
                timing_manifests += 1
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
        "prepared_seams": len(prepared_boundaries),
        "timing_manifests": timing_manifests,
        "duration_topups": duration_topups,
        "repair_attempts": repair_attempts,
        "skipped_chunks": skipped_chunks,
        "lineage_path": "CONTINUITY_LINEAGE.json",
    }
