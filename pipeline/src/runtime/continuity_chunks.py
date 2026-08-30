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
PRIMARY_SHOT_BRIDGES_KIND = "honcut.primary_shot_bridges.v2"


class ContinuityExecutionPaused(RuntimeError):
    """A durable acceptance limit stopped before the next Provider task."""

    def __init__(self, *, limit: int, executed_chunks: int) -> None:
        self.limit = limit
        self.executed_chunks = executed_chunks
        super().__init__(
            "continuity execution paused at the configured new-chunk limit: "
            f"{executed_chunks}/{limit}"
        )


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
    target_output_path: Path | None = None
    repair_attempt: int = 0


@dataclass(frozen=True)
class ChunkExecutionResult:
    """Minimum provider result needed for lineage and restart recovery."""

    output_path: Path
    provider_task_id: str | None = None
    copyright_policy_repairs: tuple[dict[str, Any], ...] = ()
    privacy_policy_repairs: tuple[dict[str, Any], ...] = ()
    provider_fallback: dict[str, Any] | None = None


@dataclass(frozen=True)
class ShotExecutionContext:
    """Output state relayed only inside one cross-shot continuity group."""

    shot_id: str
    output_path: Path
    last_chunk_id: str
    output_fingerprint: str


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
    """Read and validate the rollout mode; continuity execution is the default."""
    mode = os.environ.get(CONTINUITY_MODE_ENV, "auto").strip().lower()
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
            report.update(
                {
                    "execution_enabled": True,
                    "reason": (
                        "auto mode routes generation through continuity groups; "
                        "Phase 8 owns final replay-prefix adjudication"
                    ),
                }
            )
            if calibration_path.is_file():
                calibration = load_seam_calibration(calibration_path)
                if calibration.status != "certified":
                    raise RuntimeError(
                        "continuity seam guard requires certified calibration when "
                        f"a calibration artifact is present, got {calibration.status}"
                    )
                report.update(
                    {
                        "calibration_path": "CONTINUITY_CALIBRATION.json",
                        "calibration_fingerprint": calibration.dataset_fingerprint,
                    }
                )
            else:
                report.update(
                    {
                        "calibration_path": None,
                        "calibration_fingerprint": None,
                        "phase6_seam_policy": "observe_only",
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
    normalize_chunk: Callable[
        [Path, GenerationChunk, int], dict[str, Any]
    ] | None = None,
    chunk_context: Callable[[ContinuityShot, GenerationChunk], Any] | None = None,
    seam_calibration: dict[str, Any] | None = None,
    max_seam_repairs: int = 1,
    max_duration_topups: int = 3,
    max_workers: int = 1,
    max_new_chunks: int | None = None,
) -> dict[str, Any]:
    """Run each shot's chunks serially while independent shots run concurrently."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_seam_repairs < 0:
        raise ValueError("max_seam_repairs must not be negative")
    if max_duration_topups < 0:
        raise ValueError("max_duration_topups must not be negative")
    if (
        max_new_chunks is not None
        and (
            isinstance(max_new_chunks, bool)
            or not isinstance(max_new_chunks, int)
            or max_new_chunks < 1
        )
    ):
        raise ValueError("max_new_chunks must be a positive integer or None")

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
    copyright_policy_repairs = 0
    privacy_policy_repairs = 0
    primary_shot_bridges: list[dict[str, Any]] = []
    totals_lock = threading.Lock()
    reserved_new_chunks = 0

    def run_shot(
        shot: ContinuityShot,
        predecessor: ShotExecutionContext | None = None,
    ) -> ShotExecutionContext:
        nonlocal executed_chunks, measured_seams, repair_attempts, skipped_chunks
        nonlocal copyright_policy_repairs, privacy_policy_repairs
        nonlocal timing_manifests, duration_topups
        chunk_paths: list[Path] = []
        executed_chunk_models: list[GenerationChunk] = []
        timing_rows: list[dict[str, Any]] = []
        seam_fingerprints: list[str] = []
        if shot.extends_from_shot_id:
            if predecessor is None or predecessor.shot_id != shot.extends_from_shot_id:
                raise RuntimeError(
                    f"{shot.shot_id} requires predecessor {shot.extends_from_shot_id}"
                )
            if predecessor.last_chunk_id != shot.extends_from_chunk_id:
                raise RuntimeError(
                    f"{shot.shot_id} predecessor chunk changed: "
                    f"expected {shot.extends_from_chunk_id}, got {predecessor.last_chunk_id}"
                )
            parent_fingerprint: str | None = predecessor.output_fingerprint
            previous_output: Path | None = predecessor.output_path
            previous_chunk_id: str | None = predecessor.last_chunk_id
        else:
            if predecessor is not None:
                raise RuntimeError(f"fresh shot {shot.shot_id} received a predecessor")
            parent_fingerprint = None
            previous_output = None
            previous_chunk_id = None

        def execute_one(
            chunk: GenerationChunk,
            chunk_path: Path,
            fingerprint: str,
            *,
            attempt: int,
        ) -> None:
            nonlocal executed_chunks, repair_attempts, copyright_policy_repairs
            nonlocal privacy_policy_repairs
            nonlocal reserved_new_chunks
            if attempt == 0 and max_new_chunks is not None:
                with totals_lock:
                    if reserved_new_chunks >= max_new_chunks:
                        raise ContinuityExecutionPaused(
                            limit=max_new_chunks,
                            executed_chunks=executed_chunks,
                        )
                    reserved_new_chunks += 1
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
            except ContinuityExecutionPaused:
                raise
            except Exception as exc:
                lineage.put_chunk(
                    chunk.chunk_id,
                    {
                        "status": "failed",
                        "shot_id": shot.shot_id,
                        "resource_id": resource_id,
                        "input_fingerprint": fingerprint,
                        "repair_attempts": attempt,
                        "copyright_policy_repairs": [],
                        "privacy_policy_repairs": [],
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
                    "copyright_policy_repair_attempts": len(
                        result.copyright_policy_repairs
                    ),
                    "copyright_policy_repairs": [
                        dict(item) for item in result.copyright_policy_repairs
                    ],
                    "privacy_policy_repair_attempts": len(
                        result.privacy_policy_repairs
                    ),
                    "privacy_policy_repairs": [
                        dict(item) for item in result.privacy_policy_repairs
                    ],
                    "provider_fallback": (
                        dict(result.provider_fallback)
                        if result.provider_fallback is not None
                        else None
                    ),
                    "updated_at": _utc_now(),
                },
            )
            with totals_lock:
                executed_chunks += 1
                if attempt > 0:
                    repair_attempts += 1
                copyright_policy_repairs += len(result.copyright_policy_repairs)
                privacy_policy_repairs += len(result.privacy_policy_repairs)

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
                if (
                    prepare_seam is not None
                    and chunk.execution_strategy != "first_last_frame_bridge"
                ):
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
            cross_shot_boundary = bool(
                chunk_index == 0
                and shot.extends_from_chunk_id
                and chunk.depends_on == shot.extends_from_chunk_id
            )
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
            if previous_output is not None and not cross_shot_boundary:
                seam_fingerprint, effective_path, preparation = inspect_boundary(
                    chunk, chunk_path, fingerprint
                )
                seam_fingerprints.append(seam_fingerprint)
            else:
                # Cross-shot replay is intentionally preserved until Phase 8,
                # where the complete trajectory and transition class are known.
                effective_path = chunk_path
                preparation = None
            normalization = None
            if chunk.expected_provider_padding_frames:
                if normalize_chunk is None:
                    raise RuntimeError(
                        f"{chunk.chunk_id} has provider-minimum padding but no "
                        "chunk normalizer is configured"
                    )
                normalization = normalize_chunk(
                    effective_path,
                    chunk,
                    plan.timeline_fps,
                )
                normalized_path = Path(
                    str(normalization.get("output_path") or "")
                )
                if not normalized_path.is_file() or normalized_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"{chunk.chunk_id} provider-padding normalization produced "
                        "no video bytes"
                    )
                effective_path = normalized_path
            if chunk.execution_strategy == "first_last_frame_bridge":
                target_shot_id = str(chunk.bridge_target_shot_id or "").strip()
                if not target_shot_id:
                    raise RuntimeError(
                        f"{chunk.chunk_id} has no target primary shot for bridge archival"
                    )
                bridge_dir = root / "shot_bridges"
                bridge_dir.mkdir(parents=True, exist_ok=True)
                bridge_path = bridge_dir / f"{shot.shot_id}__{target_shot_id}.mp4"
                temporary_bridge = bridge_path.with_suffix(".mp4.tmp")
                shutil.copy2(effective_path, temporary_bridge)
                os.replace(temporary_bridge, bridge_path)
                bridge_record = {
                    "boundary_id": f"{shot.shot_id}__{target_shot_id}",
                    "source_shot_id": shot.shot_id,
                    "target_shot_id": target_shot_id,
                    "target_beat_id": chunk.bridge_target_beat_id,
                    "chunk_id": chunk.chunk_id,
                    "path": _portable_path(bridge_path, root),
                    "embedded_in_preceding_shot_output": True,
                    "phase8_transition_policy": "hard_cut_after_generated_camera_bridge",
                }
                with totals_lock:
                    primary_shot_bridges.append(bridge_record)
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
                        "provider_padding_frames": chunk.expected_provider_padding_frames,
                        "provider_padding_normalization": normalization,
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
                    "pre_phase8_policy": "minimum_target_allow_excess",
                    "deferred_cross_shot_prefix": bool(shot.extends_from_chunk_id),
                },
            }
        )
        shot_record = lineage.get_shot(shot.shot_id)
        deferred_prefix_frames = (
            int(shot.chunks[0].expected_overlap_frames)
            if shot.extends_from_chunk_id
            else 0
        )
        if not _valid_record(shot_record, materialization_fingerprint, shot_output):
            materialize_shot(tuple(chunk_paths), shot_output)
            if not shot_output.is_file() or shot_output.stat().st_size == 0:
                raise RuntimeError(f"{shot.shot_id} materialization produced no video bytes")
            closure = None
            if finalize_shot is not None and deferred_prefix_frames <= 0:
                closure = finalize_shot(shot_output, target_frames, plan.timeline_fps)
            elif deferred_prefix_frames > 0:
                closure = {
                    "method": "deferred_phase8_prefix_trim",
                    "before_frames": probe_frames(shot_output, plan.timeline_fps)["frames"]
                    if probe_frames is not None
                    else None,
                    "target_frames": target_frames,
                    "deferred_prefix_frames": deferred_prefix_frames,
                }
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
                "pre_phase8_duration_policy": "minimum_target_allow_excess",
                "minimum_target_met": int(final_timing["frames"]) >= target_frames,
                "excess_frames_before_phase8": max(
                    0, int(final_timing["frames"]) - target_frames
                ),
                "duration_closure": closure,
                "internal_seams_finalized": True,
                "boundary_before": shot.boundary_before,
                "deferred_cross_shot_prefix_frames": deferred_prefix_frames,
            }
            _atomic_write_json(
                root / "shots" / shot.shot_id / "CONTINUITY_TIMING.json",
                timing_receipt,
            )
            with totals_lock:
                timing_manifests += 1
        return ShotExecutionContext(
            shot_id=shot.shot_id,
            output_path=shot_output,
            last_chunk_id=executed_chunk_models[-1].chunk_id,
            output_fingerprint=_canonical_hash(
                {
                    "shot_id": shot.shot_id,
                    "output_sha256": _file_hash(shot_output),
                    "last_chunk_id": executed_chunk_models[-1].chunk_id,
                }
            ),
        )

    groups: list[list[ContinuityShot]] = []
    for shot in plan.shots:
        if shot.extends_from_shot_id:
            if not groups or groups[-1][-1].shot_id != shot.extends_from_shot_id:
                raise RuntimeError(
                    f"{shot.shot_id} cross-shot continuation is not adjacent to "
                    f"{shot.extends_from_shot_id}"
                )
            groups[-1].append(shot)
        else:
            groups.append([shot])

    def run_group(
        group: list[ContinuityShot],
    ) -> tuple[list[str], list[dict[str, str]]]:
        group_outputs: list[str] = []
        group_errors: list[dict[str, str]] = []
        predecessor: ShotExecutionContext | None = None
        for index, shot in enumerate(group):
            try:
                predecessor = run_shot(shot, predecessor)
                group_outputs.append(str(predecessor.output_path.relative_to(root)))
            except Exception as exc:
                if isinstance(exc, ContinuityExecutionPaused):
                    raise
                group_errors.append({"shot_id": shot.shot_id, "error": str(exc)})
                for blocked in group[index + 1 :]:
                    group_errors.append(
                        {
                            "shot_id": blocked.shot_id,
                            "error": f"blocked by failed continuity predecessor {shot.shot_id}",
                        }
                    )
                break
        return group_outputs, group_errors

    if max_new_chunks is not None:
        # A paid acceptance gate must not enqueue work beyond its durable
        # submission allowance.  Run groups serially so an already-submitted
        # future cannot race past the gate while the caller handles the pause.
        for group in groups:
            group_outputs, group_errors = run_group(group)
            outputs.extend(group_outputs)
            errors.extend(group_errors)
    else:
        worker_count = min(max_workers, max(1, len(groups)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(run_group, group): group for group in groups}
            for future in as_completed(futures):
                group = futures[future]
                try:
                    group_outputs, group_errors = future.result()
                    outputs.extend(group_outputs)
                    errors.extend(group_errors)
                except Exception as exc:
                    errors.append({"shot_id": group[0].shot_id, "error": str(exc)})

    outputs.sort()
    errors.sort(key=lambda item: item["shot_id"])
    # Cross-primary bridges are intentionally deferred until every primary
    # video has completed. This guarantees FLF2V uses the actual adjacent
    # outputs rather than a storyboard proxy or an unfinished chunk.
    if not errors:
        shots_by_id = {shot.shot_id: shot for shot in plan.shots}
        for bridge in plan.bridges:
            source_shot = shots_by_id[bridge.source_shot_id]
            source_path = root / "shots" / bridge.source_shot_id / "output.mp4"
            target_path = root / "shots" / bridge.target_shot_id / "output.mp4"
            bridge_path = root / "shot_bridges" / f"{bridge.bridge_id}.mp4"
            bridge_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_id = f"{bridge.bridge_id}_B01"
            bridge_chunk = GenerationChunk(
                chunk_id=chunk_id,
                sequence=1,
                target_duration_s=bridge.target_duration_s,
                requested_frames=bridge.requested_frames,
                expected_unique_frames=bridge.requested_frames,
                mode="native_extend",
                depends_on=source_shot.chunks[-1].chunk_id,
                execution_strategy="first_last_frame_bridge",
                bridge_target_shot_id=bridge.target_shot_id,
                action_prompt=bridge.action_prompt,
                start_state=bridge.start_state,
                end_state=bridge.end_state,
            )
            fingerprint = _canonical_hash(
                {
                    "bridge": bridge.model_dump(mode="json"),
                    "source_video_sha256": _file_hash(source_path),
                    "target_video_sha256": _file_hash(target_path),
                    "source_anchors": source_shot.anchors.model_dump(mode="json"),
                }
            )
            record = lineage.get_chunk(chunk_id)
            try:
                if _valid_record(record, fingerprint, bridge_path):
                    with totals_lock:
                        skipped_chunks += 1
                else:
                    request = ChunkExecutionRequest(
                        resource_id=chunk_id,
                        shot_id=bridge.source_shot_id,
                        chunk=bridge_chunk,
                        anchors=source_shot.anchors.model_dump(mode="json"),
                        output_path=bridge_path,
                        previous_output_path=source_path,
                        target_output_path=target_path,
                        input_fingerprint=fingerprint,
                        memory_context=render_continuity_memory_context(
                            root, plan, bridge.source_shot_id
                        ),
                    )
                    result = execute_chunk(request)
                    produced_path = Path(result.output_path)
                    if produced_path.resolve() != bridge_path.resolve():
                        raise RuntimeError(
                            f"{chunk_id} wrote {produced_path}, expected {bridge_path}"
                        )
                    if not bridge_path.is_file() or bridge_path.stat().st_size == 0:
                        raise RuntimeError(f"{chunk_id} produced no video bytes")
                    lineage.put_chunk(
                        chunk_id,
                        {
                            "status": "succeeded",
                            "input_fingerprint": fingerprint,
                            "output_path": _portable_path(bridge_path, root),
                            "output_sha256": _file_hash(bridge_path),
                            "provider_task_id": result.provider_task_id,
                            "privacy_policy_repair_attempts": len(
                                result.privacy_policy_repairs
                            ),
                            "privacy_policy_repairs": [
                                dict(item) for item in result.privacy_policy_repairs
                            ],
                            "provider_fallback": (
                                dict(result.provider_fallback)
                                if result.provider_fallback is not None
                                else None
                            ),
                            "updated_at": _utc_now(),
                        },
                    )
                    with totals_lock:
                        executed_chunks += 1
                        privacy_policy_repairs += len(
                            result.privacy_policy_repairs
                        )
                bridge_lineage = lineage.get_chunk(chunk_id) or {}
                bridge_fallback = bridge_lineage.get("provider_fallback")
                primary_shot_bridges.append(
                    {
                        "boundary_id": bridge.bridge_id,
                        "source_shot_id": bridge.source_shot_id,
                        "target_shot_id": bridge.target_shot_id,
                        "chunk_id": chunk_id,
                        "duration_s": bridge.target_duration_s,
                        "generation_duration_s": (
                            bridge.generation_duration_s or bridge.target_duration_s
                        ),
                        "visible_duration_s": (
                            bridge.visible_duration_s or bridge.target_duration_s
                        ),
                        "source_handle_s": bridge.source_handle_s,
                        "target_handle_s": bridge.target_handle_s,
                        "timeline_insertion_policy": bridge.timeline_insertion_policy,
                        "path": _portable_path(bridge_path, root),
                        "generated_after_primary_shots": True,
                        "storyboard_transition_image": (
                            bridge.storyboard_transition_image
                        ),
                        "storyboard_transition_usage": (
                            bridge.storyboard_transition_usage
                        ),
                        "first_frame_source": "source_primary_video_tail_frame",
                        "last_frame_source": "target_primary_video_first_frame",
                        "video_endpoint_policy": (
                            "local_actual_boundary_handle_passthrough"
                            if bridge_fallback
                            else "actual_completed_primary_frames_not_storyboard_transition"
                        ),
                        "embedded_in_preceding_shot_output": False,
                        "phase8_transition_policy": (
                            "replace_boundary_handles_with_local_passthrough"
                            if bridge_fallback
                            else "insert_generated_bridge_without_effect"
                        ),
                        "provider_fallback": bridge_fallback,
                    }
                )
            except Exception as exc:
                lineage.put_chunk(
                    chunk_id,
                    {
                        "status": "failed",
                        "input_fingerprint": fingerprint,
                        "error": str(exc),
                        "updated_at": _utc_now(),
                    },
                )
                errors.append({"shot_id": bridge.bridge_id, "error": str(exc)})
                break
    errors.sort(key=lambda item: item["shot_id"])
    primary_shot_bridges.sort(
        key=lambda item: (item["source_shot_id"], item["target_shot_id"])
    )
    bridge_manifest = {
        "kind": PRIMARY_SHOT_BRIDGES_KIND,
        "status": "partial" if errors else "done",
        "count": len(primary_shot_bridges),
        "bridges": primary_shot_bridges,
    }
    _atomic_write_json(root / "PRIMARY_SHOT_BRIDGES.json", bridge_manifest)
    error_summary = None
    if errors:
        details = "; ".join(
            f"{item['shot_id']}: {item['error']}" for item in errors[:6]
        )
        if len(errors) > 6:
            details += f"; and {len(errors) - 6} more"
        error_summary = f"Phase 6 continuity generation failed: {details}"
    return {
        "status": "error" if errors else "done",
        "error": error_summary,
        "outputs": outputs,
        "errors": errors,
        "executed_chunks": executed_chunks,
        "measured_seams": measured_seams,
        "prepared_seams": len(prepared_boundaries),
        "timing_manifests": timing_manifests,
        "duration_topups": duration_topups,
        "repair_attempts": (
            repair_attempts + copyright_policy_repairs + privacy_policy_repairs
        ),
        "seam_repair_attempts": repair_attempts,
        "copyright_policy_repair_attempts": copyright_policy_repairs,
        "privacy_policy_repair_attempts": privacy_policy_repairs,
        "skipped_chunks": skipped_chunks,
        "lineage_path": "CONTINUITY_LINEAGE.json",
        "primary_shot_bridges_path": "PRIMARY_SHOT_BRIDGES.json",
        "bridge_outputs": [item["path"] for item in primary_shot_bridges],
    }
