"""One fail-closed precedence rule for HonCut resume state."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph.migrations import StateMigrationError, migrate_state
from utils.artifact_chain import PHASE_SEQUENCE

STAGE_CHECKPOINT_SCHEMA_VERSION = 1


class ResumeResolutionError(RuntimeError):
    """Raised when persisted resume evidence is present but untrustworthy."""


@dataclass(frozen=True)
class ResumeSnapshot:
    source: str
    state: dict[str, Any]
    completed_phases: tuple[str, ...]
    phase_results: dict[str, dict[str, Any]]


def _ordered_completed(raw_completed: Any) -> list[str]:
    if not isinstance(raw_completed, list):
        raise ResumeResolutionError("checkpoint completed phases must be a list")
    unknown = [phase for phase in raw_completed if phase not in PHASE_SEQUENCE]
    if unknown:
        raise ResumeResolutionError(
            "checkpoint contains unknown completed phases: " + ", ".join(map(str, unknown))
        )
    completed = set(raw_completed)
    return [phase for phase in PHASE_SEQUENCE if phase in completed]


def _validated_phase_results(raw_results: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_results, Mapping):
        raise ResumeResolutionError("checkpoint phase results must be an object")
    results: dict[str, dict[str, Any]] = {}
    for phase, value in raw_results.items():
        if phase not in PHASE_SEQUENCE or not isinstance(value, Mapping):
            continue
        results[str(phase)] = dict(value)
    return results


def _apply_artifact_receipts(
    output_dir: Path,
    completed: list[str],
) -> list[str]:
    """Honor explicit stale/failed receipts without rejecting legacy absence."""

    valid = list(completed)
    for phase in completed:
        path = output_dir / f"checkpoint_{phase}.json"
        if not path.is_file():
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeResolutionError(
                f"artifact receipt for {phase} is unreadable"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise ResumeResolutionError(
                f"artifact receipt for {phase} must be an object"
            )
        if receipt.get("phase") not in (None, phase):
            raise ResumeResolutionError(
                f"artifact receipt phase mismatch for {phase}"
            )
        if receipt.get("status") != "done":
            boundary = PHASE_SEQUENCE.index(phase)
            valid = [
                candidate
                for candidate in valid
                if PHASE_SEQUENCE.index(candidate) < boundary
            ]
            break
    return valid


def _validate_identity(
    state: Mapping[str, Any],
    *,
    run_fingerprint: str,
    project_id: str,
    source: str,
) -> None:
    stored_fingerprint = state.get("run_fingerprint", state.get("run_id"))
    if stored_fingerprint != run_fingerprint:
        raise ResumeResolutionError(
            f"{source} run fingerprint does not match RUN_MANIFEST.json"
        )
    if state.get("project_id", "local") != project_id:
        raise ResumeResolutionError(
            f"{source} project_id does not match requested project {project_id!r}"
        )


def _from_graph_state(
    output_dir: Path,
    source: str,
    raw_state: Mapping[str, Any],
    *,
    run_fingerprint: str,
    project_id: str,
) -> ResumeSnapshot:
    graph_state = dict(raw_state)
    if "completed_phases" not in graph_state and "completed" in graph_state:
        graph_state["completed_phases"] = graph_state.get("completed", [])
    if "phase_results" not in graph_state and "results" in graph_state:
        graph_state["phase_results"] = graph_state.get("results", {})
    try:
        state = migrate_state(graph_state)
    except StateMigrationError as exc:
        raise ResumeResolutionError(f"{source} cannot be migrated: {exc}") from exc
    _validate_identity(
        state,
        run_fingerprint=run_fingerprint,
        project_id=project_id,
        source=source,
    )
    completed = _ordered_completed(state.get("completed_phases", []))
    completed = _apply_artifact_receipts(output_dir, completed)
    phase_results = _validated_phase_results(state.get("phase_results", {}))
    state["completed_phases"] = completed
    state["phase_results"] = {
        phase: phase_results[phase]
        for phase in completed
        if phase in phase_results
    }
    return ResumeSnapshot(
        source=source,
        state=state,
        completed_phases=tuple(completed),
        phase_results=state["phase_results"],
    )


def _from_json_checkpoint(
    output_dir: Path,
    *,
    run_fingerprint: str,
    project_id: str,
) -> ResumeSnapshot | None:
    path = output_dir / "checkpoint.json"
    if not path.is_file():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeResolutionError("checkpoint.json is unreadable") from exc
    if not isinstance(checkpoint, Mapping):
        raise ResumeResolutionError("checkpoint.json must contain an object")
    raw_version = checkpoint.get("checkpoint_schema_version", 0)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ResumeResolutionError("checkpoint_schema_version must be an integer")
    if raw_version > STAGE_CHECKPOINT_SCHEMA_VERSION:
        raise ResumeResolutionError(
            "checkpoint schema version "
            f"{raw_version} is newer than supported version "
            f"{STAGE_CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("run_fingerprint") != run_fingerprint:
        raise ResumeResolutionError(
            "checkpoint.json run fingerprint does not match RUN_MANIFEST.json"
        )
    if checkpoint.get("project_id", "local") != project_id:
        raise ResumeResolutionError(
            "checkpoint.json project_id does not match requested project "
            f"{project_id!r}"
        )
    completed = _ordered_completed(checkpoint.get("completed"))
    completed = _apply_artifact_receipts(output_dir, completed)
    phase_results = _validated_phase_results(checkpoint.get("results", {}))
    state = migrate_state(
        {
            "state_schema_version": 1,
            "run_id": run_fingerprint,
            "run_fingerprint": run_fingerprint,
            "project_id": project_id,
            "completed_phases": completed,
            "phase_results": {
                phase: phase_results[phase]
                for phase in completed
                if phase in phase_results
            },
        }
    )
    return ResumeSnapshot(
        source="json-stage",
        state=state,
        completed_phases=tuple(completed),
        phase_results=state["phase_results"],
    )


def resolve_resume_snapshot(
    output_dir: str | Path,
    *,
    run_fingerprint: str,
    project_id: str,
    graph_states: Iterable[tuple[str, Mapping[str, Any] | None]] = (),
) -> ResumeSnapshot:
    """Resolve graph → artifact receipts → JSON stage evidence in that order."""

    root = Path(output_dir)
    for source, state in graph_states:
        if state:
            return _from_graph_state(
                root,
                source,
                state,
                run_fingerprint=run_fingerprint,
                project_id=project_id,
            )
    json_snapshot = _from_json_checkpoint(
        root,
        run_fingerprint=run_fingerprint,
        project_id=project_id,
    )
    if json_snapshot is not None:
        return json_snapshot
    empty_state = migrate_state(
        {
            "state_schema_version": 1,
            "run_id": run_fingerprint,
            "run_fingerprint": run_fingerprint,
            "project_id": project_id,
            "completed_phases": [],
            "phase_results": {},
        }
    )
    return ResumeSnapshot(
        source="fresh",
        state=empty_state,
        completed_phases=(),
        phase_results={},
    )


__all__ = [
    "ResumeResolutionError",
    "ResumeSnapshot",
    "STAGE_CHECKPOINT_SCHEMA_VERSION",
    "resolve_resume_snapshot",
]
