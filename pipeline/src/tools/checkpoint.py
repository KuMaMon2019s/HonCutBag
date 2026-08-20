"""Atomic HonCut stage checkpoint persistence."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CheckpointValidationError(ValueError):
    pass


def validate_checkpoint(value: dict[str, Any]) -> None:
    if not isinstance(value.get("completed"), list):
        raise CheckpointValidationError("completed must be a list")
    if not isinstance(value.get("results"), dict):
        raise CheckpointValidationError("results must be an object")


def read_checkpoint(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    value = json.loads(target.read_text(encoding="utf-8"))
    validate_checkpoint(value)
    return value


def write_checkpoint(
    path: str | Path, stage: str, result: dict[str, Any], *, run_fingerprint: str | None = None
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = read_checkpoint(target) or {"completed": [], "results": {}, "timestamp": ""}
    if run_fingerprint is not None and value.get("run_fingerprint") not in (None, run_fingerprint):
        value = {
            "completed": [],
            "results": {},
            "timestamp": "",
            "run_fingerprint": run_fingerprint,
        }
    if stage not in value["completed"]:
        value["completed"].append(stage)
    value["results"][stage] = result
    value["timestamp"] = datetime.now(UTC).isoformat()
    if run_fingerprint is not None:
        value["run_fingerprint"] = run_fingerprint
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return value


def invalidate_checkpoint_from(path: str | Path, boundary: str, stages: list[str]) -> list[str]:
    """Atomically mark the boundary and every downstream stage incomplete."""
    if boundary not in stages:
        raise ValueError(f"unknown checkpoint boundary: {boundary}")
    target = Path(path)
    value = read_checkpoint(target)
    if value is None:
        return []
    stale = set(stages[stages.index(boundary) :])
    invalidated = [stage for stage in value["completed"] if stage in stale]
    value["completed"] = [stage for stage in value["completed"] if stage not in stale]
    for stage in stale:
        value["results"].pop(stage, None)
    value.setdefault("invalidations", []).append(
        {
            "resume_from": boundary,
            "invalidated": invalidated,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )
    value["timestamp"] = datetime.now(UTC).isoformat()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return invalidated


def get_next_stage(path: str | Path, stages: list[str]) -> str | None:
    value = read_checkpoint(path) or {"completed": []}
    completed = set(value["completed"])
    return next((stage for stage in stages if stage not in completed), None)
