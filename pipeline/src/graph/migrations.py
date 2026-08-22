"""Deterministic migrations for checkpoint-safe HonCut graph state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Command
from runtime.migration_registry import apply_migration_registry

CURRENT_STATE_SCHEMA_VERSION = 1
LEGACY_STATE_ALIASES = frozenset(
    {
        "text",
        "duration",
        "shot_duration",
        "transition_duration",
        "shots",
        "videos",
        "quality_report",
        "error",
    }
)


class StateMigrationError(ValueError):
    """Raised when a checkpoint cannot be migrated without guessing."""


def _version_of(state: Mapping[str, Any]) -> int:
    raw_version = state.get("state_schema_version", 0)
    if isinstance(raw_version, bool):
        raise StateMigrationError("state_schema_version must be an integer")
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise StateMigrationError("state_schema_version must be an integer") from exc
    if version < 0:
        raise StateMigrationError("state_schema_version must not be negative")
    if version > CURRENT_STATE_SCHEMA_VERSION:
        raise StateMigrationError(
            "checkpoint state schema version "
            f"{version} is newer than supported version "
            f"{CURRENT_STATE_SCHEMA_VERSION}"
        )
    return version


def _shot_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    identifiers: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            candidate = item.get("shot_id", item.get("id"))
        else:
            candidate = item
        if candidate is not None and str(candidate).strip():
            identifiers.append(str(candidate))
    return identifiers


def _canonicalize_v1_state(state: dict[str, Any]) -> dict[str, Any]:
    alias_values = {
        "input_text": state.get("text", ""),
        "target_duration_s": state.get("duration", 60),
        "shot_duration_s": state.get("shot_duration", 5),
        "transition_duration_s": state.get("transition_duration", 0.5),
        "shot_ids": _shot_ids(state.get("shots", [])),
        "generated_shots": state.get("videos", []),
        "consistency": state.get("quality_report", {}),
    }
    for canonical_name, value in alias_values.items():
        state.setdefault(canonical_name, value)

    legacy_error = state.get("error")
    errors = state.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    if legacy_error and not errors:
        errors = [{"category": "legacy", "message": str(legacy_error)}]
    state["errors"] = errors

    state.setdefault("project_id", "local")
    state.setdefault("run_id", state.get("run_fingerprint", "pipeline_run"))
    state["state_schema_version"] = CURRENT_STATE_SCHEMA_VERSION
    for alias in LEGACY_STATE_ALIASES:
        state.pop(alias, None)
    return state


def _migrate_state_v0_to_v1(state: dict[str, Any]) -> dict[str, Any]:
    return _canonicalize_v1_state(state)


STATE_MIGRATIONS = {0: _migrate_state_v0_to_v1}


def migrate_state(raw_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical v1 state through the explicit migration registry."""

    if not isinstance(raw_state, Mapping):
        raise StateMigrationError("checkpoint state must be an object")
    _version_of(raw_state)
    try:
        state = apply_migration_registry(
            raw_state,
            current_version=CURRENT_STATE_SCHEMA_VERSION,
            migrations=STATE_MIGRATIONS,
            document_name="checkpoint state schema",
            version_field="state_schema_version",
        )
    except ValueError as error:
        raise StateMigrationError(str(error)) from error
    return _canonicalize_v1_state(state)


def _canonical_patch(
    raw_update: Mapping[str, Any],
    previous_state: Mapping[str, Any],
    *,
    quality_target: str = "consistency",
) -> dict[str, Any]:
    update = dict(raw_update)
    if "videos" in update:
        update.setdefault("generated_shots", update["videos"])
    if "quality_report" in update:
        update.setdefault(quality_target, update["quality_report"])
    if update.get("error"):
        prior_errors = previous_state.get("errors", [])
        if not isinstance(prior_errors, list):
            prior_errors = []
        update.setdefault(
            "errors",
            [
                *prior_errors,
                {"category": "workflow", "message": str(update["error"])},
            ],
        )
    for alias in LEGACY_STATE_ALIASES:
        update.pop(alias, None)
    return update


def canonicalize_node_result(
    result: dict[str, Any] | Command,
    previous_state: Mapping[str, Any],
    *,
    quality_target: str = "consistency",
) -> dict[str, Any] | Command:
    """Strip compatibility aliases from a production node result."""

    if isinstance(result, Command):
        update = result.update
        if not isinstance(update, Mapping):
            return result
        return Command(
            graph=result.graph,
            update=_canonical_patch(
                update,
                previous_state,
                quality_target=quality_target,
            ),
            resume=result.resume,
            goto=result.goto,
        )
    return _canonical_patch(
        result,
        previous_state,
        quality_target=quality_target,
    )


def latest_error_message(state: Mapping[str, Any]) -> str | None:
    errors = state.get("errors", [])
    if not isinstance(errors, list) or not errors:
        return None
    last = errors[-1]
    if isinstance(last, Mapping) and last.get("message"):
        return str(last["message"])
    return str(last) if last else None


__all__ = [
    "CURRENT_STATE_SCHEMA_VERSION",
    "LEGACY_STATE_ALIASES",
    "STATE_MIGRATIONS",
    "StateMigrationError",
    "canonicalize_node_result",
    "latest_error_message",
    "migrate_state",
]
