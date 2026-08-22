"""Explicit, deterministic registries for persisted schema upgrades."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


Migration = Callable[[dict[str, Any]], dict[str, Any]]


class MigrationRegistryError(ValueError):
    """Raised when a persisted document cannot be upgraded exactly."""


def apply_migration_registry(
    raw_document: Mapping[str, Any],
    *,
    current_version: int,
    migrations: Mapping[int, Migration],
    document_name: str,
    version_field: str = "schema_version",
) -> dict[str, Any]:
    if not isinstance(raw_document, Mapping):
        raise MigrationRegistryError(f"{document_name} must be an object")
    document = deepcopy(dict(raw_document))
    raw_version = document.get(version_field, 0)
    if isinstance(raw_version, bool):
        raise MigrationRegistryError(f"{document_name} version must be an integer")
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as error:
        raise MigrationRegistryError(
            f"{document_name} version must be an integer"
        ) from error
    if version < 0:
        raise MigrationRegistryError(f"{document_name} version must not be negative")
    if version > current_version:
        raise MigrationRegistryError(
            f"{document_name} version {version} is newer than supported "
            f"version {current_version}"
        )
    while version < current_version:
        migrate = migrations.get(version)
        if migrate is None:
            raise MigrationRegistryError(
                f"{document_name} has no migration from version {version}"
            )
        document = migrate(document)
        next_version = document.get(version_field)
        if next_version != version + 1:
            raise MigrationRegistryError(
                f"{document_name} migration {version} must produce version "
                f"{version + 1}"
            )
        version += 1
    return document


__all__ = [
    "Migration",
    "MigrationRegistryError",
    "apply_migration_registry",
]
