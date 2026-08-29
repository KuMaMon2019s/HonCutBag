"""Project-scoped registry for immutable, approved character reference packs.

The SQLite database is authoritative for metadata only.  Image and JSON assets
remain ordinary files under the configured library root.  Vector search is
deliberately outside this owner: exact identity reuse is decided by project,
canonical character ID, a versioned static-spec fingerprint, QA receipts, and
content hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quality.character_reference_qa import (
    CHARACTER_REFERENCE_QA_SCHEMA,
    SEEDANCE_REFERENCE_VIEWS,
    file_sha256,
    validate_character_reference_qa_receipt,
)
from tools.character_reference_board import (
    CHARACTER_REFERENCE_BOARD_SCHEMA,
    validate_character_reference_board,
)

CHARACTER_REGISTRY_SCHEMA_VERSION = 1
CHARACTER_SPEC_SCHEMA = "honcut.character-library-spec.v2"
CHARACTER_APPROVAL_SCHEMA = "honcut.character-library-approval.v1"
CHARACTER_REGISTRY_RECEIPT_SCHEMA = "honcut.character-registry-receipt.v1"
CANONICAL_STATUS = "canonical_approved"
CURRENT_REFERENCE_CONTRACT_VERSION = 6


class CharacterRegistryError(RuntimeError):
    """Raised when a character pack is ineligible for promotion or reuse."""


class CharacterRegistryCorruptionError(CharacterRegistryError):
    """Raised when persisted registry evidence is unknown, missing, or changed."""


class CharacterRegistryConflictError(CharacterRegistryError):
    """Raised when one exact spec already names different approved pixels."""


@dataclass(frozen=True)
class ApprovedCharacterAsset:
    role: str
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ApprovedCharacterVersion:
    version_id: str
    project_id: str
    character_id: str
    spec_fingerprint: str
    status: str
    source_run_id: str
    approval_relative_path: str
    approval_sha256: str
    assets: tuple[ApprovedCharacterAsset, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _character_id(value: Any) -> str:
    character_id = str(value or "").strip()
    if (
        not character_id
        or character_id in {".", ".."}
        or "/" in character_id
        or "\\" in character_id
    ):
        raise CharacterRegistryError("character ID must be one safe path component")
    return character_id


def _normalized_approval_assets(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        return None
    required = {"role", "relative_path", "sha256", "size_bytes"}
    if any(not isinstance(item, dict) or set(item) != required for item in payload):
        return None
    return sorted(payload, key=lambda item: item["role"])


def character_has_unapproved_variants(character: Mapping[str, Any]) -> bool:
    """Return whether the character requests state assets outside v1 approval."""
    appearance = character.get("appearance")
    return bool(isinstance(appearance, Mapping) and appearance.get("variants"))


def character_spec_payload(character: Mapping[str, Any]) -> dict[str, Any]:
    """Build the static identity input used for exact approved-pack reuse."""
    appearance = character.get("appearance")
    appearance = appearance if isinstance(appearance, Mapping) else {}
    return {
        "schema": CHARACTER_SPEC_SCHEMA,
        "character_id": _character_id(character.get("id")),
        "description": str(character.get("description") or "").strip(),
        "style": str(character.get("style") or "").strip(),
        "negative": str(character.get("negative") or "").strip(),
        "identity_props": appearance.get("identity_props") or [],
        "synthetic_styling": appearance.get("synthetic_styling"),
        "visual_identity_policy": character.get("visual_identity_policy"),
        "reference_contract_version": CURRENT_REFERENCE_CONTRACT_VERSION,
        "reference_qa_schema": CHARACTER_REFERENCE_QA_SCHEMA,
        "reference_board_schema": CHARACTER_REFERENCE_BOARD_SCHEMA,
    }


def character_spec_fingerprint(character: Mapping[str, Any]) -> str:
    payload = character_spec_payload(character)
    return _sha256_json(payload)


def _resolve_run_file(output_dir: Path, value: str) -> Path:
    root = output_dir.resolve()
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CharacterRegistryError("character asset escapes its run directory") from error
    if not candidate.is_file():
        raise CharacterRegistryError(f"character asset is missing: {value}")
    return candidate


def _asset_records(output_dir: Path, character: Mapping[str, Any]) -> list[tuple[str, Path]]:
    char_id = _character_id(character.get("id"))
    char_dir = output_dir / "characters" / char_id
    card_path = char_dir / "character_card.json"
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CharacterRegistryError("approved character card is missing or invalid") from error
    if card.get("id") != char_id:
        raise CharacterRegistryError("character card ID does not match the canonical character")
    if card.get("reference_contract_version") != CURRENT_REFERENCE_CONTRACT_VERSION:
        raise CharacterRegistryError("character reference contract version is not current")

    declared = card.get("reference_images")
    if not isinstance(declared, dict) or set(SEEDANCE_REFERENCE_VIEWS) - set(declared):
        raise CharacterRegistryError("character card does not declare the canonical four views")
    view_paths = {
        view: _resolve_run_file(output_dir, str(declared[view]))
        for view in SEEDANCE_REFERENCE_VIEWS
    }
    qa_value = str(card.get("reference_qa_report") or "")
    qa_path = _resolve_run_file(output_dir, qa_value)
    synthetic_styling = card.get("synthetic_styling")
    if synthetic_styling is not None and not isinstance(synthetic_styling, dict):
        raise CharacterRegistryError("character card synthetic styling is invalid")
    generation_contract = card.get("reference_generation_contract")
    if not isinstance(generation_contract, dict):
        raise CharacterRegistryError("character reference generation contract is missing")
    prompt_hashes = generation_contract.get("prompt_sha256")
    if (
        generation_contract.get("schema")
        != "honcut.character-reference-generation.v1"
        or generation_contract.get("reference_contract_version")
        != CURRENT_REFERENCE_CONTRACT_VERSION
        or not str(generation_contract.get("model") or "").strip()
        or not isinstance(prompt_hashes, dict)
        or set(SEEDANCE_REFERENCE_VIEWS) - set(prompt_hashes)
        or any(not _is_sha256(prompt_hashes.get(view)) for view in SEEDANCE_REFERENCE_VIEWS)
    ):
        raise CharacterRegistryError("character reference generation contract is invalid")
    if not validate_character_reference_qa_receipt(
        qa_path,
        view_paths,
        synthetic_styling=synthetic_styling,
        generation_contract=generation_contract,
    ):
        raise CharacterRegistryError("character reference QA receipt is missing, stale, or failed")
    if not validate_character_reference_board(char_dir, character_id=char_id):
        raise CharacterRegistryError("character reference board receipt is missing or stale")

    records: list[tuple[str, Path]] = [
        ("character_card", card_path),
        ("angle_map", char_dir / "angle_map.json"),
        ("reference_qa", qa_path),
        *[(view, view_paths[view]) for view in SEEDANCE_REFERENCE_VIEWS],
        ("reference_board", char_dir / "reference_board.png"),
        ("reference_board_receipt", char_dir / "reference_board.json"),
    ]
    identity_props = card.get("identity_props")
    if isinstance(identity_props, list) and identity_props:
        detail_path = _resolve_run_file(
            output_dir, str(card.get("identity_detail_reference") or "")
        )
        detail_qa_path = _resolve_run_file(
            output_dir, str(card.get("identity_detail_qa_report") or "")
        )
        try:
            detail_qa = json.loads(detail_qa_path.read_text(encoding="utf-8"))
            detail_input = detail_qa["inputs"]["identity_detail"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise CharacterRegistryError("identity-detail QA receipt is invalid") from error
        if (
            detail_qa.get("schema") != "honcut.identity-detail-qa.v1"
            or detail_qa.get("status") != "passed"
            or detail_input.get("sha256") != file_sha256(detail_path)
        ):
            raise CharacterRegistryError("identity-detail QA receipt is stale or failed")
        records.extend([("identity_detail", detail_path), ("identity_detail_qa", detail_qa_path)])

    for _role, path in records:
        if not path.is_file():
            raise CharacterRegistryError(f"approved character asset is missing: {path.name}")
        try:
            path.resolve().relative_to(char_dir.resolve())
        except ValueError as error:
            raise CharacterRegistryError(
                "approved character package contains an external asset"
            ) from error
    return records


class CharacterRegistry:
    """Plain-SQLite source of truth for project-scoped approved characters."""

    def __init__(self, root: str | Path, *, project_id: str) -> None:
        if not str(project_id).strip():
            raise ValueError("project_id must not be empty")
        self.root = Path(root).expanduser().resolve()
        self.project_id = str(project_id).strip()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "characters.db"
        self.assets_root = self.root / "assets"
        self._initialize()
        self.assets_root.mkdir(parents=True, exist_ok=True)

    def _connect(self, *, configure_journal: bool = True) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if configure_journal:
            connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect(configure_journal=False) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > CHARACTER_REGISTRY_SCHEMA_VERSION:
                raise CharacterRegistryCorruptionError(
                    f"character registry uses unknown future schema {version}"
                )
            if version == 0:
                existing_tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if existing_tables:
                    raise CharacterRegistryCorruptionError(
                        "unversioned character registry already contains tables"
                    )
                connection.executescript(
                    """
                    CREATE TABLE character_versions (
                        version_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        character_id TEXT NOT NULL,
                        spec_fingerprint TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status = 'canonical_approved'),
                        source_run_id TEXT NOT NULL,
                        approval_relative_path TEXT NOT NULL,
                        approval_sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(project_id, character_id, spec_fingerprint)
                    );
                    CREATE TABLE character_assets (
                        version_id TEXT NOT NULL REFERENCES character_versions(version_id),
                        role TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                        PRIMARY KEY(version_id, role)
                    );
                    CREATE INDEX character_versions_exact
                        ON character_versions(project_id, character_id, spec_fingerprint);
                    PRAGMA user_version = 1;
                    """
                )
            elif version != CHARACTER_REGISTRY_SCHEMA_VERSION:
                raise CharacterRegistryCorruptionError(
                    f"unsupported character registry schema {version}"
                )
            connection.execute("PRAGMA journal_mode = WAL")

    def _load_version(self, version_id: str) -> ApprovedCharacterVersion:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM character_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                raise CharacterRegistryCorruptionError(
                    f"character registry references missing version {version_id}"
                )
            asset_rows = connection.execute(
                "SELECT role, relative_path, sha256, size_bytes "
                "FROM character_assets WHERE version_id = ? ORDER BY role",
                (version_id,),
            ).fetchall()
        if not asset_rows:
            raise CharacterRegistryCorruptionError("approved character version has no assets")
        version = ApprovedCharacterVersion(
            version_id=row["version_id"],
            project_id=row["project_id"],
            character_id=row["character_id"],
            spec_fingerprint=row["spec_fingerprint"],
            status=row["status"],
            source_run_id=row["source_run_id"],
            approval_relative_path=row["approval_relative_path"],
            approval_sha256=row["approval_sha256"],
            assets=tuple(
                ApprovedCharacterAsset(
                    role=item["role"],
                    relative_path=item["relative_path"],
                    sha256=item["sha256"],
                    size_bytes=item["size_bytes"],
                )
                for item in asset_rows
            ),
        )
        self._verify_version(version)
        return version

    def _verify_version(self, version: ApprovedCharacterVersion) -> None:
        if version.project_id != self.project_id or version.status != CANONICAL_STATUS:
            raise CharacterRegistryCorruptionError(
                "approved character identity does not match this project registry"
            )
        if not all(
            _is_sha256(value)
            for value in (
                version.version_id,
                version.spec_fingerprint,
                version.approval_sha256,
            )
        ):
            raise CharacterRegistryCorruptionError(
                "character registry contains an invalid content fingerprint"
            )
        approval = self.root / version.approval_relative_path
        try:
            approval.resolve().relative_to(self.root)
        except ValueError as error:
            raise CharacterRegistryCorruptionError(
                "character approval receipt escapes the library root"
            ) from error
        if not approval.is_file() or file_sha256(approval) != version.approval_sha256:
            raise CharacterRegistryCorruptionError(
                "character approval receipt is missing or hash mismatch"
            )
        try:
            payload = json.loads(approval.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CharacterRegistryCorruptionError(
                "character approval receipt is invalid"
            ) from error
        if (
            payload.get("schema") != CHARACTER_APPROVAL_SCHEMA
            or payload.get("status") != CANONICAL_STATUS
            or payload.get("version_id") != version.version_id
            or payload.get("project_id") != version.project_id
            or payload.get("character_id") != version.character_id
            or payload.get("spec_fingerprint") != version.spec_fingerprint
            or payload.get("source_run_id") != version.source_run_id
            or payload.get("quality_grade") != "A"
            or _sha256_json(payload.get("spec")) != version.spec_fingerprint
        ):
            raise CharacterRegistryCorruptionError(
                "character approval receipt does not match its registry row"
            )
        approval_assets = _normalized_approval_assets(payload.get("assets"))
        expected_assets = [
            {
                "role": asset.role,
                "relative_path": asset.relative_path,
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
            }
            for asset in version.assets
        ]
        if approval_assets != expected_assets:
            raise CharacterRegistryCorruptionError(
                "character approval assets do not match the registry rows"
            )
        for asset in version.assets:
            if not _is_sha256(asset.sha256) or asset.size_bytes <= 0:
                raise CharacterRegistryCorruptionError(
                    "character registry contains invalid asset metadata"
                )
            path = self.root / asset.relative_path
            try:
                path.resolve().relative_to(self.root)
            except ValueError as error:
                raise CharacterRegistryCorruptionError(
                    "character registry asset escapes the library root"
                ) from error
            if not path.is_file():
                raise CharacterRegistryCorruptionError(
                    f"approved character asset is missing: {asset.role}"
                )
            if path.stat().st_size != asset.size_bytes or file_sha256(path) != asset.sha256:
                raise CharacterRegistryCorruptionError(
                    f"approved character asset hash mismatch: {asset.role}"
                )

    def _adopt_existing_package(
        self,
        destination: Path,
        *,
        version_id: str,
        character_id: str,
        spec_fingerprint: str,
        copied_assets: list[dict[str, Any]],
    ) -> tuple[str, str, str]:
        """Validate a crash-left package before creating its missing DB row."""
        approval_path = destination / "approval.json"
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CharacterRegistryConflictError(
                "existing character package has no valid approval receipt"
            ) from error
        expected_assets = sorted(copied_assets, key=lambda item: item["role"])
        if (
            approval.get("schema") != CHARACTER_APPROVAL_SCHEMA
            or approval.get("status") != CANONICAL_STATUS
            or approval.get("quality_grade") != "A"
            or approval.get("version_id") != version_id
            or approval.get("project_id") != self.project_id
            or approval.get("character_id") != character_id
            or approval.get("spec_fingerprint") != spec_fingerprint
            or _sha256_json(approval.get("spec")) != spec_fingerprint
            or _normalized_approval_assets(approval.get("assets")) != expected_assets
            or not str(approval.get("source_run_id") or "").strip()
        ):
            raise CharacterRegistryConflictError(
                "existing character package conflicts with this exact approval"
            )
        for asset in expected_assets:
            path = self.root / asset["relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != asset["size_bytes"]
                or file_sha256(path) != asset["sha256"]
            ):
                raise CharacterRegistryConflictError(
                    "existing character package contains different pixels"
                )
        return (
            approval_path.relative_to(self.root).as_posix(),
            file_sha256(approval_path),
            str(approval["source_run_id"]),
        )

    def find_exact(self, character: Mapping[str, Any]) -> ApprovedCharacterVersion | None:
        if character_has_unapproved_variants(character):
            return None
        char_id = _character_id(character.get("id"))
        spec_fingerprint = character_spec_fingerprint(character)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version_id FROM character_versions "
                "WHERE project_id = ? AND character_id = ? AND spec_fingerprint = ?",
                (self.project_id, char_id, spec_fingerprint),
            ).fetchone()
        return self._load_version(row["version_id"]) if row is not None else None

    def promote_from_run(
        self,
        output_dir: str | Path,
        character: Mapping[str, Any],
        *,
        quality_grade: str,
        source_run_id: str,
    ) -> ApprovedCharacterVersion:
        if quality_grade != "A":
            raise CharacterRegistryError(
                "only Phase 3 grade A character packs are eligible for promotion"
            )
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise CharacterRegistryError("source run ID is required for promotion")
        if character_has_unapproved_variants(character):
            raise CharacterRegistryError(
                "character packs with unapproved state variants cannot be promoted in v1"
            )
        output_dir = Path(output_dir).resolve()
        char_id = _character_id(character.get("id"))
        spec_fingerprint = character_spec_fingerprint(character)
        source_assets = _asset_records(output_dir, character)
        asset_identity = [
            {"role": role, "filename": path.name, "sha256": file_sha256(path)}
            for role, path in sorted(source_assets)
        ]
        version_id = _sha256_json(
            {
                "project_id": self.project_id,
                "character_id": char_id,
                "spec_fingerprint": spec_fingerprint,
                "assets": asset_identity,
            }
        )

        existing = self.find_exact(character)
        if existing is not None:
            if existing.version_id != version_id:
                raise CharacterRegistryConflictError(
                    "the exact character spec already has different approved pixels"
                )
            return existing

        project_key = hashlib.sha256(self.project_id.encode("utf-8")).hexdigest()[:16]
        relative_package = Path("assets") / project_key / version_id
        destination = self.root / relative_package
        temporary = Path(tempfile.mkdtemp(prefix=f".{version_id[:12]}.", dir=self.assets_root))
        try:
            copied_assets = []
            for role, source in source_assets:
                target = temporary / source.name
                shutil.copy2(source, target)
                copied_assets.append(
                    {
                        "role": role,
                        "relative_path": (relative_package / source.name).as_posix(),
                        "sha256": file_sha256(target),
                        "size_bytes": target.stat().st_size,
                    }
                )
            approval = {
                "schema": CHARACTER_APPROVAL_SCHEMA,
                "status": CANONICAL_STATUS,
                "version_id": version_id,
                "project_id": self.project_id,
                "character_id": char_id,
                "spec_fingerprint": spec_fingerprint,
                "spec": character_spec_payload(character),
                "source_run_id": source_run_id,
                "quality_grade": quality_grade,
                "assets": copied_assets,
            }
            approval_path = temporary / "approval.json"
            approval_path.write_text(
                json.dumps(approval, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                approval_relative, approval_sha256, source_run_id = self._adopt_existing_package(
                    destination,
                    version_id=version_id,
                    character_id=char_id,
                    spec_fingerprint=spec_fingerprint,
                    copied_assets=copied_assets,
                )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(temporary, destination)
                except OSError:
                    if not destination.exists():
                        raise
                approval_relative = (relative_package / "approval.json").as_posix()
                if destination.exists() and temporary.exists():
                    approval_relative, approval_sha256, source_run_id = (
                        self._adopt_existing_package(
                            destination,
                            version_id=version_id,
                            character_id=char_id,
                            spec_fingerprint=spec_fingerprint,
                            copied_assets=copied_assets,
                        )
                    )
                else:
                    approval_sha256 = file_sha256(destination / "approval.json")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO character_versions "
                    "(version_id, project_id, character_id, spec_fingerprint, status, "
                    "source_run_id, approval_relative_path, approval_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        version_id,
                        self.project_id,
                        char_id,
                        spec_fingerprint,
                        CANONICAL_STATUS,
                        source_run_id,
                        approval_relative,
                        approval_sha256,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                connection.executemany(
                    "INSERT INTO character_assets "
                    "(version_id, role, relative_path, sha256, size_bytes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            version_id,
                            item["role"],
                            item["relative_path"],
                            item["sha256"],
                            item["size_bytes"],
                        )
                        for item in copied_assets
                    ],
                )
        except sqlite3.IntegrityError as error:
            existing = self.find_exact(character)
            if existing is not None and existing.version_id == version_id:
                return existing
            raise CharacterRegistryConflictError(
                "character promotion collided with another approved version"
            ) from error
        return self._load_version(version_id)

    def import_into_run(
        self,
        version: ApprovedCharacterVersion,
        output_dir: str | Path,
    ) -> Path:
        self._verify_version(version)
        output_dir = Path(output_dir).resolve()
        destination = output_dir / "characters" / version.character_id
        expected = {Path(asset.relative_path).name: asset for asset in version.assets}
        if len(expected) != len(version.assets):
            raise CharacterRegistryCorruptionError(
                "approved character package contains duplicate filenames"
            )
        if destination.exists():
            for filename, asset in expected.items():
                path = destination / filename
                if not path.is_file() or file_sha256(path) != asset.sha256:
                    raise CharacterRegistryConflictError(
                        "run-local character directory conflicts with approved reuse"
                    )
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{version.character_id}.registry-import.",
                dir=destination.parent,
            )
        )
        try:
            for filename, asset in expected.items():
                source = self.root / asset.relative_path
                shutil.copy2(source, temporary / filename)
            for filename, asset in expected.items():
                if file_sha256(temporary / filename) != asset.sha256:
                    raise CharacterRegistryCorruptionError(
                        f"copied character asset hash mismatch: {asset.role}"
                    )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination


__all__ = [
    "CANONICAL_STATUS",
    "CHARACTER_APPROVAL_SCHEMA",
    "CHARACTER_REGISTRY_RECEIPT_SCHEMA",
    "CHARACTER_REGISTRY_SCHEMA_VERSION",
    "ApprovedCharacterAsset",
    "ApprovedCharacterVersion",
    "CharacterRegistry",
    "CharacterRegistryConflictError",
    "CharacterRegistryCorruptionError",
    "CharacterRegistryError",
    "character_has_unapproved_variants",
    "character_spec_fingerprint",
    "character_spec_payload",
]
