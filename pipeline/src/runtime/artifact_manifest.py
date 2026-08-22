"""Atomic filesystem artifact manifests bound to one project and run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from runtime.artifact_migrations import migrate_artifact_manifest
from runtime.security_boundaries import resolve_within_workspace
from schemas.artifact import ArtifactManifest, ArtifactRef


ARTIFACT_MANIFEST_FILENAME = "ARTIFACT_MANIFEST.json"


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _artifact_id(
    *,
    project_id: str,
    run_id: str,
    artifact_type: str,
    relative_path: str,
    content_sha256: str,
    semantic_fingerprint: str | None,
) -> str:
    identity = {
        "project_id": project_id,
        "run_id": run_id,
        "type": artifact_type,
        "relative_path": relative_path,
        "content_sha256": content_sha256,
    }
    if semantic_fingerprint is not None:
        identity["semantic_fingerprint"] = semantic_fingerprint
    semantic_identity = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "artifact_" + hashlib.sha256(semantic_identity).hexdigest()


class ArtifactManifestStore:
    """Register and resolve immutable artifact metadata over local files."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        run_id: str,
        project_id: str,
    ) -> None:
        if not run_id.strip() or not project_id.strip():
            raise ValueError("run_id and project_id must not be empty")
        self.run_directory = Path(run_directory).resolve()
        self.run_id = run_id
        self.project_id = project_id
        self.path = self.run_directory / ARTIFACT_MANIFEST_FILENAME

    @classmethod
    def from_run_directory(
        cls,
        run_directory: str | Path,
        *,
        required: bool = True,
    ) -> "ArtifactManifestStore | None":
        root = Path(run_directory)
        run_manifest_path = root / "RUN_MANIFEST.json"
        if not run_manifest_path.is_file():
            if required:
                raise RuntimeError("RUN_MANIFEST.json is required for artifact identity")
            return None
        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_id = run_manifest["run_fingerprint"]
            project_id = run_manifest["resolved_config"]["project_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("RUN_MANIFEST.json has no valid artifact identity") from error
        if not isinstance(run_id, str) or not isinstance(project_id, str):
            raise RuntimeError("RUN_MANIFEST.json artifact identity must be textual")
        return cls(root, run_id=run_id, project_id=project_id)

    def load(self) -> ArtifactManifest:
        if not self.path.is_file():
            return ArtifactManifest(run_id=self.run_id, project_id=self.project_id)
        try:
            raw_manifest = json.loads(self.path.read_text(encoding="utf-8"))
            migrated = migrate_artifact_manifest(raw_manifest)
            manifest = ArtifactManifest.model_validate_json(
                json.dumps(migrated, ensure_ascii=False)
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid artifact manifest: {error}") from error
        if manifest.run_id != self.run_id or manifest.project_id != self.project_id:
            raise RuntimeError("artifact manifest belongs to a different project or run")
        return manifest

    def register_file(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        producer_node: str,
        producer_task_id: str | None = None,
        parent_artifact_ids: Iterable[str] = (),
        expected_sha256: str | None = None,
        semantic_fingerprint: str | None = None,
    ) -> ArtifactRef:
        source = resolve_within_workspace(
            self.run_directory,
            path,
            must_exist=True,
        )
        relative_path = source.relative_to(self.run_directory).as_posix()
        if not source.is_file():
            raise ValueError("only regular artifact files can be registered")
        content_sha256 = file_sha256(source)
        if expected_sha256 is not None and content_sha256 != expected_sha256:
            raise RuntimeError("artifact content hash does not match the expected hash")
        parents = tuple(parent_artifact_ids)
        manifest = self.load()
        known = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
        missing = set(parents) - set(known)
        if missing:
            raise RuntimeError(
                "artifact registration references missing parents: "
                + ", ".join(sorted(missing))
            )
        artifact_id = _artifact_id(
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            content_sha256=content_sha256,
            semantic_fingerprint=semantic_fingerprint,
        )
        existing = known.get(artifact_id)
        if existing is not None:
            if (
                existing.producer_node != producer_node
                or existing.producer_task_id != producer_task_id
                or existing.parent_artifact_ids != parents
                or existing.semantic_fingerprint != semantic_fingerprint
            ):
                raise RuntimeError("artifact ID collides with different provenance")
            return self.resolve(existing.artifact_id)
        artifact = ArtifactRef(
            artifact_id=artifact_id,
            run_id=self.run_id,
            project_id=self.project_id,
            type=artifact_type,
            content_sha256=content_sha256,
            semantic_fingerprint=semantic_fingerprint,
            relative_path=relative_path,
            producer_node=producer_node,
            producer_task_id=producer_task_id,
            parent_artifact_ids=parents,
            created_at=datetime.now(UTC),
        )
        updated = ArtifactManifest(
            run_id=self.run_id,
            project_id=self.project_id,
            artifacts=(*manifest.artifacts, artifact),
        )
        self._write_atomic(updated)
        return artifact

    def resolve(self, artifact_id: str, *, verify_content: bool = True) -> ArtifactRef:
        manifest = self.load()
        artifact = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            raise KeyError(f"unknown artifact ID: {artifact_id}")
        if verify_content:
            try:
                path = resolve_within_workspace(
                    self.run_directory,
                    artifact.relative_path,
                )
            except ValueError as error:
                raise RuntimeError("artifact path escapes its run directory") from error
            if not path.is_file():
                raise RuntimeError(f"artifact file is missing: {artifact.relative_path}")
            if file_sha256(path) != artifact.content_sha256:
                raise RuntimeError(
                    f"artifact content hash mismatch: {artifact.relative_path}"
                )
        return artifact

    def _write_atomic(self, manifest: ArtifactManifest) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{ARTIFACT_MANIFEST_FILENAME}.",
                suffix=".tmp",
                dir=self.run_directory,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(
                    json.dumps(
                        manifest.model_dump(mode="json", by_alias=True),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ArtifactManifestStore",
    "file_sha256",
]
