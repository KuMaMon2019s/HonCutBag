"""Immutable identity for one HonCut pipeline run and all of its caches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA = "honcut.run-manifest.v1"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_version(repo_root: Path) -> str:
    commit = "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        commit = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    digest = hashlib.sha256()
    source_paths = []
    for source_root in (repo_root / "pipeline" / "src", repo_root / "vendor" / "legacy"):
        if source_root.is_dir():
            source_paths.extend(source_root.rglob("*.py"))
    for extra in (repo_root / "pyproject.toml", repo_root / "pipeline" / "config.yaml"):
        if extra.is_file():
            source_paths.append(extra)
    for path in sorted(set(source_paths)):
        try:
            digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            continue
    return f"{commit}:{digest.hexdigest()}"


def prepare_run_manifest(
    output_dir: str | Path,
    *,
    source_text: str,
    resolved_config: dict[str, Any],
    repo_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Create a run identity or reject a resume whose immutable inputs changed."""
    root = Path(output_dir)
    path = root / "RUN_MANIFEST.json"
    existing = None
    if path.is_file():
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                existing = candidate
        except (OSError, json.JSONDecodeError):
            existing = None

    if resume and existing is None:
        raise RuntimeError(
            "resume refused: RUN_MANIFEST.json is missing or invalid; "
            "the checkpoint has no trustworthy input identity"
        )

    input_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if resume and not source_text:
        input_sha256 = str(existing.get("input_sha256") or "")
        if not input_sha256:
            raise RuntimeError("resume refused: stored input fingerprint is missing")

    normalized_config = json.loads(
        json.dumps(resolved_config, ensure_ascii=False, sort_keys=True, default=str)
    )
    identity = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "input_sha256": input_sha256,
        "config_sha256": _sha256_json(normalized_config),
        "provider": normalized_config.get("video_provider"),
        "model": normalized_config.get("video_model"),
        "project_video_spec": normalized_config.get("project_video_spec"),
        "code_version": _code_version(Path(repo_root)),
    }
    identity["run_fingerprint"] = _sha256_json(identity)

    if resume:
        mismatches = {
            key: {"stored": existing.get(key), "current": identity.get(key)}
            for key in (
                "schema_version",
                "input_sha256",
                "config_sha256",
                "provider",
                "model",
                "project_video_spec",
                "code_version",
                "run_fingerprint",
            )
            if existing.get(key) != identity.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "resume refused: immutable run identity changed: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        return existing

    manifest = {**identity, "resolved_config": normalized_config}
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return manifest
