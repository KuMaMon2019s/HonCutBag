"""Immutable identity for one HonCut pipeline run and all of its caches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA = "honcut.run-manifest.v1"
CODE_CHANGE_ACCEPTANCE_SCHEMA = "honcut.code-change-acceptance.v1"

_SEMANTIC_IDENTITY_KEYS = (
    "schema_version",
    "input_sha256",
    "config_sha256",
    "provider",
    "model",
    "project_video_spec",
)

_RUN_OWNED_MARKERS = (
    "STORYBOARD.json",
    "CHARACTERS.json",
    "PROJECT_VIDEO_SPEC.json",
    "director_plan.json",
    "director_storyboard.json",
    "director_storyboard.png",
    "storyboard.png",
    "visual-style.md",
    "visual_style_spec.md",
    "phase1_events.json",
    "phase1_characters.json",
    "beat_skeleton.json",
    "shots_partial.json",
    "checkpoint.json",
    "checkpoint.db",
    "runtime.db",
    "ARTIFACT_MANIFEST.json",
    "shots",
    "director_panels",
    "shot_storyboards",
    "storyboard_groups",
    "storyboard_images",
    "storyboard_beats",
    "video_first_frames",
    "characters",
    "scenes",
    "audio",
    "audio_layer",
    "bgm.mp3",
    "bgm.wav",
    "bg_music.mp3",
    "background_music.mp3",
    "music.mp3",
    "soundtrack.mp3",
    "ost.mp3",
    "raw_assembly.mp4",
    "polished.mp4",
)


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
    source_root = repo_root / "pipeline" / "src"
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


def _identity_with_code(manifest: dict[str, Any], code_version: str) -> dict[str, Any]:
    return {
        **{key: manifest.get(key) for key in _SEMANTIC_IDENTITY_KEYS},
        "code_version": code_version,
    }


def _validated_code_history(
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Verify the stable run fingerprint and every admitted code transition."""
    origin_code_version = manifest.get(
        "origin_code_version",
        manifest.get("code_version"),
    )
    if not isinstance(origin_code_version, str) or not origin_code_version:
        raise RuntimeError(
            "resume refused: stored RUN_MANIFEST.json origin code version is invalid"
        )
    expected_fingerprint = _sha256_json(
        _identity_with_code(manifest, origin_code_version)
    )
    run_fingerprint = manifest.get("run_fingerprint")
    if run_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "resume refused: stored RUN_MANIFEST.json fingerprint is invalid"
        )

    history = manifest.get("code_change_history", [])
    if not isinstance(history, list):
        raise RuntimeError(
            "resume refused: stored RUN_MANIFEST.json code change history is invalid"
        )
    admitted_code_version = origin_code_version
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(
                "resume refused: stored RUN_MANIFEST.json code change history "
                f"entry {index} is invalid"
            )
        valid_entry = (
            entry.get("schema_version") == CODE_CHANGE_ACCEPTANCE_SCHEMA
            and entry.get("acceptance") == "explicit_cli_flag"
            and entry.get("from_code_version") == admitted_code_version
            and isinstance(entry.get("to_code_version"), str)
            and bool(entry.get("to_code_version"))
            and isinstance(entry.get("resume_from"), str)
            and bool(entry.get("resume_from"))
            and entry.get("run_fingerprint") == run_fingerprint
            and isinstance(entry.get("accepted_at"), str)
            and bool(entry.get("accepted_at"))
        )
        if not valid_entry:
            raise RuntimeError(
                "resume refused: stored RUN_MANIFEST.json code change history "
                f"entry {index} is invalid"
            )
        admitted_code_version = entry["to_code_version"]
    if admitted_code_version != manifest.get("code_version"):
        raise RuntimeError(
            "resume refused: stored RUN_MANIFEST.json code version is not backed "
            "by its acceptance history"
        )
    return origin_code_version, history


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_run_manifest(
    output_dir: str | Path,
    *,
    source_text: str,
    resolved_config: dict[str, Any],
    repo_root: str | Path,
    resume: bool,
    accepted_code_change_from: str | None = None,
) -> dict[str, Any]:
    """Create a run identity or reject a resume whose immutable inputs changed."""
    if accepted_code_change_from is not None:
        if not resume:
            raise RuntimeError("code change acceptance requires resume mode")
        if not isinstance(accepted_code_change_from, str) or not (
            accepted_code_change_from.strip()
        ):
            raise RuntimeError("code change acceptance requires an explicit resume phase")
        accepted_code_change_from = accepted_code_change_from.strip()

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

    if not resume:
        if existing is not None and existing.get("run_fingerprint") != identity[
            "run_fingerprint"
        ]:
            raise RuntimeError(
                "new run refused: output_dir belongs to a different immutable run; "
                "choose a new output directory or resume the stored run"
            )
        if existing is None:
            stale_markers = [name for name in _RUN_OWNED_MARKERS if (root / name).exists()]
            if path.exists() or stale_markers:
                details = [path.name] if path.exists() else []
                details.extend(stale_markers)
                raise RuntimeError(
                    "new run refused: output_dir contains unowned pipeline artifacts: "
                    + ", ".join(details[:8])
                    + "; choose an empty output directory"
                )

    if resume:
        _origin_code_version, history = _validated_code_history(existing)
        semantic_mismatches = {
            key: {"stored": existing.get(key), "current": identity.get(key)}
            for key in _SEMANTIC_IDENTITY_KEYS
            if existing.get(key) != identity.get(key)
        }
        code_changed = existing.get("code_version") != identity["code_version"]
        mismatches = dict(semantic_mismatches)
        if code_changed:
            mismatches["code_version"] = {
                "stored": existing.get("code_version"),
                "current": identity["code_version"],
            }
        if mismatches and existing.get("run_fingerprint") != identity["run_fingerprint"]:
            mismatches["run_fingerprint"] = {
                "stored": existing.get("run_fingerprint"),
                "current": identity["run_fingerprint"],
            }
        if semantic_mismatches or (code_changed and accepted_code_change_from is None):
            raise RuntimeError(
                "resume refused: immutable run identity changed: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        if code_changed:
            transition = {
                "schema_version": CODE_CHANGE_ACCEPTANCE_SCHEMA,
                "accepted_at": datetime.now(UTC).isoformat(),
                "acceptance": "explicit_cli_flag",
                "from_code_version": existing["code_version"],
                "to_code_version": identity["code_version"],
                "resume_from": accepted_code_change_from,
                "run_fingerprint": existing["run_fingerprint"],
            }
            migrated = {
                **existing,
                "origin_code_version": _origin_code_version,
                "code_version": identity["code_version"],
                "code_change_history": [*history, transition],
            }
            _write_manifest(path, migrated)
            return migrated
        return existing

    manifest = {
        **identity,
        "origin_code_version": identity["code_version"],
        "code_change_history": [],
        "resolved_config": normalized_config,
    }
    _write_manifest(path, manifest)
    return manifest
