"""Project/run-isolated cache identity for semantic generation results."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from runtime.generation_fingerprint import GenerationFingerprint


CACHE_KEY_SCHEMA = "honcut.cache-key.v1"


@dataclass(frozen=True)
class CacheKey:
    value: str
    material: dict[str, Any]

    def task_metadata(self) -> dict[str, Any]:
        return {"cache_key": self.value, "cache_identity": self.material}


def build_cache_key(
    *,
    project_id: str,
    run_id: str,
    input_lineage: Iterable[str],
    semantic_fingerprint: str,
) -> CacheKey:
    if not project_id.strip() or not run_id.strip():
        raise ValueError("cache project_id and run_id must not be empty")
    if not re.fullmatch(r"[0-9a-f]{64}", semantic_fingerprint):
        raise ValueError("cache semantic fingerprint must be a SHA-256 digest")
    lineage = tuple(sorted(set(str(item).strip() for item in input_lineage)))
    if not lineage or any(not item for item in lineage):
        raise ValueError("cache input lineage must not be empty")
    material = {
        "schema": CACHE_KEY_SCHEMA,
        "project_id": project_id,
        "run_id": run_id,
        "input_lineage": list(lineage),
        "semantic_fingerprint": semantic_fingerprint,
    }
    digest = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CacheKey(value=f"honcut-cache-v1:{digest}", material=material)


def generation_cache_key(
    *,
    project_id: str,
    run_id: str,
    fingerprint: GenerationFingerprint,
) -> CacheKey:
    hashes = fingerprint.material["input_artifact_hashes"]
    prompt_hash = fingerprint.material["prompt"]["content_sha256"]
    lineage = [
        f"artifact:{name}:{content_hash}"
        for name, content_hash in sorted(hashes.items())
    ]
    lineage.append(f"prompt:{prompt_hash}")
    return build_cache_key(
        project_id=project_id,
        run_id=run_id,
        input_lineage=lineage,
        semantic_fingerprint=fingerprint.value,
    )


def validate_cache_identity(
    stored: dict[str, Any],
    expected: CacheKey,
) -> None:
    if stored.get("cache_key") != expected.value:
        raise RuntimeError("cache key does not match project/run lineage")
    if stored.get("cache_identity") != expected.material:
        raise RuntimeError("cache identity material does not match its key")


__all__ = [
    "CACHE_KEY_SCHEMA",
    "CacheKey",
    "build_cache_key",
    "generation_cache_key",
    "validate_cache_identity",
]
