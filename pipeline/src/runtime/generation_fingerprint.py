"""Deterministic, secret-free fingerprints for paid generation requests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GENERATION_FINGERPRINT_SCHEMA = "honcut.generation-fingerprint.v1"
PHASE6_VIDEO_PROMPT_TEMPLATE_ID = "honcut.phase6.video"
PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION = "2"

_SECRET_KEY_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_cookie",
    "session_token",
    "token",
}


def _is_secret_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    return normalized in _SECRET_KEY_NAMES or normalized.endswith(
        ("_api_key", "_credential", "_password", "_private_key", "_secret", "_token")
    )


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation parameters cannot contain non-finite floats")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_secret_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    raise TypeError(f"unsupported generation fingerprint value: {type(value).__name__}")


@dataclass(frozen=True)
class GenerationFingerprint:
    value: str
    material: dict[str, Any]

    def task_metadata(self) -> dict[str, Any]:
        return {
            "input_fingerprint": self.value,
            "generation_fingerprint": self.material,
        }


def build_generation_fingerprint(
    *,
    prompt_text: str,
    prompt_template_id: str,
    prompt_template_version: str,
    provider_id: str,
    provider_version: str,
    model_id: str,
    model_version: str,
    parameters: Mapping[str, Any],
    input_artifact_hashes: Mapping[str, str],
) -> GenerationFingerprint:
    """Hash all semantic generation inputs while excluding credential fields."""
    identifiers = {
        "prompt_template_id": prompt_template_id,
        "prompt_template_version": prompt_template_version,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "model_id": model_id,
        "model_version": model_version,
    }
    if any(not value.strip() for value in identifiers.values()):
        raise ValueError("generation fingerprint identifiers must not be empty")
    normalized_hashes = {}
    for name, content_hash in sorted(input_artifact_hashes.items()):
        if len(content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in content_hash
        ):
            raise ValueError(f"invalid input artifact SHA-256 for {name}")
        normalized_hashes[str(name)] = content_hash
    material = {
        "schema": GENERATION_FINGERPRINT_SCHEMA,
        "prompt": {
            "template_id": prompt_template_id,
            "template_version": prompt_template_version,
            "content_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        },
        "provider": {"id": provider_id, "version": provider_version},
        "model": {"id": model_id, "version": model_version},
        "parameters": _canonicalize(parameters),
        "input_artifact_hashes": normalized_hashes,
    }
    serialized = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return GenerationFingerprint(
        value=hashlib.sha256(serialized).hexdigest(),
        material=material,
    )


__all__ = [
    "GENERATION_FINGERPRINT_SCHEMA",
    "PHASE6_VIDEO_PROMPT_TEMPLATE_ID",
    "PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION",
    "GenerationFingerprint",
    "build_generation_fingerprint",
]
