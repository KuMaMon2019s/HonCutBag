"""Immutable canonical anchors plus bounded generated motion memory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from schemas.continuity import ContinuityPlan

MEMORY_KIND = "honcut.continuity_memory.v1"
CANONICAL_ASSET_NAMES = {"face_closeup.png", "full_body.png", "front.png"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_assets(output_dir: Path) -> list[dict[str, str]]:
    characters_dir = output_dir / "characters"
    if not characters_dir.is_dir():
        return []
    assets = []
    for path in sorted(characters_dir.rglob("*.png")):
        if path.name not in CANONICAL_ASSET_NAMES or not path.is_file():
            continue
        assets.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _file_hash(path),
                "role": "canonical",
            }
        )
    return assets


def _canonical_anchors(plan: ContinuityPlan) -> dict[str, Any]:
    return {
        shot.shot_id: {
            "boundary_before": shot.boundary_before,
            "anchors": shot.anchors.model_dump(mode="json"),
        }
        for shot in plan.shots
    }


def initialize_continuity_memory(
    output_dir: str | Path,
    plan: ContinuityPlan,
) -> dict[str, Any]:
    """Create canonical memory once and reject any later attempt to rewrite it."""
    root = Path(output_dir)
    path = root / "CONTINUITY_MEMORY.json"
    canonical = {
        "anchors": _canonical_anchors(plan),
        "assets": _canonical_assets(root),
    }
    canonical_fingerprint = _canonical_hash(canonical)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("kind") != MEMORY_KIND:
            raise ValueError(f"unsupported continuity memory in {path}")
        if existing.get("canonical_fingerprint") != canonical_fingerprint:
            raise RuntimeError("canonical continuity anchors changed after memory initialization")
        return existing
    memory = {
        "kind": MEMORY_KIND,
        "canonical_fingerprint": canonical_fingerprint,
        "canonical": canonical,
        "recent_motion": [],
    }
    _atomic_write(path, memory)
    return memory


def render_continuity_memory_context(
    output_dir: str | Path,
    plan: ContinuityPlan,
    shot_id: str,
) -> str:
    """Render a bounded provider prompt context with explicit trust precedence."""
    memory = initialize_continuity_memory(output_dir, plan)
    anchor = memory["canonical"]["anchors"].get(shot_id, {})
    recent = [
        {
            "shot_id": item.get("shot_id"),
            "chunk_id": item.get("chunk_id"),
            "screen_direction": item.get("screen_direction", ""),
            "camera_motion": item.get("camera_motion", ""),
        }
        for item in memory.get("recent_motion", [])
        if item.get("shot_id") == shot_id
    ]
    recent = recent[-3:]
    packet = {
        "trust_order": [
            "canonical anchors are authoritative and immutable",
            "generated recent motion is advisory and must never redefine identity",
        ],
        "canonical": anchor,
        "recent_motion": recent,
    }
    return "CONTINUITY_MEMORY " + json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _frame_pixels(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((96, 54))
    return np.asarray(image, dtype=np.float32)


def _frame_hash(pixels: np.ndarray) -> np.ndarray:
    image = Image.fromarray(pixels.astype(np.uint8)).resize((9, 8))
    values = np.asarray(image, dtype=np.float32)
    return values[:, 1:] > values[:, :-1]


def _frame_quality(pixels: np.ndarray) -> float:
    exposure = 1.0 - abs(float(pixels.mean()) - 127.5) / 127.5
    horizontal = np.diff(pixels, axis=1)
    vertical = np.diff(pixels, axis=0)
    sharpness = min(1.0, (float(horizontal.var()) + float(vertical.var())) / 2000.0)
    return round(max(0.0, 0.65 * exposure + 0.35 * sharpness), 6)


def select_memory_keyframes(
    candidates: Sequence[str | Path],
    *,
    max_count: int = 4,
    min_hash_distance: float = 0.12,
) -> list[dict[str, Any]]:
    """Select high-quality, perceptually distinct generated frames."""
    if max_count < 1:
        raise ValueError("max_count must be at least 1")
    scored = []
    for candidate in candidates:
        path = Path(candidate)
        pixels = _frame_pixels(path)
        scored.append((path, _frame_quality(pixels), _frame_hash(pixels)))
    scored.sort(key=lambda item: (-item[1], str(item[0])))
    selected: list[tuple[Path, float, np.ndarray]] = []
    for item in scored:
        if any(
            float(np.not_equal(item[2], prior[2]).mean()) < min_hash_distance for prior in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= max_count:
            break
    return [
        {
            "source_path": str(path),
            "source_sha256": _file_hash(path),
            "quality_score": quality,
            "role": "generated",
        }
        for path, quality, _ in selected
    ]


def record_recent_motion(
    output_dir: str | Path,
    plan: ContinuityPlan,
    *,
    shot_id: str,
    chunk_id: str,
    candidate_frames: Sequence[str | Path],
    screen_direction: str = "",
    camera_motion: str = "",
    recent_limit: int = 6,
) -> dict[str, Any]:
    """Store generated keyframes separately without mutating canonical memory."""
    if recent_limit < 1:
        raise ValueError("recent_limit must be at least 1")
    root = Path(output_dir)
    memory = initialize_continuity_memory(root, plan)
    selected = select_memory_keyframes(candidate_frames)
    generated_dir = root / "continuity_memory" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    stored_frames = []
    for frame in selected:
        source = Path(frame["source_path"])
        suffix = source.suffix.lower() or ".jpg"
        destination = generated_dir / f"{chunk_id}_{frame['source_sha256'][:12]}{suffix}"
        if not destination.is_file():
            shutil.copy2(source, destination)
        stored_frames.append(
            {
                "path": str(destination.relative_to(root)),
                "sha256": frame["source_sha256"],
                "quality_score": frame["quality_score"],
                "role": "generated",
            }
        )
    entry = {
        "shot_id": shot_id,
        "chunk_id": chunk_id,
        "screen_direction": screen_direction,
        "camera_motion": camera_motion,
        "keyframes": stored_frames,
    }
    recent = [item for item in memory.get("recent_motion", []) if item.get("chunk_id") != chunk_id]
    recent.append(entry)
    memory["recent_motion"] = recent[-recent_limit:]
    _atomic_write(root / "CONTINUITY_MEMORY.json", memory)
    return entry
