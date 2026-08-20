"""Fail-closed Phase 8 shot inventory validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class Phase8InventoryError(RuntimeError):
    """The storyboard, videos, and shot metadata do not describe one exact set."""


def normalize_shot_id(value: Any) -> str | None:
    """Return the current canonical ``SNN`` identity for a shot."""
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    if not text.isdigit():
        return None
    return f"S{int(text):02d}"


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def load_phase8_inventory(output_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load clips and metadata only after proving the exact shot-set invariant."""
    output_dir = Path(output_dir)
    storyboard_path = output_dir / "STORYBOARD.json"
    try:
        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8InventoryError(f"STORYBOARD.json is missing or invalid: {exc}") from exc

    shots = storyboard.get("shots")
    if not isinstance(shots, list) or not shots:
        raise Phase8InventoryError("STORYBOARD.json must contain a non-empty shots array")

    expected_ids: list[str] = []
    invalid_storyboard_entries: list[str] = []
    for index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            invalid_storyboard_entries.append(f"index {index}")
            continue
        raw_id = shot.get("shot_id") or shot.get("id")
        shot_id = normalize_shot_id(raw_id)
        if shot_id is None:
            invalid_storyboard_entries.append(f"index {index}: {raw_id!r}")
        else:
            expected_ids.append(shot_id)
    duplicate_expected = _duplicates(expected_ids)
    if invalid_storyboard_entries or duplicate_expected:
        details = []
        if invalid_storyboard_entries:
            details.append("invalid IDs: " + ", ".join(invalid_storyboard_entries))
        if duplicate_expected:
            details.append("duplicate IDs: " + ", ".join(duplicate_expected))
        raise Phase8InventoryError("invalid storyboard shot inventory; " + "; ".join(details))

    shots_dir = output_dir / "shots"
    if not shots_dir.is_dir():
        raise Phase8InventoryError("shots directory is missing")

    video_ids: list[str] = []
    metadata_ids: list[str] = []
    directories: dict[str, Path] = {}
    invalid_directories: list[str] = []
    for shot_dir in sorted(path for path in shots_dir.iterdir() if path.is_dir()):
        has_video = (shot_dir / "output.mp4").is_file()
        has_metadata = (shot_dir / "SHOT_META.json").is_file()
        if not (has_video or has_metadata):
            continue
        shot_id = normalize_shot_id(shot_dir.name)
        if shot_id is None or shot_dir.name != shot_id:
            invalid_directories.append(shot_dir.name)
            continue
        if shot_id in directories:
            invalid_directories.append(shot_dir.name)
            continue
        directories[shot_id] = shot_dir
        if has_video:
            video_ids.append(shot_id)
        if has_metadata:
            metadata_ids.append(shot_id)

    expected_set = set(expected_ids)
    video_set = set(video_ids)
    metadata_set = set(metadata_ids)
    if (
        invalid_directories
        or _duplicates(video_ids)
        or _duplicates(metadata_ids)
        or video_set != expected_set
        or metadata_set != expected_set
    ):
        details = {
            "expected": sorted(expected_set),
            "videos": sorted(video_set),
            "metadata": sorted(metadata_set),
            "missing_videos": sorted(expected_set - video_set),
            "unexpected_videos": sorted(video_set - expected_set),
            "missing_metadata": sorted(expected_set - metadata_set),
            "unexpected_metadata": sorted(metadata_set - expected_set),
            "invalid_directories": sorted(invalid_directories),
        }
        raise Phase8InventoryError(
            "Phase 8 exact shot inventory failed: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )

    clip_paths: list[str] = []
    shot_metas: list[dict[str, Any]] = []
    for shot_id in expected_ids:
        shot_dir = directories[shot_id]
        meta_path = shot_dir / "SHOT_META.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase8InventoryError(f"{shot_id} SHOT_META.json is invalid: {exc}") from exc
        if not isinstance(meta, dict):
            raise Phase8InventoryError(f"{shot_id} SHOT_META.json must contain an object")
        embedded_id = normalize_shot_id(meta.get("shot_id") or meta.get("id"))
        if embedded_id != shot_id:
            raise Phase8InventoryError(
                f"{shot_id} SHOT_META.json identity mismatch: {embedded_id!r}"
            )
        clip_paths.append(str(shot_dir / "output.mp4"))
        shot_metas.append(meta)

    return clip_paths, shot_metas
