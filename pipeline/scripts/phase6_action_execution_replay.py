#!/usr/bin/env python3
"""Recompile one persisted Phase 6 request without contacting any Provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from phases.phase6.action_execution_prompt import render_canonical_identity_projection
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import (
    _bind_final_media_index_prompt,
    _provider_prompt_metadata,
)
from schemas.continuity import ContinuityPlan
from utils.prompt_budget import enforce_prompt_budget

REPLAY_SCHEMA = "honcut.phase6-action-execution-replay.v1"
DEFAULT_SOURCE_RECEIPT = "phase6_storyboard_pose_atlas_live_acceptance.json"
REPLAY_COUNT = 10


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _content_item_from_manifest(item: dict[str, Any]) -> dict[str, Any]:
    media_type = str(item.get("media_type") or "")
    media_hash = str(item.get("sha256") or "")
    if media_type not in {"image_url", "video_url", "audio_url"}:
        raise ValueError("persisted replay media type is unsupported")
    if len(media_hash) != 64:
        raise ValueError("persisted replay media hash is invalid")
    return {
        "type": media_type,
        media_type: {"url": f"offline://sha256/{media_hash}"},
        "role": item.get("role"),
        "_reference_kind": item.get("responsibility"),
        "_reference_description": item.get("description"),
        "_reference_path": item.get("path"),
        "_reference_sha256": media_hash,
        "_character_id": item.get("character_id"),
        "_narrative_beat_id": item.get("narrative_beat_id"),
        "_narrative_cell_ids": list(item.get("narrative_cell_ids") or []),
        "_narrative_zero_time_anchor_cell_ids": list(
            item.get("narrative_zero_time_anchor_cell_ids") or []
        ),
        "_authority_roles": list(item.get("authority_roles") or []),
        "_non_authority_roles": list(item.get("non_authority_roles") or []),
        "_semantic_payload_sha256": item.get("semantic_payload_sha256"),
        "_pose_atlas_strategy": item.get("pose_atlas_strategy"),
        "_pose_atlas_page_index": item.get("pose_atlas_page_index"),
        "_pose_atlas_page_count": item.get("pose_atlas_page_count"),
        "_pose_atlas_plan_sha256": item.get("pose_atlas_plan_sha256"),
        "_pose_atlas_timing_contract": item.get("pose_atlas_timing_contract"),
        "_pose_atlas_camera_motion_contract_sha256": item.get(
            "pose_atlas_camera_motion_contract_sha256"
        ),
        "_performance_beat_id": item.get("performance_beat_id"),
        "_performance_cell_ids": list(item.get("performance_cell_ids") or []),
        "_performance_source_action_unit_ids": list(
            item.get("performance_source_action_unit_ids") or []
        ),
        "_performance_prop_ids": list(item.get("performance_prop_ids") or []),
        "_performance_source_board_sha256": item.get("performance_source_board_sha256"),
        "_mandatory_reference": item.get("mandatory") is True,
    }


def replay_persisted_action_request(
    run_dir: Path,
    *,
    source_receipt_path: Path,
    output_receipt_path: Path,
    beat_id: str | None = None,
) -> dict[str, Any]:
    """Hash-verify persisted evidence and rebuild the Provider prompt ten times."""
    run_dir = run_dir.resolve()
    source_receipt_path = source_receipt_path.resolve()
    output_receipt_path = output_receipt_path.resolve()
    if output_receipt_path == source_receipt_path or run_dir in output_receipt_path.parents:
        raise ValueError("replay receipt must be written outside the immutable source run")
    source = _read_object(source_receipt_path)
    preflight = source.get("preflight") or {}
    selected_beat = str(beat_id or preflight.get("beat_id") or "").strip()
    if not selected_beat:
        raise ValueError("replay source does not identify a storyboard beat")

    plan_path = run_dir / "CONTINUITY_PLAN.json"
    canonical_path = run_dir / "CANONICAL_VISUAL_CONTRACT.json"
    shot_id = selected_beat.split("_P", 1)[0]
    shot_meta_path = run_dir / "shots" / shot_id / "SHOT_META.json"
    immutable_paths = [source_receipt_path, plan_path, canonical_path, shot_meta_path]
    immutable_before = {str(path): _sha256_file(path) for path in immutable_paths}

    plan = ContinuityPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    chunks = [
        chunk
        for shot in plan.shots
        for chunk in shot.chunks
        if chunk.storyboard_beat_id == selected_beat
    ]
    if len(chunks) != 1:
        raise ValueError("replay beat does not resolve to exactly one continuity chunk")
    chunk = chunks[0]
    if not chunk.storyboard_pose_atlas_plan_schema:
        raise ValueError("replay requires a persisted pose-atlas request")

    canonical = _read_object(canonical_path)
    visible_ids = [str(value) for value in preflight.get("visible_character_ids") or []]
    identity_projection = render_canonical_identity_projection(
        canonical,
        character_ids=visible_ids,
    )
    media_manifest = preflight.get("media_index_manifest")
    if not isinstance(media_manifest, list) or not media_manifest:
        raise ValueError("replay source has no final media index manifest")
    for item in media_manifest:
        relative_path = str(item.get("path") or "").strip()
        expected_hash = str(item.get("sha256") or "").strip()
        asset_path = run_dir / relative_path
        if not relative_path or not asset_path.is_file():
            raise ValueError("replay media evidence is missing")
        if _sha256_file(asset_path) != expected_hash:
            raise ValueError("replay media evidence hash mismatch")
        immutable_paths.append(asset_path)
        immutable_before[str(asset_path)] = expected_hash

    shot_meta = _read_object(shot_meta_path)
    base_content = [
        {
            "type": "text",
            "text": "persisted prompt intentionally replaced by action-first projection",
            "_canonical_identity_projection": identity_projection,
            "_canonical_visual_contract_sha256": canonical["contract_sha256"],
            "_phase6_prompt_context": shot_meta,
        }
    ]
    base_content.extend(_content_item_from_manifest(dict(item)) for item in media_manifest)
    request = ChunkExecutionRequest(
        resource_id=f"REPLAY_{chunk.chunk_id}",
        shot_id=shot_id,
        chunk=chunk,
        anchors={},
        output_path=output_receipt_path.with_suffix(".mp4"),
        previous_output_path=None,
        input_fingerprint="provider-deny-replay",
        memory_context="",
    )

    replay_results: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for _index in range(REPLAY_COUNT):
        bound, rebuilt_manifest = _bind_final_media_index_prompt(base_content, request)
        prompt = next(str(item.get("text") or "") for item in bound if item.get("type") == "text")
        enforce_prompt_budget(
            prompt,
            provider="seedance",
            model=str((source.get("task_payload") or {}).get("model") or "seedance"),
            purpose="video_generation",
        )
        replay_results.append((prompt, _provider_prompt_metadata(bound), rebuilt_manifest))
    prompt, prompt_metadata, rebuilt_manifest = replay_results[0]
    if any(result != replay_results[0] for result in replay_results[1:]):
        raise RuntimeError("action request replay is not deterministic")
    if rebuilt_manifest != media_manifest:
        raise RuntimeError("replay changed persisted media order or hashes")
    action_position = prompt.find("[honcut.action-execution-brief.v1]")
    identity_position = prompt.find("[honcut.phase6-identity-projection.v1]")
    if action_position < 0 or identity_position < 0 or action_position >= identity_position:
        raise RuntimeError("replay prompt is not action-first")
    action_group_ids = list(prompt_metadata.get("action_execution_group_ids") or [])
    group_positions = {group_id: prompt.find(group_id) for group_id in action_group_ids}
    if any(
        position < action_position or position >= identity_position
        for position in group_positions.values()
    ):
        raise RuntimeError("replay action group is missing from the front action brief")
    marker_counts = {
        "action_execution_brief": prompt.count("[honcut.action-execution-brief.v1]"),
        "identity_projection": prompt.count("[honcut.phase6-identity-projection.v1]"),
        "primary_camera": prompt.count("唯一主运镜"),
        "legacy_live_pacing": prompt.count("[honcut.live-paced-action-window.v1]"),
        "legacy_video_contract": prompt.count("[honcut-video-generation-contract-v2]"),
    }
    if marker_counts != {
        "action_execution_brief": 1,
        "identity_projection": 1,
        "primary_camera": 1,
        "legacy_live_pacing": 0,
        "legacy_video_contract": 0,
    }:
        raise RuntimeError("replay prompt contains duplicate or legacy motion authority")
    if any(prompt.count(group_id) != 1 for group_id in action_group_ids):
        raise RuntimeError("replay prompt does not contain each action group exactly once")

    immutable_after = {str(path): _sha256_file(path) for path in immutable_paths}
    if immutable_after != immutable_before:
        raise RuntimeError("provider-deny replay modified immutable source evidence")
    receipt = {
        "schema": REPLAY_SCHEMA,
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "source_run": str(run_dir),
        "source_receipt_path": str(source_receipt_path),
        "source_receipt_sha256": immutable_before[str(source_receipt_path)],
        "continuity_plan_sha256": immutable_before[str(plan_path)],
        "canonical_visual_contract_sha256": canonical["contract_sha256"],
        "shot_meta_sha256": immutable_before[str(shot_meta_path)],
        "beat_id": selected_beat,
        "media_index_manifest": rebuilt_manifest,
        "media_sha256": [item["sha256"] for item in rebuilt_manifest],
        "prompt_chars": len(prompt),
        "prompt_metadata": prompt_metadata,
        "action_brief_position": action_position,
        "identity_projection_position": identity_position,
        "prompt_contract_checks": {
            "action_group_positions": group_positions,
            "marker_counts": marker_counts,
        },
        "legacy_prompt_sha256": preflight.get("prompt_sha256"),
        "recovery_replay_count": REPLAY_COUNT,
        "provider_request_count": 0,
    }
    _atomic_write_json(output_receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--beat-id")
    args = parser.parse_args()
    source_receipt = args.source_receipt or args.run_dir / DEFAULT_SOURCE_RECEIPT
    receipt = replay_persisted_action_request(
        args.run_dir,
        source_receipt_path=source_receipt,
        output_receipt_path=args.output_receipt,
        beat_id=args.beat_id,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
