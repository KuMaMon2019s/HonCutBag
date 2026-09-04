"""Adaptive identity-neutral pose-atlas planning owned by Phase 2.

The planner samples canonical action lineage and an Adaptation-owned camera
contract.  It never invents actions, selects camera technique, or calls a
Provider.  Rendering and packaging consume the resulting JSON-safe plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from phases.phase2.storyboard_guide_pose import compile_pose_contracts, render_pose_cell
from utils.camera_motion_contracts import (
    build_camera_motion_contract,
    camera_projection_at_progress,
    validate_camera_motion_duration,
)
from utils.video_capabilities import (
    VideoModelCapabilities,
    capabilities_for,
)

POSE_ATLAS_PLAN_SCHEMA = "honcut.storyboard-pose-atlas-plan.v1"
POSE_ATLAS_CANDIDATE_SCHEMA = "honcut.storyboard-pose-atlas-candidate.v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _duration_s(beat: Mapping[str, Any]) -> float:
    for field in (
        "effective_story_duration_s",
        "unique_duration_s",
        "duration_s",
        "target_duration_s",
        "duration",
    ):
        value = beat.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError("storyboard beat is missing an effective duration")


def _canonical_units(beat: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_units = beat.get("generation_action_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("storyboard beat is missing canonical generation action units")
    units = [dict(item) for item in raw_units if isinstance(item, Mapping)]
    if len(units) != len(raw_units):
        raise ValueError("generation action units must be objects")
    unit_ids = [str(unit.get("unit_id") or "").strip() for unit in units]
    if any(not unit_id for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        raise ValueError("generation action units need unique non-empty unit IDs")
    return units


def _action_groups(units: Sequence[Mapping[str, Any]], beat_id: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, unit in enumerate(units, 1):
        raw_source_event_id = unit.get("source_event_id")
        source_event_ids = (
            [raw_source_event_id]
            if isinstance(raw_source_event_id, int)
            and not isinstance(raw_source_event_id, bool)
            and raw_source_event_id > 0
            else []
        )
        lineage = {
            "unit_ids": [str(unit["unit_id"])],
            "source_action_unit_ids": [str(unit.get("source_action_unit_id") or "")]
            if str(unit.get("source_action_unit_id") or "")
            else [],
            "source_event_ids": source_event_ids,
            "source_generation_unit_indexes": list(
                unit.get("source_generation_unit_indexes") or []
            ),
            "source_micro_action_indexes": list(unit.get("source_micro_action_indexes") or []),
            "source_ledger_indexes": list(unit.get("ledger_indexes") or []),
        }
        group = {
            "action_group_id": f"{beat_id}_A{index:02d}",
            "order": index,
            "lineage": lineage,
            "actions_sha256": _canonical_sha256(list(unit.get("actions") or [])),
        }
        group["group_sha256"] = _canonical_sha256(group)
        groups.append(group)
    return groups


def _camera_contract(beat: Mapping[str, Any]) -> dict[str, Any]:
    persisted = beat.get("camera_motion_contract")
    if isinstance(persisted, Mapping):
        contract = dict(persisted)
        stored_sha = str(contract.get("contract_sha256") or "")
        if stored_sha:
            unhashed = dict(contract)
            unhashed.pop("contract_sha256", None)
            if stored_sha != _canonical_sha256(unhashed):
                raise ValueError("camera motion contract hash mismatch")
        return contract
    # Legacy artifacts may not yet contain the persisted contract.  Rebuilding
    # from already-authored shot fields is deterministic; no technique choice
    # or feasibility repair occurs here.
    return build_camera_motion_contract(beat)


def _candidate_contracts(
    *,
    sample_count: int,
    action_group_count: int,
    page_cell_count: int,
    high_fidelity_group_limit: int,
    plan_payload_sha256: str,
) -> list[dict[str, Any]]:
    sample_ids = [f"G{index:02d}" for index in range(1, sample_count + 1)]
    single = {
        "schema": POSE_ATLAS_CANDIDATE_SCHEMA,
        "strategy": "single_atlas",
        "page_count": 1,
        "preferred": action_group_count <= high_fidelity_group_limit,
        "pages": [
            {
                "page_index": 1,
                "sample_ids": sample_ids,
                "sample_range": [1, sample_count],
            }
        ],
        "plan_payload_sha256": plan_payload_sha256,
    }
    single["candidate_sha256"] = _canonical_sha256(single)
    candidates = [single]
    if sample_count > page_cell_count:
        pages = []
        for offset in range(0, sample_count, page_cell_count):
            page_ids = sample_ids[offset : offset + page_cell_count]
            pages.append(
                {
                    "page_index": len(pages) + 1,
                    "sample_ids": page_ids,
                    "sample_range": [offset + 1, offset + len(page_ids)],
                }
            )
        paged = {
            "schema": POSE_ATLAS_CANDIDATE_SCHEMA,
            "strategy": "paged_atlas",
            "page_count": len(pages),
            "preferred": action_group_count > high_fidelity_group_limit,
            "pages": pages,
            "plan_payload_sha256": plan_payload_sha256,
        }
        paged["candidate_sha256"] = _canonical_sha256(paged)
        candidates.append(paged)
    return candidates


def build_pose_atlas_plan(
    beat: Mapping[str, Any],
    *,
    known_actor_roles: tuple[str, ...] = (),
    capabilities: VideoModelCapabilities | None = None,
) -> dict[str, Any]:
    """Compile ordered action groups and camera-projected pose samples."""

    beat_id = str(beat.get("beat_id") or "").strip()
    if not beat_id:
        raise ValueError("storyboard beat requires beat_id")
    duration = _duration_s(beat)
    selected_capabilities = capabilities or capabilities_for(dict(beat))
    capacity = selected_capabilities.storyboard_pose_capacity(duration)
    units = _canonical_units(beat)
    groups = _action_groups(units, beat_id)
    if len(groups) > capacity["reliable_action_group_limit"]:
        raise ValueError(
            f"{beat_id} has {len(groups)} canonical action groups, above the "
            f"{selected_capabilities.name} reliable action-group limit "
            f"{capacity['reliable_action_group_limit']} for {duration:g}s; "
            "return to Adaptation for Pxx splitting"
        )
    camera_contract = _camera_contract(beat)
    validate_camera_motion_duration(camera_contract, duration, resource_id=beat_id)
    camera_hash = str(camera_contract.get("contract_sha256") or _canonical_sha256(camera_contract))
    sample_count = capacity["pose_sample_count"]
    cells: list[dict[str, Any]] = []
    shot_id = beat_id.split("_P", 1)[0]
    for index in range(sample_count):
        path_progress = index / max(1, sample_count - 1)
        cells.append(
            {
                "cell": index + 1,
                "label": f"{shot_id}_G{index + 1:02d}",
                "primary_shot_id": shot_id,
                "secondary_beat_id": beat_id,
                "stage": "action_progress",
                "action_progress": path_progress,
                "camera_movement": str(camera_contract.get("movement") or "static"),
                "camera_projection": camera_projection_at_progress(
                    camera_contract,
                    path_progress,
                ),
            }
        )
    compiled = compile_pose_contracts(
        beat,
        cells,
        known_actor_roles=known_actor_roles,
    )
    groups_by_units = {tuple(group["lineage"]["unit_ids"]): group for group in groups}
    has_initial_anchor = beat_id.endswith("_P01")
    timing = selected_capabilities.storyboard_timing_contract(
        duration,
        has_initial_anchor=has_initial_anchor,
        terminal_mode=str(beat.get("terminal_reference_mode") or "semantic_hold"),
    )
    samples: list[dict[str, Any]] = []
    for index, cell in enumerate(compiled, 1):
        contract = cell["pose_contract"]
        binding_ids = tuple(str(binding["unit_id"]) for binding in contract["action_bindings"])
        group = groups_by_units.get(binding_ids)
        if group is None:
            raise ValueError(f"{beat_id} pose sample crossed canonical action groups")
        timing_role = "story_action"
        story_time_weight = 1.0
        if has_initial_anchor and index == 1:
            timing_role = "initial_anchor"
            story_time_weight = 0.0
        elif index == sample_count:
            timing_role = "terminal_hold"
            story_time_weight = 0.0
        sample = {
            "sample_id": f"G{index:02d}",
            "cell_id": str(cell["label"]),
            "sample_index": index,
            "action_group_id": group["action_group_id"],
            "action_group_order": group["order"],
            "group_progress": float(contract["pose_progress"]),
            "timing_role": timing_role,
            "story_time_weight": story_time_weight,
            "pose_contract": contract,
            "camera_projection": dict(cell["camera_projection"]),
        }
        sample["pose_fingerprint"] = _canonical_sha256(
            {
                "sample_index": index,
                "action_group_id": sample["action_group_id"],
                "timing_role": timing_role,
                "pose_contract_sha256": contract["contract_sha256"],
                "camera_projection": sample["camera_projection"],
            }
        )
        samples.append(sample)
    payload = {
        "schema": POSE_ATLAS_PLAN_SCHEMA,
        "version": 1,
        "beat_id": beat_id,
        "duration_s": round(duration, 3),
        "provider_capability": selected_capabilities.name,
        "capacity": capacity,
        "timing_contract": timing,
        "camera_motion_contract": camera_contract,
        "camera_motion_contract_sha256": camera_hash,
        "action_groups": groups,
        "pose_samples": samples,
        "source_pixel_usage": "none",
    }
    plan_payload_sha256 = _canonical_sha256(payload)
    payload["atlas_candidates"] = _candidate_contracts(
        sample_count=sample_count,
        action_group_count=len(groups),
        page_cell_count=capacity["atlas_page_cell_count"],
        high_fidelity_group_limit=capacity["single_atlas_high_fidelity_group_limit"],
        plan_payload_sha256=plan_payload_sha256,
    )
    payload["plan_payload_sha256"] = plan_payload_sha256
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def select_pose_atlas_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    available_image_slots: int,
) -> dict[str, Any]:
    """Select one already-rendered candidate after authoritative media freeze."""

    if available_image_slots < 1:
        raise ValueError("pose atlas has no image slot in the provider media budget")
    by_strategy: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        strategy = str(candidate.get("strategy") or "")
        if strategy in by_strategy:
            raise ValueError(f"duplicate pose atlas strategy: {strategy}")
        pages = candidate.get("pages")
        if strategy not in {"single_atlas", "paged_atlas"} or not isinstance(pages, list):
            raise ValueError("pose atlas candidate is malformed")
        if len(pages) != int(candidate.get("page_count") or 0):
            raise ValueError("pose atlas candidate page count mismatch")
        by_strategy[strategy] = dict(candidate)
    single = by_strategy.get("single_atlas")
    paged = by_strategy.get("paged_atlas")
    preferred = next(
        (
            candidate
            for candidate in (paged, single)
            if candidate is not None and candidate.get("preferred") is True
        ),
        None,
    )
    if preferred is not None and int(preferred["page_count"]) <= available_image_slots:
        return preferred
    for candidate in (single, paged):
        if candidate is not None and int(candidate["page_count"]) <= available_image_slots:
            return candidate
    raise ValueError("no pose atlas candidate fits the provider media budget")


def _atlas_layout(cell_count: int) -> tuple[int, int]:
    candidates: list[tuple[float, int, int, int]] = []
    for columns in range(1, cell_count + 1):
        rows = math.ceil(cell_count / columns)
        aspect = columns * (16 / 9) / rows
        if 0.4 <= aspect <= 2.5:
            candidates.append((abs(aspect - (16 / 9)), columns * rows - cell_count, columns, rows))
    if not candidates:
        raise ValueError(f"cannot lay out {cell_count} pose samples")
    _distance, _blank_count, columns, rows = min(candidates)
    return columns, rows


def render_pose_atlas_candidates(
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    font_factory: Any,
) -> dict[str, Any]:
    """Render every Phase 2 candidate locally from the same pose payload."""

    beat_id = str(plan.get("beat_id") or "")
    samples = plan.get("pose_samples")
    candidates = plan.get("atlas_candidates")
    if not beat_id or not isinstance(samples, list) or not isinstance(candidates, list):
        raise ValueError("pose atlas plan is incomplete")
    expected_plan_sha = str(plan.get("plan_sha256") or "")
    unhashed_plan = dict(plan)
    unhashed_plan.pop("plan_sha256", None)
    if expected_plan_sha != _canonical_sha256(unhashed_plan):
        raise ValueError("pose atlas plan hash mismatch")
    samples_by_id = {str(sample.get("sample_id") or ""): sample for sample in samples}
    atlas_dir = output_dir / "storyboard_pose_atlases" / beat_id
    atlas_dir.mkdir(parents=True, exist_ok=True)
    rendered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        strategy = str(candidate.get("strategy") or "")
        candidate_copy = dict(candidate)
        expected_candidate_sha = str(candidate_copy.pop("candidate_sha256", ""))
        if expected_candidate_sha != _canonical_sha256(candidate_copy):
            raise ValueError("pose atlas candidate hash mismatch")
        rendered_pages: list[dict[str, Any]] = []
        for page in candidate.get("pages") or []:
            sample_ids = list(page.get("sample_ids") or [])
            page_samples = []
            for sample_id in sample_ids:
                sample = samples_by_id.get(str(sample_id))
                if sample is None:
                    raise ValueError(f"pose atlas page references unknown {sample_id}")
                page_samples.append(sample)
            columns, rows = _atlas_layout(len(page_samples))
            rendered_cells = []
            for sample in page_samples:
                pose_contract = sample.get("pose_contract")
                cell = {
                    "label": str(pose_contract.get("cell_id") or ""),
                    "secondary_beat_id": beat_id,
                    "pose_contract": pose_contract,
                }
                rendered_cells.append(
                    render_pose_cell(
                        cell,
                        width=480,
                        height=270,
                        font_factory=font_factory,
                    )
                )
            canvas = Image.new(
                "RGB",
                (columns * 480, rows * 270),
                (247, 247, 244),
            )
            for index, rendered in enumerate(rendered_cells):
                canvas.paste(rendered, ((index % columns) * 480, (index // columns) * 270))
            page_index = int(page.get("page_index") or 0)
            suffix = "dense" if strategy == "single_atlas" else f"page-{page_index:02d}"
            image_path = atlas_dir / f"{suffix}.png"
            temporary = image_path.with_suffix(".png.tmp")
            canvas.save(temporary, format="PNG", optimize=True)
            temporary.replace(image_path)
            rendered_pages.append(
                {
                    **dict(page),
                    "image": str(image_path.relative_to(output_dir)),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "layout": {
                        "columns": columns,
                        "rows": rows,
                        "reading_order": "left_to_right_top_to_bottom",
                        "blank_cells": columns * rows - len(page_samples),
                        "width": canvas.width,
                        "height": canvas.height,
                    },
                }
            )
        rendered_candidate = {
            **dict(candidate),
            "pages": rendered_pages,
            "source_pixel_usage": "none",
            "renderer": "honcut.identity-neutral-pose-atlas-renderer.v1",
            "authority_roles": [
                "narrative_order",
                "action_direction",
                "camera_motion",
                "spatial_relationship",
            ],
            "non_authority_roles": [
                "character_identity",
                "face_geometry",
                "hair_geometry",
                "wardrobe",
                "prop_appearance",
                "cinematic_pixels",
            ],
        }
        rendered_candidate["rendered_candidate_sha256"] = _canonical_sha256(rendered_candidate)
        rendered_candidates.append(rendered_candidate)
    receipt = {
        "kind": "honcut.storyboard-pose-atlas-receipt.v1",
        "version": 1,
        "status": "done",
        "beat_id": beat_id,
        "plan": dict(plan),
        "plan_sha256": expected_plan_sha,
        "candidates": rendered_candidates,
        "source_pixel_usage": "none",
        "provider_request_count": 0,
    }
    receipt_path = atlas_dir / "manifest.json"
    temporary_receipt = receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_receipt.replace(receipt_path)
    receipt["receipt"] = str(receipt_path.relative_to(output_dir))
    receipt["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return receipt


__all__ = [
    "POSE_ATLAS_CANDIDATE_SCHEMA",
    "POSE_ATLAS_PLAN_SCHEMA",
    "build_pose_atlas_plan",
    "render_pose_atlas_candidates",
    "select_pose_atlas_candidate",
]
