"""Deterministic action-first projection for Phase 6 video providers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from utils.camera_motion_contracts import camera_movement_description
from utils.canonical_visual_contracts import validate_canonical_visual_contract
from utils.prompt_budget import PromptBudgetExceededError, resolve_prompt_budget

ACTION_EXECUTION_BRIEF_SCHEMA = "honcut.action-execution-brief.v1"
ACTION_EXECUTION_BRIEF_MARKER = "[honcut.action-execution-brief.v1]"
IDENTITY_PROJECTION_SCHEMA = "honcut.phase6-identity-projection.v1"
IDENTITY_PROJECTION_MARKER = "[honcut.phase6-identity-projection.v1]"
PROMPT_PROJECTION_SCHEMA = "honcut.phase6-prompt-projection.v1"

_PROMPT_PROJECTION_POLICY = {
    "schema": PROMPT_PROJECTION_SCHEMA,
    "mandatory_order": [
        "media_index",
        "media_role_isolation",
        "action_execution_brief",
        "identity_projection",
        "output_constraints",
    ],
    "optional_order": ["scene", "emotion", "audio"],
    "arbitrary_tail_truncation": False,
    "camera_authority": "action_execution_brief_only",
    "opening_pose_hold": "forbidden",
}
PROMPT_PROJECTION_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        _PROMPT_PROJECTION_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ACTION_SEPARATOR_RE = re.compile(r"\s*(?:→|->)\s*")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in result:
            result.append(token)
    return result


def _fact_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def render_canonical_identity_projection(
    contract: Mapping[str, Any],
    *,
    character_ids: Sequence[str] | None = None,
) -> str:
    """Render the identity facts needed by the video model without audit prose."""
    validated = validate_canonical_visual_contract(dict(contract))
    requested = _ordered_unique(list(character_ids or []))
    filter_requested = character_ids is not None
    projected: list[dict[str, Any]] = []
    for record in validated["characters"]:
        instance_ids = [
            str(instance["instance_id"])
            for instance in record.get("instances") or []
            if isinstance(instance, Mapping)
        ]
        if filter_requested and not (
            record["character_id"] in requested
            or record["entity_id"] in requested
            or any(instance_id in requested for instance_id in instance_ids)
        ):
            continue
        included_instances = [
            {
                "instance_id": instance["instance_id"],
                "face_identity": _fact_value(instance["face_identity"]),
            }
            for instance in record["instances"]
            if not filter_requested
            or record["character_id"] in requested
            or record["entity_id"] in requested
            or instance["instance_id"] in requested
        ]
        hair = record["hair"]
        projected.append(
            {
                "character_id": record["character_id"],
                "entity_id": record["entity_id"],
                "instance_count": len(included_instances),
                "instances": included_instances,
                "visual_identity_policy": record["visual_identity_policy"],
                "hair": {
                    key: _fact_value(hair[key])
                    for key in ("color", "length_class", "silhouette", "parting")
                },
                "body_build": _fact_value(record["body_build"]),
                "face": _fact_value(record["face"]),
                "wardrobe": _fact_value(record["wardrobe"]),
                "identity_props": [
                    {
                        "prop_id": prop["prop_id"],
                        "name": prop["name"],
                        "geometry": {
                            key: _fact_value(prop["geometry"][key])
                            for key in (
                                "topology",
                                "shape_family",
                                "component_count",
                                "active_end_count",
                                "handle_count",
                                "relative_scale",
                                "material",
                                "colors",
                                "emissive_regions",
                            )
                        },
                    }
                    for prop in record["identity_props"]
                ],
            }
        )
    payload = {
        "schema": IDENTITY_PROJECTION_SCHEMA,
        "canonical_visual_contract_sha256": validated["contract_sha256"],
        "required_character_count": sum(
            int(character["instance_count"]) for character in projected
        ),
        "characters": projected,
    }
    return (
        f"{IDENTITY_PROJECTION_MARKER}\n"
        + _canonical_json(payload)
        + "\n身份板和本投影只锁定人物数量、脸、发型、体型、服装与专属道具；"
        "不得克隆、换脸、换装或改变道具拓扑，但不得因此冻结首帧姿态。"
    )


def validate_canonical_identity_projection(
    projection: str,
    *,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the compact identity authority before Provider prompt assembly."""
    lines = str(projection or "").splitlines()
    if len(lines) < 2 or lines[0] != IDENTITY_PROJECTION_MARKER:
        raise ValueError("canonical identity projection marker is missing")
    try:
        payload = json.loads(lines[1])
    except json.JSONDecodeError as exc:
        raise ValueError("canonical identity projection payload is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != IDENTITY_PROJECTION_SCHEMA:
        raise ValueError("canonical identity projection schema is unsupported")
    if (
        not _SHA256_RE.fullmatch(expected_contract_sha256)
        or payload.get("canonical_visual_contract_sha256") != expected_contract_sha256
    ):
        raise ValueError("canonical identity projection contract hash is invalid")
    characters = payload.get("characters")
    if not isinstance(characters, list):
        raise ValueError("canonical identity projection characters are invalid")
    expected_count = sum(
        int(character.get("instance_count") or 0)
        for character in characters
        if isinstance(character, Mapping)
    )
    if payload.get("required_character_count") != expected_count:
        raise ValueError("canonical identity projection instance count is invalid")
    return payload


def _validate_lineage(group: Mapping[str, Any], *, group_id: str) -> dict[str, list[Any]]:
    lineage = group.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(f"{group_id} action lineage is missing")
    required = ("unit_ids",)
    normalized: dict[str, list[Any]] = {}
    for key in (
        "unit_ids",
        "source_action_unit_ids",
        "source_event_ids",
        "source_generation_unit_indexes",
        "source_micro_action_indexes",
        "source_ledger_indexes",
    ):
        values = lineage.get(key) or []
        if not isinstance(values, list):
            raise ValueError(f"{group_id} lineage {key} must be an ordered array")
        normalized[key] = list(values)
    if any(not normalized[key] for key in required):
        raise ValueError(f"{group_id} action lineage is incomplete")
    return normalized


def _camera_summary(context: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> str:
    movement = context.get("camera_movement")
    if movement not in (None, ""):
        try:
            return camera_movement_description(movement)
        except (KeyError, TypeError, ValueError):
            return str(movement).strip()
    projections = [
        sample.get("camera_projection")
        for sample in samples
        if isinstance(sample.get("camera_projection"), Mapping)
    ]
    if not projections:
        return "保持单一连续摄影机路径，运镜只辅助主体动作"
    first = projections[0]
    last = projections[-1]
    return (
        "单一连续摄影机路径："
        f"yaw {float(first.get('yaw_degrees') or 0):g}°→"
        f"{float(last.get('yaw_degrees') or 0):g}°，"
        f"pitch {float(first.get('pitch_degrees') or 0):g}°→"
        f"{float(last.get('pitch_degrees') or 0):g}°，"
        f"焦段 {float(first.get('focal_length_mm') or 0):g}mm→"
        f"{float(last.get('focal_length_mm') or 0):g}mm"
    )


def _trajectory_summary(
    pose_contracts: Sequence[Mapping[str, Any]],
    *,
    pose_families: Sequence[str],
    targets: Sequence[str],
) -> dict[str, Any]:
    actor_paths: dict[str, list[list[float]]] = {}
    modifier_values: dict[str, list[float]] = {}
    progress_values: list[float] = []
    directions: list[str] = []
    for contract in pose_contracts:
        progress = contract.get("pose_progress")
        if isinstance(progress, (int, float)) and not isinstance(progress, bool):
            progress_values.append(round(float(progress), 3))
        direction = str(contract.get("direction") or "").strip()
        if direction and direction not in directions:
            directions.append(direction)
        modifiers = contract.get("mechanics_modifiers") or {}
        if isinstance(modifiers, Mapping):
            for key in ("center_drop", "torso_lean", "stance_width", "lead_step"):
                value = modifiers.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    modifier_values.setdefault(key, []).append(round(float(value), 3))
        geometry = contract.get("geometry") or {}
        for actor in geometry.get("actors") or []:
            if not isinstance(actor, Mapping):
                continue
            role = str(actor.get("role_ref") or actor.get("slot") or "actor")
            root = actor.get("root_translation")
            if (
                isinstance(root, list)
                and len(root) == 2
                and all(isinstance(value, (int, float)) for value in root)
            ):
                actor_paths.setdefault(role, []).append(
                    [round(float(root[0]), 3), round(float(root[1]), 3)]
                )
    root_trajectory = {}
    for role, path in actor_paths.items():
        peak = max(path, key=lambda point: abs(point[0]) + abs(point[1]))
        root_trajectory[role] = {
            "start": path[0],
            "peak": peak,
            "end": path[-1],
        }
    contact_families = {"strike", "kick", "block", "grab_control", "throw", "prop_use"}
    return {
        "root_trajectory": root_trajectory,
        "pose_progress": {
            "start": progress_values[0] if progress_values else 0,
            "peak": max(progress_values) if progress_values else 0,
            "end": progress_values[-1] if progress_values else 0,
        },
        "direction_profile": directions,
        "weight_transfer_profile": {
            key: {"min": min(values), "max": max(values)} for key, values in modifier_values.items()
        },
        "contact_profile": (
            {"required_targets": list(targets)}
            if set(pose_families) & contact_families and targets
            else {"required_targets": [], "mode": "no invented target contact"}
        ),
    }


def compile_action_execution_brief(
    *,
    beat_id: str,
    action_prompt: str,
    start_state: str,
    end_state: str,
    target_duration_s: float,
    action_groups: Sequence[Mapping[str, Any]],
    pose_samples: Sequence[Mapping[str, Any]],
    timing_contract: Mapping[str, Any],
    media_manifest: Sequence[Mapping[str, Any]],
    prompt_context: Mapping[str, Any] | None = None,
    canonical_visual_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Compile the current Pxx motion authority after final media indexing."""
    beat = str(beat_id or "").strip()
    if not beat:
        raise ValueError("action execution brief requires a storyboard beat id")
    if timing_contract.get("schema") != "honcut.storyboard-action-timing.v1":
        raise ValueError(f"{beat} action timing schema is unsupported")
    duration = float(timing_contract.get("duration_s") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{beat} action timing duration is invalid")
    if not math.isclose(duration, float(target_duration_s), abs_tol=0.05):
        raise ValueError(
            f"{beat} action timing duration {duration:g}s conflicts with chunk "
            f"duration {float(target_duration_s):g}s"
        )
    story_action = timing_contract.get("story_action")
    terminal_hold = timing_contract.get("terminal_hold")
    if not isinstance(story_action, Mapping) or not isinstance(terminal_hold, Mapping):
        raise ValueError(f"{beat} action timing contract is incomplete")
    completion_window = story_action.get("completion_window_s")
    if (
        not isinstance(completion_window, list)
        or len(completion_window) != 2
        or not all(isinstance(value, (int, float)) for value in completion_window)
        or not 0 < float(completion_window[0]) <= float(completion_window[1]) <= duration
    ):
        raise ValueError(f"{beat} action completion window is invalid")
    target_completion = float(story_action.get("target_completion_s") or 0)
    target_hold = float(terminal_hold.get("target_duration_s") or 0)
    if (
        not math.isfinite(target_completion)
        or not math.isfinite(target_hold)
        or target_completion <= 0
        or target_hold < 0
        or not float(completion_window[0]) <= target_completion <= float(completion_window[1])
        or not math.isclose(target_completion + target_hold, duration, abs_tol=0.05)
    ):
        raise ValueError(f"{beat} action and terminal timing budgets conflict")
    initial_anchor = timing_contract.get("initial_anchor") or {}
    if initial_anchor.get("present") and float(initial_anchor.get("story_time_s") or 0) != 0:
        raise ValueError(f"{beat} initial anchor must consume zero story time")

    ordered_groups = [dict(group) for group in action_groups]
    if not ordered_groups:
        raise ValueError(f"{beat} action groups are missing")
    orders = [group.get("order") for group in ordered_groups]
    if orders != list(range(1, len(ordered_groups) + 1)):
        raise ValueError(f"{beat} action group order is not canonical")
    group_ids = [str(group.get("action_group_id") or "").strip() for group in ordered_groups]
    if any(not value for value in group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError(f"{beat} action group ids are missing or duplicated")
    for group, group_id in zip(ordered_groups, group_ids, strict=True):
        for hash_field in ("actions_sha256", "group_sha256"):
            if not _SHA256_RE.fullmatch(str(group.get(hash_field) or "")):
                raise ValueError(f"{group_id} {hash_field} is invalid")

    samples = [dict(sample) for sample in pose_samples]
    if not samples:
        raise ValueError(f"{beat} pose samples are missing")
    sample_ids = [str(sample.get("sample_id") or "").strip() for sample in samples]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{beat} pose sample ids are missing or duplicated")
    sample_group_ids = [str(sample.get("action_group_id") or "") for sample in samples]
    if set(sample_group_ids) != set(group_ids):
        raise ValueError(f"{beat} pose samples do not cover every action group")
    if any(group_id not in group_ids for group_id in sample_group_ids):
        raise ValueError(f"{beat} pose samples contain a foreign action group")
    if [sample.get("sample_index") for sample in samples] != list(range(1, len(samples) + 1)):
        raise ValueError(f"{beat} pose sample order is not canonical")
    if _ordered_unique(sample_group_ids) != group_ids:
        raise ValueError(f"{beat} pose samples cross canonical action group order")
    for sample in samples:
        pose_contract = sample.get("pose_contract")
        if not isinstance(pose_contract, Mapping) or pose_contract.get("secondary_beat_id") != beat:
            raise ValueError(f"{beat} pose sample contains future or foreign action data")

    action_parts = [
        part.strip()
        for part in _ACTION_SEPARATOR_RE.split(str(action_prompt or "").strip())
        if part.strip()
    ]
    if action_parts and len(action_parts) != len(ordered_groups):
        raise ValueError(
            f"{beat} action prompt has {len(action_parts)} ordered actions for "
            f"{len(ordered_groups)} groups"
        )

    rendered_groups: list[dict[str, Any]] = []
    for index, (group, group_id) in enumerate(zip(ordered_groups, group_ids, strict=True)):
        group_samples = [sample for sample in samples if sample.get("action_group_id") == group_id]
        timing_roles = _ordered_unique([sample.get("timing_role") for sample in group_samples])
        pose_contracts = [
            sample.get("pose_contract")
            for sample in group_samples
            if isinstance(sample.get("pose_contract"), Mapping)
        ]
        mechanics: dict[str, Any] = {}
        for pose_contract in pose_contracts:
            candidate = pose_contract.get("body_mechanics")
            if isinstance(candidate, Mapping) and candidate:
                mechanics = {
                    key: candidate[key]
                    for key in (
                        "technique",
                        "footwork",
                        "torso",
                        "weight_shift",
                        "direction",
                        "contact",
                        "end_pose",
                    )
                    if candidate.get(key) not in (None, "", [])
                }
                break
        pose_families = _ordered_unique(
            [contract.get("pose_family") for contract in pose_contracts]
        )
        performers = _ordered_unique(
            [
                performer
                for contract in pose_contracts
                for performer in (contract.get("actor_roles") or contract.get("performers") or [])
            ]
        )
        targets = _ordered_unique(
            [
                target
                for contract in pose_contracts
                for target in (contract.get("object_roles") or contract.get("targets") or [])
            ]
        )
        trajectory = _trajectory_summary(
            pose_contracts,
            pose_families=pose_families,
            targets=targets,
        )
        for key, value in trajectory.items():
            mechanics.setdefault(key, value)
        rendered_groups.append(
            {
                "action_group_id": group_id,
                "order": group["order"],
                "timing_role": (
                    "initial_anchor" if timing_roles == ["initial_anchor"] else "story_action"
                ),
                "action": (
                    action_parts[index]
                    if action_parts
                    else str(mechanics.get("technique") or "/".join(pose_families)).strip()
                ),
                "sample_ids": [str(sample["sample_id"]) for sample in group_samples],
                "pose_families": pose_families,
                "performers": performers,
                "targets": targets,
                "observable_mechanics": mechanics,
                "lineage": _validate_lineage(group, group_id=group_id),
                "actions_sha256": group["actions_sha256"],
                "group_sha256": group["group_sha256"],
            }
        )

    manifest_projection: list[dict[str, Any]] = []
    for item in media_manifest:
        prompt_index = str(item.get("prompt_index") or "").strip()
        responsibility = str(item.get("responsibility") or "").strip()
        media_sha256 = str(item.get("sha256") or "").strip()
        if not prompt_index or not responsibility:
            raise ValueError(f"{beat} media role or resolved prompt index is missing")
        if not _SHA256_RE.fullmatch(media_sha256):
            raise ValueError(f"{beat} media {prompt_index} hash is missing or invalid")
        manifest_projection.append(
            {
                "prompt_index": prompt_index,
                "responsibility": responsibility,
                "character_id": item.get("character_id"),
                "narrative_cell_ids": list(item.get("narrative_cell_ids") or []),
                "performance_cell_ids": list(item.get("performance_cell_ids") or []),
                "sha256": media_sha256,
            }
        )
    if not any(item["responsibility"] == "storyboard_pose_atlas" for item in manifest_projection):
        raise ValueError(f"{beat} action execution brief lacks pose-atlas authority")
    payload: dict[str, Any] = {
        "schema": ACTION_EXECUTION_BRIEF_SCHEMA,
        "beat_id": beat,
        "duration_s": duration,
        "start_state": str(start_state or "").strip(),
        "end_state": str(end_state or "").strip(),
        "initial_anchor": dict(initial_anchor),
        "initial_anchor_sample_ids": [
            str(sample["sample_id"])
            for sample in samples
            if sample.get("timing_role") == "initial_anchor"
        ],
        "completion_window_s": [
            float(completion_window[0]),
            float(completion_window[1]),
        ],
        "target_completion_s": target_completion,
        "terminal_hold": dict(terminal_hold),
        "terminal_sample_ids": [
            str(sample["sample_id"])
            for sample in samples
            if sample.get("timing_role") == "terminal_hold"
        ],
        "ordered_action_group_ids": group_ids,
        "action_groups": rendered_groups,
        "primary_camera": _camera_summary(prompt_context or {}, samples),
        "media_roles": manifest_projection,
        "canonical_visual_contract_sha256": canonical_visual_contract_sha256,
        "prompt_projection_policy_sha256": PROMPT_PROJECTION_POLICY_SHA256,
    }
    payload["brief_sha256"] = _sha256_payload(payload)
    return payload


def render_action_execution_brief(brief: Mapping[str, Any]) -> str:
    if brief.get("schema") != ACTION_EXECUTION_BRIEF_SCHEMA:
        raise ValueError("action execution brief schema is unsupported")
    expected_hash = brief.get("brief_sha256")
    unsigned = {key: value for key, value in brief.items() if key != "brief_sha256"}
    if not _SHA256_RE.fullmatch(str(expected_hash or "")) or expected_hash != _sha256_payload(
        unsigned
    ):
        raise ValueError("action execution brief hash is invalid")
    lines = [
        ACTION_EXECUTION_BRIEF_MARKER,
        (
            f"当前分镜={brief['beat_id']}；"
            f"唯一允许动作组数量={len(brief['ordered_action_group_ids'])}；"
            f"视频总时长={brief['duration_s']:g}秒。"
        ),
        "首帧只表示t=0初始状态；首帧出现后立即开始主体动作，禁止准备、戒备停留、慢动作或回到首帧姿态。",
    ]
    media_by_role: dict[str, list[str]] = {}
    for item in brief.get("media_roles") or []:
        role = str(item.get("responsibility") or "").strip()
        prompt_index = str(item.get("prompt_index") or "").strip()
        if role and prompt_index:
            media_by_role.setdefault(role, []).append(prompt_index)
    role_lines = []
    for role, explanation in (
        ("character_identity_board", "只锁定身份外观，不锁定姿态"),
        ("cinematic_composition", "只锁定t=0构图，不得冻结动作"),
        ("predecessor_tail_video", "只锁定前序连续状态，不得回放"),
        ("character_performance_guide", "只提供当前角色姿态与握持参考"),
        ("storyboard_pose_atlas", "是当前动作顺序、根位移和重心轨迹权威"),
        ("terminal_pose_reference", "只约束完整动作后的精确终态"),
        ("ordered_tail_frame", "只提供前序真实尾帧连续性"),
    ):
        indexes = media_by_role.get(role) or []
        if indexes:
            role_lines.append("、".join(indexes) + explanation)
    if role_lines:
        lines.append("媒体执行职责：" + "；".join(role_lines) + "。")
    initial_samples = list(brief.get("initial_anchor_sample_ids") or [])
    if (brief.get("initial_anchor") or {}).get("present"):
        if not initial_samples:
            raise ValueError("action execution brief initial anchor samples are missing")
        lines.append(
            "零时长初始锚点="
            + "→".join(initial_samples)
            + "；它已由首帧成立，视频时间从紧随其后的动态姿态开始。"
        )
    for group in brief["action_groups"]:
        sample_range = "→".join(group["sample_ids"])
        if group["timing_role"] == "initial_anchor":
            lines.append(
                f"{group['action_group_id']}（{sample_range}）=零时长初始锚点："
                f"{group['action']}；只确认起始姿态，不得占用动作时间。"
            )
            continue
        mechanics = group.get("observable_mechanics") or {}
        observable = "；".join(
            f"{key}={_canonical_json(value) if isinstance(value, (dict, list)) else value}"
            for key, value in mechanics.items()
            if value not in (None, "", [])
        )
        lines.append(
            f"{group['action_group_id']}（{sample_range}）必须完整可见执行："
            f"{group['action']}" + (f"；{observable}" if observable else "") + "。"
        )
    completion = brief["completion_window_s"]
    terminal = brief["terminal_hold"]
    lines.extend(
        [
            (
                f"所有动态动作必须在{completion[0]:g}～{completion[1]:g}秒完成；"
                f"随后只保持已完成的终态约{float(terminal.get('target_duration_s') or 0):g}秒。"
            ),
            f"要求终态：{brief['end_state']}。终态保持不得回放、重置或返回初始戒备姿态。",
            f"唯一主运镜：{brief['primary_camera']}；运镜只能辅助，不能替代主体的肢体、重心、接触与根位移。",
            "动作图集和姿态板仅为运动控制：不得把多姿态画成克隆、分栏、拼贴、网格、文字、序号、箭头或边框。",
        ]
    )
    return "\n".join(lines)


def project_action_first_prompt(
    *,
    media_index_preamble: str,
    media_role_isolation: str,
    action_brief_text: str,
    identity_projection: str,
    prompt_context: Mapping[str, Any],
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """Build named prompt sections without arbitrary string truncation."""
    output_constraints = (
        "[honcut.phase6-output-constraints.v1]\n"
        "只生成一个连续电影画面；保持正确人物数量、同一身份、自然人体运动、正确肢体与手指。"
        "禁止克隆、瞬移、漂浮、变形、闪烁、颜色漂移、字幕、文字、Logo、水印、网格、分栏、箭头和编号。"
    )
    mandatory_sections = [
        ("media_index", media_index_preamble.strip()),
        ("media_role_isolation", media_role_isolation.strip()),
        ("action_execution_brief", action_brief_text.strip()),
        ("identity_projection", identity_projection.strip()),
        ("output_constraints", output_constraints),
    ]
    scene_values = _ordered_unique(
        [
            prompt_context.get("where"),
            prompt_context.get("visual"),
            prompt_context.get("lighting_description"),
        ]
    )
    optional_sections: list[tuple[str, str]] = []
    if scene_values:
        optional_sections.append(("scene", "[honcut.phase6-scene.v1]\n" + "；".join(scene_values)))
    emotion = str(prompt_context.get("emotion") or "").strip()
    if emotion:
        optional_sections.append(
            (
                "emotion",
                "[honcut.phase6-emotion.v1]\n情绪只通过当前动作内的表情、呼吸与重心表现："
                + emotion,
            )
        )
    audio_values = _ordered_unique(
        [
            prompt_context.get("dialogue"),
            prompt_context.get("sound_effect"),
            prompt_context.get("background_music"),
            prompt_context.get("music"),
            prompt_context.get("audio"),
            prompt_context.get("sound"),
        ]
    )
    if audio_values:
        optional_sections.append(("audio", "[honcut.phase6-audio.v1]\n" + "；".join(audio_values)))

    budget = resolve_prompt_budget(
        provider=provider,
        model=model,
        purpose="video_generation",
    )
    mandatory_text = "\n".join(text for _name, text in mandatory_sections if text)
    if len(mandatory_text) >= budget.hard_chars:
        raise PromptBudgetExceededError(
            "mandatory Phase 6 action-first prompt sections exceed the provider hard limit: "
            f"chars={len(mandatory_text)}, hard={budget.hard_chars}"
        )
    included = list(mandatory_sections)
    omitted: list[str] = []
    current = mandatory_text
    for name, section in optional_sections:
        candidate = f"{current}\n{section}" if current else section
        if len(candidate) < budget.hard_chars:
            included.append((name, section))
            current = candidate
        else:
            omitted.append(name)
    metadata = {
        "schema": PROMPT_PROJECTION_SCHEMA,
        "policy_sha256": PROMPT_PROJECTION_POLICY_SHA256,
        "hard_chars": budget.hard_chars,
        "soft_chars": budget.soft_chars,
        "total_chars": len(current),
        "section_chars": {name: len(text) for name, text in included},
        "omitted_optional_sections": omitted,
    }
    metadata["projection_sha256"] = _sha256_payload(metadata)
    return current, metadata
