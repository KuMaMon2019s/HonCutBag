"""Deterministic, identity-neutral pose semantics for Phase 2 story guides."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from PIL import Image, ImageDraw, ImageFont

POSE_CONTRACT_SCHEMA = "honcut.storyboard-guide-pose-contract.v1"

_POSE_CLASSIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("throw", ("摔", "投掷", "甩向", "throw", "toss")),
    ("grab_control", ("抓住", "扣住", "控制", "擒", "grab", "grapple", "control")),
    ("kick", ("踢", "蹬", "扫腿", "kick")),
    ("block", ("格挡", "抵挡", "防御", "架住", "护盾", "block", "parry", "shield")),
    (
        "evade",
        ("闪避", "躲", "后仰", "后倾", "侧身", "避开", "evade", "dodge", "avoid", "lean back"),
    ),
    (
        "strike",
        ("攻击", "挥砍", "挥拳", "刺", "重击", "反击", "attack", "strike", "punch", "swing"),
    ),
    ("fall_land", ("跌倒", "倒地", "落地", "跃下", "坠落", "fall", "land", "drop")),
    (
        "prop_use",
        ("使用", "操作", "启动", "激活", "发射", "展开", "use", "operate", "activate", "fire"),
    ),
    (
        "locomotion",
        (
            "走",
            "跑",
            "冲刺",
            "靠近",
            "进入",
            "踏入",
            "移动",
            "滑入",
            "驶入",
            "walk",
            "run",
            "sprint",
            "enter",
            "approach",
            "move",
        ),
    ),
    ("reveal", ("现身", "出现", "显现", "暴露", "reveal", "appear", "emerge")),
    ("prop_hold", ("手持", "握", "持", "拿", "举起", "hold", "wield", "carry")),
    ("ready", ("准备", "戒备", "对峙", "架势", "站稳", "ready", "stance")),
)

_DIRECTION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("left", ("向左", "左移", "左侧", "leftward", "to the left", "left")),
    ("right", ("向右", "右移", "右侧", "rightward", "to the right", "right")),
    ("backward", ("向后", "后撤", "后退", "后仰", "backward", "back step", "lean back")),
    (
        "forward",
        ("向前", "前冲", "靠近", "进入", "踏入", "冲刺", "forward", "approach", "enter", "sprint"),
    ),
    ("up", ("向上", "跃起", "抬起", "升起", "upward", "jump", "raise")),
    ("down", ("向下", "落地", "砸地", "下坠", "downward", "land", "drop")),
)

_ACTOR_MARKERS = (
    "角色",
    "人物",
    "男子",
    "男性",
    "女子",
    "女性",
    "敌人",
    "战斗人员",
    "士兵",
    "乘客",
    "actor",
    "guard",
    "agent",
    "person",
    "man",
    "woman",
    "soldier",
    "fighter",
)

_POSE_POLICY = {
    "schema": "honcut.storyboard-guide-pose-policy.v1",
    "classifiers": _POSE_CLASSIFIERS,
    "direction_markers": _DIRECTION_MARKERS,
    "actor_markers": _ACTOR_MARKERS,
    "role_resolution": "source_actor_roster_then_controlled_actor_markers_v1",
    "geometry": "normalized_joint_templates_v1",
    "phase_samples": {"start": 0.55, "action_progress": 1.0, "end": 0.76},
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


POSE_POLICY_SHA256 = _canonical_sha256(_POSE_POLICY)


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item or "").strip()]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        folded = value.casefold()
        if folded not in seen:
            seen.add(folded)
            result.append(value)
    return result


def _is_explicit_actor_role(value: str) -> bool:
    folded = value.casefold()
    return any(marker in folded for marker in _ACTOR_MARKERS)


def _pose_family(action_text: str, mechanics: Mapping[str, Any]) -> tuple[str, str]:
    structured = " ".join(
        str(mechanics.get(key) or "")
        for key in ("technique", "footwork", "torso", "weight_shift", "contact", "end_pose")
    )
    haystack = f"{structured} {action_text}".casefold()
    for family, markers in _POSE_CLASSIFIERS:
        marker = next((item for item in markers if item.casefold() in haystack), None)
        if marker is not None:
            return family, marker
    return "spatial", "no_controlled_marker"


def _direction(action_text: str, mechanics: Mapping[str, Any]) -> tuple[str, str]:
    structured = " ".join(
        str(mechanics.get(key) or "")
        for key in ("direction", "footwork", "torso", "weight_shift", "end_pose")
    )
    haystack = f"{structured} {action_text}".casefold()
    for direction, markers in _DIRECTION_MARKERS:
        marker = next((item for item in markers if item.casefold() in haystack), None)
        if marker is not None:
            return direction, marker
    return "right", "canonical_default_right"


def _camera_vector(camera_movement: str) -> dict[str, int]:
    folded = camera_movement.casefold()
    if any(marker in folded for marker in ("静止", "固定", "locked", "static")):
        return {"x": 0, "y": 0}
    if any(marker in folded for marker in ("左", "left")):
        return {"x": -1, "y": 0}
    if any(marker in folded for marker in ("右", "right")):
        return {"x": 1, "y": 0}
    if any(marker in folded for marker in ("上", "up", "rise", "crane")):
        return {"x": 0, "y": -1}
    if any(marker in folded for marker in ("下", "down", "drop")):
        return {"x": 0, "y": 1}
    if any(marker in folded for marker in ("拉远", "pull", "zoom out")):
        return {"x": -1, "y": 0}
    return {"x": 1, "y": 0}


def _action_vector(direction: str, *, static: bool) -> dict[str, int]:
    if static:
        return {"x": 0, "y": 0}
    return {
        "left": {"x": -1, "y": 0},
        "right": {"x": 1, "y": 0},
        "backward": {"x": -1, "y": 0},
        "forward": {"x": 1, "y": 0},
        "up": {"x": 0, "y": -1},
        "down": {"x": 0, "y": 1},
    }[direction]


_NEUTRAL = {
    "head": (0, -72),
    "neck": (0, -52),
    "left_shoulder": (-25, -40),
    "right_shoulder": (25, -40),
    "left_elbow": (-29, -8),
    "right_elbow": (29, -8),
    "left_hand": (-24, 20),
    "right_hand": (24, 20),
    "left_hip": (-15, 20),
    "right_hip": (15, 20),
    "left_knee": (-20, 52),
    "right_knee": (20, 52),
    "left_foot": (-24, 84),
    "right_foot": (24, 84),
}

_POSES: dict[str, dict[str, tuple[int, int]]] = {
    "spatial": _NEUTRAL,
    "ready": {
        **_NEUTRAL,
        "left_hand": (-8, -34),
        "right_hand": (18, -25),
        "left_knee": (-30, 53),
        "right_foot": (40, 82),
    },
    "locomotion": {
        **_NEUTRAL,
        "head": (8, -70),
        "neck": (5, -50),
        "left_elbow": (12, -8),
        "left_hand": (30, 18),
        "right_elbow": (35, -20),
        "right_hand": (52, -4),
        "left_knee": (24, 48),
        "left_foot": (48, 76),
        "right_knee": (-25, 58),
        "right_foot": (-43, 86),
    },
    "evade": {
        **_NEUTRAL,
        "head": (-28, -68),
        "neck": (-20, -47),
        "left_shoulder": (-42, -34),
        "right_shoulder": (8, -42),
        "left_hand": (-50, 9),
        "right_hand": (35, 2),
        "left_hip": (-10, 22),
        "right_hip": (19, 17),
        "left_knee": (-38, 55),
        "left_foot": (-57, 82),
    },
    "block": {
        **_NEUTRAL,
        "left_elbow": (-34, -44),
        "left_hand": (-7, -58),
        "right_elbow": (31, -42),
        "right_hand": (7, -50),
        "left_knee": (-30, 55),
        "right_foot": (39, 84),
    },
    "strike": {
        **_NEUTRAL,
        "right_elbow": (48, -38),
        "right_hand": (78, -39),
        "left_elbow": (-5, -30),
        "left_hand": (14, -22),
        "left_knee": (-31, 55),
        "right_foot": (42, 80),
    },
    "kick": {
        **_NEUTRAL,
        "left_elbow": (-37, -27),
        "left_hand": (-48, -4),
        "right_elbow": (31, -38),
        "right_hand": (20, -18),
        "left_knee": (-8, 46),
        "left_foot": (70, 43),
        "right_knee": (18, 55),
        "right_foot": (25, 86),
    },
    "grab_control": {
        **_NEUTRAL,
        "left_elbow": (-5, -25),
        "left_hand": (35, -16),
        "right_elbow": (18, -24),
        "right_hand": (48, -8),
        "left_knee": (-31, 56),
        "right_foot": (42, 82),
    },
    "throw": {
        **_NEUTRAL,
        "head": (12, -69),
        "left_shoulder": (-34, -34),
        "right_shoulder": (29, -45),
        "left_hand": (35, -8),
        "right_hand": (60, -33),
        "left_knee": (-39, 51),
        "left_foot": (-52, 78),
        "right_foot": (41, 87),
    },
    "fall_land": {
        **_NEUTRAL,
        "head": (11, -43),
        "neck": (5, -25),
        "left_shoulder": (-29, -16),
        "right_shoulder": (32, -20),
        "left_hand": (-45, 28),
        "right_hand": (50, 25),
        "left_hip": (-18, 24),
        "right_hip": (18, 24),
        "left_knee": (-45, 51),
        "right_knee": (48, 50),
        "left_foot": (-65, 70),
        "right_foot": (68, 70),
    },
    "prop_use": {
        **_NEUTRAL,
        "left_elbow": (-8, -20),
        "left_hand": (17, -34),
        "right_elbow": (20, -18),
        "right_hand": (18, -34),
        "left_knee": (-29, 54),
        "right_foot": (37, 82),
    },
    "prop_hold": {
        **_NEUTRAL,
        "left_elbow": (-18, -16),
        "left_hand": (-6, 3),
        "right_elbow": (18, -16),
        "right_hand": (6, 3),
    },
    "reveal": {
        **_NEUTRAL,
        "left_elbow": (-43, -26),
        "left_hand": (-61, -12),
        "right_elbow": (43, -26),
        "right_hand": (61, -12),
        "left_foot": (-36, 84),
        "right_foot": (36, 84),
    },
}


def _phase_geometry(family: str, stage: str, direction: str) -> dict[str, tuple[int, int]]:
    target = _POSES[family]
    factor = float(_POSE_POLICY["phase_samples"].get(stage, 1.0))
    geometry: dict[str, tuple[int, int]] = {}
    for joint, neutral in _NEUTRAL.items():
        target_point = target[joint]
        x = round(neutral[0] + (target_point[0] - neutral[0]) * factor)
        y = round(neutral[1] + (target_point[1] - neutral[1]) * factor)
        if direction in {"left", "backward"}:
            x = -x
        geometry[joint] = (x, y)
    return geometry


def _global_geometry(
    family: str,
    stage: str,
    direction: str,
    actor_count: int,
) -> list[dict[str, Any]]:
    if actor_count <= 0:
        return []
    span = 700
    centers = (
        [500]
        if actor_count == 1
        else [round(150 + span * index / (actor_count - 1)) for index in range(actor_count)]
    )
    scale = max(0.42, min(0.88, 2.2 / actor_count))
    actors = []
    for index, center_x in enumerate(centers):
        actor_family = family
        actor_direction = direction
        if index > 0 and family in {"strike", "kick", "grab_control", "throw"}:
            actor_family = "evade" if family != "throw" else "fall_land"
            actor_direction = "left" if direction in {"right", "forward"} else "right"
        local = _phase_geometry(actor_family, stage, actor_direction)
        joints = {
            joint: [
                max(35, min(965, round(center_x + point[0] * scale))),
                max(180, min(925, round(520 + point[1] * scale * 3.1))),
            ]
            for joint, point in local.items()
        }
        actors.append(
            {
                "slot": index + 1,
                "pose_family": actor_family,
                "facing": actor_direction,
                "joints": joints,
            }
        )
    return actors


def _matching_mechanics(
    beat: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = beat.get("body_action_contract")
    raw_body_beats = contract.get("beats") if isinstance(contract, Mapping) else None
    body_beats = raw_body_beats if isinstance(raw_body_beats, list) else []
    source_indexes = {
        int(index)
        for unit in units
        for index in (unit.get("source_micro_action_indexes") or [])
        if isinstance(index, int) and index > 0
    }
    matches = [
        item
        for item in body_beats
        if isinstance(item, Mapping) and int(item.get("micro_action_index") or 0) in source_indexes
    ]
    if not matches and len(body_beats) == 1 and isinstance(body_beats[0], Mapping):
        matches = [body_beats[0]]
    combined: dict[str, Any] = {}
    for key in (
        "performer",
        "technique",
        "side",
        "limbs",
        "footwork",
        "torso",
        "weight_shift",
        "direction",
        "contact",
        "end_pose",
    ):
        values = []
        for item in matches:
            raw = item.get(key)
            values.extend(_strings(raw))
        if values:
            combined[key] = values if key == "limbs" else " | ".join(_dedupe(values))
    matched_beats = [
        {
            "micro_action_index": int(item.get("micro_action_index") or 0),
            "body_action_sha256": _canonical_sha256(
                {
                    key: item.get(key)
                    for key in (
                        "micro_action",
                        "performer",
                        "technique",
                        "side",
                        "limbs",
                        "footwork",
                        "torso",
                        "weight_shift",
                        "direction",
                        "contact",
                        "end_pose",
                    )
                }
            ),
        }
        for item in matches
    ]
    return combined, matched_beats


def _validated_units(beat: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw_units = beat.get("generation_action_units")
    if isinstance(raw_units, list) and raw_units:
        units = [dict(item) for item in raw_units if isinstance(item, Mapping)]
        if len(units) != len(raw_units):
            raise ValueError("generation action units must be objects")
        unit_ids = [str(item.get("unit_id") or "").strip() for item in units]
        if any(not unit_id for unit_id in unit_ids) or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("generation action units need unique non-empty unit IDs")
        for unit in units:
            if not _strings(unit.get("actions")):
                raise ValueError(f"{unit['unit_id']} has no canonical actions")
            if not (
                str(unit.get("source_action_unit_id") or "").strip()
                or int(unit.get("source_event_id") or 0) > 0
                or any(
                    isinstance(index, int) and index > 0
                    for index in (unit.get("source_generation_unit_indexes") or [])
                )
                or any(
                    isinstance(index, int) and not isinstance(index, bool) and index >= 0
                    for index in (unit.get("ledger_indexes") or [])
                )
            ):
                raise ValueError(f"{unit['unit_id']} has no source action/event lineage")
            for field in (
                "source_micro_action_indexes",
                "source_generation_unit_indexes",
            ):
                indexes = unit.get(field) or []
                if not isinstance(indexes, list) or any(
                    isinstance(index, bool) or not isinstance(index, int) or index <= 0
                    for index in indexes
                ):
                    raise ValueError(f"{unit['unit_id']} has invalid {field} lineage")
                if len(indexes) != len(set(indexes)):
                    raise ValueError(f"{unit['unit_id']} has duplicate {field} lineage")
            ledger_indexes = unit.get("ledger_indexes") or []
            if not isinstance(ledger_indexes, list) or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in ledger_indexes
            ):
                raise ValueError(f"{unit['unit_id']} has invalid ledger_indexes lineage")
            if len(ledger_indexes) != len(set(ledger_indexes)):
                raise ValueError(f"{unit['unit_id']} has duplicate ledger_indexes lineage")
        return units, "canonical"
    raise ValueError("storyboard beat is missing canonical generation action units")


def _partition_units(units: list[dict[str, Any]], cell_count: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    unit_count = len(units)
    for cell_index in range(cell_count):
        start = cell_index * unit_count // cell_count
        end = (cell_index + 1) * unit_count // cell_count
        if end <= start:
            groups.append([units[min(start, unit_count - 1)]])
        else:
            groups.append(units[start:end])
    covered = {str(unit["unit_id"]) for group in groups for unit in group}
    expected = {str(unit["unit_id"]) for unit in units}
    if covered != expected:
        raise ValueError("Gxx action partition lost canonical generation units")
    return groups


def compile_pose_contracts(
    beat: Mapping[str, Any],
    cells: list[dict[str, Any]],
    *,
    known_actor_roles: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Attach one source-bound, deterministic pose contract to every Gxx cell."""
    units, lineage_status = _validated_units(beat)
    groups = _partition_units(units, len(cells))
    source_actor_roles = {value.casefold() for value in known_actor_roles if value}
    result: list[dict[str, Any]] = []
    for cell, group in zip(cells, groups, strict=True):
        actions = [action for unit in group for action in _strings(unit.get("actions"))]
        action_text = " → ".join(actions)
        mechanics, matched_body_action_beats = _matching_mechanics(beat, group)
        family, family_evidence = _pose_family(action_text, mechanics)
        direction, direction_evidence = _direction(action_text, mechanics)
        performers = _dedupe(
            [value for unit in group for value in _strings(unit.get("performers"))]
        )
        targets = _dedupe([value for unit in group for value in _strings(unit.get("targets"))])
        actor_roles = _dedupe(
            [
                value
                for value in performers
                if value.casefold() in source_actor_roles or _is_explicit_actor_role(value)
            ]
        )
        if family in {"strike", "kick", "grab_control", "throw", "block", "evade"}:
            actor_roles = _dedupe(
                actor_roles
                + [
                    value
                    for value in targets
                    if value.casefold() in source_actor_roles or _is_explicit_actor_role(value)
                ]
            )
        if family != "spatial" and not actor_roles and not performers and not targets:
            actor_roles = _dedupe(_strings(beat.get("character_ids") or beat.get("who")))
        object_roles = _dedupe(
            [value for value in performers + targets if value not in actor_roles]
        )
        static_spatial_state = family == "spatial" and not actor_roles
        actors = _global_geometry(
            family,
            str(cell.get("stage") or "action_progress"),
            direction,
            len(actor_roles),
        )
        for actor, role in zip(actors, actor_roles, strict=True):
            actor["role_ref"] = role
            actor["pose_fingerprint"] = _canonical_sha256(actor)
        geometry = {
            "actors": actors,
            "objects": [
                {"slot": index + 1, "role_ref": role} for index, role in enumerate(object_roles)
            ],
        }
        action_bindings = [
            {
                "unit_id": str(unit["unit_id"]),
                "source_action_unit_id": (str(unit.get("source_action_unit_id") or "") or None),
                "source_event_id": int(unit.get("source_event_id") or 0) or None,
                "source_micro_action_indexes": list(unit.get("source_micro_action_indexes") or []),
                "source_generation_unit_indexes": list(
                    unit.get("source_generation_unit_indexes") or []
                ),
                "source_ledger_indexes": list(unit.get("ledger_indexes") or []),
                "action_sha256": _canonical_sha256(_strings(unit.get("actions"))),
            }
            for unit in group
        ]
        action_vector = _action_vector(direction, static=static_spatial_state)
        camera_vector = _camera_vector(str(cell.get("camera_movement") or ""))
        pose_fingerprint = _canonical_sha256(
            {
                "family": family,
                "stage": str(cell.get("stage") or ""),
                "direction": direction,
                "actors": actors,
                "objects": geometry["objects"],
                "action_vector": action_vector,
                "camera_vector": camera_vector,
                "static_spatial_state": static_spatial_state,
            }
        )
        contract = {
            "schema": POSE_CONTRACT_SCHEMA,
            "pose_policy_sha256": POSE_POLICY_SHA256,
            "cell_id": str(cell.get("label") or ""),
            "primary_shot_id": str(cell.get("primary_shot_id") or ""),
            "secondary_beat_id": str(cell.get("secondary_beat_id") or ""),
            "stage": str(cell.get("stage") or ""),
            "lineage_status": lineage_status,
            "action_bindings": action_bindings,
            "performers": performers,
            "targets": targets,
            "action_text_sha256": _canonical_sha256(actions),
            "body_mechanics": mechanics,
            "matched_body_action_beats": matched_body_action_beats,
            "pose_family": family,
            "pose_phase": str(cell.get("stage") or ""),
            "classification_evidence": family_evidence,
            "direction": direction,
            "direction_evidence": direction_evidence,
            "actor_roles": actor_roles,
            "object_roles": object_roles,
            "action_vector": action_vector,
            "camera_vector": camera_vector,
            "static_spatial_state": static_spatial_state,
            "geometry": geometry,
            "pose_fingerprint": pose_fingerprint,
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        enriched = dict(cell)
        enriched["neutral_subject_count"] = len(actor_roles)
        enriched["pose_contract"] = contract
        result.append(enriched)
    return result


def validate_pose_contract(contract: Mapping[str, Any], *, cell_id: str, beat_id: str) -> None:
    if (
        contract.get("schema") != POSE_CONTRACT_SCHEMA
        or contract.get("pose_policy_sha256") != POSE_POLICY_SHA256
        or contract.get("cell_id") != cell_id
        or contract.get("secondary_beat_id") != beat_id
    ):
        raise ValueError("story guide pose contract identity is invalid")
    stored_contract_sha = str(contract.get("contract_sha256") or "")
    unhashed = dict(contract)
    unhashed.pop("contract_sha256", None)
    if stored_contract_sha != _canonical_sha256(unhashed):
        raise ValueError("story guide pose contract hash mismatch")
    geometry = contract.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("story guide pose geometry is missing")
    actors = geometry.get("actors")
    if not isinstance(actors, list):
        raise ValueError("story guide actor geometry must be an array")
    if len(actors) != len(contract.get("actor_roles") or []):
        raise ValueError("story guide actor roles do not match geometry")
    for actor in actors:
        joints = actor.get("joints") if isinstance(actor, Mapping) else None
        if not isinstance(joints, Mapping) or set(joints) != set(_NEUTRAL):
            raise ValueError("story guide actor joints are incomplete")
        for point in joints.values():
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(not isinstance(value, int) or not 0 <= value <= 1000 for value in point)
            ):
                raise ValueError("story guide actor joint is outside normalized bounds")
        actor_payload = dict(actor)
        actor_fingerprint = str(actor_payload.pop("pose_fingerprint", ""))
        if actor_fingerprint != _canonical_sha256(actor_payload):
            raise ValueError("story guide actor pose fingerprint mismatch")
    expected_fingerprint = _canonical_sha256(
        {
            "family": contract.get("pose_family"),
            "stage": contract.get("stage"),
            "direction": contract.get("direction"),
            "actors": actors,
            "objects": geometry.get("objects") or [],
            "action_vector": contract.get("action_vector"),
            "camera_vector": contract.get("camera_vector"),
            "static_spatial_state": bool(contract.get("static_spatial_state")),
        }
    )
    if contract.get("pose_fingerprint") != expected_fingerprint:
        raise ValueError("story guide pose fingerprint mismatch")
    if not contract.get("action_bindings"):
        raise ValueError("story guide pose has no action-unit binding")


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    size = max(10, width * 3)
    base = (end[0] - ux * size, end[1] - uy * size)
    draw.polygon(
        [
            end,
            (round(base[0] - uy * size * 0.55), round(base[1] + ux * size * 0.55)),
            (round(base[0] + uy * size * 0.55), round(base[1] - ux * size * 0.55)),
        ],
        fill=fill,
    )


def render_pose_cell(
    cell: Mapping[str, Any],
    *,
    width: int = 480,
    height: int = 270,
    font_factory: Callable[[int], ImageFont.ImageFont | ImageFont.FreeTypeFont],
) -> Image.Image:
    """Render only normalized pose geometry; never read source storyboard pixels."""
    contract = cell.get("pose_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("story guide cell has no pose contract")
    validate_pose_contract(
        contract,
        cell_id=str(cell.get("label") or ""),
        beat_id=str(cell.get("secondary_beat_id") or ""),
    )
    background = (247, 247, 244)
    neutral = (116, 121, 126)
    faint = (196, 199, 201)
    action_red = (205, 48, 54)
    camera_blue = (42, 104, 190)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    border = max(3, round(min(width, height) / 90))
    draw.rectangle((2, 2, width - 3, height - 3), outline=faint, width=border)
    label = str(cell.get("label") or "G??")
    draw.rounded_rectangle(
        (14, 12, 186, 52), radius=9, fill=(231, 232, 230), outline=neutral, width=2
    )
    draw.text((25, 19), label, fill=(52, 55, 58), font=font_factory(24))

    geometry = contract["geometry"]
    for actor in geometry["actors"]:
        joints = {
            name: (
                round(point[0] * width / 1000),
                round(point[1] * height / 1000),
            )
            for name, point in actor["joints"].items()
        }
        head = joints["head"]
        radius = max(7, round(13 * min(1.0, 2.2 / max(1, len(geometry["actors"])))))
        draw.ellipse(
            (head[0] - radius, head[1] - radius, head[0] + radius, head[1] + radius),
            fill=(211, 213, 213),
            outline=neutral,
            width=2,
        )
        for start, end in (
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_hand"),
            ("neck", "left_hip"),
            ("neck", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_foot"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_foot"),
        ):
            draw.line((*joints[start], *joints[end]), fill=neutral, width=5)

    for index, _object in enumerate(geometry["objects"][:4]):
        left = 42 + index * 34
        draw.rectangle((left, 92, left + 22, 118), outline=neutral, width=3)
        draw.line((left + 4, 113, left + 18, 97), fill=faint, width=2)

    action_vector = contract["action_vector"]
    if action_vector != {"x": 0, "y": 0}:
        if int(action_vector["x"]):
            endpoints = ((95, 205), (width - 95, 195))
            start, end = endpoints if int(action_vector["x"]) > 0 else (endpoints[1], endpoints[0])
        else:
            endpoints = ((95, height - 55), (105, 86))
            start, end = endpoints if int(action_vector["y"]) < 0 else (endpoints[1], endpoints[0])
        _draw_arrow(draw, start, end, fill=action_red, width=6)

    camera_vector = contract["camera_vector"]
    if camera_vector != {"x": 0, "y": 0}:
        if int(camera_vector["x"]):
            endpoints = ((width - 205, 76), (width - 74, 76))
            start, end = endpoints if int(camera_vector["x"]) > 0 else (endpoints[1], endpoints[0])
        else:
            endpoints = ((width - 76, 130), (width - 76, 66))
            start, end = endpoints if int(camera_vector["y"]) < 0 else (endpoints[1], endpoints[0])
        _draw_arrow(draw, start, end, fill=camera_blue, width=6)

    draw.line((66, 111, 136, 132), fill=neutral, width=3)
    draw.ellipse((59, 104, 73, 118), outline=neutral, width=3)
    draw.line((width - 58, height - 50, width - 30, height - 22), fill=neutral, width=4)
    draw.line((width - 30, height - 50, width - 58, height - 22), fill=neutral, width=4)
    return image


def pose_contracts_sha256(cells: list[Mapping[str, Any]]) -> str:
    return _canonical_sha256([cell.get("pose_contract") for cell in cells])


def pose_fingerprints(cells: list[Mapping[str, Any]]) -> list[str]:
    return [str(cell["pose_contract"]["pose_fingerprint"]) for cell in cells]


def validate_pose_sequence(cells: list[Mapping[str, Any]], *, beat_id: str) -> None:
    for cell in cells:
        contract = cell.get("pose_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"{beat_id} guide cell lacks pose contract")
        validate_pose_contract(
            contract,
            cell_id=str(cell.get("label") or ""),
            beat_id=beat_id,
        )
    for previous, current in pairwise(cells):
        previous_contract = previous["pose_contract"]
        current_contract = current["pose_contract"]
        semantics_changed = (
            previous_contract["action_bindings"] != current_contract["action_bindings"]
            or previous_contract["stage"] != current_contract["stage"]
        )
        both_static = (
            previous_contract["static_spatial_state"] and current_contract["static_spatial_state"]
        )
        if (
            semantics_changed
            and not both_static
            and previous_contract["pose_fingerprint"] == current_contract["pose_fingerprint"]
        ):
            raise ValueError(f"{beat_id} has distinct action semantics collapsed to one pose")
