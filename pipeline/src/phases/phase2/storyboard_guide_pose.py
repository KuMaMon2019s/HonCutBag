"""Deterministic, identity-neutral pose semantics for Phase 2 story guides."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from PIL import Image, ImageDraw, ImageFont

POSE_CONTRACT_SCHEMA = "honcut.storyboard-guide-pose-contract.v3"

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
        (
            "攻击",
            "挥砍",
            "挥击",
            "挥棍",
            "斜挥",
            "挥出",
            "挥拳",
            "刺",
            "重击",
            "反击",
            "attack",
            "strike",
            "punch",
            "swing",
        ),
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
            "走进",
            "跑进",
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
    ("ready", ("准备", "戒备", "对峙", "架势", "站稳", "ready", "stance")),
    ("prop_hold", ("手持", "握", "持", "拿", "举起", "hold", "wield", "carry")),
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

_POSE_EVIDENCE_FIELDS = (
    "technique",
    "footwork",
    "torso",
    "weight_shift",
    "end_pose",
    "action_text",
    "contact",
)

_CHINESE_NEGATION_PREFIXES = (
    "无",
    "无实际",
    "没有",
    "未",
    "并无",
    "并未",
    "不",
    "并不",
    "非",
    "并非",
    "无需",
    "禁止",
    "避免",
)

_MECHANICS_MARKERS = {
    "center_drop": ("降低重心", "低重心", "重心下沉", "下沉", "crouch", "lower the center"),
    "lean_back": ("后仰", "后倾", "向后倾", "lean back", "backward-leaning"),
    "lean_forward": ("前倾", "向前倾", "lean forward", "forward-leaning"),
    "wide_stance": ("滑步", "侧步", "支撑", "站稳", "side-step", "sidestep", "wide stance"),
    "two_hand_hold": ("双手", "两手", "both hands", "two-handed"),
}

_POSE_POLICY = {
    "schema": "honcut.storyboard-guide-pose-policy.v3",
    "classifiers": _POSE_CLASSIFIERS,
    "direction_markers": _DIRECTION_MARKERS,
    "actor_markers": _ACTOR_MARKERS,
    "evidence_fields": _POSE_EVIDENCE_FIELDS,
    "negation_prefixes": _CHINESE_NEGATION_PREFIXES,
    "mechanics_markers": _MECHANICS_MARKERS,
    "role_resolution": "source_actor_roster_then_controlled_actor_markers_v1",
    "geometry": "normalized_joint_templates_with_mechanics_v2",
    "phase_samples": {"start": 0.2, "action_progress": 0.7, "end": 1.0},
    "minimum_adjacent_joint_delta": 2,
    "minimum_action_span_joint_delta": 12,
    "initial_anchor": {
        "eligible_family": "ready",
        "requires_later_dynamic_family": True,
        "cell_count": 1,
        "pose_progress": 1.0,
        "story_time_weight": 0.0,
    },
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


def _match_is_negated(text: str, marker_start: int) -> bool:
    """Return whether a lexical match is governed by a local negation clause."""
    prefix = text[max(0, marker_start - 64) : marker_start].casefold()
    prefix = re.split(r"(?:但是|但|而是|however|\bbut\b)", prefix)[-1]
    chinese_clause = re.split(r"[，。；！？,.;!?]", prefix)[-1]
    compact = re.sub(r"\s+", "", chinese_clause)
    if any(
        re.search(re.escape(negation) + r"[^，。；！？,.;!?]{0,12}$", compact)
        for negation in _CHINESE_NEGATION_PREFIXES
    ):
        return True
    english_clause = re.split(r"[,.;!?]", prefix)[-1]
    return bool(
        re.search(
            r"(?:\bno\b|\bnot\b|\bwithout\b|\bnever\b|\bneither\b|\bnor\b|"
            r"\bdo not\b|\bdoes not\b|\bdid not\b|\bis not\b|\bwas not\b|"
            r"\bdon't\b|\bdoesn't\b|\bdidn't\b)"
            r"(?:\s+[\w-]+){0,5}\s*$",
            english_clause,
        )
    )


def _marker_occurrences(text: str, marker: str) -> list[tuple[int, bool]]:
    folded = text.casefold()
    result: list[tuple[int, bool]] = []
    start = 0
    needle = marker.casefold()
    while True:
        index = folded.find(needle, start)
        if index < 0:
            return result
        result.append((index, _match_is_negated(folded, index)))
        start = index + max(1, len(needle))


def _pose_family(action_text: str, mechanics: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    sources = {
        **{field: str(mechanics.get(field) or "") for field in _POSE_EVIDENCE_FIELDS},
        "action_text": action_text,
    }
    rejected: list[dict[str, str]] = []
    for field in _POSE_EVIDENCE_FIELDS:
        text = sources[field]
        for family, markers in _POSE_CLASSIFIERS:
            for marker in markers:
                rejected.extend(
                    {
                        "field": field,
                        "family": family,
                        "marker": marker,
                        "polarity": "negated",
                    }
                    for _index, negated in _marker_occurrences(text, marker)
                    if negated
                )
    for field in _POSE_EVIDENCE_FIELDS:
        text = sources[field]
        for family, markers in _POSE_CLASSIFIERS:
            for marker in markers:
                occurrences = _marker_occurrences(text, marker)
                positive = next((index for index, negated in occurrences if not negated), None)
                if positive is not None:
                    return family, {
                        "field": field,
                        "family": family,
                        "marker": marker,
                        "polarity": "positive",
                        "rejected_negated_matches": rejected,
                    }
    return "spatial", {
        "field": "none",
        "family": "spatial",
        "marker": "no_controlled_marker",
        "polarity": "none",
        "rejected_negated_matches": rejected,
    }


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


def _has_positive_marker(text: str, markers: Sequence[str]) -> bool:
    return any(
        not negated
        for marker in markers
        for _index, negated in _marker_occurrences(text, marker)
    )


def _mechanics_modifiers(mechanics: Mapping[str, Any], direction: str) -> dict[str, Any]:
    movement_text = " ".join(
        str(mechanics.get(key) or "")
        for key in ("technique", "footwork", "torso", "weight_shift", "direction", "end_pose")
    )
    contact_text = str(mechanics.get("contact") or "")
    center_drop = 18 if _has_positive_marker(movement_text, _MECHANICS_MARKERS["center_drop"]) else 0
    lean_back = _has_positive_marker(movement_text, _MECHANICS_MARKERS["lean_back"])
    lean_forward = _has_positive_marker(movement_text, _MECHANICS_MARKERS["lean_forward"])
    torso_lean = -24 if lean_back and not lean_forward else 18 if lean_forward and not lean_back else 0
    stance_width = 16 if _has_positive_marker(movement_text, _MECHANICS_MARKERS["wide_stance"]) else 0
    lead_step = 24 if stance_width and direction in {"right", "forward"} else -24 if stance_width else 0
    two_hand_hold = _has_positive_marker(
        f"{movement_text} {contact_text}",
        _MECHANICS_MARKERS["two_hand_hold"],
    )
    return {
        "center_drop": center_drop,
        "torso_lean": torso_lean,
        "stance_width": stance_width,
        "lead_step": lead_step,
        "two_hand_hold": two_hand_hold,
    }


def _mechanics_target(
    family: str,
    mechanics_modifiers: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    target = dict(_POSES[family])
    center_drop = int(mechanics_modifiers.get("center_drop") or 0)
    torso_lean = int(mechanics_modifiers.get("torso_lean") or 0)
    stance_width = int(mechanics_modifiers.get("stance_width") or 0)
    lead_step = int(mechanics_modifiers.get("lead_step") or 0)

    upper_joints = (
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hand",
        "right_hand",
    )
    for joint in upper_joints:
        x, y = target[joint]
        target[joint] = (x + torso_lean, y + center_drop)
    for joint in ("left_hip", "right_hip"):
        x, y = target[joint]
        target[joint] = (x + round(lead_step * 0.25), y + center_drop)
    for joint in ("left_knee", "right_knee"):
        x, y = target[joint]
        target[joint] = (x, y + round(center_drop * 0.6))
    left_x, left_y = target["left_foot"]
    right_x, right_y = target["right_foot"]
    target["left_foot"] = (left_x - stance_width, left_y)
    target["right_foot"] = (right_x + stance_width + lead_step, right_y)

    if mechanics_modifiers.get("two_hand_hold") and family in {
        "ready",
        "evade",
        "locomotion",
        "prop_hold",
        "prop_use",
    }:
        target["left_hand"] = (-13 + torso_lean, -5 + center_drop)
        target["right_hand"] = (13 + torso_lean, -5 + center_drop)
    return target


def _phase_geometry(
    family: str,
    stage: str,
    direction: str,
    *,
    pose_progress: float,
    mechanics_modifiers: Mapping[str, Any],
    origin_family: str | None = None,
    origin_direction: str = "right",
    origin_mechanics_modifiers: Mapping[str, Any] | None = None,
) -> dict[str, tuple[int, int]]:
    target = _mechanics_target(family, mechanics_modifiers)
    if direction in {"left", "backward"}:
        target = {joint: (-point[0], point[1]) for joint, point in target.items()}
    if origin_family is None:
        origin = _NEUTRAL
    else:
        origin = _mechanics_target(
            origin_family,
            origin_mechanics_modifiers or {},
        )
        if origin_direction in {"left", "backward"}:
            origin = {joint: (-point[0], point[1]) for joint, point in origin.items()}
    factor = max(0.0, min(1.0, pose_progress))
    geometry: dict[str, tuple[int, int]] = {}
    for joint, origin_point in origin.items():
        target_point = target[joint]
        x = round(origin_point[0] + (target_point[0] - origin_point[0]) * factor)
        y = round(origin_point[1] + (target_point[1] - origin_point[1]) * factor)
        geometry[joint] = (x, y)
    return geometry


def _root_motion(family: str, direction: str, pose_progress: float) -> tuple[int, int]:
    magnitude = {
        "locomotion": 110,
        "evade": 90,
        "strike": 42,
        "kick": 48,
        "grab_control": 38,
        "throw": 54,
        "fall_land": 82,
        "reveal": 24,
        "ready": 16,
        "block": 18,
        "prop_use": 20,
        "prop_hold": 12,
        # An unresolved actor action still needs a small, auditable pose-to-pose
        # weight shift. Environment-only spatial cells have no actors and remain
        # static, so this never fabricates a person for object/camera motion.
        "spatial": 16,
    }[family]
    travel = round(magnitude * max(0.0, min(1.0, pose_progress)))
    if direction == "left":
        return -travel, 0
    if direction in {"right", "forward"}:
        return travel, 0
    if direction == "backward":
        return -travel, 0
    if direction == "up":
        return 0, -travel
    if direction == "down":
        return 0, travel
    return 0, 0


def _actor_pose_for_slot(family: str, direction: str, slot_index: int) -> tuple[str, str]:
    if slot_index <= 0 or family not in {"strike", "kick", "grab_control", "throw"}:
        return family, direction
    actor_family = "evade" if family != "throw" else "fall_land"
    actor_direction = "left" if direction in {"right", "forward"} else "right"
    return actor_family, actor_direction


def _global_geometry(
    family: str,
    stage: str,
    direction: str,
    actor_roles: Sequence[str],
    *,
    pose_progress: float,
    mechanics_modifiers: Mapping[str, Any],
    prior_actions: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    actor_count = len(actor_roles)
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
    for index, (center_x, actor_role) in enumerate(zip(centers, actor_roles, strict=True)):
        actor_family, actor_direction = _actor_pose_for_slot(family, direction, index)
        role_prior_actions: list[tuple[Mapping[str, Any], int]] = []
        for prior_action in reversed(prior_actions):
            prior_actor_roles = list(prior_action.get("actor_roles") or [])
            if actor_role not in prior_actor_roles:
                break
            role_prior_actions.append((prior_action, prior_actor_roles.index(actor_role)))
        role_prior_actions.reverse()
        origin_entry = role_prior_actions[-1] if role_prior_actions else None
        origin_action = origin_entry[0] if origin_entry else None
        origin_slot_index = origin_entry[1] if origin_entry else 0
        if origin_action is None:
            origin_family = None
            origin_direction = "right"
            origin_modifiers: Mapping[str, Any] = {}
        else:
            origin_family, origin_direction = _actor_pose_for_slot(
                str(origin_action["family"]),
                str(origin_action["direction"]),
                origin_slot_index,
            )
            origin_modifiers = (
                origin_action["mechanics_modifiers"] if origin_slot_index == 0 else {}
            )
        local = _phase_geometry(
            actor_family,
            stage,
            actor_direction,
            pose_progress=pose_progress,
            mechanics_modifiers=mechanics_modifiers if index == 0 else {},
            origin_family=origin_family,
            origin_direction=origin_direction,
            origin_mechanics_modifiers=origin_modifiers,
        )
        root_origin_x = 0
        root_origin_y = 0
        for prior_action, prior_slot_index in role_prior_actions:
            prior_family, prior_direction = _actor_pose_for_slot(
                str(prior_action["family"]),
                str(prior_action["direction"]),
                prior_slot_index,
            )
            prior_x, prior_y = _root_motion(prior_family, prior_direction, 1.0)
            root_origin_x += prior_x
            root_origin_y += prior_y
        root_delta_x, root_delta_y = _root_motion(
            actor_family,
            actor_direction,
            pose_progress,
        )
        root_x = root_origin_x + root_delta_x
        root_y = root_origin_y + root_delta_y
        joints = {
            joint: [
                max(35, min(965, round(center_x + root_x + point[0] * scale))),
                max(180, min(925, round(520 + root_y + point[1] * scale * 3.1))),
            ]
            for joint, point in local.items()
        }
        actors.append(
            {
                "slot": index + 1,
                "pose_family": actor_family,
                "facing": actor_direction,
                "root_origin": [root_origin_x, root_origin_y],
                "root_translation": [root_x, root_y],
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
    if (
        not matches
        and not source_indexes
        and len(body_beats) == 1
        and isinstance(body_beats[0], Mapping)
    ):
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


def _initial_anchor_unit_ids(
    beat: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    cell_count: int,
) -> frozenset[str]:
    """Identify a ready pose already established by the first cinematic frame."""
    if (
        len(units) < 2
        or cell_count < 2
        or not str(beat.get("beat_id") or "").endswith("_P01")
    ):
        return frozenset()
    first_mechanics, _ = _matching_mechanics(beat, [units[0]])
    first_actions = " → ".join(_strings(units[0].get("actions")))
    first_family, _ = _pose_family(first_actions, first_mechanics)
    if first_family != "ready":
        return frozenset()
    dynamic_families = {
        family for family, _markers in _POSE_CLASSIFIERS
        if family not in {"ready", "prop_hold"}
    }
    for unit in units[1:]:
        mechanics, _ = _matching_mechanics(beat, [unit])
        action_text = " → ".join(_strings(unit.get("actions")))
        family, _ = _pose_family(action_text, mechanics)
        if family in dynamic_families:
            return frozenset({str(units[0]["unit_id"])})
    return frozenset()


def _partition_units(
    units: list[dict[str, Any]],
    cell_count: int,
    *,
    initial_anchor_unit_ids: frozenset[str] = frozenset(),
) -> list[list[dict[str, Any]]]:
    if initial_anchor_unit_ids:
        first_id = str(units[0]["unit_id"])
        if initial_anchor_unit_ids != {first_id} or cell_count < 2 or len(units) < 2:
            raise ValueError("initial pose anchor must bind only the first canonical unit")
        groups = [[units[0]]] + _partition_units(units[1:], cell_count - 1)
        covered = {str(unit["unit_id"]) for group in groups for unit in group}
        expected = {str(unit["unit_id"]) for unit in units}
        if covered != expected:
            raise ValueError("Gxx initial-anchor partition lost canonical generation units")
        return groups
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


def _pose_progress_samples(
    cells: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    initial_anchor_unit_ids: frozenset[str] = frozenset(),
) -> list[float]:
    group_keys = [tuple(str(unit["unit_id"]) for unit in group) for group in groups]
    positions: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(group_keys):
        positions.setdefault(key, []).append(index)
    samples: list[float] = []
    for index, (cell, key) in enumerate(zip(cells, group_keys, strict=True)):
        if index == 0 and initial_anchor_unit_ids == frozenset(key):
            samples.append(1.0)
            continue
        occurrences = positions[key]
        if len(occurrences) > 1:
            ordinal = occurrences.index(index)
            progress = ordinal / (len(occurrences) - 1)
        else:
            explicit = cell.get("action_progress")
            if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
                progress = float(explicit)
            else:
                progress = float(
                    _POSE_POLICY["phase_samples"].get(
                        str(cell.get("stage") or "action_progress"),
                        0.7,
                    )
                )
        samples.append(round(max(0.0, min(1.0, progress)), 3))
    return samples


def compile_pose_contracts(
    beat: Mapping[str, Any],
    cells: list[dict[str, Any]],
    *,
    known_actor_roles: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Attach one source-bound, deterministic pose contract to every Gxx cell."""
    units, lineage_status = _validated_units(beat)
    initial_anchor_unit_ids = _initial_anchor_unit_ids(
        beat,
        units,
        cell_count=len(cells),
    )
    groups = _partition_units(
        units,
        len(cells),
        initial_anchor_unit_ids=initial_anchor_unit_ids,
    )
    progress_samples = _pose_progress_samples(
        cells,
        groups,
        initial_anchor_unit_ids=initial_anchor_unit_ids,
    )
    source_actor_roles = {value.casefold() for value in known_actor_roles if value}
    result: list[dict[str, Any]] = []
    active_group_key: tuple[str, ...] | None = None
    active_action: dict[str, Any] | None = None
    prior_actions: list[dict[str, Any]] = []
    for cell, group, pose_progress in zip(cells, groups, progress_samples, strict=True):
        actions = [action for unit in group for action in _strings(unit.get("actions"))]
        action_text = " → ".join(actions)
        mechanics, matched_body_action_beats = _matching_mechanics(beat, group)
        family, family_evidence = _pose_family(action_text, mechanics)
        direction, direction_evidence = _direction(action_text, mechanics)
        mechanics_modifiers = _mechanics_modifiers(mechanics, direction)
        group_key = tuple(str(unit["unit_id"]) for unit in group)
        timing_role = (
            "initial_anchor"
            if initial_anchor_unit_ids == frozenset(group_key)
            else "story_action"
        )
        story_time_weight = 0.0 if timing_role == "initial_anchor" else 1.0
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
        current_action = {
            "unit_ids": list(group_key),
            "family": family,
            "direction": direction,
            "mechanics_modifiers": mechanics_modifiers,
            "actor_roles": actor_roles,
        }
        if active_group_key != group_key:
            if active_action is not None:
                prior_actions.append(active_action)
            active_group_key = group_key
            active_action = current_action
        elif active_action != current_action:
            raise ValueError("one action-unit group resolved to inconsistent pose semantics")
        actors = _global_geometry(
            family,
            str(cell.get("stage") or "action_progress"),
            direction,
            actor_roles,
            pose_progress=pose_progress,
            mechanics_modifiers=mechanics_modifiers,
            prior_actions=prior_actions,
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
        transition_origin = (
            {
                "source": "previous_canonical_action",
                "unit_ids": list(prior_actions[-1]["unit_ids"]),
                "pose_family": str(prior_actions[-1]["family"]),
                "direction": str(prior_actions[-1]["direction"]),
            }
            if prior_actions
            else {
                "source": "neutral",
                "unit_ids": [],
                "pose_family": "neutral",
                "direction": "right",
            }
        )
        pose_fingerprint = _canonical_sha256(
            {
                "family": family,
                "stage": str(cell.get("stage") or ""),
                "pose_progress": pose_progress,
                "direction": direction,
                "mechanics_modifiers": mechanics_modifiers,
                "transition_origin": transition_origin,
                "actors": actors,
                "objects": geometry["objects"],
                "action_vector": action_vector,
                "camera_vector": camera_vector,
                "static_spatial_state": static_spatial_state,
                "timing_role": timing_role,
                "story_time_weight": story_time_weight,
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
            "pose_progress": pose_progress,
            "classification_evidence": family_evidence,
            "direction": direction,
            "direction_evidence": direction_evidence,
            "mechanics_modifiers": mechanics_modifiers,
            "transition_origin": transition_origin,
            "actor_roles": actor_roles,
            "object_roles": object_roles,
            "action_vector": action_vector,
            "camera_vector": camera_vector,
            "static_spatial_state": static_spatial_state,
            "timing_role": timing_role,
            "story_time_weight": story_time_weight,
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
    evidence = contract.get("classification_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("polarity") not in {
        "positive",
        "none",
    }:
        raise ValueError("story guide pose classification evidence is invalid")
    progress = contract.get("pose_progress")
    if (
        not isinstance(progress, (int, float))
        or isinstance(progress, bool)
        or not 0.0 <= float(progress) <= 1.0
    ):
        raise ValueError("story guide pose progress is invalid")
    timing_role = contract.get("timing_role")
    story_time_weight = contract.get("story_time_weight")
    if timing_role not in {"initial_anchor", "story_action"}:
        raise ValueError("story guide pose timing role is invalid")
    if (
        not isinstance(story_time_weight, (int, float))
        or isinstance(story_time_weight, bool)
    ):
        raise ValueError("story guide pose story-time weight is invalid")
    if timing_role == "initial_anchor":
        if (
            float(story_time_weight) != 0.0
            or contract.get("pose_family") != "ready"
            or float(progress) != 1.0
        ):
            raise ValueError("story guide initial pose anchor is invalid")
    elif float(story_time_weight) != 1.0:
        raise ValueError("story guide story action must retain story-time weight")
    modifiers = contract.get("mechanics_modifiers")
    if not isinstance(modifiers, Mapping) or set(modifiers) != {
        "center_drop",
        "torso_lean",
        "stance_width",
        "lead_step",
        "two_hand_hold",
    }:
        raise ValueError("story guide mechanics modifiers are invalid")
    transition_origin = contract.get("transition_origin")
    if not isinstance(transition_origin, Mapping):
        raise ValueError("story guide transition origin is missing")
    origin_source = transition_origin.get("source")
    origin_unit_ids = transition_origin.get("unit_ids")
    if (
        origin_source not in {"neutral", "previous_canonical_action"}
        or not isinstance(origin_unit_ids, list)
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in origin_unit_ids)
    ):
        raise ValueError("story guide transition origin is invalid")
    if origin_source == "neutral":
        if (
            origin_unit_ids
            or transition_origin.get("pose_family") != "neutral"
            or transition_origin.get("direction") != "right"
        ):
            raise ValueError("story guide neutral transition origin is invalid")
    elif (
        not origin_unit_ids
        or transition_origin.get("pose_family") not in {family for family, _ in _POSE_CLASSIFIERS}
        | {"spatial"}
        or transition_origin.get("direction")
        not in {direction for direction, _ in _DIRECTION_MARKERS}
    ):
        raise ValueError("story guide canonical transition origin is invalid")
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
            "pose_progress": progress,
            "direction": contract.get("direction"),
            "mechanics_modifiers": modifiers,
            "transition_origin": transition_origin,
            "actors": actors,
            "objects": geometry.get("objects") or [],
            "action_vector": contract.get("action_vector"),
            "camera_vector": contract.get("camera_vector"),
            "static_spatial_state": bool(contract.get("static_spatial_state")),
            "timing_role": timing_role,
            "story_time_weight": float(story_time_weight),
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


def zero_time_anchor_cell_ids(cells: list[Mapping[str, Any]]) -> list[str]:
    return [
        str(cell.get("label") or "")
        for cell in cells
        if (cell.get("pose_contract") or {}).get("timing_role") == "initial_anchor"
    ]


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
    anchors = [
        index
        for index, cell in enumerate(cells)
        if cell["pose_contract"]["timing_role"] == "initial_anchor"
    ]
    if anchors:
        if anchors != [0] or len(cells) < 2:
            raise ValueError(f"{beat_id} initial pose anchor must be the first of multiple cells")
        if cells[1]["pose_contract"]["timing_role"] != "story_action":
            raise ValueError(f"{beat_id} initial pose anchor must precede story action")
    progress_groups: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        contract = cell["pose_contract"]
        if not contract["static_spatial_state"]:
            binding_key = _canonical_sha256(contract["action_bindings"])
            progress_groups.setdefault(binding_key, []).append(contract)
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
        if previous_contract["action_bindings"] != current_contract["action_bindings"]:
            expected_origin_ids = [
                str(binding["unit_id"])
                for binding in previous_contract["action_bindings"]
            ]
            current_origin = current_contract["transition_origin"]
            if (
                current_origin["source"] != "previous_canonical_action"
                or current_origin["unit_ids"] != expected_origin_ids
                or current_origin["pose_family"] != previous_contract["pose_family"]
                or current_origin["direction"] != previous_contract["direction"]
            ):
                raise ValueError(f"{beat_id} action transition lost its canonical origin")
        same_action = previous_contract["action_bindings"] == current_contract["action_bindings"]
        progress_changed = previous_contract["pose_progress"] != current_contract["pose_progress"]
        if same_action and progress_changed and not both_static:
            previous_actors = previous_contract["geometry"]["actors"]
            current_actors = current_contract["geometry"]["actors"]
            if not previous_actors or len(previous_actors) != len(current_actors):
                continue
            maximum_delta = max(
                (
                    abs(previous_point[0] - current_point[0])
                    + abs(previous_point[1] - current_point[1])
                    for previous_actor, current_actor in zip(
                        previous_actors, current_actors, strict=True
                    )
                    for joint, previous_point in previous_actor["joints"].items()
                    for current_point in [current_actor["joints"][joint]]
                ),
                default=0,
            )
            if maximum_delta < int(_POSE_POLICY["minimum_adjacent_joint_delta"]):
                raise ValueError(
                    f"{beat_id} adjacent action progress lacks visible joint displacement"
                )
    for contracts in progress_groups.values():
        if len(contracts) < 2:
            continue
        ordered = sorted(contracts, key=lambda item: float(item["pose_progress"]))
        first = ordered[0]
        last = ordered[-1]
        if first["pose_progress"] == last["pose_progress"]:
            continue
        first_actors = first["geometry"]["actors"]
        last_actors = last["geometry"]["actors"]
        if not first_actors or len(first_actors) != len(last_actors):
            continue
        maximum_span_delta = max(
            (
                abs(first_point[0] - last_point[0])
                + abs(first_point[1] - last_point[1])
                for first_actor, last_actor in zip(first_actors, last_actors, strict=True)
                for joint, first_point in first_actor["joints"].items()
                for last_point in [last_actor["joints"][joint]]
            ),
            default=0,
        )
        if maximum_span_delta < int(_POSE_POLICY["minimum_action_span_joint_delta"]):
            raise ValueError(f"{beat_id} action span lacks meaningful joint displacement")
