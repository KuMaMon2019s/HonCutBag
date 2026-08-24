"""Canonical camera-angle vocabulary for Phase 1 and downstream prompts.

Adaptation owns the authored value. Downstream consumers may normalize legacy
spellings, but they never choose a different angle or infer one from movement.
"""

from __future__ import annotations

from typing import Any


CAMERA_ANGLE_VALUES = (
    "eye_level",
    "low",
    "high",
    "dutch",
    "over_shoulder",
    "aerial",
    "bird",
    "worm",
)

_CAMERA_ANGLE_ALIASES = {
    "eye-level": "eye_level",
    "over-shoulder": "over_shoulder",
    "over_the_shoulder": "over_shoulder",
    "birds_eye": "bird",
    "bird_eye": "bird",
    "worms_eye": "worm",
    "worm_eye": "worm",
}

_CAMERA_ANGLE_DESCRIPTIONS = {
    "eye_level": "平视客观机位",
    "low": "低机位仰拍",
    "high": "高机位俯拍",
    "dutch": "荷兰式倾斜机位",
    "over_shoulder": "过肩机位",
    "aerial": "航拍高空机位",
    "bird": "垂直鸟瞰机位",
    "worm": "贴地虫视仰拍",
}


def canonical_camera_angle(value: Any) -> str:
    """Normalize legacy spellings at a compatibility boundary."""
    candidate = str(value or "").strip().lower().replace(" ", "_")
    candidate = _CAMERA_ANGLE_ALIASES.get(candidate, candidate)
    return candidate if candidate in CAMERA_ANGLE_VALUES else "eye_level"


def camera_angle_description(value: Any) -> str:
    """Translate one canonical angle into a generation-facing phrase."""
    return _CAMERA_ANGLE_DESCRIPTIONS[canonical_camera_angle(value)]


CAMERA_ANGLE_PLANNING_INSTRUCTIONS = (
    "【机位角度合同】camera_angle 是构图视角，不是 camera_movement。"
    "eye_level 用于客观观察和自然反应；over_shoulder 用于关系、对白和视线对照；"
    "low 用于来源支持的力量、决心或威胁；high 用于脆弱、受压或暴露；"
    "dutch 仅用于失衡、异常或心理动摇；aerial、bird 仅用于来源允许的空间规模、"
    "孤立或位置关系揭示；worm 仅用于贴地尺度或极端压迫。"
    "不得用稀有机位制造来源中不存在的无人机、摇臂、俯视空间或戏剧事实。"
)
