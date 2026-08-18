"""Shared physical camera-motion and human-perspective contracts.

Phase 1 owns the enum and persists a deterministic start/process/end contract.
Storyboard and video stages render that same metadata instead of maintaining
independent camera vocabularies.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

CAMERA_MOTION_SCHEMA_VERSION = 1

# Keep ``rack_focus`` last: parser diagnostics and contract tests historically
# use it as the visible end of the legal vocabulary.
CAMERA_MOVEMENT_SPECS: dict[str, dict[str, Any]] = {
    "static": {
        "label": "固定(fixed/locked)",
        "motion": "camera remains locked, stable eye-level framing, no drift",
    },
    "pan_left": {
        "label": "缓慢左摇 / slow pan left",
        "motion": "camera position stays fixed while it rotates smoothly left",
    },
    "pan_right": {
        "label": "缓慢右摇 / slow pan right",
        "motion": "camera position stays fixed while it rotates smoothly right",
    },
    "tilt_up": {
        "label": "缓慢上仰 / slow tilt up",
        "motion": "camera position stays fixed while it tilts smoothly upward",
    },
    "tilt_down": {
        "label": "缓慢下俯 / slow tilt down",
        "motion": "camera position stays fixed while it tilts smoothly downward",
    },
    "dolly_in": {
        "label": "推进(dolly in) / 推入(push in)",
        "motion": "camera physically moves toward the subject with natural perspective change",
    },
    "dolly_out": {
        "label": "缓慢拉镜 / slow dolly out",
        "motion": "camera physically moves backward and gradually reveals the environment",
    },
    "tracking_left": {
        "label": "向左横向跟拍 / lateral tracking left",
        "motion": "camera tracks left alongside the subject at a consistent distance and body scale",
    },
    "tracking_right": {
        "label": "向右横向跟拍 / lateral tracking right",
        "motion": "camera tracks right alongside the subject at a consistent distance and body scale",
    },
    "tracking_front": {
        "label": "前方倒退跟拍 / front backward tracking",
        "motion": "camera faces the approaching subject and moves backward at the same natural pace",
    },
    "tracking_rear": {
        "label": "后方跟拍 / rear tracking",
        "motion": "camera follows behind the subject with stable shoulder-level framing",
    },
    "pedestal_up": {
        "label": "垂直升镜 / pedestal up",
        "motion": "the entire camera rises vertically while preserving natural perspective",
    },
    "pedestal_down": {
        "label": "垂直降镜 / pedestal down",
        "motion": "the entire camera lowers vertically while preserving natural perspective",
    },
    "crane_up": {
        "label": "摇臂升镜 / crane up",
        "motion": "camera rises on a controlled physical arc with a stable horizon",
    },
    "crane_down": {
        "label": "摇臂降镜 / crane down",
        "motion": "camera descends on a controlled physical arc with a stable horizon",
    },
    "handheld": {
        "label": "克制手持跟拍 / restrained handheld",
        "motion": "subtle human-operated micro movement with stable subject framing and no chaotic shake",
    },
    "steadicam": {
        "label": "稳定器跟拍 / gimbal tracking",
        "motion": "fluid stabilized tracking with minimal vertical bounce and a stable horizon",
    },
    "orbital": {
        "label": "轻微环绕 / 20–45 degree orbit",
        "motion": "camera moves through a controlled 20–45 degree arc at a stable distance from the subject",
    },
    "orbit_semicircle": {
        "label": "半圆环绕 / semicircular orbit",
        "motion": "camera moves through a motivated 90–180 degree arc with consistent subject scale",
        "rare": True,
    },
    "zoom_in": {
        "label": "缓慢光学变焦放大 / slow optical zoom in",
        "motion": "camera position stays fixed while focal length gradually increases with minimal perspective change",
    },
    "zoom_out": {
        "label": "缓慢光学变焦缩小 / slow optical zoom out",
        "motion": "camera position stays fixed while focal length gradually decreases to reveal more context",
    },
    "subtle_zoom_in": {
        "label": "微变焦 / subtle 5–10% zoom in",
        "motion": "camera stays fixed during an almost imperceptible 5–10% focal-length increase",
    },
    "dialogue_push_in": {
        "label": "呼吸式情绪推进 / dialogue push-in",
        "motion": "extremely slow physical push toward the speaker, optionally ending with no more than 5% optical zoom",
    },
    "dolly_in_subtle_zoom": {
        "label": "推镜同时轻微变焦 / dolly in with subtle zoom",
        "motion": (
            "camera physically dollies toward the subject as focal length increases subtly; "
            "the physical move remains dominant (approximately 70% dolly and 30% zoom) and facial proportions stay natural"
        ),
    },
    "crash_zoom_in": {
        "label": "快速变焦推进 / crash zoom in",
        "motion": "rapid optical zoom to the authored detail, then an immediate stable final composition without overshoot",
        "rare": True,
    },
    "crash_zoom_out": {
        "label": "快速拉远变焦 / crash zoom out",
        "motion": "rapid optical zoom from close framing to a wider reveal, ending completely stable",
        "rare": True,
    },
    "punch_in": {
        "label": "快速物理推近 / punch in",
        "motion": "camera rapidly dollies from medium framing to close-up and decelerates smoothly before stopping",
        "rare": True,
    },
    "dolly_zoom_out": {
        "label": "后拉同步变焦 / dolly zoom",
        "motion": "camera moves backward while zooming in so subject scale stays constant and background perspective compresses",
        "rare": True,
    },
    "dolly_zoom_in": {
        "label": "前推反向变焦 / reverse dolly zoom",
        "motion": "camera moves forward while zooming out so subject scale stays constant and background perspective expands",
        "rare": True,
    },
    "whip_pan_left": {
        "label": "向左甩镜 / whip pan left",
        "motion": "camera rapidly rotates left with natural motion blur and lands precisely on the declared target",
        "rare": True,
    },
    "whip_pan_right": {
        "label": "向右甩镜 / whip pan right",
        "motion": "camera rapidly rotates right with natural motion blur and lands precisely on the declared target",
        "rare": True,
    },
    "foreground_occlusion": {
        "label": "前景遮挡移动 / foreground occlusion move",
        "motion": "camera tracks behind one declared foreground object and reveals the next composition after natural occlusion",
    },
    "rack_focus": {
        "label": "焦点转换 / rack focus",
        "motion": "camera position remains stable while focus moves smoothly between the declared foreground and background subjects",
    },
}

CAMERA_MOVEMENT_VALUES = tuple(CAMERA_MOVEMENT_SPECS)

CAMERA_MOVEMENT_ALIASES = {
    "fixed": "static",
    "locked": "static",
    "slow_pan": "pan_left",
    "tracking": "steadicam",
    "tracking_shot": "steadicam",
    "orbit": "orbital",
    "push_in": "dolly_in",
    "whip_pan": "whip_pan_right",
}

BASE_CAMERA_MOTION_CONTRACT = (
    "cinematic camera movement; one primary camera movement per shot; intentional narrative motivation; "
    "stable initial framing; smooth controlled path with natural acceleration and deceleration; stable horizon; "
    "consistent subject scale and spatial relationship; movement ends at a clearly defined final framing"
)

BASE_CAMERA_NEGATIVE = (
    "random camera movement, camera drifting, sudden framing changes, unstable horizon, excessive shaking, "
    "chaotic handheld motion, random zoom, repeated zoom in and out, camera breathing, accidental dolly zoom, "
    "sudden focal length changes, floating camera, unnatural orbit, uncontrolled rotation, warped background, "
    "subject deformation during camera movement, body stretching during camera movement"
)

HUMAN_PERSPECTIVE_CONTRACT = (
    "50–85mm equivalent cinematic lens; natural perspective; camera remains far enough from the person; "
    "full-body framing never uses an ultra-wide lens; avoid near-large/far-small exaggeration; "
    "keep the head away from frame edges; preserve identical head size and body proportions throughout the shot"
)

HUMAN_PERSPECTIVE_NEGATIVE = (
    "wide-angle distortion, fisheye distortion, oversized head, large face, short neck, narrow shoulders, "
    "childlike body proportions, bobblehead proportions, perspective-stretched head"
)

CAMERA_MOTION_PLANNING_INSTRUCTIONS = (
    "【摄影与运镜硬合同】camera_movement 必须只选择合法词表中的一个主运动。"
    "Dolly 表示摄影机真实移动并产生自然透视变化；zoom_in/zoom_out 表示机位不动、只改变焦距，禁止混写。"
    "普通对白优先 static、subtle_zoom_in、dialogue_push_in 或轻微 dolly_in；走路优先 tracking_left、"
    "tracking_right、tracking_front、tracking_rear 或 steadicam；空间揭示优先 dolly_out、pan_left、pan_right；"
    "情绪升级可用 dolly_in 或 dolly_in_subtle_zoom，后者必须保持约 70% dolly + 30% zoom；"
    "反转爆点才可用 crash_zoom_in/punch_in；dolly_zoom、whip_pan、"
    "orbit_semicircle 只用于剧本明确支持的极端震惊、注意力切换或关系反转，普通对白禁止滥用。"
    "每镜运动必须有稳定起点、平滑且受控的过程、自然加减速和明确稳定的终点；禁止随机漂移、无意义晃动、"
    "重复推拉、意外希区柯克变焦或不受控旋转。有人物时使用 50–85mm 等效焦段和自然透视，"
    "全身镜头禁止超广角，人物头部不得靠近画面边缘。"
)

_LENS_BY_SHOT_SIZE = {
    "extreme_wide": 50,
    "wide": 50,
    "establishing": 50,
    "full": 50,
    "medium_wide": 50,
    "medium": 65,
    "over_shoulder": 65,
    "medium_close": 85,
    "medium_close_up": 85,
    "close_up": 85,
    "extreme_close_up": 85,
    "insert": 85,
}


def canonical_camera_movement(value: object) -> str:
    """Normalize legacy aliases for rendering without weakening Phase 1 validation."""
    key = str(value or "static").strip().casefold()
    key = CAMERA_MOVEMENT_ALIASES.get(key, key)
    return key if key in CAMERA_MOVEMENT_SPECS else "static"


def camera_movement_description(value: object) -> str:
    """Return the shared bilingual physical description for one movement."""
    key = canonical_camera_movement(value)
    spec = CAMERA_MOVEMENT_SPECS[key]
    return f"{spec['label']}: {spec['motion']}"


def _has_human(shot: Mapping[str, Any]) -> bool:
    requested = shot.get("who") if "who" in shot else shot.get("characters")
    if isinstance(requested, list):
        return bool(requested)
    return bool(requested)


def lens_for_shot(shot_size: object, *, has_human: bool) -> int | None:
    """Choose a deterministic natural-perspective lens for human shots."""
    if not has_human:
        return None
    return _LENS_BY_SHOT_SIZE.get(str(shot_size or "medium").strip().casefold(), 65)


def build_camera_motion_contract(
    shot: Mapping[str, Any],
    *,
    has_human: bool | None = None,
) -> dict[str, Any]:
    """Build a serializable physical camera contract for one authored shot."""
    movement = canonical_camera_movement(shot.get("camera_movement"))
    human = _has_human(shot) if has_human is None else bool(has_human)
    lens_mm = lens_for_shot(shot.get("shot_size") or shot.get("shot_type"), has_human=human)
    shot_size = str(shot.get("shot_size") or shot.get("shot_type") or "medium")
    spec = CAMERA_MOVEMENT_SPECS[movement]
    return {
        "schema_version": CAMERA_MOTION_SCHEMA_VERSION,
        "movement": movement,
        "movement_label": spec["label"],
        "primary_movement_count": 1,
        "rare_movement": bool(spec.get("rare")),
        "lens_mm": lens_mm,
        "start": f"stable authored {shot_size} framing before movement begins",
        "process": spec["motion"],
        "end": "smoothly decelerate and stop at a clearly defined stable final framing",
        "human_perspective": HUMAN_PERSPECTIVE_CONTRACT if human else "",
        "negative": ", ".join(
            part
            for part in (BASE_CAMERA_NEGATIVE, HUMAN_PERSPECTIVE_NEGATIVE if human else "")
            if part
        ),
    }


def apply_camera_motion_contract(shot: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Persist lens and start/process/end metadata on a shot in place."""
    contract = build_camera_motion_contract(shot)
    shot["camera_movement"] = contract["movement"]
    shot["camera_motion_contract"] = contract
    if contract["lens_mm"] is not None:
        shot["lens_mm"] = contract["lens_mm"]
    else:
        shot.pop("lens_mm", None)
    return shot


def camera_motion_prompt(shot: Mapping[str, Any]) -> str:
    """Render the persisted contract, rebuilding it only for legacy artifacts."""
    contract = shot.get("camera_motion_contract")
    if not isinstance(contract, Mapping):
        contract = build_camera_motion_contract(shot)
    parts = [
        "Camera-motion hard contract",
        f"one primary movement only: {contract.get('movement_label')}",
        f"start: {contract.get('start')}",
        f"process: {contract.get('process')}",
        f"end: {contract.get('end')}",
    ]
    if contract.get("lens_mm"):
        parts.append(f"lens: {contract['lens_mm']}mm equivalent")
    if contract.get("human_perspective"):
        parts.append(str(contract["human_perspective"]))
    return "; ".join(parts)


def camera_motion_negative_prompt(shot: Mapping[str, Any]) -> str:
    """Return camera/perspective negatives from persisted or legacy metadata."""
    contract = shot.get("camera_motion_contract")
    if not isinstance(contract, Mapping):
        contract = build_camera_motion_contract(shot)
    return str(contract.get("negative") or BASE_CAMERA_NEGATIVE)
