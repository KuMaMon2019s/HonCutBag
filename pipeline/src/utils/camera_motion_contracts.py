"""Shared physical camera-motion and human-perspective contracts.

Phase 1 owns the enum and persists a deterministic start/process/end contract.
Storyboard and video stages render that same metadata instead of maintaining
independent camera vocabularies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping
from typing import Any

CAMERA_MOTION_SCHEMA_VERSION = 3

_CAMERA_PARAMETER_KEYS = (
    "camera_height_m",
    "translation_distance_m",
    "translation_speed_m_per_s",
    "focal_length_start_mm",
    "focal_length_end_mm",
    "focal_length_speed_mm_per_s",
    "pan_degrees",
    "pan_speed_degrees_per_s",
    "tilt_degrees",
    "tilt_speed_degrees_per_s",
    "segment_count",
    "segment_pause_s",
)

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
    raw_parameters = shot.get("camera_motion_parameters")
    if raw_parameters is not None and not isinstance(raw_parameters, Mapping):
        raise ValueError("camera_motion_parameters must be an object")
    parameters: dict[str, float | int] = {}
    for key in _CAMERA_PARAMETER_KEYS:
        if not isinstance(raw_parameters, Mapping) or key not in raw_parameters:
            continue
        value = raw_parameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"camera motion parameter {key} must be numeric")
        if key == "segment_count":
            integer = int(value)
            if integer != value or integer < 1:
                raise ValueError("camera segment_count must be a positive integer")
            parameters[key] = integer
        else:
            parameters[key] = round(float(value), 6)
    for key in (
        "translation_speed_m_per_s",
        "focal_length_start_mm",
        "focal_length_end_mm",
        "focal_length_speed_mm_per_s",
        "pan_speed_degrees_per_s",
        "tilt_speed_degrees_per_s",
    ):
        if key in parameters and float(parameters[key]) <= 0:
            raise ValueError(f"camera motion parameter {key} must be positive")
    for key in ("camera_height_m", "translation_distance_m", "segment_pause_s"):
        if key in parameters and float(parameters[key]) < 0:
            raise ValueError(f"camera motion parameter {key} cannot be negative")
    contract = {
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
        "technical_parameters": parameters,
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return contract


def apply_camera_motion_contract(shot: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Persist lens and start/process/end metadata on a shot in place."""
    contract = build_camera_motion_contract(shot)
    shot["camera_movement"] = contract["movement"]
    shot["camera_motion_contract"] = contract
    shot["camera_motion_contract_sha256"] = contract["contract_sha256"]
    if contract["lens_mm"] is not None:
        shot["lens_mm"] = contract["lens_mm"]
    else:
        shot.pop("lens_mm", None)
    return shot


def camera_motion_minimum_duration_s(contract: Mapping[str, Any]) -> float:
    """Return the deterministic minimum time required by authored camera parameters."""

    parameters = contract.get("technical_parameters")
    if not isinstance(parameters, Mapping):
        return 0.0
    requirements: list[float] = []
    pairs = (
        ("translation_distance_m", "translation_speed_m_per_s"),
        ("pan_degrees", "pan_speed_degrees_per_s"),
        ("tilt_degrees", "tilt_speed_degrees_per_s"),
    )
    for amount_key, speed_key in pairs:
        if amount_key not in parameters:
            continue
        if speed_key not in parameters:
            raise ValueError(f"{amount_key} requires {speed_key}")
        speed = float(parameters[speed_key])
        if speed <= 0:
            raise ValueError(f"{speed_key} must be positive")
        requirements.append(abs(float(parameters[amount_key])) / speed)
    focal_keys = {"focal_length_start_mm", "focal_length_end_mm"}
    if focal_keys.intersection(parameters):
        if not focal_keys.issubset(parameters):
            raise ValueError("focal-length motion requires both start and end values")
        speed_key = "focal_length_speed_mm_per_s"
        if speed_key not in parameters:
            raise ValueError("focal-length motion requires focal_length_speed_mm_per_s")
        requirements.append(
            abs(
                float(parameters["focal_length_end_mm"])
                - float(parameters["focal_length_start_mm"])
            )
            / float(parameters[speed_key])
        )
    movement_duration = max(requirements, default=0.0)
    segment_count = int(parameters.get("segment_count") or 1)
    pause_s = float(parameters.get("segment_pause_s") or 0.0)
    return round(movement_duration + max(0, segment_count - 1) * pause_s, 6)


def validate_camera_motion_duration(
    contract: Mapping[str, Any],
    duration_s: float | int,
    *,
    resource_id: str = "shot",
) -> float:
    """Fail before Provider work when Adaptation authored an impossible path."""

    available = float(duration_s)
    required = camera_motion_minimum_duration_s(contract)
    if required > available + 1e-6:
        raise ValueError(
            f"{resource_id} camera path requires at least {required:g}s, "
            f"but only {available:g}s is available; return to Adaptation"
        )
    return required


def camera_projection_at_progress(
    contract: Mapping[str, Any],
    progress: float,
) -> dict[str, Any]:
    """Project one sample along the authored continuous camera path.

    This function never selects or repairs a camera movement.  It only samples
    the Adaptation-owned contract so Phase 2 cannot introduce a second path.
    """

    normalized = max(0.0, min(1.0, float(progress)))
    movement = canonical_camera_movement(contract.get("movement"))
    raw_parameters = contract.get("technical_parameters")
    parameters = raw_parameters if isinstance(raw_parameters, Mapping) else {}
    default_pan = {
        "pan_left": -30.0,
        "pan_right": 30.0,
        "whip_pan_left": -60.0,
        "whip_pan_right": 60.0,
        "orbital": 45.0,
        "orbit_semicircle": 120.0,
    }.get(movement, 0.0)
    default_tilt = {
        "tilt_up": -25.0,
        "tilt_down": 25.0,
        "crane_up": -15.0,
        "crane_down": 15.0,
    }.get(movement, 0.0)
    yaw = float(parameters.get("pan_degrees", default_pan)) * normalized
    pitch = float(parameters.get("tilt_degrees", default_tilt)) * normalized
    translation = float(parameters.get("translation_distance_m", 0.0)) * normalized
    if movement in {"dolly_out", "tracking_front"}:
        translation *= -1
    start_focal = float(parameters.get("focal_length_start_mm") or contract.get("lens_mm") or 50.0)
    end_focal = float(parameters.get("focal_length_end_mm") or start_focal)
    focal = start_focal + (end_focal - start_focal) * normalized
    view_angle = abs(yaw) % 360.0
    if view_angle > 180:
        view_angle = 360 - view_angle
    if view_angle < 20:
        view = "front"
    elif view_angle < 65:
        view = "front_three_quarter"
    elif view_angle < 115:
        view = "profile"
    elif view_angle < 160:
        view = "rear_three_quarter"
    else:
        view = "rear"
    return {
        "schema": "honcut.camera-pose-projection.v1",
        "path_progress": round(normalized, 6),
        "yaw_degrees": round(yaw, 6),
        "pitch_degrees": round(pitch, 6),
        "translation_m": round(translation, 6),
        "focal_length_mm": round(focal, 6),
        "camera_height_m": round(float(parameters.get("camera_height_m", 1.6)), 6),
        "view": view,
        "horizontal_joint_scale": round(max(0.28, abs(math.cos(math.radians(yaw)))), 6),
        "occlusion_order": "right_over_left" if yaw >= 0 else "left_over_right",
    }


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


_OBSERVABLE_CAMERA_EVIDENCE = {
    "static": "subject scale, framing, and horizon remain visibly unchanged",
    "dolly_in": (
        "the subject becomes gradually larger while foreground/background parallax proves "
        "that the camera physically moved forward"
    ),
    "dolly_out": (
        "the subject becomes gradually smaller and more surrounding environment enters the "
        "frame while the authored subject remains composed"
    ),
    "tracking_front": (
        "the camera visibly retreats in front of the approaching subject; subject scale and "
        "frontal composition remain stable while the environment translates behind them"
    ),
    "tracking_rear": (
        "the camera visibly advances behind the moving subject at a stable following distance"
    ),
    "tracking_left": (
        "lateral background parallax moves right while subject distance and body scale stay stable"
    ),
    "tracking_right": (
        "lateral background parallax moves left while subject distance and body scale stay stable"
    ),
    "pan_left": "the fixed camera rotates left and reveals authored content on the left",
    "pan_right": "the fixed camera rotates right and reveals authored content on the right",
    "zoom_in": "framing tightens without translation parallax because camera position is fixed",
    "zoom_out": "framing widens without translation parallax because camera position is fixed",
}


def camera_motion_execution_prompt(shot: Mapping[str, Any]) -> str:
    """Render a machine-readable execution block with observable success evidence."""
    contract = shot.get("camera_motion_contract")
    if not isinstance(contract, Mapping):
        contract = build_camera_motion_contract(shot)
    movement = str(contract.get("movement") or "static")
    observable = _OBSERVABLE_CAMERA_EVIDENCE.get(
        movement,
        "the declared physical camera path creates a continuous, visible framing change",
    )
    return "\n".join(
        (
            "[camera-motion-execution-v2]",
            f"movement={movement}",
            f"start_frame={contract.get('start')}",
            f"physical_path={contract.get('process')}",
            f"end_frame={contract.get('end')}",
            f"observable_success={observable}",
            (
                "forbidden_substitution=camera drift, subject-only movement, background-only "
                "animation, optical zoom, or a diegetic character moving cannot substitute for "
                "the declared physical viewer-camera path"
            ),
        )
    )


def camera_motion_negative_prompt(shot: Mapping[str, Any]) -> str:
    """Return camera/perspective negatives from persisted or legacy metadata."""
    contract = shot.get("camera_motion_contract")
    if not isinstance(contract, Mapping):
        contract = build_camera_motion_contract(shot)
    return str(contract.get("negative") or BASE_CAMERA_NEGATIVE)
