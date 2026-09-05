"""Deterministic, identity-neutral motion-blueprint compiler.

This module is acceptance-only.  It performs no network access and is not imported
by the production CLI, Lifecycle, Graph, or Phase owners.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

BLUEPRINT_SCHEMA = "honcut.seedance-motion-blueprint.v2"
POLICY_SCHEMA = "honcut.motion-blueprint-policy.v2"
RENDERER_ID = "honcut.identity-neutral-motion-renderer.v2"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class SourceLineage(StrictModel):
    canonical_visual_contract_path: str
    canonical_visual_contract_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    continuity_plan_path: str
    continuity_plan_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_receipt_path: str
    source_receipt_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]


class MotionEvent(StrictModel):
    event_id: str
    order: Annotated[int, Field(ge=1)]
    actor_ids: tuple[str, ...]
    primitive: str
    direction: Literal["left", "right", "forward", "backward", "up", "down"] = "right"
    source_action_group_id: str
    source_action_unit_ids: tuple[str, ...]
    prop_contact: bool = False

    @model_validator(mode="after")
    def validate_event(self) -> "MotionEvent":
        if not self.event_id.strip() or not self.source_action_group_id.strip():
            raise ValueError("motion event ids must be non-empty")
        if not self.actor_ids or any(not value.strip() for value in self.actor_ids):
            raise ValueError("motion event requires canonical actors")
        if not self.source_action_unit_ids:
            raise ValueError("motion event requires source action lineage")
        return self


class CameraTrack(StrictModel):
    primitive: Literal["static", "pan_left", "pan_right", "push_in", "pull_out", "tilt_up", "tilt_down"]
    magnitude: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35


class MotionBlueprintInput(StrictModel):
    schema_id: Literal["honcut.seedance-motion-blueprint.v2"] = Field(
        default=BLUEPRINT_SCHEMA,
        alias="schema",
    )
    beat_id: str
    duration_s: Annotated[float, Field(ge=4.0, le=4.0)] = 4.0
    fps: Literal[24] = 24
    width: Literal[854] = 854
    height: Literal[480] = 480
    actor_ids: tuple[str, ...]
    events: tuple[MotionEvent, ...]
    camera: CameraTrack
    lineage: SourceLineage

    @model_validator(mode="after")
    def validate_contract(self) -> "MotionBlueprintInput":
        if not self.beat_id.strip():
            raise ValueError("beat_id must be non-empty")
        if len(self.actor_ids) != 1:
            raise ValueError("initial capability gate requires exactly one canonical actor")
        if not self.events:
            raise ValueError("motion blueprint requires at least one event")
        orders = [event.order for event in self.events]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError("motion events must be contiguous and ordered")
        actor_set = set(self.actor_ids)
        if any(set(event.actor_ids) != actor_set for event in self.events):
            raise ValueError("event actors must match the single-actor gate scope")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("motion event ids must be unique")
        return self


class MotionPolicy(StrictModel):
    schema_id: Literal["honcut.motion-blueprint-policy.v2"] = Field(
        default=POLICY_SCHEMA,
        alias="schema",
    )
    action_onset_max_s: float = 0.5
    setup_anchor_max_s: float = 0.15
    terminal_hold_max_fraction: float = 0.10
    minimum_root_displacement: float = 0.20
    minimum_joint_displacement: float = 0.25
    minimum_peak_root_speed: float = 0.60
    minimum_peak_joint_speed: float = 0.80
    minimum_major_joint_participants: int = 4
    major_joint_displacement: float = 0.10
    perceptible_root_onset: float = 0.025
    perceptible_joint_onset: float = 0.045
    apex_max_fraction: float = 0.72
    minimum_event_duration_s: float = 0.30
    minimum_actor_height_fraction: float = 0.46
    maximum_actor_height_fraction: float = 0.95
    minimum_centroid_travel_fraction: float = 0.045
    minimum_p90_foreground_change: float = 0.07
    minimum_active_transition_fraction: float = 0.25


class MotionBlueprintManifest(StrictModel):
    schema_id: Literal["honcut.seedance-motion-blueprint.v2"] = Field(alias="schema")
    policy_schema: Literal["honcut.motion-blueprint-policy.v2"]
    policy_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    renderer_id: Literal["honcut.identity-neutral-motion-renderer.v2"]
    identity_authority: Literal[False]
    authority_roles: tuple[
        Literal["motion_timing", "body_kinematics", "camera_motion", "contact_timing"],
        ...,
    ]
    non_authority_roles: tuple[
        Literal[
            "character_identity",
            "face_geometry",
            "hair_geometry",
            "wardrobe",
            "prop_appearance",
            "cinematic_pixels",
        ],
        ...,
    ]
    contract_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    semantic_frames_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    media_path: str
    media_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    media_size_bytes: Annotated[int, Field(gt=0)]
    measurements: dict[str, Any]
    technical_probe: dict[str, Any]
    lineage: SourceLineage
    provider_request_count: Literal[0]


MOTION_POLICY = MotionPolicy()

# Normalized joint targets.  Values are offsets from the actor root in a y-down
# coordinate system.  The registry contains choreography classes, never story prose.
_BASE_POSE = {
    "head": (0.00, -0.34), "neck": (0.00, -0.25),
    "left_shoulder": (-0.10, -0.21), "right_shoulder": (0.10, -0.21),
    "left_elbow": (-0.13, -0.07), "right_elbow": (0.13, -0.07),
    "left_wrist": (-0.12, 0.08), "right_wrist": (0.12, 0.08),
    "hip": (0.00, 0.05), "left_knee": (-0.07, 0.24), "right_knee": (0.07, 0.24),
    "left_ankle": (-0.10, 0.43), "right_ankle": (0.10, 0.43),
}


def _pose(**updates: tuple[float, float]) -> dict[str, tuple[float, float]]:
    result = dict(_BASE_POSE)
    result.update(updates)
    return result


MOTION_PRIMITIVES: dict[str, dict[str, Any]] = {
    "ready": {"root": (0.08, 0.02), "pose": _pose(left_wrist=(0.16, -0.15), right_wrist=(-0.15, -0.10), left_knee=(-0.12, 0.23), right_knee=(0.14, 0.24))},
    "locomotion": {"root": (0.30, 0.00), "pose": _pose(left_elbow=(-0.20, -0.10), right_elbow=(0.22, -0.02), left_knee=(0.12, 0.20), right_knee=(-0.13, 0.27), left_ankle=(0.22, 0.40), right_ankle=(-0.20, 0.43))},
    "evade": {"root": (0.30, 0.10), "pose": _pose(head=(-0.15, -0.29), neck=(-0.12, -0.20), left_shoulder=(-0.23, -0.14), right_shoulder=(-0.02, -0.19), left_wrist=(-0.30, 0.02), right_wrist=(0.08, -0.02), hip=(0.10, 0.10), left_knee=(-0.03, 0.27), right_knee=(0.25, 0.20), left_ankle=(-0.18, 0.43), right_ankle=(0.35, 0.36))},
    "strike": {"root": (0.20, 0.00), "pose": _pose(left_elbow=(0.07, -0.25), left_wrist=(0.36, -0.24), right_elbow=(-0.02, -0.12), right_wrist=(0.22, -0.18), left_knee=(-0.05, 0.25), right_knee=(0.21, 0.20), right_ankle=(0.32, 0.40))},
    "attack": {"root": (0.22, 0.00), "pose": _pose(left_elbow=(0.08, -0.27), left_wrist=(0.38, -0.30), right_elbow=(0.02, -0.11), right_wrist=(0.27, -0.16), left_knee=(-0.08, 0.25), right_knee=(0.22, 0.19), right_ankle=(0.34, 0.40))},
    "kick": {"root": (0.12, -0.03), "pose": _pose(left_wrist=(-0.23, -0.10), right_wrist=(0.19, -0.16), left_knee=(-0.08, 0.25), right_knee=(0.29, 0.05), right_ankle=(0.46, -0.02))},
    "block": {"root": (0.03, 0.02), "pose": _pose(left_elbow=(-0.08, -0.28), right_elbow=(0.10, -0.28), left_wrist=(0.03, -0.38), right_wrist=(-0.02, -0.34), left_knee=(-0.12, 0.25), right_knee=(0.15, 0.25))},
    "grapple": {"root": (0.18, 0.01), "pose": _pose(left_elbow=(0.10, -0.16), right_elbow=(0.14, -0.10), left_wrist=(0.30, -0.12), right_wrist=(0.31, -0.04), left_knee=(-0.03, 0.25), right_knee=(0.22, 0.23))},
    "throw": {"root": (0.18, -0.01), "pose": _pose(left_elbow=(-0.18, -0.31), right_elbow=(0.18, -0.31), left_wrist=(-0.30, -0.42), right_wrist=(0.32, -0.40), left_knee=(-0.16, 0.23), right_knee=(0.20, 0.22))},
    "jump": {"root": (0.12, -0.22), "pose": _pose(left_wrist=(-0.25, -0.27), right_wrist=(0.25, -0.27), left_knee=(-0.16, 0.17), right_knee=(0.16, 0.17), left_ankle=(-0.06, 0.31), right_ankle=(0.06, 0.31))},
    "crouch": {"root": (0.10, 0.16), "pose": _pose(head=(0.04, -0.24), neck=(0.03, -0.15), left_wrist=(-0.20, 0.02), right_wrist=(0.20, 0.02), left_knee=(-0.18, 0.21), right_knee=(0.18, 0.21), left_ankle=(-0.27, 0.35), right_ankle=(0.27, 0.35))},
    "lean_back": {"root": (0.08, 0.04), "pose": _pose(head=(-0.17, -0.28), neck=(-0.13, -0.20), left_shoulder=(-0.23, -0.15), right_shoulder=(-0.03, -0.20), hip=(0.09, 0.07), left_ankle=(-0.16, 0.43), right_ankle=(0.18, 0.43))},
    "lean_forward": {"root": (0.10, 0.02), "pose": _pose(head=(0.15, -0.28), neck=(0.12, -0.20), left_shoulder=(0.02, -0.19), right_shoulder=(0.22, -0.14), hip=(-0.06, 0.07), left_ankle=(-0.15, 0.43), right_ankle=(0.18, 0.43))},
    "prop_hold": {"root": (0.04, 0.00), "pose": _pose(left_elbow=(-0.05, -0.11), right_elbow=(0.06, -0.11), left_wrist=(0.04, -0.08), right_wrist=(0.08, -0.03))},
    "prop_use": {"root": (0.19, 0.00), "pose": _pose(left_elbow=(0.02, -0.23), right_elbow=(0.17, -0.18), left_wrist=(0.23, -0.29), right_wrist=(0.36, -0.23))},
    "spatial": {"root": (0.18, 0.00), "pose": _pose(left_wrist=(-0.23, -0.12), right_wrist=(0.28, -0.10), left_knee=(-0.12, 0.23), right_knee=(0.19, 0.22))},
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess_legacy_blueprint_manifest(path: Path) -> dict[str, Any]:
    """Classify a v1 manifest as audit-only without promoting or rewriting it."""
    manifest_path = path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("legacy motion blueprint manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema") != "honcut.seedance-motion-blueprint.v1":
        raise ValueError("legacy motion blueprint audit requires schema v1")
    if payload.get("policy_schema") != "honcut.motion-blueprint-policy.v1":
        raise ValueError("legacy motion blueprint policy is not auditable")
    media_path = Path(str(payload.get("media_path") or "")).resolve()
    expected_media_hash = str(payload.get("media_sha256") or "")
    if not media_path.is_file() or sha256_file(media_path) != expected_media_hash:
        raise ValueError("legacy motion blueprint media hash mismatch")
    measurements = payload.get("measurements") or {}
    fps = float(measurements.get("fps") or 0)
    if fps <= 0:
        raise ValueError("legacy motion blueprint lacks a valid frame rate")
    slow_events: list[dict[str, Any]] = []
    for event in measurements.get("events") or []:
        if event.get("primitive") in _SETUP_PRIMITIVES:
            continue
        duration_s = (
            int(event.get("end_frame") or 0) - int(event.get("start_frame") or 0) + 1
        ) / fps
        if duration_s <= 0:
            raise ValueError("legacy motion blueprint has invalid event timing")
        root_velocity = float(event.get("root_displacement") or 0) / duration_s
        joint_velocity = float(event.get("max_joint_displacement") or 0) / duration_s
        if (
            root_velocity < MOTION_POLICY.minimum_peak_root_speed
            and joint_velocity < MOTION_POLICY.minimum_peak_joint_speed
        ):
            slow_events.append({
                "event_id": str(event.get("event_id") or ""),
                "root_endpoint_velocity": round(root_velocity, 6),
                "joint_endpoint_velocity": round(joint_velocity, 6),
            })
    return {
        "schema": "honcut.motion-blueprint-legacy-assessment.v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "media_sha256": expected_media_hash,
        "source_schema": payload["schema"],
        "slow_dynamic_events": slow_events,
        "admission_status": "paid_admission_blocked",
        "audit_only": True,
        "reason": "legacy endpoint-only policy lacks v2 perceptual kinetics",
    }


def policy_sha256(policy: MotionPolicy = MOTION_POLICY) -> str:
    return hashlib.sha256(canonical_json(policy.model_dump(mode="json"))).hexdigest()


def validate_supported_actions(contract: MotionBlueprintInput) -> None:
    unsupported = sorted({event.primitive for event in contract.events} - MOTION_PRIMITIVES.keys())
    if unsupported:
        refs = sorted(
            event.source_action_group_id
            for event in contract.events
            if event.primitive in unsupported
        )
        raise ValueError(f"unsupported motion primitives {unsupported}; canonical refs={refs}")


def _ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _mirrored(value: tuple[float, float], direction: str) -> tuple[float, float]:
    if direction in {"left", "backward"}:
        return (-value[0], value[1])
    if direction == "up":
        return (value[0], value[1] - abs(value[0]) * 0.25)
    if direction == "down":
        return (value[0], value[1] + abs(value[0]) * 0.25)
    return value


_SETUP_PRIMITIVES = frozenset({"ready", "prop_hold"})
_MAJOR_JOINTS = (
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
    "hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _interpolate_pair(
    first: tuple[float, float],
    second: tuple[float, float],
    progress: float,
) -> tuple[float, float]:
    eased = _ease(min(1.0, max(0.0, progress)))
    return (
        first[0] + (second[0] - first[0]) * eased,
        first[1] + (second[1] - first[1]) * eased,
    )


def _phase_value(
    start: tuple[float, float],
    target: tuple[float, float],
    progress: float,
    *,
    root: bool = False,
) -> tuple[float, float]:
    """Compile anticipation, explosive apex, overshoot and recovery.

    The curve only exaggerates the source-derived primitive vector.  It never
    introduces a second action or a new actor/object relationship.
    """
    delta = (target[0] - start[0], target[1] - start[1])
    anticipation_scale = -0.30 if not root else -0.16
    apex_scale = 1.55
    recovery_scale = 0.35 if not root else 0.55
    terminal_scale = 1.0
    knots = (
        (0.0, start),
        (0.08, (start[0] + delta[0] * anticipation_scale, start[1] + delta[1] * anticipation_scale)),
        (0.35, (start[0] + delta[0] * apex_scale, start[1] + delta[1] * apex_scale)),
        (0.70, (start[0] + delta[0] * recovery_scale, start[1] + delta[1] * recovery_scale)),
        (1.0, (start[0] + delta[0] * terminal_scale, start[1] + delta[1] * terminal_scale)),
    )
    for index in range(1, len(knots)):
        previous_fraction, previous_value = knots[index - 1]
        fraction, value = knots[index]
        if progress <= fraction:
            local = (progress - previous_fraction) / max(1e-9, fraction - previous_fraction)
            return _interpolate_pair(previous_value, value, local)
    return knots[-1][1]


def compile_semantic_frames(
    contract: MotionBlueprintInput,
    policy: MotionPolicy = MOTION_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile ordered events to normalized, measurable actor/camera frames."""
    validate_supported_actions(contract)
    total_frames = round(contract.duration_s * contract.fps)
    onset_frames = max(1, round(0.08 * contract.fps))
    terminal_frames = max(1, math.floor(total_frames * policy.terminal_hold_max_fraction))
    active_frames = total_frames - onset_frames - terminal_frames
    minimum_event_frames = max(2, math.ceil(policy.minimum_event_duration_s * contract.fps))
    setup_indexes = [
        index
        for index, event in enumerate(contract.events)
        if event.primitive in _SETUP_PRIMITIVES
    ]
    dynamic_indexes = [
        index for index in range(len(contract.events)) if index not in setup_indexes
    ]
    if not dynamic_indexes:
        raise ValueError("motion blueprint capability gate requires a dynamic event")
    setup_cap = max(1, math.floor(policy.setup_anchor_max_s * contract.fps))
    if len(setup_indexes) > setup_cap:
        raise ValueError("setup anchors cannot fit the 0.15 second capability window")
    counts = [1 if index in setup_indexes else minimum_event_frames for index in range(len(contract.events))]
    if setup_indexes:
        for offset in range(setup_cap - len(setup_indexes)):
            counts[setup_indexes[offset % len(setup_indexes)]] += 1
    if active_frames < sum(counts):
        raise ValueError("duration cannot represent all ordered motion events")
    remaining = active_frames - sum(counts)
    for offset in range(remaining):
        counts[dynamic_indexes[offset % len(dynamic_indexes)]] += 1
    boundaries = [onset_frames]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    frames: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    previous_root = (0.28, 0.55)
    event_targets: list[tuple[dict[str, tuple[float, float]], tuple[float, float]]] = []
    for event in contract.events:
        primitive = MOTION_PRIMITIVES[event.primitive]
        pose = {joint: _mirrored(value, event.direction) for joint, value in primitive["pose"].items()}
        delta = _mirrored(primitive["root"], event.direction)
        terminal_scale = 1.0 if event.primitive in _SETUP_PRIMITIVES else 1.18
        target_root = (
            min(0.80, max(0.16, previous_root[0] + delta[0] * terminal_scale)),
            min(0.76, max(0.32, previous_root[1] + delta[1] * terminal_scale)),
        )
        event_targets.append((pose, target_root))
        previous_root = target_root
    previous_root = (0.28, 0.55)
    for frame_index in range(total_frames):
        if frame_index < onset_frames:
            event_index, progress = 0, 0.0
        elif frame_index >= boundaries[-1]:
            event_index, progress = len(contract.events) - 1, 1.0
        else:
            event_index = min(len(contract.events) - 1, next(i for i in range(len(contract.events)) if boundaries[i] <= frame_index < boundaries[i + 1]))
            start, end = boundaries[event_index], boundaries[event_index + 1]
            progress = (frame_index - start + 1) / max(1, end - start)
        start_pose = dict(_BASE_POSE) if event_index == 0 else event_targets[event_index - 1][0]
        start_root = (0.28, 0.55) if event_index == 0 else event_targets[event_index - 1][1]
        target_pose, target_root = event_targets[event_index]
        is_setup = contract.events[event_index].primitive in _SETUP_PRIMITIVES
        joints = {}
        for joint in _BASE_POSE:
            value = (
                _interpolate_pair(start_pose[joint], target_pose[joint], progress)
                if is_setup
                else _phase_value(start_pose[joint], target_pose[joint], progress)
            )
            joints[joint] = [round(value[0], 6), round(value[1], 6)]
        root_value = (
            _interpolate_pair(start_root, target_root, progress)
            if is_setup
            else _phase_value(start_root, target_root, progress, root=True)
        )
        root = [
            round(min(0.84, max(0.12, root_value[0])), 6),
            round(min(0.80, max(0.28, root_value[1])), 6),
        ]
        camera_progress = frame_index / max(1, total_frames - 1)
        frames.append({
            "frame": frame_index,
            "event_id": contract.events[event_index].event_id,
            "root": root,
            "joints": joints,
            "camera": _camera_transform(contract.camera, camera_progress),
            "prop_contact": contract.events[event_index].prop_contact,
        })
    for index, event in enumerate(contract.events):
        intervals.append({
            "event_id": event.event_id,
            "order": event.order,
            "primitive": event.primitive,
            "start_frame": boundaries[index],
            "end_frame": boundaries[index + 1] - 1,
            "source_action_group_id": event.source_action_group_id,
            "source_action_unit_ids": list(event.source_action_unit_ids),
            "admission_role": "setup_anchor" if event.primitive in _SETUP_PRIMITIVES else "dynamic_action",
        })
    return frames, intervals


def _camera_transform(camera: CameraTrack, progress: float) -> dict[str, float]:
    x = y = 0.0
    zoom = 1.0
    amount = camera.magnitude * progress
    if camera.primitive == "pan_left": x = -amount
    elif camera.primitive == "pan_right": x = amount
    elif camera.primitive == "tilt_up": y = -amount
    elif camera.primitive == "tilt_down": y = amount
    elif camera.primitive == "push_in": zoom = 1.0 + amount
    elif camera.primitive == "pull_out": zoom = 1.0 - amount * 0.35
    return {"x": round(x, 6), "y": round(y, 6), "zoom": round(zoom, 6)}


_BONES = (
    ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "hip"), ("hip", "left_knee"), ("left_knee", "left_ankle"),
    ("hip", "right_knee"), ("right_knee", "right_ankle"),
)


def _render_frame(frame: dict[str, Any], width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    camera = frame["camera"]
    scale = min(width, height) * 0.72 * camera["zoom"]
    cx = width * (0.5 - camera["x"] * 0.18)
    cy = height * (0.52 - camera["y"] * 0.18)
    root = frame["root"]
    points = {
        joint: (round(cx + (root[0] - 0.5 + value[0]) * scale), round(cy + (root[1] - 0.55 + value[1]) * scale))
        for joint, value in frame["joints"].items()
    }
    for first, second in _BONES:
        draw.line((points[first], points[second]), fill=(225, 229, 233), width=12)
    head = points["head"]
    draw.ellipse((head[0] - 22, head[1] - 22, head[0] + 22, head[1] + 22), outline=(225, 229, 233), width=10)
    for point in points.values():
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=(162, 171, 180))
    if frame["prop_contact"]:
        left, right = points["left_wrist"], points["right_wrist"]
        draw.line((left[0] - 36, left[1] - 15, right[0] + 48, right[1] + 14), fill=(190, 194, 198), width=7)
        draw.ellipse((right[0] - 8, right[1] - 8, right[0] + 8, right[1] + 8), outline=(240, 200, 92), width=4)
    return image


def _distance(first: list[float], second: list[float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def measure_semantic_frames(
    contract: MotionBlueprintInput,
    frames: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    policy: MotionPolicy = MOTION_POLICY,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("blueprint has no frames")
    dynamic_intervals = [
        interval for interval in intervals if interval["admission_role"] == "dynamic_action"
    ]
    if not dynamic_intervals:
        raise ValueError("motion blueprint has no dynamic action")
    first_dynamic = dynamic_intervals[0]
    dynamic_baseline_index = max(0, first_dynamic["start_frame"] - 1)
    dynamic_baseline = frames[dynamic_baseline_index]
    onset_frame = next((
        index
        for index in range(first_dynamic["start_frame"], len(frames))
        if _distance(dynamic_baseline["root"], frames[index]["root"])
        >= policy.perceptible_root_onset
        or sum(
            _distance(dynamic_baseline["joints"][joint], frames[index]["joints"][joint])
            >= policy.perceptible_joint_onset
            for joint in _MAJOR_JOINTS
        )
        >= 2
    ), len(frames))
    event_measurements: list[dict[str, Any]] = []
    for interval in intervals:
        baseline_index = max(0, interval["start_frame"] - 1)
        baseline = frames[baseline_index]
        event_frames = frames[interval["start_frame"]:interval["end_frame"] + 1]
        duration_s = len(event_frames) / contract.fps
        root_displacements = [
            _distance(baseline["root"], frame["root"]) for frame in event_frames
        ]
        joint_displacements = {
            joint: max(
                _distance(baseline["joints"][joint], frame["joints"][joint])
                for frame in event_frames
            )
            for joint in _MAJOR_JOINTS
        }
        root_speeds: list[float] = []
        joint_speeds: list[float] = []
        previous = baseline
        for frame in event_frames:
            root_speeds.append(_distance(previous["root"], frame["root"]) * contract.fps)
            joint_speeds.append(max(
                _distance(previous["joints"][joint], frame["joints"][joint]) * contract.fps
                for joint in _MAJOR_JOINTS
            ))
            previous = frame
        scores = [
            root_displacements[index] / max(policy.minimum_root_displacement, 1e-9)
            + max(
                _distance(baseline["joints"][joint], frame["joints"][joint])
                for joint in _MAJOR_JOINTS
            ) / max(policy.minimum_joint_displacement, 1e-9)
            for index, frame in enumerate(event_frames)
        ]
        apex_offset = max(range(len(scores)), key=scores.__getitem__)
        apex_fraction = (apex_offset + 1) / len(event_frames)
        root_delta = max(root_displacements)
        joint_delta = max(joint_displacements.values())
        participant_count = sum(
            displacement >= policy.major_joint_displacement
            for displacement in joint_displacements.values()
        )
        is_setup = interval["admission_role"] == "setup_anchor"
        passes_amplitude = is_setup or (
            (root_delta >= policy.minimum_root_displacement or joint_delta >= policy.minimum_joint_displacement)
            and (max(root_speeds) >= policy.minimum_peak_root_speed or max(joint_speeds) >= policy.minimum_peak_joint_speed)
            and participant_count >= policy.minimum_major_joint_participants
            and apex_fraction <= policy.apex_max_fraction
        )
        event_measurements.append({
            **interval,
            "duration_s": round(duration_s, 6),
            "peak_root_displacement": round(root_delta, 6),
            "peak_joint_displacement": round(joint_delta, 6),
            "peak_root_speed": round(max(root_speeds), 6),
            "peak_joint_speed": round(max(joint_speeds), 6),
            "major_joint_participants": participant_count,
            "apex_frame": interval["start_frame"] + apex_offset,
            "apex_fraction": round(apex_fraction, 6),
            "passes_kinetics": passes_amplitude,
            # Kept as a compatibility projection for existing acceptance readers.
            "passes_amplitude": passes_amplitude,
        })
    terminal_frames = len(frames) - 1 - intervals[-1]["end_frame"]
    result = {
        "duration_s": contract.duration_s,
        "fps": contract.fps,
        "frame_count": len(frames),
        "action_onset_s": round(onset_frame / contract.fps, 6),
        "terminal_hold_frames": terminal_frames,
        "terminal_hold_fraction": round(terminal_frames / len(frames), 6),
        "ordered_event_ids": [item["event_id"] for item in event_measurements],
        "events": event_measurements,
    }
    if result["action_onset_s"] > policy.action_onset_max_s:
        raise ValueError("motion blueprint action onset exceeds policy")
    if result["terminal_hold_fraction"] > policy.terminal_hold_max_fraction:
        raise ValueError("motion blueprint terminal hold exceeds policy")
    setup_overruns = [
        item["event_id"]
        for item in event_measurements
        if item["admission_role"] == "setup_anchor"
        and item["duration_s"] > policy.setup_anchor_max_s + (1 / contract.fps)
    ]
    if setup_overruns:
        raise ValueError(f"motion blueprint setup anchors exceed policy: {setup_overruns}")
    setup_duration = sum(
        item["duration_s"]
        for item in event_measurements
        if item["admission_role"] == "setup_anchor"
    )
    if setup_duration > policy.setup_anchor_max_s + 1e-9:
        raise ValueError("motion blueprint total setup window exceeds policy")
    failed = [
        item["event_id"]
        for item in event_measurements
        if item["admission_role"] == "dynamic_action" and not item["passes_kinetics"]
    ]
    if failed:
        raise ValueError(f"motion blueprint events have sub-threshold kinetics: {failed}")
    return result


def compile_motion_blueprint(contract: MotionBlueprintInput, output_path: Path, policy: MotionPolicy = MOTION_POLICY) -> dict[str, Any]:
    """Compile a deterministic H.264 MP4 and return its strict manifest payload."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("motion blueprint compilation requires ffmpeg and ffprobe")
    frames, intervals = compile_semantic_frames(contract, policy)
    measurements = measure_semantic_frames(contract, frames, intervals, policy)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="honcut-motion-blueprint-") as temporary:
        frame_dir = Path(temporary)
        for frame in frames:
            _render_frame(frame, contract.width, contract.height).save(frame_dir / f"{frame['frame']:06d}.png", optimize=False, compress_level=9)
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(contract.fps),
            "-i", str(frame_dir / "%06d.png"), "-an", "-c:v", "libx264", "-preset", "veryslow",
            "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(contract.fps), "-g", str(contract.fps * 2),
            "-sc_threshold", "0", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-movflags", "+faststart", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
        if completed.returncode != 0:
            raise RuntimeError("motion blueprint ffmpeg encoding failed")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate:format=duration,size", "-of", "json", str(output),
    ], capture_output=True, text=True, check=False, timeout=30)
    if probe.returncode != 0:
        raise RuntimeError("motion blueprint ffprobe validation failed")
    technical = json.loads(probe.stdout)
    rendered_motion = measure_rendered_blueprint_motion(output, policy)
    if rendered_motion["deterministic_motion_pass"] is not True:
        raise ValueError("rendered motion blueprint has sub-threshold perceptual kinetics")
    measurements["rendered_motion"] = rendered_motion
    semantic_sha256 = hashlib.sha256(canonical_json({"frames": frames, "events": intervals})).hexdigest()
    contract_sha256 = hashlib.sha256(canonical_json(contract.model_dump(mode="json"))).hexdigest()
    manifest = {
        "schema": BLUEPRINT_SCHEMA,
        "policy_schema": policy.schema_id,
        "policy_sha256": policy_sha256(policy),
        "renderer_id": RENDERER_ID,
        "identity_authority": False,
        "authority_roles": ["motion_timing", "body_kinematics", "camera_motion", "contact_timing"],
        "non_authority_roles": ["character_identity", "face_geometry", "hair_geometry", "wardrobe", "prop_appearance", "cinematic_pixels"],
        "contract_sha256": contract_sha256,
        "semantic_frames_sha256": semantic_sha256,
        "media_path": str(output),
        "media_sha256": sha256_file(output),
        "media_size_bytes": output.stat().st_size,
        "measurements": measurements,
        "technical_probe": technical,
        "lineage": contract.lineage.model_dump(mode="json"),
        "provider_request_count": 0,
    }
    return MotionBlueprintManifest.model_validate(manifest).model_dump(mode="json")


def inspect_identity_neutral_pixels(video_path: Path) -> dict[str, Any]:
    """Sample decoded pixels and reject annotation/control colours or empty media."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    sampled = 0
    forbidden = 0
    non_background = 0
    while sampled < 12:
        ok, frame = capture.read()
        if not ok:
            break
        if sampled % 2 == 0:
            b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            forbidden += int((((r > 180) & (g < 90) & (b < 90)) | ((b > 180) & (r < 90) & (g < 150))).sum())
            non_background += int(((r > 80) | (g > 80) | (b > 80)).sum())
        sampled += 1
    capture.release()
    if sampled == 0 or non_background == 0 or forbidden:
        raise ValueError("motion blueprint pixel guard rejected the rendered media")
    return {"sampled_frames": sampled, "forbidden_annotation_pixels": forbidden, "non_background_pixels": non_background}


def measure_rendered_blueprint_motion(
    video_path: Path,
    policy: MotionPolicy = MOTION_POLICY,
) -> dict[str, Any]:
    """Measure whether the encoded neutral actor visibly moves at playback scale."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    masks: list[Any] = []
    heights: list[float] = []
    centroids: list[tuple[float, float]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = gray > 70
        points = np.argwhere(mask)
        if points.size == 0:
            capture.release()
            raise ValueError("rendered motion blueprint has an empty actor frame")
        y_min, _x_min = points.min(axis=0)
        y_max, _x_max = points.max(axis=0)
        heights.append(float((y_max - y_min + 1) / frame.shape[0]))
        centroids.append((float(points[:, 1].mean()), float(points[:, 0].mean())))
        masks.append(mask)
    capture.release()
    if fps <= 0 or len(masks) < 2:
        raise ValueError("rendered motion blueprint has no measurable frames")
    changes: list[float] = []
    for previous, current in itertools.pairwise(masks):
        union = int(np.logical_or(previous, current).sum())
        changes.append(float(np.logical_xor(previous, current).sum() / max(1, union)))
    width = masks[0].shape[1]
    height = masks[0].shape[0]
    diagonal = math.hypot(width, height)
    first_centroid = centroids[0]
    centroid_travel = max(
        math.hypot(point[0] - first_centroid[0], point[1] - first_centroid[1]) / diagonal
        for point in centroids
    )
    active = [value >= 0.025 for value in changes]
    p90 = float(np.percentile(np.asarray(changes), 90))
    result = {
        "schema": "honcut.motion-blueprint-rendered-measurement.v2",
        "fps": round(fps, 6),
        "frame_count": len(masks),
        "median_actor_height_fraction": round(float(np.median(np.asarray(heights))), 6),
        "max_actor_height_fraction": round(max(heights), 6),
        "centroid_travel_fraction": round(centroid_travel, 6),
        "p90_foreground_change": round(p90, 6),
        "active_transition_fraction": round(sum(active) / len(active), 6),
    }
    result["deterministic_motion_pass"] = bool(
        result["median_actor_height_fraction"] >= policy.minimum_actor_height_fraction
        and result["max_actor_height_fraction"] <= policy.maximum_actor_height_fraction
        and result["centroid_travel_fraction"] >= policy.minimum_centroid_travel_fraction
        and result["p90_foreground_change"] >= policy.minimum_p90_foreground_change
        and result["active_transition_fraction"] >= policy.minimum_active_transition_fraction
    )
    return result


def measure_output_motion(video_path: Path) -> dict[str, Any]:
    """Return deterministic coarse motion evidence without semantic pass authority."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    previous = None
    changes: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if previous is not None:
            changes.append(float(np.mean(cv2.absdiff(previous, gray))))
        previous = gray
    capture.release()
    if fps <= 0 or not changes:
        raise ValueError("generated output has no measurable video motion")
    threshold = 0.8
    active = [value >= threshold for value in changes]
    onset_index = next((index for index, value in enumerate(active) if value), len(active))
    terminal_idle = 0
    for value in reversed(active):
        if value:
            break
        terminal_idle += 1
    p90 = float(np.percentile(np.asarray(changes), 90))
    result = {
        "schema": "honcut.motion-blueprint-output-measurement.v1",
        "fps": round(fps, 6),
        "frame_transition_count": len(changes),
        "mean_frame_change": round(float(np.mean(changes)), 6),
        "p90_frame_change": round(p90, 6),
        "active_transition_fraction": round(sum(active) / len(active), 6),
        "motion_onset_s": round(onset_index / fps, 6),
        "terminal_idle_fraction": round(terminal_idle / len(active), 6),
    }
    result["deterministic_motion_pass"] = bool(
        result["p90_frame_change"] >= threshold
        and result["active_transition_fraction"] >= 0.25
        and result["motion_onset_s"] <= MOTION_POLICY.action_onset_max_s
        and result["terminal_idle_fraction"] <= 0.20
    )
    return result
