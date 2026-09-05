"""Deterministic canonical body kinematics owned by the body-action contract.

The Provider-facing understanding DTO intentionally remains small.  This module
turns its already validated mechanics into JSON-safe, fixed-point motion facts
and later projects those facts onto final GAU/Pxx lineage.  It performs no I/O
and no model or Provider call.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

SOURCE_KINEMATICS_SCHEMA = "honcut.canonical-action-kinematics.v1"
KINEMATICS_PROJECTION_SCHEMA = "honcut.generation-action-kinematics.v1"

CHANNEL_ORDER = (
    "root",
    "waist_torso",
    "head",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "left_leg",
    "left_foot",
    "right_leg",
    "right_foot",
)
_PHASE_WINDOWS = (
    ("load", 0, 150, 0.20),
    ("drive", 150, 430, 0.68),
    ("apex_contact", 430, 650, 1.00),
    ("follow_through", 650, 850, 0.72),
    ("settle", 850, 1000, 0.38),
)
_TRANSFORM_PROGRESS = {
    "load": 0.0,
    "drive": 0.25,
    "apex_contact": 0.55,
    "follow_through": 0.85,
    "settle": 1.0,
}
_SOURCE_FIELDS = frozenset(
    {
        "schema",
        "micro_action_index",
        "actor_tracks",
        "source_evidence_sha256",
        "policy_sha256",
        "kinematics_sha256",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "beat_id",
        "generation_action_unit_id",
        "source_micro_action_indexes",
        "source_kinematics_sha256s",
        "actor_tracks",
        "action_units",
        "policy_sha256",
        "projection_sha256",
    }
)
_CHANNEL_FIELDS = frozenset(
    {
        "role",
        "translation_milli",
        "rotation_mdeg",
        "activation_milli",
        "amplitude",
        "support",
        "contact",
    }
)
_TRANSFORM_FIELDS = frozenset(
    {
        "kind",
        "axis",
        "direction",
        "amount_mdeg",
        "amount_range_mdeg",
        "airborne",
        "support_release",
        "landing_state",
    }
)
_POLICY = {
    "schema": "honcut.canonical-action-kinematics-policy.v1",
    "coordinate_space": "actor_local_pxx_start_yaw_zero",
    "fixed_point": {"translation": 1000, "rotation_degrees": 1000},
    "channel_order": CHANNEL_ORDER,
    "phase_windows": _PHASE_WINDOWS,
    "amplitude_minima": {
        "large": {"root": 450, "waist_rotation_mdeg": 25_000, "major_chain": 420},
        "medium": {"root": 220, "waist_rotation_mdeg": 12_000, "major_chain": 240},
        "small": {"root": 60, "waist_rotation_mdeg": 4_000, "major_chain": 80},
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


KINEMATICS_POLICY_SHA256 = _sha256(_POLICY)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_RE = re.compile(
    r"攻击|击|拳|掌|踢|蹬|闪|躲|避|冲|跨|滑|摔|投|抓|扣|挡|转|旋|翻|跃|跳|"
    r"attack|strike|punch|kick|dodge|evade|sprint|lunge|throw|grab|block|turn|spin|flip|jump",
    re.IGNORECASE,
)
_GUARD_RE = re.compile(r"戒备|准备|站立|静止|guard|ready|stand|idle", re.IGNORECASE)
_NO_TARGET_CONTACT_RE = re.compile(
    r"无(?:目标|外部|实际)?接触|不接触|未接触|without (?:target )?contact|no (?:target )?contact|none",
    re.IGNORECASE,
)


def _performers(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = [str(item).strip() for item in value]
    else:
        raw = re.split(r"\s*(?:、|，|,|/|&|\band\b|与)\s*", str(value or ""))
    return list(dict.fromkeys(item for item in raw if item))


def _mechanics_text(beat: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "micro_action",
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
        value = beat.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(str(item) for item in value)
        elif value not in (None, ""):
            values.append(str(value))
    return " ".join(values)


def _side(beat: Mapping[str, Any]) -> str:
    text = f"{beat.get('side') or ''} {' '.join(str(x) for x in beat.get('limbs') or [])}"
    left = bool(re.search(r"左|\bleft\b", text, re.IGNORECASE))
    right = bool(re.search(r"右|\bright\b", text, re.IGNORECASE))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "bilateral"


def _direction_vector(text: str, magnitude: int) -> list[int]:
    if re.search(r"向后|后撤|后退|backward|back step", text, re.IGNORECASE):
        return [0, 0, -magnitude]
    if re.search(r"向左|左移|leftward|to the left", text, re.IGNORECASE):
        return [-magnitude, 0, 0]
    if re.search(r"向右|右移|rightward|to the right", text, re.IGNORECASE):
        return [magnitude, 0, 0]
    if re.search(r"跃|跳|向上|jump|upward", text, re.IGNORECASE):
        return [0, magnitude, magnitude // 3]
    if re.search(r"落|向下|land|downward", text, re.IGNORECASE):
        return [0, -magnitude, magnitude // 4]
    return [0, 0, magnitude]


def _transform(text: str) -> dict[str, Any]:
    folded = text.casefold()
    kind = "none"
    axis = "none"
    direction = "none"
    amount: int | None = None
    amount_range: list[int] | None = None
    airborne = False
    if re.search(r"前翻|forward flip|front flip", folded):
        kind, axis, direction, amount, airborne = "flip", "pitch", "positive", 360_000, True
    elif re.search(r"后翻|back flip|backflip", folded):
        kind, axis, direction, amount, airborne = "flip", "pitch", "negative", 360_000, True
    elif re.search(r"侧翻|cartwheel", folded):
        kind, axis, direction, amount, airborne = "flip", "roll", "positive", 360_000, True
    elif re.search(r"旋转|转体|回旋|\bspin\b", folded):
        kind, axis = "spin", "yaw"
        direction = "negative" if re.search(r"逆时针|counterclockwise", folded) else "positive"
        explicit = re.search(r"(\d+(?:\.\d+)?)\s*(?:度|°)", folded)
        if explicit:
            amount = round(float(explicit.group(1)) * 1000)
        else:
            amount_range = [90_000, 360_000]
    elif re.search(r"转身|pivot|\bturn\b", folded):
        kind, axis = "turn", "yaw"
        direction = "negative" if re.search(r"向左|左转|left", folded) else "positive"
        amount_range = [45_000, 180_000]
    return {
        "kind": kind,
        "axis": axis,
        "direction": direction,
        "amount_mdeg": amount,
        "amount_range_mdeg": amount_range,
        "airborne": airborne,
        "support_release": False,
        "landing_state": "grounded",
    }


def _active_channels(beat: Mapping[str, Any], text: str) -> set[str]:
    limb_text = " ".join(str(item) for item in beat.get("limbs") or [])
    source = f"{limb_text} {text}"
    active = {"root", "waist_torso", "head"}
    markers = {
        "left_arm": r"左(?:臂|手臂|肘|肩)|left (?:arm|elbow|shoulder)",
        "left_hand": r"左(?:手|拳|掌)|left (?:hand|fist|palm)",
        "right_arm": r"右(?:臂|手臂|肘|肩)|right (?:arm|elbow|shoulder)",
        "right_hand": r"右(?:手|拳|掌)|right (?:hand|fist|palm)",
        "left_leg": r"左(?:腿|膝|髋)|left (?:leg|knee|hip)",
        "left_foot": r"左(?:脚|足)|left foot",
        "right_leg": r"右(?:腿|膝|髋)|right (?:leg|knee|hip)",
        "right_foot": r"右(?:脚|足)|right foot",
    }
    for channel, pattern in markers.items():
        if re.search(pattern, source, re.IGNORECASE):
            active.add(channel)
    if not active.intersection({"left_arm", "left_hand", "right_arm", "right_hand"}):
        if re.search(r"拳|掌|手|臂|挥|挡|抓|punch|hand|arm|swing|block|grab", text, re.IGNORECASE):
            side = _side(beat)
            for prefix in (("left",) if side == "left" else ("right",) if side == "right" else ("left", "right")):
                active.update({f"{prefix}_arm", f"{prefix}_hand"})
    if not active.intersection({"left_leg", "left_foot", "right_leg", "right_foot"}):
        if re.search(r"腿|脚|步|踢|蹬|跨|滑|kick|leg|foot|step|lunge", text, re.IGNORECASE):
            side = _side(beat)
            for prefix in (("left",) if side == "left" else ("right",) if side == "right" else ("left", "right")):
                active.update({f"{prefix}_leg", f"{prefix}_foot"})
    return active


def _amplitude(text: str) -> str:
    if _GUARD_RE.search(text) and not _DYNAMIC_RE.search(text):
        return "small"
    if _DYNAMIC_RE.search(text):
        return "large"
    return "medium"


def _has_target_contact(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and not _NO_TARGET_CONTACT_RE.search(text)


def _channel_target(
    channel: str,
    *,
    motion_side: str,
    direction: list[int],
    amplitude: str,
    factor: float,
    active: bool,
    contact_active: bool,
) -> dict[str, Any]:
    scale = {"small": 0.22, "medium": 0.55, "large": 1.0}[amplitude]
    handed = (
        -1
        if channel.startswith("left_")
        else 1
        if channel.startswith("right_")
        else -1
        if motion_side == "left"
        else 1
    )
    role = "active" if active else ("support" if channel.endswith(("leg", "foot")) else "balance")
    activation = round((1000 if active else 260) * factor)
    base = round(520 * scale * factor)
    translation = [0, 0, 0]
    rotation = [0, 0, 0]
    if channel == "root":
        translation = [round(v * factor) for v in direction]
    elif channel == "waist_torso":
        translation = [round(direction[0] * 0.18 * factor), -round(base * 0.12), round(direction[2] * 0.12 * factor)]
        rotation = [round(12_000 * scale * factor), round(38_000 * scale * factor), round(handed * 8_000 * scale * factor)]
    elif channel == "head":
        translation = [round(direction[0] * 0.08 * factor), -round(base * 0.08), round(direction[2] * 0.04 * factor)]
        rotation = [round(-8_000 * scale * factor), round(22_000 * scale * factor), 0]
    elif channel.endswith("_arm"):
        translation = [handed * base, -round(base * 0.42), round(base * 0.88)]
        rotation = [-round(35_000 * scale * factor), handed * round(48_000 * scale * factor), handed * round(18_000 * scale * factor)]
    elif channel.endswith("_hand"):
        translation = [handed * round(base * 1.28), -round(base * 0.58), round(base * 1.24)]
        rotation = [-round(22_000 * scale * factor), handed * round(62_000 * scale * factor), handed * round(28_000 * scale * factor)]
    elif channel.endswith("_leg"):
        translation = [handed * round(base * 0.45), round(base * 0.28), round(base * 0.95)]
        rotation = [round(45_000 * scale * factor), handed * round(12_000 * scale * factor), 0]
    elif channel.endswith("_foot"):
        translation = [handed * round(base * 0.62), round(base * 0.34), round(base * 1.34)]
        rotation = [round(20_000 * scale * factor), handed * round(8_000 * scale * factor), handed * round(12_000 * scale * factor)]
    if not active and channel not in {"root", "waist_torso", "head"}:
        translation = [round(value * 0.18) for value in translation]
        rotation = [round(value * 0.18) for value in rotation]
    return {
        "role": role,
        "translation_milli": translation,
        "rotation_mdeg": rotation,
        "activation_milli": activation,
        "amplitude": amplitude if active or channel in {"root", "waist_torso"} else "small",
        "support": role == "support",
        "contact": contact_active
        and active
        and channel.endswith(("hand", "arm", "foot", "leg")),
    }


def compile_source_kinematics(beat: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one source-indexed mechanics beat into deterministic motion facts."""
    try:
        index = int(beat.get("micro_action_index") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("kinematics requires a valid micro_action_index") from exc
    if index < 1:
        raise ValueError("kinematics requires a valid micro_action_index")
    performers = _performers(beat.get("performer"))
    if not performers:
        raise ValueError("kinematics requires at least one performer")
    text = _mechanics_text(beat)
    amplitude = _amplitude(text)
    magnitude = {"small": 90, "medium": 280, "large": 620}[amplitude]
    direction_text = str(beat.get("direction") or "").strip()
    direction = _direction_vector(direction_text or text, magnitude)
    active_channels = _active_channels(beat, text)
    # Only an explicitly authored move/technique may create a whole-body
    # transform.  Biomechanics such as "waist rotates" are channel mechanics,
    # not permission to invent a spin or flip.
    transform = _transform(
        f"{beat.get('micro_action') or ''} {beat.get('technique') or ''}"
    )
    has_target_contact = _has_target_contact(beat.get("contact"))
    motion_side = _side(beat)
    tracks: list[dict[str, Any]] = []
    for performer in sorted(performers):
        phases: list[dict[str, Any]] = []
        for phase_id, start, end, factor in _PHASE_WINDOWS:
            transform_factor = (
                _TRANSFORM_PROGRESS[phase_id] if transform["kind"] != "none" else 0.0
            )
            amount = transform.get("amount_mdeg")
            amount_range = transform.get("amount_range_mdeg")
            projected_amount = amount
            if projected_amount is None and isinstance(amount_range, list):
                projected_amount = round((int(amount_range[0]) + int(amount_range[1])) / 2)
            direction_sign = -1 if transform["direction"] == "negative" else 1
            yaw = (
                direction_sign * round((projected_amount or 0) * transform_factor)
                if transform["axis"] == "yaw"
                else 0
            )
            airborne = transform["kind"] == "flip" and phase_id in {
                "drive",
                "apex_contact",
                "follow_through",
            }
            channels = {
                channel: _channel_target(
                    channel,
                    motion_side=motion_side,
                    direction=direction,
                    amplitude=amplitude,
                    factor=factor,
                    active=channel in active_channels,
                    contact_active=has_target_contact and phase_id == "apex_contact",
                )
                for channel in CHANNEL_ORDER
            }
            if transform["kind"] == "flip":
                root_arc = {
                    "load": 0,
                    "drive": 380,
                    "apex_contact": 720,
                    "follow_through": 360,
                    "settle": 0,
                }[phase_id]
                channels["root"]["translation_milli"][1] = root_arc
            if airborne:
                for channel in channels.values():
                    if channel["support"]:
                        channel["support"] = False
                        channel["role"] = "balance"
            phases.append(
                {
                    "phase_id": (
                        f"M{index:03d}_apex"
                        if phase_id == "apex_contact" and not has_target_contact
                        else f"M{index:03d}_{phase_id}"
                    ),
                    "source_micro_action_index": index,
                    "start_milli": start,
                    "end_milli": end,
                    "relative_yaw_mdeg": yaw,
                    "camera_relation": "unspecified",
                    "transform": {
                        **transform,
                        "amount_mdeg": (
                            direction_sign * round(amount * transform_factor)
                            if amount is not None
                            else None
                        ),
                        "airborne": airborne,
                        "support_release": airborne,
                        "landing_state": (
                            "pending"
                            if airborne
                            else "landed"
                            if transform["kind"] == "flip" and phase_id == "settle"
                            else "grounded"
                        ),
                    },
                    "channels": channels,
                }
            )
        tracks.append(
            {
                "performer_id": performer,
                "orientation_anchor": "pxx_start_yaw_zero",
                "phases": phases,
            }
        )
    evidence = {
        key: beat.get(key)
        for key in (
            "micro_action_index",
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
    payload: dict[str, Any] = {
        "schema": SOURCE_KINEMATICS_SCHEMA,
        "micro_action_index": index,
        "actor_tracks": tracks,
        "source_evidence_sha256": _sha256(evidence),
        "policy_sha256": KINEMATICS_POLICY_SHA256,
    }
    payload["kinematics_sha256"] = _sha256(payload)
    return validate_source_kinematics(payload)


def _validate_channel(channel: Mapping[str, Any], *, name: str) -> None:
    if set(channel) != _CHANNEL_FIELDS:
        raise ValueError(f"{name} channel fields are invalid")
    if channel.get("role") not in {"active", "support", "balance", "inherit", "stabilize"}:
        raise ValueError(f"{name} channel role is invalid")
    if channel.get("amplitude") not in {"small", "medium", "large"}:
        raise ValueError(f"{name} channel amplitude is invalid")
    for field in ("translation_milli", "rotation_mdeg"):
        values = channel.get(field)
        if not isinstance(values, list) or len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"{name} {field} is invalid")
    activation = channel.get("activation_milli")
    if isinstance(activation, bool) or not isinstance(activation, int) or not 0 <= activation <= 1000:
        raise ValueError(f"{name} activation is invalid")
    if not isinstance(channel.get("support"), bool) or not isinstance(channel.get("contact"), bool):
        raise ValueError(f"{name} support/contact flags are invalid")


def _validate_tracks(tracks: Any) -> None:
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("kinematics actor tracks are missing")
    performer_ids: list[str] = []
    for track in tracks:
        if not isinstance(track, Mapping) or set(track) != {"performer_id", "orientation_anchor", "phases"}:
            raise ValueError("kinematics actor track fields are invalid")
        performer = str(track.get("performer_id") or "").strip()
        if not performer or track.get("orientation_anchor") != "pxx_start_yaw_zero":
            raise ValueError("kinematics performer or orientation anchor is invalid")
        performer_ids.append(performer)
        phases = track.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("kinematics phases are missing")
        previous_end = 0
        for phase in phases:
            required = {
                "phase_id",
                "source_micro_action_index",
                "start_milli",
                "end_milli",
                "relative_yaw_mdeg",
                "camera_relation",
                "transform",
                "channels",
            }
            if not isinstance(phase, Mapping) or set(phase) != required:
                raise ValueError("kinematics phase fields are invalid")
            start, end = phase.get("start_milli"), phase.get("end_milli")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start != previous_end
                or not start < end <= 1000
            ):
                raise ValueError("kinematics phase window is invalid")
            previous_end = end
            transform = phase.get("transform")
            if not isinstance(transform, Mapping) or set(transform) != _TRANSFORM_FIELDS:
                raise ValueError("kinematics transform fields are invalid")
            kind = transform.get("kind")
            axis = transform.get("axis")
            direction = transform.get("direction")
            if (
                kind not in {"none", "turn", "spin", "flip"}
                or axis not in {"none", "yaw", "pitch", "roll"}
                or direction not in {"none", "positive", "negative"}
                or not isinstance(transform.get("airborne"), bool)
                or not isinstance(transform.get("support_release"), bool)
                or transform.get("landing_state")
                not in {"grounded", "pending", "landed"}
            ):
                raise ValueError("kinematics transform enum is invalid")
            amount = transform.get("amount_mdeg")
            amount_range = transform.get("amount_range_mdeg")
            if amount is not None and (
                isinstance(amount, bool) or not isinstance(amount, int)
            ):
                raise ValueError("kinematics transform amount is invalid")
            if amount_range is not None and (
                not isinstance(amount_range, list)
                or len(amount_range) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in amount_range)
                or int(amount_range[0]) <= 0
                or int(amount_range[0]) > int(amount_range[1])
            ):
                raise ValueError("kinematics transform range is invalid")
            if kind == "none" and (
                axis != "none"
                or direction != "none"
                or amount is not None
                or amount_range is not None
                or transform["airborne"]
                or transform["support_release"]
                or transform["landing_state"] != "grounded"
            ):
                raise ValueError("kinematics empty transform is invalid")
            if kind in {"turn", "spin"} and axis != "yaw":
                raise ValueError("kinematics yaw transform axis is invalid")
            if kind == "flip" and axis not in {"pitch", "roll"}:
                raise ValueError("kinematics flip axis is invalid")
            if kind != "none" and (
                axis == "none"
                or direction == "none"
                or (amount is None and amount_range is None)
            ):
                raise ValueError("kinematics transform definition is incomplete")
            if amount is not None and amount_range is not None:
                raise ValueError("kinematics transform amount is ambiguous")
            if transform["airborne"] != transform["support_release"]:
                raise ValueError("kinematics airborne support state is invalid")
            if transform["airborne"] and transform["landing_state"] != "pending":
                raise ValueError("kinematics airborne landing state is invalid")
            if transform["landing_state"] == "landed" and (
                kind != "flip" or transform["airborne"]
            ):
                raise ValueError("kinematics landing state is invalid")
            channels = phase.get("channels")
            if not isinstance(channels, Mapping) or tuple(channels) != CHANNEL_ORDER:
                raise ValueError("kinematics channel order is invalid")
            for name, channel in channels.items():
                if not isinstance(channel, Mapping):
                    raise ValueError(f"{name} channel is invalid")
                _validate_channel(channel, name=name)
            if transform["support_release"] and any(
                bool(channel["support"]) for channel in channels.values()
            ):
                raise ValueError("kinematics support cannot persist while airborne")
        if previous_end != 1000:
            raise ValueError("kinematics phase timeline is incomplete")
    if performer_ids != sorted(set(performer_ids)):
        raise ValueError("kinematics performers must be unique and sorted")


def validate_source_kinematics(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    if set(data) != _SOURCE_FIELDS:
        raise ValueError("source kinematics fields are invalid")
    if data.get("schema") != SOURCE_KINEMATICS_SCHEMA:
        raise ValueError("source kinematics schema is unsupported")
    if data.get("policy_sha256") != KINEMATICS_POLICY_SHA256:
        raise ValueError("source kinematics policy hash is invalid")
    if not _SHA256_RE.fullmatch(str(data.get("source_evidence_sha256") or "")):
        raise ValueError("source kinematics evidence hash is invalid")
    _validate_tracks(data.get("actor_tracks"))
    expected = str(data.pop("kinematics_sha256", ""))
    if not _SHA256_RE.fullmatch(expected) or expected != _sha256(data):
        raise ValueError("source kinematics hash is invalid")
    data["kinematics_sha256"] = expected
    return data


def _scaled_phases(source: Mapping[str, Any], *, offset: int, span: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for phase in source.get("phases") or []:
        item = copy.deepcopy(dict(phase))
        item["start_milli"] = offset + round(int(phase["start_milli"]) * span / 1000)
        item["end_milli"] = offset + round(int(phase["end_milli"]) * span / 1000)
        result.append(item)
    if result:
        result[0]["start_milli"] = offset
        result[-1]["end_milli"] = offset + span
    return result


def _hold_phase(
    phase: Mapping[str, Any], *, phase_id: str, start: int, end: int, role: str
) -> dict[str, Any]:
    item = copy.deepcopy(dict(phase))
    item["phase_id"] = phase_id
    item["start_milli"] = start
    item["end_milli"] = end
    item["channels"] = {
        name: {
            **channel,
            "role": role if channel["role"] not in {"support", "balance"} else channel["role"],
            "activation_milli": min(int(channel["activation_milli"]), 240),
            "contact": False,
        }
        for name, channel in item["channels"].items()
    }
    return item


def _cover_track_timeline(phases: list[dict[str, Any]], *, performer: str) -> list[dict[str, Any]]:
    if not phases:
        return phases
    result = list(phases)
    if int(result[0]["start_milli"]) > 0:
        result.insert(
            0,
            _hold_phase(
                result[0],
                phase_id=f"{performer}_inherit_start",
                start=0,
                end=int(result[0]["start_milli"]),
                role="inherit",
            ),
        )
    if int(result[-1]["end_milli"]) < 1000:
        result.append(
            _hold_phase(
                result[-1],
                phase_id=f"{performer}_stabilize_end",
                start=int(result[-1]["end_milli"]),
                end=1000,
                role="stabilize",
            )
        )
    return result


def _unit_projection(
    *,
    beat_id: str,
    unit: Mapping[str, Any],
    source_by_index: Mapping[int, Mapping[str, Any]],
    body_indexes: list[int],
    inherited_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unit_id = str(unit.get("unit_id") or "").strip()
    indexes = list(body_indexes)
    if not unit_id or not indexes or any(index not in source_by_index for index in indexes):
        raise ValueError(f"{beat_id} generation unit has invalid kinematics lineage")
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{beat_id} generation unit has overlapping kinematics lineage")
    tracks_by_performer: dict[str, list[dict[str, Any]]] = {}
    span_edges = [round(position * 1000 / len(indexes)) for position in range(len(indexes) + 1)]
    for position, index in enumerate(indexes):
        source = source_by_index[index]
        offset = span_edges[position]
        span = span_edges[position + 1] - offset
        for track in source["actor_tracks"]:
            tracks_by_performer.setdefault(str(track["performer_id"]), []).extend(
                _scaled_phases(track, offset=offset, span=span)
            )
    tracks = [
        {
            "performer_id": performer,
            "orientation_anchor": "pxx_start_yaw_zero",
            "phases": _cover_track_timeline(phases, performer=performer),
        }
        for performer, phases in sorted(tracks_by_performer.items())
    ]
    for track in tracks:
        inherited = inherited_states.get(str(track["performer_id"]))
        if not isinstance(inherited, Mapping):
            continue
        phases = track["phases"]
        first = phases[0]
        first_channels = copy.deepcopy(first["channels"])
        inherited_channels = inherited.get("channels")
        if not isinstance(inherited_channels, Mapping):
            raise ValueError(f"{beat_id} inherited kinematics channels are invalid")
        first_yaw = int(first.get("relative_yaw_mdeg") or 0)
        inherited_yaw = int(inherited.get("relative_yaw_mdeg") or 0)
        for phase in phases:
            phase["relative_yaw_mdeg"] = inherited_yaw + (
                int(phase.get("relative_yaw_mdeg") or 0) - first_yaw
            )
            for channel_name, state in phase["channels"].items():
                first_state = first_channels[channel_name]
                inherited_state = inherited_channels[channel_name]
                state["translation_milli"] = [
                    int(inherited_state["translation_milli"][axis])
                    + int(state["translation_milli"][axis])
                    - int(first_state["translation_milli"][axis])
                    for axis in range(3)
                ]
                state["rotation_mdeg"] = [
                    int(inherited_state["rotation_mdeg"][axis])
                    + int(state["rotation_mdeg"][axis])
                    - int(first_state["rotation_mdeg"][axis])
                    for axis in range(3)
                ]
        first["relative_yaw_mdeg"] = inherited_yaw
        # Inherit only durable geometry.  Activation, support and contact are
        # phase-local mechanics of the new action and must not leak across a
        # GAU boundary (for example, a prior hit must not become a sustained
        # contact or guard in the next move).
        for channel_name, state in first["channels"].items():
            inherited_state = inherited_channels[channel_name]
            state["translation_milli"] = copy.deepcopy(
                inherited_state["translation_milli"]
            )
            state["rotation_mdeg"] = copy.deepcopy(
                inherited_state["rotation_mdeg"]
            )
    payload: dict[str, Any] = {
        "schema": KINEMATICS_PROJECTION_SCHEMA,
        "scope": "generation_action_unit",
        "beat_id": beat_id,
        "generation_action_unit_id": unit_id,
        "source_micro_action_indexes": indexes,
        "source_kinematics_sha256s": [source_by_index[index]["kinematics_sha256"] for index in indexes],
        "actor_tracks": tracks,
        "action_units": [],
        "policy_sha256": KINEMATICS_POLICY_SHA256,
    }
    payload["projection_sha256"] = _sha256(payload)
    return validate_generation_kinematics_projection(payload)


def apply_generation_kinematics_projection(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    """Attach final GAU/Pxx projections after generation units are finalized."""
    beat_id = str(record.get("beat_id") or "").strip()
    if not beat_id:
        raise ValueError("generation kinematics requires a Pxx beat_id")
    contract = record.get("body_action_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{beat_id} lacks a body-action contract")
    source_by_index: dict[int, dict[str, Any]] = {}
    for beat in contract.get("beats") or []:
        if not isinstance(beat, Mapping) or not isinstance(beat.get("kinematics"), Mapping):
            continue
        source = validate_source_kinematics(beat["kinematics"])
        index = int(source["micro_action_index"])
        if index in source_by_index:
            raise ValueError(f"{beat_id} has duplicate source kinematics index {index}")
        source_by_index[index] = source
    if not source_by_index:
        # A valid body-action contract can represent an environment-only or
        # deliberately incomplete compatibility beat.  Absence of canonical
        # human mechanics means there is nothing to project; it is not an
        # invitation to invent an actor track or aggregate an empty list.
        record.pop("kinematics_projection", None)
        return None
    units = [dict(unit) for unit in record.get("generation_action_units") or [] if isinstance(unit, Mapping)]
    if not units:
        raise ValueError(f"{beat_id} generation action units are missing")
    covered: set[int] = set()
    unit_projections: list[dict[str, Any]] = []
    inherited_states: dict[str, dict[str, Any]] = {}
    for unit in units:
        raw_indexes = list(unit.get("source_micro_action_indexes") or [])
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indexes):
            raise ValueError(f"{beat_id} generation unit has invalid source lineage")
        body_indexes = [index for index in raw_indexes if index in source_by_index]
        if not body_indexes:
            # Spatial, environmental and other non-body micro-actions remain in
            # the GAU but correctly receive no invented human channels.
            continue
        overlap = covered.intersection(body_indexes)
        if overlap:
            raise ValueError(f"{beat_id} source kinematics indexes overlap: {sorted(overlap)}")
        covered.update(body_indexes)
        projection = _unit_projection(
            beat_id=beat_id,
            unit=unit,
            source_by_index=source_by_index,
            body_indexes=body_indexes,
            inherited_states=inherited_states,
        )
        unit["kinematics_projection"] = projection
        unit["kinematics_projection_sha256"] = projection["projection_sha256"]
        unit_projections.append(projection)
        for track in projection["actor_tracks"]:
            terminal = track["phases"][-1]
            inherited_states[str(track["performer_id"])] = {
                "relative_yaw_mdeg": terminal["relative_yaw_mdeg"],
                "channels": copy.deepcopy(terminal["channels"]),
            }
    if covered != set(source_by_index):
        raise ValueError(
            f"{beat_id} source kinematics coverage mismatch: expected {sorted(source_by_index)}, got {sorted(covered)}"
        )
    aggregate_tracks: dict[str, list[dict[str, Any]]] = {}
    edges = [round(position * 1000 / len(unit_projections)) for position in range(len(unit_projections) + 1)]
    for position, projection in enumerate(unit_projections):
        offset, span = edges[position], edges[position + 1] - edges[position]
        for track in projection["actor_tracks"]:
            aggregate_tracks.setdefault(str(track["performer_id"]), []).extend(
                _scaled_phases(track, offset=offset, span=span)
            )
    aggregate: dict[str, Any] = {
        "schema": KINEMATICS_PROJECTION_SCHEMA,
        "scope": "pxx",
        "beat_id": beat_id,
        "generation_action_unit_id": None,
        "source_micro_action_indexes": sorted(covered),
        "source_kinematics_sha256s": [source_by_index[index]["kinematics_sha256"] for index in sorted(covered)],
        "actor_tracks": [
            {
                "performer_id": performer,
                "orientation_anchor": "pxx_start_yaw_zero",
                "phases": _cover_track_timeline(phases, performer=performer),
            }
            for performer, phases in sorted(aggregate_tracks.items())
        ],
        "action_units": [
            {
                "generation_action_unit_id": projection["generation_action_unit_id"],
                "projection_sha256": projection["projection_sha256"],
                "source_micro_action_indexes": projection["source_micro_action_indexes"],
            }
            for projection in unit_projections
        ],
        "policy_sha256": KINEMATICS_POLICY_SHA256,
    }
    aggregate["projection_sha256"] = _sha256(aggregate)
    aggregate = validate_generation_kinematics_projection(aggregate)
    record["generation_action_units"] = units
    record["kinematics_projection"] = aggregate
    return aggregate


def validate_generation_kinematics_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    if set(data) != _PROJECTION_FIELDS:
        raise ValueError("generation kinematics projection fields are invalid")
    if data.get("schema") != KINEMATICS_PROJECTION_SCHEMA:
        raise ValueError("generation kinematics projection schema is unsupported")
    if data.get("scope") not in {"generation_action_unit", "pxx"}:
        raise ValueError("generation kinematics projection scope is invalid")
    if data.get("policy_sha256") != KINEMATICS_POLICY_SHA256:
        raise ValueError("generation kinematics projection policy hash is invalid")
    indexes = data.get("source_micro_action_indexes")
    if not isinstance(indexes, list) or not indexes or indexes != sorted(set(indexes)):
        raise ValueError("generation kinematics source indexes are invalid")
    hashes = data.get("source_kinematics_sha256s")
    if not isinstance(hashes, list) or len(hashes) != len(indexes) or any(
        not _SHA256_RE.fullmatch(str(value)) for value in hashes
    ):
        raise ValueError("generation kinematics source hashes are invalid")
    _validate_tracks(data.get("actor_tracks"))
    action_units = data.get("action_units")
    if not isinstance(action_units, list):
        raise ValueError("generation kinematics action units are invalid")
    if data["scope"] == "generation_action_unit":
        if data.get("generation_action_unit_id") in (None, "") or action_units:
            raise ValueError("generation-unit kinematics scope is invalid")
    else:
        if data.get("generation_action_unit_id") is not None or not action_units:
            raise ValueError("Pxx kinematics scope is invalid")
        unit_indexes: list[int] = []
        for unit in action_units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "generation_action_unit_id",
                "projection_sha256",
                "source_micro_action_indexes",
            }:
                raise ValueError("generation kinematics action-unit lineage is invalid")
            unit_id = str(unit.get("generation_action_unit_id") or "").strip()
            unit_hash = str(unit.get("projection_sha256") or "")
            child_indexes = unit.get("source_micro_action_indexes")
            if (
                not unit_id
                or not _SHA256_RE.fullmatch(unit_hash)
                or not isinstance(child_indexes, list)
                or not child_indexes
                or child_indexes != sorted(set(child_indexes))
            ):
                raise ValueError("generation kinematics action-unit lineage is invalid")
            unit_indexes.extend(child_indexes)
        if unit_indexes != indexes:
            raise ValueError("Pxx kinematics action-unit coverage is invalid")
    expected = str(data.pop("projection_sha256", ""))
    if not _SHA256_RE.fullmatch(expected) or expected != _sha256(data):
        raise ValueError("generation kinematics projection hash is invalid")
    data["projection_sha256"] = expected
    return data


def _relation(relative_yaw_mdeg: int, camera_yaw_mdeg: int | None) -> str:
    if camera_yaw_mdeg is None:
        return "unspecified"
    delta = (relative_yaw_mdeg - camera_yaw_mdeg + 180_000) % 360_000 - 180_000
    absolute = abs(delta)
    if absolute <= 22_500:
        return "front"
    if absolute >= 157_500:
        return "back"
    if delta < -67_500:
        return "left_profile"
    if delta > 67_500:
        return "right_profile"
    return "left_three_quarter" if delta < 0 else "right_three_quarter"


def _sample_track(track: Mapping[str, Any], progress_milli: int, camera_yaw_mdeg: int | None) -> dict[str, Any]:
    phases = list(track.get("phases") or [])
    selected = next(
        (phase for phase in phases if int(phase["start_milli"]) <= progress_milli <= int(phase["end_milli"])),
        phases[-1],
    )
    yaw = int(selected.get("relative_yaw_mdeg") or 0)
    return {
        "performer_id": str(track["performer_id"]),
        "phase_id": str(selected["phase_id"]),
        "source_micro_action_index": int(selected["source_micro_action_index"]),
        "relative_yaw_mdeg": yaw,
        "camera_relation": _relation(yaw, camera_yaw_mdeg),
        "transform": copy.deepcopy(selected["transform"]),
        "channels": copy.deepcopy(selected["channels"]),
    }


def sample_projection(
    projection: Mapping[str, Any], progress: float, *, camera_yaw_mdeg: int | None = None
) -> dict[str, Any]:
    """Sample a verified projection without interpreting action prose."""
    data = validate_generation_kinematics_projection(projection)
    if not math.isfinite(progress):
        raise ValueError("kinematics sample progress is invalid")
    progress_milli = round(max(0.0, min(1.0, progress)) * 1000)
    return {
        "schema": "honcut.canonical-action-kinematics-sample.v1",
        "projection_sha256": data["projection_sha256"],
        "progress_milli": progress_milli,
        "actor_tracks": [
            _sample_track(track, progress_milli, camera_yaw_mdeg) for track in data["actor_tracks"]
        ],
    }


def sample_generation_units(
    units: Sequence[Mapping[str, Any]],
    progress: float,
    *,
    camera_yaw_mdeg: int | None = None,
) -> dict[str, Any]:
    """Sample an ordered GAU group while retaining every performer track."""
    projections: list[dict[str, Any]] = []
    for unit in units:
        projection = unit.get("kinematics_projection")
        if not isinstance(projection, Mapping):
            continue
        projections.append(validate_generation_kinematics_projection(projection))
    if not projections:
        raise ValueError("canonical kinematics projection group is empty")
    normalized = max(0.0, min(1.0, float(progress)))
    scaled = min(len(projections) - 1e-9, normalized * len(projections))
    selected_index = min(len(projections) - 1, int(scaled))
    local_progress = scaled - selected_index
    sampled = sample_projection(
        projections[selected_index],
        local_progress,
        camera_yaw_mdeg=camera_yaw_mdeg,
    )
    sampled["generation_action_unit_id"] = projections[selected_index][
        "generation_action_unit_id"
    ]
    sampled["group_projection_sha256s"] = [
        projection["projection_sha256"] for projection in projections
    ]
    sampled["group_kinematics_sha256"] = _sha256(
        sampled["group_projection_sha256s"]
    )
    return sampled
