"""Shared deterministic contracts for every paid video-generation route."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from utils.camera_motion_contracts import camera_motion_execution_prompt
from utils.privacy_visual_policy import NO_REAL_PERSON_POLICY

VIDEO_GENERATION_CONTRACT_MARKER = "[honcut-video-generation-contract-v1]"

DUPLICATE_IDENTITY_NEGATIVE = (
    "cloned named character, duplicated identity, twin copy of a reference character, "
    "mirrored character duplicate, repeated canonical costume or identity marker on background extras"
)


def _character_list(characters: Any) -> list[dict[str, Any]]:
    if isinstance(characters, Mapping):
        raw = characters.get("characters", [])
    else:
        raw = characters or []
    return [item for item in raw if isinstance(item, dict)]


def has_synthetic_identity_policy(characters: Any) -> bool:
    """Return whether persisted character data declares the synthetic policy."""
    if (
        isinstance(characters, Mapping)
        and characters.get("visual_identity_policy") == NO_REAL_PERSON_POLICY
    ):
        return True
    return any(
        character.get("visual_identity_policy") == NO_REAL_PERSON_POLICY
        for character in _character_list(characters)
    )


def select_identity_bound_characters(
    shot_meta: Mapping[str, Any],
    characters: Any,
) -> list[dict[str, Any]]:
    """Resolve the named foreground cast without inferring arbitrary prompt nouns."""
    raw_requested = shot_meta.get("who") or shot_meta.get("characters") or []
    if not isinstance(raw_requested, list):
        raw_requested = [raw_requested] if raw_requested else []
    requested = {str(value).casefold() for value in raw_requested if value}
    if not requested and "who" not in shot_meta and "characters" not in shot_meta:
        requested.update(
            str(asset)[5:].split(":", 1)[0].casefold()
            for asset in shot_meta.get("associate_assets", []) or []
            if isinstance(asset, str) and asset.startswith("char:")
        )
    if not requested:
        return []

    selected = []
    for character in _character_list(characters):
        keys = {
            str(character.get("id") or "").casefold(),
            str(character.get("name") or "").casefold(),
            *(str(alias).casefold() for alias in character.get("aliases", []) if alias),
        }
        if not requested.isdisjoint(keys):
            selected.append(character)
    return selected


def _declared_cast_names(
    shot_meta: Mapping[str, Any],
    selected: list[dict[str, Any]],
) -> list[str]:
    raw = shot_meta.get("who") or shot_meta.get("characters") or []
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    names = [str(value).strip() for value in raw if str(value).strip()]
    if not names:
        names = [
            str(character.get("name") or character.get("id") or "").strip()
            for character in selected
        ]
    return list(dict.fromkeys(name for name in names if name))


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        flattened = []
        for key, item in value.items():
            for text in _flatten_strings(item):
                flattened.append(f"{key}: {text}")
        return flattened
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_strings(item))
        return flattened
    return []


def _shot_action_text(shot_meta: Mapping[str, Any]) -> str:
    values = [
        shot_meta.get("visual"),
        shot_meta.get("what"),
        shot_meta.get("action"),
        shot_meta.get("action_description"),
        shot_meta.get("generation_actions"),
        shot_meta.get("prompt"),
    ]
    return " ".join(_flatten_strings(values)).casefold()


def _interaction_prop_is_active(owner: str, prop: str, action_text: str) -> bool:
    prop_lower = prop.casefold()
    if "iphone" in prop_lower:
        return any(token in action_text for token in ("iphone", "手机", "拍摄", "跟拍"))
    latin_tokens = re.findall(r"[a-z][a-z0-9+_.-]{2,}", prop_lower)
    if any(token in action_text for token in latin_tokens):
        return True
    owner_lower = owner.casefold()
    return bool(
        owner_lower
        and owner_lower in action_text
        and any(token in action_text for token in ("手持", "拿", "使用", "操作", "拍摄", "跟拍"))
    )


def _prop_anchor(owner: str, prop: str) -> str:
    lower = prop.casefold()
    if "iphone" in lower:
        return (
            f"{owner}: exactly one silver iPhone smartphone / 银色 iPhone 手机，"
            "薄型银色圆角矩形机身，背面清晰可辨的手机三摄镜头模组，手掌大小；"
            "它是画面内唯一拍摄道具。禁止替换成单反相机、微单相机、摄像机、"
            "带可更换长焦镜头的机身、相机握把或任何泛化的拍摄设备"
        )
    if "手机" in prop or "smartphone" in lower:
        return (
            f"{owner}: exactly one smartphone / 智能手机，薄型圆角矩形手持机身和手机镜头模组；"
            "禁止替换成单反、微单、摄像机或带可更换镜头的设备"
        )
    return f"{owner}: preserve this exact interaction prop without substitution — {prop}"


def _interaction_prop_contract(
    shot_meta: Mapping[str, Any],
    selected: list[dict[str, Any]],
) -> str:
    action_text = _shot_action_text(shot_meta)
    active: list[tuple[str, str]] = [
        ("shot", prop) for prop in _flatten_strings(shot_meta.get("interaction_props"))
    ]
    for character in selected:
        owner = str(character.get("name") or character.get("id") or "character")
        appearance = character.get("appearance") or {}
        if not isinstance(appearance, Mapping):
            continue
        for prop in _flatten_strings(appearance.get("interaction_props")):
            if _interaction_prop_is_active(owner, prop, action_text):
                active.append((owner, prop))
    anchors = list(dict.fromkeys(_prop_anchor(owner, prop) for owner, prop in active))
    if not anchors:
        return ""
    return "\n".join(("[interaction-prop-fidelity]", *anchors))


def _reshoot_feedback_contract(shot_meta: Mapping[str, Any]) -> str:
    feedback = shot_meta.get("phase8_reshoot")
    if not isinstance(feedback, Mapping):
        return ""
    issues = _flatten_strings(feedback.get("issues"))
    if not issues:
        return ""
    bounded = [issue[:1200] for issue in issues[:6]]
    return "\n".join(
        (
            "[phase8-reshoot-correction]",
            f"attempt={feedback.get('round') or 1}",
            "The previous paid take was rejected for the following observed defects:",
            *(f"- {issue}" for issue in bounded),
            (
                "Correct every listed defect in this take while preserving the authored identity, "
                "action order, interaction prop, camera path, scene, and lighting contracts. "
                "Do not repeat or cosmetically hide the failed behavior."
            ),
        )
    )


def render_video_generation_contract(
    shot_meta: Mapping[str, Any],
    characters: Any,
) -> str:
    """Render cast, prop, camera, and reshoot contracts as one untruncated block."""
    selected = select_identity_bound_characters(shot_meta, characters)
    cast_names = _declared_cast_names(shot_meta, selected)
    sections = [VIDEO_GENERATION_CONTRACT_MARKER]
    if cast_names:
        sections.append(
            "\n".join(
                (
                    "[identity-cardinality]",
                    f"identity_bound_cast_count={len(cast_names)}",
                    "required_exactly_once=" + " | ".join(cast_names),
                    (
                        "Each named identity-bound character appears as exactly one physical instance. "
                        "Never clone, mirror, duplicate, split, or reuse that identity, costume, helmet, "
                        "or signature marker on a background extra. Authored background crowds may exist, "
                        "but every extra must remain visually distinct from the named cast."
                    ),
                )
            )
        )
    prop_contract = _interaction_prop_contract(shot_meta, selected)
    if prop_contract:
        sections.append(prop_contract)
    sections.append(camera_motion_execution_prompt(shot_meta))
    feedback_contract = _reshoot_feedback_contract(shot_meta)
    if feedback_contract:
        sections.append(feedback_contract)
    return "\n".join(sections)


def ensure_video_generation_contract(
    prompt: object,
    shot_meta: Mapping[str, Any],
    characters: Any,
) -> str:
    """Append the shared contract exactly once outside any prompt-length limiter."""
    text = str(prompt or "").strip()
    if VIDEO_GENERATION_CONTRACT_MARKER in text:
        return text
    block = render_video_generation_contract(shot_meta, characters)
    return f"{text}\n{block}".strip()
