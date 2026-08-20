"""Shared deterministic contracts for every paid video-generation route."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from utils.camera_motion_contracts import camera_motion_execution_prompt
from utils.character_reference_contracts import normalize_identity_props
from utils.privacy_visual_policy import NO_REAL_PERSON_POLICY

VIDEO_GENERATION_CONTRACT_MARKER = "[honcut-video-generation-contract-v2]"

DUPLICATE_IDENTITY_NEGATIVE = (
    "cloned named character, duplicated identity, twin copy of a reference character, "
    "mirrored character duplicate, repeated canonical costume or identity marker on background extras"
)

SPATIAL_IDENTITY_NEGATIVE = (
    "canonical identity color drift, recolored helmet or costume, swapped named-character styling, "
    "swapped foreground roles, unintended side-by-side blocking, follower overtaking the lead, "
    "reversed authored depth order, unprompted stop, turn, or travel-direction reversal"
)

_CANONICAL_APPEARANCE_FIELDS = (
    ("hair", "hair/head"),
    ("face", "face/helmet"),
    ("clothing", "clothing"),
    ("build", "body build"),
    ("distinguishing", "signature marker"),
)

# These are base-material color targets, not lighting instructions. The hex
# hint keeps providers from treating neighboring dark neutrals as equivalent
# when two identity-bound characters share a helmet or visor silhouette.
_CANONICAL_COLOR_HEX = (
    ("藏蓝", "navy", "#1F2A44"),
    ("深红", "dark red", "#7A1F2B"),
    ("深灰", "dark gray", "#3A3F46"),
    ("冷白", "cool white", "#E8F4FF"),
    ("银色", "silver", "#C0C0C0"),
    ("琥珀", "amber", "#FFBF00"),
    ("冷蓝", "cool blue", "#4A6FA5"),
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


def _bounded_contract_text(value: Any, limit: int = 1200) -> str:
    text = " | ".join(_flatten_strings(value))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _character_name(character: Mapping[str, Any]) -> str:
    return str(character.get("name") or character.get("id") or "character").strip()


def _appearance_values(character: Mapping[str, Any]) -> dict[str, str]:
    appearance = character.get("appearance") or {}
    if not isinstance(appearance, Mapping):
        return {}
    values = {}
    for field, _label in _CANONICAL_APPEARANCE_FIELDS:
        value = _bounded_contract_text(appearance.get(field), 600)
        if value:
            values[field] = value
    return values


def _canonical_color_targets(text: str) -> list[str]:
    return [
        f"{token} / {english} = {hex_value}"
        for token, english, hex_value in _CANONICAL_COLOR_HEX
        if token in text
    ]


def _canonical_appearance_contract(selected: list[dict[str, Any]]) -> str:
    lines = []
    for character in selected:
        values = _appearance_values(character)
        if not values:
            continue
        traits = "; ".join(
            f"{label}={values[field]}"
            for field, label in _CANONICAL_APPEARANCE_FIELDS
            if field in values
        )
        colors = _canonical_color_targets(" ".join(values.values()))
        color_suffix = f"; canonical base colors: {' | '.join(colors)}" if colors else ""
        lines.append(
            f"{_character_name(character)} [reference_id={character.get('id') or 'unassigned'}]: "
            f"{traits}{color_suffix}"
        )
    if not lines:
        return ""
    return "\n".join(
        (
            "[canonical-identity-appearance-lock]",
            *lines,
            (
                "Every listed trait is immutable in every frame. Hex values describe the canonical "
                "base material/albedo under neutral light: scene lighting may change brightness, but "
                "must not change the hue family, recolor a helmet/visor/costume, or transfer one "
                "character's styling to another."
            ),
        )
    )


def _lookalike_disambiguation_contract(selected: list[dict[str, Any]]) -> str:
    shared: list[str] = []
    for field, label in _CANONICAL_APPEARANCE_FIELDS:
        owners_by_value: dict[str, list[str]] = {}
        original_by_value: dict[str, str] = {}
        for character in selected:
            value = _appearance_values(character).get(field)
            if not value:
                continue
            normalized = re.sub(r"\s+", "", value).casefold()
            owners_by_value.setdefault(normalized, []).append(_character_name(character))
            original_by_value[normalized] = value
        for normalized, owners in owners_by_value.items():
            if len(owners) < 2:
                continue
            shared.append(
                f"shared {label}: {' | '.join(owners)} -> {original_by_value[normalized]}"
            )
    if not shared:
        return ""
    return "\n".join(
        (
            "[lookalike-cast-disambiguation]",
            *shared,
            (
                "Shared helmet, visor, hair, build, or costume traits are not identity keys. Bind "
                "each named role to its own reference subject for the whole take; use that role's "
                "non-shared canonical traits (especially clothing), assigned action, and authored "
                "spatial position as the disambiguators. Never swap reference subjects, costumes, "
                "colors, actions, or spatial roles between lookalike characters, and never merge them "
                "into one identity."
            ),
        )
    )


def _spatial_motion_contract(shot_meta: Mapping[str, Any]) -> str:
    explicit_layout = _bounded_contract_text(
        {
            key: shot_meta.get(key)
            for key in (
                "spatial_relations",
                "spatial_layout",
                "blocking",
                "position_constraints",
                "motion_constraints",
            )
            if shot_meta.get(key)
        }
    )
    visual = _bounded_contract_text(shot_meta.get("visual"))
    start_state = _bounded_contract_text(shot_meta.get("start_state"))
    ordered_motion = _bounded_contract_text(
        shot_meta.get("generation_actions")
        or shot_meta.get("action_description")
        or shot_meta.get("what")
        or shot_meta.get("action")
    )
    end_state = _bounded_contract_text(shot_meta.get("end_state"))
    evidence = [
        ("explicit_layout", explicit_layout),
        ("authored_visual_blocking", visual),
        ("start_state", start_state),
        ("ordered_motion", ordered_motion),
        ("required_end_state", end_state),
    ]
    evidence_lines = [f"{label}={value}" for label, value in evidence if value]
    if not evidence_lines:
        return ""
    return "\n".join(
        (
            "[spatial-motion-lock]",
            *evidence_lines,
            (
                "Treat authored front/behind, left/right, depth order, distance, facing, and screen "
                "direction as persistent state variables, not suggestions. Preserve any exact authored "
                "distance throughout the action unless the contract explicitly changes it."
            ),
            (
                "If one role follows or stays behind another, the follower must remain behind along the "
                "authored travel/spatial axis and keep the authored gap: never become side-by-side, move "
                "ahead, overtake, cross through, or exchange depth order with the lead."
            ),
            (
                "If motion is authored as continuous forward travel, maintain that travel from the first "
                "relevant frame through the required end state. Do not pause, pivot toward another role, "
                "reverse, teleport, or let camera motion substitute for the subject's movement unless "
                "that change is explicitly authored."
            ),
        )
    )


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


def _identity_prop_reference_contract(
    shot_meta: Mapping[str, Any],
    selected: list[dict[str, Any]],
) -> str:
    """Bind recurring identity equipment to its supplemental detail board."""
    action_text = _shot_action_text(shot_meta)
    anchors: list[str] = []
    for character in selected:
        owner = str(character.get("name") or character.get("id") or "character")
        appearance = character.get("appearance") or {}
        if not isinstance(appearance, Mapping):
            continue
        for item in normalize_identity_props(appearance.get("identity_props")):
            exact = (
                f"{owner}/{item['id']} {item['name']}: {item['description']}; "
                f"attachment_mode={item['attachment_mode']}; persistence={item['persistence']}"
            )
            active = _interaction_prop_is_active(
                owner,
                f"{item['name']} {item['description']}",
                action_text,
            )
            if item["attachment_mode"] == "body_attached" or item["persistence"] == "always":
                anchors.append(
                    exact
                    + "; must remain visible at the authored attachment point with identical geometry, "
                    "colors, material and markings in every relevant frame"
                )
            elif active:
                anchors.append(
                    exact
                    + "; this shot activates the item, so show exactly one matching instance owned by "
                    f"{owner}; do not substitute, recolor, resize, duplicate, or transfer it"
                )
            else:
                anchors.append(
                    exact
                    + "; do not invent it into the action, but if it is visible it must match the "
                    "identity-detail reference exactly and remain owned by the declared character"
                )
    if not anchors:
        return ""
    return "\n".join(("[identity-prop-reference-lock]", *dict.fromkeys(anchors)))


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
    appearance_contract = _canonical_appearance_contract(selected)
    if appearance_contract:
        sections.append(appearance_contract)
    lookalike_contract = _lookalike_disambiguation_contract(selected)
    if lookalike_contract:
        sections.append(lookalike_contract)
    spatial_contract = _spatial_motion_contract(shot_meta)
    if spatial_contract:
        sections.append(spatial_contract)
    prop_contract = _interaction_prop_contract(shot_meta, selected)
    if prop_contract:
        sections.append(prop_contract)
    identity_prop_contract = _identity_prop_reference_contract(shot_meta, selected)
    if identity_prop_contract:
        sections.append(identity_prop_contract)
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
