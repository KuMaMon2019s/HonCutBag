"""Shared asset-boundary contract for neutral character references.

Canonical reference images describe identity, not a moment from the story.  The
distinction is relational rather than object-specific: an item secured to the
body can be a stable identity asset, while an item that requires a hand to
hold, carry, use, or operate is an interaction prop.  Keeping that rule here
prevents the discovery, generation, and QA prompts from drifting apart.
"""

from __future__ import annotations

import re
from typing import Any

CHARACTER_REFERENCE_ASSET_CONTRACT_VERSION = 2

STATIC_REFERENCE_ASSET_POLICY = (
    "Canonical neutral references contain only the character's body, garments, "
    "footwear, and accessories secured to the body without hand support. Any item "
    "whose authored relation requires a hand to hold, grip, carry, raise, wield, "
    "use, or operate it is an interaction prop, not static identity. Omit every "
    "interaction prop from every reference view; keep the hands empty, open, and "
    "relaxed. Preserve body-worn or fastened wardrobe and accessories."
)

STATIC_REFERENCE_QA_POLICY = (
    f"{STATIC_REFERENCE_ASSET_POLICY} Never penalize the absence of an interaction "
    "prop, even if residual story text assigns one to the character. The empty-hands "
    "rule overrides residual held-object wording; it does not permit removal of "
    "body-worn or fastened identity assets."
)

IDENTITY_DETAIL_ASSET_POLICY = (
    "A supplemental identity-detail board is derived from the approved neutral four-view pack. "
    "It may show close crops of body-worn identity markers and isolated turnarounds of declared "
    "signature props. Handheld equipment must be displayed by itself, never held or operated by "
    "the character, so it cannot contaminate neutral pose references. Preserve exact geometry, "
    "color, material, markings, handedness notes, and attachment points; do not invent story props, "
    "a location, an action pose, or another character."
)

# Split only at list-like boundaries commonly used in generated wardrobe fields.
# The matcher intentionally classifies the relation (held/operated), never an
# object noun, role name, character id, or story phrase.
_ASSET_CLAUSE_SEPARATOR = re.compile(r"\s*(?:\+|,|，|、|;|；|\|)\s*")
_HAND_INTERACTION_RELATION = re.compile(
    r"(?:"
    r"手持|握持|持握|手握|拿着|握着|攥着|拎着|提着|抱着|托着|举着|扛着|"
    r"挥舞着?|操作着?|使用着?|"
    r"hand[ -]?held|"
    r"\b(?:hold(?:s|ing)?|held|grip(?:s|ping|ped)?|wield(?:s|ing|ed)?|"
    r"clutch(?:es|ing|ed)?|brandish(?:es|ing|ed)?|operate(?:s|d|ing)?|"
    r"us(?:e|es|ed|ing)|rais(?:e|es|ed|ing))\b|"
    r"\b(?:carry|carries|carried|carrying)\b[^,;|]*\b(?:by|in)\s+(?:the\s+)?hand\b|"
    r"\bin\s+(?:his|her|their|the)\s+(?:left\s+|right\s+)?hand\b"
    r")",
    re.IGNORECASE,
)


def partition_reference_asset_clauses(value: object) -> tuple[str, list[str]]:
    """Separate static wardrobe clauses from hand-interaction clauses.

    The returned interaction clauses retain their authored wording so callers
    can preserve them as story metadata instead of silently discarding them.
    """
    text = str(value or "").strip()
    if not text:
        return "", []
    clauses = [part.strip() for part in _ASSET_CLAUSE_SEPARATOR.split(text)]
    clauses = [part for part in clauses if part]
    interaction = [part for part in clauses if _HAND_INTERACTION_RELATION.search(part)]
    static = [part for part in clauses if part not in interaction]
    return " + ".join(static), interaction


def static_reference_identity_text(value: object) -> str:
    """Return identity text with hand-interaction clauses removed."""
    static, _interaction = partition_reference_asset_clauses(value)
    return static


def normalize_identity_props(value: object) -> list[dict[str, Any]]:
    """Normalize flexible authored identity-prop values into an auditable list."""
    raw_items = value if isinstance(value, list) else ([value] if value else [])
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or "").strip()
            description = str(item.get("description") or name).strip()
            attachment_mode = str(
                item.get("attachment_mode") or item.get("attachment") or "isolated_handheld"
            ).strip().lower()
            persistence = str(item.get("persistence") or "role_active").strip().lower()
            reference_required = item.get("reference_required", True) is not False
            identity_id = str(item.get("id") or f"identity_prop_{index:02d}").strip()
        else:
            name = str(item or "").strip()
            description = name
            attachment_mode = "isolated_handheld"
            persistence = "role_active"
            reference_required = True
            identity_id = f"identity_prop_{index:02d}"
        if not name or not description:
            continue
        if attachment_mode not in {"body_attached", "isolated_handheld"}:
            attachment_mode = "isolated_handheld"
        if persistence not in {"always", "role_active"}:
            persistence = "role_active"
        normalized.append({
            "id": identity_id,
            "name": name,
            "description": description,
            "attachment_mode": attachment_mode,
            "persistence": persistence,
            "reference_required": reference_required,
        })
    return normalized


def character_identity_detail_items(character: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only declared props that require a visual consistency reference."""
    appearance = character.get("appearance")
    if not isinstance(appearance, dict):
        appearance = character
    return [
        item
        for item in normalize_identity_props(
            appearance.get("identity_props") or appearance.get("signature_props")
        )
        if item["reference_required"]
    ]


def identity_detail_prompt_items(items: list[dict[str, Any]]) -> str:
    """Render exact item contracts for Phase 3 generation and QA."""
    return "；".join(
        f"{item['id']}={item['name']}：{item['description']}"
        f"（展示方式={item['attachment_mode']}，持续性={item['persistence']}）"
        for item in items
    )


def normalize_character_reference_assets(character: dict[str, Any]) -> None:
    """Normalize one character at the Phase 1 contract boundary in place.

    Interaction clauses are moved to ``appearance.interaction_props`` so the
    story still owns them while Phase 3 receives a contradiction-free static
    identity. Re-running the normalizer is idempotent.
    """
    appearance = character.get("appearance")
    if not isinstance(appearance, dict):
        return

    static_clothing, interaction = partition_reference_asset_clauses(
        appearance.get("clothing")
    )
    if static_clothing:
        appearance["clothing"] = static_clothing
    elif interaction:
        appearance.pop("clothing", None)

    static_summary, summary_interaction = partition_reference_asset_clauses(
        appearance.get("summary")
    )
    if static_summary:
        appearance["summary"] = static_summary
    elif summary_interaction:
        fallback_summary = " + ".join(
            str(appearance.get(key) or "").strip()
            for key in ("hair", "face", "clothing", "build")
            if str(appearance.get(key) or "").strip()
        )
        appearance["summary"] = fallback_summary or "static character identity"

    existing = appearance.get("interaction_props")
    if isinstance(existing, list):
        preserved = [str(item).strip() for item in existing if str(item).strip()]
    elif str(existing or "").strip():
        preserved = [str(existing).strip()]
    else:
        preserved = []
    combined = list(dict.fromkeys([*preserved, *interaction, *summary_interaction]))
    if combined:
        appearance["interaction_props"] = combined

    identity_props = normalize_identity_props(
        appearance.get("identity_props") or appearance.get("signature_props")
    )
    if identity_props:
        appearance["identity_props"] = identity_props
    appearance.pop("signature_props", None)

    appearance["reference_asset_contract"] = {
        "version": CHARACTER_REFERENCE_ASSET_CONTRACT_VERSION,
        "static_identity": "body_and_body_supported_assets",
        "interaction_props": "excluded_from_neutral_references",
        "identity_detail_assets": "derived_board_body_attached_or_isolated",
        "hands": "empty_open_relaxed",
    }
