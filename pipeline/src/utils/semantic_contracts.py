"""Canonical semantic IDs joining text understanding to downstream vision QA."""

from __future__ import annotations

import hashlib
from typing import Any

from utils.character_identity import (
    normalize_character_reference,
    resolve_character_id,
)

SEMANTIC_UNDERSTANDING_SCHEMA = "honcut.semantic-understanding.v1"


def source_identity_ref(mention: Any) -> str:
    """Return a stable opaque ID without exposing a source label as identity."""

    normalized = normalize_character_reference(mention)
    if not normalized:
        raise ValueError("participant mention must not be empty")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"SRCCHAR_{digest}"


def _character_id_for_mention(
    mention: str,
    characters: list[dict[str, Any]],
) -> str | None:
    return resolve_character_id(mention, characters)


def bind_story_semantics(
    events: list[dict[str, Any]],
    characters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind every text mention to one stable character ID or fail closed.

    Human-readable names remain presentation fields.  Downstream code receives
    ``character_ids`` and opaque ``participant_refs`` so a later model synonym
    cannot silently change ownership of an action or visual observation.
    """

    character_by_id = {
        str(character.get("id") or "").strip(): character
        for character in characters
        if str(character.get("id") or "").strip()
    }
    if len(character_by_id) != len(characters):
        raise ValueError("canonical characters require unique non-empty ids")

    refs_by_character: dict[str, list[str]] = {
        character_id: [] for character_id in character_by_id
    }
    event_records: list[dict[str, Any]] = []
    for position, event in enumerate(events, 1):
        event_id = int(event.get("event_id") or event.get("id") or position)
        raw_who = event.get("who") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        mentions = [str(value).strip() for value in raw_who if str(value).strip()]
        participant_refs: list[dict[str, str]] = []
        character_ids: list[str] = []
        for mention in mentions:
            character_id = _character_id_for_mention(mention, characters)
            if character_id is None:
                raise ValueError(
                    f"unbound participant {mention!r} in event "
                    f"{event_id}"
                )
            ref_id = source_identity_ref(mention)
            participant_refs.append({
                "ref_id": ref_id,
                "mention": mention,
                "character_id": character_id,
            })
            if character_id not in character_ids:
                character_ids.append(character_id)
            if ref_id not in refs_by_character[character_id]:
                refs_by_character[character_id].append(ref_id)
        event["participant_refs"] = participant_refs
        event["character_ids"] = character_ids
        event_records.append({
            "event_id": event_id,
            "action_unit_id": str(event.get("action_unit_id") or ""),
            "participant_ref_ids": [item["ref_id"] for item in participant_refs],
            "character_ids": character_ids,
        })

    entity_records: list[dict[str, Any]] = []
    for character_id, character in character_by_id.items():
        ref_ids = refs_by_character[character_id]
        character["source_identity_ref_ids"] = ref_ids
        entity_records.append({
            "character_id": character_id,
            "name": str(character.get("name") or ""),
            "source_identity_ref_ids": ref_ids,
        })

    return {
        "schema": SEMANTIC_UNDERSTANDING_SCHEMA,
        "entities": entity_records,
        "events": event_records,
    }


__all__ = [
    "SEMANTIC_UNDERSTANDING_SCHEMA",
    "bind_story_semantics",
    "source_identity_ref",
]
