"""Shared, conservative character identity resolution.

Character references cross several pipeline boundaries: screenplay ``who``
labels, canonical names, asset ids, and aliases.  Keep their matching rules in
one place so quality review and adaptation cannot silently disagree.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

GENERIC_CHARACTER_REFERENCES = {
    "他",
    "她",
    "它",
    "其",
    "主角",
    "主人公",
    "人物",
    "角色",
}


def normalize_character_reference(value: Any) -> str:
    """Normalize presentation differences without erasing word boundaries."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[\s·•_\-]+", "", text)


def _latin_tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(re.findall(r"[a-z0-9]+", text))


def _contains_reference(mention: Any, reference: Any) -> bool:
    """Return whether a qualified mention contains one complete reference.

    Latin references are matched as complete token sequences, preventing an
    alias such as ``ann`` from matching ``joanne``.  CJK references have no
    whitespace word boundary, so a two-character minimum and substring match
    are used there.
    """
    mention_text = str(mention or "")
    reference_text = str(reference or "")
    reference_key = normalize_character_reference(reference_text)
    if len(reference_key) < 2 or reference_text.strip() in GENERIC_CHARACTER_REFERENCES:
        return False

    if re.search(r"[\u3400-\u9fff]", reference_text):
        return reference_key in normalize_character_reference(mention_text)

    reference_tokens = _latin_tokens(reference_text)
    mention_tokens = _latin_tokens(mention_text)
    if not reference_tokens or len(reference_tokens) > len(mention_tokens):
        return False
    width = len(reference_tokens)
    return any(
        mention_tokens[index:index + width] == reference_tokens
        for index in range(len(mention_tokens) - width + 1)
    )


def character_references(character: dict[str, Any]) -> list[str]:
    return [
        str(value).strip()
        for value in (
            character.get("name"),
            character.get("id"),
            *(character.get("aliases") or []),
        )
        if str(value or "").strip()
    ]


def resolve_character_name(
    value: Any,
    characters: list[dict[str, Any]] | None,
) -> str | None:
    """Resolve a source mention to one unambiguous canonical character name."""
    mention = str(value or "").strip()
    mention_key = normalize_character_reference(mention)
    if not mention_key or not characters:
        return None

    exact_matches: set[str] = set()
    qualified_matches: list[tuple[int, str]] = []
    for character in characters:
        canonical = str(character.get("name") or "").strip()
        if not canonical:
            continue
        for reference in character_references(character):
            reference_key = normalize_character_reference(reference)
            if not reference_key:
                continue
            if mention_key == reference_key:
                exact_matches.add(canonical)
            elif _contains_reference(mention, reference):
                qualified_matches.append((len(reference_key), canonical))

    if len(exact_matches) == 1:
        return next(iter(exact_matches))
    if exact_matches:
        return None
    if not qualified_matches:
        return None

    longest = max(score for score, _canonical in qualified_matches)
    candidates = {
        canonical
        for score, canonical in qualified_matches
        if score == longest
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def resolve_character_id(
    value: Any,
    characters: list[dict[str, Any]] | None,
) -> str | None:
    """Resolve one source mention to one unambiguous canonical asset ID."""

    canonical_name = resolve_character_name(value, characters)
    if canonical_name is None or not characters:
        return None
    matches = {
        str(character.get("id") or "").strip()
        for character in characters
        if str(character.get("name") or "").strip() == canonical_name
        and str(character.get("id") or "").strip()
    }
    return next(iter(matches)) if len(matches) == 1 else None


def is_declared_character_reference(
    value: Any,
    characters: list[dict[str, Any]] | None,
) -> bool:
    return resolve_character_name(value, characters) is not None
