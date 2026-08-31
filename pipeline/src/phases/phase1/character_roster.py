"""Deterministic Phase 1 character roster compiled from source events.

The roster owns entity/instance cardinality.  A model may describe the visual
facts of an existing entity, but it may not create, remove, split, or merge
roster members.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from schemas.understanding import (
    CharacterRosterUnderstanding,
    CharacterRosterV1Understanding,
)
from utils.character_identity import (
    character_identity_is_explicitly_declared,
    character_reference_is_explicit,
    compatible_human_reference_descriptors,
    human_gender_descriptor,
    is_character_identity_candidate,
    is_gender_attribute_reference,
    normalize_character_reference,
    parse_human_reference_descriptor,
)

CHARACTER_ROSTER_SCHEMA = "honcut.character-roster.v2"
CHARACTER_ROSTER_FILENAME = "CHARACTER_ROSTER.json"
CHARACTER_ROSTER_MIGRATION_SCHEMA = "honcut.character-roster-migration.v1"
CHARACTER_ROSTER_MIGRATION_FILENAME = "CHARACTER_ROSTER_MIGRATION.json"


class CharacterRosterError(ValueError):
    """Raised when source identity cardinality cannot be proven."""


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}
_ENGLISH_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_ENGLISH_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_GENERIC_SOURCE_REFERENCES = {
    "主角",
    "主人公",
    "男主",
    "女主",
    "他",
    "她",
    "它",
    "they",
    "he",
    "she",
}
_GROUP_PERSON_ENDINGS = (
    "人员",
    "人",
    "者",
    "员",
    "卫",
    "兵",
    "士",
    "手",
    "师",
    "徒",
    "生",
    "童",
    "男",
    "女",
    "guards",
    "fighters",
    "soldiers",
    "agents",
    "workers",
    "people",
    "persons",
    "men",
    "women",
)
_ZH_GROUP_PATTERN = re.compile(
    r"(?<!第)(?P<count>[一二两三四五六七八九十百千0-9]+)\s*"
    r"(?:名|位|个)\s*"
    r"(?P<label>[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9_-]{0,23}?)"
    r"(?=突然|随后|一起|同时|从|向|在|进入|出现|冲出|走出|来到|"
    r"身穿|手持|发动|开始|正在|均|都|，|。|；|、|$)"
)
_EN_GROUP_PATTERN = re.compile(
    r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\s+"
    r"(?P<label>[A-Za-z][A-Za-z -]{1,40}?)"
    r"(?=\s+(?:appear|arrive|enter|rush|wear|carry|attack|move|stand)\b|[,.;]|$)",
    re.IGNORECASE,
)
_DISTINCT_REFERENCE_PATTERNS = (
    r"(?:另一|另一个|另外一|新来的|新出现的|一名新)\s*{label}",
    r"\b(?:another|a new|newly arrived)\s+{label}\b",
)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _positive_integer(value: str) -> int | None:
    token = str(value or "").strip().casefold()
    if not token:
        return None
    if token.isdigit():
        number = int(token)
        return number if number > 0 else None
    if token in _ENGLISH_NUMBERS:
        return _ENGLISH_NUMBERS[token]
    total = 0
    digit = 0
    found = False
    for character in token:
        if character in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[character]
            found = True
            continue
        unit = _CHINESE_UNITS.get(character)
        if unit is None:
            return None
        total += (digit or 1) * unit
        digit = 0
        found = True
    number = total + digit
    return number if found and number > 0 else None


def _identity_ordinal(label: Any) -> int | None:
    text = str(label or "").strip()
    values: set[int] = set()
    for match in re.finditer(
        r"第\s*([零〇一二两三四五六七八九十百千0-9]+)\s*(?:名|位|个|号)",
        text,
    ):
        if (value := _positive_integer(match.group(1))) is not None:
            values.add(value)
    for match in re.finditer(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"[0-9]+(?:st|nd|rd|th))\b",
        text.casefold(),
    ):
        token = match.group(1)
        values.add(_ENGLISH_ORDINALS.get(token, int(token[:-2]) if token[:-2].isdigit() else 0))
    for match in re.finditer(r"(?:#|\bno\.?\s*)([0-9]+)\b", text.casefold()):
        values.add(int(match.group(1)))
    values.discard(0)
    if len(values) > 1:
        raise CharacterRosterError(f"source mention has conflicting ordinals: {text}")
    return next(iter(values), None)


def _ordinal_base(label: Any) -> str:
    text = str(label or "").strip().casefold()
    text = re.sub(
        r"^第\s*[零〇一二两三四五六七八九十百千0-9]+\s*(?:名|位|个|号)\s*",
        "",
        text,
    )
    text = re.sub(
        r"^(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"[0-9]+(?:st|nd|rd|th))\s+",
        "",
        text,
    )
    text = re.sub(r"^(?:#|no\.?\s*)[0-9]+\s*", "", text)
    return re.sub(r"[\s_-]+", "", text)


def _event_ref(event: dict[str, Any], position: int) -> str:
    event_id = event.get("event_id") or event.get("id") or position
    return f"event:{event_id}"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _source_evidence(event: dict[str, Any], position: int) -> dict[str, str]:
    excerpt = str(event.get("source_excerpt") or "").strip()
    return {
        "event_ref": _event_ref(event, position),
        "sequence_id": str(event.get("sequence_id") or "__unspecified__"),
        "source_excerpt": excerpt,
        "source_excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    }


def _event_source_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(field) or "")
        for field in ("source_excerpt", "what", "start_state", "causal_link")
    )


def _source_introduces_distinct_reference(evidence: str, label: str) -> bool:
    escaped = re.escape(str(label or "").strip())
    if not escaped:
        return False
    return any(
        re.search(pattern.format(label=escaped), evidence, flags=re.IGNORECASE)
        for pattern in _DISTINCT_REFERENCE_PATTERNS
    )


def _continuous_alias_reconciliations(
    events: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    """Prove conservative source aliases before entity cardinality is compiled."""

    first_position: dict[str, int] = {}
    event_positions: dict[str, set[int]] = defaultdict(set)
    edges: list[tuple[str, str, dict[str, Any]]] = []
    for position, event in enumerate(events, 1):
        raw_who = event.get("who") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        model_who = event.get("model_who") or []
        if isinstance(model_who, str):
            model_who = [model_who]
        for value in [*raw_who, *model_who]:
            mention = str(value or "").strip()
            if mention:
                first_position.setdefault(mention, position)
        for value in raw_who:
            mention = str(value or "").strip()
            if mention:
                event_positions[mention].add(position)

        legacy_forward_mappings = (
            []
            if event.get("who_identity_reconciliations")
            else event.get("who_reconciled_from_forward_continuity") or []
        )
        for mapping in legacy_forward_mappings:
            if not isinstance(mapping, dict):
                continue
            model_label = str(mapping.get("model_label") or "").strip()
            source_identity = str(mapping.get("source_identity") or "").strip()
            if not compatible_human_reference_descriptors(model_label, source_identity):
                continue
            evidence = _event_source_text(event)
            if not character_reference_is_explicit(source_identity, evidence):
                continue
            descriptor = parse_human_reference_descriptor(model_label)
            following_position = min(position + 1, len(events))
            edges.append((model_label, source_identity, {
                "canonical_mention": model_label,
                "source_mention": source_identity,
                "sequence_id": str(event.get("sequence_id") or "__unspecified__"),
                "event_refs": [
                    _event_ref(event, position),
                    _event_ref(events[following_position - 1], following_position),
                ],
                "evidence_kind": "continuous_source_cross_reference",
                "controlled_gender": descriptor.gender if descriptor else "male",
                "evidence_sha256": hashlib.sha256(
                    evidence.encode("utf-8")
                ).hexdigest(),
            }))
        for mapping in event.get("who_identity_reconciliations") or []:
            if not isinstance(mapping, dict):
                continue
            model_label = str(mapping.get("model_label") or "").strip()
            source_identity = str(mapping.get("source_identity") or "").strip()
            if not compatible_human_reference_descriptors(model_label, source_identity):
                continue
            direction = str(mapping.get("direction") or "").strip()
            if direction == "forward":
                adjacent_position = min(position + 1, len(events))
            elif direction == "backward":
                adjacent_position = max(position - 1, 1)
            else:
                continue
            descriptor = parse_human_reference_descriptor(model_label)
            edges.append((model_label, source_identity, {
                "canonical_mention": model_label,
                "source_mention": source_identity,
                "sequence_id": str(event.get("sequence_id") or "__unspecified__"),
                "event_refs": [
                    _event_ref(events[adjacent_position - 1], adjacent_position),
                    _event_ref(event, position),
                ],
                "evidence_kind": "continuous_source_cross_reference",
                "controlled_gender": descriptor.gender if descriptor else "male",
                "evidence_sha256": str(mapping.get("evidence_sha256") or ""),
            }))

    for position in range(1, len(events)):
        previous = events[position - 1]
        current = events[position]
        if str(previous.get("sequence_id") or "__unspecified__") != str(
            current.get("sequence_id") or "__unspecified__"
        ):
            continue
        if str(current.get("continuity_before") or "cut").strip().casefold() != "continuous":
            continue
        previous_who = _ordered_unique(
            str(value or "").strip() for value in (previous.get("who") or [])
        )
        current_who = _ordered_unique(
            str(value or "").strip() for value in (current.get("who") or [])
        )
        shared = set(previous_who) & set(current_who)
        candidates: list[tuple[str, str]] = []
        for previous_label in previous_who:
            if previous_label in shared or _identity_ordinal(previous_label) is not None:
                continue
            for current_label in current_who:
                if current_label in shared or _identity_ordinal(current_label) is not None:
                    continue
                if not compatible_human_reference_descriptors(
                    previous_label,
                    current_label,
                ):
                    continue
                if event_positions[previous_label] & event_positions[current_label]:
                    continue
                previous_evidence = _event_source_text(previous)
                current_evidence = _event_source_text(current)
                if not (
                    character_reference_is_explicit(current_label, previous_evidence)
                    or character_reference_is_explicit(previous_label, current_evidence)
                ):
                    continue
                if _source_introduces_distinct_reference(
                    current_evidence,
                    current_label,
                ):
                    continue
                candidates.append((previous_label, current_label))
        if len(candidates) != 1:
            continue
        previous_label, current_label = candidates[0]
        descriptor = parse_human_reference_descriptor(previous_label)
        evidence = {
            "previous": _event_source_text(previous),
            "current": _event_source_text(current),
        }
        edges.append((previous_label, current_label, {
            "canonical_mention": previous_label,
            "source_mention": current_label,
            "sequence_id": str(current.get("sequence_id") or "__unspecified__"),
            "event_refs": [
                _event_ref(previous, position),
                _event_ref(current, position + 1),
            ],
            "evidence_kind": "continuous_source_cross_reference",
            "controlled_gender": descriptor.gender if descriptor else "male",
            "evidence_sha256": _canonical_json_sha256(evidence),
        }))

    parent = {mention: mention for mention in first_position}

    def preferred_mention(values: Iterable[str]) -> str:
        return min(
            values,
            key=lambda mention: (
                is_gender_attribute_reference(mention),
                first_position.get(mention, 10**9),
                mention,
            ),
        )

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        members = [
            mention
            for mention in parent
            if find(mention) in {left_root, right_root}
        ]
        qualified_count = sum(
            1
            for mention in members
            if (
                (descriptor := parse_human_reference_descriptor(mention))
                and descriptor.kind == "qualified"
            )
        )
        if qualified_count > 1:
            return
        canonical_root = preferred_mention((left_root, right_root))
        other_root = right_root if canonical_root == left_root else left_root
        parent[other_root] = canonical_root

    for left, right, _record in edges:
        union(left, right)

    aliases_by_canonical: dict[str, list[str]] = defaultdict(list)
    alias_to_canonical: dict[str, str] = {}
    for mention in sorted(first_position, key=lambda item: (first_position[item], item)):
        root = find(mention)
        members = [candidate for candidate in first_position if find(candidate) == root]
        canonical = preferred_mention(members)
        alias_to_canonical[mention] = canonical
        aliases_by_canonical[canonical].append(mention)
    for canonical, aliases in list(aliases_by_canonical.items()):
        aliases_by_canonical[canonical] = [
            canonical,
            *(mention for mention in aliases if mention != canonical),
        ]

    reconciliations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left, right, record in edges:
        canonical = alias_to_canonical.get(left, left)
        if alias_to_canonical.get(right, right) != canonical:
            continue
        item = dict(record)
        item["canonical_mention"] = canonical
        item["source_mention"] = right if right != canonical else left
        if item not in reconciliations[canonical]:
            reconciliations[canonical].append(item)
    return alias_to_canonical, dict(aliases_by_canonical), dict(reconciliations)


def _plausible_group_label(label: str) -> bool:
    normalized = str(label or "").strip().casefold()
    return bool(normalized) and normalized.endswith(_GROUP_PERSON_ENDINGS)


def _group_declarations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for position, event in enumerate(events, 1):
        excerpt = str(event.get("source_excerpt") or "").strip()
        if not excerpt:
            continue
        matches = [*_ZH_GROUP_PATTERN.finditer(excerpt), *_EN_GROUP_PATTERN.finditer(excerpt)]
        for match in sorted(matches, key=lambda item: item.start()):
            count = _positive_integer(match.group("count"))
            label = re.sub(r"\s+", " ", match.group("label")).strip(" ,，。；;.-")
            introduction_prefix = excerpt[max(0, match.start() - 3):match.start()]
            if (
                count is None
                or count < 2
                or re.search(r"(?:另|另外|新)$", introduction_prefix)
                or not _plausible_group_label(label)
            ):
                continue
            declarations.append({
                "count": count,
                "label": label,
                "sequence_id": str(event.get("sequence_id") or "__unspecified__"),
                "event_ref": _event_ref(event, position),
                "event_position": position,
            })
    return declarations


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = _canonical_json_sha256([str(part) for part in parts])[:16]
    return f"{prefix}_{digest}"


def _entity_record(
    *,
    display_name: str,
    instance_mentions: list[list[str]],
    mention_events: dict[str, list[tuple[int, dict[str, Any]]]],
    declaration_events: list[tuple[int, dict[str, Any]]] | None = None,
    origin: str,
    mention_reconciliations: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    normalized_mentions = [
        _ordered_unique(str(value).strip() for value in mentions)
        for mentions in instance_mentions
    ]
    entity_id = _stable_id("ENTITY", display_name, normalized_mentions)
    instances = []
    evidence_by_ref: dict[str, dict[str, str]] = {}
    for ordinal, mentions in enumerate(normalized_mentions, 1):
        event_pairs = [
            pair
            for mention in mentions
            for pair in mention_events.get(mention, [])
        ]
        event_pairs.sort(key=lambda pair: pair[0])
        event_refs = _ordered_unique(
            _event_ref(event, position) for position, event in event_pairs
        )
        action_refs = _ordered_unique(
            str(event.get("action_unit_id") or "").strip()
            for _position, event in event_pairs
        )
        instances.append({
            "instance_id": f"{entity_id}_I{ordinal:02d}",
            "ordinal": ordinal,
            "source_mentions": mentions or [display_name],
            "event_refs": event_refs,
            "action_unit_refs": action_refs,
            "identity_reconciliations": _ordered_unique_reconciliations(
                reconciliation
                for mention in mentions
                for reconciliation in (mention_reconciliations or {}).get(mention, [])
            ),
        })
        for position, event in event_pairs:
            item = _source_evidence(event, position)
            evidence_by_ref.setdefault(item["event_ref"], item)
    for position, event in declaration_events or []:
        item = _source_evidence(event, position)
        evidence_by_ref.setdefault(item["event_ref"], item)
    return {
        "entity_id": entity_id,
        "display_name": display_name,
        "instance_count": len(instances),
        "instances": instances,
        "source_visual_evidence": list(evidence_by_ref.values()),
        "reconciliation_origin": origin,
    }


def _ordered_unique_reconciliations(
    values: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        key = _canonical_json_sha256(value)
        if key not in seen:
            seen.add(key)
            result.append(copy.deepcopy(value))
    return result


def compile_character_roster(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile cardinality and identity ownership without a Provider call."""

    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise CharacterRosterError("character roster events must be an object array")
    source_hash = _canonical_json_sha256(events)
    alias_to_canonical, proven_aliases, mention_reconciliations = (
        _continuous_alias_reconciliations(events)
    )
    mention_events: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    mention_aliases: dict[str, list[str]] = {}
    mentions_by_sequence: dict[str, list[str]] = defaultdict(list)
    for position, event in enumerate(events, 1):
        sequence_id = str(event.get("sequence_id") or "__unspecified__")
        raw_who = event.get("who") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        if not isinstance(raw_who, list):
            raise CharacterRosterError(f"event {position} who must be an array")
        for value in raw_who:
            mention = str(value or "").strip()
            if not mention or mention.casefold() in _GENERIC_SOURCE_REFERENCES:
                continue
            if not is_character_identity_candidate(mention) and not (
                character_identity_is_explicitly_declared(
                    mention,
                    _event_source_text(event),
                )
            ):
                continue
            canonical = alias_to_canonical.get(mention, mention)
            mention_events[canonical].append((position, event))
            mention_aliases[canonical] = _ordered_unique([
                *mention_aliases.get(canonical, []),
                *proven_aliases.get(canonical, [mention]),
            ])
            if canonical not in mentions_by_sequence[sequence_id]:
                mentions_by_sequence[sequence_id].append(canonical)

    declarations = _group_declarations(events)
    grouped_mentions: set[str] = set()
    records: list[dict[str, Any]] = []
    for sequence_id, sequence_mentions in mentions_by_sequence.items():
        ordinal_mentions: dict[int, list[str]] = defaultdict(list)
        for mention in sequence_mentions:
            ordinal = _identity_ordinal(mention)
            if ordinal is not None:
                ordinal_mentions[ordinal].append(mention)
        if not ordinal_mentions:
            continue
        duplicate_ordinals = {
            ordinal: names
            for ordinal, names in ordinal_mentions.items()
            if len({_ordinal_base(name) for name in names}) > 1
        }
        if duplicate_ordinals:
            raise CharacterRosterError(
                f"duplicate ordinal identity in {sequence_id}: {duplicate_ordinals}"
            )
        candidates = [
            declaration
            for declaration in declarations
            if declaration["sequence_id"] == sequence_id
        ]
        if not candidates:
            continue
        bases = {_ordinal_base(names[0]) for names in ordinal_mentions.values()}
        exact = [
            candidate
            for candidate in candidates
            if _ordinal_base(candidate["label"]) in bases
        ]
        viable_pool = exact or candidates
        if len(viable_pool) != 1:
            raise CharacterRosterError(
                f"ambiguous group declarations in {sequence_id}: "
                f"{[item['label'] for item in viable_pool]}"
            )
        declaration = viable_pool[0]
        expected = list(range(1, int(declaration["count"]) + 1))
        observed = sorted(ordinal_mentions)
        if observed != expected:
            raise CharacterRosterError(
                f"group count conflict in {sequence_id}: "
                f"declared={declaration['count']} observed_ordinals={observed}"
            )
        instance_mentions = [
            _ordered_unique(
                alias
                for mention in ordinal_mentions[ordinal]
                for alias in mention_aliases.get(mention, [mention])
            )
            for ordinal in expected
        ]
        ordinal_source_mentions = {
            alias
            for names in ordinal_mentions.values()
            for name in names
            for alias in mention_aliases.get(name, [name])
        }
        group_generic_aliases: list[str] = []
        for mention in sequence_mentions:
            if (
                _identity_ordinal(mention) is not None
                or mention == declaration["label"]
                or _ordinal_base(mention) not in bases
            ):
                continue
            event_pairs = mention_events.get(mention, [])
            # A label that existed before the counted group is an independent
            # source identity, not a retroactive alias of the group.
            if not event_pairs or any(
                position <= declaration["event_position"]
                for position, _event in event_pairs
            ):
                continue
            if any(
                _source_introduces_distinct_reference(
                    _event_source_text(event),
                    mention,
                )
                for _position, event in event_pairs
            ):
                continue
            for position, event in event_pairs:
                source_text = _event_source_text(event)
                if not character_reference_is_explicit(mention, source_text):
                    raise CharacterRosterError(
                        f"group generic reference lacks source evidence in "
                        f"{_event_ref(event, position)}: {mention}"
                    )
                raw_who = event.get("who") or []
                if isinstance(raw_who, str):
                    raw_who = [raw_who]
                if any(
                    source_mention in raw_who
                    for source_mention in ordinal_source_mentions
                ):
                    raise CharacterRosterError(
                        f"group generic reference co-occurs with a numbered "
                        f"member in {_event_ref(event, position)}: {mention}"
                    )
            group_generic_aliases.append(mention)

        if group_generic_aliases:
            instance_mentions = [
                _ordered_unique([*mentions, *group_generic_aliases])
                for mentions in instance_mentions
            ]
        declaration_pair = (
            declaration["event_position"],
            events[declaration["event_position"] - 1],
        )
        records.append(_entity_record(
            display_name=declaration["label"],
            instance_mentions=instance_mentions,
            mention_events=mention_events,
            declaration_events=[declaration_pair],
            origin="deterministic_group_completion",
            mention_reconciliations=mention_reconciliations,
        ))
        grouped_mentions.update(
            mention for mentions in instance_mentions for mention in mentions
        )

    # Explicit numbered people without a proven group remain separate.  Plain
    # repeated mentions form one independent entity with one instance.
    declared_group_labels = {declaration["label"] for declaration in declarations}
    first_group_declaration_position = {
        label: min(
            declaration["event_position"]
            for declaration in declarations
            if declaration["label"] == label
        )
        for label in declared_group_labels
    }
    for mention, event_pairs in sorted(
        mention_events.items(), key=lambda item: item[1][0][0]
    ):
        if mention in grouped_mentions:
            continue
        if mention in declared_group_labels and not any(
            position < first_group_declaration_position[mention]
            for position, _event in event_pairs
        ):
            continue
        records.append(_entity_record(
            display_name=mention,
            instance_mentions=[mention_aliases.get(mention, [mention])],
            mention_events=mention_events,
            origin="explicit_source",
            mention_reconciliations=mention_reconciliations,
        ))

    # An explicit counted group without numbered member labels still preserves
    # source cardinality.  Its anonymous instances remain owned by the group and
    # are never guessed into separately named people.
    used_group_names = {record["display_name"] for record in records}
    for declaration in declarations:
        if declaration["label"] in used_group_names:
            continue
        same_key = [
            candidate
            for candidate in declarations
            if candidate["sequence_id"] == declaration["sequence_id"]
            and candidate["label"] == declaration["label"]
        ]
        counts = {candidate["count"] for candidate in same_key}
        if len(counts) != 1:
            raise CharacterRosterError(
                f"group count conflict for {declaration['label']}: {sorted(counts)}"
            )
        declaration_pairs = [
            (candidate["event_position"], events[candidate["event_position"] - 1])
            for candidate in same_key
        ]
        records.append(_entity_record(
            display_name=declaration["label"],
            instance_mentions=[
                [declaration["label"]]
                for _ordinal in range(1, int(declaration["count"]) + 1)
            ],
            mention_events=mention_events,
            declaration_events=declaration_pairs,
            origin="explicit_source",
            mention_reconciliations=mention_reconciliations,
        ))
        used_group_names.add(declaration["label"])

    records.sort(
        key=lambda record: min(
            (
                int(ref.removeprefix("event:"))
                for instance in record["instances"]
                for ref in instance["event_refs"]
                if ref.removeprefix("event:").isdigit()
            ),
            default=10**9,
        )
    )
    unsigned = {
        "schema": CHARACTER_ROSTER_SCHEMA,
        "source_events_sha256": source_hash,
        "entities": records,
    }
    roster = {**unsigned, "roster_sha256": _canonical_json_sha256(unsigned)}
    return validate_character_roster(roster)


def validate_character_roster(value: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = CharacterRosterUnderstanding.model_validate(value)
    except ValidationError as error:
        raise CharacterRosterError("invalid character roster schema") from error
    payload = parsed.model_dump(by_alias=True)
    claimed = payload.pop("roster_sha256")
    if _canonical_json_sha256(payload) != claimed:
        raise CharacterRosterError("character roster hash mismatch")
    entity_ids: set[str] = set()
    instance_ids: set[str] = set()
    for entity in payload["entities"]:
        if entity["entity_id"] in entity_ids:
            raise CharacterRosterError("duplicate character roster entity ID")
        entity_ids.add(entity["entity_id"])
        if entity["instance_count"] != len(entity["instances"]):
            raise CharacterRosterError("character roster instance count mismatch")
        if [item["ordinal"] for item in entity["instances"]] != list(
            range(1, entity["instance_count"] + 1)
        ):
            raise CharacterRosterError("character roster ordinals are incomplete")
        for instance in entity["instances"]:
            if instance["instance_id"] in instance_ids:
                raise CharacterRosterError("duplicate character roster instance ID")
            instance_ids.add(instance["instance_id"])
    payload["roster_sha256"] = claimed
    return copy.deepcopy(payload)


def persist_character_roster(
    path: str | Path,
    roster: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist the source-owned roster before any model request."""

    validated = validate_character_roster(roster)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return validated


def migrate_character_roster_v1(
    value: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    receipt_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompile a hash-verified v1 roster from its original source events."""

    if not isinstance(value, dict) or value.get("schema") != "honcut.character-roster.v1":
        raise CharacterRosterError("only character roster v1 can migrate")
    unsigned = copy.deepcopy(value)
    claimed = str(unsigned.pop("roster_sha256", ""))
    if _canonical_json_sha256(unsigned) != claimed:
        raise CharacterRosterError("legacy character roster hash mismatch")
    try:
        CharacterRosterV1Understanding.model_validate(value)
    except ValidationError as error:
        raise CharacterRosterError("invalid legacy character roster schema") from error
    events_sha256 = _canonical_json_sha256(events)
    if events_sha256 != value.get("source_events_sha256"):
        raise CharacterRosterError("legacy character roster source lineage mismatch")

    migrated = compile_character_roster(events)
    legacy_entity_ids = [
        str(entity.get("entity_id") or "") for entity in value.get("entities") or []
    ]
    migrated_entity_ids = [
        str(entity.get("entity_id") or "") for entity in migrated["entities"]
    ]
    legacy_instance_ids = [
        str(instance.get("instance_id") or "")
        for entity in value.get("entities") or []
        for instance in entity.get("instances") or []
    ]
    migrated_instance_ids = [
        str(instance.get("instance_id") or "")
        for entity in migrated["entities"]
        for instance in entity["instances"]
    ]
    downstream_reuse_allowed = (
        legacy_entity_ids == migrated_entity_ids
        and legacy_instance_ids == migrated_instance_ids
    )
    receipt = {
        "schema": CHARACTER_ROSTER_MIGRATION_SCHEMA,
        "status": "migrated",
        "source_schema": "honcut.character-roster.v1",
        "target_schema": CHARACTER_ROSTER_SCHEMA,
        "source_roster_sha256": claimed,
        "target_roster_sha256": migrated["roster_sha256"],
        "source_events_sha256": events_sha256,
        "legacy_entity_count": len(legacy_entity_ids),
        "target_entity_count": len(migrated_entity_ids),
        "legacy_instance_count": len(legacy_instance_ids),
        "target_instance_count": len(migrated_instance_ids),
        "downstream_reuse_allowed": downstream_reuse_allowed,
        "legacy_artifact_preserved": True,
    }
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    if receipt_path is not None:
        target = Path(receipt_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    return migrated, receipt


def _stable_completion(
    entity_id: str,
    values: tuple[str, ...],
    label: str,
) -> str:
    digest = hashlib.sha256(f"{entity_id}:{label}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:2], "big") % len(values)]


_VISUAL_EVIDENCE_MARKERS = (
    "岁", "cm", "厘米", "身高", "体型", "偏瘦", "健壮", "结实", "纤细",
    "发", "脸", "眼", "眉", "鼻", "唇", "皮肤", "肤色", "穿", "服", "衣",
    "裤", "裙", "鞋", "靴", "帽", "甲", "盔", "纹", "scar", "hair", "face",
    "skin", "wear", "coat", "shirt", "dress", "trouser", "boot", "armor",
)


def _static_visual_evidence(entity: dict[str, Any]) -> list[str]:
    clauses: list[str] = []
    for evidence in entity.get("source_visual_evidence") or []:
        excerpt = str(evidence.get("source_excerpt") or "")
        for clause in re.split(r"[。；;，,]", excerpt):
            clause = clause.strip()
            if clause and any(
                marker.casefold() in clause.casefold()
                for marker in _VISUAL_EVIDENCE_MARKERS
            ):
                clauses.append(clause)
    return list(dict.fromkeys(clauses))[:8]


def _source_grounded_character(entity: dict[str, Any]) -> dict[str, Any]:
    """Build a complete deterministic DTO when an observation omits an entity."""

    entity_id = str(entity["entity_id"])
    display_name = str(entity["display_name"])
    visual_evidence = _static_visual_evidence(entity)
    gender = human_gender_descriptor(display_name) or "unknown"
    build = _stable_completion(
        entity_id,
        ("lean", "balanced", "athletic", "sturdy"),
        "body_build",
    )
    hair = _stable_completion(
        entity_id,
        (
            "short dark straight hair",
            "medium dark softly textured hair",
            "short dark softly textured hair",
            "medium dark straight hair",
        ),
        "hair",
    )
    face_code = hashlib.sha256(
        f"{entity_id}:face".encode("utf-8")
    ).hexdigest()[:8]
    evidence_text = "；".join(visual_evidence)
    clothing = (
        f"source-defined wardrobe: {evidence_text}"
        if evidence_text
        else "stable source-compatible cinematic wardrobe"
    )
    aliases = list(dict.fromkeys(
        mention
        for instance in entity.get("instances") or []
        for mention in instance.get("source_mentions") or []
        if mention and mention != display_name
    ))
    return {
        "id": entity_id,
        "entity_id": entity_id,
        "name": display_name,
        "aliases": aliases,
        "role": "extra",
        "appearance": {
            "gender": gender,
            "age_range": "adult",
            "height": "source-defined or stable adult proportion",
            "build": build,
            "hair": hair,
            "face": f"stable fictional face geometry {face_code}",
            "clothing": clothing,
            "interaction_props": [],
            "identity_props": [],
            "distinguishing": "",
            "summary": f"{display_name}: {hair}, {build} build, {clothing}",
            "variants": [],
        },
        "personality": {"traits": [], "speech_style": "", "motivation": ""},
        "style": "cinematic source-derived character design",
        "negative": "identity drift, accidental cloning, inconsistent wardrobe",
        "size": "2K",
        "first_appearance": 0,
        "appearance_count": 0,
        "relationships": [],
    }


def _roster_entity_labels(entity: dict[str, Any]) -> set[str]:
    return {
        normalize_character_reference(value)
        for value in (
            entity.get("display_name"),
            *(
                mention
                for instance in (entity.get("instances") or [])
                for mention in (instance.get("source_mentions") or [])
            ),
        )
        if normalize_character_reference(value)
    }


def _observation_labels(observation: dict[str, Any]) -> set[str]:
    return {
        normalize_character_reference(value)
        for value in (
            observation.get("name"),
            *(observation.get("aliases") or []),
        )
        if normalize_character_reference(value)
    }


def reconcile_character_observations(
    observations: list[dict[str, Any]],
    roster: dict[str, Any],
    *,
    semantic_qa_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project probabilistic visual observations onto the source-owned roster."""

    if not isinstance(semantic_qa_enabled, bool):
        raise ValueError("semantic_qa_enabled must be a boolean")
    roster = validate_character_roster(roster)
    entities = list(roster["entities"])
    by_entity = {str(entity["entity_id"]): [] for entity in entities}
    entity_labels = {
        str(entity["entity_id"]): _roster_entity_labels(entity)
        for entity in entities
    }
    diagnostics: list[dict[str, Any]] = []

    for observation in observations:
        observation_id = str(observation.get("id") or "").strip()
        labels = _observation_labels(observation)
        exact_ids = {
            entity_id
            for entity_id, known_labels in entity_labels.items()
            if labels & known_labels
        }
        if observation_id in by_entity:
            exact_ids.add(observation_id)
        if not exact_ids:
            observation_bases = {
                _ordinal_base(label) for label in labels if _ordinal_base(label)
            }
            exact_ids = {
                entity_id
                for entity_id, known_labels in entity_labels.items()
                if observation_bases
                & {
                    _ordinal_base(label)
                    for label in known_labels
                    if _ordinal_base(label)
                }
            }
        if len(exact_ids) != 1:
            diagnostics.append({
                "code": (
                    "model_entity_unbound"
                    if not exact_ids
                    else "model_entity_merge"
                ),
                "model_observation_id": observation_id,
                "candidate_entity_ids": sorted(exact_ids),
            })
            continue
        entity_id = next(iter(exact_ids))
        by_entity[entity_id].append(observation)

    characters: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = str(entity["entity_id"])
        candidates = by_entity[entity_id]
        if not candidates:
            diagnostics.append({
                "code": "model_entity_missing",
                "entity_id": entity_id,
            })
            character = _source_grounded_character(entity)
            origin = str(entity["reconciliation_origin"])
        else:
            if len(candidates) > 1:
                diagnostics.append({
                    "code": "model_entity_split",
                    "entity_id": entity_id,
                    "model_observation_ids": sorted(
                        str(item.get("id") or "") for item in candidates
                    ),
                })
            character = dict(min(
                candidates,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ))
            model_id = str(character.get("id") or "")
            if model_id != entity_id:
                diagnostics.append({
                    "code": "model_entity_id_rewritten",
                    "entity_id": entity_id,
                    "model_observation_id": model_id,
                })
            expected_mentions = list(dict.fromkeys(
                mention
                for instance in entity["instances"]
                for mention in instance["source_mentions"]
            ))
            observed_labels = _observation_labels(character)
            missing_aliases = [
                mention
                for mention in expected_mentions
                if normalize_character_reference(mention) not in observed_labels
            ]
            if missing_aliases:
                diagnostics.append({
                    "code": "model_alias_omission",
                    "entity_id": entity_id,
                    "source_mentions": missing_aliases,
                })
            character["model_observation_id"] = model_id
            origin = "model_observation"

        source_mentions = list(dict.fromkeys(
            mention
            for instance in entity["instances"]
            for mention in instance["source_mentions"]
        ))
        event_ids = sorted({
            int(ref.removeprefix("event:"))
            for instance in entity["instances"]
            for ref in instance["event_refs"]
            if ref.removeprefix("event:").isdigit()
        })
        character.update({
            "id": entity_id,
            "entity_id": entity_id,
            "name": entity["display_name"],
            "aliases": [
                mention
                for mention in source_mentions
                if mention != entity["display_name"]
            ],
            "instance_count": entity["instance_count"],
            "instances": entity["instances"],
            "source_visual_evidence": entity["source_visual_evidence"],
            "reconciliation_origin": origin,
            "character_roster_sha256": roster["roster_sha256"],
            "first_appearance": event_ids[0] if event_ids else 0,
            "appearance_count": len(event_ids),
            "source_identity_evidence": {
                "event_ids": event_ids,
                "source_mentions": source_mentions,
                "inferred_aliases": [],
            },
        })
        characters.append(character)

    if semantic_qa_enabled and diagnostics:
        codes = ", ".join(item["code"] for item in diagnostics)
        raise ValueError(f"strict character roster semantic QA failed: {codes}")
    return characters, diagnostics


__all__ = [
    "CHARACTER_ROSTER_SCHEMA",
    "CHARACTER_ROSTER_FILENAME",
    "CHARACTER_ROSTER_MIGRATION_SCHEMA",
    "CHARACTER_ROSTER_MIGRATION_FILENAME",
    "CharacterRosterError",
    "compile_character_roster",
    "persist_character_roster",
    "migrate_character_roster_v1",
    "reconcile_character_observations",
    "validate_character_roster",
]
