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

from schemas.understanding import CharacterRosterUnderstanding
from utils.character_identity import (
    human_gender_descriptor,
    normalize_character_reference,
)

CHARACTER_ROSTER_SCHEMA = "honcut.character-roster.v1"
CHARACTER_ROSTER_FILENAME = "CHARACTER_ROSTER.json"


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
            if count is None or not _plausible_group_label(label):
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


def compile_character_roster(
    events: list[dict[str, Any]],
    *,
    source_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile cardinality and identity ownership without a Provider call."""

    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise CharacterRosterError("character roster events must be an object array")
    source_hash = _canonical_json_sha256(events)
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
            mention_events[mention].append((position, event))
            mention_aliases.setdefault(mention, [mention])
            if mention not in mentions_by_sequence[sequence_id]:
                mentions_by_sequence[sequence_id].append(mention)

    if source_stats is not None:
        if not isinstance(source_stats, dict):
            raise CharacterRosterError("character roster source_stats must be an object")
        event_by_id = {
            str(event.get("event_id") or event.get("id") or position): (position, event)
            for position, event in enumerate(events, 1)
        }
        compiled_mentions: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        compiled_sequences: dict[str, list[str]] = defaultdict(list)
        compiled_aliases: dict[str, list[str]] = {}
        for canonical, info in source_stats.items():
            canonical = str(canonical or "").strip()
            if not canonical or not isinstance(info, dict):
                raise CharacterRosterError("source_stats identity is invalid")
            aliases = _ordered_unique([
                canonical,
                *(
                    str(alias or "").strip()
                    for alias in (info.get("source_aliases") or [])
                ),
            ])
            compiled_aliases[canonical] = aliases
            pairs = [
                event_by_id[str(event_id)]
                for event_id in (info.get("events") or [])
                if str(event_id) in event_by_id
            ]
            pairs.sort(key=lambda pair: pair[0])
            compiled_mentions[canonical].extend(pairs)
            for _position, event in pairs:
                sequence_id = str(event.get("sequence_id") or "__unspecified__")
                if canonical not in compiled_sequences[sequence_id]:
                    compiled_sequences[sequence_id].append(canonical)
        mention_events = compiled_mentions
        mentions_by_sequence = compiled_sequences
        mention_aliases = compiled_aliases

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
        ))
        grouped_mentions.update(
            mention for mentions in instance_mentions for mention in mentions
        )

    # Explicit numbered people without a proven group remain separate.  Plain
    # repeated mentions form one independent entity with one instance.
    declared_group_labels = {declaration["label"] for declaration in declarations}
    for mention, event_pairs in sorted(
        mention_events.items(), key=lambda item: item[1][0][0]
    ):
        if mention in grouped_mentions or mention in declared_group_labels:
            continue
        records.append(_entity_record(
            display_name=mention,
            instance_mentions=[mention_aliases.get(mention, [mention])],
            mention_events=mention_events,
            origin="explicit_source",
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
    "CharacterRosterError",
    "compile_character_roster",
    "persist_character_roster",
    "reconcile_character_observations",
    "validate_character_roster",
]
