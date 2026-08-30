"""Canonical, source-bound visual facts shared by every downstream Phase.

The Phase 1 character discoverer remains the only probabilistic extractor.
This module converts its accepted output into a versioned geometry contract and
deterministically fills only fields the source left unspecified.  Downstream
owners consume this contract and must never infer replacement visual facts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from schemas.understanding import CanonicalVisualContractUnderstanding
from utils.character_reference_contracts import normalize_identity_props


CANONICAL_VISUAL_CONTRACT_SCHEMA = "honcut.canonical-visual-contract.v2"
LEGACY_CANONICAL_VISUAL_CONTRACT_SCHEMA = "honcut.canonical-visual-contract.v1"
CANONICAL_VISUAL_CONTRACT_FILENAME = "CANONICAL_VISUAL_CONTRACT.json"
CANONICAL_VISUAL_MIGRATION_RECEIPT_FILENAME = (
    "CANONICAL_VISUAL_CONTRACT_MIGRATION.json"
)
CANONICAL_VISUAL_MIGRATION_RECEIPT_SCHEMA = (
    "honcut.canonical-visual-contract-migration.v1"
)

SOURCE_DERIVED_POLICY = "source_derived"
FICTIONAL_CINEMATIC_HUMAN_POLICY = "fictional_cinematic_human_v1"
SYNTHETIC_STYLIZED_POLICY = "synthetic_stylized_character_v3"
CHARACTER_VISUAL_POLICIES = (
    SOURCE_DERIVED_POLICY,
    FICTIONAL_CINEMATIC_HUMAN_POLICY,
    SYNTHETIC_STYLIZED_POLICY,
)
CHARACTER_RUNTIME_POLICIES = (
    FICTIONAL_CINEMATIC_HUMAN_POLICY,
    SYNTHETIC_STYLIZED_POLICY,
)

_SYNTHETIC_MARKERS = re.compile(
    r"(?:合成人|仿生人|机器人|机械人|安卓人|数字角色|人工智能躯体|"
    r"android|synthetic\s+(?:person|character|body)|robot(?:ic)?|cyborg)",
    re.IGNORECASE,
)


class CanonicalVisualContractError(RuntimeError):
    """Raised before downstream media generation when visual facts are invalid."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_character_visual_policy(value: object) -> str:
    normalized = str(value or SOURCE_DERIVED_POLICY).strip().lower().replace("-", "_")
    aliases = {
        "source": SOURCE_DERIVED_POLICY,
        "source_derived": SOURCE_DERIVED_POLICY,
        "fictional_cinematic_human": FICTIONAL_CINEMATIC_HUMAN_POLICY,
        FICTIONAL_CINEMATIC_HUMAN_POLICY: FICTIONAL_CINEMATIC_HUMAN_POLICY,
        "synthetic_stylized": SYNTHETIC_STYLIZED_POLICY,
        SYNTHETIC_STYLIZED_POLICY: SYNTHETIC_STYLIZED_POLICY,
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            "character_visual_policy must be one of "
            + ", ".join(CHARACTER_VISUAL_POLICIES)
        )
    return resolved


def _stable_choice(stable_id: str, field: str, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{stable_id}:{field}".encode("utf-8")).digest()
    return values[digest[0] % len(values)]


def _fact(
    value: str | int | float | bool | list[str],
    *,
    explicit: bool,
    source_ref: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "origin": "explicit_source" if explicit else "deterministic_completion",
        "source_refs": [source_ref] if explicit else [],
    }


def _first_marker(text: str, candidates: tuple[tuple[str, str], ...]) -> str | None:
    folded = text.casefold()
    for marker, value in candidates:
        if marker.casefold() in folded:
            return value
    return None


def _hair_geometry(character_id: str, value: object) -> dict[str, Any]:
    text = str(value or "").strip()
    source_ref = f"character:{character_id}:appearance.hair"
    color = _first_marker(text, (
        ("黑", "black"), ("black", "black"),
        ("深棕", "dark_brown"), ("brown", "brown"),
        ("白", "white"), ("white", "white"),
        ("银", "silver"), ("silver", "silver"),
        ("蓝", "blue"), ("blue", "blue"),
        ("红", "red"), ("red", "red"),
        ("金", "blonde"), ("blonde", "blonde"),
    ))
    length_class = _first_marker(text, (
        ("光头", "bald"), ("bald", "bald"),
        ("寸头", "buzz"), ("buzz", "buzz"),
        ("短发", "short"), ("short", "short"),
        ("齐耳", "ear"), ("ear-length", "ear"),
        ("及颈", "neck"), ("neck-length", "neck"),
        ("齐肩", "shoulder"), ("shoulder", "shoulder"),
        ("长发", "long"), ("long", "long"),
        ("及腰", "waist"), ("waist", "waist"),
    ))
    texture = _first_marker(text, (
        ("卷", "curly"), ("curly", "curly"),
        ("波浪", "wavy"), ("wavy", "wavy"),
        ("直", "straight"), ("straight", "straight"),
        ("纤维", "designed_fiber"),
    ))
    parting = _first_marker(text, (
        ("中分", "center"), ("center part", "center"),
        ("侧分", "side"), ("side part", "side"),
        ("无分缝", "none"),
    ))
    tie_state = _first_marker(text, (
        ("马尾", "ponytail"), ("ponytail", "ponytail"),
        ("发髻", "bun"), ("bun", "bun"),
        ("辫", "braided"), ("braid", "braided"),
        ("披散", "loose"), ("loose", "loose"),
    ))
    resolved_length = length_class or _stable_choice(
        character_id, "hair_length", ("short", "ear", "neck")
    )
    silhouette = {
        "bald": "bare_head",
        "buzz": "close_to_head",
        "short": "close_to_head",
        "ear": "jaw_contour",
        "neck": "neck_contour",
        "shoulder": "shoulder_contour",
        "long": "below_shoulders",
        "waist": "waist_length",
    }[resolved_length]
    return {
        "color": _fact(
            color or _stable_choice(character_id, "hair_color", ("black", "dark_brown")),
            explicit=color is not None,
            source_ref=source_ref,
        ),
        "length_class": _fact(
            resolved_length,
            explicit=length_class is not None,
            source_ref=source_ref,
        ),
        "silhouette": _fact(
            silhouette,
            explicit=length_class is not None,
            source_ref=source_ref,
        ),
        "parting": _fact(
            parting or _stable_choice(character_id, "hair_parting", ("side", "none")),
            explicit=parting is not None,
            source_ref=source_ref,
        ),
        "texture": _fact(
            texture or "straight",
            explicit=texture is not None,
            source_ref=source_ref,
        ),
        "tie_state": _fact(
            tie_state or "untied",
            explicit=tie_state is not None,
            source_ref=source_ref,
        ),
    }


def _number_marker(text: str, markers: tuple[tuple[str, int], ...]) -> int | None:
    folded = text.casefold()
    for marker, value in markers:
        if marker.casefold() in folded:
            return value
    return None


def _prop_geometry(prop_id: str, description: str) -> dict[str, Any]:
    text = description.strip()
    source_ref = f"identity_prop:{prop_id}:description"
    shape = _first_marker(text, (
        ("六边形", "hexagonal"), ("hexagon", "hexagonal"),
        ("圆形", "circular"), ("circular", "circular"),
        ("矩形", "rectangular"), ("rectangular", "rectangular"),
        ("三角", "triangular"), ("triangular", "triangular"),
        ("刀刃", "blade"), ("blade", "blade"),
        ("短棍", "short_rod"), ("rod", "rod"),
        ("盾", "shield"), ("shield", "shield"),
        ("枪", "firearm"), ("rifle", "firearm"),
    ))
    composite = bool(re.search(r"(?:组合|复合|可拆|双节|multi[ -]?part|composite)", text, re.I))
    component_count = _number_marker(text, (
        ("三段", 3), ("three-part", 3),
        ("双段", 2), ("两段", 2), ("two-part", 2),
        ("一体", 1), ("single-piece", 1),
    ))
    end_count = _number_marker(text, (
        ("双头", 2), ("双端", 2), ("double-ended", 2),
        ("单头", 1), ("单端", 1), ("single-ended", 1),
    ))
    handle_count = _number_marker(text, (
        ("双握柄", 2), ("two handles", 2),
        ("单握柄", 1), ("握柄", 1), ("handle", 1),
    ))
    is_weapon = bool(re.search(r"(?:武器|刀|剑|刃|枪|棍|weapon|blade|sword|rifle)", text, re.I))
    material = _first_marker(text, (
        ("透明", "transparent"), ("transparent", "transparent"),
        ("玻璃", "glass"), ("glass", "glass"),
        ("金属", "metal"), ("metal", "metal"),
        ("陶瓷", "ceramic"), ("ceramic", "ceramic"),
        ("木", "wood"), ("wood", "wood"),
    ))
    colors = [
        value
        for marker, value in (
            ("黑", "black"), ("白", "white"), ("蓝", "blue"),
            ("红", "red"), ("绿", "green"), ("金", "gold"),
            ("银", "silver"), ("紫", "purple"),
        )
        if marker in text
    ]
    if not colors:
        colors = [
            value
            for marker, value in (
                ("black", "black"), ("white", "white"), ("blue", "blue"),
                ("red", "red"), ("green", "green"), ("gold", "gold"),
                ("silver", "silver"), ("purple", "purple"),
            )
            if marker in text.casefold()
        ]
    emissive = []
    if re.search(r"(?:发光|能量|光纹|emissive|glow|energy)", text, re.I):
        emissive.append("declared_emissive_region")
    scale = _first_marker(text, (
        ("微型", "tiny"), ("掌心", "palm_sized"), ("手掌", "palm_sized"),
        ("短", "short"), ("长", "long"), ("大型", "large"),
        ("tiny", "tiny"), ("palm", "palm_sized"), ("short", "short"),
        ("long", "long"), ("large", "large"),
    ))
    resolved_components = component_count or (2 if composite else 1)
    resolved_ends = end_count if end_count is not None else (1 if is_weapon else 0)
    resolved_handles = handle_count if handle_count is not None else (1 if is_weapon else 0)
    topology = "composite" if resolved_components > 1 else "single_object"
    return {
        "topology": _fact(topology, explicit=composite or component_count is not None, source_ref=source_ref),
        "shape_family": _fact(shape or "authored_generic", explicit=shape is not None, source_ref=source_ref),
        "component_count": _fact(resolved_components, explicit=component_count is not None, source_ref=source_ref),
        "active_end_count": _fact(resolved_ends, explicit=end_count is not None, source_ref=source_ref),
        "handle_count": _fact(resolved_handles, explicit=handle_count is not None, source_ref=source_ref),
        "relative_scale": _fact(scale or "character_relative_default", explicit=scale is not None, source_ref=source_ref),
        "material": _fact(material or "authored_generic", explicit=material is not None, source_ref=source_ref),
        "colors": _fact(colors or ["source_unspecified"], explicit=bool(colors), source_ref=source_ref),
        "emissive_regions": _fact(emissive, explicit=bool(emissive), source_ref=source_ref),
        "forbidden_topology_changes": [
            f"topology_must_remain_{topology}",
            f"component_count_must_remain_{resolved_components}",
            f"active_end_count_must_remain_{resolved_ends}",
            f"handle_count_must_remain_{resolved_handles}",
        ],
    }


def _source_policy(character: dict[str, Any]) -> str:
    appearance = character.get("appearance")
    appearance = appearance if isinstance(appearance, dict) else {}
    evidence = " ".join(
        str(value or "")
        for value in (
            appearance.get("gender"), appearance.get("face"),
            appearance.get("summary"), character.get("style"),
        )
    )
    return (
        SYNTHETIC_STYLIZED_POLICY
        if _SYNTHETIC_MARKERS.search(evidence)
        else FICTIONAL_CINEMATIC_HUMAN_POLICY
    )


def apply_character_visual_policy(
    characters_data: dict[str, Any],
    requested_policy: str,
) -> dict[str, Any]:
    """Apply one canonical policy without exposing legacy flags downstream."""
    policy = normalize_character_visual_policy(requested_policy)
    rewritten = copy.deepcopy(characters_data)
    characters = rewritten.get("characters")
    if not isinstance(characters, list):
        raise CanonicalVisualContractError("characters payload has no character list")

    if policy == SYNTHETIC_STYLIZED_POLICY:
        # The existing checked-in aesthetic profile remains the current
        # synthetic implementation.  The legacy boolean selector is not used.
        from utils.privacy_visual_policy import (
            apply_synthetic_stylized_character_policy,
        )

        rewritten = apply_synthetic_stylized_character_policy(rewritten)
        characters = rewritten.get("characters") or []

    resolved_policies: set[str] = set()
    for index, character in enumerate(list(characters)):
        if not isinstance(character, dict):
            raise CanonicalVisualContractError("character entry must be an object")
        resolved = (
            _source_policy(character)
            if policy == SOURCE_DERIVED_POLICY
            else policy
        )
        if (
            policy == SOURCE_DERIVED_POLICY
            and resolved == SYNTHETIC_STYLIZED_POLICY
        ):
            # ``source_derived`` is per-character, not merely a label.  Apply
            # the same current v3 transform to the synthetic subset so mixed
            # human/synthetic casts receive complete, reviewable contracts.
            from utils.privacy_visual_policy import (
                apply_synthetic_stylized_character_policy,
            )

            styled = apply_synthetic_stylized_character_policy(
                {"characters": [character]}
            )
            styled_character = styled.get("characters", [None])[0]
            if not isinstance(styled_character, dict):
                raise CanonicalVisualContractError(
                    "source-derived synthetic styling produced no character"
                )
            characters[index] = styled_character
            character = styled_character
        character["visual_identity_policy"] = resolved
        resolved_policies.add(resolved)
    rewritten["character_visual_policy"] = policy
    rewritten["resolved_character_visual_policies"] = sorted(resolved_policies)
    rewritten.pop("visual_identity_policy", None)
    return rewritten


def build_canonical_visual_contract(
    characters_data: dict[str, Any],
    *,
    requested_policy: str,
) -> dict[str, Any]:
    policy = normalize_character_visual_policy(requested_policy)
    characters = characters_data.get("characters")
    if not isinstance(characters, list):
        raise CanonicalVisualContractError("characters payload has no character list")
    source_hash = canonical_json_sha256(characters)
    roster_hash = str(characters_data.get("character_roster_sha256") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", roster_hash):
        # One-to-one programmatic callers predate the roster artifact.  Their
        # source character hash is a deterministic compatibility roster.  New
        # Phase 1 output always provides the actual roster hash.
        if all(
            isinstance(character, dict)
            and int(character.get("instance_count") or 1) == 1
            for character in characters
        ):
            roster_hash = source_hash
        else:
            raise CanonicalVisualContractError(
                "multi-instance characters require a verified character roster"
            )
    records: list[dict[str, Any]] = []
    for index, character in enumerate(characters, 1):
        if not isinstance(character, dict):
            raise CanonicalVisualContractError("character entry must be an object")
        character_id = str(character.get("id") or f"character_{index:02d}").strip()
        if not character_id:
            raise CanonicalVisualContractError("character ID must not be empty")
        appearance = character.get("appearance")
        appearance = appearance if isinstance(appearance, dict) else {}
        resolved_policy = str(character.get("visual_identity_policy") or "").strip()
        if resolved_policy not in CHARACTER_RUNTIME_POLICIES:
            raise CanonicalVisualContractError(
                f"{character_id} has no current visual identity policy"
            )
        raw_count = character.get("instance_count", 1)
        count = raw_count if isinstance(raw_count, int) and raw_count > 0 else 1
        raw_instances = character.get("instances")
        if raw_instances is None and count == 1:
            raw_instances = [{
                "instance_id": character_id,
                "ordinal": 1,
                "source_mentions": [str(character.get("name") or character_id)],
                "event_refs": [],
                "action_unit_refs": [],
            }]
        if not isinstance(raw_instances, list) or len(raw_instances) != count:
            raise CanonicalVisualContractError(
                f"{character_id} instances do not match instance_count"
            )
        instance_records = []
        for ordinal, instance in enumerate(raw_instances, 1):
            if not isinstance(instance, dict):
                raise CanonicalVisualContractError(
                    f"{character_id} instance must be an object"
                )
            instance_id = str(instance.get("instance_id") or "").strip()
            if not instance_id or int(instance.get("ordinal") or 0) != ordinal:
                raise CanonicalVisualContractError(
                    f"{character_id} instances require stable ordered IDs"
                )
            instance_records.append({
                "instance_id": instance_id,
                "ordinal": ordinal,
                "source_mentions": [
                    str(value).strip()
                    for value in (instance.get("source_mentions") or [])
                    if str(value).strip()
                ],
                "source_event_refs": [
                    str(value).strip()
                    for value in (instance.get("event_refs") or [])
                    if str(value).strip()
                ],
                "source_action_unit_refs": [
                    str(value).strip()
                    for value in (instance.get("action_unit_refs") or [])
                    if str(value).strip()
                ],
                "face_identity": _fact(
                    f"stable_fictional_face_{hashlib.sha256((instance_id + ':face').encode('utf-8')).hexdigest()[:12]}",
                    explicit=False,
                    source_ref=f"instance:{instance_id}:face",
                ),
            })
        props = []
        for prop in normalize_identity_props(appearance.get("identity_props")):
            geometry = _prop_geometry(prop["id"], prop["description"])
            props.append({
                "prop_id": prop["id"],
                "name": prop["name"],
                "geometry": geometry,
            })
            prop["geometry"] = geometry
        if props:
            appearance["identity_props"] = [
                {
                    **prop,
                    "geometry": props[prop_index]["geometry"],
                }
                for prop_index, prop in enumerate(
                    normalize_identity_props(appearance.get("identity_props"))
                )
            ]
        records.append({
            "character_id": character_id,
            "entity_id": str(character.get("entity_id") or character_id),
            "instance_count": _fact(
                count,
                explicit="instance_count" in character,
                source_ref=f"character:{character_id}:instance_count",
            ),
            "instances": instance_records,
            "visual_identity_policy": resolved_policy,
            "hair": _hair_geometry(character_id, appearance.get("hair")),
            "body_build": _fact(
                str(appearance.get("build") or "stable_character_proportions"),
                explicit=bool(str(appearance.get("build") or "").strip()),
                source_ref=f"character:{character_id}:appearance.build",
            ),
            "face": _fact(
                str(appearance.get("face") or "stable_fictional_face_geometry"),
                explicit=bool(str(appearance.get("face") or "").strip()),
                source_ref=f"character:{character_id}:appearance.face",
            ),
            "wardrobe": _fact(
                str(appearance.get("clothing") or "stable_authored_wardrobe"),
                explicit=bool(str(appearance.get("clothing") or "").strip()),
                source_ref=f"character:{character_id}:appearance.clothing",
            ),
            "identity_props": props,
        })
    unsigned = {
        "schema": CANONICAL_VISUAL_CONTRACT_SCHEMA,
        "requested_policy": policy,
        "source_characters_sha256": source_hash,
        "character_roster_sha256": roster_hash,
        "characters": records,
    }
    contract = {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}
    CanonicalVisualContractUnderstanding.model_validate(contract)
    return contract


def _migrate_one_to_one_v1_contract(
    value: dict[str, Any],
    *,
    characters_data: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy = copy.deepcopy(value)
    claimed = str(legacy.pop("contract_sha256", ""))
    if canonical_json_sha256(legacy) != claimed:
        raise CanonicalVisualContractError("legacy canonical visual contract hash mismatch")
    records = legacy.get("characters")
    if not isinstance(records, list):
        raise CanonicalVisualContractError("legacy canonical contract has no characters")
    source_characters = (
        characters_data.get("characters")
        if isinstance(characters_data, dict)
        else None
    )
    if not isinstance(source_characters, list):
        raise CanonicalVisualContractError(
            "legacy canonical contract has no source character lineage; audit-only"
        )
    lineage_by_id = {
        str(character.get("id") or "").strip(): character
        for character in source_characters
        if isinstance(character, dict) and str(character.get("id") or "").strip()
    }
    migrated_records = []
    for record in records:
        if not isinstance(record, dict):
            raise CanonicalVisualContractError("legacy canonical character is invalid")
        character_id = str(record.get("character_id") or "").strip()
        count = record.get("instance_count")
        if (
            not character_id
            or not isinstance(count, dict)
            or count.get("value") != 1
        ):
            raise CanonicalVisualContractError(
                "legacy grouped canonical contract is audit-only; rerun Phase 1"
            )
        source_character = lineage_by_id.get(character_id)
        identity_evidence = (
            source_character.get("source_identity_evidence")
            if isinstance(source_character, dict)
            else None
        )
        source_mentions = (
            [
                str(mention).strip()
                for mention in identity_evidence.get("source_mentions") or []
                if str(mention).strip()
            ]
            if isinstance(identity_evidence, dict)
            else []
        )
        source_event_refs = (
            [
                f"event:{int(event_id)}"
                for event_id in identity_evidence.get("event_ids") or []
                if isinstance(event_id, int) and not isinstance(event_id, bool)
            ]
            if isinstance(identity_evidence, dict)
            else []
        )
        if not source_mentions or not source_event_refs:
            raise CanonicalVisualContractError(
                "legacy canonical character lacks source lineage; audit-only"
            )
        migrated_records.append({
            **record,
            "entity_id": character_id,
            "instances": [{
                "instance_id": character_id,
                "ordinal": 1,
                "source_mentions": source_mentions,
                "source_event_refs": source_event_refs,
                "source_action_unit_refs": [],
                "face_identity": _fact(
                    f"stable_fictional_face_{hashlib.sha256((character_id + ':face').encode('utf-8')).hexdigest()[:12]}",
                    explicit=False,
                    source_ref=f"instance:{character_id}:face",
                ),
            }],
        })
    unsigned = {
        "schema": CANONICAL_VISUAL_CONTRACT_SCHEMA,
        "requested_policy": legacy.get("requested_policy"),
        "source_characters_sha256": legacy.get("source_characters_sha256"),
        "character_roster_sha256": legacy.get("source_characters_sha256"),
        "characters": migrated_records,
    }
    return {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}


def validate_canonical_visual_contract(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") == LEGACY_CANONICAL_VISUAL_CONTRACT_SCHEMA:
        raise CanonicalVisualContractError(
            "legacy canonical contract requires the explicit load migration boundary"
        )
    parsed = CanonicalVisualContractUnderstanding.model_validate(value)
    payload = parsed.model_dump(by_alias=True)
    claimed = payload.pop("contract_sha256")
    if canonical_json_sha256(payload) != claimed:
        raise CanonicalVisualContractError("canonical visual contract hash mismatch")
    payload["contract_sha256"] = claimed
    return payload


def render_canonical_visual_prompt_contract(
    contract: dict[str, Any],
    *,
    character_ids: list[str] | None = None,
) -> str:
    """Render the Phase 1 facts as the first structured Phase 6 authority."""
    validated = validate_canonical_visual_contract(contract)
    filter_requested = character_ids is not None
    requested = {
        str(value).strip()
        for value in (character_ids or [])
        if str(value).strip()
    }
    records = []
    for record in validated["characters"]:
        instance_ids = {
            str(instance["instance_id"])
            for instance in record["instances"]
        }
        if (
            not filter_requested
            or record["character_id"] in requested
            or record["entity_id"] in requested
            or bool(instance_ids & requested)
        ):
            records.append(record)

    def fact(value: dict[str, Any]) -> Any:
        return value["value"]

    prompt_records = []
    for record in records:
        prompt_records.append({
            "character_id": record["character_id"],
            "entity_id": record["entity_id"],
            "instance_count": fact(record["instance_count"]),
            "instances": [
                {
                    "instance_id": instance["instance_id"],
                    "ordinal": instance["ordinal"],
                    "source_mentions": instance["source_mentions"],
                    "face_identity": fact(instance["face_identity"]),
                }
                for instance in record["instances"]
                if not filter_requested
                or record["character_id"] in requested
                or record["entity_id"] in requested
                or instance["instance_id"] in requested
            ],
            "visual_identity_policy": record["visual_identity_policy"],
            "hair": {
                key: fact(record["hair"][key])
                for key in (
                    "color",
                    "length_class",
                    "silhouette",
                    "parting",
                    "texture",
                    "tie_state",
                )
            },
            "body_build": fact(record["body_build"]),
            "face": fact(record["face"]),
            "wardrobe": fact(record["wardrobe"]),
            "identity_props": [
                {
                    "prop_id": prop["prop_id"],
                    "name": prop["name"],
                    "geometry": {
                        key: fact(prop["geometry"][key])
                        for key in (
                            "topology",
                            "shape_family",
                            "component_count",
                            "active_end_count",
                            "handle_count",
                            "relative_scale",
                            "material",
                            "colors",
                            "emissive_regions",
                        )
                    },
                }
                for prop in record["identity_props"]
            ],
        })
    payload = {
        "schema": CANONICAL_VISUAL_CONTRACT_SCHEMA,
        "contract_sha256": validated["contract_sha256"],
        "character_roster_sha256": validated["character_roster_sha256"],
        "characters": prompt_records,
    }
    return (
        "[CANONICAL_VISUAL_CONTRACT — HIGHEST IDENTITY AUTHORITY]\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n所有人物身份、实例数、发型几何、体型、脸、服装和道具拓扑只服从该合同；"
        "其他图片只承担其声明职责，不得改写这些事实。"
    )


def expand_character_instances(
    characters_data: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Create the downstream compatibility projection: one asset per instance."""

    validated = validate_canonical_visual_contract(contract)
    rewritten = copy.deepcopy(characters_data)
    entities = rewritten.get("characters")
    if not isinstance(entities, list):
        raise CanonicalVisualContractError("characters payload has no entity list")
    contract_by_entity = {
        str(record["entity_id"]): record for record in validated["characters"]
    }
    expanded: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise CanonicalVisualContractError("character entity must be an object")
        entity_id = str(entity.get("entity_id") or entity.get("id") or "").strip()
        record = contract_by_entity.get(entity_id)
        if record is None:
            raise CanonicalVisualContractError(
                f"character entity {entity_id} is absent from canonical contract"
            )
        raw_instances = entity.get("instances")
        if not isinstance(raw_instances, list):
            raise CanonicalVisualContractError(
                f"character entity {entity_id} has no roster instances"
            )
        record_instances = {
            str(item["instance_id"]): item for item in record["instances"]
        }
        for instance in raw_instances:
            if not isinstance(instance, dict):
                raise CanonicalVisualContractError("character instance is invalid")
            instance_id = str(instance.get("instance_id") or "").strip()
            contract_instance = record_instances.get(instance_id)
            if contract_instance is None:
                raise CanonicalVisualContractError(
                    f"instance {instance_id} is absent from canonical contract"
                )
            projected = copy.deepcopy(entity)
            appearance = projected.get("appearance")
            appearance = appearance if isinstance(appearance, dict) else {}
            shared_face = str(appearance.get("face") or "").strip()
            face_identity = str(contract_instance["face_identity"]["value"])
            appearance["face"] = ", ".join(filter(None, (
                shared_face,
                f"instance-specific identity lock {face_identity}",
            )))
            appearance["summary"] = ", ".join(filter(None, (
                str(appearance.get("summary") or "").strip(),
                f"unique instance identity {face_identity}",
            )))
            source_mentions = [
                str(value).strip()
                for value in (instance.get("source_mentions") or [])
                if str(value).strip()
            ]
            projected.update({
                "id": instance_id,
                "entity_id": entity_id,
                "instance_id": instance_id,
                "instance_ordinal": int(instance.get("ordinal") or 0),
                "entity_instance_count": int(entity.get("instance_count") or 1),
                "name": source_mentions[0] if source_mentions else str(entity.get("name") or entity_id),
                "aliases": source_mentions[1:],
                "appearance": appearance,
                "asset_path": f"characters/{instance_id}/",
                "canonical_visual_contract": {
                    "artifact": CANONICAL_VISUAL_CONTRACT_FILENAME,
                    "contract_sha256": validated["contract_sha256"],
                    "character_id": entity_id,
                    "entity_id": entity_id,
                    "instance_id": instance_id,
                },
            })
            expanded.append(projected)
    rewritten["entities"] = entities
    rewritten["characters"] = expanded
    rewritten["total_character_entities"] = len(entities)
    rewritten["total_characters"] = len(expanded)
    rewritten["total_character_instances"] = len(expanded)
    return rewritten


def persist_canonical_visual_contract(
    output_dir: str | Path,
    characters_data: dict[str, Any],
    *,
    requested_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rewritten = apply_character_visual_policy(characters_data, requested_policy)
    contract = build_canonical_visual_contract(
        rewritten,
        requested_policy=requested_policy,
    )
    contract_hash = contract["contract_sha256"]
    for character in rewritten["characters"]:
        character["canonical_visual_contract"] = {
            "artifact": CANONICAL_VISUAL_CONTRACT_FILENAME,
            "contract_sha256": contract_hash,
            "character_id": character["id"],
        }
    rewritten["canonical_visual_contract"] = CANONICAL_VISUAL_CONTRACT_FILENAME
    rewritten["canonical_visual_contract_sha256"] = contract_hash
    rewritten["character_roster_sha256"] = contract["character_roster_sha256"]
    path = root / CANONICAL_VISUAL_CONTRACT_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return rewritten, contract


def load_canonical_visual_contract(
    output_dir: str | Path,
    *,
    characters_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one current contract and bind it to the current character artifact."""
    path = Path(output_dir) / CANONICAL_VISUAL_CONTRACT_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalVisualContractError(
            "canonical visual contract is missing or invalid; rerun Phase 1"
        ) from error
    if not isinstance(raw, dict):
        raise CanonicalVisualContractError("canonical visual contract must be an object")
    migrated_from_v1 = (
        raw.get("schema") == LEGACY_CANONICAL_VISUAL_CONTRACT_SCHEMA
    )
    if migrated_from_v1:
        source_sha256 = canonical_json_sha256(raw)
        receipt_path = (
            Path(output_dir) / CANONICAL_VISUAL_MIGRATION_RECEIPT_FILENAME
        )
        try:
            migrated = _migrate_one_to_one_v1_contract(
                raw,
                characters_data=characters_data,
            )
        except CanonicalVisualContractError as error:
            _persist_visual_migration_receipt(
                receipt_path,
                {
                    "schema": CANONICAL_VISUAL_MIGRATION_RECEIPT_SCHEMA,
                    "status": "audit_only",
                    "source_schema": LEGACY_CANONICAL_VISUAL_CONTRACT_SCHEMA,
                    "source_document_sha256": source_sha256,
                    "reason": str(error),
                    "provider_request_count": 0,
                },
            )
            raise
        contract = validate_canonical_visual_contract(migrated)
        _persist_visual_migration_receipt(
            receipt_path,
            {
                "schema": CANONICAL_VISUAL_MIGRATION_RECEIPT_SCHEMA,
                "status": "migrated",
                "source_schema": LEGACY_CANONICAL_VISUAL_CONTRACT_SCHEMA,
                "source_document_sha256": source_sha256,
                "migrated_schema": CANONICAL_VISUAL_CONTRACT_SCHEMA,
                "migrated_contract_sha256": contract["contract_sha256"],
                "provider_request_count": 0,
            },
        )
    else:
        contract = validate_canonical_visual_contract(raw)
    if characters_data is not None:
        claimed = str(
            characters_data.get("canonical_visual_contract_sha256") or ""
        )
        allowed_contract_claims = {contract["contract_sha256"]}
        if migrated_from_v1:
            allowed_contract_claims.add(str(raw.get("contract_sha256") or ""))
        if claimed not in allowed_contract_claims:
            raise CanonicalVisualContractError(
                "CHARACTERS.json canonical visual contract hash mismatch"
            )
        roster_claim = str(characters_data.get("character_roster_sha256") or "")
        allowed_roster_claims = {contract["character_roster_sha256"]}
        if migrated_from_v1:
            allowed_roster_claims.add(str(raw.get("source_characters_sha256") or ""))
        if roster_claim not in allowed_roster_claims:
            raise CanonicalVisualContractError(
                "CHARACTERS.json character roster hash mismatch"
            )
        raw_characters = characters_data.get("characters")
        if not isinstance(raw_characters, list):
            raise CanonicalVisualContractError("CHARACTERS.json has no character list")
        expected = {
            record["character_id"]: record["visual_identity_policy"]
            for record in contract["characters"]
        }
        actual = {
            str(character.get("entity_id") or character.get("id") or ""): str(
                character.get("visual_identity_policy") or ""
            )
            for character in raw_characters
            if isinstance(character, dict)
        }
        if actual != expected:
            raise CanonicalVisualContractError(
                "CHARACTERS.json identity policies disagree with canonical contract"
            )
    return contract


def _persist_visual_migration_receipt(
    path: Path,
    receipt: dict[str, Any],
) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CanonicalVisualContractError(
                "canonical visual migration receipt is invalid"
            ) from error
        if existing != receipt:
            raise CanonicalVisualContractError(
                "canonical visual migration receipt conflicts with source evidence"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "CANONICAL_VISUAL_CONTRACT_FILENAME",
    "CANONICAL_VISUAL_MIGRATION_RECEIPT_FILENAME",
    "CANONICAL_VISUAL_MIGRATION_RECEIPT_SCHEMA",
    "CANONICAL_VISUAL_CONTRACT_SCHEMA",
    "CHARACTER_VISUAL_POLICIES",
    "FICTIONAL_CINEMATIC_HUMAN_POLICY",
    "SOURCE_DERIVED_POLICY",
    "SYNTHETIC_STYLIZED_POLICY",
    "CanonicalVisualContractError",
    "apply_character_visual_policy",
    "build_canonical_visual_contract",
    "expand_character_instances",
    "load_canonical_visual_contract",
    "normalize_character_visual_policy",
    "persist_canonical_visual_contract",
    "render_canonical_visual_prompt_contract",
    "validate_canonical_visual_contract",
]
