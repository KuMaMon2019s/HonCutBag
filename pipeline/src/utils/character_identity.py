"""Shared, conservative character identity resolution.

Character references cross several pipeline boundaries: screenplay ``who``
labels, canonical names, asset ids, and aliases.  Keep their matching rules in
one place so quality review and adaptation cannot silently disagree.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

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

HumanGender = Literal["male", "female"]
HumanReferenceKind = Literal["exact", "qualified"]


@dataclass(frozen=True)
class HumanReferenceDescriptor:
    """Machine meaning of one source-level human descriptor."""

    gender: HumanGender
    kind: HumanReferenceKind
    base_label: str
    qualifier: str

# Source prose may alternate between a gender adjective and a referential noun.
# Keep those presentation labels language-aware while reducing their machine
# meaning to one controlled English enum.  These labels are never character
# identities by themselves; callers may reconcile them only when a unique
# canonical identity is independently anchored.
_HUMAN_GENDER_DESCRIPTOR_GROUPS: dict[HumanGender, frozenset[str]] = {
    "male": frozenset({"男子", "男性", "男人", "man", "male"}),
    "female": frozenset({"女子", "女性", "女人", "woman", "female"}),
}
_HUMAN_GENDER_ATTRIBUTE_REFERENCES = frozenset({
    "男性",
    "女性",
    "male",
    "female",
})

# These are identity categories, not screenplay participants.  Event ``who``
# may legitimately contain a vehicle, weather system, prop, or environment
# object because it owns an action.  Such a participant must remain in the
# source event while staying out of the character-asset roster.
_INTELLIGENT_CHARACTER_MARKERS = (
    "无人机", "机器人", "机械臂", "传感器", "合成人", "复制体",
    "android", "robot", "drone", "synthetic being",
)
_CHARACTER_ROLE_SUFFIXES = (
    "机器人", "号", "型", "级", "者", "员", "师", "家", "王", "后",
    "公主", "王子", "先生", "小姐", "佣兵", "机械体", "合成人", "复制体",
    "复制品", "生命体", "机甲", "战士", "卫兵", "执法体", "仙女", "女子",
    "少女", "女人", "男子", "男人", "女孩", "男孩", "姑娘", "妇人",
    "夫人", "老者", "枪手", "剑客", "车手",
)
_NON_CHARACTER_EXACT_REFERENCES = frozenset({
    "冷空气", "风", "雨", "雪", "雷", "电", "鸡", "鸭", "狗", "猫",
    "鸟", "鱼", "桌子", "椅子", "车", "书", "刀", "剑", "枪", "武器",
    "积水", "路面", "钢梁", "混凝土", "高楼", "玻璃", "灰尘",
})
_NON_CHARACTER_SUFFIXES = (
    "霓虹牌", "电缆", "路面", "钢梁", "混凝土", "高楼", "塑料布",
    "纸屑", "玻璃", "水浪", "灰尘", "碎石", "残骸", "护臂", "手掌",
    "手指", "车门", "刀具", "武器", "护甲", "铠甲", "弓箭", "云海",
    "山脉", "河流", "道路", "车辆", "建筑", "房间", "走廊", "列车",
    "火车", "汽车", "卡车", "电车", "地铁", "飞船", "飞机", "舰船",
    "轮船", "船只", "train", "vehicle", "car", "truck", "tram",
    "subway", "aircraft", "spaceship", "ship",
)
_ABSTRACT_IDENTITY_REFERENCES = frozenset({
    "说话者", "观察者", "记录者", "思考者", "行走者", "试验者", "打探人员",
})
_EXPLICIT_IDENTITY_MARKER = re.compile(
    r"(?:代号|化名|名为|名叫|昵称(?:是|为)?|codename|alias(?:ed)?(?:\s+as)?)",
    re.IGNORECASE,
)


def normalize_character_reference(value: Any) -> str:
    """Normalize presentation differences without erasing word boundaries."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[\s·•_\-]+", "", text)


def character_identity_is_explicitly_declared(
    reference: Any,
    evidence: Any,
) -> bool:
    """Return whether source prose explicitly promotes a label to identity."""

    label = str(reference or "").strip()
    text = str(evidence or "")
    if not label or not text or not _EXPLICIT_IDENTITY_MARKER.search(text):
        return False
    marker_then_label = re.compile(
        rf"{_EXPLICIT_IDENTITY_MARKER.pattern}\s*[：:]?\s*"
        rf"[\"'“”‘’「」『』]?{re.escape(label)}[\"'“”‘’「」『』]?",
        re.IGNORECASE,
    )
    return bool(marker_then_label.search(text))


def is_character_identity_candidate(value: Any) -> bool:
    """Classify whether a source participant requires a character asset.

    This intentionally answers only clear identity-vs-object cases.  Unknown
    labels remain candidates and are resolved fail-closed by Character Roster;
    clear vehicles and environment objects remain event facts without becoming
    faces, identity boards, or Phase 6 character references.
    """

    label = str(value or "").strip()
    if not label or label.endswith("们"):
        return False
    lowered = label.casefold()
    if any(marker in lowered for marker in _INTELLIGENT_CHARACTER_MARKERS):
        return True
    if label in _ABSTRACT_IDENTITY_REFERENCES:
        return False
    if any(label.endswith(suffix) for suffix in _CHARACTER_ROLE_SUFFIXES):
        return True
    if label in _NON_CHARACTER_EXACT_REFERENCES:
        return False
    for suffix in _NON_CHARACTER_SUFFIXES:
        if re.fullmatch(r"[a-z ]+", suffix):
            if re.search(rf"(?:^|[\s_-]){re.escape(suffix)}$", lowered):
                return False
        elif lowered.endswith(suffix):
            return False
    return True


def human_gender_descriptor(value: Any) -> HumanGender | None:
    """Return the controlled gender meaning of one exact source descriptor."""

    key = normalize_character_reference(value)
    for gender, labels in _HUMAN_GENDER_DESCRIPTOR_GROUPS.items():
        if key in labels:
            return gender
    return None


def parse_human_reference_descriptor(
    value: Any,
) -> HumanReferenceDescriptor | None:
    """Parse exact or qualified human descriptors without inventing identity.

    The result is only presentation semantics.  A caller still needs sequence,
    source, co-occurrence, and uniqueness evidence before treating two labels as
    one person.
    """

    raw = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    if not raw:
        return None
    exact_gender = human_gender_descriptor(raw)
    if exact_gender is not None:
        return HumanReferenceDescriptor(
            gender=exact_gender,
            kind="exact",
            base_label=normalize_character_reference(raw),
            qualifier="",
        )

    if re.search(r"[\u3400-\u9fff]", raw):
        key = normalize_character_reference(raw)
        candidates: list[HumanReferenceDescriptor] = []
        for gender, labels in _HUMAN_GENDER_DESCRIPTOR_GROUPS.items():
            for label in labels:
                label_key = normalize_character_reference(label)
                if not re.search(r"[\u3400-\u9fff]", label) or not key.endswith(label_key):
                    continue
                qualifier = key[:-len(label_key)]
                if qualifier:
                    candidates.append(HumanReferenceDescriptor(
                        gender=gender,
                        kind="qualified",
                        base_label=label_key,
                        qualifier=qualifier,
                    ))
        return max(candidates, key=lambda item: len(item.base_label), default=None)

    tokens = _latin_tokens(raw)
    if len(tokens) < 2:
        return None
    base = tokens[-1]
    for gender, labels in _HUMAN_GENDER_DESCRIPTOR_GROUPS.items():
        if base in labels:
            return HumanReferenceDescriptor(
                gender=gender,
                kind="qualified",
                base_label=base,
                qualifier=" ".join(tokens[:-1]),
            )
    return None


def compatible_human_reference_descriptors(left: Any, right: Any) -> bool:
    """Return whether two labels may be the same human presentation.

    Two qualified descriptions are deliberately not compatible: without one
    exact source descriptor, merging them would turn shared gender into identity.
    """

    left_descriptor = parse_human_reference_descriptor(left)
    right_descriptor = parse_human_reference_descriptor(right)
    return bool(
        left_descriptor
        and right_descriptor
        and left_descriptor.gender == right_descriptor.gender
        and (
            left_descriptor.kind == "exact"
            or right_descriptor.kind == "exact"
        )
    )


def character_reference_is_explicit(reference: Any, evidence: Any) -> bool:
    """Match one complete source reference in bounded source evidence."""

    label = str(reference or "").strip()
    text = str(evidence or "")
    if not label or not text:
        return False
    if re.search(r"[\u3400-\u9fff]", label):
        return normalize_character_reference(label) in normalize_character_reference(text)
    tokens = _latin_tokens(label)
    evidence_tokens = _latin_tokens(text)
    if not tokens or len(tokens) > len(evidence_tokens):
        return False
    width = len(tokens)
    return any(
        evidence_tokens[index:index + width] == tokens
        for index in range(len(evidence_tokens) - width + 1)
    )


def is_gender_attribute_reference(value: Any) -> bool:
    """Return whether ``value`` is an attribute label, not a stable identity."""

    return normalize_character_reference(value) in _HUMAN_GENDER_ATTRIBUTE_REFERENCES


def equivalent_human_descriptors(left: Any, right: Any) -> bool:
    """Compare exact human descriptors through controlled machine semantics."""

    left_gender = human_gender_descriptor(left)
    return left_gender is not None and left_gender == human_gender_descriptor(right)


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
