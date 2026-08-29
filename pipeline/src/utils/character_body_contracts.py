"""Deterministic body-proportion contracts for adult lead characters.

The contract is stored as structured data in ``CHARACTERS.json`` and rendered
through the helpers in this module.  Image, storyboard, QA, and video prompts
must not maintain private copies of these values.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from utils.character_reference_contracts import (
    STATIC_REFERENCE_ASSET_POLICY,
    static_reference_identity_text,
)

BODY_CONTRACT_SCHEMA_VERSION = 2

COMMON_ADULT_HUMAN_PROPORTION_CONTRACT: dict[str, Any] = {
    "anatomy": "realistic adult human proportions",
    "head_scale": "slightly small and natural head size",
    "head_to_body_ratio_range": [7.6, 8.0],
    "max_head_width_to_shoulder_width": 0.43,
    "shoulder_head_relation": "shoulders visibly wider than the head",
    "neck": "natural neck length",
    "clavicle_and_shoulder_line": "clear clavicle and shoulder line",
    "torso_pelvis_legs": "realistic torso, pelvis, and leg proportions",
    "leg_length": "long legs without exaggeration",
    "extremity_scale": "hands and feet proportionate to height",
    "cross_shot_consistency": (
        "same head size and body proportions for the same character in every shot"
    ),
    "forbidden": [
        "large head on small body",
        "childlike body proportions",
        "bobblehead proportions",
        "large face",
        "short neck",
        "narrow shoulders",
    ],
}

ADULT_LEAD_BODY_CONTRACTS: dict[str, dict[str, Any]] = {
    "male": {
        "profile": "adult_male_lead",
        "height_cm": 182,
        "head_to_body_ratio": 7.8,
        "build": "lean athletic",
        "shoulders": "moderately broad shoulders",
        "leg_proportion": "slightly long legs",
        "body_fat": "low-to-normal body fat",
        "posture": "upright, confident",
        "forbidden": [
            "oversized head",
            "extremely narrow waist",
            "bodybuilder physique",
        ],
    },
    "female": {
        "profile": "adult_female_lead",
        "height_cm": 166,
        "head_to_body_ratio": 7.6,
        "build": "slender balanced",
        "shoulders_and_hips": "natural proportional shoulders and hips",
        "leg_proportion": "slightly long legs",
        "waistline": "naturally defined waist",
        "body_fat": "healthy slim",
        "forbidden": [
            "oversized head",
            "extremely tiny waist",
            "exaggerated curves",
        ],
    },
}

ADULT_LEAD_DISCOVERY_INSTRUCTIONS = """
【成年主角身体比例硬合同】
- 只对已明确为成年人且 role=protagonist 的男主、女主应用；未成年人、配角和性别/年龄不明者不得套用。
- 成年男主：height=182cm；head_to_body_ratio=7.8；build=lean athletic；
  shoulders=moderately broad shoulders；leg_proportion=slightly long legs；
  body_fat=low-to-normal body fat；posture=upright, confident；
  禁止 oversized head、extremely narrow waist、bodybuilder physique。
- 成年女主：height=166cm；head_to_body_ratio=7.6；build=slender balanced；
  shoulders_and_hips=natural proportional shoulders and hips；
  leg_proportion=slightly long legs；waistline=naturally defined waist；
  body_fat=healthy slim；
  禁止 oversized head、extremely tiny waist、exaggerated curves。
- appearance.summary 不得写入与上述合同冲突的身高、头身比、体型、肩胯、腰线或体脂描述。
- 所有成年主角共同遵守：成年真人比例；头部尺寸偏小且自然；头身比必须在 7.6–8.0；
  头宽不得超过肩宽的 43%，肩部必须明显宽于头部；颈长自然，锁骨与肩线清晰；
  躯干、骨盆、腿部比例真实，腿长但不夸张，手掌和脚掌与身高匹配；同一角色所有镜头
  保持相同头部尺寸和身体比例。
- 共同禁止：大头身小、幼态头身比例、large face、short neck、narrow shoulders、
  childlike body proportions、bobblehead proportions。
- appearance 中增加 body_contract 对象，逐字段使用上述英文值和 forbidden 数组；不要自行改写数值或同义替换。
""".strip()

_LEAD_LABELS = {"男主", "女主", "male lead", "female lead"}
_NON_ADULT_MARKERS = (
    "未成年",
    "儿童",
    "小孩",
    "孩子",
    "男孩",
    "女孩",
    "少年",
    "少女",
    "child",
    "kid",
    "teen",
    "minor",
    "boy",
    "girl",
)
_ADULT_MARKERS = ("成年人", "成年", "adult", "grown-up", "grown up")


def _normalized_gender(value: object) -> str | None:
    normalized = str(value or "").strip().casefold()
    if normalized in {"male", "man", "男", "男性"}:
        return "male"
    if normalized in {"female", "woman", "女", "女性"}:
        return "female"
    return None


def _is_adult(appearance: dict[str, Any]) -> bool:
    age_text = str(appearance.get("age_range") or "").strip().casefold()
    if any(marker in age_text for marker in _NON_ADULT_MARKERS):
        return False
    if any(marker in age_text for marker in _ADULT_MARKERS):
        return True
    ages = [int(value) for value in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", age_text)]
    # A range is adult only when its youngest possible age is adult.  Unknown
    # or ambiguous ages fail closed instead of assigning an adult body preset.
    return bool(ages) and min(ages) >= 18


def _is_lead(character: dict[str, Any]) -> bool:
    if str(character.get("role") or "").strip().casefold() == "protagonist":
        return True
    labels = {
        str(character.get("name") or "").strip().casefold(),
        *(str(value).strip().casefold() for value in character.get("aliases", [])),
    }
    return bool(labels.intersection(_LEAD_LABELS))


def apply_adult_lead_body_contracts(
    characters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach one deterministic contract to the primary adult lead per gender.

    ``characters`` is expected to be sorted by narrative prominence.  Applying
    at most one preset per gender prevents a cast of adult supporting characters
    from being homogenized into copies of the leads.
    """
    assigned: set[str] = set()
    for character in characters:
        if not isinstance(character, dict) or not _is_lead(character):
            continue
        appearance = character.get("appearance")
        if not isinstance(appearance, dict) or not _is_adult(appearance):
            continue
        gender = _normalized_gender(appearance.get("gender"))
        if not gender or gender in assigned:
            continue

        contract = copy.deepcopy(ADULT_LEAD_BODY_CONTRACTS[gender])
        contract["human_proportion_constraints"] = copy.deepcopy(
            COMMON_ADULT_HUMAN_PROPORTION_CONTRACT
        )
        contract["schema_version"] = BODY_CONTRACT_SCHEMA_VERSION
        appearance["body_contract"] = contract
        # Keep legacy consumers that only read appearance.height/build aligned
        # with the structured contract.
        appearance["height"] = f"{contract['height_cm']}cm"
        appearance["build"] = contract["build"]
        assigned.add(gender)
    return characters


def body_contract_prompt(character: dict[str, Any]) -> str:
    """Render a complete positive body lock without dropping optional fields."""
    appearance = character.get("appearance")
    if not isinstance(appearance, dict):
        return ""
    contract = appearance.get("body_contract")
    if not isinstance(contract, dict):
        return ""

    details = [
        f"height exactly {contract.get('height_cm')} cm",
        f"exactly {contract.get('head_to_body_ratio')} heads tall",
        f"{contract.get('build')} build",
    ]
    for key in (
        "shoulders",
        "shoulders_and_hips",
        "leg_proportion",
        "waistline",
        "body_fat",
        "posture",
    ):
        value = str(contract.get(key) or "").strip()
        if value:
            details.append(value)
    common = contract.get("human_proportion_constraints")
    if isinstance(common, dict):
        ratio = common.get("head_to_body_ratio_range") or []
        ratio_text = (
            f"adult head-to-body ratio stays within {ratio[0]}–{ratio[1]}"
            if isinstance(ratio, list) and len(ratio) == 2
            else ""
        )
        max_width = common.get("max_head_width_to_shoulder_width")
        common_details = [
            common.get("anatomy"),
            common.get("head_scale"),
            ratio_text,
            (
                f"head width never exceeds {float(max_width) * 100:g}% of shoulder width"
                if isinstance(max_width, (int, float))
                else ""
            ),
            common.get("shoulder_head_relation"),
            common.get("neck"),
            common.get("clavicle_and_shoulder_line"),
            common.get("torso_pelvis_legs"),
            common.get("leg_length"),
            common.get("extremity_scale"),
            common.get("cross_shot_consistency"),
        ]
        details.extend(str(value).strip() for value in common_details if value)
    forbidden = body_contract_forbidden(character)
    negative_clause = f" Do not depict: {', '.join(forbidden)}." if forbidden else ""
    return (
        "Body-proportion lock: "
        + "; ".join(details)
        + ". Preserve this apparent height, silhouette, head scale, and limb proportions "
        "in full-body, side, and back views and across every shot." + negative_clause
    )


def body_contract_forbidden(character: dict[str, Any]) -> list[str]:
    """Return normalized negative traits from a structured body contract."""
    appearance = character.get("appearance")
    contract = appearance.get("body_contract") if isinstance(appearance, dict) else None
    if not isinstance(contract, dict):
        return []
    forbidden = contract.get("forbidden")
    forbidden = forbidden if isinstance(forbidden, list) else []
    common = contract.get("human_proportion_constraints")
    common_forbidden = common.get("forbidden") if isinstance(common, dict) else []
    common_forbidden = common_forbidden if isinstance(common_forbidden, list) else []
    return list(
        dict.fromkeys(
            str(value).strip() for value in [*forbidden, *common_forbidden] if str(value).strip()
        )
    )


def _reference_body_contract_prompt(character: dict[str, Any]) -> str:
    """Render the same structured body facts in a compact Phase 3 form."""
    appearance = character.get("appearance")
    contract = appearance.get("body_contract") if isinstance(appearance, dict) else None
    if not isinstance(contract, dict):
        return ""
    details = [
        f"height exactly {contract.get('height_cm')} cm",
        f"exactly {contract.get('head_to_body_ratio')} heads tall",
        f"{contract.get('build')} build",
    ]
    for key in (
        "shoulders",
        "shoulders_and_hips",
        "leg_proportion",
        "waistline",
        "body_fat",
        "posture",
    ):
        value = str(contract.get(key) or "").strip()
        if value:
            details.append(value)
    common = contract.get("human_proportion_constraints")
    if isinstance(common, dict):
        max_width = common.get("max_head_width_to_shoulder_width")
        details.extend(
            str(value).strip()
            for value in (
                common.get("anatomy"),
                common.get("shoulder_head_relation"),
                (
                    f"head width at most {float(max_width) * 100:g}% of shoulder width"
                    if isinstance(max_width, (int, float))
                    else ""
                ),
                common.get("extremity_scale"),
            )
            if value
        )
    forbidden = body_contract_forbidden(character)
    forbidden_clause = f" Avoid: {', '.join(forbidden)}." if forbidden else ""
    return (
        "Phase 3 body lock: "
        + "; ".join(details)
        + "; identical in front, side and back views."
        + forbidden_clause
    )


def character_visual_description(
    character: dict[str, Any],
    fallback: str = "",
) -> str:
    """Combine the authored appearance summary with its deterministic body lock."""
    appearance = character.get("appearance")
    if isinstance(appearance, dict):
        summary = str(
            appearance.get("summary")
            or appearance.get("description")
            or fallback
            or character.get("description")
            or character.get("visual_description")
            or ""
        ).strip()
    else:
        summary = str(appearance or fallback or character.get("description") or "").strip()
    contract = body_contract_prompt(character)
    if contract and summary:
        return (
            f"{contract} Authored face, hair, and costume details: {summary}. "
            "The body-proportion lock has priority over any conflicting body wording."
        )
    return contract or summary


def character_reference_identity_description(
    character: dict[str, Any],
    fallback: str = "",
) -> str:
    """Render static identity facts for neutral Phase 3 reference images.

    Story summaries and ``distinguishing`` often contain actions, poses, camera
    interaction, or locations.  Those facts belong in shot prompts, not in the
    canonical character pack where they would contaminate every view.
    """
    appearance = character.get("appearance")
    if not isinstance(appearance, dict):
        identity = static_reference_identity_text(
            fallback or character.get("description") or ""
        )
        return " ".join(part for part in (identity, STATIC_REFERENCE_ASSET_POLICY) if part)

    labels = {
        "age_range": "age",
        "gender": "gender",
        "height": "apparent height",
        "build": "build",
        "hair": "hair",
        "face": "face",
        "clothing": "clothing and static accessories",
    }
    synthetic_styling = appearance.get("synthetic_styling")
    compact_synthetic_face = bool(
        isinstance(synthetic_styling, dict)
        and synthetic_styling.get("schema") == "honcut.synthetic-styling.v3"
        and synthetic_styling.get("mode") == "synthetic_porcelain_makeup"
    )
    static_details = []
    for key, label in labels.items():
        # Phase 3 receives the exact synthetic face material and three visual
        # anchors through its verified, high-priority aesthetic profile.  Do
        # not repeat the long prose face summary and bury those exact facts.
        if key == "face" and compact_synthetic_face:
            continue
        value = str(appearance.get(key) or "").strip()
        if key == "clothing":
            value = static_reference_identity_text(value)
        if value:
            static_details.append(f"{label}: {value}")
    contract = _reference_body_contract_prompt(character)
    parts = [
        contract,
        "Static authored identity: " + "; ".join(static_details)
        if static_details
        else "",
        STATIC_REFERENCE_ASSET_POLICY,
    ]
    result = " ".join(part for part in parts if part).strip()
    return result or str(fallback or character.get("description") or "").strip()
