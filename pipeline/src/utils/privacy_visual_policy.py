"""Project-level visual identity policy for privacy-safe synthetic characters."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

NO_REAL_PERSON_ENV = "HONCUT_NO_REAL_PERSON"
NO_REAL_PERSON_POLICY = "synthetic_stylized_character_v2"
LEGACY_NO_REAL_PERSON_POLICIES = frozenset({"synthetic_faceless_android_v1"})
SUPPORTED_NO_REAL_PERSON_POLICIES = frozenset(
    {NO_REAL_PERSON_POLICY, *LEGACY_NO_REAL_PERSON_POLICIES}
)
SYNTHETIC_QA_CONTRACT = "synthetic_character_styling_consistency_v2"

SYNTHETIC_STYLE_CONTRACT = (
    "高成本风格化三维 CGI 动画，所有角色都是完全虚构的数字角色；"
    "每个角色必须拥有独立且逐镜持久的非真人妆造组合，例如面纱/遮罩、图形化妆、面部纹样、"
    "机械拼接纹理、瓷质或晶体合成皮肤、非人眼部设计与设计化发丝；"
    "每人至少保留两个可见妆造锚点和一种明确非人材质，不要求也禁止默认给所有角色套同一种头盔；"
    "材质、轮廓与面部设计明确属于数字角色，不是真人实拍，不模仿任何现实人物"
)

SYNTHETIC_NEGATIVE_CONTRACT = (
    "真人，真人实拍，写真人脸，未经妆造的自然人脸，照片级人类皮肤，自然人类眼睛，"
    "自然裸露皮肤，普通真人发丝，名人，现实人物，身份证照片，肖像摄影，换脸，live-action，"
    "photorealistic human，real person，natural human skin，所有角色同款头盔，统一面甲"
)


def is_synthetic_visual_identity_policy(value: Any) -> bool:
    """Accept the current diverse styling policy and durable legacy artifacts."""
    return str(value or "").strip() in SUPPORTED_NO_REAL_PERSON_POLICIES


def is_no_real_person_enabled() -> bool:
    """Return whether the current pipeline process requires synthetic identities."""
    return os.environ.get(NO_REAL_PERSON_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def synthetic_character_review_evidence(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return auditable runtime/artifact evidence for synthetic-character QA."""
    env_enabled = is_no_real_person_enabled()
    evidence: dict[str, Any] = {
        "enabled": env_enabled,
        "sources": ([f"environment:{NO_REAL_PERSON_ENV}"] if env_enabled else []),
        "policy": NO_REAL_PERSON_POLICY,
        "artifact": "CHARACTERS.json",
        "artifact_present": False,
        "artifact_valid": False,
        "top_level_policy_match": False,
        "all_characters_policy_tagged": False,
        "all_characters_gender_synthetic": False,
        "identity_contract_complete": False,
        "characters": [],
    }
    if output_dir is None:
        return evidence
    artifact_path = Path(output_dir) / "CHARACTERS.json"
    evidence["artifact_present"] = artifact_path.is_file()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return evidence
    if not isinstance(payload, dict):
        return evidence
    raw_characters = payload.get("characters")
    if not isinstance(raw_characters, list):
        return evidence

    characters: list[dict[str, Any]] = []
    for character in raw_characters:
        if not isinstance(character, dict):
            continue
        appearance = character.get("appearance")
        if not isinstance(appearance, dict):
            appearance = {}
        gender = str(appearance.get("gender") or character.get("gender") or "").strip()
        features = character.get("distinguishing_features")
        if not isinstance(features, list):
            features = []
        aliases = character.get("aliases")
        aliases = aliases if isinstance(aliases, list) else []
        characters.append({
            "id": str(character.get("id") or "").strip(),
            "name": str(character.get("name") or "").strip(),
            "aliases": [
                str(value).strip()
                for value in aliases
                if str(value).strip()
            ],
            "gender": gender,
            "visual_identity_policy": str(
                character.get("visual_identity_policy") or ""
            ).strip(),
            "face_styling": str(appearance.get("face") or "").strip(),
            # Compatibility alias for older report consumers.
            "helmet_or_face": str(appearance.get("face") or "").strip(),
            "clothing": str(appearance.get("clothing") or "").strip(),
            "distinguishing": str(appearance.get("distinguishing") or "").strip(),
            "summary": str(appearance.get("summary") or "").strip(),
            "distinguishing_features": [
                str(value).strip() for value in features if str(value).strip()
            ],
            "identity_props": (
                appearance.get("identity_props", [])
                if isinstance(appearance.get("identity_props"), list)
                else []
            ),
            "synthetic_styling": (
                appearance.get("synthetic_styling")
                if isinstance(appearance.get("synthetic_styling"), dict)
                else {}
            ),
        })

    evidence["artifact_valid"] = True
    evidence["characters"] = characters
    top_level_match = is_synthetic_visual_identity_policy(
        payload.get("visual_identity_policy")
    )
    all_policy_tagged = bool(characters) and all(
        is_synthetic_visual_identity_policy(character["visual_identity_policy"])
        for character in characters
    )
    all_gender_synthetic = bool(characters) and all(
        character["gender"].casefold() == "synthetic"
        for character in characters
    )
    def complete_identity(character: dict[str, Any]) -> bool:
        base = bool(
            character["id"]
            and (
                is_synthetic_visual_identity_policy(character["visual_identity_policy"])
                or character["gender"].casefold() == "synthetic"
            )
            and character["face_styling"]
            and (
                character["clothing"]
                or character["distinguishing"]
                or character["distinguishing_features"]
            )
        )
        if not base:
            return False
        if character["visual_identity_policy"] != NO_REAL_PERSON_POLICY:
            return True
        styling = character.get("synthetic_styling") or {}
        anchors = styling.get("visible_anchors") or []
        return bool(
            styling.get("schema") == "honcut.synthetic-styling.v2"
            and styling.get("mode")
            and styling.get("non_human_material")
            and isinstance(anchors, list)
            and len([anchor for anchor in anchors if str(anchor).strip()]) >= 2
        )

    identity_complete = bool(characters) and all(
        complete_identity(character) for character in characters
    )
    evidence.update({
        "top_level_policy_match": top_level_match,
        "all_characters_policy_tagged": all_policy_tagged,
        "all_characters_gender_synthetic": all_gender_synthetic,
        "identity_contract_complete": identity_complete,
    })
    if top_level_match:
        evidence["sources"].append("artifact:top_level_policy")
    if all_policy_tagged:
        evidence["sources"].append("artifact:all_character_policies")
    if all_gender_synthetic:
        evidence["sources"].append("artifact:all_character_genders")
    evidence["enabled"] = bool(
        env_enabled or top_level_match or all_policy_tagged or all_gender_synthetic
    )
    return evidence


def uses_synthetic_character_review(output_dir: str | Path | None = None) -> bool:
    """Resolve synthetic QA mode from runtime and durable artifact evidence."""
    return bool(synthetic_character_review_evidence(output_dir)["enabled"])


def no_real_person_prompt_contract() -> str:
    """Prompt block shared by image and video generation paths."""
    return (
        "【非真人视觉硬约束】"
        f"{SYNTHETIC_STYLE_CONTRACT}。"
        f"负面约束：{SYNTHETIC_NEGATIVE_CONTRACT}。"
        "男性/女性等词只表示服装与表演呈现，不得恢复自然真人生物特征；剧情中的脸、头发描述"
        "必须经过该角色已声明的非真人妆造合同重解释。以角色自己的妆造锚点为最高优先级，"
        "不得把不同角色统一改成同款头盔、面甲或机器人。"
    )


def _synthetic_identity(character: dict[str, Any], index: int) -> dict[str, Any]:
    """Choose a deterministic, visibly non-human styling route per character."""
    appearance = character.get("appearance")
    appearance = appearance if isinstance(appearance, dict) else {}
    source_gender = str(
        appearance.get("gender") or character.get("gender") or ""
    ).casefold()
    stable_key = str(
        character.get("id") or character.get("name") or f"character-{index}"
    )
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    palette = (
        ("钴蓝", "荧光洋红", "冷银"),
        ("孔雀绿", "熔岩橙", "石墨黑"),
        ("群青", "酸性黄", "乳白瓷"),
        ("深紫", "冰青", "暗金"),
        ("朱红", "电光蓝", "钛灰"),
    )[digest[0] % 5]
    primary, accent, material_color = palette

    female_presenting = any(
        token in source_gender
        for token in ("female", "woman", "girl", "女")
    )
    mode_index = index % 4
    if female_presenting:
        mode_index = 0
    modes: tuple[dict[str, Any], ...] = (
        {
            "mode": "veiled_graphic_couture",
            "kind": "完全虚构的面纱图形妆数字角色",
            "hair": f"{material_color}设计化纤维发束，边缘呈规则切面高光，不是自然真人发丝",
            "face": (
                f"{primary}不透明薄纱面纱固定遮住鼻梁以下，额头与眼周覆盖{accent}非对称几何彩妆和"
                f"发光面部纹样；可见区域是{material_color}哑光瓷质合成表面，不是人类皮肤"
            ),
            "anchors": [f"{primary}不透明面纱", f"{accent}发光眼周纹样", f"{material_color}瓷质表面"],
            "material": f"{material_color}哑光瓷质合成表面",
        },
        {
            "mode": "biomechanical_face_seams",
            "kind": "完全虚构的生物机械妆数字角色",
            "hair": f"{primary}硬质纤维束发型，发梢带{accent}导光丝，不是自然头发",
            "face": (
                f"脸颊和太阳穴嵌入{material_color}机械拼接板与可见接缝，眉骨下是{accent}图形光带，"
                "表面为设计化合成材质，无照片级人类皮肤"
            ),
            "anchors": [f"{material_color}脸颊机械拼接板", f"{accent}眉骨光带", f"{primary}导光纤维发束"],
            "material": f"{material_color}机械陶瓷与导光纤维",
        },
        {
            "mode": "tattoo_editorial_makeup",
            "kind": "完全虚构的纹样特效妆数字角色",
            "hair": f"{material_color}雕塑感整块发型，带{primary}印刷网点纹理，不是自然发丝",
            "face": (
                f"整张脸覆盖{primary}/{accent}高对比编辑彩妆与跨越鼻梁的电路式面部纹身，"
                f"底层是{material_color}丝绒合成皮肤并带规则微网格，不是自然人类皮肤"
            ),
            "anchors": [f"{primary}/{accent}跨鼻梁电路纹身", f"{material_color}微网格合成皮肤", "雕塑感印刷发型"],
            "material": f"{material_color}丝绒微网格合成皮肤",
        },
        {
            "mode": "crystalline_facial_texture",
            "kind": "完全虚构的晶体纹理数字角色",
            "hair": f"{primary}半透明片状头饰与短纤维冠，边缘发出{accent}冷光",
            "face": (
                f"面部由{material_color}半透明晶体纹理与{primary}放射状裂纹构成，颧骨有{accent}金属箔妆，"
                "眼部为抽象图形光孔，不是自然眼睛或皮肤"
            ),
            "anchors": [f"{primary}放射晶体裂纹", f"{accent}颧骨金属箔妆", "抽象图形光孔"],
            "material": f"{material_color}半透明晶体合成材质",
        },
    )
    identity = dict(modes[mode_index])
    identity["mark"] = (
        f"锁骨位置保留{accent}短横识别灯，服装左侧有仅属于该角色的"
        f"{digest[1]:02X}{digest[2]:02X}几何编号章"
    )
    return identity


def apply_no_real_person_character_policy(
    characters_data: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite human identity fields into stable, visibly synthetic styling.

    The transform intentionally preserves names, roles, relationships, costume
    colors, and story semantics while replacing natural biometric cues with
    persistent veil/makeup/tattoo/mechanical/material anchors that cannot be
    mistaken for an untreated real-person likeness.
    """
    rewritten = copy.deepcopy(characters_data)
    characters = rewritten.get("characters")
    if not isinstance(characters, list):
        return rewritten
    if (
        rewritten.get("visual_identity_policy") == NO_REAL_PERSON_POLICY
        and characters
        and all(
            isinstance(character, dict)
            and character.get("visual_identity_policy") == NO_REAL_PERSON_POLICY
            and isinstance(
                (character.get("appearance") or {}).get("synthetic_styling"),
                dict,
            )
            and (character.get("appearance") or {})
            .get("synthetic_styling", {})
            .get("schema")
            == "honcut.synthetic-styling.v2"
            for character in characters
        )
    ):
        return rewritten

    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            continue
        appearance = character.get("appearance")
        if not isinstance(appearance, dict):
            appearance = {}
            character["appearance"] = appearance
        identity = _synthetic_identity(character, index)
        clothing = str(appearance.get("clothing") or "").strip()
        clothing = clothing or "全身密闭式科幻战术装甲和防滑太空作战靴"
        appearance.update(
            {
                "gender": "synthetic",
                "age_range": "not applicable",
                "hair": identity["hair"],
                "face": identity["face"],
                "clothing": clothing,
                "distinguishing": identity["mark"],
                "synthetic_styling": {
                    "schema": "honcut.synthetic-styling.v2",
                    "mode": identity["mode"],
                    "non_human_material": identity["material"],
                    "visible_anchors": list(identity["anchors"]),
                    "minimum_visible_anchors_per_shot": 2,
                    "visibility_contract": (
                        "every character-bearing shot must visibly preserve at least two listed anchors; "
                        "occlusion may hide one anchor but may not restore an untreated natural human face"
                    ),
                },
                "summary": (
                    f"{identity['kind']}，{identity['face']}；{identity['hair']}；{clothing}；"
                    f"{identity['mark']}。妆造锚点：{'、'.join(identity['anchors'])}。"
                    "明确为风格化 CGI 数字角色而非真人"
                ),
            }
        )
        character["style"] = SYNTHETIC_STYLE_CONTRACT
        old_negative = str(character.get("negative") or "").strip()
        character["negative"] = "，".join(
            part for part in (SYNTHETIC_NEGATIVE_CONTRACT, old_negative) if part
        )
        character["visual_identity_policy"] = NO_REAL_PERSON_POLICY
        character["distinguishing_features"] = [
            clothing,
            identity["face"],
            identity["mark"],
            *identity["anchors"],
        ]
        character["prompt_definition"] = (
            f"将{{图片N}}中的[{identity['kind']}、{identity['face']}、{identity['hair']}、{clothing}、"
            f"{identity['mark']}]定义为{{主体N}}；主体是虚构数字角色，至少两个妆造锚点逐镜可见，"
            "不得恢复未经妆造的自然真人脸，也不得替换成通用头盔"
        )
        existing_guardrails = str(character.get("negative_guardrails") or "").strip()
        character["negative_guardrails"] = "，".join(
            part
            for part in (SYNTHETIC_NEGATIVE_CONTRACT, existing_guardrails)
            if part
        )
    rewritten["visual_identity_policy"] = NO_REAL_PERSON_POLICY
    rewritten["synthetic_styling_policy"] = {
        "schema": "honcut.synthetic-styling-policy.v2",
        "minimum_visible_anchors_per_character": 2,
        "same_headgear_for_every_character_forbidden": True,
        "natural_human_face_without_declared_styling_forbidden": True,
    }
    return rewritten
