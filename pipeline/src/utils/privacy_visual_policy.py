"""Project-level visual identity policy for privacy-safe synthetic characters."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

NO_REAL_PERSON_ENV = "HONCUT_NO_REAL_PERSON"
NO_REAL_PERSON_POLICY = "synthetic_stylized_character_v3"
LEGACY_NO_REAL_PERSON_POLICIES = frozenset({
    "synthetic_faceless_android_v1",
    "synthetic_stylized_character_v2",
})
SUPPORTED_NO_REAL_PERSON_POLICIES = frozenset(
    {NO_REAL_PERSON_POLICY, *LEGACY_NO_REAL_PERSON_POLICIES}
)
SYNTHETIC_QA_CONTRACT = "synthetic_character_styling_consistency_v3"

SYNTHETIC_STYLE_CONTRACT = (
    "高成本风格化三维 CGI 动画，所有角色都是完全虚构的数字角色；"
    "面部统一采用精致的珍珠生体瓷妆：完整无遮挡的协调五官、珍珠陶瓷合成皮肤、"
    "从太阳穴延伸到颧骨的细窄虹彩电路妆纹、柔和发光虹膜环与设计化纤维发丝；"
    "每个角色使用独立且逐镜持久的配色、妆纹走向和识别码，至少两个合成人锚点清晰可见；"
    "整体优雅克制、干净美观、明确属于数字合成人，不是真人实拍，不模仿任何现实人物"
)

SYNTHETIC_NEGATIVE_CONTRACT = (
    "真人，真人实拍，写真人脸，未经妆造的自然人脸，照片级人类皮肤，自然人类眼睛，"
    "自然裸露皮肤，普通真人发丝，名人，现实人物，身份证照片，肖像摄影，换脸，live-action，"
    "photorealistic human，real person，natural human skin，面纱，遮脸面具，统一头盔，统一面甲，"
    "粗大机械面板，破裂面孔，伤疤，恐怖化，畸形五官，廉价塑料感"
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
            return False
        styling = character.get("synthetic_styling") or {}
        anchors = styling.get("visible_anchors") or []
        return bool(
            styling.get("schema") == "honcut.synthetic-styling.v3"
            and styling.get("mode") == "synthetic_porcelain_makeup"
            and styling.get("makeup_design_id")
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
        "必须经过该角色已声明的非真人妆造合同重解释。面部必须完整可见，以角色自己的妆造锚点"
        "为最高优先级，不得把不同角色统一改成同款头盔、面甲或机器人。"
    )


def _synthetic_identity(character: dict[str, Any], index: int) -> dict[str, Any]:
    """Build one deterministic porcelain-makeup language per character."""
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

    presentation = "柔和利落" if any(
        token in source_gender for token in ("female", "woman", "girl", "女")
    ) else "克制利落"
    design_id = f"porcelain-{digest[1]:02x}{digest[2]:02x}{digest[3]:02x}"
    identity = {
        "mode": "synthetic_porcelain_makeup",
        "makeup_design_id": design_id,
        "kind": "完全虚构的珍珠生体瓷妆数字角色",
        "hair": (
            f"{primary}设计化纤维发束，发梢带极细{accent}导光丝与规则切面高光，"
            "不是自然真人发丝"
        ),
        "face": (
            f"面部完整无遮挡，五官比例协调、表情自然且{presentation}；表面为{material_color}珍珠陶瓷"
            f"合成皮肤，无真人毛孔；从太阳穴到颧骨只有一条{accent}细窄虹彩电路妆纹，"
            f"虹膜保留一圈柔和{primary}非自然光环；妆造编号{design_id}，不是人类皮肤或真人肖像"
        ),
        "anchors": [
            f"{material_color}珍珠陶瓷合成皮肤",
            f"{accent}太阳穴至颧骨细窄电路妆纹",
            f"{primary}柔和发光虹膜环",
        ],
        "material": f"{material_color}珍珠陶瓷合成皮肤",
    }
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
    persistent porcelain-makeup anchors that cannot be mistaken for an
    untreated real-person likeness.
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
            == "honcut.synthetic-styling.v3"
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
                    "schema": "honcut.synthetic-styling.v3",
                    "mode": identity["mode"],
                    "makeup_design_id": identity["makeup_design_id"],
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
        "schema": "honcut.synthetic-styling-policy.v3",
        "allowed_mode": "synthetic_porcelain_makeup",
        "minimum_visible_anchors_per_character": 2,
        "same_headgear_for_every_character_forbidden": True,
        "natural_human_face_without_declared_styling_forbidden": True,
    }
    return rewritten
