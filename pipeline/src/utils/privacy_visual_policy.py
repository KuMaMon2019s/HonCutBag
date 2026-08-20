"""Project-level visual identity policy for privacy-safe synthetic characters."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

NO_REAL_PERSON_ENV = "HONCUT_NO_REAL_PERSON"
NO_REAL_PERSON_POLICY = "synthetic_faceless_android_v1"

SYNTHETIC_STYLE_CONTRACT = (
    "高成本风格化三维 CGI 科幻动画，所有角色都是完全虚构的合成人或安保机器人；"
    "角色始终佩戴不透明全封闭机械头盔和反光面甲，不露出脸、皮肤、头发、眼睛或任何真人生物特征；"
    "材质和轮廓明确属于设计化数字角色，不是真人实拍，不模仿任何现实人物"
)

SYNTHETIC_NEGATIVE_CONTRACT = (
    "真人，真人实拍，写真人脸，照片级人类皮肤，可见面孔，可见眼睛，可见头发，"
    "裸露皮肤，名人，现实人物，身份证照片，肖像摄影，换脸，live-action，"
    "photorealistic human，real person，natural human skin"
)


def is_no_real_person_enabled() -> bool:
    """Return whether the current pipeline process requires synthetic identities."""
    return os.environ.get(NO_REAL_PERSON_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def uses_synthetic_character_review(output_dir: str | Path | None = None) -> bool:
    """Resolve synthetic QA mode from live policy or persisted character data.

    Final QA often runs in a fresh/resumed process where the original privacy
    environment variable is no longer present.  ``CHARACTERS.json`` is the
    durable second source of truth, so reviewers must consult both sources.
    """
    if is_no_real_person_enabled():
        return True
    if output_dir is None:
        return False
    try:
        payload = json.loads(
            (Path(output_dir) / "CHARACTERS.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("visual_identity_policy") == NO_REAL_PERSON_POLICY:
        return True
    characters = payload.get("characters")
    if not isinstance(characters, list):
        return False
    return any(
        isinstance(character, dict)
        and character.get("visual_identity_policy") == NO_REAL_PERSON_POLICY
        for character in characters
    )


def no_real_person_prompt_contract() -> str:
    """Prompt block shared by image and video generation paths."""
    return (
        "【非真人视觉硬约束】"
        f"{SYNTHETIC_STYLE_CONTRACT}。"
        f"负面约束：{SYNTHETIC_NEGATIVE_CONTRACT}。"
        "如果剧情文字含有男性、女性、脸、头发或真人写实等旧描述，一律忽略这些旧描述，"
        "以本非真人视觉硬约束为最高优先级。"
    )


def _synthetic_identity(character: dict[str, Any]) -> dict[str, str]:
    role = str(character.get("role") or "").strip().lower()
    char_id = str(character.get("id") or "").strip().lower()
    protagonist = role == "protagonist" or char_id == "agent"
    if protagonist:
        return {
            "kind": "完全虚构的深灰色战术合成人",
            "helmet": "哑光深灰色全封闭机械头盔，单片式不透明黑色反光面甲，无可见面孔或皮肤",
            "mark": "面甲右眉位置有一条细窄琥珀色识别灯带，胸甲有三角形冷白编号灯",
        }
    return {
        "kind": "完全虚构的藏蓝色安保机器人",
        "helmet": "哑光藏蓝色全封闭安保头盔，横向不透明深红色机械面甲，无可见面孔或皮肤",
        "mark": "左前臂是发光安保编号板，肩部保留银色安保识别章",
    }


def apply_no_real_person_character_policy(
    characters_data: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite human identity fields into stable, faceless synthetic designs.

    The transform intentionally preserves names, roles, relationships, costume
    colors, and story semantics while removing every facial/biometric cue that
    could steer image or video providers toward a real-person likeness.
    """
    rewritten = copy.deepcopy(characters_data)
    characters = rewritten.get("characters")
    if not isinstance(characters, list):
        return rewritten

    for character in characters:
        if not isinstance(character, dict):
            continue
        appearance = character.get("appearance")
        if not isinstance(appearance, dict):
            appearance = {}
            character["appearance"] = appearance
        identity = _synthetic_identity(character)
        clothing = str(appearance.get("clothing") or "").strip()
        clothing = clothing or "全身密闭式科幻战术装甲和防滑太空作战靴"
        appearance.update(
            {
                "gender": "synthetic",
                "age_range": "not applicable",
                "hair": "无可见头发；全部隐藏在不可开启的全封闭机械头盔内",
                "face": identity["helmet"],
                "clothing": clothing,
                "distinguishing": identity["mark"],
                "summary": (
                    f"{identity['kind']}，{identity['helmet']}；{clothing}；"
                    f"{identity['mark']}。全身无裸露皮肤，明确为风格化 CGI 数字角色而非真人"
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
            identity["helmet"],
            identity["mark"],
        ]
        character["prompt_definition"] = (
            f"将{{图片N}}中的[{identity['kind']}、{identity['helmet']}、{clothing}、"
            f"{identity['mark']}]定义为{{主体N}}；主体是虚构合成人，不得生成人脸或皮肤"
        )
        existing_guardrails = str(character.get("negative_guardrails") or "").strip()
        character["negative_guardrails"] = "，".join(
            part
            for part in (SYNTHETIC_NEGATIVE_CONTRACT, existing_guardrails)
            if part
        )
    rewritten["visual_identity_policy"] = NO_REAL_PERSON_POLICY
    return rewritten
