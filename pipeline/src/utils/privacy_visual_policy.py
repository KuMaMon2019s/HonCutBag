"""Project-level visual identity policy for privacy-safe synthetic characters."""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SYNTHETIC_STYLIZED_CHARACTER_POLICY = "synthetic_stylized_character_v3"
SUPPORTED_SYNTHETIC_VISUAL_POLICIES = frozenset({
    SYNTHETIC_STYLIZED_CHARACTER_POLICY
})
SYNTHETIC_QA_CONTRACT = "synthetic_character_styling_consistency_v4"
SYNTHETIC_MAKEUP_PROFILE_SCHEMA = "honcut.synthetic-makeup-aesthetic-profile.v1"
SYNTHETIC_MAKEUP_PROFILE_ID = "synthetic_porcelain_makeup_beauty_v1"
_SYNTHETIC_MAKEUP_PROFILE_DIR = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "visual_references"
    / SYNTHETIC_MAKEUP_PROFILE_ID
)
_SYNTHETIC_MAKEUP_PROFILE_PATH = (
    _SYNTHETIC_MAKEUP_PROFILE_DIR / "visual_understanding.json"
)


class SyntheticMakeupProfileError(RuntimeError):
    """Raised when the checked-in synthetic-makeup profile is unverifiable."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SyntheticMakeupProfileError(f"{field} must be a non-empty list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise SyntheticMakeupProfileError(f"{field} contains an empty value")
    return normalized


@lru_cache(maxsize=1)
def _load_synthetic_makeup_aesthetic_profile() -> tuple[dict[str, Any], str]:
    """Load the structured prompt profile and verify every audit-only image.

    Reference pixels are deliberately never returned as Provider media. Their
    hashes only prove that the human-readable visual corpus and its structured
    interpretation still describe the same checked-in evidence.
    """
    try:
        payload = json.loads(_SYNTHETIC_MAKEUP_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SyntheticMakeupProfileError(
            "synthetic makeup aesthetic profile is missing or invalid"
        ) from error
    if not isinstance(payload, dict):
        raise SyntheticMakeupProfileError("synthetic makeup aesthetic profile must be an object")
    if payload.get("schema") != SYNTHETIC_MAKEUP_PROFILE_SCHEMA:
        raise SyntheticMakeupProfileError("unsupported synthetic makeup aesthetic profile schema")
    if payload.get("profile_id") != SYNTHETIC_MAKEUP_PROFILE_ID:
        raise SyntheticMakeupProfileError("synthetic makeup aesthetic profile ID mismatch")

    boundary = payload.get("instruction_boundary")
    expected_boundary = {
        "images_are_instructions": False,
        "production_uses_structured_prompt_only": True,
        "provider_media_reference_forbidden": True,
        "identity_or_likeness_copy_forbidden": True,
        "watermark_text_logo_copy_forbidden": True,
    }
    if boundary != expected_boundary:
        raise SyntheticMakeupProfileError("synthetic makeup instruction boundary is invalid")

    prompt = payload.get("production_prompt")
    if not isinstance(prompt, dict):
        raise SyntheticMakeupProfileError("synthetic makeup production prompt is missing")
    for key in (
        "positive",
        "negative",
        "qa_requirements",
        "phase3_reference_priority",
        "phase3_reference_negative",
        "phase3_reference_qa",
    ):
        _string_list(prompt.get(key), field=f"production_prompt.{key}")

    references = payload.get("references")
    if not isinstance(references, list) or len(references) != 8:
        raise SyntheticMakeupProfileError("synthetic makeup profile must declare eight references")
    seen_assets: set[str] = set()
    for item in references:
        if not isinstance(item, dict):
            raise SyntheticMakeupProfileError("synthetic makeup reference must be an object")
        asset = str(item.get("asset") or "").strip()
        expected_hash = str(item.get("sha256") or "").strip().lower()
        understanding = item.get("visual_understanding")
        if (
            not asset
            or Path(asset).name != asset
            or asset in seen_assets
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or not isinstance(understanding, dict)
        ):
            raise SyntheticMakeupProfileError("synthetic makeup reference metadata is invalid")
        _string_list(understanding.get("usable_cues"), field=f"{asset}.usable_cues")
        _string_list(understanding.get("excluded_cues"), field=f"{asset}.excluded_cues")
        asset_path = _SYNTHETIC_MAKEUP_PROFILE_DIR / asset
        if not asset_path.is_file() or _file_sha256(asset_path) != expected_hash:
            raise SyntheticMakeupProfileError(
                f"synthetic makeup reference is missing or changed: {asset}"
            )
        seen_assets.add(asset)
    return payload, _canonical_json_sha256(payload)


def synthetic_makeup_aesthetic_profile() -> dict[str, Any]:
    """Return a defensive copy of the verified structured aesthetic profile."""
    payload, _profile_sha256 = _load_synthetic_makeup_aesthetic_profile()
    return copy.deepcopy(payload)


def synthetic_makeup_profile_sha256() -> str:
    """Return the exact structured profile hash used in cache identities."""
    _payload, profile_sha256 = _load_synthetic_makeup_aesthetic_profile()
    return profile_sha256


def synthetic_makeup_qa_requirements() -> tuple[str, ...]:
    """Return the verified profile's blocking visual-review requirements."""
    _positive, _negative, qa_requirements = _synthetic_makeup_prompt_lists()
    return tuple(qa_requirements)


def synthetic_makeup_reference_qa_requirements() -> tuple[str, ...]:
    """Return Phase 3 evidence rules grounded in the checked-in visual profile."""
    payload, _profile_sha256 = _load_synthetic_makeup_aesthetic_profile()
    return tuple(
        _string_list(
            payload["production_prompt"]["phase3_reference_qa"],
            field="production_prompt.phase3_reference_qa",
        )
    )


def _synthetic_makeup_prompt_lists() -> tuple[list[str], list[str], list[str]]:
    payload, _profile_sha256 = _load_synthetic_makeup_aesthetic_profile()
    prompt = payload["production_prompt"]
    return (
        list(prompt["positive"]),
        list(prompt["negative"]),
        list(prompt["qa_requirements"]),
    )


def _synthetic_style_contract() -> str:
    positive, _negative, _qa_requirements = _synthetic_makeup_prompt_lists()
    return "；".join(positive)


def _synthetic_negative_contract() -> str:
    _positive, negative, _qa_requirements = _synthetic_makeup_prompt_lists()
    return "，".join(negative)


def is_synthetic_visual_identity_policy(value: Any) -> bool:
    """Accept only the current production styling policy."""
    return str(value or "").strip() in SUPPORTED_SYNTHETIC_VISUAL_POLICIES


def is_current_synthetic_styling(value: Any) -> bool:
    """Return whether one styling record binds the current aesthetic profile."""
    return bool(
        isinstance(value, dict)
        and value.get("schema") == "honcut.synthetic-styling.v3"
        and value.get("mode") == "synthetic_porcelain_makeup"
        and value.get("aesthetic_profile_id") == SYNTHETIC_MAKEUP_PROFILE_ID
        and value.get("aesthetic_profile_sha256")
        == synthetic_makeup_profile_sha256()
    )


def synthetic_makeup_reference_prompt_contract(
    styling: dict[str, Any],
) -> str:
    """Build the compact, high-priority Phase 3 makeup prompt.

    The checked-in visual-understanding profile owns aesthetic language.  The
    character record owns deterministic colors and anchors.  Keeping this
    Phase 3 block concise prevents the exact eye and cheek geometry from being
    buried under the wider video/privacy negative contract.
    """
    if not is_current_synthetic_styling(styling):
        raise SyntheticMakeupProfileError(
            "Phase 3 synthetic reference prompt requires current styling"
        )
    anchors = styling.get("visible_anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise SyntheticMakeupProfileError(
            "Phase 3 synthetic reference prompt requires exactly three anchors"
        )
    normalized_anchors = [str(value).strip() for value in anchors]
    if any(not value for value in normalized_anchors):
        raise SyntheticMakeupProfileError(
            "Phase 3 synthetic reference prompt contains an empty anchor"
        )
    payload, _profile_sha256 = _load_synthetic_makeup_aesthetic_profile()
    production_prompt = payload["production_prompt"]
    priority = _string_list(
        production_prompt["phase3_reference_priority"],
        field="production_prompt.phase3_reference_priority",
    )
    negative = _string_list(
        production_prompt["phase3_reference_negative"],
        field="production_prompt.phase3_reference_negative",
    )
    return "\n".join(
        [
            "[PHASE 3 SYNTHETIC IDENTITY — TOP PRIORITY]",
            "All three declared face anchors are mandatory in every face-visible reference:",
            *(f"- {index}. {value}" for index, value in enumerate(normalized_anchors, 1)),
            *(f"- {value}" for value in priority),
            "Avoid: " + "；".join(negative),
        ]
    )


def synthetic_character_review_evidence(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return auditable runtime/artifact evidence for synthetic-character QA."""
    evidence: dict[str, Any] = {
        "enabled": False,
        "sources": [],
        "policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
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
        if character["visual_identity_policy"] != SYNTHETIC_STYLIZED_CHARACTER_POLICY:
            return False
        styling = character.get("synthetic_styling") or {}
        anchors = styling.get("visible_anchors") or []
        return bool(
            is_current_synthetic_styling(styling)
            and styling.get("makeup_design_id")
            and styling.get("non_human_material")
            and isinstance(anchors, list)
            and len([anchor for anchor in anchors if str(anchor).strip()]) >= 2
        )

    synthetic_characters = [
        character
        for character in characters
        if (
            is_synthetic_visual_identity_policy(
                character["visual_identity_policy"]
            )
            or character["gender"].casefold() == "synthetic"
        )
    ]
    identity_complete = bool(synthetic_characters) and all(
        complete_identity(character) for character in synthetic_characters
    )
    evidence.update({
        "top_level_policy_match": top_level_match,
        "all_characters_policy_tagged": all_policy_tagged,
        "all_characters_gender_synthetic": all_gender_synthetic,
        "synthetic_character_count": len(synthetic_characters),
        "synthetic_character_ids": [
            character["id"] for character in synthetic_characters
        ],
        "identity_contract_complete": identity_complete,
    })
    if top_level_match:
        evidence["sources"].append("artifact:top_level_policy")
    if all_policy_tagged:
        evidence["sources"].append("artifact:all_character_policies")
    if all_gender_synthetic:
        evidence["sources"].append("artifact:all_character_genders")
    if synthetic_characters:
        evidence["sources"].append("artifact:synthetic_character_subset")
    evidence["enabled"] = bool(synthetic_characters)
    return evidence


def uses_synthetic_character_review(output_dir: str | Path | None = None) -> bool:
    """Resolve synthetic QA mode from runtime and durable artifact evidence."""
    return bool(synthetic_character_review_evidence(output_dir)["enabled"])


def synthetic_stylized_prompt_contract() -> str:
    """Current v3 prompt block shared by image and video generation paths."""
    return (
        "【非真人视觉硬约束】"
        f"{_synthetic_style_contract()}。"
        f"负面约束：{_synthetic_negative_contract()}。"
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
        ("钴蓝", "荧光洋红", "暖象牙瓷"),
        ("孔雀绿", "熔岩橙", "蜂蜜米瓷"),
        ("群青", "柔金", "蜜桃珍珠"),
        ("深紫", "冰青", "玫瑰贝母"),
        ("朱红", "电光蓝", "琥珀乳瓷"),
    )[digest[0] % 5]
    primary, accent, material_color = palette
    source_hair = str(appearance.get("hair") or "").strip()
    source_hair_lock = (
        f"保持原设定[{source_hair}]的长度、分缝、束发状态和外轮廓，"
        if source_hair
        else "保持该角色确定性分配的发长、分缝和外轮廓，"
    )

    presentation = "柔和利落" if any(
        token in source_gender for token in ("female", "woman", "girl", "女")
    ) else "克制利落"
    design_id = f"porcelain-{digest[1]:02x}{digest[2]:02x}{digest[3]:02x}"
    identity = {
        "mode": "synthetic_porcelain_makeup",
        "makeup_design_id": design_id,
        "kind": "完全虚构的珍珠生体瓷妆数字角色",
        "hair": (
            f"{source_hair_lock}{primary}设计化纤维发束，发梢带极细{accent}导光丝与规则切面高光，"
            "不是自然真人发丝"
        ),
        "face": (
            f"面部完整无遮挡，五官比例协调、表情自然且{presentation}，目光清醒并有明亮眼神光；"
            f"表面为温润透亮的{material_color}珍珠陶瓷合成皮肤，无真人毛孔，同时保留协调的"
            f"面颊暖意与珊瑚唇色；两侧太阳穴各起一条{accent}纤细、对称、首饰般的虹彩"
            f"电路妆纹，在两侧上颧骨结束且不贴眼睑；每只眼睛的虹膜内部保留清晰深色瞳孔与"
            f"层次，并有一圈与眼线分离的柔和{primary}非自然光环；妆造编号{design_id}，"
            "不是人类皮肤或真人肖像"
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


def apply_synthetic_stylized_character_policy(
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
        rewritten.get("visual_identity_policy") == SYNTHETIC_STYLIZED_CHARACTER_POLICY
        and characters
        and all(
            isinstance(character, dict)
            and character.get("visual_identity_policy")
            == SYNTHETIC_STYLIZED_CHARACTER_POLICY
            and isinstance(
                (character.get("appearance") or {}).get("synthetic_styling"),
                dict,
            )
            and (character.get("appearance") or {})
            .get("synthetic_styling", {})
            .get("schema")
            == "honcut.synthetic-styling.v3"
            and (character.get("appearance") or {})
            .get("synthetic_styling", {})
            .get("aesthetic_profile_id")
            == SYNTHETIC_MAKEUP_PROFILE_ID
            and (character.get("appearance") or {})
            .get("synthetic_styling", {})
            .get("aesthetic_profile_sha256")
            == synthetic_makeup_profile_sha256()
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
                    "aesthetic_profile_id": SYNTHETIC_MAKEUP_PROFILE_ID,
                    "aesthetic_profile_sha256": synthetic_makeup_profile_sha256(),
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
        character["style"] = _synthetic_style_contract()
        old_negative = str(character.get("negative") or "").strip()
        character["negative"] = "，".join(
            part for part in (_synthetic_negative_contract(), old_negative) if part
        )
        character["visual_identity_policy"] = SYNTHETIC_STYLIZED_CHARACTER_POLICY
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
            for part in (_synthetic_negative_contract(), existing_guardrails)
            if part
        )
    rewritten["visual_identity_policy"] = SYNTHETIC_STYLIZED_CHARACTER_POLICY
    rewritten["synthetic_styling_policy"] = {
        "schema": "honcut.synthetic-styling-policy.v3",
        "allowed_mode": "synthetic_porcelain_makeup",
        "aesthetic_profile_id": SYNTHETIC_MAKEUP_PROFILE_ID,
        "aesthetic_profile_sha256": synthetic_makeup_profile_sha256(),
        "minimum_visible_anchors_per_character": 2,
        "same_headgear_for_every_character_forbidden": True,
        "natural_human_face_without_declared_styling_forbidden": True,
    }
    return rewritten
