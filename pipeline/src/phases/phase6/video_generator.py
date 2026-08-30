"""Asset-aware Phase 6 video generation routing."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from clients.video_client import VideoClient
from prompt.eight_layer_summary import build_subject_summary
from utils.camera_motion_contracts import (
    apply_camera_motion_contract,
    camera_motion_negative_prompt,
    camera_motion_prompt,
    camera_movement_description,
)
from utils.body_action_contracts import body_action_prompt
from utils.character_body_contracts import (
    body_contract_forbidden,
    body_contract_prompt,
)
from utils.pixel_text_policy import (
    PIXEL_TEXT_METADATA_CONTRACT,
    strip_pixel_text_identity_markers,
)
from utils.storyboard_motion_policy import append_storyboard_motion_policy
from utils.style_slices import get_slice
from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    build_temporal_visual_contract,
    normalized_temporal_visual_contract,
    temporal_visual_negative_prompt,
    temporal_visual_prompt,
)
from utils.video_generation_contracts import (
    DUPLICATE_IDENTITY_NEGATIVE,
    SPATIAL_IDENTITY_NEGATIVE,
    ensure_video_generation_contract,
)
from utils.video_geometry import resolve_video_geometry
from utils.visual_style_contract import BASE_STYLE_SCHEMA

BASE_NEGATIVE_PROMPT = (
    "变形扭曲(warping), 形态渐变(morphing), 面部扭曲(distorted faces), "
    "多余手指(extra fingers), 模糊纹理(blurry textures), "
    "抖动运动(jittery motion), 伪影(artifacts)"
)

SEEDANCE_OUTPUT_CONSTRAINTS = (
    "保持无字幕，避免生成任何文字或字幕，不要生成Logo，不要生成水印；"
    "人物和物体不得变形，不得出现多余肢体或手指，不得闪烁、漂浮、瞬移或颜色漂移"
)

_EMOTION_EXTERNALIZATION = (
    (("悲伤", "难过", "伤心"), "轻微低头、肩膀微微收紧、眼眶渐渐泛红"),
    (("喜悦", "开心", "欢喜"), "嘴角自然上扬、眉眼舒展、步伐变得轻快"),
    (("紧张", "焦虑", "不安"), "呼吸略微急促、肩颈保持警觉、目光短促扫视当前威胁"),
    (("愤怒", "生气", "怒"), "下颌线紧绷、胸口起伏加重、目光变得锐利"),
    (("释然", "放松"), "缓慢呼出一口气、紧绷的肩膀自然放松、露出克制的微笑"),
)


def _seedance_emotion_prompt(shot_meta: dict[str, Any]) -> str:
    emotion = str(shot_meta.get("emotion") or shot_meta.get("mood") or "").strip()
    if not emotion:
        return ""
    for keywords, observable in _EMOTION_EXTERNALIZATION:
        if any(keyword in emotion for keyword in keywords):
            return (
                "情绪外化（不新增剧情动作）："
                f"{observable}；仅用可见的表情、呼吸、肩背与重心变化表达{emotion}，"
                "不新增道具、不打断既定动作顺序"
            )
    return (
        f"情绪外化（不新增剧情动作）：{emotion}必须通过已写明的"
        "面部、呼吸、肩背、重心与动作节奏变化可见表达，不得只改变背景氛围"
    )


def _seedance_audio_prompts(shot_meta: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    dialogue = shot_meta.get("dialogue")
    if isinstance(dialogue, dict):
        speaker = str(dialogue.get("speaker") or "角色").strip()
        line = str(dialogue.get("line") or "").strip()
        language = str(dialogue.get("language") or "").strip()
        if line:
            speech = f"{speaker}{'用' + language if language else ''}说道{{{line}}}"
            prompts.append(f"台词：{speech}")
    elif str(dialogue or "").strip():
        prompts.append(f"台词：{{{str(dialogue).strip()}}}")

    sound_effect = str(shot_meta.get("sound_effect") or "").strip()
    if sound_effect:
        prompts.append(f"音效：<{sound_effect}>")
    music = str(
        shot_meta.get("background_music") or shot_meta.get("music") or ""
    ).strip()
    if music:
        prompts.append(f"音乐：（{music}）")
    generic_audio = str(shot_meta.get("audio") or shot_meta.get("sound") or "").strip()
    if generic_audio and generic_audio not in {sound_effect, music}:
        prompts.append(f"音效：<{generic_audio}>")
    return prompts


def _time_continuity_contract(*values: object) -> tuple[str, str]:
    """Backward-compatible wrapper around the structured temporal contract."""
    contract = build_temporal_visual_contract(
        time_of_day=values[0] if values else "",
        source_time=values[1] if len(values) > 1 else "",
        visual_context=" ".join(str(value) for value in values[2:] if value),
    )
    return temporal_visual_prompt(contract), temporal_visual_negative_prompt(contract)

def _characters_list(characters: Any) -> list[dict[str, Any]]:
    if isinstance(characters, dict):
        return list(characters.get("characters", []))
    return list(characters or [])


def _shot_number(shot_meta: dict[str, Any]) -> int:
    raw = (
        shot_meta.get("shot_number")
        or shot_meta.get("shot_order")
        or shot_meta.get("id")
        or shot_meta.get("shot_id")
        or 1
    )
    try:
        return int(str(raw).lstrip("Ss"))
    except (TypeError, ValueError):
        return 1


def resolve_video_lighting(
    scene: dict[str, Any],
    global_lighting: object = "",
) -> str:
    """Keep continuity lighting subordinate to an authored visual-style anchor."""
    if scene.get("style_anchor") or scene.get("style_suffix"):
        return (
            "光影、调色、饱和度、舞台介质与主辅光颜色严格服从项目美术风格"
            "和 Phase 4 成片首帧；场景连续性只维持同一风格内的光位与明暗关系，"
            "任何冲突的天气、色温或饱和度描述均忽略"
        )
    return str(
        scene.get("lighting_description")
        or scene.get("lighting_note")
        or global_lighting
        or "与全片美术风格一致的自然光照，明暗关系真实克制"
    )


def build_video_prompt(
    shot_meta: dict[str, Any],
    characters: Any,
    scene_consistency: dict[str, Any],
    model: str,
) -> str | dict[str, str]:
    """Assemble the eight layers and apply Seedance/Kling guardrail routing."""
    apply_camera_motion_contract(shot_meta)
    number = _shot_number(shot_meta)
    scene = scene_consistency.get("shots", {}).get(f"S{number:02d}", {})
    temporal_contract = normalized_temporal_visual_contract(
        shot_meta.get("temporal_visual_contract")
    ) or normalized_temporal_visual_contract(scene.get("temporal_visual_contract"))
    if temporal_contract is None:
        temporal_contract = apply_temporal_visual_contract(shot_meta)
    elif shot_meta.get("temporal_visual_contract") != temporal_contract:
        shot_meta["temporal_visual_contract"] = temporal_contract
    requested_declared = "who" in shot_meta or "characters" in shot_meta
    requested = shot_meta.get("who") or shot_meta.get("characters") or []
    requested = requested if isinstance(requested, list) else [requested]
    explicit_scenery = requested_declared and not requested
    selected = []
    for character in _characters_list(characters):
        matches_requested = (
            character.get("name") in requested
            or character.get("id") in requested
            or bool(set(character.get("aliases", [])).intersection(requested))
        )
        # An explicit who=[] is a hard scenery contract, not shorthand for
        # "select every known character". Legacy metadata with no character
        # field at all retains the historical all-character fallback.
        if matches_requested or (not requested_declared and not requested):
            selected.append(character)

    parts = []
    definitions = [str(char.get("prompt_definition", "")).strip() for char in selected]
    definitions = [definition for definition in definitions if definition]
    if definitions:
        parts.append("元素参考声明：" + "；".join(definitions))
    if selected:
        subject_names = "、".join(
            str(character.get("name") or character.get("id") or "角色")
            for character in selected
        )
        parts.append(
            "主体指代硬约束："
            f"全镜只使用固定名称（{subject_names}）指代对应主体；"
            "每次涉及主体都明确指名，不省略、不改名、不合并、不互换"
        )
    body_locks = [body_contract_prompt(char) for char in selected]
    body_locks = [contract for contract in body_locks if contract]
    if body_locks:
        # Keep body geometry outside the bounded eight-layer summary so height,
        # head scale, and silhouette cannot be truncated in action-heavy shots.
        parts.append("角色身体比例逐镜硬合同：" + "；".join(body_locks))
    parts.append(f"[镜头{number}｜按事件顺序]")
    shot_type = shot_meta.get("shot_type") or shot_meta.get("shot_size") or "中景"
    subject = shot_meta.get("subject_description")
    if not subject:
        traits = []
        for character in selected:
            appearance = character.get("appearance", {})
            stable = [
                strip_pixel_text_identity_markers(appearance.get(key))
                for key in ("hair", "face", "clothing")
                if appearance.get(key)
            ][:3]
            traits.append(f"{character.get('name')}—{'，'.join(stable)}")
        subject = "；".join(traits) or shot_meta.get("visual") or "场景主体"
    subject = strip_pixel_text_identity_markers(subject)
    generation_actions = shot_meta.get("generation_actions") or []
    if isinstance(generation_actions, str):
        generation_actions = [generation_actions]
    generation_actions = [
        str(value).strip() for value in generation_actions if str(value).strip()
    ]
    action = str(
        " → ".join(generation_actions)
        or shot_meta.get("action_description")
        or shot_meta.get("what")
        or "保持自然姿态"
    )
    camera = camera_movement_description(shot_meta.get("camera_movement"))
    layout = scene.get("spatial_layout", {})
    setting = scene.get("scene_description") or shot_meta.get("where") or "当前场景"
    lighting = resolve_video_lighting(
        scene,
        scene_consistency.get("global_lighting"),
    )
    if temporal_contract is None:
        temporal_contract = build_temporal_visual_contract(
            visual_context=" ".join(
                str(value or "")
                for value in (
                    lighting,
                    scene_consistency.get("global_style_lock"),
                )
            )
        )
        if temporal_contract is not None:
            shot_meta["temporal_visual_contract"] = temporal_contract
    scene_and_lighting = f"{setting}，{layout.get('subject', '')}，{lighting}".replace("，，", "，")
    subject_summary = build_subject_summary([
        ("景别与主体：", f"{shot_type}，{subject}"),
        ("动作：", action),
        ("场景与光影：", scene_and_lighting),
        ("运镜：", camera),
    ])
    parts.append(f"主体总结：{subject_summary}")
    if generation_actions or action != "保持自然姿态":
        parts.append(
            "动作细节执行：只细化上述已写动作，明确对应执行肢体、"
            "幅度、速度、力度、重心变化和前后惯性；动作自然承接，"
            "不增加新动作，不改变已写的接触点、先后顺序与结果"
        )
    emotion_prompt = _seedance_emotion_prompt(shot_meta)
    if emotion_prompt:
        parts.append(emotion_prompt)
    parts.append(f"场景与光影硬合同：{scene_and_lighting}")
    parts.append("运镜规则：每个镜头只使用一种主运镜，其他取景词只描述构图结果")
    parts.append(f"运镜物理硬合同：{camera_motion_prompt(shot_meta)}")
    if generation_actions:
        # The bounded eight-layer summary is deliberately short. Keep the full
        # authored action ledger outside that limiter so legacy Phase 6 routing
        # cannot silently drop later body actions and animate only the scenery.
        parts.append("主体动作逐项硬合同：" + " → ".join(generation_actions))
    choreography_prompt = body_action_prompt(shot_meta)
    if choreography_prompt:
        # This contract is never placed inside the bounded subject summary.
        parts.append(choreography_prompt)
    parts.extend(_seedance_audio_prompts(shot_meta))
    # The per-shot storyboard frame already carries the project's visual style.
    # Project-level summaries often contain plot nouns (characters, palaces,
    # props, future locations). Repeating that prose in every prompt caused a
    # cloud-only opening shot to invent a fairy and a palace.
    style_contract = scene.get("style_contract") or scene_consistency.get(
        "style_contract"
    )
    if (
        isinstance(style_contract, dict)
        and style_contract.get("schema") == BASE_STYLE_SCHEMA
        and style_contract.get("base_style")
        and style_contract.get("positive_prompt")
        and style_contract.get("negative_prompt")
    ):
        style = (
            "BASE VISUAL STYLE: "
            f"{style_contract['base_style']}; {style_contract['positive_prompt']}; "
            "同时仅继承该镜头参考成片首帧的色彩科学、光影质感、材质真实度、"
            "镜头语言与氛围；不得从项目级风格描述引入本镜头动作与视觉契约未明确列出的"
            "人物、建筑、地点、道具或剧情元素"
        )
    else:
        style = (
            "仅继承该镜头参考分镜帧的渲染方式、色彩科学、光影质感、材质真实度、"
            "镜头语言与氛围；不得从项目级风格描述引入本镜头动作与视觉契约未明确列出的"
            "人物、建筑、地点、道具或剧情元素"
        )
    style = get_slice(style, "video")
    ratio, _width, _height = resolve_video_geometry({**scene, **shot_meta})
    quality = str(
        scene.get("quality_suffix")
        or f"高清细节, {ratio}, {shot_meta.get('duration', 5)}秒"
    )
    # Resolution is a provider semantic parameter owned by ``media_profile``;
    # keep legacy scene artifacts from asking the model to hallucinate "4K".
    quality = re.sub(r"^\s*4[kK]\s*,?\s*", "高清细节, ", quality, count=1)
    quality = re.sub(r"(?<!\d)\d+(?:\.\d+)?:\d+(?:\.\d+)?(?!\d)", ratio, quality)
    time_lock = temporal_visual_prompt(temporal_contract)
    time_negative = temporal_visual_negative_prompt(temporal_contract)
    if time_lock:
        # This is intentionally outside build_subject_summary's character budget.
        parts.append(f"时空连续性硬约束：{time_lock}")
    # Layer 8 is appended after the bounded summary and is never truncated.
    parts.append(f"全局收尾：视觉风格与画质：{style}；{quality}")
    parts.append(f"输出约束：{SEEDANCE_OUTPUT_CONSTRAINTS}")
    parts.append(PIXEL_TEXT_METADATA_CONTRACT)

    negatives = [str(scene.get("negative_prompt", "")).strip()]
    if (
        isinstance(style_contract, dict)
        and style_contract.get("schema") == BASE_STYLE_SCHEMA
    ):
        negatives.append(str(style_contract.get("negative_prompt") or "").strip())
    if explicit_scenery:
        negatives.append(
            "人物(people), 人形主体(humanoid figures), 角色(characters), "
            "服装(costumes), 未明确列出的建筑或道具(unlisted architecture or props)"
        )
    if time_negative:
        negatives.append(time_negative)
    negatives.extend(str(char.get("negative_guardrails", "")).strip() for char in selected)
    negatives.extend(
        ", ".join(body_contract_forbidden(char))
        for char in selected
        if body_contract_forbidden(char)
    )
    if selected:
        negatives.append(DUPLICATE_IDENTITY_NEGATIVE)
        negatives.append(SPATIAL_IDENTITY_NEGATIVE)
    negatives.append(camera_motion_negative_prompt(shot_meta))
    # Keep the long-standing base guardrail at the tail for downstream tools
    # that verify the prompt suffix while still carrying the richer contracts.
    negatives.append(BASE_NEGATIVE_PROMPT)
    normalized_negatives = list(dict.fromkeys(item for item in negatives if item))
    normalized_negatives = [
        item for item in normalized_negatives if item != BASE_NEGATIVE_PROMPT
    ]
    normalized_negatives.append(BASE_NEGATIVE_PROMPT)
    negative_prompt = ", ".join(normalized_negatives)
    additional_negative_prompt = ", ".join(normalized_negatives[:-1])
    negative_suffix = (
        (f"附加约束条件：{additional_negative_prompt}。" if additional_negative_prompt else "")
        + f"约束条件：{BASE_NEGATIVE_PROMPT}"
    )
    prompt = append_storyboard_motion_policy("。".join(parts))
    from utils.privacy_visual_policy import is_synthetic_visual_identity_policy

    synthetic_identity = bool(selected) and (
        all(
            is_synthetic_visual_identity_policy(
                character.get("visual_identity_policy")
            )
            for character in selected
        )
    )
    if synthetic_identity:
        from utils.privacy_visual_policy import synthetic_stylized_prompt_contract

        prompt = f"{synthetic_stylized_prompt_contract()}\n{prompt}"
    prompt = ensure_video_generation_contract(prompt, shot_meta, characters)
    if explicit_scenery:
        scenery_lock = (
            "纯环境镜头硬约束：画面中保持零人物、零人形主体、零服装与零角色道具，"
            "只呈现本镜头主体总结和动作中明确列出的环境元素"
        )
        return f"{prompt}。{scenery_lock}。{negative_suffix}"
    fictional_decl = "虚拟形象声明：片中角色均为 AI 生成的虚构角色，非真实人物"
    if "kling" in model.lower():
        return {"prompt": f"{fictional_decl}。{prompt}", "negative_prompt": negative_prompt}
    identity_lock = (
        "synthetic-identity-lock/synthetic-styling-lock：保持参考图中每个角色自己的温暖珍珠陶瓷肤色、"
        "清晰瞳孔与虹膜层次、太阳穴至上颧骨的纤细电路彩妆、完整无遮挡五官、发型、服装类别与主色；"
        "整体必须健康、美观、清醒且明确为风格化合成人，禁止尸妆、鬼妆、面部裂纹、粗大机械面板、"
        "面纱、面具或同款头盔替换"
        if synthetic_identity
        else "identity-lock：保持参考图中的面部骨骼、发型、服装类别与主色不变"
    )
    return f"{prompt}。{fictional_decl}。{identity_lock}。{negative_suffix}"


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _read_state(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, MutableMapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _write_state(state: Any, name: str, value: Any) -> None:
    if isinstance(state, MutableMapping):
        state[name] = value
    else:
        setattr(state, name, value)


class VideoGenerator:
    """Generate Phase 6 video, preferring Bridge assets when available."""

    def __init__(self, video_client: VideoClient | None = None) -> None:
        self.video_client = video_client or VideoClient()

    async def run(self, state: Any) -> Any:
        asset_id = _read_state(state, "character_asset_id") or _read_state(state, "asset_id")
        prompt = _read_state(state, "prompt", "")
        negative_prompt = None
        output_dir = Path(_read_state(state, "output_dir", "."))
        shot_meta = _read_state(state, "shot_meta") or _read_state(state, "shot")
        characters = _read_state(state, "characters") or _load_json(output_dir / "CHARACTERS.json", {})
        scene_contract = _read_state(state, "scene_consistency") or _load_json(output_dir / "SCENE_CONSISTENCY.json", {})
        model = str(_read_state(state, "model") or "seedance")
        if shot_meta and scene_contract:
            routed = build_video_prompt(shot_meta, characters, scene_contract, model)
            if isinstance(routed, dict):
                prompt = routed["prompt"]
                negative_prompt = routed["negative_prompt"]
            else:
                prompt = routed
            _write_state(state, "assembled_prompt", routed)
        if asset_id:
            result = await self.video_client.generate_with_assets(
                asset_id=asset_id,
                asset_type="character",
                image_index=0,
                model=_read_state(state, "model") or "wan22",
                prompt=prompt,
            )
        else:
            generation_kwargs = {
                "reference_images": _read_state(state, "reference_images", []),
            }
            if negative_prompt:
                generation_kwargs["negative_prompt"] = negative_prompt
            result = self.video_client.generate(
                prompt=prompt,
                **generation_kwargs,
            )
            if inspect.isawaitable(result):
                result = await result
        _write_state(state, "video_result", result)
        return state
