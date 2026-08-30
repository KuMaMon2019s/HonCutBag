#!/usr/bin/env python3
"""
prompt_router.py — M4: HonCut 模型路由（4 种提示词模式）
按模型名自动匹配提示词格式。
"""

from typing import Optional

from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    temporal_visual_negative_prompt,
    temporal_visual_prompt,
)
from utils.video_generation_contracts import VIDEO_GENERATION_CONTRACT_MARKER


def route_prompt(model_name: str, mode: str, shot_data: dict, assets: list = None) -> str:
    """
    路由入口：按模型名 + 模式选择提示词格式。
    
    Args:
        model_name: 模型名（如 "seedance-2-0", "wan2.6"）
        mode: "multi_shot" | "single_shot" | "first_last_frame"
        shot_data: 镜头数据 dict（含 prompt/visual/who/where/emotion 等）
        assets: 角色资产列表 [{"name": "CHARACTER_A", "description": "..."}]
    
    Returns:
        格式化后的提示词字符串
    """
    assets = assets or []
    model_lower = model_name.lower()
    
    if mode == "multi_ref":
        return _build_generic_multi_ref(shot_data, assets)
    
    if "seedance" in model_lower and "2" in model_lower:
        if mode == "multi_shot":
            return _build_seedance2_multi(shot_data, assets)
        else:
            return _build_seedance2_single(shot_data, assets)
    elif "wan" in model_lower and "2.6" in model_lower:
        return _build_wan26_narrative(shot_data)
    else:
        return _build_generic_first_last_frame(shot_data)


def _build_seedance2_multi(shot_data: dict, assets: list) -> str:
    """Seedance 2.0 ordered shots without brittle per-shot timecodes."""
    style = shot_data.get("style", "真人写实, 电影风格, 都市暖调")

    parts = []

    # Important reference bindings stay ahead of style and shot metadata.
    if assets:
        parts.append("图片定义：")
        for i, asset in enumerate(assets, 1):
            name = asset.get("name", f"角色{i}")
            desc = asset.get("description", "")
            parts.append(f"@图片{i}：{name}，{desc}")
        parts.append("")

    shots = shot_data.get("shots", [shot_data])
    parts.append(f"按以下事件顺序生成 {len(shots)} 个连续镜头：\n")

    for i, shot in enumerate(shots, 1):
        raw_who = shot.get("who") or shot.get("characters") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        subject = "、".join(str(value) for value in raw_who if value)
        subject = subject or str(shot.get("subject") or "场景主体")
        visual = str(shot.get("visual") or shot.get("prompt") or "").strip()
        action = str(shot.get("action") or visual or "保持自然连续运动").strip()
        time_desc = str(shot.get("time") or shot.get("time_of_day") or "").strip()
        where = str(shot.get("where") or "当前场景").strip()
        setting = "，".join(value for value in (time_desc, where) if value)
        lighting = str(shot.get("lighting") or "服从参考画面光影").strip()
        color_tone = str(shot.get("color_tone") or "").strip()
        if color_tone:
            lighting = f"{lighting}，{color_tone}"
        camera = str(shot.get("camera") or "固定机位").strip()

        layers = [
            f"精准主体：{subject}",
            f"动作细节：{action}",
        ]
        if visual and visual != action:
            layers.append(f"画面细节：{visual}")
        expression = str(shot.get("expression") or "").strip()
        if expression:
            layers.append(f"可见表情与身体状态：{expression}")
        layers.extend(
            (
                f"场景环境：{setting}",
                f"光影色调：{lighting}",
                f"主运镜：{camera}",
            )
        )
        rhythm = str(shot.get("rhythm") or "").strip()
        if rhythm:
            layers.append(f"动作节奏：{rhythm}")
        sound_effect = str(shot.get("sound_effect") or "").strip()
        if sound_effect:
            layers.append(f"音效：<{sound_effect}>")
        dialogue = shot.get("dialogue")
        if isinstance(dialogue, dict):
            dialogue = dialogue.get("line")
        if str(dialogue or "").strip():
            layers.append(f"台词：{{{str(dialogue).strip()}}}")
        music = str(shot.get("background_music") or shot.get("music") or "").strip()
        if music:
            layers.append(f"音乐：（{music}）")
        parts.append(f"镜头{i}：" + "；".join(layers))

    parts.append(f"画面风格和类型：{style}")
    parts.append("输出约束：无字幕、无Logo、无水印；主体身份稳定，动作连续自然")
    return "\n".join(parts)


def _build_seedance2_single(shot_data: dict, assets: list) -> str:
    """Seedance 2.0 单镜头模式：保留上游完整的镜头契约。

    Phase 4/6 已经把角色、动作、场景、光照和全局风格组装进
    ``prompt``。模型路由只能增强这份契约，不能退回只使用 ``visual``，
    否则会丢失雨夜等跨镜头约束。
    """
    source_prompt = str(
        shot_data.get("prompt") or shot_data.get("visual") or ""
    ).strip()
    if VIDEO_GENERATION_CONTRACT_MARKER in source_prompt:
        return source_prompt
    emotion = shot_data.get("emotion", "")
    where = shot_data.get("where", "")
    camera = shot_data.get("camera") or shot_data.get("shot_size") or "medium shot"
    time_desc = shot_data.get("time_of_day") or shot_data.get("time") or ""
    lighting = (
        shot_data.get("lighting_description")
        or shot_data.get("lighting")
        or shot_data.get("lighting_key")
        or ""
    )
    temporal_contract = apply_temporal_visual_contract(shot_data)
    
    # 构建英文 prompt
    parts = [source_prompt]
    if where:
        parts.append(f"Scene: {where}.")
    parts.append(f"Shot: {camera}.")
    if time_desc:
        parts.append(f"Time and weather: {time_desc}.")
    if lighting:
        parts.append(f"Lighting continuity: {lighting}.")
    if temporal_contract:
        parts.append(
            "Temporal visual hard contract: "
            f"{temporal_visual_prompt(temporal_contract)}. "
            f"Forbidden cues: {temporal_visual_negative_prompt(temporal_contract)}."
        )
    if emotion:
        parts.append(f"Mood: {emotion}.")
    # ``source_prompt`` is the complete Phase 6 contract and already carries a
    # reference-frame style lock. Re-appending ``style_anchor`` here can leak
    # project-level plot nouns (future characters, architecture, props) into a
    # scenery-only shot after the safe prompt has already been assembled.
    from utils.privacy_visual_policy import (
        is_synthetic_visual_identity_policy,
    )

    synthetic_identity = bool(
        is_synthetic_visual_identity_policy(
            shot_data.get("visual_identity_policy")
        )
    )

    # 角色参考绑定
    if assets:
        names = [a.get("name", "") for a in assets]
        identity_traits = (
            "declared warm pearl bio-ceramic complexion, clear pupils and layered irises, "
            "fine symmetric temple-to-upper-cheek circuit cosmetics, unobscured facial geometry, "
            "designed hair silhouette, costume colors, and identity markers"
            if synthetic_identity
            else "face features, hairstyle, costume details"
        )
        parts.append(
            f"Based on the reference image of {', '.join(names)}, "
            f"maintain consistent: {identity_traits}."
        )
    
    # 风格锚定：纯环境镜不得携带 skin/hair 等人物诱导词。
    character_requested = bool(
        assets or shot_data.get("who") or shot_data.get("characters")
    )
    if synthetic_identity and character_requested:
        parts.append(
            "High-end stylized 3D CGI cinematography with beautiful, warm and clearly synthetic "
            "pearl bio-ceramic facial styling. Preserve each character's declared cosmetic anchors; "
            "no untreated natural human face, corpse-gray skin, blank glowing eyes, facial cracks, "
            "coarse mechanical plates, veil, mask, or one generic helmet copied to all roles. "
            "cinematic quality, ultra-fine material detail, ultra-sharp detail, "
            "no subtitles, no Logo, no watermark."
        )
    else:
        detail_contract = (
            "delicate skin texture, strand-by-strand hair detail"
            if character_requested
            else "realistic volumetric atmosphere, fine environmental material detail"
        )
        parts.append(
            "Photorealistic cinematography, cinematic quality, ultra-fine detail, "
            f"{detail_contract}. Ultra-sharp detail, no subtitles, no Logo, no watermark."
        )
    
    return " ".join(parts)


def _build_wan26_narrative(shot_data: dict) -> str:
    """Wan 2.6 模式：叙事式英文（风格→主体→光线→镜头）"""
    visual = shot_data.get("visual", shot_data.get("prompt", ""))
    emotion = shot_data.get("emotion", "calm")
    where = shot_data.get("where", "")
    
    parts = [
        "Cinematic photorealistic style.",
        f"Subject: {visual}",
        f"Environment: {where}.",
        f"Lighting: natural soft light, {emotion} atmosphere.",
        "Camera: smooth dolly movement, shallow depth of field.",
    ]
    return " ".join(parts)


def _build_generic_first_last_frame(shot_data: dict) -> str:
    """通用首尾帧模式：[Visual][Motion][Camera][Audio][Narrative] 五维度"""
    visual = shot_data.get("visual", shot_data.get("prompt", ""))
    emotion = shot_data.get("emotion", "")
    where = shot_data.get("where", "")
    
    parts = [
        f"[Visual] {visual}",
        f"[Motion] Natural character movement, subtle gestures.",
        f"[Camera] Medium shot, slight pan.",
        f"[Audio] Ambient sound of {where or 'the environment'}.",
        f"[Narrative] Emotional tone: {emotion or 'neutral'}.",
    ]
    return " ".join(parts)


def _build_generic_multi_ref(shot_data: dict, assets: list) -> str:
    """HonCut 通用多参模式：[References] + [Instruction]。"""
    parts = []
    # References 段
    if assets:
        parts.append("[References]")
        for i, asset in enumerate(assets, 1):
            name = asset.get("name", f"Ref{i}")
            desc = asset.get("description", "")
            parts.append(f"Image {i}: {name} — {desc}")
        parts.append("")
    # Instruction 段
    visual = shot_data.get("visual", shot_data.get("prompt", ""))
    where = shot_data.get("where", "")
    emotion = shot_data.get("emotion", "")
    camera = shot_data.get("camera", "medium shot")
    parts.append("[Instruction]")
    parts.append(f"Scene: {where}. Camera: {camera}. Mood: {emotion}.")
    parts.append(f"Action: {visual}")
    parts.append("Maintain character consistency with reference images.")
    return "\n".join(parts)
