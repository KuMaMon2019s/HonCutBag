#!/usr/bin/env python3
"""
prompt_router.py — M4: HonCut 模型路由（4 种提示词模式）
按模型名自动匹配提示词格式。
"""

from typing import Optional


def route_prompt(model_name: str, mode: str, shot_data: dict, assets: list = None) -> str:
    """
    路由入口：按模型名 + 模式选择提示词格式。
    
    Args:
        model_name: 模型名（如 "seedance-2-0", "wan2.6"）
        mode: "multi_shot" | "single_shot" | "first_last_frame"
        shot_data: 镜头数据 dict（含 prompt/visual/who/where/emotion 等）
        assets: 角色资产列表 [{"name": "林夏", "description": "..."}]
    
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
    """Seedance 2.0 多分镜模式：中文结构化12维编码 + @图N 引用 + 毫秒时长"""
    style = shot_data.get("style", "真人写实, 电影风格, 都市暖调")
    duration_ms = shot_data.get("duration", 6) * 1000
    
    parts = [f"画面风格和类型: {style}\n"]
    
    # 图片定义
    if assets:
        parts.append("图片定义:")
        for i, asset in enumerate(assets, 1):
            name = asset.get("name", f"角色{i}")
            desc = asset.get("description", "")
            parts.append(f"@图{i}: {name}，{desc}")
        parts.append("")
    
    # 分镜内容
    shots = shot_data.get("shots", [shot_data])
    parts.append(f"生成一个由以下 {len(shots)} 个分镜组成的视频:\n")
    
    for i, shot in enumerate(shots, 1):
        dur = shot.get("duration", shot.get("suggested_duration", 6))
        time_desc = shot.get("time", "白天")
        where = shot.get("where", "室内")
        camera = shot.get("camera", "中景")
        visual = shot.get("visual", shot.get("prompt", ""))
        who = ", ".join(shot.get("who", []))
        
        parts.append(f"分镜{i} {dur}s: 时间：{time_desc}，场景：{where}，"
                     f"镜头：{camera}，{who}，{visual}")
        
        # HonCut 12维编码补充
        dims = []
        if shot.get("action"): dims.append(f"动作：{shot['action']}")
        if shot.get("expression"): dims.append(f"表情：{shot['expression']}")
        if shot.get("lighting"): dims.append(f"光影：{shot['lighting']}")
        if shot.get("color_tone"): dims.append(f"色彩：{shot['color_tone']}")
        if shot.get("sound_effect"): dims.append(f"音效：{shot['sound_effect']}")
        if shot.get("rhythm"): dims.append(f"节奏：{shot['rhythm']}")
        if dims:
            parts.append("  " + "，".join(dims))
    
    return "\n".join(parts)


def _build_seedance2_single(shot_data: dict, assets: list) -> str:
    """Seedance 2.0 单镜头模式：reference_image + 英文 prompt"""
    visual = shot_data.get("visual", shot_data.get("prompt", ""))
    emotion = shot_data.get("emotion", "")
    where = shot_data.get("where", "")
    camera = shot_data.get("camera", "medium shot")
    
    # 构建英文 prompt
    parts = []
    if where:
        parts.append(f"Scene: {where}.")
    parts.append(f"Shot: {camera}.")
    if emotion:
        parts.append(f"Mood: {emotion}.")
    parts.append(visual)
    
    # 角色参考绑定
    if assets:
        names = [a.get("name", "") for a in assets]
        parts.append(f"Based on the reference image of {', '.join(names)}, "
                     "maintain consistent: face features, hairstyle, costume details.")
    
    # 风格锚定
    parts.append("Photorealistic cinematography, cinematic quality, ultra-fine detail, "
                 "delicate skin texture, strand-by-strand hair detail. "
                 "Ultra-sharp 4K, no subtitles, no watermark.")
    
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
