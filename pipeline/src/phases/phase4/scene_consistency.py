"""Phase 4 scene consistency contract for eight-layer video prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.visual_style_spec import VisualStyle, parse_visual_style


BASE_NEGATIVE_PROMPTS = [
    "变形扭曲(warping)",
    "形态渐变(morphing)",
    "面部扭曲(distorted faces)",
    "多余手指(extra fingers)",
    "模糊纹理(blurry textures)",
    "抖动运动(jittery motion)",
    "伪影(artifacts)",
]


def _load_style(path: Optional[Path]) -> VisualStyle:
    if path and path.exists():
        return parse_visual_style(path.read_text(encoding="utf-8"))
    bundled = Path(__file__).resolve().parents[3] / "prompts" / "default_visual_style.md"
    if bundled.exists():
        return parse_visual_style(bundled.read_text(encoding="utf-8"))
    return VisualStyle(
        name="cinematic-fallback",
        style_prompt_short="电影叙事风格，35mm胶片质感，高清细节",
        style_prompt_full="电影质感，冷色调，高对比度，高清细节",
        layout_aspect_ratio="16:9",
    )


def _shot_key(shot: Mapping[str, Any], index: int) -> str:
    raw = shot.get("shot_number") or shot.get("shot_order") or shot.get("id") or index
    try:
        number = int(str(raw).lstrip("Ss"))
    except (TypeError, ValueError):
        number = index
    return f"S{number:02d}"


def _lighting_for(shot: Mapping[str, Any], visual_style: VisualStyle) -> str:
    explicit = str(shot.get("lighting_description") or shot.get("lighting_key") or "").strip()
    if explicit and any(token in explicit for token in ("左", "右", "上", "下", "逆光", "侧光")):
        lighting = explicit if any(token in explicit.upper() for token in ("K", "暖", "冷")) else f"{explicit}，色温4800K"
        if not any(token in lighting for token in ("气氛", "氛围", "雾", "雨", "尘", "颗粒", "潮湿")):
            lighting += "，空气颗粒轻微可见，气氛与剧情情绪一致"
        return lighting
    place = str(shot.get("where") or shot.get("scene_description") or shot.get("visual") or "")
    if any(token in place for token in ("夜", "月", "暗", "工厂")):
        return "冷蓝月光从镜头右上方射入，色温5600K，左侧暖橙实景灯补亮人物轮廓，空气颗粒清晰"
    if any(token in place for token in ("室内", "房", "店", "办公室")):
        return "暖白LED面板从镜头左上方照射，色温4200K，右后方冷色窗光勾勒轮廓，明暗层次稳定"

    style_text = " ".join(filter(None, (
        visual_style.style_prompt_short,
        visual_style.style_prompt_full,
        " ".join(visual_style.mood_keywords),
        " ".join(visual_style.tags),
    ))).lower()
    has_rain = any(token in style_text for token in ("雨", "rain", "storm", "暴风"))
    has_night = any(token in style_text for token in ("夜", "night", "moon", "月"))
    if has_rain and has_night:
        return "冷蓝雨夜光从镜头右上方漫射照入，湿润表面反射微光，低饱和度，气氛湿冷压抑"
    if has_night:
        return "冷蓝夜间主光从镜头右上方照入，微弱环境光勾勒轮廓，气氛与全片美术风格一致"
    if has_rain or any(token in style_text for token in ("阴", "overcast", "cloudy")):
        return "阴雨天漫射冷光从上方均匀落下，低饱和度，空气潮湿，气氛与全片美术风格一致"
    if any(token in style_text for token in ("黄金时段", "golden hour", "日落", "sunset", "黄昏", "dusk")):
        return "黄金时段暖调主光从镜头左上方侧逆光照射，斜射余晖拉出长影，气氛与全片美术风格一致"
    if any(token in style_text for token in ("黎明", "清晨", "dawn", "sunrise")):
        return "清晨柔和自然光从镜头左上方斜射，薄雾中轮廓清晰，气氛与全片美术风格一致"
    return "与全片美术风格一致的自然光照，主光从镜头左上方照射，明暗关系真实克制"


def _appearance_constraints(characters: list[Mapping[str, Any]], shot: Mapping[str, Any]) -> list[str]:
    requested = shot.get("who", [])
    requested = requested if isinstance(requested, list) else [requested]
    constraints = []
    for character in characters:
        if character.get("name") not in requested and not set(character.get("aliases", [])).intersection(requested):
            continue
        appearance = character.get("appearance", {})
        stable = "，".join(
            str(appearance.get(key, "")).strip()
            for key in ("hair", "face", "clothing", "distinguishing")
            if appearance.get(key)
        )
        constraints.append(f"{character.get('name')}的{stable}与CHARACTERS.json一致")
    return constraints


def generate_scene_consistency(
    storyboard: Mapping[str, Any],
    characters_data: Optional[Mapping[str, Any]] = None,
    visual_style_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Create deterministic spatial, lighting, style, and guardrail metadata."""
    style = _load_style(visual_style_path)
    characters = list((characters_data or {}).get("characters", []))
    shots = list(storyboard.get("shots", []))
    global_style = style.style_prompt_full or style.style_prompt_short or "电影质感，高清细节"
    aspect_ratio = style.layout_aspect_ratio or "16:9"
    fixed_elements = []
    for shot in shots:
        place = str(shot.get("where") or shot.get("scene_description") or "").strip()
        if place and place not in fixed_elements:
            fixed_elements.append(place)
        props = shot.get("props", [])
        prop_names = list(props) if isinstance(props, Mapping) else props
        for prop in prop_names if isinstance(prop_names, (list, tuple, set)) else []:
            prop = str(prop).strip()
            if prop and prop not in fixed_elements:
                fixed_elements.append(prop)
    if not fixed_elements:
        fixed_elements = ["场景建筑结构", "主要家具", "门窗位置"]

    contract: dict[str, Any] = {
        "version": "1.0",
        "global_style_lock": global_style,
        "global_lighting": "主光方向与色温跨镜头连续，冷暖实景光保持一致",
        "spatial_anchors": {
            "fixed_elements": fixed_elements,
            "dynamic_elements": ["角色位置", "光影角度", "镜头视角"],
        },
        "shots": {},
    }
    for index, shot in enumerate(shots, 1):
        key = _shot_key(shot, index)
        duration = shot.get("duration", 5)
        lighting = _lighting_for(shot, style)
        style_anchor = style.style_prompt_short or global_style
        constraints = _appearance_constraints(characters, shot)
        constraints.extend([
            "光照方向与全局设定一致",
            f"画面比例{aspect_ratio}",
            "固定场景元素的位置和颜色跨镜头一致",
        ])
        subject_names = shot.get("who", [])
        subject_names = subject_names if isinstance(subject_names, list) else [subject_names]
        subject = "、".join(map(str, filter(None, subject_names))) or "场景主体"
        scene_description = str(shot.get("where") or shot.get("visual") or shot.get("prompt") or "当前场景")
        negative = ", ".join(BASE_NEGATIVE_PROMPTS)
        quality = f"4K, {aspect_ratio}, {duration}秒"
        contract["shots"][key] = {
            "scene_description": scene_description,
            "spatial_layout": {
                "camera": str(shot.get("shot_size") or "中景") + "，镜头位置沿用空间轴线",
                "subject": f"{subject}保持既定站位与朝向",
                "props": {element: "位置固定" for element in fixed_elements[:4]},
            },
            "lighting_description": lighting,
            "lighting_note": lighting,
            "style_anchor": style_anchor,
            "style_suffix": f"保持{style_anchor}",
            "negative_prompt": negative,
            "quality_suffix": quality,
            "consistency_constraints": constraints,
        }
    return contract


def write_scene_consistency(
    output_path: Path,
    storyboard: Mapping[str, Any],
    characters_data: Optional[Mapping[str, Any]] = None,
    visual_style_path: Optional[Path] = None,
) -> dict[str, Any]:
    contract = generate_scene_consistency(storyboard, characters_data, visual_style_path)
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract
