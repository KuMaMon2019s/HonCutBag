"""Controlled visual-medium contract shared by image and video phases."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from utils.visual_style_spec import VisualStyle

BASE_STYLE_SCHEMA = "honcut.image-style.v1"
CLIP_STYLE_SCHEMA = "honcut.clip-style-classification.v1"
STYLE_REFERENCE_POLICY_VERSION = "1"
STYLE_REFERENCE_MIN_MARGIN = 0.008


class BaseStyle(StrEnum):
    PHOTOREALISTIC = "photorealistic"
    CINEMATIC = "cinematic"
    ANIME = "anime"
    DONGHUA = "donghua"
    THREE_D_ANIMATION = "3d_animation"
    GAME_CINEMATIC = "game_cinematic"
    CONCEPT_ART = "concept_art"
    DIGITAL_ILLUSTRATION = "digital_illustration"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    INK_WASH = "ink_wash"
    SHADOW_PUPPET = "shadow_puppet"


_STYLE_DEFINITIONS: dict[BaseStyle, dict[str, Any]] = {
    BaseStyle.PHOTOREALISTIC: {
        "label": "a photograph",
        "positive": (
            "photorealistic live-action cinematic film still, real adult skin texture, "
            "physically based materials, photographic optics and natural anatomy"
        ),
        "aliases": (
            "photoreal", "photo-real", "live action", "live-action", "realistic film",
            "写实", "真人", "实拍", "真实电影", "真实皮肤", "摄影质感",
        ),
    },
    BaseStyle.CINEMATIC: {
        "label": "a live action movie still",
        "positive": (
            "live-action cinematic movie still, photographic lens rendering, "
            "natural materials and film color science"
        ),
        "aliases": ("cinematic", "film still", "电影质感", "电影感", "电影风格"),
    },
    BaseStyle.ANIME: {
        "label": "an anime drawing",
        "positive": "Japanese anime cel illustration, intentional drawn line art and cel shading",
        "aliases": ("anime", "manga", "cel animation", "cel-shaded", "赛璐璐", "二次元", "日漫"),
    },
    BaseStyle.DONGHUA: {
        "label": "a Chinese animation drawing",
        "positive": "Chinese donghua animation illustration with authored drawn rendering",
        "aliases": ("donghua", "国漫", "中国动画"),
    },
    BaseStyle.THREE_D_ANIMATION: {
        "label": "a 3d animation render",
        "positive": "stylized three-dimensional animated film rendering",
        "aliases": ("3d animation", "3d animated", "三维动画", "3d动画"),
    },
    BaseStyle.GAME_CINEMATIC: {
        "label": "a video game render",
        "positive": "high-end video game cinematic rendering with real-time computer graphics",
        "aliases": ("game cinematic", "video game render", "游戏过场", "游戏电影", "游戏cg"),
    },
    BaseStyle.CONCEPT_ART: {
        "label": "concept art",
        "positive": "painted production concept art with authored brushwork",
        "aliases": ("concept art", "概念艺术", "概念设计", "设定图"),
    },
    BaseStyle.DIGITAL_ILLUSTRATION: {
        "label": "a digital illustration",
        "positive": "two-dimensional digital illustration with intentional drawn rendering",
        "aliases": ("digital illustration", "digital art", "数字插画", "数字绘画", "插画"),
    },
    BaseStyle.OIL_PAINTING: {
        "label": "an oil painting",
        "positive": "traditional oil painting on canvas with visible paint texture",
        "aliases": ("oil painting", "oil paint", "油画"),
    },
    BaseStyle.WATERCOLOR: {
        "label": "a watercolor painting",
        "positive": "traditional watercolor painting on paper with translucent pigment",
        "aliases": ("watercolor", "watercolour", "水彩"),
    },
    BaseStyle.INK_WASH: {
        "label": "an ink wash painting",
        "positive": "traditional Chinese ink wash painting on paper",
        "aliases": ("ink wash", "sumi-e", "水墨", "水墨画", "墨韵"),
    },
    BaseStyle.SHADOW_PUPPET: {
        "label": "a shadow puppet image",
        "positive": (
            "traditional Chinese shadow-puppet image, flat backlit cutout silhouettes "
            "on a translucent stage screen"
        ),
        "aliases": ("shadow puppet", "shadow-puppet", "皮影", "皮影戏", "幕布剪影"),
    },
}

_INFERENCE_ORDER = (
    BaseStyle.SHADOW_PUPPET,
    BaseStyle.INK_WASH,
    BaseStyle.WATERCOLOR,
    BaseStyle.OIL_PAINTING,
    BaseStyle.GAME_CINEMATIC,
    BaseStyle.THREE_D_ANIMATION,
    BaseStyle.DONGHUA,
    BaseStyle.ANIME,
    BaseStyle.CONCEPT_ART,
    BaseStyle.DIGITAL_ILLUSTRATION,
    BaseStyle.PHOTOREALISTIC,
    BaseStyle.CINEMATIC,
)

_COMPATIBLE_STYLES = {
    BaseStyle.PHOTOREALISTIC: {
        BaseStyle.PHOTOREALISTIC,
        BaseStyle.CINEMATIC,
    },
    BaseStyle.CINEMATIC: {
        BaseStyle.PHOTOREALISTIC,
        BaseStyle.CINEMATIC,
    },
}


def normalize_base_style(value: object) -> BaseStyle:
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return BaseStyle(normalized)
    except ValueError as exc:
        raise ValueError(f"unknown visual base_style: {value!r}") from exc


def infer_base_style(text: object) -> BaseStyle:
    normalized = re.sub(r"\s+", " ", str(text or "")).casefold()
    for base_style in _INFERENCE_ORDER:
        if any(alias.casefold() in normalized for alias in _STYLE_DEFINITIONS[base_style]["aliases"]):
            return base_style
    return BaseStyle.CINEMATIC


def style_classification_labels() -> dict[str, str]:
    """Return the fixed custom LabelTable vocabulary in stable enum order."""
    return {
        base_style.value: str(_STYLE_DEFINITIONS[base_style]["label"])
        for base_style in BaseStyle
    }


def _negative_prompt(base_style: BaseStyle) -> str:
    incompatible = [
        style.value
        for style in BaseStyle
        if style not in _COMPATIBLE_STYLES.get(base_style, {base_style})
    ]
    return "forbid rendering-medium drift into: " + ", ".join(incompatible)


def build_visual_style_contract(style: VisualStyle) -> dict[str, Any]:
    source_text = " ".join(
        value
        for value in (
            style.style_prompt_short,
            style.style_prompt_full,
            " ".join(style.style_tags),
            " ".join(style.tags),
            style.name,
        )
        if str(value or "").strip()
    )
    base_style = (
        normalize_base_style(style.base_style)
        if str(style.base_style or "").strip()
        else infer_base_style(source_text)
    )
    style_tags = list(
        dict.fromkeys(
            str(value).strip().casefold().replace(" ", "_")
            for value in [*style.style_tags, *style.tags]
            if str(value).strip()
        )
    )
    return {
        "schema": BASE_STYLE_SCHEMA,
        "base_style": base_style.value,
        "style_tags": style_tags,
        "positive_prompt": str(_STYLE_DEFINITIONS[base_style]["positive"]),
        "negative_prompt": _negative_prompt(base_style),
        "classification_label": str(_STYLE_DEFINITIONS[base_style]["label"]),
        "allowed_reference_styles": sorted(
            style.value
            for style in _COMPATIBLE_STYLES.get(base_style, {base_style})
        ),
        "reference_policy_version": STYLE_REFERENCE_POLICY_VERSION,
    }


def style_reference_compatible(
    expected_base_style: object,
    classification: dict[str, Any],
    *,
    min_margin: float = STYLE_REFERENCE_MIN_MARGIN,
) -> bool:
    """Reject only a material fixed-enum mismatch, not an ambiguous CLIP vote."""
    expected = normalize_base_style(expected_base_style)
    if classification.get("schema") != CLIP_STYLE_SCHEMA:
        raise ValueError("style classification schema mismatch")
    if classification.get("status") != "done":
        raise ValueError("style classification is not complete")
    rankings = classification.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        raise ValueError("style classification rankings are empty")
    scores: dict[BaseStyle, float] = {}
    for item in rankings:
        if not isinstance(item, dict):
            raise ValueError("style classification ranking must be an object")
        style = normalize_base_style(item.get("base_style"))
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("style classification score must be numeric") from exc
        scores[style] = score
    top_style = normalize_base_style(classification.get("top_style"))
    compatible = _COMPATIBLE_STYLES.get(expected, {expected})
    if top_style in compatible:
        return True
    expected_scores = [scores[style] for style in compatible if style in scores]
    if not expected_scores or top_style not in scores:
        return False
    return scores[top_style] - max(expected_scores) < float(min_margin)
