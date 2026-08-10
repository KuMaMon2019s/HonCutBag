"""Route a visual-style document to task-specific prompt slices."""

from __future__ import annotations

import re


_KEYWORDS = {
    "character": (
        "character", "appearance", "outfit", "costume", "clothing", "wardrobe",
        "body", "face", "hair", "人物", "角色", "外貌", "外观", "服装", "衣着",
        "体型", "身体", "面部", "发型",
    ),
    "scene": (
        "environment", "setting", "location", "lighting", "light", "rain", "weather",
        "palette", "color", "colour", "环境", "场景", "地点", "光影", "光照", "灯光",
        "雨", "天气", "调色板", "色彩", "色调",
    ),
    "storyboard": (
        "camera", "composition", "framing", "frame", "shot", "lens", "angle",
        "perspective", "depth of field", "镜头", "摄影机", "构图", "取景", "景别",
        "焦距", "视角", "景深",
    ),
    "video": (
        "motion", "movement", "action", "physics", "tempo", "pace", "animation",
        "gesture", "动作", "运动", "动态", "物理", "节奏", "速度", "动画",
    ),
    "global": (
        "palette", "color", "colour", "mood", "tone", "atmosphere", "lighting",
        "light", "rain", "weather", "environment", "调色板", "色彩", "色调", "情绪",
        "氛围", "基调", "光影", "光照", "雨", "天气", "环境",
    ),
}


def _blocks(style_text: str) -> list[str]:
    """Keep Markdown sections together, otherwise split on blank lines."""
    text = str(style_text or "").strip()
    if not text:
        return []
    # ``visual-style.md`` may be pure prose or YAML frontmatter. Treat YAML
    # fields independently so unrelated prompt fields do not become one blob.
    if text.startswith("---\n") and text.rstrip().endswith("---"):
        return [line.strip() for line in text.splitlines() if line.strip() != "---"]
    chunks = re.split(r"(?=^#{1,6}\s+)", text, flags=re.MULTILINE)
    blocks: list[str] = []
    for chunk in chunks:
        blocks.extend(part.strip() for part in re.split(r"\n\s*\n", chunk) if part.strip())
    return blocks


def split_visual_style(style_text: str) -> dict[str, str]:
    """Split style prose using generic task vocabulary.

    A block may intentionally appear in more than one slice. In particular,
    environment, lighting, weather, and palette guidance is both scene-level
    and global guidance.
    """
    routed: dict[str, list[str]] = {key: [] for key in _KEYWORDS}
    for block in _blocks(style_text):
        lowered = block.casefold()
        for task_type, keywords in _KEYWORDS.items():
            if any(keyword.casefold() in lowered for keyword in keywords):
                routed[task_type].append(block)
    return {key: "\n\n".join(parts) for key, parts in routed.items()}


def get_slice(style_text: str, task_type: str) -> str:
    """Return a task slice plus global guidance, falling back to the source."""
    source = str(style_text or "").strip()
    normalized_type = str(task_type or "").strip().casefold()
    if normalized_type not in {"character", "scene", "storyboard", "video"}:
        return source

    slices = split_visual_style(source)
    task_slice = slices[normalized_type]
    if not task_slice:
        return source

    parts = [task_slice]
    global_slice = slices["global"]
    if global_slice and global_slice != task_slice:
        parts.append(global_slice)
    return "\n\n".join(parts)
