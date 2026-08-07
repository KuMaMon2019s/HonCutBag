"""Parser and serializer for the portable ``visual-style.md`` format."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ColorEntry:
    name: str
    hex: str
    role: str


@dataclass
class TypographyEntry:
    family: str
    weight: str = "regular"
    style: str = ""


@dataclass
class VisualStyle:
    name: str
    version: str = "1.0"
    style_prompt_short: str = ""
    style_prompt_full: str = ""
    colors_primary: list[ColorEntry] = field(default_factory=list)
    colors_accent: list[ColorEntry] = field(default_factory=list)
    colors_neutral: list[ColorEntry] = field(default_factory=list)
    typography_display: Optional[TypographyEntry] = None
    typography_body: Optional[TypographyEntry] = None
    typography_caption: Optional[TypographyEntry] = None
    typography_rules: list[str] = field(default_factory=list)
    layout_grid: str = ""
    layout_alignment: str = ""
    layout_aspect_ratio: str = "16:9"
    motion_transitions: list[str] = field(default_factory=list)
    motion_animation_style: str = ""
    motion_pacing: str = ""
    mood_keywords: list[str] = field(default_factory=list)
    mood_era: str = ""
    mood_avoid: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _frontmatter(md_text: str) -> dict[str, Any]:
    lines = md_text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("visual-style.md must start with YAML frontmatter")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("visual-style.md frontmatter is missing its closing ---") from exc
    data = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(data, dict):
        raise ValueError("visual-style.md frontmatter must be a YAML mapping")
    return data


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _colors(value: Any) -> list[ColorEntry]:
    entries = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            entries.append(
                ColorEntry(
                    name=str(item.get("name", "")),
                    hex=str(item.get("hex", "")),
                    role=str(item.get("role", "")),
                )
            )
    return entries


def _typography(value: Any) -> Optional[TypographyEntry]:
    if not isinstance(value, dict) or not value:
        return None
    return TypographyEntry(
        family=str(value.get("family", "")),
        weight=str(value.get("weight", "regular")),
        style=str(value.get("style", "")),
    )


def parse_visual_style(md_text: str) -> VisualStyle:
    """Parse YAML frontmatter from an OpenMontage-compatible style file."""
    data = _frontmatter(md_text)
    if not data.get("name"):
        raise ValueError("visual-style.md is missing required field: name")

    colors = _mapping(data.get("colors"))
    typography = _mapping(data.get("typography"))
    layout = _mapping(data.get("layout"))
    motion = _mapping(data.get("motion"))
    mood = _mapping(data.get("mood"))
    return VisualStyle(
        name=str(data["name"]),
        version=str(data.get("version", "1.0")),
        style_prompt_short=str(data.get("style_prompt_short", "") or ""),
        style_prompt_full=str(data.get("style_prompt_full", "") or ""),
        colors_primary=_colors(colors.get("primary")),
        colors_accent=_colors(colors.get("accent")),
        colors_neutral=_colors(colors.get("neutral")),
        typography_display=_typography(typography.get("display")),
        typography_body=_typography(typography.get("body")),
        typography_caption=_typography(typography.get("caption")),
        typography_rules=_strings(typography.get("rules")),
        layout_grid=str(layout.get("grid", "") or ""),
        layout_alignment=str(layout.get("alignment", "") or ""),
        layout_aspect_ratio=str(layout.get("aspect_ratio", "16:9") or "16:9"),
        motion_transitions=_strings(motion.get("transitions")),
        motion_animation_style=str(motion.get("animation_style", "") or ""),
        motion_pacing=str(motion.get("pacing", "") or ""),
        mood_keywords=_strings(mood.get("keywords")),
        mood_era=str(mood.get("era", "") or ""),
        mood_avoid=_strings(mood.get("avoid")),
        tags=_strings(data.get("tags")),
    )


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    return value


def serialize_visual_style(style: VisualStyle) -> str:
    """Serialize a :class:`VisualStyle` to OpenMontage-compatible Markdown."""
    typography: dict[str, Any] = {
        "display": asdict(style.typography_display) if style.typography_display else None,
        "body": asdict(style.typography_body) if style.typography_body else None,
        "caption": asdict(style.typography_caption) if style.typography_caption else None,
        "rules": style.typography_rules,
    }
    data = {
        "name": style.name,
        "version": style.version,
        "tags": style.tags,
        "style_prompt_short": style.style_prompt_short,
        "style_prompt_full": style.style_prompt_full,
        "colors": {
            "primary": [asdict(entry) for entry in style.colors_primary],
            "accent": [asdict(entry) for entry in style.colors_accent],
            "neutral": [asdict(entry) for entry in style.colors_neutral],
        },
        "typography": typography,
        "layout": {
            "grid": style.layout_grid,
            "alignment": style.layout_alignment,
            "aspect_ratio": style.layout_aspect_ratio,
        },
        "motion": {
            "transitions": style.motion_transitions,
            "animation_style": style.motion_animation_style,
            "pacing": style.motion_pacing,
        },
        "mood": {
            "keywords": style.mood_keywords,
            "era": style.mood_era,
            "avoid": style.mood_avoid,
        },
    }
    yaml_text = yaml.safe_dump(
        _without_none(data), sort_keys=False, allow_unicode=True, width=1000
    ).rstrip()
    return f"---\n{yaml_text}\n---\n"
