"""Create, load, and apply portable visual design systems."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from utils.config import ARK_AGENT_API_KEY_ENV, ARK_BASE_URL, DEFAULT_TEXT_MODEL
from utils.visual_style_spec import VisualStyle as VisualStyleData
from utils.visual_style_spec import parse_visual_style


_CREATE_SYSTEM_PROMPT = """You create portable visual design systems for AI video generation.
Return only a complete visual-style.md document with YAML frontmatter delimited by ---.
It must follow visual-style.md spec 1.0 and include: name, version, tags,
style_prompt_short, a detailed standalone style_prompt_full, at least two primary
colors plus accent and neutral colors (all with name/hex/role), typography display/body/caption,
layout, motion, and mood including avoid rules. style_prompt_full must explicitly cover colors,
lighting, composition, texture, motion, mood, and anti-patterns so it can be used verbatim by an AI tool.
Do not include a fenced code block or explanatory text."""


class VisualStyle(BaseTool):
    name = "visual_style"
    version = "0.1.0"
    # BaseTool currently calls its graphics/generation tier GENERATE.
    tier = ToolTier.GENERATE
    capability = "visual_style"
    provider = "honcut"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {"type": "string", "enum": ["create", "load", "apply"]},
            "style_path": {
                "type": "string",
                "description": "Path to visual-style.md (for load/apply)",
            },
            "script_text": {
                "type": "string",
                "description": "Script/storyboard text (for create)",
            },
            "shot_prompt": {
                "type": "string",
                "description": "Shot prompt to enrich (for apply)",
            },
            "output_path": {
                "type": "string",
                "description": "Where to save created style (for create)",
            },
        },
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation")
        try:
            if operation == "create":
                return self._create(inputs)
            if operation == "load":
                return ToolResult(success=True, data=self._load(inputs))
            if operation == "apply":
                style = inputs.get("visual_style")
                if style is None:
                    style = self._load(inputs)
                if not isinstance(style, VisualStyleData):
                    raise ValueError("visual_style must be a VisualStyle instance")
                shot_prompt = str(inputs.get("shot_prompt", "")).strip()
                if not shot_prompt:
                    raise ValueError("shot_prompt is required for apply")
                enriched = shot_prompt
                if style.style_prompt_full:
                    enriched = f"{shot_prompt}\n\nVisual style: {style.style_prompt_full}"
                return ToolResult(success=True, data=enriched)
            return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _load(inputs: dict[str, Any]) -> VisualStyleData:
        raw_path = inputs.get("style_path")
        if not raw_path:
            raise ValueError("style_path is required for load/apply")
        style_path = Path(str(raw_path))
        if not style_path.is_file():
            raise FileNotFoundError(f"Visual style not found: {style_path}")
        return parse_visual_style(style_path.read_text(encoding="utf-8"))

    @staticmethod
    def _create(inputs: dict[str, Any]) -> ToolResult:
        script_text = str(inputs.get("script_text", "")).strip()
        if not script_text:
            raise ValueError("script_text is required for create")
        api_key = os.environ.get(ARK_AGENT_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{ARK_AGENT_API_KEY_ENV} is required for create")

        client = OpenAI(api_key=api_key, base_url=ARK_BASE_URL)
        response = client.chat.completions.create(
            model=DEFAULT_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _CREATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this script/storyboard:\n\n{script_text}"},
            ],
            timeout=60,
        )
        md_text = response.choices[0].message.content or ""
        md_text = re.sub(r"^```(?:ya?ml|markdown)?\s*", "", md_text.strip())
        md_text = re.sub(r"\s*```$", "", md_text).strip() + "\n"
        # Validate the model output before writing an artifact.
        style = parse_visual_style(md_text)
        if not style.style_prompt_full:
            raise ValueError("Generated visual style is missing style_prompt_full")

        output_path = Path(str(inputs.get("output_path") or "visual-style.md"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_text, encoding="utf-8")
        return ToolResult(
            success=True,
            data={"output": str(output_path), "visual_style": style},
            artifacts=[str(output_path)],
            model=DEFAULT_TEXT_MODEL,
        )
