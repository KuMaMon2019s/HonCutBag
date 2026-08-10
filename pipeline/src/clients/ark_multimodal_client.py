"""Multi-image review client for the OpenAI-compatible Volcano Ark endpoint."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils.config import ARK_BASE_URL, DEFAULT_TEXT_MODEL, get_api_key


class ArkMultimodalClient:
    """Send a prompt and multiple local images to an Ark vision-capable model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        key = api_key or get_api_key("ARK_AGENT_API_KEY")
        if client is None and not key:
            raise RuntimeError("ARK_AGENT_API_KEY is required for real storyboard review")
        self.model = model or os.environ.get(
            "HONCUT_STORYBOARD_REVIEW_MODEL", DEFAULT_TEXT_MODEL
        )
        self.client = client or OpenAI(
            api_key=key,
            base_url=base_url
            or os.environ.get("HONCUT_STORYBOARD_REVIEW_BASE_URL", ARK_BASE_URL),
        )

    @staticmethod
    def _image_url(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"storyboard image not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def review(self, image_paths: list[Path], prompt: str) -> str:
        """Return the model's textual review for the complete ordered image set."""
        if not image_paths:
            raise ValueError("at least one storyboard image is required")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for position, path in enumerate(image_paths, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Storyboard image {position}: {path.stem}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_url(path)},
                    },
                ]
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("ARK multimodal review returned empty content")
        return result
