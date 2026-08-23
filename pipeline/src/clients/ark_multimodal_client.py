"""Ark Responses client for ordered image/video/document/audio review."""

from __future__ import annotations

import mimetypes
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from utils.config import (
    ARK_RESPONSES_BASE_URL,
    DEFAULT_MULTIMODAL_MODEL,
    get_api_key,
)
from utils.prompt_budget import enforce_prompt_budget


class ArkMultimodalClient:
    """Send ordered local media to an Ark multimodal understanding model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        media_url_resolver: Callable[[Path], str] | None = None,
    ) -> None:
        key = api_key or get_api_key("ARK_AGENT_API_KEY")
        if client is None and not key:
            raise RuntimeError("ARK_AGENT_API_KEY is required for real storyboard review")
        self.model = model or os.environ.get(
            "HONCUT_STORYBOARD_REVIEW_MODEL", DEFAULT_MULTIMODAL_MODEL
        )
        self.max_tokens = int(
            os.environ.get("HONCUT_STORYBOARD_REVIEW_MAX_TOKENS", "4096")
        )
        if self.max_tokens <= 0:
            raise ValueError("HONCUT_STORYBOARD_REVIEW_MAX_TOKENS must be positive")
        self.thinking_type = os.environ.get(
            "HONCUT_STORYBOARD_REVIEW_THINKING", "disabled"
        ).strip().lower()
        if self.thinking_type not in {"disabled", "enabled", "auto"}:
            raise ValueError(
                "HONCUT_STORYBOARD_REVIEW_THINKING must be disabled, enabled, or auto"
            )
        timeout_s = float(os.environ.get("HONCUT_STORYBOARD_REVIEW_TIMEOUT_S", "120"))
        if timeout_s <= 0:
            raise ValueError("HONCUT_STORYBOARD_REVIEW_TIMEOUT_S must be positive")
        self.wall_timeout_s = float(
            os.environ.get("HONCUT_STORYBOARD_REVIEW_WALL_TIMEOUT_S", "240")
        )
        if self.wall_timeout_s <= 0:
            raise ValueError("HONCUT_STORYBOARD_REVIEW_WALL_TIMEOUT_S must be positive")
        self.client = client or OpenAI(
            api_key=key,
            base_url=base_url or ARK_RESPONSES_BASE_URL,
            timeout=timeout_s,
            max_retries=0,
        )
        self._media_url_resolver = media_url_resolver or self._upload_media

    @staticmethod
    def _upload_media(path: Path) -> str:
        from clients.tos_uploader import upload_multimodal_media_file_required

        return upload_multimodal_media_file_required(
            path,
            label=f"multimodal input {path.name}",
        )

    @staticmethod
    def _media_kind(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or ""
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("audio/"):
            return "audio"
        if path.suffix.lower() == ".pdf":
            return "document"
        raise ValueError(f"unsupported multimodal input format: {path.name}")

    def _content_item(self, path: Path) -> tuple[str, dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"multimodal input not found: {path}")
        kind = self._media_kind(path)
        media_url = self._media_url_resolver(path)
        if kind == "image":
            return kind, {
                "type": "input_image",
                "image_url": media_url,
                "detail": "high",
            }
        if kind == "video":
            fps = float(os.environ.get("HONCUT_MULTIMODAL_VIDEO_FPS", "1"))
            if not 0.2 <= fps <= 5:
                raise ValueError("HONCUT_MULTIMODAL_VIDEO_FPS must be between 0.2 and 5")
            return kind, {
                "type": "input_video",
                "video_url": media_url,
                "fps": fps,
            }
        if kind == "document":
            return kind, {"type": "input_file", "file_url": media_url}
        return kind, {"type": "input_audio", "audio_url": media_url}

    def review(self, image_paths: list[Path], prompt: str) -> str:
        """Return the model's textual review for the complete ordered image set."""
        if not image_paths:
            raise ValueError("at least one storyboard image is required")
        if any(self._media_kind(Path(path)) != "image" for path in image_paths):
            raise ValueError("review() accepts images only; use review_media()")
        return self.review_media(image_paths, prompt)

    def review_media(self, media_paths: list[Path], prompt: str) -> str:
        """Review an ordered mixed-media set through Ark Responses API."""
        if not media_paths:
            raise ValueError("at least one multimodal input is required")
        normalized_paths = [Path(path) for path in media_paths]
        kinds = [self._media_kind(path) for path in normalized_paths]
        counters: dict[str, int] = {}
        labels = []
        for path, kind in zip(normalized_paths, kinds, strict=True):
            counters[kind] = counters.get(kind, 0) + 1
            labels.append(f"Input {kind} {counters[kind]}: {path.stem}")
        enforce_prompt_budget(
            "\n".join((prompt, *labels)),
            provider="ark",
            model=self.model,
            purpose="multimodal_review",
        )

        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path, label in zip(normalized_paths, labels, strict=True):
            _kind, media_item = self._content_item(path)
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    media_item,
                ]
            )

        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def _request() -> None:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    text={"format": {"type": "json_object"}},
                    max_output_tokens=self.max_tokens,
                    extra_body={"thinking": {"type": self.thinking_type}},
                )
                result_queue.put((True, response))
            except BaseException as exc:
                result_queue.put((False, exc))

        # A daemon worker gives this call a true wall-clock ceiling even when
        # server keepalives repeatedly reset the SDK's socket read timeout.
        threading.Thread(target=_request, daemon=True).start()
        try:
            succeeded, payload = result_queue.get(timeout=self.wall_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                "ARK multimodal review wall timeout after "
                f"{self.wall_timeout_s:g}s"
            ) from exc
        if not succeeded:
            raise payload
        response = payload
        result = getattr(response, "output_text", None)
        if not isinstance(result, str) or not result.strip():
            fragments: list[str] = []
            for item in getattr(response, "output", []) or []:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if item_type != "message":
                    continue
                item_content = (
                    item.get("content", [])
                    if isinstance(item, dict)
                    else getattr(item, "content", [])
                )
                for block in item_content or []:
                    block_type = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if block_type != "output_text":
                        continue
                    text = (
                        block.get("text")
                        if isinstance(block, dict)
                        else getattr(block, "text", None)
                    )
                    if isinstance(text, str) and text.strip():
                        fragments.append(text)
            result = "\n".join(fragments)
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("ARK multimodal review returned empty content")
        return result
