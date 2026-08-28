"""Ark Responses client for ordered image/video/document/audio review."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from openai import DefaultHttpxClient, OpenAI
from pydantic import BaseModel

from schemas.understanding import (
    native_json_schema_format,
    parse_structured_output,
)
from utils.config import (
    ARK_RESPONSES_BASE_URL,
    DEFAULT_MULTIMODAL_MODEL,
    get_api_key,
)
from utils.prompt_budget import enforce_prompt_budget

StructuredReviewT = TypeVar("StructuredReviewT", bound=BaseModel)
_ENVELOPE_AUDIT_LIMIT = 8
_SAFE_ENVELOPE_TOKENS = frozenset({
    "assistant",
    "cancelled",
    "completed",
    "failed",
    "in_progress",
    "incomplete",
    "message",
    "output_text",
    "queued",
    "reasoning",
    "refusal",
})


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_envelope_token(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    return value if value in _SAFE_ENVELOPE_TOKENS else "<unknown>"


def _response_envelope_audit(response: Any) -> dict[str, Any]:
    """Describe the response shape without retaining Provider text."""

    output = _response_field(response, "output")
    output_items = list(output) if isinstance(output, (list, tuple)) else []
    inspected_items = output_items[:_ENVELOPE_AUDIT_LIMIT]
    messages = [
        item
        for item in inspected_items
        if _response_field(item, "type") == "message"
    ]
    content_blocks: list[Any] = []
    for message in messages:
        content = _response_field(message, "content")
        if isinstance(content, (list, tuple)):
            content_blocks.extend(content[:_ENVELOPE_AUDIT_LIMIT])
    output_texts = [
        _response_field(block, "text")
        for block in content_blocks
        if _response_field(block, "type") == "output_text"
    ]
    return {
        "response_status": _safe_envelope_token(
            _response_field(response, "status")
        ),
        "error_present": _response_field(response, "error") is not None,
        "incomplete_details_present": (
            _response_field(response, "incomplete_details") is not None
        ),
        "output_is_sequence": isinstance(output, (list, tuple)),
        "output_count": len(output_items),
        "output_types": [
            _safe_envelope_token(_response_field(item, "type"))
            for item in inspected_items
        ],
        "message_roles": [
            _safe_envelope_token(_response_field(message, "role"))
            for message in messages
        ],
        "message_statuses": [
            _safe_envelope_token(_response_field(message, "status"))
            for message in messages
        ],
        "content_block_count": len(content_blocks),
        "content_types": [
            _safe_envelope_token(_response_field(block, "type"))
            for block in content_blocks
        ],
        "output_text_count": len(output_texts),
        "output_text_utf8_lengths": [
            len(text.encode("utf-8")) if isinstance(text, str) else None
            for text in output_texts
        ],
        "output_text_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if isinstance(text, str)
            else None
            for text in output_texts
        ],
        "audit_truncated": (
            len(output_items) > _ENVELOPE_AUDIT_LIMIT
            or len(content_blocks) > _ENVELOPE_AUDIT_LIMIT
        ),
    }


def _reject_response_envelope(reason: str, response: Any) -> None:
    audit = json.dumps(
        _response_envelope_audit(response),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    raise json.JSONDecodeError(
        f"ARK multimodal response envelope rejected: {reason}; envelope={audit}",
        "",
        0,
    )


def _extract_single_completed_output_text(response: Any) -> str:
    """Return one authoritative message block without SDK aggregation."""

    if _response_field(response, "status") != "completed":
        _reject_response_envelope("response status must be completed", response)
    if _response_field(response, "error") is not None:
        _reject_response_envelope(
            "completed response must not contain error",
            response,
        )
    if _response_field(response, "incomplete_details") is not None:
        _reject_response_envelope(
            "completed response must not contain incomplete details",
            response,
        )

    output = _response_field(response, "output")
    if not isinstance(output, (list, tuple)) or len(output) != 1:
        _reject_response_envelope(
            "expected exactly one assistant message output item",
            response,
        )
    message = output[0]
    if _response_field(message, "type") != "message":
        _reject_response_envelope(
            "expected exactly one assistant message output item",
            response,
        )
    if _response_field(message, "role") != "assistant":
        _reject_response_envelope("message role must be assistant", response)
    if _response_field(message, "status") != "completed":
        _reject_response_envelope("message status must be completed", response)

    content = _response_field(message, "content")
    if not isinstance(content, (list, tuple)) or len(content) != 1:
        _reject_response_envelope(
            "expected exactly one output_text content block",
            response,
        )
    block = content[0]
    if _response_field(block, "type") != "output_text":
        _reject_response_envelope(
            "expected exactly one output_text content block",
            response,
        )
    text = _response_field(block, "text")
    if not isinstance(text, str) or not text.strip():
        _reject_response_envelope("output_text must be non-empty", response)
    return text


def review_as(
    client: Any,
    media_paths: list[Path],
    prompt: str,
    response_model: type[StructuredReviewT],
) -> StructuredReviewT:
    """Use native structured output, with a narrow adapter for test reviewers."""

    structured_review = getattr(client, "review_structured", None)
    if callable(structured_review):
        return structured_review(media_paths, prompt, response_model)
    raw = client.review(media_paths, prompt)
    return parse_structured_output(raw, response_model)


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
            # Ark endpoints are direct-routed.  Do not let an unrelated
            # desktop SOCKS proxy become an undeclared production dependency.
            http_client=DefaultHttpxClient(
                timeout=timeout_s,
                trust_env=False,
            ),
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
        return self._review_media_with_format(
            media_paths,
            prompt,
            {"type": "json_object"},
        )

    def review_structured(
        self,
        media_paths: list[Path],
        prompt: str,
        response_model: type[StructuredReviewT],
    ) -> StructuredReviewT:
        """Return a schema-validated understanding object, never free text."""

        raw = self._review_media_with_format(
            media_paths,
            prompt,
            native_json_schema_format(response_model),
        )
        return parse_structured_output(raw, response_model)

    def _review_media_with_format(
        self,
        media_paths: list[Path],
        prompt: str,
        text_format: dict[str, Any],
    ) -> str:
        """Execute one Responses request using the supplied output contract."""
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
                    text={"format": text_format},
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
        return _extract_single_completed_output_text(payload)
