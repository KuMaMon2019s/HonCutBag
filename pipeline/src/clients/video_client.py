"""Unified dual-track video generation client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from utils.config import get_bridge_api_url, get_video_route


DirectGenerator = Callable[..., Any]


@dataclass(frozen=True)
class VideoResult:
    """Normalized result returned by :class:`VideoClient`."""

    provider: str
    route: str
    task_id: str | None = None
    output_path: str | None = None
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoClient:
    """Route video generation through Bridge or a provider fallback."""

    def __init__(self, provider: str = "auto", direct_generator: DirectGenerator | None = None):
        self.provider = provider.strip().lower()
        self.mode = get_video_route(self.provider)
        self.bridge_url = get_bridge_api_url()
        self.direct_generator = direct_generator

    def generate(self, prompt: str, **kwargs: Any) -> VideoResult:
        if self.mode == "bridge":
            return self._generate_via_bridge(prompt, **kwargs)
        return self._generate_direct(prompt, **kwargs)

    def _generate_via_bridge(self, prompt: str, **kwargs: Any) -> VideoResult:
        # Reuse the established Bridge session, polling, download, and proxy bypass.
        from clients import local_video_client

        bridge_kwargs = self._bridge_kwargs(kwargs)
        output_path = bridge_kwargs.pop("output_path", None)
        bridge_kwargs.setdefault("model", self._bridge_model())
        if output_path:
            value = local_video_client.generate_video(prompt, output_path, **bridge_kwargs)
            return VideoResult(
                provider=self.provider,
                route="bridge",
                output_path=str(value),
                value=value,
            )
        bridge_kwargs.pop("duration", None)
        bridge_kwargs.pop("reference_image_base64", None)
        task_id = local_video_client.submit(prompt=prompt, **bridge_kwargs)
        if not task_id:
            raise RuntimeError(f"Bridge did not return a task id for provider {self.provider}")
        return VideoResult(
            provider=self.provider,
            route="bridge",
            task_id=str(task_id),
            value=str(task_id),
        )

    def _generate_direct(self, prompt: str, **kwargs: Any) -> VideoResult:
        if self.direct_generator is None:
            raise RuntimeError(f"No direct/local fallback configured for provider {self.provider}")
        value = self.direct_generator(prompt=prompt, **kwargs)
        output_path = kwargs.get("output_path")
        task_id = None if output_path else str(value)
        return VideoResult(
            provider=self.provider,
            route=self.mode,
            task_id=task_id,
            output_path=str(value) if output_path else None,
            value=value,
        )

    def _bridge_model(self) -> str:
        return {"wan": "wan22"}.get(self.provider, self.provider)

    def _bridge_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate provider arguments to the established Bridge contract."""
        allowed = {
            "output_path",
            "reference_image_base64",
            "seed",
            "duration",
            "width",
            "height",
            "fps",
            "asset_zip_path",
            "image_base64_list",
            "content",
            "batch_id",
            "model",
        }
        bridge_kwargs = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
        if self.provider == "seedance":
            reference = kwargs.get("reference_image_base64") or kwargs.get("first_frame_base64")
            if reference is not None:
                bridge_kwargs["reference_image_base64"] = reference
        return bridge_kwargs
