"""Asset-aware Phase 5 video generation routing."""

from __future__ import annotations

import inspect
from collections.abc import MutableMapping
from typing import Any

from clients.video_client import VideoClient


def _read_state(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, MutableMapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _write_state(state: Any, name: str, value: Any) -> None:
    if isinstance(state, MutableMapping):
        state[name] = value
    else:
        setattr(state, name, value)


class VideoGenerator:
    """Generate Phase 5 video, preferring Bridge assets when available."""

    def __init__(self, video_client: VideoClient | None = None) -> None:
        self.video_client = video_client or VideoClient()

    async def run(self, state: Any) -> Any:
        asset_id = _read_state(state, "character_asset_id") or _read_state(state, "asset_id")
        prompt = _read_state(state, "prompt", "")
        if asset_id:
            result = await self.video_client.generate_with_assets(
                asset_id=asset_id,
                asset_type="character",
                image_index=0,
                model=_read_state(state, "model") or "wan22",
                prompt=prompt,
            )
        else:
            result = self.video_client.generate(
                prompt=prompt,
                reference_images=_read_state(state, "reference_images", []),
            )
            if inspect.isawaitable(result):
                result = await result
        _write_state(state, "video_result", result)
        return state
