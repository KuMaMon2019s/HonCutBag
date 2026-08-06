"""Client for the Bridge v3.2 InvokeAI asset import endpoint."""

from __future__ import annotations

from typing import Any

from clients.video_client import _AsyncRequestsSession, _response_json
from utils.config import get_bridge_api_url


class InvokeAIClient:
    """Import images from an InvokeAI gallery into Bridge Assets."""

    def __init__(
        self,
        bridge_url: str | None = None,
        session: Any | None = None,
    ) -> None:
        self.bridge_url = (bridge_url or get_bridge_api_url()).rstrip("/")
        self.session = session or _AsyncRequestsSession()

    async def import_from_invokeai(
        self,
        asset_id: str,
        image_names: list[str],
        asset_type: str = "character",
        target_view: str = "front",
        invokeai_url: str = "http://127.0.0.1:9090",
    ) -> dict[str, Any]:
        """Import named InvokeAI Gallery images into a Bridge asset."""
        payload = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "image_names": image_names,
            "target_view": target_view,
            "invokeai_url": invokeai_url,
        }
        response = await self.session.post(
            f"{self.bridge_url}/import_from_invokeai", json=payload
        )
        return await _response_json(response)
