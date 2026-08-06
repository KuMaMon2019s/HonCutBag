"""Volcano Ark OpenAI-compatible text-to-speech narration client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


class TTSNarration:
    """Generate narration through Ark's ``/audio/speech`` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        session: Any = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ARK_AGENT_API_KEY")
        self.base_url = (base_url or os.environ.get(
            "ARK_TTS_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3"
        )).rstrip("/")
        self.model = model or os.environ.get("ARK_TTS_MODEL", "doubao-tts")
        self.voice = voice or os.environ.get("ARK_TTS_VOICE", "zh_female_vv_uranus_bigtts")
        self.session = session or requests.Session()
        self.timeout = timeout

    def generate(
        self,
        text: str,
        output_path: str,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> str:
        """Synthesize non-empty text and atomically write returned audio bytes."""
        if not self.api_key:
            raise ValueError("ARK_AGENT_API_KEY is required for TTS narration")
        if not text or not text.strip():
            raise ValueError("TTS narration text must not be empty")
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        if not 0.5 <= pitch <= 2.0:
            raise ValueError("pitch must be between 0.5 and 2.0")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        response_format = "wav" if destination.suffix.lower() == ".wav" else "mp3"
        response = self.session.post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model, "input": text.strip(), "voice": self.voice,
                "speed": speed, "pitch": pitch, "response_format": response_format,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(response, "status_code", "unknown")
            raise RuntimeError(f"Ark TTS request failed with HTTP {status}") from exc
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type:
            raise RuntimeError("Ark TTS returned JSON instead of audio")
        if not response.content:
            raise RuntimeError("Ark TTS returned an empty audio response")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
        return str(destination)
