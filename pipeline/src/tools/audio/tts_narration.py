"""Volcano Ark plan TTS narration client (unidirectional endpoint).

Ark's text-to-speech does NOT use the OpenAI-compatible ``/audio/speech``
endpoint. It requires the dedicated unidirectional endpoint with X-Api-Key
authentication and a submit/query request body. The response is JSON whose
``data`` field carries the base64-encoded audio.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import requests

DEFAULT_TTS_URL = "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"


class TTSNarration:
    """Generate narration through Ark's plan TTS unidirectional endpoint."""

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
        self.base_url = (base_url or os.environ.get("ARK_TTS_BASE_URL", DEFAULT_TTS_URL)).rstrip("/")
        # Resource id (used in the header), not a chat-style model name.
        self.model = model or os.environ.get("ARK_TTS_RESOURCE_ID", DEFAULT_RESOURCE_ID)
        self.voice = voice or os.environ.get(
            "DOUBAO_SPEECH_VOICE_TYPE",
            os.environ.get("ARK_TTS_VOICE", "zh_female_vv_uranus_bigtts"),
        )
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
        audio_format = "wav" if destination.suffix.lower() == ".wav" else "mp3"

        request_id = str(uuid.uuid4())
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.model,
            "X-Api-Request-Id": request_id,
            "Content-Type": "application/json",
        }
        body = {
            "user": {"uid": "honcut"},
            "unique_id": request_id,
            "req_params": {
                "text": text.strip(),
                "speaker": self.voice,
                "audio_params": {
                    "format": audio_format,
                    "sample_rate": 24000,
                    # speech_rate is a -12..20 integer offset; 0 == normal.
                    "speech_rate": max(-12, min(20, int(round((speed - 1.0) * 10)))),
                    "pitch_rate": pitch,
                    "enable_timestamp": False,
                },
            },
        }
        response = self.session.post(self.base_url, headers=headers, json=body, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(response, "status_code", "unknown")
            raise RuntimeError(f"Ark TTS request failed with HTTP {status}") from exc
        if not response.content:
            raise RuntimeError("Ark TTS returned an empty audio response")

        audio_bytes = self._extract_audio(response)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(audio_bytes)
        temporary.replace(destination)
        return str(destination)

    @staticmethod
    def _extract_audio(response: requests.Response) -> bytes:
        """Decode base64 audio chunks from the NDJSON envelope.

        The unidirectional endpoint returns newline-delimited JSON chunks.
        Each chunk's ``data`` field is a base64 slice of the audio stream;
        a final chunk with ``code=20000000`` marks completion.
        """
        content = response.content
        if content[:1] in (b"{", b"["):
            chunks: list[bytes] = []
            parsed_any = False
            for raw_line in content.split(b"\n"):
                raw_line = raw_line.strip()
                if not raw_line or raw_line[:1] not in (b"{", b"["):
                    continue
                try:
                    payload = json.loads(raw_line.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                parsed_any = True
                if not isinstance(payload, dict):
                    continue
                code = payload.get("code")
                if code not in (0, 20000000):
                    raise RuntimeError(
                        f"Ark TTS failed with code {code}: {payload.get('message', 'unknown error')}"
                    )
                data = payload.get("data")
                if isinstance(data, str) and data:
                    chunks.append(base64.b64decode(data))
            if parsed_any:
                if not chunks:
                    raise RuntimeError("Ark TTS returned JSON without decodable audio data")
                return b"".join(chunks)
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type:
            raise RuntimeError("Ark TTS returned JSON without decodable audio data")
        if not content:
            raise RuntimeError("Ark TTS returned an empty audio response")
        return content
