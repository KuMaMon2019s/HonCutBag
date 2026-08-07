"""ElevenLabs text-to-sound-effects generation tool."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, ClassVar

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class SoundEffects(BaseTool):
    name = "sound_effects"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "sound_effect_generation"
    provider = "elevenlabs"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies: ClassVar[list[str]] = []
    install_instructions = "Set ELEVENLABS_API_KEY to an ElevenLabs API key."
    capabilities: ClassVar[list[str]] = ["text_to_sound_effect", "looping_sound_effect"]
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Description of the sound effect"},
            "duration_seconds": {
                "type": "number",
                "description": "Duration 0.5-30s, auto if null",
            },
            "prompt_influence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.3,
            },
            "loop": {"type": "boolean", "default": False},
            "model_id": {"type": "string", "default": "eleven_text_to_sound_v2"},
            "output_path": {"type": "string"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=128, disk_mb=20, network_required=True)

    _URL = "https://api.elevenlabs.io/v1/sound-generation"

    def get_status(self) -> ToolStatus:
        return (
            ToolStatus.AVAILABLE if os.environ.get("ELEVENLABS_API_KEY") else ToolStatus.UNAVAILABLE
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            return ToolResult(
                success=False, error="No ElevenLabs API key. " + self.install_instructions
            )

        text = str(inputs.get("text", "")).strip()
        if not text:
            return ToolResult(success=False, error="text is required")

        duration = inputs.get("duration_seconds")
        if duration is not None and (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not 0.5 <= float(duration) <= 30
        ):
            return ToolResult(success=False, error="duration_seconds must be between 0.5 and 30")

        influence = inputs.get("prompt_influence", 0.3)
        if (
            not isinstance(influence, (int, float))
            or isinstance(influence, bool)
            or not 0 <= float(influence) <= 1
        ):
            return ToolResult(success=False, error="prompt_influence must be between 0 and 1")

        model_id = inputs.get("model_id", "eleven_text_to_sound_v2")
        loop = inputs.get("loop", False)
        if loop and model_id != "eleven_text_to_sound_v2":
            return ToolResult(success=False, error="loop requires eleven_text_to_sound_v2")

        payload: dict[str, Any] = {
            "text": text,
            "prompt_influence": float(influence),
            "loop": bool(loop),
            "model_id": model_id,
        }
        if duration is not None:
            payload["duration_seconds"] = float(duration)

        output_path = Path(inputs.get("output_path") or "sound_effect.mp3").expanduser()
        started = time.monotonic()
        try:
            response = requests.post(
                self._URL,
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=60,
            )
            self._raise_api_error(response)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as audio_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        audio_file.write(chunk)
            if output_path.stat().st_size == 0:
                raise ValueError("ElevenLabs returned an empty audio response")
        except (requests.RequestException, OSError, ValueError) as exc:
            return ToolResult(success=False, error=f"Sound effect generation failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "output": str(output_path),
                "duration_seconds": duration,
                "provider": "elevenlabs",
                "model_id": model_id,
                "loop": bool(loop),
            },
            artifacts=[str(output_path)],
            duration_seconds=round(time.monotonic() - started, 2),
            model=model_id,
        )

    @staticmethod
    def _raise_api_error(response: requests.Response) -> None:
        if response.ok:
            return
        labels = {401: "invalid API key", 422: "invalid parameters", 429: "rate limit exceeded"}
        reason = labels.get(response.status_code)
        if reason is None:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            reason = str(detail or response.reason)
        raise requests.HTTPError(
            f"ElevenLabs sound generation failed ({response.status_code}): {reason}",
            response=response,
        )
