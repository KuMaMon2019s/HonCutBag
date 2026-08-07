"""ElevenLabs text-to-music generation tool."""

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


class ElevenLabsMusic(BaseTool):
    name = "elevenlabs_music"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "elevenlabs"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies: ClassVar[list[str]] = []
    install_instructions = (
        "Set ELEVENLABS_API_KEY to an ElevenLabs API key. "
        "Suno remains available as an alternative music provider."
    )
    fallback_tools: ClassVar[list[str]] = ["suno_music", "google_music"]
    capabilities: ClassVar[list[str]] = ["compose_music", "compose_music_with_plan"]
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["compose", "compose_with_plan"],
            },
            "prompt": {"type": "string"},
            "duration_ms": {"type": "integer", "default": 30000},
            "output_path": {"type": "string"},
            "negative_prompt": {
                "type": "string",
                "description": "What to avoid in generation",
            },
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=256, disk_mb=100, network_required=True)
    side_effects: ClassVar[list[str]] = [
        "calls ElevenLabs Music API",
        "writes generated audio",
    ]

    _BASE_URL = "https://api.elevenlabs.io/v1/music"
    _TIMEOUT_SECONDS = 300

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

        operation = inputs.get("operation")
        if operation not in {"compose", "compose_with_plan"}:
            return ToolResult(success=False, error=f"Unsupported operation: {operation!r}")

        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        duration_ms = inputs.get("duration_ms", 30000)
        if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0:
            return ToolResult(success=False, error="duration_ms must be a positive integer")

        output_path = Path(inputs.get("output_path") or "elevenlabs_music.mp3").expanduser()
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        started = time.monotonic()

        try:
            payload: dict[str, Any]
            plan = None
            if operation == "compose_with_plan":
                plan_payload = {"prompt": prompt, "music_length_ms": duration_ms}
                self._add_negative_prompt(plan_payload, inputs)
                plan_response = requests.post(
                    f"{self._BASE_URL}/composition-plan",
                    headers=headers,
                    json=plan_payload,
                    timeout=self._TIMEOUT_SECONDS,
                )
                self._raise_api_error(plan_response, "composition plan")
                plan = plan_response.json()
                payload = {"composition_plan": plan, "music_length_ms": duration_ms}
            else:
                payload = {"prompt": prompt, "music_length_ms": duration_ms}
                self._add_negative_prompt(payload, inputs)

            response = requests.post(
                self._BASE_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self._TIMEOUT_SECONDS,
            )
            self._raise_api_error(response, "music composition")
            self._write_stream(response, output_path)
        except (requests.RequestException, OSError, ValueError) as exc:
            return ToolResult(success=False, error=f"ElevenLabs music generation failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "output": str(output_path),
                "duration_ms": duration_ms,
                "provider": "elevenlabs",
                "composition_plan": plan,
            },
            artifacts=[str(output_path)],
            duration_seconds=round(time.monotonic() - started, 2),
        )

    @staticmethod
    def _add_negative_prompt(payload: dict[str, Any], inputs: dict[str, Any]) -> None:
        negative_prompt = inputs.get("negative_prompt")
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

    @staticmethod
    def _write_stream(response: requests.Response, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as audio_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    audio_file.write(chunk)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ValueError("ElevenLabs returned an empty audio response")

    @staticmethod
    def _raise_api_error(response: requests.Response, action: str) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            code = detail.get("status") or detail.get("code") or detail.get("type")
            message = detail.get("message") or detail.get("detail") or response.text
            suggestion = (
                detail.get("prompt_suggestion")
                or detail.get("composition_plan_suggestion")
                or payload.get("prompt_suggestion")
                or payload.get("composition_plan_suggestion")
            )
        else:
            code, message, suggestion = None, str(detail), None
        labels = {401: "invalid API key", 422: "invalid parameters", 429: "rate limit exceeded"}
        reason = labels.get(response.status_code, message or response.reason)
        if code == "bad_prompt":
            reason = f"content restriction: {message}"
            if suggestion:
                reason += f"; prompt suggestion: {suggestion}"
        raise requests.HTTPError(
            f"ElevenLabs {action} failed ({response.status_code}): {reason}", response=response
        )
