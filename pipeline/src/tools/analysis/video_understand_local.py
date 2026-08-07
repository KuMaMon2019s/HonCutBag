"""Lightweight local video understanding with ffmpeg and optional Whisper."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

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


class VideoUnderstandLocal(BaseTool):
    name = "video_understand_local"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "local"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies: ClassVar[list[str]] = ["cmd:ffmpeg", "cmd:ffprobe"]
    install_instructions = "Install ffmpeg and ffprobe; optionally install openai-whisper."
    capabilities: ClassVar[list[str]] = [
        "extract_video_frames",
        "transcribe_video",
        "analyze_video_local",
    ]
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["operation", "input_path"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["extract_frames", "transcribe", "analyze"],
            },
            "input_path": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["scene", "keyframe", "interval"],
                "default": "scene",
            },
            "max_frames": {"type": "integer", "default": 20, "minimum": 1},
            "output_dir": {"type": "string"},
            "whisper_model": {
                "type": "string",
                "enum": ["tiny", "base", "small", "medium", "large"],
                "default": "base",
            },
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2048, disk_mb=500)

    _TIMESTAMP_PATTERN = re.compile(r"pts_time:\s*([\d.]+)")

    def get_status(self) -> ToolStatus:
        return (
            ToolStatus.AVAILABLE
            if shutil.which("ffmpeg") and shutil.which("ffprobe")
            else ToolStatus.UNAVAILABLE
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation", "analyze")
        if operation not in {"extract_frames", "transcribe", "analyze"}:
            return ToolResult(success=False, error=f"Unsupported operation: {operation!r}")
        input_path = Path(inputs.get("input_path", "")).expanduser()
        if not input_path.is_file():
            return ToolResult(success=False, error=f"Input video not found: {input_path}")
        if self.get_status() == ToolStatus.UNAVAILABLE:
            return ToolResult(success=False, error=self.install_instructions)

        started = time.monotonic()
        try:
            metadata = self._probe(input_path)
            if operation == "extract_frames":
                frames = self._extract_frames(input_path, metadata, inputs)
                payload: dict[str, Any] = {"video": str(input_path), "frames": frames}
            elif operation == "transcribe":
                transcript = self._transcribe(input_path, inputs.get("whisper_model", "base"))
                payload = {"video": str(input_path), **transcript}
            else:
                frames = self._extract_frames(input_path, metadata, inputs)
                transcript = self._transcribe(input_path, inputs.get("whisper_model", "base"))
                payload = {
                    "video": str(input_path),
                    "metadata": metadata,
                    "mode": inputs.get("mode", "scene"),
                    "frames": frames,
                    "frame_count": len(frames),
                    **transcript,
                }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult(success=False, error=f"Local video understanding failed: {exc}")

        artifacts = [frame["path"] for frame in payload.get("frames", [])]
        return ToolResult(
            success=True,
            data=payload,
            artifacts=artifacts,
            duration_seconds=round(time.monotonic() - started, 2),
        )

    @staticmethod
    def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if completed.returncode != 0:
            message = completed.stderr.strip().splitlines()
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=message[-1] if message else "unknown ffmpeg error",
            )
        return completed

    def _probe(self, input_path: Path) -> dict[str, Any]:
        completed = self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(input_path),
            ],
            timeout=30,
        )
        probe = json.loads(completed.stdout)
        video_stream = next(
            (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
            {},
        )
        duration = float(probe.get("format", {}).get("duration") or 0)
        return {
            "duration": duration,
            "resolution": {
                "width": int(video_stream.get("width") or 0),
                "height": int(video_stream.get("height") or 0),
            },
            "has_audio": any(
                stream.get("codec_type") == "audio" for stream in probe.get("streams", [])
            ),
        }

    def _extract_frames(
        self, input_path: Path, metadata: dict[str, Any], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        mode = inputs.get("mode", "scene")
        max_frames = inputs.get("max_frames", 20)
        if mode not in {"scene", "keyframe", "interval"}:
            raise ValueError(f"Unsupported frame extraction mode: {mode!r}")
        if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")

        output_root = Path(
            inputs.get("output_dir") or input_path.parent / f"{input_path.stem}_frames"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        frames_dir = Path(tempfile.mkdtemp(prefix="extract_", dir=output_root))
        output_pattern = frames_dir / "frame_%04d.jpg"

        if mode == "scene":
            video_filter = "select='gt(scene,0.3)',showinfo"
        elif mode == "keyframe":
            video_filter = "select='eq(pict_type,I)',showinfo"
        else:
            duration = metadata["duration"]
            if duration <= 0:
                raise ValueError("Cannot interval-sample a video with unknown duration")
            video_filter = f"fps=1/{max(duration / max_frames, 0.1):.6f},showinfo"

        completed = self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                video_filter,
                "-vsync",
                "vfr",
                "-q:v",
                "2",
                str(output_pattern),
            ],
            timeout=120,
        )
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        timestamps = [
            float(match.group(1))
            for line in completed.stderr.splitlines()
            if "showinfo" in line and (match := self._TIMESTAMP_PATTERN.search(line))
        ]

        if mode == "scene" and not frame_paths:
            retry_inputs = {**inputs, "mode": "interval", "output_dir": str(output_root)}
            return self._extract_frames(input_path, metadata, retry_inputs)

        if len(frame_paths) > max_frames:
            indices = self._even_indices(len(frame_paths), max_frames)
            frame_paths = [frame_paths[index] for index in indices]
            timestamps = [timestamps[index] for index in indices if index < len(timestamps)]
        timestamps = self._complete_timestamps(len(frame_paths), timestamps, metadata["duration"])
        return [
            {
                "path": str(path.resolve()),
                "timestamp": round(timestamp, 3),
                "timestamp_formatted": self._format_timestamp(timestamp),
            }
            for path, timestamp in zip(frame_paths, timestamps, strict=True)
        ]

    def _transcribe(self, input_path: Path, model_name: str) -> dict[str, Any]:
        if model_name not in {"tiny", "base", "small", "medium", "large"}:
            raise ValueError(f"Unsupported Whisper model: {model_name!r}")
        try:
            import whisper
        except ImportError:
            return {
                "transcript": [],
                "text": "",
                "transcription_available": False,
                "transcription_note": "Whisper is not installed; transcription skipped.",
            }

        with tempfile.TemporaryDirectory(prefix="honcut_whisper_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            audio = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if audio.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
                return {
                    "transcript": [],
                    "text": "",
                    "transcription_available": True,
                    "transcription_note": "Video has no readable audio track.",
                }
            model = whisper.load_model(model_name)
            transcription = model.transcribe(str(wav_path))

        segments = [
            {
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text": segment.get("text", "").strip(),
            }
            for segment in transcription.get("segments", [])
        ]
        return {
            "transcript": segments,
            "text": transcription.get("text", "").strip(),
            "transcription_available": True,
            "whisper_model": model_name,
        }

    @staticmethod
    def _even_indices(total: int, maximum: int) -> list[int]:
        if maximum == 1:
            return [0]
        return [round(index * (total - 1) / (maximum - 1)) for index in range(maximum)]

    @staticmethod
    def _complete_timestamps(count: int, timestamps: list[float], duration: float) -> list[float]:
        if len(timestamps) >= count:
            return timestamps[:count]
        if count == 0:
            return []
        if count == 1:
            return [timestamps[0] if timestamps else 0.0]
        return [index * duration / (count - 1) for index in range(count)]

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
