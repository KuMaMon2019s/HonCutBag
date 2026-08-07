"""Unified local FFmpeg video editing tool."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from tools.base_tool import (
        BaseTool,
        Determinism,
        ExecutionMode,
        ToolResult,
        ToolRuntime,
        ToolStability,
        ToolStatus,
        ToolTier,
    )
except ModuleNotFoundError:  # Support imports through pipeline.src.tools.
    from ..base_tool import (
        BaseTool,
        Determinism,
        ExecutionMode,
        ToolResult,
        ToolRuntime,
        ToolStability,
        ToolStatus,
        ToolTier,
    )


logger = logging.getLogger(__name__)

PLATFORM_PRESETS = {
    "tiktok": (1080, 1920),
    "youtube": (1920, 1080),
    "instagram": (1080, 1350),
    "square": (1080, 1080),
    "twitter": (1920, 1080),
}


class VideoEdit(BaseTool):
    """Perform common deterministic video edits with FFmpeg and FFprobe."""

    name = "video_edit"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_edit"
    provider = "ffmpeg"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["cmd:ffmpeg", "cmd:ffprobe"]
    capabilities = [
        "trim",
        "concat",
        "resize",
        "speed",
        "extract_audio",
        "replace_audio",
        "compress",
        "info",
        "overlay",
    ]
    install_instructions = "Install FFmpeg (including ffprobe): https://ffmpeg.org/download.html"

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "trim",
                    "concat",
                    "resize",
                    "speed",
                    "extract_audio",
                    "replace_audio",
                    "compress",
                    "info",
                    "overlay",
                ],
            },
            "input_path": {"type": "string"},
            "input_paths": {"type": "array", "items": {"type": "string"}},
            "output_path": {"type": "string"},
            "start_time": {"type": ["string", "number"]},
            "end_time": {"type": ["string", "number"]},
            "transition": {"type": "string", "enum": ["cut", "crossfade"], "default": "cut"},
            "crossfade_duration": {"type": "number", "default": 0.5},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "platform": {"type": "string"},
            "factor": {"type": "number", "minimum": 0.25, "maximum": 4.0},
            "format": {"type": "string"},
            "audio_path": {"type": "string"},
            "crf": {"type": "integer", "default": 23},
            "preset": {"type": "string", "default": "medium"},
            "overlay_path": {"type": "string"},
            "position": {"type": "string"},
            "margin": {"type": "integer", "default": 10},
        },
    }

    def get_status(self) -> ToolStatus:
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        operation = inputs.get("operation")
        handlers = {
            "trim": self._trim,
            "concat": self._concat,
            "resize": self._resize,
            "speed": self._speed,
            "extract_audio": self._extract_audio,
            "replace_audio": self._replace_audio,
            "compress": self._compress,
            "info": self._info,
            "overlay": self._overlay,
        }
        if operation not in handlers:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")
        if self.get_status() is not ToolStatus.AVAILABLE:
            return ToolResult(success=False, error=self.install_instructions)

        try:
            result = handlers[operation](inputs)
        except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
            result = ToolResult(success=False, error=str(exc))
        result.duration_seconds = round(time.monotonic() - started, 2)
        return result

    def _trim(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        output = self._output_file(inputs)
        start = self._required(inputs, "start_time")
        end = self._required(inputs, "end_time")
        self._run_ffmpeg(["-ss", str(start), "-to", str(end), "-i", str(source), "-c", "copy", str(output)])
        return self._output_result(output)

    def _concat(self, inputs: dict[str, Any]) -> ToolResult:
        raw_paths = inputs.get("input_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("input_paths must contain at least one file")
        paths = [self._existing_file(path, "Input") for path in raw_paths]
        output = self._output_file(inputs)
        transition = inputs.get("transition", "cut")
        if transition == "cut":
            self._concat_cut(paths, output)
        elif transition == "crossfade":
            self._concat_crossfade(paths, output, float(inputs.get("crossfade_duration", 0.5)))
        else:
            raise ValueError("transition must be 'cut' or 'crossfade'")
        return self._output_result(output)

    def _concat_cut(self, paths: list[Path], output: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="video_edit_concat_") as temp_dir:
            list_path = Path(temp_dir) / "inputs.txt"
            lines = []
            for path in paths:
                escaped = str(path.resolve()).replace("'", "'\\''")
                lines.append(f"file '{escaped}'")
            list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)])

    def _concat_crossfade(self, paths: list[Path], output: Path, fade: float) -> None:
        if len(paths) == 1:
            logger.warning("Crossfade needs multiple clips; falling back to cut concat")
            self._concat_cut(paths, output)
            return
        if fade <= 0:
            raise ValueError("crossfade_duration must be greater than zero")

        durations = [self._duration(path) for path in paths]
        if any(duration <= fade for duration in durations):
            raise ValueError("crossfade_duration must be shorter than every input clip")

        filters: list[str] = []
        cumulative = durations[0]
        video_in = "[0:v]"
        audio_in = "[0:a]"
        for index in range(1, len(paths)):
            video_out = f"[v{index}]"
            audio_out = f"[a{index}]"
            offset = cumulative - fade
            filters.append(
                f"{video_in}[{index}:v]xfade=transition=fade:duration={fade}:offset={offset}{video_out}"
            )
            filters.append(f"{audio_in}[{index}:a]acrossfade=d={fade}{audio_out}")
            video_in, audio_in = video_out, audio_out
            cumulative += durations[index] - fade

        args: list[str] = []
        for path in paths:
            args.extend(["-i", str(path)])
        args.extend([
            "-filter_complex",
            ";".join(filters),
            "-map",
            video_in,
            "-map",
            audio_in,
            str(output),
        ])
        self._run_ffmpeg(args)

    def _resize(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        output = self._output_file(inputs)
        platform = inputs.get("platform")
        if platform:
            if platform not in PLATFORM_PRESETS:
                raise ValueError(f"Unknown platform preset: {platform}")
            width, height = PLATFORM_PRESETS[platform]
        else:
            width = int(self._required(inputs, "width"))
            height = int(self._required(inputs, "height"))
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        self._run_ffmpeg(["-i", str(source), "-vf", video_filter, "-c:a", "copy", str(output)])
        return self._output_result(output)

    def _speed(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        output = self._output_file(inputs)
        factor = float(self._required(inputs, "factor"))
        if not 0.25 <= factor <= 4.0:
            raise ValueError("factor must be between 0.25 and 4.0")
        self._run_ffmpeg([
            "-i",
            str(source),
            "-filter:v",
            f"setpts={1 / factor}*PTS",
            "-filter:a",
            self._atempo_filter(factor),
            str(output),
        ])
        return self._output_result(output)

    def _extract_audio(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        output = self._output_file(inputs)
        audio_format = inputs.get("format", "mp3")
        codecs = {"mp3": "libmp3lame", "wav": "pcm_s16le", "aac": "aac"}
        if audio_format not in codecs:
            raise ValueError("format must be one of: mp3, wav, aac")
        self._run_ffmpeg(["-i", str(source), "-vn", "-acodec", codecs[audio_format], str(output)])
        return self._output_result(output)

    def _replace_audio(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        audio = self._input_file(inputs, "audio_path")
        output = self._output_file(inputs)
        self._run_ffmpeg([
            "-i", str(source), "-i", str(audio), "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output),
        ])
        return self._output_result(output)

    def _compress(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        output = self._output_file(inputs)
        crf = int(inputs.get("crf", 23))
        preset = str(inputs.get("preset", "medium"))
        if not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        self._run_ffmpeg(["-i", str(source), "-crf", str(crf), "-preset", preset, "-c:a", "copy", str(output)])
        return self._output_result(output)

    def _info(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        return ToolResult(success=True, data={"info": self._metadata(source)})

    def _overlay(self, inputs: dict[str, Any]) -> ToolResult:
        source = self._input_file(inputs, "input_path")
        overlay = self._input_file(inputs, "overlay_path")
        output = self._output_file(inputs)
        margin = int(inputs.get("margin", 10))
        if margin < 0:
            raise ValueError("margin must not be negative")
        positions = {
            "top-right": f"W-w-{margin}:{margin}",
            "top-left": f"{margin}:{margin}",
            "bottom-right": f"W-w-{margin}:H-h-{margin}",
            "bottom-left": f"{margin}:H-h-{margin}",
            "center": "(W-w)/2:(H-h)/2",
        }
        position = inputs.get("position", "top-right")
        if position not in positions:
            raise ValueError(f"Unknown overlay position: {position}")
        self._run_ffmpeg([
            "-i", str(source), "-i", str(overlay),
            "-filter_complex", f"overlay={positions[position]}", "-c:a", "copy", str(output),
        ])
        return self._output_result(output)

    def _run_ffmpeg(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        timeout = 600 if self._is_concat_command(args) else 300
        try:
            return subprocess.run(
                ["ffmpeg", "-y", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = self._stderr_text(exc.stderr)
            raise RuntimeError(f"FFmpeg timed out after {timeout}s. stderr: {stderr}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFmpeg failed with exit code {exc.returncode}. stderr: {self._stderr_text(exc.stderr)}"
            ) from exc

    def _probe(self, path: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", path,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return json.loads(completed.stdout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FFprobe timed out after 300s. stderr: {self._stderr_text(exc.stderr)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFprobe failed with exit code {exc.returncode}. stderr: {self._stderr_text(exc.stderr)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"FFprobe returned invalid JSON: {exc}") from exc

    def _metadata(self, path: Path) -> dict[str, Any]:
        probe = self._probe(str(path))
        streams = probe.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
        frame_rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
        fps = self._parse_rate(frame_rate)
        format_info = probe.get("format", {})
        duration = format_info.get("duration", video.get("duration"))
        size = format_info.get("size")
        return {
            "duration": float(duration) if duration is not None else None,
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": fps,
            "codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"),
            "file_size": int(size) if size is not None else path.stat().st_size,
        }

    def _output_result(self, output: Path) -> ToolResult:
        metadata = self._metadata(output)
        return ToolResult(
            success=True,
            data={"output": str(output), "duration_seconds": metadata["duration"]},
            artifacts=[str(output)],
        )

    def _duration(self, path: Path) -> float:
        duration = self._metadata(path)["duration"]
        if duration is None:
            raise ValueError(f"Could not determine duration: {path}")
        return float(duration)

    @staticmethod
    def _atempo_filter(factor: float) -> str:
        filters: list[float] = []
        remaining = factor
        while remaining > 2.0:
            filters.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            filters.append(0.5)
            remaining /= 0.5
        filters.append(remaining)
        return ",".join(f"atempo={value:g}" for value in filters)

    @staticmethod
    def _parse_rate(rate: str) -> float | None:
        try:
            numerator, denominator = rate.split("/", 1)
            return float(numerator) / float(denominator) if float(denominator) else None
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _is_concat_command(args: list[str]) -> bool:
        return "concat" in args or any("xfade=" in argument for argument in args)

    @staticmethod
    def _stderr_text(stderr: str | bytes | None) -> str:
        if stderr is None:
            return "(no stderr)"
        if isinstance(stderr, bytes):
            return stderr.decode(errors="replace")
        return stderr

    @staticmethod
    def _required(inputs: dict[str, Any], key: str) -> Any:
        value = inputs.get(key)
        if value is None or value == "":
            raise ValueError(f"{key} is required")
        return value

    def _input_file(self, inputs: dict[str, Any], key: str) -> Path:
        return self._existing_file(self._required(inputs, key), key)

    @staticmethod
    def _existing_file(raw_path: Any, label: str) -> Path:
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        return path

    def _output_file(self, inputs: dict[str, Any]) -> Path:
        output = Path(str(self._required(inputs, "output_path")))
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
