"""Phase 6 quality gate for character animation output."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from itertools import pairwise
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


class CharacterAnimationQA(BaseTool):
    name = "character_animation_qa"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "quality_assurance"
    provider = "honcut"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies: ClassVar[list[str]] = ["cmd:ffmpeg", "cmd:ffprobe", "python:PIL"]
    install_instructions = "Install ffmpeg/ffprobe and Pillow."
    capabilities: ClassVar[list[str]] = [
        "character_schema_qa",
        "animation_motion_qa",
        "video_probe_qa",
    ]
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["probe", "frame_sample", "motion_check", "schema_check", "full_qa"],
            },
            "input_path": {"type": "string"},
            "video_path": {"type": "string"},
            "characters_json_path": {"type": "string"},
            "num_samples": {"type": "integer", "default": 5, "minimum": 2},
            "threshold": {"type": "number", "default": 0.01, "minimum": 0},
            "output_dir": {"type": "string"},
            "expected_width": {"type": "integer"},
            "expected_height": {"type": "integer"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=512, disk_mb=100)

    def get_status(self) -> ToolStatus:
        try:
            import PIL  # noqa: F401
        except ImportError:
            return ToolStatus.UNAVAILABLE
        return (
            ToolStatus.AVAILABLE
            if shutil.which("ffmpeg") and shutil.which("ffprobe")
            else ToolStatus.UNAVAILABLE
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation")
        if operation not in {"probe", "frame_sample", "motion_check", "schema_check", "full_qa"}:
            return ToolResult(success=False, error=f"Unsupported operation: {operation!r}")
        if self.get_status() == ToolStatus.UNAVAILABLE:
            return ToolResult(success=False, error=self.install_instructions)

        started = time.monotonic()
        try:
            if operation == "schema_check":
                payload = self._schema_check(self._required_path(inputs, "characters_json_path"))
                artifacts: list[str] = []
            else:
                video_key = "video_path" if operation == "full_qa" else "input_path"
                video_path = self._required_path(inputs, video_key)
                if operation == "probe":
                    payload = self._probe(video_path, inputs)
                    artifacts = []
                elif operation == "frame_sample":
                    frames = self._sample_frames(video_path, inputs)
                    payload = {"frames": frames, "sample_count": len(frames)}
                    artifacts = frames
                elif operation == "motion_check":
                    payload = self._motion_check(video_path, inputs)
                    artifacts = payload["frames"]
                else:
                    payload = self._full_qa(video_path, inputs)
                    artifacts = payload["checks"]["frame_sample"]["frames"]
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            return ToolResult(success=False, error=f"Character animation QA failed: {exc}")

        return ToolResult(
            success=True,
            data=payload,
            artifacts=artifacts,
            duration_seconds=round(time.monotonic() - started, 2),
        )

    @staticmethod
    def _required_path(inputs: dict[str, Any], key: str) -> Path:
        path = Path(inputs.get(key, "")).expanduser()
        if not path.is_file():
            raise ValueError(f"{key} not found: {path}")
        return path

    @staticmethod
    def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if completed.returncode != 0:
            lines = completed.stderr.strip().splitlines()
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=lines[-1] if lines else "unknown media command error",
            )
        return completed

    def _probe(self, video_path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        completed = self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ],
            timeout=30,
        )
        raw = json.loads(completed.stdout)
        video = next(
            (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        audio = next(
            (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"),
            None,
        )
        duration = float(raw.get("format", {}).get("duration") or 0)
        issues = []
        if duration <= 0:
            issues.append("Video duration must be greater than zero")
        if video is None:
            issues.append("No video stream found")
        width = int((video or {}).get("width") or 0)
        height = int((video or {}).get("height") or 0)
        expected_width = inputs.get("expected_width")
        expected_height = inputs.get("expected_height")
        if expected_width and width != expected_width:
            issues.append(f"Resolution width mismatch: expected {expected_width}, got {width}")
        if expected_height and height != expected_height:
            issues.append(f"Resolution height mismatch: expected {expected_height}, got {height}")
        fps = self._parse_rate(
            (video or {}).get("avg_frame_rate") or (video or {}).get("r_frame_rate")
        )
        if video is not None and not 24 <= fps <= 30:
            issues.append(f"Frame rate {fps:.3f} is outside the expected 24-30 fps range")
        return {
            "passed": not issues,
            "metadata": {
                "duration": duration,
                "resolution": {"width": width, "height": height},
                "fps": round(fps, 3),
                "has_video": video is not None,
                "has_audio": audio is not None,
            },
            "issues": issues,
        }

    def _sample_frames(self, video_path: Path, inputs: dict[str, Any]) -> list[str]:
        num_samples = inputs.get("num_samples", 5)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 2:
            raise ValueError("num_samples must be an integer of at least 2")
        probe = self._probe(video_path, {})
        duration = probe["metadata"]["duration"]
        if duration <= 0:
            raise ValueError("Cannot sample a zero-duration video")
        output_root = Path(inputs.get("output_dir") or video_path.parent / f"{video_path.stem}_qa")
        output_root.mkdir(parents=True, exist_ok=True)
        sample_dir = Path(tempfile.mkdtemp(prefix="samples_", dir=output_root))
        frames = []
        for index in range(num_samples):
            # Keep the final sample inside the decodable range. Container duration
            # can extend slightly beyond the last video frame, especially for
            # short clips with audio.
            timestamp = duration * index / num_samples
            frame_path = sample_dir / f"sample_{index + 1:04d}.jpg"
            self._run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                timeout=120,
            )
            frames.append(str(frame_path.resolve()))
        return frames

    def _motion_check(self, video_path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image, ImageChops, ImageStat

        threshold = inputs.get("threshold", 0.01)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold < 0:
            raise ValueError("threshold must be a non-negative number")
        frames = self._sample_frames(video_path, inputs)
        scores = []
        for first_path, second_path in pairwise(frames):
            with Image.open(first_path) as first_raw, Image.open(second_path) as second_raw:
                first = first_raw.convert("RGB")
                second = second_raw.convert("RGB").resize(first.size)
                difference = ImageChops.difference(first, second)
                mean_channels = ImageStat.Stat(difference).mean
                score = sum(mean_channels) / (len(mean_channels) * 255)
                scores.append(round(score, 6))
        average = sum(scores) / len(scores) if scores else 0.0
        passed = average >= float(threshold)
        return {
            "passed": passed,
            "average_difference": round(average, 6),
            "threshold": float(threshold),
            "scores": scores,
            "frames": frames,
            "risk": None if passed else "static/slideshow risk",
            "issues": [] if passed else ["Average sampled-frame motion is below threshold"],
        }

    @staticmethod
    def _schema_check(characters_path: Path) -> dict[str, Any]:
        document = json.loads(characters_path.read_text(encoding="utf-8"))
        if isinstance(document, list):
            characters = document
        elif isinstance(document, dict) and isinstance(document.get("characters"), list):
            characters = document["characters"]
        elif isinstance(document, dict):
            characters = [document]
        else:
            return {
                "passed": False,
                "character_count": 0,
                "issues": ["Root must be an object or list"],
            }
        issues = []
        required = ("name", "appearance", "distinguishing_features")
        for index, character in enumerate(characters):
            if not isinstance(character, dict):
                issues.append(f"Character {index} must be an object")
                continue
            for field in required:
                if field not in character or character[field] in (None, "", []):
                    issues.append(f"Character {index} is missing required field: {field}")
        if not characters:
            issues.append("No characters found")
        return {"passed": not issues, "character_count": len(characters), "issues": issues}

    def _full_qa(self, video_path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        probe = self._probe(video_path, inputs)
        frames = self._sample_frames(video_path, inputs)
        motion_inputs = {**inputs, "output_dir": inputs.get("output_dir")}
        motion = self._motion_check(video_path, motion_inputs)
        characters_path = self._required_path(inputs, "characters_json_path")
        schema = self._schema_check(characters_path)
        checks = {
            "probe": probe,
            "frame_sample": {"passed": bool(frames), "frames": frames, "issues": []},
            "motion_check": motion,
            "schema_check": schema,
        }
        issues = [issue for check in checks.values() for issue in check.get("issues", [])]
        failed = [name for name, check in checks.items() if not check.get("passed")]
        if not probe["metadata"]["has_video"] or probe["metadata"]["duration"] <= 0:
            verdict = "fail"
        elif failed:
            verdict = "revise"
        else:
            verdict = "pass"
        return {"verdict": verdict, "checks": checks, "issues": issues}

    @staticmethod
    def _parse_rate(value: Any) -> float:
        if not value:
            return 0.0
        numerator, separator, denominator = str(value).partition("/")
        if separator:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(numerator)
