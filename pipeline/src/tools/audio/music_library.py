"""User music library — local royalty-free track discovery.

Surfaces the tracks a user has dropped into ``music_library/`` so the agent can
present them at the *proposal* stage, before creative direction is approved.

AGENT_GUIDE.md requires the music decision to be made at the proposal stage, but
the only check for ``music_library/`` historically lived in the asset-director
skills, which run later. That meant a user could approve a creative direction
without ever being told a free, intentional music option was sitting on disk.
This read-only tool makes the library a first-class, auto-discovered provider so
it shows up in the preflight provider menu alongside ``music_gen`` and the stock
music sources.
"""

# ruff: noqa: RUF012

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

# Repository root: tools/audio/music_library.py -> parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common royalty-free audio container extensions a user might drop in.
_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
}

_MOODS = {"happy", "sad", "epic", "calm"}


@dataclass(frozen=True)
class MusicTrack:
    """A normalized local or remote music-library entry."""

    id: str
    path: str
    duration: float
    format: str
    mood: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MusicLibrary(BaseTool):
    name = "music_library"
    version = "0.1.0"
    tier = ToolTier.SOURCE
    capability = "music_library"
    provider = "local"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = []  # pure filesystem scan; ffprobe is used opportunistically
    install_instructions = (
        "Create a 'music_library/' folder in the project root and drop "
        "royalty-free audio tracks into it (e.g. .mp3, .wav, .m4a, .flac, .ogg). "
        "Sources: your own files, YouTube Audio Library, Jamendo, Freesound, etc. "
        "Override the location with the MUSIC_LIBRARY_DIR environment variable."
    )

    agent_skills = ["music"]

    capabilities = ["list_user_music_tracks"]
    supports = {
        "local_offline": True,
        "free": True,
        "duration_when_ffprobe_present": True,
    }
    best_for = [
        "user-provided, intentional background music",
        "free music with no API key or generation cost",
        "knowing music options at the proposal stage",
    ]
    not_good_for = [
        "generating new music (use music_gen / suno_music)",
        "searching an external catalog (use freesound_music / pixabay_music)",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "library_dir": {
                "type": "string",
                "description": (
                    "Optional override for the library folder. Defaults to the "
                    "MUSIC_LIBRARY_DIR env var, then '<project root>/music_library'."
                ),
            },
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "library_dir": {"type": "string"},
            "exists": {"type": "boolean"},
            "track_count": {"type": "integer"},
            "total_duration_seconds": {"type": ["number", "null"]},
            "tracks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string"},
                        "size_bytes": {"type": "integer"},
                        "duration_seconds": {"type": ["number", "null"]},
                    },
                },
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=64, vram_mb=0, disk_mb=0, network_required=False
    )
    side_effects = []  # read-only
    user_visible_verification = [
        "Confirm the listed tracks are the ones you intend to choose from",
    ]

    def __init__(self, music_dir: str = "~/.honcut/music/") -> None:
        self.music_dir = Path(music_dir).expanduser()
        self._tracks: dict[str, MusicTrack] = {}

    # ---- Library resolution ----

    def _library_dir(self, inputs: dict[str, Any] | None = None) -> Path:
        if inputs and inputs.get("library_dir"):
            return Path(inputs["library_dir"]).expanduser()
        env_dir = os.environ.get("MUSIC_LIBRARY_DIR")
        if env_dir:
            return Path(env_dir).expanduser()
        honcut_dir = os.environ.get("HONCUT_MUSIC_DIR")
        if honcut_dir:
            return Path(honcut_dir).expanduser()
        return self.music_dir

    def _list_tracks(self, library_dir: Path) -> list[Path]:
        if not library_dir.is_dir():
            return []
        tracks = [
            p
            for p in library_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
        ]
        return sorted(tracks, key=lambda p: p.as_posix().lower())

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        """Best-effort track duration via ffprobe; None if unavailable."""
        if shutil.which("ffprobe") is None:
            return None

    @staticmethod
    def _probe_metadata(path: Path) -> tuple[float, dict[str, Any]]:
        """Extract duration, bitrate, tags, and stream information with ffprobe."""
        if shutil.which("ffprobe") is None:
            raise RuntimeError("ffprobe is required to scan music metadata")
        command = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,bit_rate,format_name:format_tags=title,artist,album,genre,mood",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=15, check=True
            )
            probe = json.loads(completed.stdout)
            format_info = probe.get("format", {})
            duration = float(format_info.get("duration") or 0.0)
            bitrate = int(format_info.get("bit_rate") or 0)
        except (subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read audio metadata for {path}: {exc}") from exc
        metadata = {
            "bitrate": bitrate,
            "tags": format_info.get("tags", {}),
            "streams": probe.get("streams", []),
            "size_bytes": path.stat().st_size,
        }
        return duration, metadata

    @staticmethod
    def _infer_mood(path: Path, metadata: dict[str, Any]) -> str:
        tags = metadata.get("tags", {})
        searchable = " ".join(
            [path.stem, path.parent.name]
            + [str(value) for value in tags.values()]
        ).lower()
        aliases = {
            "happy": ("happy", "upbeat", "joy", "cheerful", "欢快", "快乐"),
            "sad": ("sad", "melancholy", "sorrow", "悲伤", "忧郁"),
            "epic": ("epic", "cinematic", "trailer", "heroic", "史诗"),
            "calm": ("calm", "ambient", "peaceful", "relax", "宁静", "舒缓"),
        }
        for mood, words in aliases.items():
            if any(word in searchable for word in words):
                return mood
        return "calm"

    @staticmethod
    def _track_id(path: Path) -> str:
        identity = str(path.resolve()).encode("utf-8")
        return hashlib.sha256(identity).hexdigest()[:16]

    def scan(self) -> list[MusicTrack]:
        """Scan the configured directory recursively and cache valid tracks."""
        tracks: dict[str, MusicTrack] = {}
        for path in self._list_tracks(self._library_dir()):
            try:
                duration, metadata = self._probe_metadata(path)
            except RuntimeError:
                continue
            track = MusicTrack(
                id=self._track_id(path),
                path=str(path),
                duration=duration,
                format=path.suffix.lower().lstrip("."),
                mood=self._infer_mood(path, metadata),
                source="local",
                metadata=metadata,
            )
            tracks[track.id] = track
        self._tracks = tracks
        return list(tracks.values())

    def search(
        self,
        mood: str | None = None,
        duration_range: tuple[float | None, float | None] | None = None,
    ) -> list[MusicTrack]:
        """Return cached tracks matching mood and inclusive duration bounds."""
        if mood is not None and mood not in _MOODS:
            raise ValueError(f"Unsupported mood {mood!r}; expected one of {sorted(_MOODS)}")
        if duration_range is not None:
            if len(duration_range) != 2:
                raise ValueError("duration_range must be a (minimum, maximum) pair")
            minimum, maximum = duration_range
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("duration_range minimum cannot exceed maximum")
        else:
            minimum = maximum = None
        candidates = self.scan()
        return [
            track for track in candidates
            if (mood is None or track.mood == mood)
            and (minimum is None or track.duration >= minimum)
            and (maximum is None or track.duration <= maximum)
        ]

    def get_track(self, track_id: str) -> MusicTrack:
        """Return a track by stable id, rescanning once when needed."""
        if track_id not in self._tracks:
            self.scan()
        try:
            return self._tracks[track_id]
        except KeyError as exc:
            raise KeyError(f"Music track not found: {track_id}") from exc

    def search_external(
        self,
        query: str,
        source: str,
        duration_range: tuple[float, float] = (30.0, 120.0),
    ) -> list[MusicTrack]:
        """Search Pixabay or Freesound using the project's provider adapters."""
        providers = {
            "pixabay": ("tools.audio.pixabay_music", "PixabayMusic"),
            "freesound": ("tools.audio.freesound_music", "FreesoundMusic"),
        }
        if source not in providers:
            raise ValueError("source must be 'pixabay' or 'freesound'")
        module_name, class_name = providers[source]
        import importlib

        provider = getattr(importlib.import_module(module_name), class_name)()
        response = provider.execute({
            "query": query,
            "min_duration": duration_range[0],
            "max_duration": duration_range[1],
        })
        if not response.success:
            raise RuntimeError(response.error or f"{source} search failed")
        payload = response.data or {}
        path = payload.get("output")
        if not path:
            return []
        duration = float(payload.get("duration_seconds") or 0.0)
        track = MusicTrack(
            id=self._track_id(Path(path)), path=str(path), duration=duration,
            format=str(payload.get("format") or Path(path).suffix.lstrip(".")),
            mood=self._infer_mood(Path(path), {"tags": payload}), source=source,
            metadata=payload,
        )
        self._tracks[track.id] = track
        return [track]
        try:
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            value = out.stdout.strip()
            return round(float(value), 2) if value else None
        except (subprocess.SubprocessError, ValueError):
            return None

    # ---- Status ----

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._list_tracks(self._library_dir()) else ToolStatus.UNAVAILABLE

    # ---- Execution ----

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 1.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        library_dir = self._library_dir(inputs)
        track_paths = self._list_tracks(library_dir)

        tracks: list[dict[str, Any]] = []
        total_duration = 0.0
        have_any_duration = False
        for path in track_paths:
            duration = self._probe_duration(path)
            if duration is not None:
                have_any_duration = True
                total_duration += duration
            tracks.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "duration_seconds": duration,
                    "track": asdict(self._tracks.get(self._track_id(path)))
                    if self._track_id(path) in self._tracks else None,
                }
            )

        return ToolResult(
            success=True,
            data={
                "library_dir": str(library_dir),
                "exists": library_dir.is_dir(),
                "track_count": len(tracks),
                "total_duration_seconds": round(total_duration, 2) if have_any_duration else None,
                "tracks": tracks,
            },
            duration_seconds=round(time.time() - start, 2),
        )
