"""Canonical Phase 4 shot metadata normalization and materialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from phases.phase2.storyboard_assets import _normalize_shot_id


def _first_text(shot: Mapping[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = shot.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _parse_captions(
    caption: str,
    caption_frames: str,
    *,
    fps: int = 30,
) -> list[dict[str, Any]]:
    if not caption or not caption_frames or fps <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    for part in caption_frames.split(","):
        if "-" not in part:
            continue
        start_frame, end_frame = part.split("-", 1)
        ranges.append((int(start_frame.strip()), int(end_frame.strip())))
    text_segments = [text.strip() for text in caption.split("/") if text.strip()]
    if not ranges or not text_segments:
        return []
    if len(ranges) != len(text_segments):
        ranges = [ranges[0]]
    if len(ranges) == 1 and len(text_segments) > 1:
        start_frame, end_frame = ranges[0]
        segment_frames = (end_frame - start_frame) / len(text_segments)
        ranges = [
            (
                round(start_frame + index * segment_frames),
                round(start_frame + (index + 1) * segment_frames),
            )
            for index in range(len(text_segments))
        ]
    return [
        {
            "text": text,
            "start": round(start_frame / fps, 3),
            "end": round(end_frame / fps, 3),
        }
        for text, (start_frame, end_frame) in zip(text_segments, ranges)
    ]


def _route_first_frame(
    first_frame: str | None,
    storyboard_dir: Path | None,
) -> dict[str, Any]:
    if not first_frame:
        return {
            "route": "txt2vid",
            "route_reason": "No reference image — pure text-to-video generation",
            "ref_type": None,
            "first_frame_path": None,
            "first_frame_exists": False,
        }
    if "characters/" in first_frame:
        ref_type = "character"
        reason = f"Character-locked: reference image '{first_frame}' provides character consistency"
    elif "scenes/" in first_frame:
        ref_type = "scene"
        reason = f"Scene-locked: reference image '{first_frame}' provides visual continuity"
    elif "props/" in first_frame:
        ref_type = "prop"
        reason = f"Prop-locked: reference image '{first_frame}' anchors the shot"
    else:
        ref_type = "unknown"
        reason = f"Reference image '{first_frame}' available"
    frame_path = Path(first_frame)
    if not frame_path.is_absolute() and storyboard_dir is not None:
        frame_path = storyboard_dir / frame_path
    return {
        "route": "img2vid",
        "route_reason": reason,
        "ref_type": ref_type,
        "first_frame_path": str(frame_path),
        "first_frame_exists": frame_path.is_file(),
    }


def normalize_shots(
    storyboard: Mapping[str, Any],
    *,
    storyboard_dir: Path | None,
) -> list[dict[str, Any]]:
    """Validate canonical shots and build the Phase 4 metadata contract."""
    source_shots = storyboard.get("shots")
    if not isinstance(source_shots, list):
        raise ValueError("Phase 4 storyboard must contain a shots array")
    project_aspect_ratio = storyboard.get("aspect_ratio")
    project_width = storyboard.get("width")
    project_height = storyboard.get("height")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, source in enumerate(source_shots):
        if not isinstance(source, dict):
            raise ValueError(f"Phase 4 shot at index {index} must be an object")
        shot_id = _normalize_shot_id(source)
        if shot_id is None:
            raise ValueError(f"Phase 4 shot at index {index} has no usable shot ID")
        try:
            numeric_id = int(shot_id[1:])
        except ValueError as exc:
            raise ValueError(
                f"Phase 4 shot at index {index} has a non-numeric shot ID: {shot_id}"
            ) from exc
        if numeric_id <= 0:
            raise ValueError(
                f"Phase 4 shot at index {index} has an invalid shot ID: {shot_id}"
            )
        if shot_id in seen_ids:
            raise ValueError(f"Phase 4 storyboard contains duplicate shot ID: {shot_id}")
        seen_ids.add(shot_id)
        name = _first_text(source, ("name", "shot_intent", "caption", "action", "what", "visual", "prompt")) or shot_id
        prompt = _first_text(source, ("prompt", "visual", "action", "what"))
        if prompt is None:
            raise ValueError(
                f"Phase 4 shot {shot_id} has no prompt-compatible visual, action, or what field"
            )
        first_frame = _first_text(source, ("first_frame",))
        shot = {
            "id": numeric_id,
            "shot_id": shot_id,
            "name": name,
            "duration": source.get("duration", source.get("suggested_duration", 7)),
            "prompt": prompt,
            "first_frame": first_frame,
            "caption": source.get("caption", ""),
            "caption_frames": source.get("caption_frames", ""),
            "who": source.get("who", []),
            "character_ids": source.get("character_ids", []),
            "participant_refs": source.get("participant_refs", []),
            "associate_assets": source.get("associate_assets", []),
            "gen_strategy": source.get("gen_strategy", "i2v"),
            "visual": source.get("visual", ""),
            "what": source.get("what", ""),
            "action_description": source.get("action_description", source.get("what", "")),
            "generation_actions": source.get("generation_actions", []),
            "body_action_choreography": source.get("body_action_choreography", []),
            "body_action_contract": source.get("body_action_contract"),
            "generation_load": source.get("generation_load"),
            "source_action_unit_ids": source.get("source_action_unit_ids", []),
            "start_state": source.get("start_state", ""),
            "end_state": source.get("end_state", ""),
            "causal_link": source.get("causal_link", ""),
            "shot_type": source.get("shot_type"),
            "shot_size": source.get("shot_size"),
            "camera_movement": source.get("camera_movement"),
            "where": source.get("where", ""),
            "time": source.get("time", ""),
            "time_of_day": source.get("time_of_day", ""),
            "time_window": source.get("time_window", ""),
            "source_time_values": source.get("source_time_values", []),
            "temporal_visual_contract": source.get("temporal_visual_contract"),
            "lighting_description": source.get("lighting_description", ""),
            "emotion": source.get("emotion", ""),
            "dialogue": source.get("dialogue"),
            "speech_duration_s": source.get("speech_duration_s", 0),
            "audio": source.get("audio", source.get("sound")),
            "aspect_ratio": source.get("aspect_ratio", project_aspect_ratio),
            "width": source.get("width", project_width),
            "height": source.get("height", project_height),
        }
        shot.update(_route_first_frame(first_frame, storyboard_dir))
        normalized.append(shot)
    return normalized


def materialize_shot_directories(
    shots_dir: Path,
    shots: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Atomically write one canonical ``SHOT_META.json`` per normalized shot."""
    shots_dir.mkdir(parents=True, exist_ok=True)
    meta_paths: list[Path] = []
    for shot in shots:
        shot_id = str(shot["shot_id"])
        shot_dir = shots_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "frames").mkdir(exist_ok=True)
        meta = {
            key: shot.get(key)
            for key in (
                "shot_id", "name", "duration", "prompt", "route", "route_reason",
                "ref_type", "first_frame_path", "first_frame_exists", "caption",
                "caption_frames", "who", "character_ids", "participant_refs",
                "associate_assets", "gen_strategy", "visual",
                "what", "action_description", "generation_actions",
                "body_action_choreography", "body_action_contract", "generation_load",
                "source_action_unit_ids", "start_state", "end_state", "causal_link",
                "shot_type", "shot_size", "camera_movement", "where", "time",
                "time_of_day", "time_window", "source_time_values",
                "temporal_visual_contract", "lighting_description", "emotion", "dialogue",
                "speech_duration_s", "audio", "aspect_ratio", "width", "height",
            )
        }
        meta.update(
            {
                "captions": _parse_captions(
                    str(shot.get("caption") or ""),
                    str(shot.get("caption_frames") or ""),
                ),
                "status": "pending",
                "task_id": None,
                "video_path": None,
            }
        )
        meta_path = shot_dir / "SHOT_META.json"
        temporary = meta_path.with_suffix(meta_path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, meta_path)
        finally:
            temporary.unlink(missing_ok=True)
        meta_paths.append(meta_path)
    return meta_paths
