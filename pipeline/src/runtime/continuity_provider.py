"""Phase 6 provider adapters for calibrated continuity chunk execution."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from quality.seam_calibration import SeamCalibration
from runtime.artifact_manifest import ArtifactManifestStore
from runtime.bridge_execution import execute_bridge_video_task
from runtime.cache_lineage import generation_cache_key
from runtime.capacity import (
    CrossProcessSlotTable,
    SlotTable,
    default_capacity_lease_path,
)
from runtime.continuity_chunks import (
    ChunkExecutionRequest,
    ChunkExecutionResult,
    execute_continuity_plan,
)
from runtime.execution_errors import ProviderPreparationError
from runtime.generation_fingerprint import (
    PHASE6_VIDEO_PROMPT_TEMPLATE_ID,
    PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION,
    build_generation_fingerprint,
)
from runtime.generation_tasks import GenerationTaskStore
from runtime.provider_policy import ProviderExecutionPolicy
from runtime.security_boundaries import validate_subprocess_args
from runtime.seedance_execution import execute_seedance_video_task
from schemas.continuity import ContinuityPlan, GenerationChunk
from utils.storyboard_motion_policy import apply_storyboard_motion_policy
from utils.video_capabilities import SEEDANCE_2_CAPABILITIES
from utils.video_generation_contracts import ensure_video_generation_contract
from utils.video_geometry import resolve_video_geometry

CONTINUITY_BRIDGE_ENV = "HONCUT_CONTINUITY_BRIDGE"
CONTINUITY_BRIDGE_MODES = {"off", "auto"}
SEAM_DECISIONS_KIND = "honcut.continuity_seam_decisions.v1"
SEEDANCE_MAX_REFERENCE_IMAGES = SEEDANCE_2_CAPABILITIES.max_reference_images or 9
CONTINUITY_ANCHOR_FRAME_COUNT = SEEDANCE_2_CAPABILITIES.continuity_anchor_frame_count
STORYBOARD_GUIDE_BASE_IMAGE_BUDGET = (
    SEEDANCE_MAX_REFERENCE_IMAGES - CONTINUITY_ANCHOR_FRAME_COUNT
)
SEEDANCE_MIN_IMAGE_ASPECT = 0.40
SEEDANCE_MAX_IMAGE_ASPECT = 2.50
SEEDANCE_IMAGE_ASPECT_MARGIN = 0.01
MAX_COPYRIGHT_POLICY_REPAIRS = 2
COPYRIGHT_POLICY_REPAIR_VERSION = "original_audio_frame_fallback_v1"
MAX_PRIVACY_POLICY_REPAIRS = SEEDANCE_MAX_REFERENCE_IMAGES + 1
PRIVACY_POLICY_REPAIR_VERSION = "provider_indexed_media_fallback_v2"
MEDIA_ROLE_ISOLATION_CONTRACT = "honcut.phase6-media-role-isolation.v1"
SEEDANCE_ALL_MODAL_PROMPT_CONTRACT = (
    "all_modal_reference_with_role_isolation_v4"
)
_COPYRIGHT_SAFE_AUDIO_CONTRACT = (
    "[copyright-safe audio contract] Generate original ambient location sounds only: "
    "natural footsteps, clothing movement, crowd presence, and location ambience. "
    "No music, soundtrack, song, melody, lyrics, or recognizable tune. No copyrighted "
    "audio, sampled recording, or imitation of any existing artist. This instruction overrides "
    "any earlier soundtrack, music, or rhythm-audio request; visual body-motion timing remains unchanged.\n"
)
_FRAME_ONLY_CONTINUITY_CONTRACT = (
    "[copyright-safe frame-only continuity fallback] The rejected reference-video item "
    "has been removed. Continue only from the remaining ordered reference images and the "
    "authored storyboard start/action/end states. Preserve their visible subject identity, "
    "position, direction, camera, and lighting without reconstructing or quoting any audio "
    "from the rejected media.\n"
)
_PRIVACY_FRAME_ONLY_CONTINUITY_CONTRACT = (
    "[privacy-safe frame-only continuity fallback] The provider-rejected tail-window "
    "video has been removed. Continue from the remaining ordered tail-state frames. "
    "They preserve the predecessor's final position, direction, speed, camera and "
    "lighting. Earlier picture numbers may have changed after rejected media removal; "
    "use the remaining continuity anchors in their current order and do not reset or "
    "replay the action.\n"
)
_PRIVACY_VIDEO_ONLY_CONTINUITY_CONTRACT = (
    "[privacy-safe tail-video continuity fallback] One or more provider-rejected still "
    "references have been removed. Continue from the remaining tail-window video and "
    "continuity anchors only. Earlier picture numbers may have changed; never infer or "
    "recreate a removed reference.\n"
)


class RequiredContinuityEndpointPrivacyError(RuntimeError):
    """A provider rejected an inseparable first/last-frame bridge endpoint."""


def probe_continuity_frames(path: Path, timeline_fps: int) -> dict[str, Any]:
    """Count decoded video frames; duration metadata is only a fallback."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot count continuity frames in {path}")
    streams = json.loads(completed.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"continuity artifact has no video stream: {path}")
    stream = streams[0]
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frames in (None, "N/A"):
        duration = float(stream.get("duration") or 0.0)
        frames = round(duration * timeline_fps)
    numerator, denominator = str(stream.get("avg_frame_rate") or f"{timeline_fps}/1").split(
        "/", 1
    )
    source_fps = float(numerator) / max(float(denominator), 1.0)
    return {
        "frames": int(frames),
        "duration_s": round(int(frames) / source_fps, 6),
        "source_fps": round(source_fps, 6),
    }


def finalize_continuity_shot(
    output_path: Path,
    target_frames: int,
    timeline_fps: int,
) -> dict[str, Any]:
    """Enforce the pre-Phase-8 minimum duration without discarding valid frames."""
    before = probe_continuity_frames(output_path, timeline_fps)
    actual_frames = int(before["frames"])
    delta = target_frames - actual_frames
    if delta == 0:
        return {
            "method": "none",
            "before_frames": actual_frames,
            "target_frames": target_frames,
            "after_frames": actual_frames,
        }
    if delta < 0:
        return {
            "method": "deferred_phase8_excess_trim",
            "before_frames": actual_frames,
            "target_frames": target_frames,
            "after_frames": actual_frames,
            "excess_frames": -delta,
            "minimum_target_met": True,
        }
    if delta > max(2, math.ceil(target_frames * 0.02)):
        raise RuntimeError(
            f"{output_path.parent.name} is short by {delta} frames; "
            "additional continuation generation is required"
        )

    temporary = output_path.with_name(f"{output_path.stem}.duration_closure{output_path.suffix}")
    stretch = target_frames / actual_frames
    video_filter = (
        f"setpts={stretch:.12f}*PTS,fps={timeline_fps},"
        f"trim=end_frame={target_frames},setpts=PTS-STARTPTS"
    )
    method = "bounded_micro_retime"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_path),
            "-vf",
            video_filter,
            "-frames:v",
            str(target_frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot close {output_path.parent.name} duration: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    after = probe_continuity_frames(temporary, timeline_fps)
    if int(after["frames"]) != target_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"duration closure produced {after['frames']} frames, expected {target_frames}"
        )
    os.replace(temporary, output_path)
    return {
        "method": method,
        "before_frames": actual_frames,
        "target_frames": target_frames,
        "after_frames": int(after["frames"]),
        "adjustment_frames": delta,
    }


def normalize_provider_minimum_padding(
    input_path: Path,
    chunk: GenerationChunk,
    timeline_fps: int,
) -> dict[str, Any]:
    """Retime provider-minimum padding away while preserving both endpoints."""
    padding_frames = int(chunk.expected_provider_padding_frames)
    if padding_frames <= 0:
        return {
            "method": "none",
            "output_path": str(input_path),
            "provider_padding_frames": 0,
        }
    if chunk.expected_unique_frames is None:
        raise ValueError(
            f"{chunk.chunk_id} provider padding requires expected_unique_frames"
        )
    before = probe_continuity_frames(input_path, timeline_fps)
    source_duration = float(before["duration_s"])
    target_frames = int(chunk.expected_unique_frames)
    target_duration = target_frames / timeline_fps
    if source_duration <= 0:
        raise RuntimeError(f"{chunk.chunk_id} provider output has no positive duration")
    speed_ratio = target_duration / source_duration
    destination = input_path.with_name(
        f"{input_path.stem}.story_clock{input_path.suffix}"
    )
    temporary = destination.with_name(
        f"{destination.stem}.tmp{destination.suffix}"
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            (
                f"setpts={speed_ratio:.12f}*PTS,fps={timeline_fps},"
                f"trim=end_frame={target_frames},setpts=PTS-STARTPTS"
            ),
            "-frames:v",
            str(target_frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot normalize provider padding for {chunk.chunk_id}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    after = probe_continuity_frames(temporary, timeline_fps)
    if int(after["frames"]) != target_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{chunk.chunk_id} provider-padding normalization produced "
            f"{after['frames']} frames, expected {target_frames}"
        )
    os.replace(temporary, destination)
    return {
        "method": "provider_minimum_endpoint_preserving_retime",
        "output_path": str(destination),
        "source_frames": int(before["frames"]),
        "target_frames": target_frames,
        "provider_padding_frames": padding_frames,
    }


def _read_shot_meta(output_dir: Path, shot_id: str) -> dict[str, Any]:
    path = output_dir / "shots" / shot_id / "SHOT_META.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"continuity auto requires {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _generation_seed(request: ChunkExecutionRequest) -> int | None:
    scene = str(request.anchors.get("scene") or "").strip()
    if not scene:
        return None
    seed_material = json.dumps(
        {
            "scene": scene,
            "chunk_id": request.chunk.chunk_id,
            "repair_attempt": request.repair_attempt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % 2_147_483_647


def _chunk_unique_duration(chunk: GenerationChunk) -> float:
    """Recover effective story time without counting replay/reference overlap."""
    request_duration = float(chunk.target_duration_s)
    if chunk.requested_frames and chunk.expected_unique_frames:
        return request_duration * chunk.expected_unique_frames / chunk.requested_frames
    return request_duration


def _validated_seedance_chunk_duration(
    chunk: GenerationChunk,
    resource_id: str,
) -> int:
    request_duration, _unique_duration = SEEDANCE_2_CAPABILITIES.validate_chunk_durations(
        float(chunk.target_duration_s),
        _chunk_unique_duration(chunk),
        chunk.execution_strategy,
        resource_id=resource_id,
    )
    return int(request_duration)


def _chunk_duration(request: ChunkExecutionRequest) -> int:
    return _validated_seedance_chunk_duration(request.chunk, request.resource_id)


def _validate_seedance_continuity_plan(plan: ContinuityPlan) -> None:
    """Reject every invalid request before initializing a paid provider route."""
    errors: list[str] = []
    for shot in plan.shots:
        for chunk in shot.chunks:
            try:
                _validated_seedance_chunk_duration(chunk, chunk.chunk_id)
            except ValueError as exc:
                errors.append(str(exc))
            if chunk.storyboard_beat_id and not chunk.storyboard_narrative_guide:
                errors.append(
                    f"{chunk.chunk_id} has no validated storyboard narrative guide"
                )
            if chunk.character_performance_required and not (
                chunk.character_performance_guides
            ):
                errors.append(
                    f"{chunk.chunk_id} has no validated character performance guide"
                )
            if chunk.sequence == 1 and chunk.execution_strategy == "multi_image":
                if not chunk.storyboard_image:
                    errors.append(
                        f"{chunk.chunk_id} has no primary-shot cinematic composition"
                    )
            elif chunk.sequence > 1 and chunk.storyboard_image:
                errors.append(
                    f"{chunk.chunk_id} must extend predecessor video without a new cinematic frame"
                )
    for bridge in plan.bridges:
        try:
            SEEDANCE_2_CAPABILITIES.validate_chunk_durations(
                bridge.target_duration_s,
                bridge.target_duration_s,
                "first_last_frame_bridge",
                resource_id=bridge.bridge_id,
            )
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        detail = "; ".join(errors[:8])
        if len(errors) > 8:
            detail += f"; and {len(errors) - 8} more"
        raise RuntimeError(
            "Seedance continuity preflight failed before any paid provider "
            f"submission: {detail}. Regenerate Phase 1-4 artifacts with the current "
            "secondary-storyboard contract"
        )


def _video_geometry(shot_meta: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve provider ratio and Bridge dimensions from the authored shot."""
    return resolve_video_geometry(shot_meta)


def _storyboard_group_for_shot(
    output_dir: Path,
    shot_id: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    path = output_dir / "STORYBOARD_GROUPS.json"
    if not path.is_file():
        return None, None
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    group_id = (contract.get("shot_to_group") or {}).get(shot_id)
    group = next(
        (
            item for item in contract.get("groups", [])
            if isinstance(item, dict) and item.get("group_id") == group_id
        ),
        None,
    )
    if group is None:
        return None, None
    board_value = group.get("storyboard_board")
    board = output_dir / str(board_value) if board_value else None
    return group, board if board is not None and board.is_file() else None


def _storyboard_group_prompt(
    group: dict[str, Any] | None,
    shot_id: str,
    chunk: GenerationChunk | None = None,
) -> str:
    if not group:
        return ""
    beats = [beat for beat in group.get("beats", []) if isinstance(beat, dict)]
    position = next((index for index, beat in enumerate(beats) if beat.get("shot_id") == shot_id), None)
    if position is None:
        return ""
    current = beats[position]
    previous = beats[position - 1] if position > 0 else None
    following = beats[position + 1] if position + 1 < len(beats) else None
    handoff = group.get("handoff_from_previous") or {}
    has_inner_beats = bool(current.get("storyboard_beats"))
    if has_inner_beats and chunk is not None and chunk.storyboard_beat_id:
        return "\n".join(
            (
                f"[storyboard group {group.get('group_id')}; step {position + 1}/{len(beats)}]",
                "Text map only; render the authoritative current Pxx beat.",
                f"Start: {chunk.start_state or 'continue the supplied state'}",
                f"End: {chunk.end_state or 'complete the current beat action'}",
                "Use credible body mechanics for only the current 1-2 actions.",
                "No later/earlier Pxx, combined action, replay, or collage.",
            )
        )
    actions = " -> ".join(str(value) for value in current.get("generation_actions", []))
    lines = [
        f"[storyboard group {group.get('group_id')}; step {position + 1}/{len(beats)}]",
        "The text-only group contract is a chronological narrative map, not a request to perform every panel now.",
        (
            f"Previous shot final state: {previous.get('end_state', '')}"
            if previous
            else (
                f"Previous generation group {group.get('previous_group_id')} ended at: "
                f"{handoff.get('previous_end_state', '')}. Begin this fresh editorial cut from the declared current start state."
                if handoff
                else "This is the fresh story entry."
            )
        ),
        f"Current shot starting state: {current.get('start_state', '')}",
        (
            "The authoritative Pxx contract below defines the only action to execute now."
            if has_inner_beats
            else f"Execute only this current shot action contract: {actions or 'no authored body action'}"
        ),
        f"Current shot required result: {current.get('end_state', '')}",
        f"Next shot will begin from: {following.get('start_state', '')}" if following else "This is the final shot in the group.",
        "Do not jump to a later panel, combine later actions, replay the previous panel, or render a collage.",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def _chunk_scoped_shot_meta(
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
) -> dict[str, Any]:
    """Return a provider-only view containing the current beat, never the full ledger."""
    from utils.body_action_contracts import apply_body_action_contract

    scoped = copy.deepcopy(shot_meta)
    beat_id = str(request.chunk.storyboard_beat_id or "").strip()
    action = str(request.chunk.action_prompt or "").strip()
    if (
        request.chunk.execution_strategy != "first_last_frame_bridge"
        and not beat_id
        and not action
    ):
        return scoped
    scoped.pop("prompt", None)
    scoped.pop("storyboard_beats", None)
    scoped.pop("body_action_contract", None)
    for ledger_field in (
        "generation_action_units",
        "generation_action_categories",
        "source_action_unit_ids",
        "source_event_slices",
        "subject_description",
    ):
        scoped.pop(ledger_field, None)

    if request.chunk.execution_strategy == "first_last_frame_bridge":
        bridge_action = str(request.chunk.action_prompt or "").strip()
        scoped.pop("phase8_reshoot", None)
        scoped["generation_actions"] = []
        scoped["micro_actions"] = []
        scoped["body_action_choreography"] = []
        scoped["action"] = bridge_action
        scoped["action_description"] = bridge_action
        scoped["what"] = bridge_action or "仅生成两个已完成主镜头之间的连续过渡"
        scoped["visual"] = scoped["what"]
        scoped["start_state"] = request.chunk.start_state
        scoped["end_state"] = request.chunk.end_state
        return scoped

    source_beats = [
        beat
        for beat in (shot_meta.get("storyboard_beats") or [])
        if isinstance(beat, dict)
    ]
    matched = next(
        (beat for beat in source_beats if str(beat.get("beat_id") or "") == beat_id),
        {},
    )

    def _action_terms(beat: dict[str, Any]) -> set[str]:
        terms: set[str] = set()
        for field_name in ("action", "action_prompt", "micro_actions", "generation_actions"):
            values = beat.get(field_name) or []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                terms.update(
                    str(value).strip()
                    for value in values
                    if len(str(value).strip()) >= 3
                )
        for body_beat in beat.get("body_action_choreography") or []:
            if not isinstance(body_beat, dict):
                continue
            for field_name in ("technique", "micro_action", "description"):
                value = str(body_beat.get(field_name) or "").strip()
                if len(value) >= 3:
                    terms.add(value)
        return terms

    current_terms = _action_terms(matched)
    if action:
        current_terms.add(action)
    offbeat_terms: set[str] = set()
    for source_beat in source_beats:
        if source_beat is not matched:
            offbeat_terms.update(_action_terms(source_beat))
    offbeat_terms.difference_update(current_terms)
    scoped["_forbidden_offbeat_action_terms"] = sorted(offbeat_terms)

    action = action or str(
        matched.get("action")
        or matched.get("action_prompt")
        or matched.get("visual")
        or ""
    ).strip()
    choreography = copy.deepcopy(matched.get("body_action_choreography") or [])
    if not choreography and action:
        choreography = [
            copy.deepcopy(beat)
            for beat in (shot_meta.get("body_action_choreography") or [])
            if isinstance(beat, dict)
            and str(beat.get("micro_action") or beat.get("description") or "").strip()
            == action
        ]
    if not choreography and beat_id:
        match = re.search(r"_P(\d+)$", beat_id, re.IGNORECASE)
        if match:
            position = int(match.group(1))
            choreography = [
                copy.deepcopy(beat)
                for beat in (shot_meta.get("body_action_choreography") or [])
                if isinstance(beat, dict)
                and int(beat.get("micro_action_index") or beat.get("beat") or 0)
                == position
            ]

    scoped["generation_actions"] = [action] if action else []
    scoped["micro_actions"] = [action] if action else []
    scoped["body_action_choreography"] = choreography
    scoped["action"] = action
    scoped["action_description"] = action
    scoped["what"] = action or "自然延续当前分镜拍"
    scoped["visual"] = str(matched.get("visual") or action or scoped["what"])
    scoped["start_state"] = request.chunk.start_state or matched.get("start_state") or ""
    scoped["end_state"] = request.chunk.end_state or matched.get("end_state") or ""
    source_fact_echoes = matched.get("source_fact_echoes") or []
    if isinstance(source_fact_echoes, str):
        source_fact_echoes = [source_fact_echoes]
    scoped["source_fact_echoes"] = [
        str(value).strip()
        for value in source_fact_echoes
        if str(value).strip()
    ]
    scoped["source_generation_unit_indexes"] = list(
        matched.get("source_generation_unit_indexes") or []
    )
    if "character_ids" in matched:
        visible_character_ids = matched.get("character_ids") or []
        if not isinstance(visible_character_ids, list):
            raise ValueError(f"{beat_id} character_ids must be an ordered array")
        scoped["who"] = list(visible_character_ids)
        scoped["characters"] = list(visible_character_ids)
        scoped["_char_ids"] = list(visible_character_ids)
    if matched.get("subject_description"):
        scoped["subject_description"] = matched["subject_description"]
    feedback = scoped.get("phase8_reshoot")
    if isinstance(feedback, dict) and offbeat_terms:
        issues = feedback.get("issues") or []
        if isinstance(issues, str):
            issues = [issues]
        feedback["issues"] = [
            issue
            for issue in issues
            if not any(
                term.casefold() in str(issue).casefold()
                for term in offbeat_terms
            )
        ]
        if not feedback["issues"]:
            scoped.pop("phase8_reshoot", None)
    apply_body_action_contract(scoped)
    return scoped


def _chunk_prompt(
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
    group_prompt: str = "",
    characters: Any = None,
) -> str:
    prompt = str(shot_meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"{request.shot_id} SHOT_META.json has no prompt")
    strategy = request.chunk.execution_strategy
    continuation = {
        "multi_image": (
            "Generate the opening secondary beat from the ordered multi-image "
            "identity, environment, and current-storyboard references."
        ),
        "tail_video_extend": (
            "Extend the extracted tail section of the previous secondary-beat "
            "video without a reset, cut, replay, or re-entry."
        ),
        "first_last_frame_bridge": (
            "Generate only the transition between the completed source primary "
            "video's actual final frame and the completed target primary video's "
            "actual first frame. Both primary actions are already complete: do not "
            "add, finish, repeat, or reinterpret plot action."
        ),
    }.get(
        strategy,
        (
            "Generate the opening chunk of this editorial shot."
            if request.chunk.mode == "fresh"
            else "Continue directly from the reference video's final state without a reset or cut."
        ),
    )
    repair = ""
    if request.repair_attempt > 0:
        repair = (
            "\n[approved seam repair] Preserve subject position, scale, orientation, "
            "screen direction, camera framing, lighting, and surrounding motion state. "
            "Do not skip forward in time or reposition the subject before continuing."
        )
    beat_contract = ""
    if request.chunk.storyboard_beat_id:
        scoped_actions = shot_meta.get("generation_actions") or []
        if isinstance(scoped_actions, str):
            scoped_actions = [scoped_actions]
        current_action = str(
            request.chunk.action_prompt
            or next((value for value in scoped_actions if str(value).strip()), "")
            or shot_meta.get("action")
            or "natural scene progression"
        ).strip()
        current_start = str(
            request.chunk.start_state
            or shot_meta.get("start_state")
            or "continue the supplied state"
        ).strip()
        current_end = str(
            request.chunk.end_state
            or shot_meta.get("end_state")
            or "complete that action"
        ).strip()
        source_fact_echoes = shot_meta.get("source_fact_echoes") or []
        if isinstance(source_fact_echoes, str):
            source_fact_echoes = [source_fact_echoes]
        source_fact_contract = (
            " Source facts that must remain visibly represented in this Pxx: "
            + " | ".join(
                str(value).strip()
                for value in source_fact_echoes
                if str(value).strip()
            )
            + "."
            if source_fact_echoes
            else ""
        )
        beat_contract = (
            f"\n[authoritative storyboard beat {request.chunk.storyboard_beat_id}] "
            f"Start state: {current_start}. "
            f"Execute only this visible action: {current_action}. "
            f"Required end state: {current_end}. "
            f"{source_fact_contract} "
            "Specify the current limbs, contact force, center of gravity, weight "
            "transfer, resistance, and inertia with physically credible motion. "
            "Do not execute another Pxx panel, skip ahead, or replay an earlier panel."
        )
    prompt = apply_storyboard_motion_policy(
        f"{prompt}\n\n[continuity chunk {request.chunk.sequence}] {continuation}\n"
        f"{group_prompt}\n{request.memory_context}{beat_contract}{repair}"
    )
    prompt = ensure_video_generation_contract(prompt, shot_meta, characters or {})
    from utils.body_action_contracts import body_action_prompt

    choreography = body_action_prompt(shot_meta)
    marker_count = prompt.count("[逐拍肢体动作谱｜不可摘要]")
    expected_count = 1 if choreography else 0
    if marker_count != expected_count:
        raise ValueError(
            f"{request.resource_id} choreography marker count is {marker_count}, "
            f"expected {expected_count} for the current beat"
        )
    leaked_terms = [
        term
        for term in shot_meta.get("_forbidden_offbeat_action_terms", [])
        if str(term).casefold() in prompt.casefold()
    ]
    if leaked_terms:
        raise ValueError(
            f"{request.resource_id} prompt leaked non-current beat action terms: "
            + ", ".join(leaked_terms[:5])
        )
    return prompt


def _seedance_reference_image_payload(board_path: Path) -> tuple[bytes, str]:
    """Return a complete, uncropped board inside Seedance's aspect limits."""
    raw = board_path.read_bytes()
    try:
        import io

        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw)) as source:
            image_format = str(source.format or "PNG").upper()
            image = ImageOps.exif_transpose(source).convert("RGB")
        content_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(image_format, "image/png")
        width, height = image.size
        aspect = width / height
        if SEEDANCE_MIN_IMAGE_ASPECT <= aspect <= SEEDANCE_MAX_IMAGE_ASPECT:
            return raw, content_type

        safe_min = SEEDANCE_MIN_IMAGE_ASPECT + SEEDANCE_IMAGE_ASPECT_MARGIN
        safe_max = SEEDANCE_MAX_IMAGE_ASPECT - SEEDANCE_IMAGE_ASPECT_MARGIN
        target_width = max(width, math.ceil(height * safe_min))
        target_height = max(height, math.ceil(width / safe_max))
        corners = (
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        )
        fill = tuple(
            round(sum(pixel[channel] for pixel in corners) / 4)
            for channel in range(3)
        )
        canvas = Image.new("RGB", (target_width, target_height), fill)
        canvas.paste(image, ((target_width - width) // 2, (target_height - height) // 2))

        encoded = io.BytesIO()
        output_format = image_format if image_format in {"JPEG", "PNG", "WEBP"} else "PNG"
        save_options = (
            {"quality": 95, "optimize": True}
            if output_format in {"JPEG", "WEBP"}
            else {"optimize": True}
        )
        canvas.save(encoded, format=output_format, **save_options)
        print(
            "  [continuity] padded storyboard board for Seedance: "
            f"{width}x{height} ({aspect:.3f}) -> "
            f"{target_width}x{target_height} ({target_width / target_height:.3f})"
        )
        return encoded.getvalue(), content_type
    except (ImportError, OSError, ValueError):
        import mimetypes

        content_type = mimetypes.guess_type(board_path.name)[0] or "application/octet-stream"
        return raw, content_type


def _append_group_board_reference(
    content: list[dict[str, Any]],
    board_path: Path | None,
) -> list[dict[str, Any]]:
    """Retain the legacy continuity-group board seam as an enforced no-op.

    ``storyboard_groups/CGxx`` is not the Phase 2 per-Sxx narrative guide.
    Keeping this helper as a no-op prevents that obsolete group-board pixel
    contract from being confused with ``storyboard_narrative_guide``.
    """
    _ = board_path
    return content


def _base_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
    shot_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    from phases.phase6.video_generator import build_video_prompt
    from tools.asset_packager import build_content_for_shot

    content_meta = _chunk_scoped_shot_meta(request, shot_meta)
    group, _ = _storyboard_group_for_shot(output_dir, request.shot_id)
    try:
        characters_data = json.loads(
            (output_dir / "CHARACTERS.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        characters_data = {}
    try:
        scene_consistency = json.loads(
            (output_dir / "SCENE_CONSISTENCY.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        scene_consistency = {}
    model = os.environ.get("VIDEO_MODEL") or os.environ.get("SEEDANCE_MODEL") or "seedance"
    should_render_current_beat = bool(
        request.chunk.storyboard_beat_id
        or request.chunk.action_prompt
        or request.chunk.execution_strategy == "first_last_frame_bridge"
    )
    if should_render_current_beat:
        rendered = build_video_prompt(
            content_meta,
            characters_data,
            scene_consistency,
            model,
        )
        if isinstance(rendered, dict):
            rendered = rendered.get("prompt") or ""
        content_meta["prompt"] = str(rendered)
    content_meta["prompt"] = _chunk_prompt(
        request,
        content_meta,
        _storyboard_group_prompt(group, request.shot_id, request.chunk),
        characters_data,
    )
    from utils.privacy_visual_policy import (
        is_no_real_person_enabled,
        no_real_person_prompt_contract,
    )

    if is_no_real_person_enabled():
        privacy_contract = no_real_person_prompt_contract()
        if privacy_contract not in content_meta["prompt"]:
            content_meta["prompt"] = (
                f"{privacy_contract}\n{content_meta['prompt']}"
            )
    strategy = request.chunk.execution_strategy
    if strategy == "first_last_frame_bridge":
        # Frame roles cannot be mixed with reference media in Seedance.  The
        # bridge helper adds the actual predecessor tail and next P01 below.
        return [{"type": "text", "text": content_meta["prompt"]}]
    content_meta["_include_cinematic_frame"] = request.chunk.sequence == 1
    if request.chunk.storyboard_beat_id:
        content_meta["_storyboard_beat_id"] = request.chunk.storyboard_beat_id
    if strategy in {"multi_image", "tail_video_extend"}:
        content_meta["gen_strategy"] = "phantom"
    if request.chunk.storyboard_image:
        content_meta["_storyboard_frame_path"] = request.chunk.storyboard_image
        content_meta["_storyboard_frame_kind"] = request.chunk.storyboard_image_kind
        content_meta["generation_actions"] = [request.chunk.action_prompt]
        if strategy not in {"multi_image", "tail_video_extend"}:
            content_meta["gen_strategy"] = "i2v"
    if request.chunk.storyboard_beat_id:
        if not request.chunk.storyboard_narrative_guide:
            raise ValueError(
                f"{request.resource_id} has no storyboard narrative guide"
            )
        content_meta.update({
            "_storyboard_narrative_guide_path": (
                request.chunk.storyboard_narrative_guide
            ),
            "_storyboard_narrative_guide_kind": (
                request.chunk.storyboard_narrative_guide_kind
            ),
            "_storyboard_narrative_guide_usage": (
                request.chunk.storyboard_narrative_guide_usage
            ),
            "_storyboard_narrative_guide_cell_ids": list(
                request.chunk.storyboard_narrative_guide_cell_ids
            ),
            "_storyboard_narrative_guide_sha256": (
                request.chunk.storyboard_narrative_guide_sha256
            ),
            "_storyboard_narrative_guide_source_board": (
                request.chunk.storyboard_narrative_guide_source_board
            ),
            "_storyboard_narrative_guide_source_board_sha256": (
                request.chunk.storyboard_narrative_guide_source_board_sha256
            ),
            "_storyboard_narrative_guide_receipt": (
                request.chunk.storyboard_narrative_guide_receipt
            ),
        })
        content_meta["_character_performance_required"] = (
            request.chunk.character_performance_required
        )
        content_meta["_character_performance_guides"] = [
            guide.model_dump(mode="json")
            for guide in request.chunk.character_performance_guides
        ]
    if request.chunk.storyboard_beat_id:
        # Keep the mandatory cinematic/identity/guide set, then admit optional
        # detail assets only up to the same six-image base budget used before
        # the three P02+ tail anchors are appended. This deterministic priority
        # bound also prevents optional media descriptions from exhausting the
        # Seedance prompt budget.
        content_meta["_max_reference_images"] = SEEDANCE_MAX_REFERENCE_IMAGES
    if request.chunk.mode == "native_extend":
        content_meta["_max_reference_images"] = (
            SEEDANCE_MAX_REFERENCE_IMAGES
            - CONTINUITY_ANCHOR_FRAME_COUNT
        )
    content = build_content_for_shot(
        output_dir=output_dir,
        shot_id=request.shot_id,
        shot_meta=content_meta,
    )
    if not content:
        content = [{"type": "text", "text": content_meta["prompt"]}]
    return content


def _extension_content(
    content: Sequence[dict[str, Any]],
    previous_output_path: Path,
) -> list[dict[str, Any]]:
    from clients.tos_uploader import upload_media_file_required
    from quality.continuity_seam import (
        extract_ordered_video_frames,
        extract_video_tail_window,
    )

    with previous_output_path.open("rb") as predecessor:
        predecessor_hash = hashlib.file_digest(predecessor, "sha256").hexdigest()
    anchor_dir = previous_output_path.parent / "continuity_anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    asset_stem = f"{previous_output_path.stem}_{predecessor_hash[:16]}_tail_2000ms"
    tail_video_path = anchor_dir / f"{asset_stem}.mp4"
    if not tail_video_path.is_file() or tail_video_path.stat().st_size == 0:
        extract_video_tail_window(previous_output_path, tail_video_path, window_s=2.0)
    frame_paths = tuple(anchor_dir / f"{asset_stem}_frame_{index:02d}.jpg" for index in range(1, 4))
    if any(not path.is_file() or path.stat().st_size == 0 for path in frame_paths):
        extract_ordered_video_frames(tail_video_path, frame_paths)
    video_url = upload_media_file_required(
        tail_video_path,
        prefix="volcengine/video",
        label="continuity tail video",
    )
    frame_urls = []
    for index, path in enumerate(frame_paths, 1):
        frame_urls.append(
            upload_media_file_required(
                path,
                prefix="volcengine/image",
                label=f"ordered continuity anchor image {index}",
            )
        )

    base_images = [dict(item) for item in content if item.get("type") == "image_url"]
    base_images = base_images[
        : SEEDANCE_MAX_REFERENCE_IMAGES - CONTINUITY_ANCHOR_FRAME_COUNT
    ]
    first_anchor_number = len(base_images) + 1
    anchor_numbers = tuple(
        range(first_anchor_number, first_anchor_number + CONTINUITY_ANCHOR_FRAME_COUNT)
    )
    anchor_labels = "、".join(f"图片{number}" for number in anchor_numbers)
    directive = (
        f"向后延长视频1。{anchor_labels}是视频1末段按时间先后顺序截取的状态帧，"
        "只用于判断主体的速度、方向和连续运动。新生成内容的第一帧必须紧接视频1最后一帧，"
        f"并严格参考图片{anchor_numbers[-1]}的最终状态；保持主体位置、大小、朝向、水波相位、机位和灯光，"
        f"继续最后的速度与方向。不得重播视频1中的运动轨迹，不得回到图片{anchor_numbers[0]}或更早位置，"
        "不得让主体重新入画。\n"
    )
    text_items = [dict(item) for item in content if item.get("type") == "text"]
    normalized: list[dict[str, Any]] = text_items or [{"type": "text", "text": ""}]
    text_item = next((item for item in normalized if item.get("type") == "text"), None)
    if text_item is not None:
        text_item["text"] = f"{directive}{text_item.get('text', '')}"
    # P02+ ordering is a semantic contract: predecessor video first, then
    # current-character and narrative-guide images, then local tail anchors.
    normalized.append(
        {
            "type": "video_url",
            "video_url": {"url": video_url},
            "role": "reference_video",
            "_continuity_role": "tail_window_video",
            "_reference_kind": "predecessor_tail_video",
            "_reference_description": "上一 Pxx 已完成视频的末段",
            "_reference_path": str(tail_video_path),
            "_reference_sha256": hashlib.sha256(
                tail_video_path.read_bytes()
            ).hexdigest(),
        }
    )
    for item in base_images:
        item["role"] = "reference_image"
        item.pop("priority", None)
        normalized.append(item)
    normalized.extend(
        {
            "type": "image_url",
            "image_url": {"url": url},
            "role": "reference_image",
            "_continuity_role": "ordered_tail_frame",
            "_reference_kind": "ordered_tail_frame",
            "_reference_description": "视频1末段按时间顺序截取的连续性状态帧",
            "_reference_path": str(path),
            "_reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for url, path in zip(frame_urls, frame_paths, strict=True)
    )
    return normalized


def _media_index_manifest(content: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe final provider media order without storing URLs or media bytes."""
    counters = {"image_url": 0, "video_url": 0, "audio_url": 0}
    labels = {"image_url": "图片", "video_url": "视频", "audio_url": "音频"}
    manifest: list[dict[str, Any]] = []
    for content_index, item in enumerate(content):
        media_type = str(item.get("type") or "")
        if media_type not in counters:
            continue
        counters[media_type] += 1
        manifest.append({
            "content_index": content_index,
            "media_type": media_type,
            "media_index": counters[media_type],
            "prompt_index": f"{labels[media_type]}{counters[media_type]}",
            "role": item.get("role"),
            "responsibility": (
                item.get("_reference_kind")
                or item.get("_continuity_role")
                or item.get("role")
            ),
            "description": item.get("_reference_description"),
            "path": item.get("_reference_path"),
            "sha256": item.get("_reference_sha256"),
            "character_id": item.get("_character_id"),
            "narrative_beat_id": item.get("_narrative_beat_id"),
            "narrative_cell_ids": list(item.get("_narrative_cell_ids") or []),
            "performance_beat_id": item.get("_performance_beat_id"),
            "performance_cell_ids": list(
                item.get("_performance_cell_ids") or []
            ),
            "performance_source_action_unit_ids": list(
                item.get("_performance_source_action_unit_ids") or []
            ),
            "performance_prop_ids": list(
                item.get("_performance_prop_ids") or []
            ),
            "performance_source_board_sha256": item.get(
                "_performance_source_board_sha256"
            ),
            "mandatory": item.get("_mandatory_reference") is True
            or bool(item.get("_continuity_role")),
        })
    return manifest


def _bind_final_media_index_prompt(
    content: Sequence[dict[str, Any]],
    request: ChunkExecutionRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind 图片N/视频N only after the final content order is fixed."""
    normalized = [dict(item) for item in content]
    manifest = _media_index_manifest(normalized)
    if request.chunk.execution_strategy == "first_last_frame_bridge":
        if any(
            item.get("responsibility") in {
                "storyboard_narrative_guide",
                "character_performance_guide",
            }
            for item in manifest
        ):
            raise ValueError(
                "cross-primary bridge must not contain narrative or performance guides"
            )
        return normalized, manifest

    guides = [
        item
        for item in manifest
        if item.get("responsibility") == "storyboard_narrative_guide"
    ]
    if request.chunk.storyboard_beat_id:
        if len(guides) != 1:
            raise ValueError(
                f"{request.resource_id} must contain exactly one current narrative guide"
            )
        guide = guides[0]
        expected_cells = list(request.chunk.storyboard_narrative_guide_cell_ids)
        if (
            guide.get("narrative_beat_id") != request.chunk.storyboard_beat_id
            or guide.get("narrative_cell_ids") != expected_cells
        ):
            raise ValueError(
                f"{request.resource_id} narrative guide does not match its persisted Gxx assignment"
            )
    performance_guides = [
        item
        for item in manifest
        if item.get("responsibility") == "character_performance_guide"
    ]
    expected_performance = request.chunk.character_performance_guides
    if bool(performance_guides) != request.chunk.character_performance_required:
        raise ValueError(
            f"{request.resource_id} performance-guide requirement is not satisfied"
        )
    if len(performance_guides) != len(expected_performance):
        raise ValueError(
            f"{request.resource_id} performance-guide count does not match its plan"
        )
    for observed, expected in zip(
        performance_guides,
        expected_performance,
        strict=True,
    ):
        if (
            observed.get("character_id") != expected.character_id
            or observed.get("performance_beat_id") != expected.beat_id
            or observed.get("performance_cell_ids") != expected.cell_ids
            or observed.get("performance_source_action_unit_ids")
            != expected.source_action_unit_ids
            or observed.get("performance_prop_ids") != expected.prop_ids
            or observed.get("performance_source_board_sha256")
            != expected.source_board_sha256
        ):
            raise ValueError(
                f"{request.resource_id} performance guide does not match persisted Axx lineage"
            )
    if (
        request.chunk.sequence == 1
        and request.chunk.execution_strategy == "multi_image"
        and request.chunk.storyboard_beat_id
    ):
        image_items = [
            item for item in manifest if item.get("media_type") == "image_url"
        ]
        if not image_items or image_items[0].get("responsibility") != (
            "cinematic_composition"
        ):
            raise ValueError(
                f"{request.resource_id} 图片1 must be the primary-shot cinematic composition"
            )
    if request.chunk.mode == "native_extend":
        videos = [
            item for item in manifest if item.get("media_type") == "video_url"
        ]
        if len(videos) != 1 or videos[0].get("prompt_index") != "视频1":
            raise ValueError(
                f"{request.resource_id} must extend exactly one predecessor as 视频1"
            )

    index_lines = []
    for item in manifest:
        description = str(item.get("description") or "").strip()
        if not description:
            responsibility = str(item.get("responsibility") or "参考素材")
            description = responsibility.replace("_", " ")
        index_lines.append(f"{item['prompt_index']}：{description}")
    guide_contract = ""
    if guides:
        guide = guides[0]
        cells = [str(value) for value in guide["narrative_cell_ids"]]
        guide_contract = (
            f"\n当前剧情导航图是{guide['prompt_index']}；本次只按"
            + "→".join(cells)
            + "的顺序演绎，不得提前演绎其他 Gxx 或后续 Pxx。"
            "红色箭头表示主体或物体运动方向；蓝色箭头表示摄影机运动；"
            "其他指示标识表示空间、视线、接触点和动作接续关系。"
            "Gxx 序号、文字、箭头、轨迹线、指示标识、边框和九宫格只供理解，"
            "严禁渲染进视频画面。"
        )
    performance_contract = ""
    if performance_guides:
        directives = []
        for guide in performance_guides:
            cells = [str(value) for value in guide["performance_cell_ids"]]
            directives.append(
                f"{guide['prompt_index']}是角色{guide['character_id']}在当前"
                f"{guide['performance_beat_id']}的动作姿态图，只允许参考"
                + "→".join(cells)
            )
        performance_contract = (
            "\n当前动作姿态图中的多个人形是同一个角色的不同参考姿态，不是多个角色。"
            + "；".join(directives)
            + "。只执行本次明确列出的 Axx，不得提前执行其他 Axx 或后续 Pxx；"
            "不得生成角色克隆、分栏、拼贴、网格、文字、序号、箭头、边框或参考板布局。"
            "道具只属于其绑定角色，并保持声明的几何、材质、颜色与握持关系。"
        )
    role_isolation_contract = ""
    if guides:
        narrative_index = guides[0]["prompt_index"]
        identity_indices = [
            item["prompt_index"]
            for item in manifest
            if item.get("responsibility") == "character_identity_board"
        ]
        performance_indices = [
            item["prompt_index"] for item in performance_guides
        ]
        boundary_indices = [
            item["prompt_index"]
            for item in manifest
            if item.get("responsibility") in {
                "cinematic_composition",
                "predecessor_tail_video",
                "ordered_tail_frame",
            }
        ]
        identity_authority = (
            "、".join(identity_indices) if identity_indices else "结构化角色合同"
        )
        performance_authority = (
            "、".join(performance_indices)
            if performance_indices
            else "结构化动作与道具合同"
        )
        boundary_authority = (
            "、".join(boundary_indices)
            if boundary_indices
            else "结构化起始状态"
        )
        role_isolation_contract = (
            f"\n[{MEDIA_ROLE_ISOLATION_CONTRACT}] "
            f"{narrative_index}只负责 Gxx 动作顺序、关节与重心轨迹、红蓝箭头和空间关系；"
            "其中的人物脸、头发长度与发型轮廓、妆造、服装、身体比例，以及道具外形、"
            "总长度、端部数量、握柄/刃部结构、颜色和材质都是非权威占位像素，严禁带入成片。"
            f"角色身份外观只以{identity_authority}为权威；开场构图与连续状态只以"
            f"{boundary_authority}为权威；当前姿态、握持关系和道具几何只以"
            f"{performance_authority}及结构化文字合同为权威。发生冲突时必须按上述职责覆盖"
            f"{narrative_index}，不得折中混合。逐帧保持角色头发长度、发型外轮廓、分缝和发束"
            "数量不变；逐帧保持同一道具的总长度、端部数量、握柄与发光部位分段不变；"
            "禁止生长、缩短、变形、增减端部或在参考职责之间渐变。"
        )
    prefix = (
        "【参考素材索引｜顺序不可交换】\n"
        + "\n".join(index_lines)
        + guide_contract
        + performance_contract
        + role_isolation_contract
        + "\n"
    )
    text_item = next(
        (item for item in normalized if item.get("type") == "text"),
        None,
    )
    if text_item is None:
        normalized.insert(0, {"type": "text", "text": prefix})
    else:
        text_item["text"] = prefix + str(text_item.get("text") or "")
    return normalized, manifest


def _first_last_bridge_content(
    content: Sequence[dict[str, Any]],
    previous_output_path: Path,
    target_output_path: Path,
) -> list[dict[str, Any]]:
    """Bind a transition to completed source-tail and target-head frames."""
    from clients.tos_uploader import require_tos_url, upload_image
    from quality.continuity_seam import (
        extract_video_head_frame,
        extract_video_tail_frame,
    )

    for label, path in (
        ("source primary video", previous_output_path),
        ("target primary video", target_output_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} missing: {path}")
    with previous_output_path.open("rb") as predecessor:
        predecessor_hash = hashlib.file_digest(predecessor, "sha256").hexdigest()
    anchor_dir = previous_output_path.parent / "continuity_anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    tail_frame = anchor_dir / (
        f"{previous_output_path.stem}_{predecessor_hash[:16]}_final.jpg"
    )
    if not tail_frame.is_file() or tail_frame.stat().st_size == 0:
        extract_video_tail_frame(previous_output_path, tail_frame)
    with target_output_path.open("rb") as target:
        target_hash = hashlib.file_digest(target, "sha256").hexdigest()
    head_frame = anchor_dir / (
        f"{target_output_path.parent.name}_{target_hash[:16]}_first.jpg"
    )
    if not head_frame.is_file() or head_frame.stat().st_size == 0:
        extract_video_head_frame(target_output_path, head_frame)

    first_payload, first_content_type = _seedance_reference_image_payload(tail_frame)
    last_payload, last_content_type = _seedance_reference_image_payload(head_frame)
    first_url = require_tos_url(
        upload_image(first_payload, first_content_type),
        label="secondary-storyboard first-frame image",
    )
    last_url = require_tos_url(
        upload_image(last_payload, last_content_type),
        label="secondary-storyboard last-frame image",
    )
    directive = (
        "图片1来自已完成的上一一级分镜视频真实尾帧，必须作为新视频第一帧；"
        "图片2来自已完成的下一一级分镜视频真实首帧，必须作为新视频最后一帧。"
        "只生成两帧之间的连续摄影机与动作过渡，不得重复两侧已完成剧情，"
        "不得新增剧情、角色、道具、文字、箭头、轨迹线、编号或分格标记。\n"
    )
    text_value = next(
        (str(item.get("text") or "") for item in content if item.get("type") == "text"),
        "",
    )
    return [
        {"type": "text", "text": f"{directive}{text_value}"},
        {
            "type": "image_url",
            "image_url": {"url": first_url},
            "role": "first_frame",
            "priority": "high",
            "_continuity_role": "required_first_frame",
            "_reference_kind": "cross_shot_source_tail_frame",
            "_reference_description": "上一一级分镜视频真实尾帧",
            "_reference_path": str(tail_frame),
            "_reference_sha256": hashlib.sha256(
                tail_frame.read_bytes()
            ).hexdigest(),
        },
        {
            "type": "image_url",
            "image_url": {"url": last_url},
            "role": "last_frame",
            "priority": "high",
            "_continuity_role": "required_last_frame",
            "_reference_kind": "cross_shot_target_head_frame",
            "_reference_description": "下一一级分镜视频真实首帧",
            "_reference_path": str(head_frame),
            "_reference_sha256": hashlib.sha256(
                head_frame.read_bytes()
            ).hexdigest(),
        },
    ]


def _provider_content(
    output_dir: Path,
    request: ChunkExecutionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any], int | None, int]:
    shot_meta = _read_shot_meta(output_dir, request.shot_id)
    content = _base_content(output_dir, request, shot_meta)
    strategy = request.chunk.execution_strategy
    if strategy == "first_last_frame_bridge":
        if request.previous_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no predecessor video")
        if request.target_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no completed target video")
        content = _first_last_bridge_content(
            content,
            request.previous_output_path,
            request.target_output_path,
        )
    elif request.chunk.mode == "native_extend":
        if request.previous_output_path is None:
            raise RuntimeError(f"{request.resource_id} has no predecessor video")
        content = _extension_content(content, request.previous_output_path)
    content, _media_manifest = _bind_final_media_index_prompt(content, request)
    prompt = "\n".join(
        str(item.get("text") or "")
        for item in content
        if item.get("type") == "text"
    )
    from utils.config import SEEDANCE_MODEL
    from utils.prompt_budget import enforce_prompt_budget

    enforce_prompt_budget(
        prompt,
        provider="seedance",
        model=SEEDANCE_MODEL,
        purpose="video_generation",
    )
    return content, shot_meta, _generation_seed(request), _chunk_duration(request)


def _task_payload(
    request: ChunkExecutionRequest,
    *,
    model: str,
    provider_id: str,
    provider_version: str,
    project_id: str,
    run_id: str,
    duration: int,
    seed: int | None,
    generation_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unique_duration = _chunk_unique_duration(request.chunk)
    requested_frames = request.chunk.requested_frames
    overlap_duration = (
        duration * request.chunk.expected_overlap_frames / requested_frames
        if requested_frames
        else 0.0
    )
    padding_duration = (
        duration * request.chunk.expected_provider_padding_frames / requested_frames
        if requested_frames
        else 0.0
    )
    payload = {
        "shot_id": request.shot_id,
        "chunk_id": request.chunk.chunk_id,
        "resource_id": request.resource_id,
        "output_path": str(request.output_path),
        "model": model,
        "mode": request.chunk.mode,
        "execution_strategy": request.chunk.execution_strategy,
        "continuation_contract": (
            "next_primary_p01_first_last_frame_v1"
            if request.chunk.execution_strategy == "first_last_frame_bridge"
            else "tail_window_ordered_frames_v1"
            if request.chunk.mode == "native_extend"
            else "multi_image_storyboard_identity_v1"
            if request.chunk.execution_strategy == "multi_image"
            else None
        ),
        "tail_window_seconds": (
            SEEDANCE_2_CAPABILITIES.tail_reference_window_s
            if request.chunk.execution_strategy == "tail_video_extend"
            else None
        ),
        "ordered_frame_fractions": (
            list(SEEDANCE_2_CAPABILITIES.tail_reference_frame_fractions)
            if request.chunk.execution_strategy == "tail_video_extend"
            else None
        ),
        "bridge_target_beat_id": request.chunk.bridge_target_beat_id,
        "storyboard_narrative_guide": request.chunk.storyboard_narrative_guide,
        "storyboard_narrative_guide_kind": (
            request.chunk.storyboard_narrative_guide_kind
        ),
        "storyboard_narrative_guide_usage": (
            request.chunk.storyboard_narrative_guide_usage
        ),
        "storyboard_narrative_guide_cell_ids": list(
            request.chunk.storyboard_narrative_guide_cell_ids
        ),
        "storyboard_narrative_guide_sha256": (
            request.chunk.storyboard_narrative_guide_sha256
        ),
        "storyboard_narrative_guide_source_board_sha256": (
            request.chunk.storyboard_narrative_guide_source_board_sha256
        ),
        "storyboard_narrative_guide_source_board": (
            request.chunk.storyboard_narrative_guide_source_board
        ),
        "storyboard_narrative_guide_receipt": (
            request.chunk.storyboard_narrative_guide_receipt
        ),
        "character_performance_required": (
            request.chunk.character_performance_required
        ),
        "character_performance_guides": [
            guide.model_dump(mode="json")
            for guide in request.chunk.character_performance_guides
        ],
        "duration": duration,
        "provider_request_duration_s": duration,
        "effective_story_duration_s": round(unique_duration, 6),
        "reference_overlap_duration_s": round(overlap_duration, 6),
        "provider_minimum_padding_duration_s": round(padding_duration, 6),
        "seed": seed,
        "repair_attempt": request.repair_attempt,
        "privacy_fallback": PRIVACY_POLICY_REPAIR_VERSION,
        **(generation_parameters or {}),
    }
    upstream_fingerprint = str(request.input_fingerprint)
    if not re.fullmatch(r"[0-9a-f]{64}", upstream_fingerprint):
        upstream_fingerprint = hashlib.sha256(
            upstream_fingerprint.encode("utf-8")
        ).hexdigest()
    fingerprint = build_generation_fingerprint(
        prompt_text=(
            str(request.chunk.action_prompt or "").strip()
            or request.memory_context
        ),
        prompt_template_id=PHASE6_VIDEO_PROMPT_TEMPLATE_ID,
        prompt_template_version=PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION,
        provider_id=provider_id,
        provider_version=provider_version,
        model_id=model,
        model_version=model,
        parameters={
            key: value for key, value in payload.items() if key != "output_path"
        },
        input_artifact_hashes={"continuity_input": upstream_fingerprint},
    )
    cache_key = generation_cache_key(
        project_id=project_id,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    return {
        **payload,
        **fingerprint.task_metadata(),
        **cache_key.task_metadata(),
    }


def _provider_input_context(
    output_dir: Path,
    shot_id: str,
    chunk_id: str,
    *,
    storyboard_image: str | None = None,
    storyboard_narrative_guide: str | None = None,
    storyboard_narrative_guide_cell_ids: Sequence[str] = (),
    character_performance_guides: Sequence[Any] = (),
    bridge_target_storyboard_image: str | None = None,
) -> dict[str, Any]:
    shot_meta = output_dir / "shots" / shot_id / "SHOT_META.json"
    storyboard_frame = output_dir / "storyboard_images" / f"{shot_id}.png"
    group, group_board = _storyboard_group_for_shot(output_dir, shot_id)
    group_contract = output_dir / "STORYBOARD_GROUPS.json"
    from utils.privacy_visual_policy import (
        NO_REAL_PERSON_POLICY,
        is_no_real_person_enabled,
    )

    def optional_hash(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = output_dir / path
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    return {
        "shot_meta_sha256": hashlib.sha256(shot_meta.read_bytes()).hexdigest(),
        "storyboard_frame_sha256": (
            hashlib.sha256(storyboard_frame.read_bytes()).hexdigest()
            if storyboard_frame.is_file()
            else None
        ),
        "storyboard_group_id": str(group.get("group_id")) if group else None,
        "storyboard_groups_sha256": (
            hashlib.sha256(group_contract.read_bytes()).hexdigest()
            if group_contract.is_file()
            else None
        ),
        "storyboard_group_board_sha256": (
            hashlib.sha256(group_board.read_bytes()).hexdigest()
            if group_board is not None
            else None
        ),
        "chunk_id": chunk_id,
        "secondary_storyboard_image_sha256": optional_hash(storyboard_image),
        "storyboard_narrative_guide_sha256": optional_hash(
            storyboard_narrative_guide
        ),
        "storyboard_narrative_guide_cell_ids": list(
            storyboard_narrative_guide_cell_ids
        ),
        "character_performance_guides": [
            {
                "character_id": guide.character_id,
                "beat_id": guide.beat_id,
                "image_sha256": optional_hash(guide.image),
                "cell_ids": list(guide.cell_ids),
                "source_action_unit_ids": list(guide.source_action_unit_ids),
                "prop_ids": list(guide.prop_ids),
                "source_board_sha256": optional_hash(guide.source_board),
            }
            for guide in character_performance_guides
        ],
        "bridge_target_storyboard_image_sha256": optional_hash(
            bridge_target_storyboard_image
        ),
        "generation_seed_strategy": "scene_chunk_repair_v1",
        "continuation_contract": "tail_window_ordered_frames_v1",
        "tail_window_seconds": "2.0",
        "ordered_frame_fractions": "0.2,0.6,0.95",
        "visual_identity_policy": (
            NO_REAL_PERSON_POLICY if is_no_real_person_enabled() else None
        ),
    }


def _privacy_rejected_media_indices(
    content: Sequence[dict[str, Any]],
    error: BaseException,
) -> tuple[int, ...]:
    """Resolve provider-addressed image/video items from a privacy rejection."""
    message = str(error)
    privacy_markers = ("PrivacyInformation", "real person")
    if not any(marker.casefold() in message.casefold() for marker in privacy_markers):
        return ()
    indices = []
    for raw_index in re.findall(r"content\[(\d+)\]", message):
        index = int(raw_index)
        if (
            0 <= index < len(content)
            and content[index].get("type") in {"image_url", "video_url"}
            and index not in indices
        ):
            indices.append(index)
    return tuple(indices)


def _privacy_rejected_image_indices(
    content: Sequence[dict[str, Any]],
    error: BaseException,
) -> tuple[int, ...]:
    """Compatibility wrapper returning only provider-addressed image items."""
    return tuple(
        index
        for index in _privacy_rejected_media_indices(content, error)
        if content[index].get("type") == "image_url"
    )


def _without_content_indices(
    content: Sequence[dict[str, Any]],
    rejected_indices: Sequence[int],
) -> list[dict[str, Any]]:
    rejected = set(rejected_indices)
    return [dict(item) for index, item in enumerate(content) if index not in rejected]


def _provider_ready_content(
    content: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip local continuity annotations without changing provider indices."""
    return [
        {key: value for key, value in item.items() if not str(key).startswith("_")}
        for item in content
    ]


def _privacy_policy_repair_budget(content: Sequence[dict[str, Any]]) -> int:
    """Bound progressive privacy repairs by the original addressable media set."""
    media_items = sum(
        item.get("type") in {"image_url", "video_url"} for item in content
    )
    return min(media_items, MAX_PRIVACY_POLICY_REPAIRS)


def _privacy_repair_content(
    content: Sequence[dict[str, Any]],
    rejected_indices: Sequence[int],
    *,
    request: ChunkExecutionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove only rejected optional media while preserving continuity structure."""
    rejected = tuple(dict.fromkeys(int(index) for index in rejected_indices))
    removed = [content[index] for index in rejected if 0 <= index < len(content)]
    if not removed or len(removed) != len(rejected):
        raise RuntimeError("privacy fallback could not resolve every rejected media item")

    required_endpoints = [
        index
        for index in rejected
        if str(content[index].get("_continuity_role") or "").startswith("required_")
    ]
    if required_endpoints:
        raise RequiredContinuityEndpointPrivacyError(
            "provider privacy review rejected required first/last bridge endpoint(s): "
            + ", ".join(f"content[{index}]" for index in required_endpoints)
        )
    required_references = [
        index
        for index in rejected
        if content[index].get("_mandatory_reference") is True
    ]
    if required_references:
        raise RuntimeError(
            "provider privacy review rejected mandatory cinematic/identity/"
            "narrative-guide reference(s); automatic media removal is forbidden: "
            + ", ".join(f"content[{index}]" for index in required_references)
        )

    corrected = _without_content_indices(content, rejected)
    native_extend = request.chunk.mode == "native_extend"
    if native_extend:
        remaining_continuity = [
            item for item in corrected if item.get("_continuity_role")
        ]
        if not remaining_continuity:
            raise RuntimeError(
                "privacy fallback would remove every tail continuity anchor"
            )

    removed_types = sorted({str(item.get("type") or "unknown") for item in removed})
    removed_video = any(item.get("type") == "video_url" for item in removed)
    contract = (
        _PRIVACY_FRAME_ONLY_CONTINUITY_CONTRACT
        if native_extend and removed_video
        else _PRIVACY_VIDEO_ONLY_CONTINUITY_CONTRACT
        if native_extend
        else ""
    )
    if contract:
        text_item = next(
            (item for item in corrected if item.get("type") == "text"),
            None,
        )
        if text_item is None:
            corrected.insert(0, {"type": "text", "text": contract})
        elif contract not in str(text_item.get("text") or ""):
            text_item["text"] = contract + str(text_item.get("text") or "")

    return corrected, {
        "reason_code": "InputMediaSensitiveContentDetected.PrivacyInformation",
        "policy": (
            "drop_rejected_video_keep_ordered_tail_frames_v1"
            if removed_video
            else "drop_provider_rejected_optional_images_v2"
        ),
        "removed_content_indices": list(rejected),
        "removed_media_types": removed_types,
        "remaining_continuity_anchors": sum(
            bool(item.get("_continuity_role")) for item in corrected
        ),
    }


def _copyright_policy_violation_kind(error: BaseException) -> str | None:
    """Classify only the two copyright-policy failures with safe remediations."""
    message = str(error).casefold()
    if "policyviolation" not in message:
        return None
    if "outputaudiosensitivecontentdetected" in message:
        return "output_audio"
    if "inputvideosensitivecontentdetected" in message:
        return "input_video"
    return None


def _copyright_rejected_video_indices(
    content: Sequence[dict[str, Any]],
    error: BaseException,
) -> tuple[int, ...]:
    """Resolve provider-addressed video items without dropping unrelated inputs."""
    if _copyright_policy_violation_kind(error) != "input_video":
        return ()
    indices: list[int] = []
    for raw_index in re.findall(r"content\[(\d+)\]", str(error)):
        index = int(raw_index)
        if (
            0 <= index < len(content)
            and content[index].get("type") == "video_url"
            and index not in indices
        ):
            indices.append(index)
    return tuple(indices)


def _sanitize_music_language(value: str) -> str:
    """Remove soundtrack requests while preserving the authored body action."""
    substitutions = (
        (
            r"(?i)\b(?:soundtrack|music|song|melody|lyrics?|bpm|rhythm)\b",
            "visual motion timing",
        ),
        (r"配乐|音乐|歌曲|旋律|歌词", "动作时序提示"),
        (r"节奏", "动作时序"),
    )
    sanitized = value
    for pattern, replacement in substitutions:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def _copyright_repair_content(
    content: Sequence[dict[str, Any]],
    *,
    audio_safe: bool,
    rejected_video_indices: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Build a bounded compliant retry without reusing a rejected video item."""
    corrected = _without_content_indices(content, rejected_video_indices)
    if rejected_video_indices:
        removed = [
            content[index]
            for index in rejected_video_indices
            if 0 <= index < len(content)
        ]
        if not removed or any(item.get("type") != "video_url" for item in removed):
            raise RuntimeError("copyright fallback did not resolve a rejected video item")
    frame_only = bool(rejected_video_indices) or any(
        _FRAME_ONLY_CONTINUITY_CONTRACT in str(item.get("text") or "")
        for item in content
        if item.get("type") == "text"
    )
    text_contract = (
        (_FRAME_ONLY_CONTINUITY_CONTRACT if frame_only else "")
        + (_COPYRIGHT_SAFE_AUDIO_CONTRACT if audio_safe else "")
    )
    for item in corrected:
        if item.get("type") != "text":
            continue
        original = str(item.get("text") or "")
        original = original.replace(_COPYRIGHT_SAFE_AUDIO_CONTRACT, "").replace(
            _FRAME_ONLY_CONTINUITY_CONTRACT,
            "",
        )
        item["text"] = text_contract + (
            _sanitize_music_language(original) if audio_safe else original
        )
    if text_contract and not any(item.get("type") == "text" for item in corrected):
        corrected.insert(0, {"type": "text", "text": text_contract})
    return corrected


def _copyright_repair_seed(
    seed: int | None,
    repairs: Sequence[dict[str, Any]],
) -> int | None:
    """Deterministically vary repaired output while keeping reruns reproducible."""
    if seed is None or not repairs:
        return seed
    material = json.dumps(
        {"seed": seed, "repairs": list(repairs)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) % 2_147_483_647


def _bridge_handle_contract(
    output_dir: Path,
    request: ChunkExecutionRequest,
) -> tuple[float, float, int]:
    """Load the exact additive-handle budget for one post-primary bridge."""
    plan_path = output_dir / "CONTINUITY_PLAN.json"
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    target_shot_id = str(request.chunk.bridge_target_shot_id or "").strip()
    bridge = next(
        (
            item
            for item in document.get("bridges", [])
            if isinstance(item, dict)
            and item.get("source_shot_id") == request.shot_id
            and item.get("target_shot_id") == target_shot_id
        ),
        None,
    )
    if bridge is None:
        raise RuntimeError(
            f"cannot resolve handle contract for {request.shot_id}__{target_shot_id}"
        )
    source_handle_s = float(bridge.get("source_handle_s") or 0.0)
    target_handle_s = float(bridge.get("target_handle_s") or 0.0)
    duration_s = float(bridge.get("visible_duration_s") or request.chunk.target_duration_s)
    if source_handle_s <= 0 or target_handle_s <= 0:
        raise RuntimeError("privacy bridge fallback requires two positive handle durations")
    if not math.isclose(source_handle_s + target_handle_s, duration_s, abs_tol=1e-6):
        raise RuntimeError(
            "privacy bridge fallback requires source_handle + target_handle "
            "to equal visible bridge duration"
        )
    timeline_fps = int(document.get("timeline_fps") or 24)
    if timeline_fps <= 0:
        raise RuntimeError("privacy bridge fallback requires a positive timeline fps")
    return source_handle_s, target_handle_s, timeline_fps


def _render_privacy_safe_handle_bridge(
    source_video: Path,
    target_video: Path,
    output_path: Path,
    *,
    source_handle_s: float,
    target_handle_s: float,
    timeline_fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Emit an honest local bridge by preserving the two budgeted handles.

    Phase 8 removes these same handles from the adjacent primaries. Re-emitting
    them here therefore preserves timing and every real source frame without
    inventing faces, duplicating story time, or pretending a hard cut is a
    provider-generated transition.
    """
    for label, path in (("source", source_video), ("target", target_video)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"privacy fallback {label} video missing: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.privacy_handle_fallback{output_path.suffix}"
    )
    total_frames = round((source_handle_s + target_handle_s) * timeline_fps)
    common = (
        f"fps={timeline_fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    )
    filter_graph = (
        f"[0:v]trim=duration={source_handle_s:.6f},setpts=PTS-STARTPTS,{common}[src];"
        f"[1:v]trim=duration={target_handle_s:.6f},setpts=PTS-STARTPTS,{common}[dst];"
        f"[src][dst]concat=n=2:v=1:a=0,trim=end_frame={total_frames},"
        "setpts=PTS-STARTPTS[outv]"
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-sseof",
            f"-{source_handle_s:.6f}",
            "-i",
            str(source_video),
            "-i",
            str(target_video),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-frames:v",
            str(total_frames),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            "cannot render privacy-safe handle bridge: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    observed = probe_continuity_frames(temporary, timeline_fps)
    if int(observed["frames"]) != total_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "privacy-safe handle bridge frame mismatch: "
            f"expected {total_frames}, got {observed['frames']}"
        )
    os.replace(temporary, output_path)
    return {
        "policy": "local_boundary_handle_passthrough_v1",
        "reason": "provider privacy review rejected an inseparable FLF2V endpoint",
        "provider_generation": False,
        "transition_effect": "none_hard_cut_preserved",
        "source_handle_s": source_handle_s,
        "target_handle_s": target_handle_s,
        "timeline_fps": timeline_fps,
        "frames": total_frames,
        "phase8_contract": "replace_the_same_boundary_handles",
    }


def _direct_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
    *,
    media_profile: str = "480p",
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import seedance_client
    from utils.config import SEEDANCE_MODEL, get_api_key_or_raise
    from utils.video_validation import is_valid_video

    api_key = get_api_key_or_raise("ARK_AGENT")
    model = SEEDANCE_MODEL
    resolution = seedance_client.resolution_for_media_profile(media_profile, model)
    provider_policy = ProviderExecutionPolicy.from_environment("seedance")
    artifact_store = ArtifactManifestStore.from_run_directory(
        output_dir,
        required=False,
    )
    run_id = artifact_store.run_id if artifact_store else str(output_dir.resolve())
    project_id = artifact_store.project_id if artifact_store else "local"
    fallback_workers = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    capacity = provider_policy.capacity(fallback_workers)
    slots = SlotTable()
    leases = CrossProcessSlotTable(default_capacity_lease_path())

    def execute(request: ChunkExecutionRequest) -> ChunkExecutionResult:
        seed = _generation_seed(request)
        duration = _chunk_duration(request)
        ratio, _width, _height = _video_geometry(
            _read_shot_meta(output_dir, request.shot_id)
        )
        provider_ratio = (
            "adaptive"
            if request.chunk.execution_strategy == "first_last_frame_bridge"
            else ratio
        )
        generation_parameters = {
            "ratio": provider_ratio,
            "resolution": resolution,
            "return_last_frame": True,
            "seedance_prompt_contract": (
                "first_last_frame_v1"
                if request.chunk.execution_strategy == "first_last_frame_bridge"
                else "video1_tail_extend_v1"
                if request.chunk.execution_strategy == "tail_video_extend"
                else SEEDANCE_ALL_MODAL_PROMPT_CONTRACT
            ),
        }
        repairs: list[dict[str, Any]] = []
        privacy_repairs: list[dict[str, Any]] = []
        privacy_resubmission_attempt = 0
        privacy_repair_budget: int | None = None
        content_override: list[dict[str, Any]] | None = None
        execution = None
        with slots.reserve("seedance", "video", request.resource_id, capacity=capacity):
            with leases.reserve(
                "seedance",
                "video",
                request.resource_id,
                capacity=capacity,
            ):
                # Terminal provider failures require a new durable task. Privacy and
                # copyright repairs keep independent bounded budgets below.
                for _provider_attempt in range(
                    MAX_COPYRIGHT_POLICY_REPAIRS + MAX_PRIVACY_POLICY_REPAIRS + 1
                ):
                    try:
                        if content_override is None:
                            content, _shot_meta, _seed, _duration = _provider_content(
                                output_dir,
                                request,
                            )
                        else:
                            content = [dict(item) for item in content_override]
                    except Exception as exc:
                        raise ProviderPreparationError(
                            "cannot prepare provider content for "
                            f"{request.resource_id}: {exc}"
                        ) from exc
                    if privacy_repair_budget is None:
                        privacy_repair_budget = _privacy_policy_repair_budget(content)
                    assert privacy_repair_budget is not None

                    media_index_manifest = _media_index_manifest(content)
                    provider_prompt = next(
                        (
                            str(item.get("text") or "")
                            for item in content
                            if item.get("type") == "text"
                        ),
                        "",
                    )
                    payload = _task_payload(
                        request,
                        model=model,
                        provider_id="seedance",
                        provider_version="ark-agent-plan-v3",
                        project_id=project_id,
                        run_id=run_id,
                        duration=duration,
                        seed=seed,
                        generation_parameters={
                            **generation_parameters,
                            "media_index_manifest": media_index_manifest,
                            "provider_prompt_sha256": hashlib.sha256(
                                provider_prompt.encode("utf-8")
                            ).hexdigest(),
                        },
                    )

                    attempt_seed = _copyright_repair_seed(seed, repairs)
                    policy_attempt = len(repairs)
                    submitted_content: list[dict[str, Any]] = []
                    attempt_payload = dict(payload)
                    if repairs:
                        attempt_payload.update(
                            {
                                "seed": attempt_seed,
                                "copyright_policy_repair_version": (
                                    COPYRIGHT_POLICY_REPAIR_VERSION
                                ),
                                "copyright_policy_repair_attempt": policy_attempt,
                                "copyright_policy_repairs": [
                                    dict(item) for item in repairs
                                ],
                            }
                        )
                    if privacy_resubmission_attempt:
                        attempt_payload.update(
                            {
                                "privacy_policy_repair_version": (
                                    PRIVACY_POLICY_REPAIR_VERSION
                                ),
                                "privacy_policy_repair_attempt": (
                                    privacy_resubmission_attempt
                                ),
                                "privacy_policy_repairs": [
                                    dict(item) for item in privacy_repairs
                                ],
                            }
                        )
                    attempt_suffixes: list[str] = []
                    if policy_attempt:
                        attempt_suffixes.append(f"CP{policy_attempt:02d}")
                    if privacy_resubmission_attempt:
                        attempt_suffixes.append(
                            f"PP{privacy_resubmission_attempt:02d}"
                        )
                    attempt_resource_id = request.resource_id
                    if attempt_suffixes:
                        attempt_resource_id += "_" + "_".join(attempt_suffixes)

                    def submit(
                        current_content: Sequence[dict[str, Any]] = content,
                        current_seed: int | None = attempt_seed,
                        submitted: list[dict[str, Any]] = submitted_content,
                        current_privacy_budget: int = privacy_repair_budget,
                    ) -> str:
                        def submit_selected(
                            selected: Sequence[dict[str, Any]],
                        ) -> str:
                            submitted[:] = [dict(item) for item in selected]
                            return provider_policy.execute_rate_limited(
                                partial(
                                    seedance_client.submit_content,
                                    _provider_ready_content(selected),
                                    api_key=api_key,
                                    model=model,
                                    duration=duration,
                                    ratio=provider_ratio,
                                    resolution=resolution,
                                    seed=current_seed,
                                    return_last_frame=True,
                                    timeout=(
                                        provider_policy.submit_timeout_seconds
                                    ),
                                )
                            )

                        selected = [dict(item) for item in current_content]
                        while True:
                            try:
                                return submit_selected(selected)
                            except Exception as submit_exc:
                                rejected_indices = _privacy_rejected_media_indices(
                                    selected,
                                    submit_exc,
                                )
                                if (
                                    not rejected_indices
                                    or len(privacy_repairs)
                                    >= current_privacy_budget
                                ):
                                    raise
                                try:
                                    corrected_content, repair = _privacy_repair_content(
                                        selected,
                                        rejected_indices,
                                        request=request,
                                    )
                                except RequiredContinuityEndpointPrivacyError as endpoint_exc:
                                    privacy_repairs.append({
                                        "attempt": len(privacy_repairs) + 1,
                                        "reason_code": (
                                            "InputImageSensitiveContentDetected."
                                            "PrivacyInformation"
                                        ),
                                        "policy": (
                                            "required_endpoints_inseparable_"
                                            "local_handle_fallback_v1"
                                        ),
                                        "removed_content_indices": [],
                                        "rejected_content_indices": list(
                                            rejected_indices
                                        ),
                                    })
                                    raise endpoint_exc from submit_exc
                                repair = {
                                    "attempt": len(privacy_repairs) + 1,
                                    **repair,
                                }
                                privacy_repairs.append(repair)
                                selected = corrected_content
                                print(
                                    "  🛡 Seedance 隐私合规降级："
                                    f"{repair['policy']}，移除 "
                                    + ", ".join(
                                        f"content[{index}]"
                                        for index in rejected_indices
                                    )
                                    + f" ({len(privacy_repairs)}/"
                                    f"{current_privacy_budget})",
                                    flush=True,
                                )

                    try:
                        succeeded = task_store.find_succeeded(
                            run_id=run_id,
                            task_type="video.generate",
                            resource_id=attempt_resource_id,
                            payload=attempt_payload,
                            provider_id="seedance",
                        )
                        failed = (
                            None
                            if succeeded is not None
                            else task_store.find_failed(
                                run_id=run_id,
                                task_type="video.generate",
                                resource_id=attempt_resource_id,
                                payload=attempt_payload,
                                provider_id="seedance",
                            )
                        )
                        if failed is not None:
                            failed_error = RuntimeError(failed.error_message or "")
                            if (
                                _copyright_policy_violation_kind(failed_error)
                                or _privacy_rejected_media_indices(
                                    content,
                                    failed_error,
                                )
                            ):
                                raise failed_error
                        execution = execute_seedance_video_task(
                            task_store,
                            run_id=run_id,
                            resource_id=attempt_resource_id,
                            payload=attempt_payload,
                            provider_endpoint=seedance_client.BASE_URL,
                            output_path=request.output_path,
                            submit=submit,
                            poll=provider_policy.bind_poll(
                                seedance_client.poll,
                                interval_seconds=15,
                                api_key=api_key,
                            ),
                            download=seedance_client.download,
                            validate_output=is_valid_video,
                            artifact_store=artifact_store,
                        )
                        break
                    except Exception as exc:
                        submitted = submitted_content or content
                        rejected_indices = _privacy_rejected_media_indices(
                            submitted,
                            exc,
                        )
                        if rejected_indices:
                            if len(privacy_repairs) >= privacy_repair_budget:
                                raise
                            try:
                                corrected_content, repair = _privacy_repair_content(
                                    submitted,
                                    rejected_indices,
                                    request=request,
                                )
                            except RequiredContinuityEndpointPrivacyError as endpoint_exc:
                                privacy_repairs.append(
                                    {
                                        "attempt": len(privacy_repairs) + 1,
                                        "reason_code": (
                                            "InputImageSensitiveContentDetected."
                                            "PrivacyInformation"
                                        ),
                                        "policy": (
                                            "required_endpoints_inseparable_"
                                            "local_handle_fallback_v1"
                                        ),
                                        "removed_content_indices": [],
                                        "rejected_content_indices": list(
                                            rejected_indices
                                        ),
                                    }
                                )
                                exc = endpoint_exc
                            else:
                                repair = {
                                    "attempt": len(privacy_repairs) + 1,
                                    **repair,
                                }
                                privacy_repairs.append(repair)
                                content_override = corrected_content
                                privacy_resubmission_attempt += 1
                                print(
                                    "  🛡 Seedance 异步隐私审核合规重生成："
                                    f"{repair['policy']}，移除 "
                                    + ", ".join(
                                        f"content[{index}]"
                                        for index in rejected_indices
                                    )
                                    + f" ({len(privacy_repairs)}/"
                                    f"{privacy_repair_budget})",
                                    flush=True,
                                )
                                continue
                        if isinstance(
                            exc,
                            RequiredContinuityEndpointPrivacyError,
                        ):
                            if (
                                request.previous_output_path is None
                                or request.target_output_path is None
                            ):
                                raise RuntimeError(
                                    "privacy-safe bridge fallback requires completed "
                                    "source and target primary videos"
                                ) from exc
                            source_handle_s, target_handle_s, timeline_fps = (
                                _bridge_handle_contract(output_dir, request)
                            )
                            fallback = _render_privacy_safe_handle_bridge(
                                request.previous_output_path,
                                request.target_output_path,
                                request.output_path,
                                source_handle_s=source_handle_s,
                                target_handle_s=target_handle_s,
                                timeline_fps=timeline_fps,
                                width=_width,
                                height=_height,
                            )
                            print(
                                "  🛡 FLF2V 必需端点被隐私审核拒绝：改用本地等长把手"
                                "透传，Phase 8 仍按原合同替换边界把手",
                                flush=True,
                            )
                            return ChunkExecutionResult(
                                output_path=request.output_path,
                                provider_task_id=None,
                                privacy_policy_repairs=tuple(
                                    dict(item) for item in privacy_repairs
                                ),
                                provider_fallback=fallback,
                            )
                        violation_kind = _copyright_policy_violation_kind(exc)
                        if (
                            violation_kind is None
                            or len(repairs) >= MAX_COPYRIGHT_POLICY_REPAIRS
                        ):
                            raise
                        if violation_kind == "output_audio":
                            if any(
                                item.get("reason_code")
                                == "OutputAudioSensitiveContentDetected.PolicyViolation"
                                for item in repairs
                            ):
                                raise
                            content_override = _copyright_repair_content(
                                submitted,
                                audio_safe=True,
                            )
                            repair = {
                                "attempt": len(repairs) + 1,
                                "reason_code": (
                                    "OutputAudioSensitiveContentDetected.PolicyViolation"
                                ),
                                "policy": "original_ambient_no_music_v1",
                                "removed_content_indices": [],
                            }
                        else:
                            rejected_video_indices = (
                                _copyright_rejected_video_indices(submitted, exc)
                            )
                            if not rejected_video_indices:
                                raise
                            content_override = _copyright_repair_content(
                                submitted,
                                audio_safe=True,
                                rejected_video_indices=rejected_video_indices,
                            )
                            repair = {
                                "attempt": len(repairs) + 1,
                                "reason_code": (
                                    "InputVideoSensitiveContentDetected.PolicyViolation"
                                ),
                                "policy": "drop_rejected_video_keep_ordered_frames_v1",
                                "removed_content_indices": list(
                                    rejected_video_indices
                                ),
                                "retained_reference_images": sum(
                                    item.get("type") == "image_url"
                                    for item in content_override
                                ),
                            }
                        repairs.append(repair)
                        print(
                            "  🛡 Seedance 版权审核合规重生成："
                            f"{repair['reason_code']} → {repair['policy']} "
                            f"({len(repairs)}/{MAX_COPYRIGHT_POLICY_REPAIRS})",
                            flush=True,
                        )
                if execution is None:
                    raise RuntimeError(
                        f"{request.resource_id} provider policy retry exited without result"
                    )
        return ChunkExecutionResult(
            output_path=Path(execution.output_path),
            provider_task_id=execution.provider_job_id,
            copyright_policy_repairs=tuple(dict(item) for item in repairs),
            privacy_policy_repairs=tuple(
                dict(item) for item in privacy_repairs
            ),
        )

    return execute


def _bridge_seedance_executor(
    output_dir: Path,
    task_store: GenerationTaskStore,
) -> Callable[[ChunkExecutionRequest], ChunkExecutionResult]:
    from clients import local_video_client
    from utils.video_validation import is_valid_video

    provider_policy = ProviderExecutionPolicy.from_environment("bridge")
    artifact_store = ArtifactManifestStore.from_run_directory(
        output_dir,
        required=False,
    )
    run_id = artifact_store.run_id if artifact_store else str(output_dir.resolve())
    project_id = artifact_store.project_id if artifact_store else "local"
    capacity = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    slots = SlotTable()

    def execute(request: ChunkExecutionRequest) -> ChunkExecutionResult:
        seed = _generation_seed(request)
        duration = _chunk_duration(request)
        model = "seedance"
        ratio, width, height = _video_geometry(
            _read_shot_meta(output_dir, request.shot_id)
        )
        content, _shot_meta, _seed, _duration = _provider_content(
            output_dir,
            request,
        )
        prompt = next(
            (str(item.get("text")) for item in content if item.get("type") == "text"),
            "",
        )
        payload = _task_payload(
            request,
            model=model,
            provider_id="bridge",
            provider_version="bridge-api-v1",
            project_id=project_id,
            run_id=run_id,
            duration=duration,
            seed=seed,
            generation_parameters={
                "ratio": ratio,
                "width": width,
                "height": height,
                "media_index_manifest": _media_index_manifest(content),
                "provider_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            },
        )

        def generate(**runtime_kwargs: Any) -> str | dict[str, Any]:
            return local_video_client.generate_video(
                prompt=prompt,
                output_path=str(request.output_path),
                seed=seed if seed is not None else -1,
                duration=duration,
                width=width,
                height=height,
                fps=24,
                content=_provider_ready_content(content),
                batch_id=output_dir.name,
                model=model,
                submit_timeout=int(provider_policy.submit_timeout_seconds),
                status_timeout=provider_policy.status_timeout_seconds,
                poll_deadline=provider_policy.poll_deadline_seconds,
                **runtime_kwargs,
            )

        with slots.reserve("bridge", "video", request.resource_id, capacity=capacity):
            execution = execute_bridge_video_task(
                task_store,
                run_id=run_id,
                resource_id=request.resource_id,
                payload=payload,
                provider_endpoint=local_video_client.get_api_url(),
                output_path=request.output_path,
                generate=generate,
                validate_output=is_valid_video,
                artifact_store=artifact_store,
            )
        return ChunkExecutionResult(
            output_path=Path(execution.output_path),
            provider_task_id=execution.provider_job_id,
        )

    return execute


def materialize_continuity_shot(
    chunk_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Concatenate accepted internal boundaries into one editorial-shot clip."""
    if not chunk_paths:
        raise ValueError("cannot materialize a shot without chunks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    command = ["ffmpeg", "-y"]
    for path in chunk_paths:
        command.extend(["-i", str(path)])
    streams = "".join(f"[{index}:v:0]" for index in range(len(chunk_paths)))
    command.extend(
        [
            "-filter_complex",
            f"{streams}concat=n={len(chunk_paths)}:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    completed = subprocess.run(
        validate_subprocess_args(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot materialize continuity shot {output_path}: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    os.replace(temporary, output_path)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_phase8_seam_decisions(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "CONTINUITY_SEAM_DECISIONS.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != SEAM_DECISIONS_KIND:
        raise ValueError(f"unsupported Phase 8 continuity decisions in {path}")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"{path} must contain a decisions object")
    for boundary_id, decision in decisions.items():
        if not isinstance(boundary_id, str) or not isinstance(decision, dict):
            raise ValueError(f"{path} contains an invalid boundary decision")
        if decision.get("action") != "hard_trim":
            raise ValueError(f"{boundary_id} has unsupported Phase 8 action")
        if int(decision.get("trim_frames") or 0) <= 0:
            raise ValueError(f"{boundary_id} requires a positive trim_frames value")
    return decisions


def _render_phase8_hard_trim(
    following: Path,
    output_path: Path,
    *,
    trim_frames: int,
    timeline_fps: int,
) -> None:
    """Materialize a frame-addressed, interpolation-free Phase 8 cut."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.phase8{output_path.suffix}")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(following),
            "-vf",
            (
                f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS,"
                f"fps={timeline_fps},format=yuv420p"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            "cannot apply Phase 8 continuity trim: "
            f"{detail[-1] if detail else 'unknown ffmpeg error'}"
        )
    os.replace(temporary, output_path)


def _continuity_bridge_preparer(
    output_dir: Path,
    *,
    planned_overlap_seconds: float = 0.0,
    timeline_fps: int = 24,
) -> Callable[[Path, Path, str], dict[str, Any]] | None:
    """Build the optional local trim/interpolation step used before seam scoring."""
    phase8_decisions = _load_phase8_seam_decisions(output_dir)
    mode = os.environ.get(CONTINUITY_BRIDGE_ENV, "off").strip().lower()
    if mode not in CONTINUITY_BRIDGE_MODES:
        expected = ", ".join(sorted(CONTINUITY_BRIDGE_MODES))
        raise ValueError(f"{CONTINUITY_BRIDGE_ENV} must be one of {expected}, got {mode!r}")
    if mode == "off" and not phase8_decisions:
        return None
    raw_candidates = os.environ.get("HONCUT_CONTINUITY_BRIDGE_FRAMES", "4,6,8")
    try:
        candidate_frames = tuple(int(item.strip()) for item in raw_candidates.split(","))
    except ValueError as exc:
        raise ValueError(
            "HONCUT_CONTINUITY_BRIDGE_FRAMES must be comma-separated integers"
        ) from exc
    candidate_frames = tuple(sorted(set(candidate_frames)))
    if not candidate_frames or any(frame < 4 or frame > 24 for frame in candidate_frames):
        raise ValueError("HONCUT_CONTINUITY_BRIDGE_FRAMES values must be between 4 and 24")

    def prepare(previous: Path, following: Path, boundary_id: str) -> dict[str, Any]:
        from quality.continuity_bridge import detect_replayed_prefix, repair_continuity_boundary

        boundary_dir = output_dir / "continuity_bridges" / boundary_id
        output_path = boundary_dir / "effective_following.mp4"
        receipt_path = boundary_dir / "CONTINUITY_BRIDGE.json"
        phase8_decision = phase8_decisions.get(boundary_id)
        input_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "previous_sha256": _file_sha256(previous),
                    "following_sha256": _file_sha256(following),
                    "candidate_frames": candidate_frames,
                    "planned_overlap_seconds": planned_overlap_seconds,
                    "phase8_decision": phase8_decision,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if receipt_path.is_file():
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            cached_output = Path(str(cached.get("output_path") or following))
            if (
                cached.get("status") != "fallback"
                and cached.get("input_fingerprint") == input_fingerprint
                and cached_output.is_file()
                and cached_output.stat().st_size > 0
            ):
                return cached

        boundary_dir.mkdir(parents=True, exist_ok=True)
        if phase8_decision is not None:
            expected = phase8_decision.get("source_fingerprint") or {}
            actual = {
                "previous_sha256": _file_sha256(previous),
                "following_sha256": _file_sha256(following),
            }
            if expected != actual:
                raise RuntimeError(
                    f"{boundary_id} Phase 8 decision is stale; rerun seam adjudication"
                )
            trim_frames = int(phase8_decision["trim_frames"])
            _render_phase8_hard_trim(
                following,
                output_path,
                trim_frames=trim_frames,
                timeline_fps=timeline_fps,
            )
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "adjudicated_trim",
                "reason": phase8_decision["reason"],
                "input_fingerprint": input_fingerprint,
                "output_path": str(output_path),
                "selected_bridge_frames": None,
                "ghost_safe_fallback": True,
                "phase8_decision": phase8_decision,
                "overlap": {
                    "detected": True,
                    "overlap_seconds": trim_frames / timeline_fps,
                    "overlap_frames": trim_frames,
                    "source": "phase8_temporal_adjudication",
                    "reason": phase8_decision["reason"],
                },
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return receipt
        if mode == "off":
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "skipped",
                "reason": "continuity bridge is disabled for this unadjudicated boundary",
                "input_fingerprint": input_fingerprint,
                "output_path": str(following),
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return receipt
        try:
            overlap = detect_replayed_prefix(previous, following)
            if not overlap["detected"]:
                if planned_overlap_seconds <= 0:
                    receipt = {
                        "kind": "honcut.continuity_bridge.v1",
                        "status": "skipped",
                        "reason": overlap["reason"],
                        "overlap": overlap,
                        "input_fingerprint": input_fingerprint,
                        "output_path": str(following),
                    }
                else:
                    trial = repair_continuity_boundary(
                        previous,
                        following,
                        output_path,
                        work_dir=boundary_dir / "work",
                        overlap_seconds=planned_overlap_seconds,
                        candidate_frames=candidate_frames,
                    )
                    baseline = float(trial["baseline_boundary_frame_mae"])
                    selected = float(trial["selected_boundary_frame_mae"])
                    trimmed = float(trial["trimmed_boundary_frame_mae"])
                    if (
                        trial.get("improved")
                        and selected <= baseline * 0.5
                        and trimmed < baseline
                    ):
                        # The interpolation result is evidence that the planned
                        # cut point is useful, not necessarily a safe output.
                        # With an uncertain trajectory, synthesized in-betweens
                        # can create double contours around moving subjects.
                        # Prefer the decoded hard trim: one small discontinuity
                        # is less objectionable than a multi-frame ghost trail.
                        trimmed_path = boundary_dir / "work" / "trimmed_hard_cut.mp4"
                        temporary = output_path.with_name(
                            f"{output_path.stem}.ghost_safe{output_path.suffix}"
                        )
                        shutil.copy2(trimmed_path, temporary)
                        os.replace(temporary, output_path)
                        receipt = trial
                        receipt["status"] = "trimmed"
                        receipt["output_path"] = str(output_path)
                        receipt["selected_bridge_frames"] = None
                        receipt["selected_boundary_frame_mae"] = trimmed
                        receipt["ghost_safe_fallback"] = True
                        receipt["detector_overlap"] = overlap
                        receipt["overlap"] = {
                            "detected": True,
                            "overlap_seconds": planned_overlap_seconds,
                            "source": "phase4_planned_budget",
                            "reason": (
                                "planned overlap trial reduced boundary-frame error "
                                "by at least fifty percent; emitted a hard trim to "
                                "avoid interpolation ghosting"
                            ),
                        }
                    else:
                        receipt = {
                            "kind": "honcut.continuity_bridge.v1",
                            "status": "skipped",
                            "reason": (
                                "planned overlap trial did not reduce boundary-frame "
                                "error by at least fifty percent"
                            ),
                            "overlap": overlap,
                            "planned_overlap_trial": trial,
                            "input_fingerprint": input_fingerprint,
                            "output_path": str(following),
                        }
            else:
                receipt = repair_continuity_boundary(
                    previous,
                    following,
                    output_path,
                    work_dir=boundary_dir / "work",
                    overlap_seconds=float(overlap["overlap_seconds"]),
                    candidate_frames=candidate_frames,
                )
                receipt["input_fingerprint"] = input_fingerprint
                receipt["overlap"] = overlap
            receipt["input_fingerprint"] = input_fingerprint
        except Exception as exc:
            receipt = {
                "kind": "honcut.continuity_bridge.v1",
                "status": "fallback",
                "reason": f"local continuity bridge failed: {exc}",
                "input_fingerprint": input_fingerprint,
                "output_path": str(following),
            }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt

    return prepare


def execute_phase6_auto_continuity(
    output_dir: str | Path,
    plan: ContinuityPlan,
    calibration: SeamCalibration | None,
    *,
    media_profile: str = "480p",
    _test_executor_factory: Callable[
        [Path, GenerationTaskStore],
        Callable[[ChunkExecutionRequest], ChunkExecutionResult],
    ]
    | None = None,
) -> dict[str, Any]:
    """Run continuity groups through exactly one configured provider."""
    root = Path(output_dir)
    test_only = _test_executor_factory is not None
    if test_only:
        provider = "offline_fixture"
        executor_factory = _test_executor_factory
        executor_factory_kwargs: dict[str, Any] = {}
    else:
        provider = os.environ.get("VIDEO_PROVIDER", "seedance").strip().lower()
        if provider == "seedance":
            executor_factory = _direct_seedance_executor
            parameters = inspect.signature(executor_factory).parameters
            executor_factory_kwargs = (
                {"media_profile": media_profile}
                if "media_profile" in parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                else {}
            )
        elif provider == "bridge":
            executor_factory = _bridge_seedance_executor
            executor_factory_kwargs = {}
        else:
            raise RuntimeError(
                "continuity auto requires VIDEO_PROVIDER=seedance or bridge; "
                f"{provider!r} has no verified native video-extension contract"
            )

    _validate_seedance_continuity_plan(plan)

    requires_media_upload = any(
        chunk.mode == "native_extend"
        or chunk.execution_strategy == "first_last_frame_bridge"
        or bool(chunk.storyboard_image)
        for shot in plan.shots
        for chunk in shot.chunks
    )
    if requires_media_upload and not test_only:
        from clients.tos_uploader import is_media_upload_configured

        if not is_media_upload_configured():
            raise RuntimeError(
                "continuity auto preflight requires TOS_ACCESS_KEY, "
                "TOS_SECRET_KEY, and TOS_BUCKET before any paid provider "
                "submission containing image or video media"
            )

    planned_overlap_seconds = max(
        (
            chunk.expected_overlap_frames / plan.timeline_fps
            for shot in plan.shots
            for chunk in shot.chunks
        ),
        default=0.0,
    )
    prepare_seam = _continuity_bridge_preparer(
        root,
        planned_overlap_seconds=planned_overlap_seconds,
        timeline_fps=plan.timeline_fps,
    )
    missing_replay_overlap = [
        chunk.chunk_id
        for shot in plan.shots
        for chunk in shot.chunks
        if (
            chunk.mode == "native_extend"
            and chunk.execution_strategy == "legacy"
            and chunk.expected_overlap_frames <= 0
        )
    ]
    if prepare_seam is not None and missing_replay_overlap:
        raise RuntimeError(
            "continuity bridge requires an overlap-budgeted CONTINUITY_PLAN.json; "
            "rerun Phase 4 with HONCUT_CONTINUITY_BRIDGE=auto; missing overlap on "
            + ", ".join(missing_replay_overlap[:6])
        )
    task_store = GenerationTaskStore(root / "runtime.db")
    execute_chunk = executor_factory(root, task_store, **executor_factory_kwargs)

    max_repairs = int(os.environ.get("HONCUT_CONTINUITY_MAX_REPAIRS", "1"))
    if not 0 <= max_repairs <= 3:
        raise ValueError("HONCUT_CONTINUITY_MAX_REPAIRS must be between 0 and 3")
    workers = max(1, int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1")))
    report = execute_continuity_plan(
        plan,
        root,
        execute_chunk=execute_chunk,
        prepare_seam=prepare_seam,
        materialize_shot=materialize_continuity_shot,
        probe_frames=probe_continuity_frames,
        finalize_shot=finalize_continuity_shot,
        normalize_chunk=normalize_provider_minimum_padding,
        chunk_context=lambda shot, chunk: _provider_input_context(
            root,
            shot.shot_id,
            chunk.chunk_id,
            storyboard_image=chunk.storyboard_image,
            storyboard_narrative_guide=chunk.storyboard_narrative_guide,
            storyboard_narrative_guide_cell_ids=(
                chunk.storyboard_narrative_guide_cell_ids
            ),
            character_performance_guides=chunk.character_performance_guides,
            bridge_target_storyboard_image=chunk.bridge_target_storyboard_image,
        ),
        seam_calibration=(
            calibration.model_dump(mode="json") if calibration is not None else None
        ),
        max_seam_repairs=max_repairs,
        max_workers=workers,
    )
    report.update(
        {
            "provider": provider,
            "mode": "continuity_auto",
            "test_only": test_only,
            "continuity_bridge": os.environ.get(CONTINUITY_BRIDGE_ENV, "off").strip().lower(),
            "calibration_fingerprint": (
                calibration.dataset_fingerprint if calibration is not None else None
            ),
        }
    )
    return report
