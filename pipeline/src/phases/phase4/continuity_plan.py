"""Build the backward-compatible Phase 4 continuity plan artifact."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from phases.phase1.storyboard_beats import (
    secondary_contract_declared,
    secondary_storyboard_contract_errors,
)
from schemas.continuity import (
    ContinuityAnchors,
    ContinuityPlan,
    ContinuityShot,
    GenerationChunk,
    PrimaryShotBridge,
)
from utils.video_capabilities import SEEDANCE_2_CAPABILITIES, capabilities_for

DEFAULT_PROVIDER_CHUNK_LIMIT_S = SEEDANCE_2_CAPABILITIES.max_shot_duration_s
DEFAULT_TIMELINE_FPS = 24
DEFAULT_CONTINUITY_GROUP_MAX_SHOTS = 3


def _storyboard_group_contract(
    storyboard: Mapping[str, Any],
    plan: ContinuityPlan,
) -> dict[str, Any]:
    """Build the narrative map that sits above per-shot continuity chunks."""
    storyboard_shots = {
        _shot_id(shot, index): shot
        for index, shot in enumerate(storyboard.get("shots", []), 1)
        if isinstance(shot, Mapping)
    }
    grouped: dict[str, list[ContinuityShot]] = {}
    for planned in plan.shots:
        grouped.setdefault(planned.continuity_group_id, []).append(planned)

    groups: list[dict[str, Any]] = []
    shot_to_group: dict[str, str] = {}
    def compact(value: Any, limit: int = 320) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    for group_id, planned_shots in grouped.items():
        beats = []
        for position, planned in enumerate(planned_shots, 1):
            source = storyboard_shots.get(planned.shot_id, {})
            actions = source.get("generation_actions") or []
            if isinstance(actions, str):
                actions = [actions]
            beat = {
                "position": position,
                "shot_id": planned.shot_id,
                "mode": planned.chunks[0].mode,
                "storyboard_image": f"storyboard_images/{planned.shot_id}.png",
                "storyboard_board": source.get("storyboard_board"),
                "storyboard_beats": source.get("storyboard_beats") or [],
                "lens_mm": source.get("lens_mm"),
                "camera_motion_contract": source.get("camera_motion_contract") or {},
                "start_state": compact(
                    source.get("start_state")
                    or source.get("prev_shot_context")
                    or source.get("what")
                ),
                "generation_actions": [str(action) for action in actions if str(action).strip()],
                "body_action_choreography": source.get("body_action_choreography") or [],
                "body_action_contract": source.get("body_action_contract") or {},
                "end_state": compact(source.get("end_state") or source.get("what")),
                "causal_link": compact(source.get("causal_link") or planned.continuity_reason),
                "continuity_reason": planned.continuity_reason,
            }
            beats.append(beat)
            shot_to_group[planned.shot_id] = group_id
        groups.append({
            "group_id": group_id,
            "shot_ids": [shot.shot_id for shot in planned_shots],
            "entry_shot_id": planned_shots[0].shot_id,
            "extension_shot_ids": [shot.shot_id for shot in planned_shots[1:]],
            "storyboard_board": f"storyboard_groups/{group_id}.jpg",
            "beats": beats,
        })
    for index, group in enumerate(groups):
        previous = groups[index - 1] if index > 0 else None
        following = groups[index + 1] if index + 1 < len(groups) else None
        previous_last = previous["beats"][-1] if previous and previous.get("beats") else None
        current_first = group["beats"][0] if group.get("beats") else None
        group["previous_group_id"] = previous.get("group_id") if previous else None
        group["next_group_id"] = following.get("group_id") if following else None
        group["handoff_from_previous"] = (
            {
                "previous_shot_id": previous_last.get("shot_id"),
                "previous_end_state": previous_last.get("end_state", ""),
                "entry_shot_id": current_first.get("shot_id") if current_first else None,
                "entry_start_state": current_first.get("start_state", "") if current_first else "",
                "edit": "fresh_editorial_cut",
            }
            if previous_last
            else None
        )
    return {
        "kind": "honcut.storyboard_groups.v1",
        "version": 1,
        "director_storyboard": storyboard.get("director_storyboard"),
        "groups": groups,
        "shot_to_group": shot_to_group,
    }


def _render_group_board(output_dir: Path, group: Mapping[str, Any]) -> str | None:
    """Render one chronological contact sheet from the group's shot boards."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    beats = [beat for beat in group.get("beats", []) if isinstance(beat, Mapping)]
    # In the two-level pipeline one continuity group normally equals one Sxx.
    # Reuse its Phase 2 multi-panel board directly so P01/P02 remain visible.
    if len(beats) == 1 and beats[0].get("storyboard_board"):
        authored = output_dir / str(beats[0]["storyboard_board"])
        if authored.is_file():
            return str(authored.relative_to(output_dir))

    panels = []
    for beat in beats:
        source = output_dir / str(beat.get("storyboard_image") or "")
        if not source.is_file():
            continue
        with Image.open(source) as image:
            panel = ImageOps.fit(image.convert("RGB"), (640, 360), method=Image.Resampling.LANCZOS)
        panels.append(panel)
    if not panels:
        return None
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    gutter = 12
    board = Image.new(
        "RGB",
        (columns * 640 + (columns + 1) * gutter, rows * 360 + (rows + 1) * gutter),
        color=(8, 10, 14),
    )
    for index, panel in enumerate(panels):
        column, row = index % columns, index // columns
        board.paste(panel, (gutter + column * (640 + gutter), gutter + row * (360 + gutter)))
    relative = Path(str(group["storyboard_board"]))
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, format="JPEG", quality=88, optimize=True)
    return str(relative)


def write_storyboard_groups(
    output_dir: Path,
    storyboard: Mapping[str, Any],
    plan: ContinuityPlan,
) -> dict[str, Any]:
    """Persist group-level plot contracts and chronological storyboard boards."""
    output_dir = Path(output_dir)
    contract = _storyboard_group_contract(storyboard, plan)
    for group in contract["groups"]:
        rendered = _render_group_board(output_dir, group)
        group["storyboard_board"] = rendered
    destination = output_dir / "STORYBOARD_GROUPS.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return contract


def _chunk_frame_budgets(
    target_frames: int,
    limit_frames: int,
    overlap_frames: int,
    *,
    fps: int,
    initial_extension: bool = False,
) -> list[tuple[int, int, int]]:
    """Return requested, replay-overlap, and unique frame budgets per chunk."""
    if target_frames <= limit_frames and not initial_extension:
        return [(target_frames, 0, target_frames)]
    if overlap_frames >= limit_frames:
        raise ValueError("continuation overlap must be shorter than the provider chunk limit")

    usable_extension_frames = limit_frames - overlap_frames
    if initial_extension:
        chunk_count = max(1, math.ceil(target_frames / usable_extension_frames))
        overlap_count = chunk_count
    else:
        chunk_count = 1 + max(
            1,
            math.ceil(max(0, target_frames - limit_frames) / usable_extension_frames),
        )
        overlap_count = chunk_count - 1
    requested_total = target_frames + overlap_count * overlap_frames

    # Preserve whole-second provider requests whenever all inputs permit it.
    if requested_total % fps == 0 and limit_frames % fps == 0:
        total_units = requested_total // fps
        base, remainder = divmod(total_units, chunk_count)
        requested = [(base + (index < remainder)) * fps for index in range(chunk_count)]
    else:
        base, remainder = divmod(requested_total, chunk_count)
        requested = [base + (index < remainder) for index in range(chunk_count)]

    if max(requested) > limit_frames:
        raise ValueError("cannot fit overlap-aware chunk budget within provider limit")
    budgets = []
    for index, requested_frames in enumerate(requested):
        reserved_overlap = overlap_frames if initial_extension or index > 0 else 0
        unique_frames = requested_frames - reserved_overlap
        if unique_frames <= 0:
            raise ValueError("continuation overlap leaves no unique frames in a chunk")
        budgets.append((requested_frames, reserved_overlap, unique_frames))
    return budgets


def _shot_id(shot: Mapping[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip()
    if re.fullmatch(r"[Ss]?\d+", text):
        return f"S{int(text.lstrip('Ss')):02d}"
    return text or f"S{index:02d}"


def _target_duration(shot: Mapping[str, Any]) -> float:
    raw = shot.get("duration")
    if raw is None:
        raw = shot.get("suggested_duration")
    if raw is None:
        raw = 5.0
    return float(raw)


def _boundary_before(
    shot: Mapping[str, Any],
    previous_shot: Mapping[str, Any] | None,
    index: int,
    *,
    allow_unverified_explicit: bool = False,
) -> tuple[str, str]:
    from quality.shot_continuity import classify_boundary

    return classify_boundary(
        previous_shot,
        shot,
        index=index,
        allow_unverified_explicit=allow_unverified_explicit,
    )


def _anchors(shot: Mapping[str, Any], scene_contract: Mapping[str, Any]) -> ContinuityAnchors:
    who = (
        shot.get("character_ids")
        or shot.get("who")
        or shot.get("characters")
        or []
    )
    if not isinstance(who, list):
        who = [who] if who else []
    camera_motion = str(
        shot.get("camera_motion") or shot.get("camera_movement") or shot.get("camera") or ""
    )
    tracking_prompt = str(
        shot.get("continuity_subject")
        or shot.get("tracking_prompt")
        or shot.get("subject_description")
        or ", ".join(str(value) for value in who if value)
        or ""
    ).strip()
    return ContinuityAnchors(
        characters=[str(value) for value in who if value],
        scene=str(shot.get("where") or scene_contract.get("scene_description") or ""),
        screen_direction=str(shot.get("screen_direction") or shot.get("camera_axis") or ""),
        camera_motion=camera_motion,
        style=str(scene_contract.get("style_anchor") or scene_contract.get("style_suffix") or ""),
        tracking_prompt=tracking_prompt[:240],
    )


def _authored_storyboard_beats(shot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        beat
        for beat in (shot.get("storyboard_beats") or [])
        if isinstance(beat, Mapping)
    ]


def _secondary_strategy(beat: Mapping[str, Any], sequence: int) -> str:
    """Normalize the secondary strategy while keeping legacy artifacts readable."""
    value = str(
        beat.get("execution_strategy") or beat.get("generation_mode") or ""
    ).strip().lower()
    aliases = {
        "multi_image": "multi_image",
        "tail_video_extend": "tail_video_extend",
        "first_last_frame_bridge": "first_last_frame_bridge",
    }
    if value in aliases:
        return aliases[value]
    return "legacy"


def _beat_action_prompt(beat: Mapping[str, Any]) -> str:
    """Render story action plus stable edge handles used by bridge replacement."""
    clauses: list[str] = []
    incoming = float(beat.get("incoming_bridge_handle_s") or 0)
    outgoing = float(beat.get("outgoing_bridge_handle_s") or 0)
    if incoming > 0:
        clauses.append(
            f"开头前{incoming:g}秒保持 start_state 的同一构图、姿态与运动趋势，"
            "只允许自然微动，不执行本格新的剧情动作；随后再开始本格动作"
        )
    clauses.append(str(beat.get("action") or ""))
    if outgoing > 0:
        clauses.append(
            f"必须在结尾前{outgoing:g}秒完成本格剧情动作；最后{outgoing:g}秒"
            "稳定保持 end_state，只允许自然微动，不新增动作、台词或剧情结果"
        )
    return "。".join(value for value in clauses if value)


def _validate_secondary_strategy_sequence(
    shot_id: str,
    strategies: list[str],
) -> None:
    """Enforce content-first ordering without assigning semantics by position."""
    valid = (
        1 <= len(strategies) <= 3
        and strategies[0] == "multi_image"
        and all(strategy == "tail_video_extend" for strategy in strategies[1:])
    )
    if not valid:
        raise ValueError(
            f"{shot_id} has invalid secondary execution sequence {strategies}; "
            "expected P01 multi-image followed by zero, one, or two capacity "
            "extensions; cross-primary bridges are separate post-generation tasks"
        )


def _beat_unique_frames(
    beats: list[Mapping[str, Any]],
    target_frames: int,
) -> list[int]:
    """Allocate the editorial frame budget across authored Pxx panels."""
    weights = [max(0.001, float(beat.get("duration_s") or 1.0)) for beat in beats]
    total = sum(weights)
    endpoints = [round(sum(weights[:index]) / total * target_frames) for index in range(1, len(weights) + 1)]
    result = []
    previous = 0
    for endpoint in endpoints:
        result.append(endpoint - previous)
        previous = endpoint
    if any(value <= 0 for value in result):
        raise ValueError("storyboard beats exceed the editorial shot frame budget")
    return result


def build_continuity_plan(
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
    timeline_fps: int = DEFAULT_TIMELINE_FPS,
    continuation_overlap_s: float = 0.0,
    continuity_group_max_shots: int = DEFAULT_CONTINUITY_GROUP_MAX_SHOTS,
) -> ContinuityPlan:
    """Split long editorial shots into a linear sequence of provider-sized chunks."""
    if provider_chunk_limit_s <= 0:
        raise ValueError("provider_chunk_limit_s must be positive")
    if timeline_fps <= 0:
        raise ValueError("timeline_fps must be positive")
    if continuation_overlap_s < 0:
        raise ValueError("continuation_overlap_s must not be negative")
    if continuity_group_max_shots < 1:
        raise ValueError("continuity_group_max_shots must be positive")

    storyboard_contract = dict(storyboard)
    strict_secondary_contract = secondary_contract_declared(storyboard_contract)
    if strict_secondary_contract:
        contract_shots = [
            shot for shot in storyboard.get("shots", []) if isinstance(shot, Mapping)
        ]
        strict_errors = [
            error
            for shot_index in range(len(contract_shots))
            for error in secondary_storyboard_contract_errors(
                storyboard_contract,
                shot_index,
            )
        ]
        if strict_errors:
            summary = "; ".join(error["message"] for error in strict_errors[:8])
            raise ValueError(f"invalid secondary storyboard contract: {summary}")

    limit_frames = round(provider_chunk_limit_s * timeline_fps)
    overlap_frames = round(continuation_overlap_s * timeline_fps)

    scene_shots = (scene_consistency or {}).get("shots", {})
    planned_shots: list[ContinuityShot] = []
    cumulative_duration = 0.0
    previous_endpoint_frames = 0
    previous_storyboard_shot: Mapping[str, Any] | None = None
    previous_planned_shot: ContinuityShot | None = None
    group_number = 0
    group_shot_count = 0
    preserve_one_take = str(
        storyboard.get("continuity_mode") or ""
    ).strip().lower() in {"one_take", "single_take", "oner"}
    for index, shot in enumerate(storyboard.get("shots", []), 1):
        shot_id = _shot_id(shot, index)
        capability_profile = capabilities_for({**storyboard_contract, **dict(shot)})
        authored_beats = _authored_storyboard_beats(shot)
        boundary_before, continuity_reason = _boundary_before(
            shot,
            previous_storyboard_shot,
            index,
            allow_unverified_explicit=not strict_secondary_contract,
        )
        if preserve_one_take and index > 1:
            boundary_before = "continuous"
            continuity_reason = (
                "one-take contract overrides authored cut/location-change metadata; "
                "the preceding primary shot must bridge into this P01 composition"
            )
        requested_extension = (
            boundary_before == "continuous"
            and previous_planned_shot is not None
        )
        secondary_strategies = [
            _secondary_strategy(beat, sequence)
            for sequence, beat in enumerate(authored_beats, 1)
        ]
        uses_secondary_contract = bool(authored_beats) and all(
            strategy != "legacy" for strategy in secondary_strategies
        )
        if uses_secondary_contract:
            _validate_secondary_strategy_sequence(shot_id, secondary_strategies)
        capped_group = (
            requested_extension
            and not preserve_one_take
            and group_shot_count >= continuity_group_max_shots
        )
        # Modern Pxx shots are independently completed first. Their continuous
        # boundary is bridged later from the actual source tail to target head,
        # so primary shots must not form a cross-shot provider dependency group.
        logical_extension = (
            requested_extension and not capped_group and not uses_secondary_contract
        )
        initial_extension = logical_extension
        if capped_group:
            boundary_before = "cut"
            continuity_reason = (
                f"start a fresh generation group after {continuity_group_max_shots} "
                "continuous editorial shots to prevent accumulated visual and narrative drift"
            )
        if not logical_extension:
            group_number += 1
            group_shot_count = 0
        continuity_group_id = f"CG{group_number:03d}"
        group_shot_count += 1
        target_duration = _target_duration(shot)
        cumulative_duration += target_duration
        endpoint_frames = round(cumulative_duration * timeline_fps)
        target_frames = endpoint_frames - previous_endpoint_frames
        previous_endpoint_frames = endpoint_frames
        chunks: list[GenerationChunk] = []
        previous: str | None = None
        if initial_extension:
            previous = previous_planned_shot.chunks[-1].chunk_id
        if authored_beats:
            unique_budgets = _beat_unique_frames(authored_beats, target_frames)
            for sequence, (beat, unique_frames) in enumerate(
                zip(authored_beats, unique_budgets, strict=True),
                1,
            ):
                strategy = secondary_strategies[sequence - 1]
                reserved_overlap = (
                    overlap_frames
                    if previous is not None
                    and strategy == "legacy"
                    else 0
                )
                minimum_request_s, _maximum_request_s = (
                    capability_profile.request_duration_bounds(strategy)
                )
                minimum_request_frames = math.ceil(
                    minimum_request_s * timeline_fps - 1e-9
                )
                requested_frames = max(
                    unique_frames + reserved_overlap,
                    minimum_request_frames,
                )
                declared_request_s = beat.get("provider_request_duration_s")
                if strict_secondary_contract:
                    expected_request_s = (
                        capability_profile.request_duration_for_effective_story(
                            unique_frames / timeline_fps,
                            strategy,
                        )
                    )
                    if (
                        isinstance(declared_request_s, bool)
                        or not isinstance(declared_request_s, (int, float))
                        or not math.isclose(
                            float(declared_request_s),
                            expected_request_s,
                            abs_tol=1e-6,
                        )
                    ):
                        raise ValueError(
                            f"{shot_id} storyboard beat {sequence} declares stale "
                            "Provider request duration"
                        )
                    requested_frames = round(expected_request_s * timeline_fps)
                provider_padding_frames = (
                    requested_frames - unique_frames - reserved_overlap
                )
                if requested_frames > limit_frames:
                    raise ValueError(
                        f"{shot_id} storyboard beat {sequence} needs "
                        f"{requested_frames / timeline_fps:g}s including overlap, "
                        f"above provider limit {provider_chunk_limit_s:g}s"
                    )
                chunk_id = f"{shot_id}_C{sequence:02d}"
                capability_profile.validate_chunk_durations(
                    requested_frames / timeline_fps,
                    unique_frames / timeline_fps,
                    strategy,
                    resource_id=chunk_id,
                )
                chunks.append(GenerationChunk(
                    chunk_id=chunk_id,
                    sequence=sequence,
                    target_duration_s=round(requested_frames / timeline_fps, 6),
                    requested_frames=requested_frames,
                    expected_overlap_frames=reserved_overlap,
                    expected_provider_padding_frames=provider_padding_frames,
                    expected_unique_frames=unique_frames,
                    mode="native_extend" if previous is not None else "fresh",
                    depends_on=previous,
                    execution_strategy=strategy,
                    storyboard_beat_id=str(
                        beat.get("beat_id") or f"{shot_id}_P{sequence:02d}"
                    ),
                    storyboard_image=str(
                        beat.get("video_first_frame")
                        or beat.get("storyboard_image")
                        or ""
                    ) or None,
                    storyboard_image_kind=str(
                        beat.get("video_first_frame_kind")
                        or (
                            "legacy_storyboard_image"
                            if beat.get("storyboard_image")
                            else ""
                        )
                    ) or None,
                    bridge_target_shot_id=(
                        str(beat.get("bridge_target_shot_id") or "") or None
                    ),
                    bridge_target_beat_id=(
                        str(beat.get("bridge_target_beat_id") or "") or None
                    ),
                    bridge_target_storyboard_image=(
                        str(beat.get("bridge_target_storyboard_image") or "") or None
                    ),
                    action_prompt=_beat_action_prompt(beat),
                    start_state=str(beat.get("start_state") or ""),
                    end_state=str(beat.get("end_state") or ""),
                ))
                previous = chunk_id
        else:
            budgets = _chunk_frame_budgets(
                target_frames,
                limit_frames,
                overlap_frames,
                fps=timeline_fps,
                initial_extension=initial_extension,
            )
            for sequence, (requested_frames, reserved_overlap, unique_frames) in enumerate(budgets, 1):
                chunk_id = f"{shot_id}_C{sequence:02d}"
                chunks.append(
                    GenerationChunk(
                        chunk_id=chunk_id,
                        sequence=sequence,
                        target_duration_s=round(requested_frames / timeline_fps, 6),
                        requested_frames=requested_frames,
                        expected_overlap_frames=reserved_overlap,
                        expected_unique_frames=unique_frames,
                        mode="native_extend" if previous is not None else "fresh",
                        depends_on=previous,
                    )
                )
                previous = chunk_id

        planned_shots.append(
            ContinuityShot(
                shot_id=shot_id,
                target_duration_s=target_duration,
                target_frames=target_frames,
                boundary_before=boundary_before,
                continuity_group_id=continuity_group_id,
                extends_from_shot_id=(
                    previous_planned_shot.shot_id if initial_extension else None
                ),
                extends_from_chunk_id=(
                    previous_planned_shot.chunks[-1].chunk_id if initial_extension else None
                ),
                continuity_reason=str(
                    shot.get("continuity_reason") or continuity_reason
                ).strip(),
                anchors=_anchors(shot, scene_shots.get(shot_id, {})),
                chunks=chunks,
            )
        )
        previous_storyboard_shot = shot
        previous_planned_shot = planned_shots[-1]
    bridge_specs = [
        item
        for item in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(item, Mapping)
    ]
    planned_by_id = {shot.shot_id: shot for shot in planned_shots}
    planned_bridges: list[PrimaryShotBridge] = []
    for spec in bridge_specs:
        source_shot_id = str(spec.get("source_shot_id") or "")
        target_shot_id = str(spec.get("target_shot_id") or "")
        source = planned_by_id.get(source_shot_id)
        target = planned_by_id.get(target_shot_id)
        if source is None or target is None:
            raise ValueError(
                f"bridge {source_shot_id}__{target_shot_id} references unknown primary shot"
            )
        duration_s = float(spec.get("duration_s") or 0)
        profile = capabilities_for(storyboard_contract)
        profile.validate_chunk_durations(
            duration_s,
            duration_s,
            "first_last_frame_bridge",
            resource_id=str(spec.get("bridge_id") or f"{source_shot_id}__{target_shot_id}"),
        )
        storyboard_transition = spec.get("storyboard_transition")
        if not isinstance(storyboard_transition, Mapping):
            storyboard_transition = {}
        planned_bridges.append(
            PrimaryShotBridge(
                bridge_id=str(
                    spec.get("bridge_id") or f"{source_shot_id}__{target_shot_id}"
                ),
                source_shot_id=source_shot_id,
                target_shot_id=target_shot_id,
                target_duration_s=duration_s,
                requested_frames=round(duration_s * timeline_fps),
                generation_duration_s=float(
                    spec.get("generation_duration_s") or duration_s
                ),
                visible_duration_s=float(
                    spec.get("visible_duration_s") or duration_s
                ),
                source_handle_s=float(spec.get("source_handle_s") or 0),
                target_handle_s=float(spec.get("target_handle_s") or 0),
                timeline_insertion_policy=str(
                    spec.get("timeline_insertion_policy") or "append"
                ),
                continuity_reason=str(spec.get("boundary_reason") or ""),
                action_prompt=str(spec.get("action_prompt") or ""),
                start_state=str(spec.get("start_state") or ""),
                end_state=str(spec.get("end_state") or ""),
                storyboard_transition_image=(
                    str(storyboard_transition.get("image") or "") or None
                ),
                storyboard_transition_prompt=(
                    str(storyboard_transition.get("prompt") or "") or None
                ),
                storyboard_transition_usage=(
                    storyboard_transition.get("usage") or None
                ),
            )
        )
    return ContinuityPlan(
        provider_chunk_limit_s=provider_chunk_limit_s,
        timeline_fps=timeline_fps,
        shots=planned_shots,
        bridges=planned_bridges,
        material_budget=dict(storyboard.get("material_budget") or {}),
    )


def write_continuity_plan(
    output_path: Path,
    storyboard: Mapping[str, Any],
    scene_consistency: Mapping[str, Any] | None = None,
    *,
    provider_chunk_limit_s: float = DEFAULT_PROVIDER_CHUNK_LIMIT_S,
    timeline_fps: int = DEFAULT_TIMELINE_FPS,
    continuation_overlap_s: float = 0.0,
    continuity_group_max_shots: int = DEFAULT_CONTINUITY_GROUP_MAX_SHOTS,
) -> ContinuityPlan:
    """Persist a JSON-safe continuity plan through an atomic replace."""
    plan = build_continuity_plan(
        storyboard,
        scene_consistency,
        provider_chunk_limit_s=provider_chunk_limit_s,
        timeline_fps=timeline_fps,
        continuation_overlap_s=continuation_overlap_s,
        continuity_group_max_shots=continuity_group_max_shots,
    )
    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return plan
