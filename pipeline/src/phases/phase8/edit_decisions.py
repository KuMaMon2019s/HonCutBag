"""HonCut edit decisions builder and executor.

Builds structured edit_decisions from shot videos, then executes
frame-accurate assembly with normalization, transitions, and audio handling.

Key capabilities:
1. Frame-accurate trimming (blackdetect + static frame removal)
2. Segment normalization (resolution/fps/codec/pix_fmt/sar)
3. Silent audio track injection (anullsrc for clips without audio)
4. Per-cut speed adjustment (setpts + atempo chain)
5. xfade transition chain (dissolve/fadeblack/cut)
6. Audio-safe boundaries (equal-power crossfades and soft fades on visual cuts)
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# ─── Probing ─────────────────────────────────────────────────────────────────


def probe_video(video_path: str) -> dict:
    """Probe a video file for duration, resolution, fps, audio presence."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        format_duration = float(data.get("format", {}).get("duration", 0))
        duration = format_duration
        video_duration = audio_duration = None
        width, height, fps = 1920, 1080, 30.0
        has_audio = has_video = False

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                has_video = True
                try:
                    video_duration = float(stream.get("duration"))
                except (TypeError, ValueError):
                    pass
                width = int(stream.get("width", 1920))
                height = int(stream.get("height", 1080))
                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / max(float(den), 1)
                else:
                    fps = float(fps_str)
            elif stream.get("codec_type") == "audio":
                has_audio = True
                try:
                    audio_duration = float(stream.get("duration"))
                except (TypeError, ValueError):
                    pass

        # The visual timeline is authoritative for video artifacts. Using the
        # container's longest-stream duration lets padded audio conceal a
        # truncated video stream from both assembly offsets and duration gates.
        if has_video and video_duration is not None:
            duration = video_duration

        return {
            "duration": round(duration, 3),
            "video_duration": None if video_duration is None else round(video_duration, 3),
            "audio_duration": None if audio_duration is None else round(audio_duration, 3),
            "format_duration": round(format_duration, 3),
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": round(duration * fps),
            "has_audio": has_audio,
            "has_video": has_video,
        }
    except Exception as e:
        print(f"  [probe] Error: {e}")
        return {
            "duration": 0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "has_audio": False,
            "has_video": False,
        }


# ─── Black Frame Detection ───────────────────────────────────────────────────


def detect_black_frames(
    video_path: str,
    threshold: float = 0.1,
    *,
    trim_static_edges: bool = True,
) -> dict:
    """Detect leading/trailing black or static frames to trim.

    Returns: {"trim_start": seconds, "trim_end": seconds}
    """
    info = probe_video(video_path)
    duration = info["duration"]
    if duration <= 0:
        return {"trim_start": 0, "trim_end": 0}

    trim_start = trim_end = 0.0

    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"blackdetect=d=0.2:pix_th={threshold}",
            "-an",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        blacks = re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", result.stderr)

        for start_s, end_s in blacks:
            sf, ef = float(start_s), float(end_s)
            if sf < 0.3:
                trim_start = max(trim_start, ef)
            if ef > duration - 0.3:
                trim_end = max(trim_end, duration - sf)
    except Exception:
        pass

    # A continuity-finalized clip has already spent its boundary handles.
    # Preserve its exact frame budget unless blackdetect found real black.
    if trim_static_edges:
        trim_start = max(trim_start, 0.1)
        trim_end = max(trim_end, 0.1)

    # Don't trim more than 20%
    max_trim = duration * 0.2
    trim_start = min(trim_start, max_trim)
    trim_end = min(trim_end, max_trim)

    return {"trim_start": round(trim_start, 2), "trim_end": round(trim_end, 2)}


def _replace_boundary_handles(
    source_cut: dict[str, Any],
    target_cut: dict[str, Any],
    bridge: dict[str, Any],
) -> dict[str, Any]:
    """Replace stable primary edge handles with one visible bridge interval."""
    source_handle = float(bridge.get("source_handle_s") or 0)
    target_handle = float(bridge.get("target_handle_s") or 0)
    visible = float(
        bridge.get("visible_duration_s")
        or bridge.get("target_duration_s")
        or 0
    )
    if source_handle <= 0 or target_handle <= 0:
        raise RuntimeError("bridge replacement requires positive handles on both sides")
    if abs(source_handle + target_handle - visible) > 1e-6:
        raise RuntimeError(
            "bridge replacement handles do not match its visible duration: "
            f"{source_handle:g}+{target_handle:g}!={visible:g}"
        )

    source_before = float(source_cut["out_seconds"])
    target_before = float(target_cut["in_seconds"])
    source_boundary = float(source_cut["original_duration"]) - source_handle
    target_boundary = target_handle
    source_cut["out_seconds"] = round(
        min(source_before, source_boundary), 6
    )
    target_cut["in_seconds"] = round(max(target_before, target_boundary), 6)
    if float(source_cut["in_seconds"]) >= float(source_cut["out_seconds"]):
        raise RuntimeError(
            f"bridge source handle consumes all of {source_cut['shot_id']}"
        )
    if float(target_cut["in_seconds"]) >= float(target_cut["out_seconds"]):
        raise RuntimeError(
            f"bridge target handle consumes all of {target_cut['shot_id']}"
        )
    source_cut["trimmed"] = True
    target_cut["trimmed"] = True
    receipt = {
        "boundary_id": str(
            bridge.get("bridge_id")
            or f"{source_cut['shot_id']}__{target_cut['shot_id']}"
        ),
        "source_shot_id": str(source_cut["shot_id"]),
        "target_shot_id": str(target_cut["shot_id"]),
        "source_handle_s": source_handle,
        "target_handle_s": target_handle,
        "visible_bridge_duration_s": visible,
        "source_out_before_s": source_before,
        "source_out_after_s": float(source_cut["out_seconds"]),
        "target_in_before_s": target_before,
        "target_in_after_s": float(target_cut["in_seconds"]),
        "timeline_insertion_policy": "replace_boundary_handles",
    }
    source_cut.setdefault("bridge_handle_replacements", []).append(receipt)
    target_cut.setdefault("bridge_handle_replacements", []).append(receipt)
    return receipt


# ─── Build Edit Decisions ────────────────────────────────────────────────────


def build_edit_decisions(
    shots_dir: str,
    target_width: int = 1920,
    target_height: int = 1080,
    transition_decisions: list | None = None,
    quality_report: dict | None = None,
    shot_order: list[str] | None = None,
    target_duration: float | None = None,
    transition_duration: float = 0.5,
    fit_mode: str = "cover",
    continuity_plan: dict | None = None,
    allow_unresolved_reshoots: bool = False,
) -> dict:
    """Build reviewed edit decisions from shot videos.

    For each shot: probe → merge automatic boundary trims → build cut entry.
    Shots still marked ``reshoot`` are rejected instead of silently assembled.
    """
    shots_dir = Path(shots_dir)
    available = {
        directory.name: directory
        for directory in shots_dir.iterdir()
        if directory.is_dir() and directory.name.startswith("S")
    }
    if shot_order:
        missing_ordered = [name for name in shot_order if name not in available]
        unexpected_ordered = sorted(set(available).difference(shot_order))
        duplicate_ordered = sorted(
            {name for name in shot_order if shot_order.count(name) > 1}
        )
        if missing_ordered or unexpected_ordered or duplicate_ordered:
            raise ValueError(
                "reviewed shot order does not exactly resolve to shot directories: "
                f"missing={missing_ordered}, unexpected={unexpected_ordered}, "
                f"duplicates={duplicate_ordered}"
            )
    shot_dirs = (
        [available[name] for name in shot_order or []]
        if shot_order
        else [available[name] for name in sorted(available)]
    )
    quality_shots = (quality_report or {}).get("shots", {})

    cuts = []
    unresolved_reshoots: list[str] = []
    phase8_duration_trims: list[dict[str, Any]] = []
    for shot_dir in shot_dirs:
        video_path = shot_dir / "output.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"{shot_dir.name} output.mp4 is missing")

        info = probe_video(str(video_path))
        if info["duration"] <= 0:
            raise ValueError(f"{shot_dir.name} output.mp4 has non-positive duration")

        quality = quality_shots.get(shot_dir.name, {})
        if quality.get("action") == "reshoot":
            if not allow_unresolved_reshoots:
                raise ValueError(
                    f"{shot_dir.name} still requires reshoot: "
                    f"{'; '.join(quality.get('reasons', []))}"
                )
            unresolved_reshoots.append(shot_dir.name)
        timing_path = shot_dir / "CONTINUITY_TIMING.json"
        continuity_timing = None
        if timing_path.is_file():
            try:
                candidate = json.loads(timing_path.read_text(encoding="utf-8"))
                if candidate.get("internal_seams_finalized") is True:
                    continuity_timing = candidate
            except (OSError, json.JSONDecodeError):
                continuity_timing = None
        trims = (
            detect_black_frames(str(video_path), trim_static_edges=False)
            if continuity_timing is not None
            else detect_black_frames(str(video_path))
        )
        in_s = max(float(trims["trim_start"]), float(quality.get("trim_start_s", 0.0)))
        quality_end = float(quality.get("trim_end_s", info["duration"]) or info["duration"])
        out_s = min(info["duration"] - float(trims["trim_end"]), quality_end)
        duration_trim = None
        if continuity_timing is not None:
            timing_fps = int(continuity_timing.get("timeline_fps") or 24)
            target_frames = int(continuity_timing.get("target_frames") or 0)
            final_frames = int(continuity_timing.get("final_frames") or 0)
            available_frames = round((out_s - in_s) * timing_fps)
            if (
                target_frames > 0
                and final_frames > target_frames
                and available_frames > target_frames
            ):
                phase8_out_s = in_s + (target_frames / timing_fps)
                if phase8_out_s < out_s:
                    out_s = phase8_out_s
                    duration_trim = {
                        "method": "phase8_per_shot_excess_trim",
                        "timeline_fps": timing_fps,
                        "target_frames": target_frames,
                        "discarded_excess_frames": available_frames - target_frames,
                    }
                    phase8_duration_trims.append(
                        {"shot_id": shot_dir.name, **duration_trim}
                    )
        if out_s <= in_s:
            raise ValueError(
                f"{shot_dir.name} quality trims are invalid: {in_s:.3f}s-{out_s:.3f}s "
                f"for {info['duration']:.3f}s clip"
            )

        cut = {
            "source": str(video_path),
            "shot_id": shot_dir.name,
            "in_seconds": round(in_s, 6),
            "out_seconds": round(out_s, 6),
            "speed": 1.0,
            "has_audio": info["has_audio"],
            "original_duration": info["duration"],
            "trimmed": in_s > 0.15 or out_s < info["duration"] - 0.15,
            "quality_action": quality.get("action", "keep"),
            "quality_reasons": quality.get("reasons", []),
            "continuity_timing": continuity_timing,
            "phase8_duration_trim": duration_trim,
        }
        cuts.append(cut)

        trim_info = f" (裁切 {in_s:.1f}s-{out_s:.1f}s)" if cut["trimmed"] else ""
        audio_icon = "🔇" if not info["has_audio"] else "🔊"
        print(f"    • {shot_dir.name}: {info['duration']:.1f}s{trim_info} {audio_icon}")

    # Cross-shot native extension deliberately carries a replay prefix. Phase
    # 8 owns its frame-addressed removal because only now is the full temporal
    # trajectory available. Internal chunk cuts remain materialized in Phase 6.
    continuity_trim_receipts: list[dict[str, Any]] = []
    if continuity_plan:
        decisions_path = Path(shots_dir).parent / "CONTINUITY_SEAM_DECISIONS.json"
        seam_decisions: dict[str, Any] = {}
        if decisions_path.is_file():
            document = json.loads(decisions_path.read_text(encoding="utf-8"))
            if document.get("kind") != "honcut.continuity_seam_decisions.v1":
                raise ValueError(f"unsupported continuity seam decisions in {decisions_path}")
            seam_decisions = document.get("decisions") or {}
            if not isinstance(seam_decisions, dict):
                raise ValueError(f"{decisions_path} must contain a decisions object")
        timeline_fps = int(continuity_plan.get("timeline_fps") or 24)
        cuts_by_shot = {str(cut["shot_id"]): cut for cut in cuts}
        for shot in continuity_plan.get("shots", []):
            predecessor_chunk = str(shot.get("extends_from_chunk_id") or "")
            chunks = shot.get("chunks") or []
            if not predecessor_chunk or not chunks:
                continue
            first_chunk = chunks[0]
            planned_frames = int(first_chunk.get("expected_overlap_frames") or 0)
            if planned_frames <= 0:
                continue
            boundary_id = f"{predecessor_chunk}__{first_chunk['chunk_id']}"
            decision = seam_decisions.get(boundary_id)
            if not isinstance(decision, dict) or decision.get("action") != "hard_trim":
                raise RuntimeError(
                    f"continuous boundary {boundary_id} has no Phase 8 hard-trim decision"
                )
            trim_frames = int(decision.get("trim_frames") or 0)
            if trim_frames < planned_frames:
                raise RuntimeError(
                    f"continuous boundary {boundary_id} trims {trim_frames} frames, "
                    f"below planned replay prefix {planned_frames}"
                )
            shot_id = str(shot.get("shot_id") or "")
            cut = cuts_by_shot.get(shot_id)
            if cut is None:
                raise RuntimeError(f"continuous target shot {shot_id} is missing from edit cuts")
            trim_seconds = trim_frames / timeline_fps
            # Both values are absolute offsets from the raw shot head; taking
            # max avoids double-trimming when visual QA already noticed part
            # of the replay prefix.
            cut["in_seconds"] = round(
                max(float(cut["in_seconds"]), trim_seconds), 6
            )
            cut["trimmed"] = True
            cut["continuity_trim"] = {
                "boundary_id": boundary_id,
                "trim_frames": trim_frames,
                "trim_seconds": round(trim_seconds, 6),
                "frame_policy": "do_not_interpolate",
            }
            if float(cut["in_seconds"]) >= float(cut["out_seconds"]):
                raise RuntimeError(f"continuity trim consumes all of {shot_id}")
            continuity_trim_receipts.append(
                {"shot_id": shot_id, **cut["continuity_trim"]}
            )

    # Insert post-primary FLF2V bridges only at authored continuous boundaries.
    # Cut/transition boundaries deliberately remain adjacent so Phase 8 can
    # apply their edit effect below.
    primary_cuts = list(cuts)
    primary_pairs = [
        (str(primary_cuts[index]["shot_id"]), str(primary_cuts[index + 1]["shot_id"]))
        for index in range(max(0, len(primary_cuts) - 1))
    ]
    transition_by_pair = dict(
        zip(
            primary_pairs,
            transition_decisions or [],
            strict=False,
        )
    )
    bridge_manifest_path = Path(shots_dir).parent / "PRIMARY_SHOT_BRIDGES.json"
    bridge_records: dict[tuple[str, str], dict[str, Any]] = {}
    if bridge_manifest_path.is_file():
        bridge_document = json.loads(bridge_manifest_path.read_text(encoding="utf-8"))
        if bridge_document.get("kind") != "honcut.primary_shot_bridges.v2":
            raise ValueError(f"unsupported primary bridge manifest in {bridge_manifest_path}")
        bridge_records = {
            (str(item.get("source_shot_id")), str(item.get("target_shot_id"))): item
            for item in bridge_document.get("bridges", [])
            if isinstance(item, dict)
        }
    continuous_pairs = {
        (str(bridge.get("source_shot_id")), str(bridge.get("target_shot_id")))
        for bridge in (continuity_plan or {}).get("bridges", [])
        if isinstance(bridge, dict)
    }
    legacy_continuous_targets = {
        str(shot.get("shot_id"))
        for shot in (continuity_plan or {}).get("shots", [])
        if shot.get("boundary_before") == "continuous"
    } if not (continuity_plan or {}).get("bridges") else set()
    planned_bridge_records = {
        (str(bridge.get("source_shot_id")), str(bridge.get("target_shot_id"))): bridge
        for bridge in (continuity_plan or {}).get("bridges", [])
        if isinstance(bridge, dict)
    }
    assembled_cuts: list[dict[str, Any]] = []
    inserted_bridges: list[dict[str, Any]] = []
    bridge_handle_replacements: list[dict[str, Any]] = []
    inserted_bridge_pairs: set[tuple[str, str]] = set()
    for index, cut in enumerate(primary_cuts):
        assembled_cuts.append(cut)
        if index + 1 >= len(primary_cuts):
            continue
        pair = (str(cut["shot_id"]), str(primary_cuts[index + 1]["shot_id"]))
        bridge_record = bridge_records.get(pair)
        if pair in continuous_pairs:
            if bridge_record is None:
                raise RuntimeError(
                    f"continuous primary boundary {pair[0]}__{pair[1]} has no generated bridge"
                )
            bridge_path = Path(shots_dir).parent / str(bridge_record.get("path") or "")
            if not bridge_path.is_file() or bridge_path.stat().st_size == 0:
                raise RuntimeError(
                    f"continuous primary boundary {pair[0]}__{pair[1]} bridge is missing"
                )
            bridge_info = probe_video(str(bridge_path))
            if bridge_info["duration"] <= 0:
                raise RuntimeError(f"bridge has no decodable duration: {bridge_path}")
            planned_bridge = planned_bridge_records.get(pair) or {}
            insertion_policy = str(
                planned_bridge.get("timeline_insertion_policy") or "append"
            )
            replacement = None
            if insertion_policy == "replace_boundary_handles":
                visible_duration = float(
                    planned_bridge.get("visible_duration_s")
                    or planned_bridge.get("target_duration_s")
                    or 0
                )
                if abs(float(bridge_info["duration"]) - visible_duration) > (1 / 30):
                    raise RuntimeError(
                        f"bridge {pair[0]}__{pair[1]} measured "
                        f"{float(bridge_info['duration']):g}s but its visible budget is "
                        f"{visible_duration:g}s"
                    )
                replacement = _replace_boundary_handles(
                    cut,
                    primary_cuts[index + 1],
                    planned_bridge,
                )
                bridge_handle_replacements.append(replacement)
            bridge_id = f"BRIDGE_{pair[0]}__{pair[1]}"
            assembled_cuts.append(
                {
                    "source": str(bridge_path),
                    "shot_id": bridge_id,
                    "in_seconds": 0.0,
                    "out_seconds": bridge_info["duration"],
                    "speed": 1.0,
                    "has_audio": bridge_info["has_audio"],
                    "original_duration": bridge_info["duration"],
                    "trimmed": False,
                    "quality_action": "generated_continuity_bridge",
                    "quality_reasons": [],
                    "primary_bridge": bridge_record,
                }
            )
            inserted_bridges.append(
                {
                    "boundary_id": f"{pair[0]}__{pair[1]}",
                    "shot_id": bridge_id,
                    "path": str(bridge_path),
                    "generation_duration_s": float(
                        planned_bridge.get("generation_duration_s")
                        or planned_bridge.get("target_duration_s")
                        or bridge_info["duration"]
                    ),
                    "visible_duration_s": float(bridge_info["duration"]),
                    "timeline_insertion_policy": insertion_policy,
                    "handle_replacement": replacement,
                }
            )
            inserted_bridge_pairs.add(pair)
        elif bridge_record is not None:
            raise RuntimeError(
                f"cut/transition boundary {pair[0]}__{pair[1]} must not contain a bridge"
            )
    missing_bridges = continuous_pairs - inserted_bridge_pairs
    if missing_bridges:
        missing = ", ".join(f"{source}__{target}" for source, target in sorted(missing_bridges))
        raise RuntimeError(
            f"planned continuous primary bridges are not adjacent in assembly order: {missing}"
        )
    unexpected_bridges = set(bridge_records) - continuous_pairs
    if (continuity_plan or {}).get("bridges") and unexpected_bridges:
        unexpected = ", ".join(
            f"{source}__{target}" for source, target in sorted(unexpected_bridges)
        )
        raise RuntimeError(
            f"bridge manifest contains cut/transition or unplanned boundaries: {unexpected}"
        )
    cuts = assembled_cuts

    # Build transitions
    transitions = []
    transition_locks = []
    if transition_decisions:
        for i in range(len(cuts) - 1):
            current_id = str(cuts[i]["shot_id"])
            next_shot_id = str(cuts[i + 1]["shot_id"])
            touches_bridge = current_id.startswith("BRIDGE_") or next_shot_id.startswith(
                "BRIDGE_"
            )
            if touches_bridge:
                transition_type = "cut"
                locked = True
                lock_reason = "generated continuity bridge"
                decision = {"decision": "cut"}
            else:
                decision = transition_by_pair.get((current_id, next_shot_id), {"decision": "cut"})
                locked = next_shot_id in legacy_continuous_targets
                transition_type = "cut" if locked else str(decision["decision"])
                lock_reason = "continuous editorial boundary"
            if locked:
                transition_locks.append(
                    {
                        "index": i,
                        "before_shot_id": next_shot_id,
                        "reason": f"{lock_reason} forbids transition effects",
                    }
                )
            transition_entry = {
                "index": i,
                "type": transition_type,
                "duration": 0.0 if transition_type == "cut" else transition_duration,
                # A visual hard cut must not imply a hard cut in sound.
                # Keep picture timing frame-accurate while gently fading
                # both sides of the audio boundary. Dissolves retain a
                # true overlap, matched to the visual transition length.
                "audio_transition": (
                    "edge_fade" if transition_type == "cut" else "crossfade"
                ),
                "audio_duration": (
                    0.35 if transition_type == "cut" else transition_duration
                ),
            }
            transition_frames = round(float(transition_entry["duration"]) * 30)
            transition_entry["duration_frames"] = transition_frames
            transition_entry["duration"] = round(transition_frames / 30, 6)
            if locked:
                transition_entry.update(
                    locked=True,
                    lock_reason=lock_reason,
                )
            transitions.append(transition_entry)

    # Quality review removes obvious dead edges first.  Then use one bounded
    # pacing factor across primary clips and bridges so every authored frame is
    # retained and the editor never solves a material surplus by chopping off
    # the end of the film.  Because primary material is capped at 1.3x, the
    # required acceleration is likewise bounded by that declared ratio.
    pacing_normalization = None
    if target_duration and cuts:
        source_duration = sum(cut["out_seconds"] - cut["in_seconds"] for cut in cuts)
        overlap = sum(
            0.0 if item["type"] == "cut" else float(item["duration"]) for item in transitions
        )
        projected = source_duration - overlap
        speed = source_duration / (float(target_duration) + overlap)
        ratio_limit = float(
            ((continuity_plan or {}).get("material_budget") or {}).get(
                "pre_edit_duration_ratio_limit",
                1.3,
            )
        )
        if abs(projected - float(target_duration)) > (1 / 30):
            if 0.85 <= speed <= ratio_limit + 1e-6:
                for cut in cuts:
                    cut["speed"] = round(speed, 6)
                pacing_normalization = {
                    "method": "bounded_all_frame_pacing_normalization",
                    "source_duration_s": round(source_duration, 6),
                    "transition_overlap_s": round(overlap, 6),
                    "projected_before_s": round(projected, 6),
                    "target_duration_s": float(target_duration),
                    "speed": round(speed, 6),
                    "minimum_speed": 0.85,
                    "maximum_speed": ratio_limit,
                    "preserves_all_reviewed_frames": True,
                }
            elif projected > float(target_duration):
                raise RuntimeError(
                    "reviewed timeline cannot reach delivery duration without unsafe "
                    f"tail deletion: requires {speed:g}x pacing outside the "
                    f"0.85-{ratio_limit:g}x budget"
                )

    timeline = _build_timeline(cuts, transitions, target_fps=30)
    if (
        target_duration
        and pacing_normalization
        and pacing_normalization["speed"] > 1.0
        and timeline
    ):
        target_frames = round(float(target_duration) * 30)
        projected_frames = timeline[-1]["output_end_frame"]
        residual_frames = target_frames - projected_frames
        if residual_frames:
            # Per-cut integer rounding can miss the otherwise exact global
            # pacing solution by a frame or two.  Close that residual on the
            # longest reviewed cut with an imperceptible speed adjustment.
            candidate = max(
                cuts,
                key=lambda item: float(item["out_seconds"])
                - float(item["in_seconds"]),
            )
            duration_s = float(candidate["out_seconds"]) - float(
                candidate["in_seconds"]
            )
            current_frames = round(duration_s / float(candidate["speed"]) * 30)
            desired_frames = current_frames + residual_frames
            if desired_frames <= 0:
                raise RuntimeError("timeline frame closure would consume an entire cut")
            adjusted_speed = duration_s * 30 / desired_frames
            candidate["speed"] = round(adjusted_speed, 9)
            pacing_normalization["frame_closure"] = {
                "shot_id": candidate["shot_id"],
                "residual_frames": residual_frames,
                "adjusted_speed": candidate["speed"],
            }
            timeline = _build_timeline(cuts, transitions, target_fps=30)
            if timeline[-1]["output_end_frame"] != target_frames:
                raise RuntimeError("bounded pacing could not close the delivery frame budget")
    projected_frames = timeline[-1]["output_end_frame"] if timeline else 0
    return {
        "cuts": cuts,
        "transitions": transitions,
        "timeline": timeline,
        "metadata": {
            "compose_target": {
                "width": target_width,
                "height": target_height,
                "fit": fit_mode,
            },
            "target_fps": 30,
            "target_frames": None if target_duration is None else round(target_duration * 30),
            "projected_frames": projected_frames,
            "target_duration": target_duration,
            "quality_reviewed": bool(quality_report),
            "allow_unresolved_reshoots": allow_unresolved_reshoots,
            "unresolved_reshoots": unresolved_reshoots,
            "transition_locks": transition_locks,
            "continuity_trims": continuity_trim_receipts,
            "phase8_duration_trims": phase8_duration_trims,
            "inserted_primary_bridges": inserted_bridges,
            "bridge_handle_replacements": bridge_handle_replacements,
            "pacing_normalization": pacing_normalization,
            "audio_transition_policy": {
                "visual_cut": "equal_power_edge_fade",
                "visual_dissolve": "equal_power_crossfade",
            },
        },
    }


# ─── Execute Edit Decisions ──────────────────────────────────────────────────


def execute_edit_decisions(edit_decisions: dict, output_path: str) -> dict:
    """Execute: trim → normalize → transition → concat.

    Returns: {"success": bool, "output": str, "duration": float, "segments": int}
    """
    cuts = edit_decisions.get("cuts", [])
    transitions = edit_decisions.get("transitions", [])
    meta = edit_decisions.get("metadata", {})
    target = meta.get("compose_target", {})
    tw = target.get("width", 1920)
    th = target.get("height", 1080)
    tfps = meta.get("target_fps", 30)
    output_path = Path(output_path)

    if not cuts:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {"success": False, "error": "No cuts"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".edit_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files: list[Path] = []

    try:
        # Step 1: Trim + normalize each segment
        segments = []
        for i, cut in enumerate(cuts):
            src = Path(cut["source"])
            if not src.exists():
                raise FileNotFoundError(f"reviewed edit source is missing: {src}")

            seg = temp_dir / f"seg_{i:04d}.mp4"
            in_s = cut["in_seconds"]
            dur = cut["out_seconds"] - in_s
            speed = cut.get("speed", 1.0)
            has_audio = cut.get("has_audio", False)

            # --- P2-5c: cover fit mode（裁切填充，无黑边）---
            fit_mode = (
                edit_decisions.get("metadata", {}).get("compose_target", {}).get("fit", "pad")
            )
            if fit_mode == "cover":
                vf = [
                    f"scale={tw}:{th}:force_original_aspect_ratio=increase",
                    f"crop={tw}:{th}",
                    "setsar=1",
                    f"fps={tfps}",
                ]
            else:
                # pad 模式（默认，带黑边）
                vf = [
                    f"scale={tw}:{th}:force_original_aspect_ratio=decrease",
                    f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "setsar=1",
                    f"fps={tfps}",
                ]
            af = []
            if speed != 1.0:
                vf.append(f"setpts={1.0 / speed}*PTS")
                af.append(f"atempo={speed}")

            if has_audio:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(in_s),
                    "-t",
                    str(dur),
                    "-i",
                    str(src),
                    "-filter:v",
                    ",".join(vf),
                ]
                if af:
                    cmd += ["-filter:a", ",".join(af)]
                cmd += [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(tfps),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(seg),
                ]
            else:
                # Inject silent audio via lavfi
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(in_s),
                    "-t",
                    str(dur),
                    "-i",
                    str(src),
                    "-f",
                    "lavfi",
                    "-t",
                    str(dur),
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-filter:v",
                    ",".join(vf),
                ]
                if af:
                    cmd += ["-filter:a", ",".join(af)]
                cmd += [
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(tfps),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(seg),
                ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                segments.append(str(seg))
                temp_files.append(seg)
            except subprocess.CalledProcessError as e:
                detail = str(e.stderr)[:500] if e.stderr else str(e)
                raise RuntimeError(
                    f"reviewed segment {i} normalization failed for {src}: {detail}"
                ) from e

        if not segments:
            return {"success": False, "error": "No segments produced"}

        # Step 2: Concat with transitions
        if len(segments) == 1:
            shutil.copy2(segments[0], str(output_path))
        elif transitions:
            # Even an all-cut picture edit needs the transition renderer so
            # independently generated shot audio is not concatenated with
            # sample-level discontinuities.
            _xfade_chain(segments, transitions, output_path, temp_dir, temp_files)
        else:
            _concat_copy(segments, output_path, temp_dir, temp_files)

        info = probe_video(str(output_path))
        timeline = {
            "version": 1,
            "artifact": output_path.name,
            "duration_s": info["duration"],
            "duration_frames": round(info["duration"] * tfps),
            "shots": edit_decisions.get("timeline", []),
            "transitions": transitions,
            "compose_target": target,
        }
        timeline_path = output_path.parent / "edit_timeline.json"
        temporary = timeline_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(timeline_path)
        return {
            "success": True,
            "output": str(output_path),
            "duration": info["duration"],
            "segments": len(segments),
            "timeline": str(timeline_path),
        }

    except Exception as e:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {"success": False, "error": str(e)}

    finally:
        for f in temp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── P2-B: Boundary Frame Consistency Check ──────────────────────────────────


def _build_timeline(
    cuts: list[dict],
    transitions: list[dict],
    *,
    target_fps: int = 30,
) -> list[dict]:
    """Build the canonical output-time mapping for all reviewed cuts."""
    transitions_by_index = {int(item["index"]): item for item in transitions}
    entries: list[dict] = []
    output_start_frame = 0
    for index, cut in enumerate(cuts):
        speed = float(cut.get("speed", 1.0) or 1.0)
        source_in = float(cut["in_seconds"])
        source_out = float(cut["out_seconds"])
        output_duration = (source_out - source_in) / speed
        output_duration_frames = round(output_duration * target_fps)
        output_end_frame = output_start_frame + output_duration_frames
        transition = transitions_by_index.get(index)
        overlap_frames = 0
        if transition and transition.get("type") != "cut":
            overlap_frames = min(
                int(
                    transition.get("duration_frames")
                    or round(float(transition.get("duration", 0.0)) * target_fps)
                ),
                output_duration_frames,
            )
        entries.append(
            {
                "index": index,
                "shot_id": cut.get("shot_id") or Path(cut["source"]).parent.name,
                "source": cut["source"],
                "source_in_s": round(source_in, 6),
                "source_out_s": round(source_out, 6),
                "speed": speed,
                "output_start_frame": output_start_frame,
                "output_end_frame": output_end_frame,
                "output_duration_frames": output_duration_frames,
                "overlap_to_next_frames": overlap_frames,
                "output_start_s": round(output_start_frame / target_fps, 6),
                "output_end_s": round(output_end_frame / target_fps, 6),
                "output_duration_s": round(output_duration_frames / target_fps, 6),
                "overlap_to_next_s": round(overlap_frames / target_fps, 6),
            }
        )
        output_start_frame = output_end_frame - overlap_frames
    return entries


def check_boundary_consistency(video_a: Path, video_b: Path) -> dict:
    """检查两个相邻视频的边界帧一致性（参考 HonCut frame sampler）。

    Returns:
        {"consistent": bool, "issues": [...], "recommended_transition": str}
    """
    issues = []

    def _probe(path):
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return json.loads(result.stdout)
        except Exception:
            return None

    info_a = _probe(video_a)
    info_b = _probe(video_b)

    if not info_a or not info_b:
        return {"consistent": True, "issues": [], "recommended_transition": "dissolve"}

    # 检查分辨率一致性
    def _get_resolution(info):
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                return s.get("width", 0), s.get("height", 0)
        return (0, 0)

    res_a = _get_resolution(info_a)
    res_b = _get_resolution(info_b)
    if res_a != res_b:
        issues.append(f"分辨率不一致: {res_a} vs {res_b}")

    # 检查 fps 一致性
    def _get_fps(info):
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                fps_str = s.get("r_frame_rate", "30/1")
                try:
                    num, den = fps_str.split("/")
                    return float(num) / float(den)
                except Exception:
                    return 30.0
        return 30.0

    fps_a = _get_fps(info_a)
    fps_b = _get_fps(info_b)
    if abs(fps_a - fps_b) > 1.0:
        issues.append(f"FPS 不一致: {fps_a:.1f} vs {fps_b:.1f}")

    # 推荐转场
    if issues:
        recommended = "dissolve"  # 不一致时用 dissolve 过渡
    else:
        recommended = "cut"  # 一致时可以直接 cut

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "recommended_transition": recommended,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _concat_copy(segments: list, output_path: Path, temp_dir: Path, temp_files: list):
    """Simple concat with stream copy (all hard cuts)."""
    concat_list = temp_dir / "concat_list.txt"
    temp_files.append(concat_list)
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{Path(seg).resolve()}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)


def _xfade_chain(
    segments: list, transitions: list, output_path: Path, temp_dir: Path, temp_files: list
):
    """Apply xfade transitions as bounded pairwise renders.

    FFmpeg 9 can truncate a long, nested xfade filter graph even when every
    input has normalized timestamps. Rendering one transition at a time keeps
    each graph simple and makes every intermediate duration independently
    probeable.
    """
    trans_map = {t["index"]: t for t in transitions}
    current = Path(segments[0])
    try:
        for i, next_segment in enumerate(segments[1:]):
            following = Path(next_segment)
            current_duration = probe_video(str(current))["duration"]
            if current_duration <= 0:
                raise RuntimeError(f"invalid xfade intermediate duration: {current}")

            t = trans_map.get(i, {"type": "cut", "duration": 0.5})
            ttype = t["type"]
            tdur = t.get("duration", 0.5)
            following_duration = probe_video(str(following))["duration"]
            if following_duration <= 0:
                raise RuntimeError(f"invalid following segment duration: {following}")

            # --- P2-B: 边界帧一致性检查（参考 HonCut AI Clip Chaining）---
            try:
                boundary = check_boundary_consistency(current, following)
                if not boundary["consistent"] and ttype == "cut":
                    # The edit decision is authoritative. Record the visual
                    # recommendation without silently changing picture timing,
                    # otherwise the persisted EDL would no longer match output.
                    print(
                        f"    [P2-B] 边界不一致({boundary['issues'][0]})，"
                        "保留导演 cut 决策并使用音频软边界"
                    )
            except Exception:
                pass  # 检查失败不影响现有流程

            if ttype == "cut":
                tdur = 0.0
                audio_fade = min(
                    float(t.get("audio_duration", 0.35)),
                    current_duration / 2,
                    following_duration / 2,
                )
                fade_start = max(0.0, current_duration - audio_fade)
                filter_complex = (
                    "[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];"
                    "[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];"
                    "[v0][v1]concat=n=2:v=1:a=0[v];"
                    "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS,"
                    f"apad,atrim=duration={current_duration},"
                    f"afade=t=out:st={fade_start}:d={audio_fade}:curve=qsin[a0];"
                    "[1:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS,"
                    f"apad,atrim=duration={following_duration},"
                    f"afade=t=in:d={audio_fade}:curve=qsin[a1];"
                    "[a0][a1]concat=n=2:v=0:a=1[a]"
                )
            else:
                xname = {"dissolve": "fade", "fade": "fadeblack", "cut": "fade"}.get(ttype, "fade")
                offset = round(max(0, current_duration - tdur), 3)
                filter_complex = (
                    "[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];"
                    "[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];"
                    f"[v0][v1]xfade=transition={xname}:duration={tdur}:offset={offset}[v];"
                    "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a0];"
                    "[1:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a1];"
                    f"[a0][a1]acrossfade=d={tdur}:c1=qsin:c2=qsin[a]"
                )
            intermediate = temp_dir / f"xfade_{i:04d}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(current),
                "-i",
                str(following),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(intermediate),
            ]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            rendered_duration = probe_video(str(intermediate))["duration"]
            expected_duration = current_duration + following_duration - tdur
            if abs(rendered_duration - expected_duration) > 0.25:
                raise RuntimeError(
                    "xfade intermediate video duration mismatch: "
                    f"expected={expected_duration:.3f}s actual={rendered_duration:.3f}s "
                    f"at transition {i}"
                )
            temp_files.append(intermediate)
            current = intermediate
        shutil.copy2(current, output_path)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        stderr = getattr(e, "stderr", None)
        detail = str(stderr)[:500] if stderr else str(e)
        # A concat fallback changes the authored overlap durations and makes
        # edit_timeline.json false. Fail closed so callers can retry or retain
        # the previous assembly instead of publishing mistimed captions/audio.
        raise RuntimeError(f"transition render failed: {detail}") from e
