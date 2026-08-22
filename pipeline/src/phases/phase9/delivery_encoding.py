"""Final delivery encoding filters and reviewed-timeline validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils.file_integrity import _file_sha256


def _final_encode_duration_gate(
    encode_input_durations: dict[str, float | None],
    encoded_durations: dict[str, float | None],
    *,
    delivery_contract: dict[str, Any],
    requested_duration: float | None,
    fps: float,
) -> dict[str, Any]:
    """Build the non-contradictory Phase 9 final-encode duration receipt.

    Phase 8 and the rhythm editor own the reviewed edit timeline. Phase 9's
    encoder must preserve that input and match the cryptographically bound
    delivery receipt rather than silently re-trimming it to an earlier target.
    """
    duration_tolerance_s = 2 / float(fps)
    audio_duration_tolerance_s = max(duration_tolerance_s, 0.05)
    comparison_epsilon_s = 1e-6
    duration_deltas = {
        kind: (
            None
            if encode_input_durations[kind] is None or encoded_durations[kind] is None
            else round(
                abs(encoded_durations[kind] - encode_input_durations[kind]),
                6,
            )
        )
        for kind in ("video", "audio")
    }
    encode_conserved = all(
        encode_input_durations[kind] is None
        or (
            encoded_durations[kind] is not None
            and duration_deltas[kind]
            <= (
                audio_duration_tolerance_s
                if kind == "audio"
                else duration_tolerance_s
            )
            + comparison_epsilon_s
        )
        for kind in ("video", "audio")
    )
    reviewed_duration = float(delivery_contract["duration_s"])
    encode_input_to_reviewed_delta = round(
        abs(float(encode_input_durations["video"]) - reviewed_duration),
        6,
    )
    encoded_to_reviewed_delta = (
        None
        if encoded_durations["video"] is None
        else round(abs(float(encoded_durations["video"]) - reviewed_duration), 6)
    )
    reviewed_timeline_matched = (
        encode_input_to_reviewed_delta
        <= duration_tolerance_s + comparison_epsilon_s
        and encoded_to_reviewed_delta is not None
        and encoded_to_reviewed_delta
        <= duration_tolerance_s + comparison_epsilon_s
    )
    requested_duration_delta = (
        None
        if requested_duration is None or encoded_durations["video"] is None
        else round(
            abs(encoded_durations["video"] - float(requested_duration)),
            6,
        )
    )
    requested_duration_within_tolerance = (
        None
        if requested_duration_delta is None
        else requested_duration_delta
        <= duration_tolerance_s + comparison_epsilon_s
    )
    return {
        "passed": encode_conserved and reviewed_timeline_matched,
        "artifact": "polished.mp4",
        "expected": encode_input_durations,
        "actual": encoded_durations,
        "absolute_delta_s": duration_deltas,
        "authoritative_duration_s": reviewed_duration,
        "authoritative_duration_source": "delivery_timeline.json",
        "encode_input_to_authoritative_delta_s": encode_input_to_reviewed_delta,
        "encoded_to_authoritative_delta_s": encoded_to_reviewed_delta,
        "reviewed_timeline": delivery_contract,
        "requested_duration_s": requested_duration,
        "requested_duration_delta_s": requested_duration_delta,
        "requested_duration_within_tolerance": requested_duration_within_tolerance,
        "requested_duration_enforced_by_final_encode": False,
        "tolerance_s": {
            "video": duration_tolerance_s,
            "audio": audio_duration_tolerance_s,
        },
        "tolerance_frames": 2,
        "basis": (
            "Phase 9 encode input is SHA-256-bound to delivery_timeline.json; "
            "the earlier requested duration is diagnostic only"
        ),
    }


def _final_encode_filters(profile: dict[str, Any]) -> tuple[str, str]:
    """Return delivery filters that normalize format without changing runtime."""
    video_filters = (
        "setpts=PTS-STARTPTS,"
        f"scale={profile['width']}:{profile['height']}:"
        "force_original_aspect_ratio=increase,"
        f"crop={profile['width']}:{profile['height']},setsar=1,"
        f"fps={profile['fps']}"
    )
    audio_filters = "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0"
    return video_filters, audio_filters


def _validated_reviewed_delivery_contract(
    output_dir: Path,
    encode_input: Path,
    encode_input_durations: dict[str, float | None],
    *,
    fps: float,
) -> dict[str, Any]:
    """Validate that final encoding consumes the current reviewed timeline."""
    from phases.phase9.rhythm_editor import DELIVERY_TIMELINE_SCHEMA

    receipt_path = Path(output_dir) / "delivery_timeline.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Final encode requires a readable delivery_timeline.json from the current rhythm edit"
        ) from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != DELIVERY_TIMELINE_SCHEMA:
        raise RuntimeError(
            f"Final encode requires delivery timeline schema {DELIVERY_TIMELINE_SCHEMA}"
        )
    if receipt.get("artifact") != encode_input.name:
        raise RuntimeError(
            "Delivery timeline artifact does not match the final encode input: "
            f"{receipt.get('artifact')!r} != {encode_input.name!r}"
        )
    expected_sha = str(receipt.get("source_sha256") or "")
    actual_sha = _file_sha256(encode_input)
    if len(expected_sha) != 64 or expected_sha != actual_sha:
        raise RuntimeError(
            "Delivery timeline SHA-256 does not match the final encode input"
        )
    source_size = receipt.get("source_size_bytes")
    if not isinstance(source_size, int) or source_size != encode_input.stat().st_size:
        raise RuntimeError(
            "Delivery timeline byte size does not match the final encode input"
        )

    try:
        duration = float(receipt["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Delivery timeline has no valid duration_s") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Delivery timeline duration_s must be finite and positive")
    shots = receipt.get("shots")
    if not isinstance(shots, list) or not shots:
        raise RuntimeError("Delivery timeline must contain at least one reviewed shot")

    comparison_epsilon_s = 1e-6
    previous_end = 0.0
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not str(shot.get("shot_id") or "").strip():
            raise RuntimeError(f"Delivery timeline shot {index} has no shot_id")
        try:
            start = float(shot["output_start_s"])
            end = float(shot["output_end_s"])
            item_duration = float(shot["output_duration_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Delivery timeline shot {index} has invalid output timing"
            ) from exc
        if not all(math.isfinite(value) for value in (start, end, item_duration)):
            raise RuntimeError(f"Delivery timeline shot {index} timing must be finite")
        if abs(start - previous_end) > comparison_epsilon_s:
            raise RuntimeError(
                f"Delivery timeline shot {index} is not contiguous with its predecessor"
            )
        if end <= start or abs((end - start) - item_duration) > comparison_epsilon_s:
            raise RuntimeError(f"Delivery timeline shot {index} duration is inconsistent")
        previous_end = end
    if abs(previous_end - duration) > comparison_epsilon_s:
        raise RuntimeError("Delivery timeline final boundary does not match duration_s")

    video_duration = encode_input_durations.get("video")
    if video_duration is None:
        raise RuntimeError("Final encode input has no measurable video duration")
    tolerance_s = 2 / float(fps)
    input_delta = abs(float(video_duration) - duration)
    if input_delta > tolerance_s + comparison_epsilon_s:
        raise RuntimeError(
            "Delivery timeline duration does not match the final encode input: "
            f"timeline={duration:.6f}s input={float(video_duration):.6f}s"
        )
    return {
        "schema": DELIVERY_TIMELINE_SCHEMA,
        "artifact": encode_input.name,
        "source_sha256": actual_sha,
        "source_size_bytes": source_size,
        "duration_s": duration,
        "shot_count": len(shots),
        "timing_contiguous": True,
        "input_duration_delta_s": round(input_delta, 6),
    }
