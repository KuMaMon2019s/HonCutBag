"""Phase 8 per-shot visual QA and actionable edit/reshoot decisions."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    temporal_visual_qa_instruction,
)
from utils.body_action_contracts import body_action_qa_instruction
from utils.privacy_visual_policy import SYNTHETIC_QA_CONTRACT


SemanticReviewer = Callable[[list[Path], dict[str, Any]], dict[str, Any]]


def _multimodal_circuit_is_open(output_path: Path) -> bool:
    """Avoid repeated per-shot calls after the same run's order review failed."""
    receipt = Path(output_path).parent / "storyboard_order_review.json"
    if not receipt.is_file():
        return False
    try:
        review = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return review.get("source") == "deterministic_fallback"


def _read_pixels(path: Path) -> np.ndarray:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {path}")
        return image.astype(np.float32)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def is_black_frame(path: Path, threshold: float = 20.0) -> bool:
    return float(_read_pixels(Path(path)).mean()) < threshold


def is_static_frame(first: Path, last: Path, threshold: float = 3.0) -> bool:
    first_pixels = _read_pixels(Path(first))
    last_pixels = _read_pixels(Path(last))
    if first_pixels.shape != last_pixels.shape:
        return False
    return float(np.abs(first_pixels - last_pixels).mean()) < threshold


def probe_duration(video_path: Path) -> float:
    # Prefer the actual video-stream duration. Container duration is often the
    # longest stream and can hide a truncated picture behind a correctly sized
    # (or padded) audio track.
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    value = completed.stdout.strip().splitlines()
    if value and value[0] not in {"N/A", ""}:
        return float(value[0])
    fallback = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return float(fallback.stdout.strip().splitlines()[0])


def _sample_timestamps(duration: float, max_frames: int, interval_s: float) -> list[float]:
    if duration <= 0:
        return []
    usable_end = max(0.0, duration - min(0.1, duration / 10))
    count = max(3, min(max_frames, int(math.ceil(duration / interval_s)) + 1))
    return [round(float(value), 4) for value in np.linspace(0.05, usable_end, count)]


def extract_review_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    max_frames: int = 12,
    interval_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract dense, bounded samples across a complete generated shot."""
    duration = probe_duration(video_path)
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)

    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(_sample_timestamps(duration, max_frames, interval_s)):
        output = frames_dir / f"frame_{index:03d}.jpg"
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.6f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(output),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = completed.stderr.strip().splitlines()
            raise RuntimeError(
                f"ffmpeg frame extraction failed for {video_path} at {timestamp:.3f}s: "
                f"{detail[-1] if detail else 'unknown error'}"
            )
        frames.append({"path": output, "timestamp_s": timestamp})
    return frames


def extract_three_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    """Compatibility wrapper returning representative first/middle/tail frames."""
    frames = extract_review_frames(video_path, frames_dir, max_frames=3, interval_s=9999)
    return [item["path"] for item in frames]


def _frame_metrics(path: Path) -> dict[str, Any]:
    pixels = _read_pixels(path)
    gray = pixels.mean(axis=2)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    blur_score = float(laplacian.var())
    issues: list[str] = []
    if brightness < 20:
        issues.append("black")
    elif brightness < 35:
        issues.append("underexposed")
    elif brightness > 225:
        issues.append("overexposed")
    if blur_score < 45:
        issues.append("blurry")
    if contrast < 18:
        issues.append("low_contrast")
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "issues": issues,
    }


def measure_motion_activity(frame_paths: list[Path]) -> dict[str, float]:
    """Measure visible inter-sample change, including animated-still failures."""
    arrays: list[np.ndarray] = []
    for path in frame_paths:
        pixels = _read_pixels(Path(path))
        gray = pixels.mean(axis=2)
        # Block averaging suppresses rain, grain, sparks, and codec shimmer so
        # those high-frequency effects cannot masquerade as subject motion.
        height, width = gray.shape
        factor = max(1, min(height // 90, width // 160))
        pooled_height = (height // factor) * factor
        pooled_width = (width // factor) * factor
        pooled = gray[:pooled_height, :pooled_width].reshape(
            pooled_height // factor,
            factor,
            pooled_width // factor,
            factor,
        ).mean(axis=(1, 3))
        arrays.append(pooled)
    if len(arrays) < 2:
        return {"median_mae": 0.0, "changed_pixel_ratio": 0.0, "pair_count": 0}
    maes: list[float] = []
    changed_ratios: list[float] = []
    for previous, current in zip(arrays, arrays[1:]):
        if previous.shape != current.shape:
            continue
        difference = np.abs(current - previous)
        maes.append(float(difference.mean()))
        changed_ratios.append(float((difference > 12.0).mean()))
    return {
        "median_mae": round(float(np.median(maes)), 4) if maes else 0.0,
        "changed_pixel_ratio": (
            round(float(np.median(changed_ratios)), 6) if changed_ratios else 0.0
        ),
        "pair_count": len(maes),
    }


def _detect_black_segments(video_path: Path) -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path), "-vf", "blackdetect=d=0.08:pix_th=0.10",
            "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    segments: list[dict[str, float]] = []
    pattern = re.compile(
        r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)\s+"
        r"black_duration:(?P<duration>[\d.]+)"
    )
    for match in pattern.finditer(completed.stderr):
        segments.append({key: round(float(value), 4) for key, value in match.groupdict().items()})
    return segments


def _detect_freeze_segments(video_path: Path) -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path), "-vf", "freezedetect=n=-50dB:d=1.5",
            "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    starts: list[float] = []
    durations: list[float] = []
    ends: list[float] = []
    for line in completed.stderr.splitlines():
        start = re.search(r"freeze_start:\s*([\d.]+)", line)
        duration = re.search(r"freeze_duration:\s*([\d.]+)", line)
        end = re.search(r"freeze_end:\s*([\d.]+)", line)
        if start:
            starts.append(float(start.group(1)))
        if duration:
            durations.append(float(duration.group(1)))
        if end:
            ends.append(float(end.group(1)))
    segments: list[dict[str, float]] = []
    for index, start in enumerate(starts):
        duration = durations[index] if index < len(durations) else 0.0
        end = ends[index] if index < len(ends) else start + duration
        segments.append({"start": round(start, 4), "end": round(end, 4), "duration": round(duration, 4)})
    return segments


def decide_shot_action(
    duration: float,
    frames: list[dict[str, Any]],
    black_segments: list[dict[str, float]],
    freeze_segments: list[dict[str, float]],
    semantic_review: dict[str, Any] | None = None,
    shot_meta: dict[str, Any] | None = None,
    motion_activity: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Turn technical/semantic evidence into one keep, trim, or reshoot action."""
    boundary_tolerance = min(0.35, max(0.12, duration * 0.04))
    trim_start = 0.0
    trim_end = duration
    reasons: list[str] = []
    interior_black: list[dict[str, float]] = []

    for segment in black_segments:
        if segment["start"] <= boundary_tolerance:
            trim_start = max(trim_start, segment["end"])
        elif segment["end"] >= duration - boundary_tolerance:
            trim_end = min(trim_end, segment["start"])
        else:
            interior_black.append(segment)

    semantic_review = semantic_review or {}
    if semantic_review.get("verdict") in {"fail", "reshoot", "unavailable"}:
        reasons.extend(str(issue) for issue in semantic_review.get("issues", []) or ["semantic visual review failed"])
        return {"action": "reshoot", "reasons": reasons, "trim_start_s": 0.0, "trim_end_s": duration}

    if interior_black:
        reasons.append(f"interior black segment detected: {interior_black}")

    severe_samples = [
        frame for frame in frames
        if "black" in frame.get("metrics", {}).get("issues", [])
        or (
            "blurry" in frame.get("metrics", {}).get("issues", [])
            and any(
                issue in frame.get("metrics", {}).get("issues", [])
                for issue in ("underexposed", "overexposed")
            )
        )
    ]
    if len(severe_samples) >= max(2, math.ceil(len(frames) * 0.4)):
        reasons.append(f"{len(severe_samples)}/{len(frames)} sampled frames have severe quality issues")

    meta = shot_meta or {}
    expected_motion = " ".join(
        str(meta.get(key, "")) for key in ("action", "camera", "camera_movement", "motion")
    ).strip().lower()
    intentionally_static = not expected_motion or any(word in expected_motion for word in ("static", "locked", "固定"))
    long_freezes = [
        segment for segment in freeze_segments
        if segment.get("duration", 0) >= max(1.5, duration * 0.3)
    ]
    if long_freezes and not intentionally_static:
        reasons.append(f"unexpected long freeze detected: {long_freezes}")

    activity = motion_activity or {}
    camera_motion = str(
        meta.get("camera_movement")
        or meta.get("camera_movement_en")
        or meta.get("camera")
        or ""
    ).lower()
    generation_actions = meta.get("generation_actions") or []
    requires_visible_motion = bool(generation_actions) or camera_motion not in {
        "", "static", "fixed", "locked", "unspecified"
    } or duration > 6.0
    animated_still = (
        activity.get("pair_count", 0) >= 2
        and activity.get("median_mae", 0.0) < 3.5
        and activity.get("changed_pixel_ratio", 0.0) < 0.06
    )
    if requires_visible_motion and animated_still:
        reasons.append(
            "animated-still motion failure: "
            f"median_mae={activity.get('median_mae', 0.0):.3f}, "
            f"changed_pixel_ratio={activity.get('changed_pixel_ratio', 0.0):.3f}"
        )

    remaining = trim_end - trim_start
    if remaining < max(1.0, duration * 0.5):
        reasons.append(f"boundary trimming would leave only {remaining:.2f}s of {duration:.2f}s")

    if reasons:
        return {"action": "reshoot", "reasons": reasons, "trim_start_s": 0.0, "trim_end_s": duration}
    if trim_start > 0.05 or trim_end < duration - 0.05:
        return {
            "action": "trim",
            "reasons": [f"trim boundary defects to {trim_start:.3f}s-{trim_end:.3f}s"],
            "trim_start_s": round(trim_start, 4),
            "trim_end_s": round(trim_end, 4),
        }
    return {"action": "keep", "reasons": [], "trim_start_s": 0.0, "trim_end_s": round(duration, 4)}


def _character_reference_paths(
    output_dir: Path | None,
    shot_meta: dict[str, Any],
) -> list[tuple[str, Path]]:
    """Resolve the canonical character images explicitly owned by one shot."""
    if output_dir is None:
        return []
    requested: set[str] = set()
    who = shot_meta.get("who")
    if who == []:
        return []
    if isinstance(who, list):
        requested.update(str(value).casefold() for value in who if value)
    elif who:
        requested.add(str(who).casefold())
    for asset in shot_meta.get("associate_assets") or []:
        if isinstance(asset, str) and asset.startswith("char:"):
            requested.add(asset[5:].split(":", 1)[0].casefold())
    if not requested:
        return []

    try:
        characters = json.loads(
            (output_dir / "CHARACTERS.json").read_text(encoding="utf-8")
        ).get("characters", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []

    references: list[tuple[str, Path]] = []
    for character in characters:
        character_id = str(character.get("id") or "").strip()
        keys = {
            character_id.casefold(),
            str(character.get("name") or "").casefold(),
            *(
                str(value).casefold()
                for value in character.get("aliases", [])
                if value
            ),
        }
        if not character_id or requested.isdisjoint(keys):
            continue
        for character_dir in (
            output_dir / "characters" / character_id,
            output_dir / "characters" / "characters" / character_id,
        ):
            reference = next(
                (
                    path
                    for path in (
                        character_dir / "face_closeup.png",
                        character_dir / "full_body.png",
                        *sorted(character_dir.glob("variant_*.png")),
                        character_dir / "front.png",
                    )
                    if path.is_file() and path.stat().st_size > 0
                ),
                None,
            )
            if reference is not None:
                references.append((character_id, reference))
                identity_detail = character_dir / "identity_detail.png"
                if identity_detail.is_file() and identity_detail.stat().st_size > 0:
                    references.append(
                        (f"{character_id}:identity_detail", identity_detail)
                    )
                break
    return references


def _uses_synthetic_character_review(output_dir: Path | None) -> bool:
    """Resolve privacy-safe review mode from runtime state or persisted artifacts."""
    from utils.privacy_visual_policy import uses_synthetic_character_review

    return uses_synthetic_character_review(output_dir)


def _automatic_semantic_reviewer(
    output_dir: Path | None = None,
) -> SemanticReviewer | None:
    setting = os.environ.get("HONCUT_SHOT_VLM_REVIEW", "auto").strip().lower()
    if setting in {"0", "false", "off", "no"}:
        return None
    try:
        from clients.ark_multimodal_client import ArkMultimodalClient, review_as
        from schemas.understanding import ShotSemanticReview

        client = ArkMultimodalClient()
    except Exception:
        return None

    synthetic_review = _uses_synthetic_character_review(output_dir)
    qa_contract = (
        SYNTHETIC_QA_CONTRACT
        if synthetic_review
        else "human_visual_anatomy_v1"
    )

    def review(frame_paths: list[Path], shot_meta: dict[str, Any]) -> dict[str, Any]:
        temporal_contract = apply_temporal_visual_contract(shot_meta)
        character_references = _character_reference_paths(output_dir, shot_meta)
        expected = {
            key: shot_meta.get(key)
            for key in (
                "shot_id", "visual", "action", "action_description", "generation_actions",
                "body_action_choreography", "body_action_contract",
                "who", "where",
                "characters", "time", "time_of_day", "lighting", "lighting_key",
                "time_window", "temporal_visual_contract", "lighting_description",
                "style_anchor", "camera_movement",
                "camera_motion_contract", "interaction_props", "phase8_reshoot",
            )
            if shot_meta.get(key) not in (None, "", [])
        }
        reference_labels = [character_id for character_id, _path in character_references]
        review_paths = [path for _character_id, path in character_references] + frame_paths
        structure_contract = (
            (
                "This project intentionally uses fully synthetic stylized CGI characters. Declared veils/masks, "
                "graphic makeup, facial tattoos, mechanical seams, porcelain/crystalline synthetic surfaces, "
                "designed hair/head silhouettes and other non-human materials are required identity styling, "
                "not human-anatomy defects. A helmet is optional and must never be copied onto every role. Do not "
                "reject a shot merely because a character is veiled, masked, robotic, graphically made up, "
                "faceless, or unlike a normal human. Judge synthetic-character consistency against the canonical "
                "references: part count, attachment continuity, silhouette, styling-anchor geometry, costume, "
                "color blocks, non-human material and identity markers must remain stable, with at least two "
                "declared styling anchors visibly preserved per character whenever framing permits. Reject an "
                "untreated natural human face or visible positive evidence of an "
                "unintended break, detachment, merge, extra/missing part, impossible self-intersection, or "
                "reference-inconsistent deformation. "
            )
            if synthetic_review
            else (
                "Detect broken anatomy and extra or missing limbs. Do not call a hand or limb anatomically "
                "broken merely because it is partly hidden by a sleeve, prop, railing, crop, or camera angle; "
                "require visible positive evidence of malformation. "
            )
        )
        temporal_qa = temporal_visual_qa_instruction(temporal_contract)
        body_action_qa = body_action_qa_instruction(shot_meta)
        prompt = (
            f"QA contract: {qa_contract}. "
            f"The first {len(character_references)} supplied image(s) are canonical character references "
            f"in this exact order: {json.dumps(reference_labels, ensure_ascii=False)}. All remaining images "
            "are ordered frames from one generated video shot. Compare the video frames with those references "
            "and with the expected "
            f"shot metadata: {json.dumps(expected, ensure_ascii=False)}. Detect character identity drift, "
            "extra or missing limbs/objects, impossible geometry, continuity jumps, text or "
            "watermark artifacts, wrong time of day, daylight/night drift, weather drift, and lighting that "
            "contradicts the shot or changes materially between the first and last supplied frame. Natural acting "
            "micro-movements (small gaze/head/hand changes) are allowed unless they reverse the narrative action or "
            "create a true continuity jump. "
            f"{structure_contract}{temporal_qa} {body_action_qa} "
            "Each canonical named identity may appear only once unless the metadata explicitly requests clones; "
            "background extras must not duplicate a canonical identity, costume, styling-anchor combination, or signature marker. "
            "Verify that generation_actions occur in their listed order with visible subject "
            "displacement and a recognizable result; hair, rain, smoke, blinking, or camera drift alone do not "
            "count as completion of body action. Judge shot-size drift only against explicit camera movement: a dolly-in must not become "
            "a sustained pull-back, while a fixed shot may contain minor stabilization drift. Return JSON only: "
            '{"verdict":"pass|reshoot","issues":["..."],"confidence":0.0}.'
        )
        parsed = review_as(
            client,
            review_paths,
            prompt,
            ShotSemanticReview,
        ).model_dump()
        parsed["qa_contract"] = qa_contract
        return parsed

    return review


def analyze_shot_frames(
    shots_dir: Path,
    output_path: Path,
    *,
    semantic_reviewer: SemanticReviewer | bool | None = None,
    max_frames: int = 12,
    interval_s: float = 1.0,
) -> dict[str, Any]:
    """Analyze every generated shot and persist actionable assembly decisions."""
    circuit_open = semantic_reviewer is None and _multimodal_circuit_is_open(output_path)
    if circuit_open:
        reviewer = None
        print(
            "  ⚠ [8.2] 多模态审稿熔断已打开；跳过逐镜头远端语义请求，继续本地逐帧检查",
            flush=True,
        )
    elif semantic_reviewer is None or semantic_reviewer is True:
        reviewer = _automatic_semantic_reviewer(Path(output_path).parent)
    elif semantic_reviewer is False:
        reviewer = None
    else:
        reviewer = semantic_reviewer

    report: dict[str, Any] = {
        "shots": {},
        "has_issues": False,
        "summary": {"keep": [], "trim": [], "reshoot": []},
        "semantic_review": (
            "circuit_open" if circuit_open else "enabled" if reviewer else "unavailable"
        ),
    }
    for shot_dir in sorted(Path(shots_dir).iterdir()) if Path(shots_dir).is_dir() else []:
        video = shot_dir / "output.mp4"
        if not shot_dir.is_dir() or not shot_dir.name.startswith("S") or not video.is_file():
            continue
        meta_path = shot_dir / "SHOT_META.json"
        try:
            shot_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            shot_meta = {}

        try:
            duration = probe_duration(video)
            extracted = extract_review_frames(
                video, shot_dir / "frames", max_frames=max_frames, interval_s=interval_s
            )
            frames: list[dict[str, Any]] = []
            for item in extracted:
                frames.append({
                    "path": str(item["path"].relative_to(Path(shots_dir).parent)),
                    "timestamp_s": item["timestamp_s"],
                    "metrics": _frame_metrics(item["path"]),
                })
            black_segments = _detect_black_segments(video)
            freeze_segments = _detect_freeze_segments(video)
            motion_activity = measure_motion_activity([item["path"] for item in extracted])
            semantic: dict[str, Any] | None = None
            if reviewer:
                try:
                    representative = [item["path"] for item in extracted]
                    if len(representative) > 5:
                        positions = np.linspace(0, len(representative) - 1, 5).astype(int)
                        representative = [representative[index] for index in positions]
                    semantic = reviewer(representative, shot_meta)
                except Exception as exc:
                    semantic = {"verdict": "unavailable", "issues": [], "error": str(exc)}
            decision = decide_shot_action(
                duration, frames, black_segments, freeze_segments, semantic, shot_meta,
                motion_activity,
            )
            entry = {
                "duration_s": round(duration, 4),
                "frames": frames,
                "black_segments": black_segments,
                "freeze_segments": freeze_segments,
                "motion_activity": motion_activity,
                "semantic_review": semantic,
                **decision,
            }
        except Exception as exc:
            entry = {
                "duration_s": 0.0,
                "frames": [],
                "black_segments": [],
                "freeze_segments": [],
                "semantic_review": None,
                "action": "reshoot",
                "reasons": [f"frame analysis failed: {exc}"],
                "trim_start_s": 0.0,
                "trim_end_s": 0.0,
            }

        action = entry["action"]
        report["summary"][action].append(shot_dir.name)
        if action != "keep":
            report["has_issues"] = True
            print(f"  ⚠ [8.2] {shot_dir.name}: {action} — {'; '.join(entry['reasons'])}", flush=True)
        else:
            print(f"  ✓ [8.2] {shot_dir.name}: {len(entry['frames'])} 帧通过", flush=True)
        report["shots"][shot_dir.name] = entry

    report["summary"].update({
        "reviewed_shots": len(report["shots"]),
        "sampled_frames": sum(len(item["frames"]) for item in report["shots"].values()),
    })
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
