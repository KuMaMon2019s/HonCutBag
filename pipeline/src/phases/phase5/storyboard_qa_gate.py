"""Phase 5: pre-generation storyboard quality gate.

Every judgment is derived from project artifacts.  The gate never repairs a
storyboard; it reports the shots that need to be redrawn before Phase 6.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clients.ark_multimodal_client import ArkMultimodalClient
from utils.video_capabilities import capabilities_for

DEFAULT_SIMILARITY_THRESHOLD = 0.85
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

_LIGHT_PERIODS = {
    "night": ("night", "midnight", "moonlight", "夜", "午夜", "月光", "星空"),
    "day": ("daylight", "midday", "noon", "daytime", "日间", "白天", "正午", "午后"),
    "dawn": ("dawn", "sunrise", "morning", "黎明", "清晨", "日出", "晨光"),
    "dusk": ("dusk", "sunset", "evening", "golden hour", "夕阳", "黄昏", "傍晚", "日落", "黄金时段"),
}


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id", shot.get("id", index + 1))
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _periods(text: str) -> set[str]:
    lowered = text.lower()
    return {period for period, terms in _LIGHT_PERIODS.items() if any(term in lowered for term in terms)}


def _issue(layer: str, severity: str, code: str, message: str, shot_ids: list[str] | None = None, **details: Any) -> dict:
    result = {"layer": layer, "severity": severity, "code": code, "message": message, "shot_ids": shot_ids or []}
    if details:
        result["details"] = details
    return result


def run_l1_checks(storyboard: dict, visual_style: str) -> tuple[list[dict], dict[str, dict]]:
    """Check artifact-level lighting, spoken-text fields, and duration."""
    shots = storyboard.get("shots") if isinstance(storyboard.get("shots"), list) else []
    issues: list[dict] = []
    per_shot: dict[str, dict] = {}
    style_periods = _periods(visual_style)
    durations: list[float] = []
    dialogue_fields = ("dialogue", "narration", "voiceover", "voice_over")

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        profile = capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index)
        per_shot[sid] = {"issues": [], "characters": []}
        character_assets = [
            str(value)
            for value in (shot.get("associate_assets") or [])
            if str(value).startswith("char:")
        ]
        if shot.get("who") == [] and character_assets:
            item = _issue(
                "L1",
                "severe",
                "no_character_contract_conflict",
                f"{sid} declares who=[] but binds character assets",
                [sid],
                character_assets=character_assets,
            )
            issues.append(item)
            per_shot[sid]["issues"].append(item)
        lighting = " ".join(_text(shot.get(key)) for key in ("lighting_description", "lighting", "prompt", "description", "name"))
        shot_periods = _periods(lighting)
        if style_periods and shot_periods and style_periods.isdisjoint(shot_periods):
            item = _issue("L1", "severe", "lighting_period_mismatch", f"{sid} lighting period conflicts with visual-style.md", [sid], expected=sorted(style_periods), observed=sorted(shot_periods))
            issues.append(item)
            per_shot[sid]["issues"].append(item)

        duration = shot.get("duration", shot.get("duration_seconds"))
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            item = _issue("L1", "moderate", "invalid_duration", f"{sid} has no positive numeric duration", [sid])
            issues.append(item)
            per_shot[sid]["issues"].append(item)
        else:
            durations.append(float(duration))

        present = [
            key
            for key in dialogue_fields
            if key in shot and shot.get(key) is not None
        ]
        if present and all(not _text(shot.get(key)).strip() for key in present):
            item = _issue("L1", "moderate", "empty_spoken_content", f"{sid} declares spoken-content fields but all are empty", [sid])
            issues.append(item)
            per_shot[sid]["issues"].append(item)

    target = storyboard.get("target_duration", storyboard.get("duration"))
    if isinstance(target, (int, float)) and target > 0 and shots:
        actual = sum(durations)
        tolerance = max(1.0, float(target) * 0.05)
        if abs(actual - float(target)) > tolerance:
            issues.append(_issue("L1", "moderate", "duration_budget_mismatch", f"Storyboard duration {actual:g}s differs from target {float(target):g}s", details={"actual_seconds": actual, "target_seconds": float(target), "tolerance_seconds": tolerance}))
    return issues, per_shot


def run_generation_capacity_checks(
    storyboard: dict,
    events_data: dict | None = None,
) -> list[dict]:
    """Block storyboards that exceed one video clip's narrative capacity."""
    issues: list[dict] = []
    observed_units: set[str] = set()
    for index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        profile = capabilities_for({**storyboard, **shot})
        sid = _shot_id(shot, index)
        raw_units = shot.get("source_action_unit_ids") or []
        if isinstance(raw_units, str):
            raw_units = [raw_units]
        units = {str(value) for value in raw_units if str(value).strip()}
        observed_units.update(units)
        generation_actions = shot.get("generation_actions") or []
        if isinstance(generation_actions, str):
            generation_actions = [generation_actions]
        duration = float(shot.get("duration") or shot.get("suggested_duration") or 0)
        camera = str(
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or ""
        ).lower()
        storyboard_beats = [
            beat
            for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]

        if storyboard_beats:
            beat_duration_total = 0.0
            beat_units_seen: list[str] = []
            for position, beat in enumerate(storyboard_beats, 1):
                beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
                beat_duration = float(beat.get("duration_s") or 0)
                beat_duration_total += beat_duration
                expected_mode = (
                    "extend"
                    if position > 1
                    or (
                        position == 1
                        and str(shot.get("boundary_before") or "")
                        .strip()
                        .lower()
                        == "continuous"
                    )
                    else "fresh"
                )
                if beat.get("generation_mode") != expected_mode:
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_mode_invalid",
                        f"{beat_id} must use {expected_mode}",
                        [sid], beat_id=beat_id, expected_mode=expected_mode,
                    ))
                if not str(beat.get("action") or "").strip():
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_action_missing",
                        f"{beat_id} has no executable action contract",
                        [sid], beat_id=beat_id,
                    ))
                beat_units = beat.get("source_action_unit_ids") or []
                if isinstance(beat_units, str):
                    beat_units = [beat_units]
                beat_units = [str(value) for value in beat_units if str(value).strip()]
                beat_units_seen.extend(beat_units)
                if len(set(beat_units)) > profile.max_action_units_per_beat:
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_action_unit_overload",
                        f"{beat_id} exceeds {profile.name}'s action-unit capacity "
                        f"({profile.max_action_units_per_beat})",
                        [sid], beat_id=beat_id, action_unit_ids=beat_units,
                    ))
                micro_actions = beat.get("micro_actions") or []
                if isinstance(micro_actions, str):
                    micro_actions = [micro_actions]
                if (
                    len([value for value in micro_actions if str(value).strip()])
                    > profile.max_micro_actions_per_beat
                ):
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_action_overload",
                        f"{beat_id} exceeds {profile.name}'s visible-action capacity "
                        f"({profile.max_micro_actions_per_beat})",
                        [sid], beat_id=beat_id,
                    ))
                if (
                    beat_duration < profile.min_unique_beat_s
                    or beat_duration > profile.max_unique_beat_s
                ):
                    issues.append(_issue(
                        "L1", "severe", "storyboard_beat_duration_invalid",
                        f"{beat_id} lasts {beat_duration:g}s; expected "
                        f"{profile.min_unique_beat_s:g}-{profile.max_unique_beat_s:g}s "
                        f"for {profile.name}",
                        [sid], beat_id=beat_id, duration_seconds=beat_duration,
                    ))
            if duration and not math.isclose(
                beat_duration_total,
                duration,
                abs_tol=0.05,
            ):
                issues.append(_issue(
                    "L1", "severe", "storyboard_beat_duration_mismatch",
                    f"{sid} internal beats total {beat_duration_total:g}s, "
                    f"expected {duration:g}s",
                    [sid], beat_duration_seconds=beat_duration_total,
                    shot_duration_seconds=duration,
                ))
            if units and set(beat_units_seen) != units:
                issues.append(_issue(
                    "L1", "severe", "storyboard_beat_action_unit_coverage_mismatch",
                    f"{sid} Pxx action-unit coverage differs from the director shot",
                    [sid], expected_action_unit_ids=sorted(units),
                    observed_action_unit_ids=sorted(set(beat_units_seen)),
                ))

        if len(units) > profile.max_action_units_per_beat and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "action_unit_overload",
                f"{sid} contains {len(units)} action units; {profile.name} supports "
                f"{profile.max_action_units_per_beat}",
                [sid], action_unit_ids=sorted(units),
            ))
        if units and not generation_actions and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "missing_generation_actions",
                f"{sid} has screenplay action but no bounded generation action contract",
                [sid],
            ))
        action_limit = profile.action_limit(duration)
        if len(generation_actions) > action_limit and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "generation_action_overload",
                f"{sid} asks the video model to perform {len(generation_actions)} actions "
                f"(max {action_limit} for {duration:g}s)",
                [sid], prompted_actions=len(generation_actions), action_limit=action_limit,
            ))
        if units and duration > profile.max_unique_beat_s and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "action_shot_too_long",
                f"{sid} action shot lasts {duration:g}s; split within "
                f"{profile.name}'s {profile.min_unique_beat_s:g}-"
                f"{profile.max_unique_beat_s:g}s beat range",
                [sid], duration_seconds=duration,
            ))
        if units and camera in {"static", "fixed", "locked", "unspecified", ""} and not storyboard_beats:
            issues.append(_issue(
                "L1", "severe", "static_action_camera",
                f"{sid} action shot uses a locked/static camera contract",
                [sid], camera_movement=camera or "missing",
            ))
        if (
            not units
            and duration > profile.max_unique_beat_s
            and camera in {"static", "fixed", "locked", "unspecified", ""}
            and not storyboard_beats
        ):
            issues.append(_issue(
                "L1", "severe", "static_hold_risk",
                f"{sid} holds a static composition for {duration:g}s",
                [sid], duration_seconds=duration,
            ))

    expected_units = {
        str(event.get("action_unit_id"))
        for event in (events_data or {}).get("events", [])
        if isinstance(event, dict) and str(event.get("action_unit_id") or "").strip()
    }
    missing_units = sorted(expected_units - observed_units)
    if missing_units:
        issues.append(_issue(
            "L1", "severe", "action_unit_coverage_missing",
            f"Storyboard drops {len(missing_units)} screenplay action unit(s)",
            [], missing_action_unit_ids=missing_units,
        ))
    return issues


def _characters_in_shot(shot: dict, characters: list[dict]) -> list[str]:
    explicit = shot.get("character_ids", shot.get("characters", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    found = {str(value) for value in explicit if value} if isinstance(explicit, list) else set()
    haystack = _text(shot).casefold()
    for character in characters:
        cid = str(character.get("id", ""))
        names = [cid, str(character.get("name", "")), *[str(x) for x in character.get("aliases", []) if x]]
        if cid and any(name and name.casefold() in haystack for name in names):
            found.add(cid)
    return sorted(found)


def find_storyboard_images(output_dir: Path, storyboard: dict) -> dict[str, Path]:
    image_dir = output_dir / "storyboard_images"
    paths = list(image_dir.iterdir()) if image_dir.is_dir() else []
    result: dict[str, Path] = {}
    for index, shot in enumerate(storyboard.get("shots", [])):
        sid = _shot_id(shot, index)
        for path in paths:
            if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.upper() == sid.upper():
                result[sid] = path
                break
    return result


def find_storyboard_beat_images(output_dir: Path, storyboard: dict) -> dict[str, Path]:
    """Resolve every exact Pxx image; callers can detect omissions by count."""
    result: dict[str, Path] = {}
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        sid = _shot_id(shot, shot_index)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{sid}_P{position:02d}")
            value = str(beat.get("storyboard_image") or "").strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file() and path.stat().st_size > 1024:
                result[beat_id] = path
    return result


def _parent_shot_id(image_id: str) -> str:
    match = re.match(r"^(.*)_P\d+$", image_id, flags=re.IGNORECASE)
    return match.group(1) if match else image_id


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    if norm == 0:
        raise ValueError("embedding vectors must have non-zero norms")
    return dot / norm


def run_l2_checks(storyboard: dict, characters_data: dict, images: dict[str, Path], threshold: float = DEFAULT_SIMILARITY_THRESHOLD, embedder: Callable[[str], list[float] | None] | None = None) -> tuple[list[dict], dict]:
    """Embed whole frames as scene diagnostics, never as character crops.

    A whole-frame embedding cannot isolate a particular person.  Older code
    copied the same frame matrix under every character ID and treated it as
    identity evidence.  L2 now reports one honest scene-level matrix; canonical
    character-reference comparison is delegated to the multimodal L3 review.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("similarity threshold must be between 0 and 1")
    if embedder is None:
        from utils.shot_embedder import embed_image
        embedder = embed_image
    vectors: dict[str, list[float]] = {}
    errors: list[str] = []
    for sid, path in images.items():
        try:
            vector = embedder(str(path))
            if vector:
                vectors[sid] = [float(value) for value in vector]
        except Exception as exc:  # preserve the failure in the report
            errors.append(f"{sid}: {exc}")

    image_ids = [image_id for image_id in images if image_id in vectors]
    matrix: list[list[float]] = []
    for left in image_ids:
        row: list[float] = []
        for right in image_ids:
            try:
                score = cosine_similarity(vectors[left], vectors[right])
            except ValueError as exc:
                errors.append(f"{left}/{right}: {exc}")
                score = 0.0
            row.append(round(score, 6))
        matrix.append(row)
    status = "completed" if vectors else "skipped"
    reason = None if vectors else ("ARK_AGENT_API_KEY missing or embedding service returned no vectors")
    return [], {
        "status": status,
        "skipped_reason": reason,
        "scope": "whole_frame_scene_consistency",
        "character_isolation": False,
        "identity_review_layer": "L3_canonical_references",
        "threshold": threshold,
        "embedded_shots": sorted(vectors),
        "scene_matrix": {"storyboard_ids": image_ids, "matrix": matrix},
        "errors": errors,
    }


def create_storyboard_grid(image_paths: list[Path], output_path: Path, columns: int = 5) -> Path:
    """Create a labelled, artifact-derived contact sheet using Pillow."""
    if not image_paths:
        raise ValueError("at least one storyboard image is required")
    from PIL import Image, ImageDraw, ImageFont
    opened = [Image.open(path).convert("RGB") for path in image_paths]
    width = min(480, max(image.width for image in opened))
    height = max(1, int(max(image.height / image.width for image in opened) * width))
    rows = math.ceil(len(opened) / columns)
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(grid)
    font_size = max(28, width // 10)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # Pillow < 10
            font = ImageFont.load_default()
    for index, (path, source) in enumerate(zip(image_paths, opened, strict=True)):
        source.thumbnail((width, height))
        x = (index % columns) * width + (width - source.width) // 2
        y = (index // columns) * height
        grid.paste(source, (x, y))
        # Put a large high-contrast ID inside each frame. Small captions below
        # a dense contact sheet caused the VLM to associate S15 with S11 and
        # neighbouring final shots with S24 in a real 24-shot run.
        label = path.stem.upper()
        label_x = (index % columns) * width + 10
        label_y = y + 10
        bbox = draw.textbbox((label_x, label_y), label, font=font, stroke_width=1)
        padding = 8
        draw.rounded_rectangle(
            (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ),
            radius=6,
            fill="black",
        )
        draw.text(
            (label_x, label_y),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def _ordered_storyboard_images(
    storyboard: dict, images: dict[str, Path]
) -> list[Path]:
    """Order beat-level images by shot and beat, falling back to shot images."""
    ordered: list[Path] = []
    for shot_index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        beat_paths: list[Path] = []
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
            if beat_id in images:
                beat_paths.append(images[beat_id])
        if beat_paths:
            ordered.extend(beat_paths)
        elif shot_id in images:
            ordered.append(images[shot_id])
    return ordered


def find_character_reference_images(
    output_dir: Path,
    characters_data: dict,
) -> dict[str, list[Path]]:
    """Resolve canonical Phase 3 references in stable identity-first order."""
    output_dir = Path(output_dir)
    result: dict[str, list[Path]] = {}
    for character in characters_data.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id") or "").strip()
        if not character_id:
            continue
        character_dir = output_dir / "characters" / character_id
        card_path = character_dir / "character_card.json"
        declared: dict[str, Any] = {}
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(card.get("reference_images"), dict):
                declared = card["reference_images"]
        except (OSError, json.JSONDecodeError):
            pass
        paths: list[Path] = []
        for view_name in (
            "face_closeup",
            "full_body",
            "closeup",
            "front",
            "side",
            "back",
            "three_quarter",
            "detail",
        ):
            value = declared.get(view_name)
            path = Path(str(value)) if value else character_dir / f"{view_name}.png"
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file() and path.stat().st_size > 0 and path not in paths:
                paths.append(path)
        if paths:
            result[character_id] = paths
    return result


def _calibrate_l3_severity(red_line: str, severity: str, message: str) -> str:
    """Keep L3 blocking for production-breaking mismatches, not pose minutiae."""
    if severity != "severe":
        return severity
    normalized_line = red_line.upper()
    text = message.casefold()
    # Match explicit mismatch assertions, not isolated descriptor words. The
    # former terms "male" and "female" made any sentence beginning with
    # "the female character..." a hard blocker even when no gender mismatch
    # was alleged. Likewise, a VLM's unsupported celebrity resemblance claim
    # is not evidence of a wrong character identity.
    hard_blockers = (
        "wrong identity",
        "different identity",
        "identity mismatch",
        "wrong gender",
        "different gender",
        "gender mismatch",
        "male instead of",
        "female instead of",
        "missing character",
        "wrong character",
        "reversed attacker",
        "wrong location",
        "reset",
        "replay",
        "replays prior",
        "wholly unrelated",
        "unrelated core action",
        "身份错误",
        "身份不一致",
        "性别错误",
        "性别不一致",
        "男性而非",
        "女性而非",
        "角色缺失",
        "人物错误",
        "攻守颠倒",
        "场景错误",
        "重置",
        "重放",
        "重复前格",
        "回到初始",
        "核心动作缺失",
    )
    if normalized_line == "R1" and not any(term in text for term in hard_blockers):
        return "moderate"
    if normalized_line in {"R3", "R4"} and not any(
        term in text for term in hard_blockers
    ):
        return "moderate"
    return severity


_R1_ATTRIBUTE_TERMS = (
    "color",
    "colour",
    "clothing",
    "costume",
    "uniform",
    "outfit",
    "颜色",
    "服装",
    "制服",
    "衣服",
    "穿着",
)


def _normalized_visual_attribute(value: Any) -> str:
    """Normalize concise expected/observed R1 attribute claims."""
    text = str(value or "").casefold()
    aliases = {
        "dark grey": "darkgray",
        "dark gray": "darkgray",
        "深灰色": "深灰",
        "navy blue": "navy",
        "dark blue": "navy",
        "藏蓝色": "藏蓝",
    }
    for source, target in aliases.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def _r1_attribute_evidence(
    value: dict[str, Any],
    storyboard_ids: list[str],
    reference_count: int,
) -> tuple[bool, dict[str, Any]]:
    """Validate evidence for clothing/color continuity claims.

    R1 identity and gender findings remain governed by the severe mismatch
    calibration. Attribute drift is more vulnerable to lighting/style
    hallucinations, so it needs explicit canonical and per-panel evidence
    before it can block paid generation.
    """
    mismatch_type = str(value.get("mismatch_type") or "").casefold()
    message = str(value.get("message") or "").casefold()
    is_attribute_claim = mismatch_type in {
        "clothing_color",
        "clothing",
        "costume",
        "uniform",
        "color",
        "colour",
    } or any(term in message for term in _R1_ATTRIBUTE_TERMS)
    if not is_attribute_claim:
        return True, {"evidence_status": "not_attribute_claim"}

    expected = str(value.get("expected") or "").strip()
    observed = str(value.get("observed") or "").strip()
    raw_reference_indices = value.get("reference_input_indices") or []
    if not isinstance(raw_reference_indices, list):
        raw_reference_indices = [raw_reference_indices]
    reference_indices = sorted({
        int(index)
        for index in raw_reference_indices
        if str(index).isdigit() and 1 <= int(index) <= reference_count
    })
    raw_panel_evidence = value.get("panel_evidence") or []
    if not isinstance(raw_panel_evidence, list):
        raw_panel_evidence = []
    evidence_ids = {
        str(item.get("shot_id") or "")
        for item in raw_panel_evidence
        if isinstance(item, dict)
        and str(item.get("observed") or "").strip()
    }
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    expected_normalized = _normalized_visual_attribute(expected)
    observed_normalized = _normalized_visual_attribute(observed)
    reasons = []
    if not expected_normalized or not observed_normalized:
        reasons.append("missing_expected_or_observed")
    elif expected_normalized == observed_normalized:
        reasons.append("expected_equals_observed")
    if not reference_indices:
        reasons.append("missing_canonical_reference")
    if not storyboard_ids or not set(storyboard_ids).issubset(evidence_ids):
        reasons.append("missing_per_panel_evidence")
    if not math.isfinite(confidence) or confidence < 0.75 or confidence > 1.0:
        reasons.append("confidence_below_0.75")
    return not reasons, {
        "evidence_status": "validated" if not reasons else "unverified",
        "evidence_reasons": reasons,
        "mismatch_type": mismatch_type or "unspecified_attribute",
        "expected": expected,
        "observed": observed,
        "reference_input_indices": reference_indices,
        "confidence": confidence,
    }


def run_l3_review(
    storyboard: dict,
    characters_data: dict,
    visual_style: str,
    images: dict[str, Path],
    grid_path: Path,
    client: ArkMultimodalClient | None = None,
    *,
    character_reference_images: dict[str, list[Path]] | None = None,
) -> tuple[list[dict], dict]:
    if not images:
        return [], {"status": "skipped", "skipped_reason": "no storyboard images available"}
    ordered = _ordered_storyboard_images(storyboard, images)
    if not ordered:
        return [], {
            "status": "skipped",
            "skipped_reason": "no storyboard images match storyboard IDs",
        }
    create_storyboard_grid(ordered, grid_path)
    if client is None and not (os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("ARK_API_KEY")):
        return [], {"status": "skipped", "grid_path": str(grid_path), "skipped_reason": "ARK multimodal API key missing"}
    valid_storyboard_ids = list(images)
    reference_inputs: list[Path] = []
    reference_manifest: list[dict[str, Any]] = []
    references = character_reference_images or {}
    ordered_character_ids = [
        str(character.get("id") or "")
        for character in characters_data.get("characters", [])
        if isinstance(character, dict) and str(character.get("id") or "")
    ]
    ordered_character_ids.extend(
        sorted(set(references) - set(ordered_character_ids))
    )
    for character_id in ordered_character_ids:
        for path in references.get(character_id, []):
            path = Path(path)
            if not path.is_file() or path in reference_inputs:
                continue
            reference_inputs.append(path)
            reference_manifest.append({
                "input_index": len(reference_inputs),
                "character_id": character_id,
                "view": path.stem,
            })
    grid_input_index = len(reference_inputs) + 1
    prompt = f"""Review the final input image, which is the storyboard grid, against the supplied project artifacts and canonical character reference images. Inputs 1 through {len(reference_inputs)} are character references described by REFERENCE INPUTS below; input {grid_input_index} is the storyboard grid. The grid has exactly 5 columns in row-major order, and every frame has its exact Sxx or Sxx_Pxx ID in a large black badge at the top-left. Associate observations only with that in-frame badge; never infer an ID from a neighbouring cell or row position.

Apply red lines R1-R4: R1 character identity/gender/build/clothing continuity against the canonical references; R2 time-of-day and lighting continuity; R3 scene/action continuity; R4 storyboard-to-image semantic fidelity. Do not perform face recognition or infer a public identity from appearance alone. Each Pxx image represents only its own authored action and must progress from the previous Pxx without pose reset or premature future action.

Evidence rules:
- Report only visible contradictions. Absence of proof is not proof of mismatch. Do not infer clothing-color drift from red warning light, shadow, monochrome PREVIS rendering, highlights, or low saturation.
- For every R1 clothing/color/uniform claim, set mismatch_type="clothing_color" and provide: canonical reference input indices, a concise expected attribute, a concise observed attribute, confidence from 0 to 1, and separate panel_evidence for every listed ID. Expected and observed must name genuinely different visible attributes. If both are dark gray, both are navy, or the difference is only illumination/style, emit no issue.
- Never copy one panel observation across a range. List multiple IDs only after independently checking each badge; panel_evidence must contain one observation per listed ID.
- For R4, compare the literal actor → action → target → prop ownership → end state. A mutual weapon-disarm action is not satisfied by one actor aiming a weapon. Do not reverse attacker/defender or weapon ownership.
- For the final Pxx, verify the authored result is visibly complete. If the contract says stable/stopped/freeze-frame while another character flies toward or hits a target, ongoing mutual fighting or generic floating is a mismatch.

Reserve severe for production-breaking mismatches: wrong/missing character identity or gender, wrong location/time-of-day, reversed attacker/defender, wholly unrelated core action, or a continuation panel that visibly resets/replays the prior state. Use moderate for material but recoverable semantic mismatches, exact prop/action-state errors, blocking offsets, or intermediate/final-state omissions. Only identify problems; do not propose or perform edits.

Return JSON only: {{"issues":[{{"red_line":"R1|R2|R3|R4","severity":"severe|moderate|minor","mismatch_type":"identity|gender|clothing_color|lighting|action|end_state|other","shot_ids":["S01_P01"],"message":"...","reference_input_indices":[1],"expected":"canonical concise fact","observed":"visibly different concise fact","confidence":0.90,"panel_evidence":[{{"shot_id":"S01_P01","observed":"specific visible evidence in this panel"}}]}}]}}. Use only these exact IDs: {json.dumps(valid_storyboard_ids, ensure_ascii=False)}.
REFERENCE INPUTS:
{json.dumps(reference_manifest, ensure_ascii=False)}
STORYBOARD:
{json.dumps(storyboard, ensure_ascii=False)}
CHARACTERS:
{json.dumps(characters_data, ensure_ascii=False)}
VISUAL STYLE:
{visual_style}"""
    try:
        raw = (client or ArkMultimodalClient()).review(
            [*reference_inputs, grid_path], prompt
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("issues"), list):
            raise ValueError("response must contain an issues array")
        valid_ids = set(images)
        issues = []
        for value in parsed["issues"]:
            if not isinstance(value, dict):
                continue
            red_line = str(value.get("red_line", "semantic_review"))
            message = str(value.get("message", "Multimodal review issue"))
            requested_severity = value.get("severity") if value.get("severity") in {"severe", "moderate", "minor"} else "moderate"
            severity = _calibrate_l3_severity(
                red_line, requested_severity, message
            )
            storyboard_ids = [sid for sid in value.get("shot_ids", []) if sid in valid_ids]
            evidence_valid, evidence_details = _r1_attribute_evidence(
                value,
                storyboard_ids,
                len(reference_inputs),
            ) if red_line.upper() == "R1" else (True, {"evidence_status": "not_required"})
            if not evidence_valid:
                severity = "minor"
            shot_ids = sorted({_parent_shot_id(sid) for sid in storyboard_ids})
            issues.append(_issue(
                "L3", severity, red_line, message, shot_ids,
                storyboard_ids=storyboard_ids,
                **evidence_details,
            ))
        return issues, {"status": "completed", "grid_path": str(grid_path), "raw_issue_count": len(parsed["issues"])}
    except Exception as exc:
        return [], {"status": "skipped", "grid_path": str(grid_path), "skipped_reason": f"multimodal review unavailable: {exc}"}


def is_blocking_issue(issue: dict) -> bool:
    """Return whether an issue must stop paid video generation."""
    if issue.get("severity") == "severe":
        return True
    return (
        issue.get("severity") == "moderate"
        and issue.get("layer") == "L3"
        and str(issue.get("code") or "").upper() in {"R1", "R2", "R3", "R4"}
        and len(set(issue.get("shot_ids") or [])) >= 2
    )


def blocking_issues(issues: list[dict]) -> list[dict]:
    """Include individually severe and collectively systemic L3 findings."""
    blocking_ids = {
        id(issue) for issue in issues if is_blocking_issue(issue)
    }
    moderate_groups: dict[tuple[str, str], list[dict]] = {}
    for issue in issues:
        if (
            issue.get("severity") == "moderate"
            and issue.get("layer") == "L3"
            and str(issue.get("code") or "").upper() in {"R1", "R2", "R3", "R4"}
        ):
            key = ("L3", str(issue.get("code") or "").upper())
            moderate_groups.setdefault(key, []).append(issue)
    for grouped in moderate_groups.values():
        affected_shots = {
            shot_id
            for issue in grouped
            for shot_id in issue.get("shot_ids") or []
        }
        if len(affected_shots) >= 2:
            blocking_ids.update(id(issue) for issue in grouped)
    return [issue for issue in issues if id(issue) in blocking_ids]


def grade_issues(issues: list[dict]) -> str:
    blocking = blocking_issues(issues)
    severe = sum(issue.get("severity") == "severe" for issue in blocking)
    moderate = sum(issue.get("severity") == "moderate" for issue in issues)
    if severe >= 3 or len(blocking) >= 3:
        return "D"
    if blocking:
        return "C"
    if moderate > 2:
        return "B"
    return "A"


def run_storyboard_qa_gate(output_dir: Path, similarity_threshold: float | None = None, embedder: Callable[[str], list[float] | None] | None = None, multimodal_client: ArkMultimodalClient | None = None) -> dict:
    """Run all QA layers, persist the report, and return a phase result."""
    output_dir = Path(output_dir)
    report_path = output_dir / "storyboard_qa_report.json"
    try:
        storyboard = json.loads((output_dir / "STORYBOARD.json").read_text(encoding="utf-8"))
        characters_path = output_dir / "CHARACTERS.json"
        characters = json.loads(characters_path.read_text(encoding="utf-8")) if characters_path.is_file() else {"characters": []}
        style_path = output_dir / "visual-style.md"
        visual_style = style_path.read_text(encoding="utf-8") if style_path.is_file() else ""
        events_path = output_dir / "phase1_events.json"
        events_data = (
            json.loads(events_path.read_text(encoding="utf-8"))
            if events_path.is_file()
            else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "error", "grade": "D", "gate_passed": False, "error": f"required artifact unreadable: {exc}", "issues": [_issue("L1", "severe", "artifact_unreadable", str(exc))], "failed_shot_ids": []}
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    beat_images = find_storyboard_beat_images(output_dir, storyboard)
    expected_beats = [
        str(beat.get("beat_id") or f"{_shot_id(shot, shot_index)}_P{position:02d}")
        for shot_index, shot in enumerate(storyboard.get("shots", []))
        if isinstance(shot, dict)
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1)
        if isinstance(beat, dict)
    ]
    images = beat_images or find_storyboard_images(output_dir, storyboard)
    l1_issues, per_shot = run_l1_checks(storyboard, visual_style)
    # These checks inspect only storyboard metadata, so they belong before the
    # paid video boundary. Running them in Phase 7 used to discover an
    # unfixable storyboard defect only after Phase 6 had spent quota.
    from quality.slideshow_risk import score_slideshow_risk
    from quality.variation_checker import check_scene_variation

    scenes = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    variation = check_scene_variation(scenes)
    slideshow = score_slideshow_risk(scenes)
    variation_quality = round(5.0 - float(variation.get("score", 5.0)), 2)
    slideshow_risk = round(float(slideshow.get("average", 5.0)) / 5.0, 3)
    (output_dir / "variation_report.json").write_text(
        json.dumps(variation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "slideshow_risk_report.json").write_text(
        json.dumps(slideshow, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    structural_issues = []
    if variation_quality < 3.0:
        structural_issues.append(
            _issue(
                "L1",
                "severe",
                "scene_variation_insufficient",
                f"Storyboard variation quality {variation_quality:g}/5 requires revision",
                details={"violations": variation.get("violations", [])},
            )
        )
    if slideshow_risk > 0.7:
        structural_issues.append(
            _issue(
                "L1",
                "severe",
                "slideshow_risk_high",
                f"Storyboard slideshow risk {slideshow_risk:.3f} exceeds 0.7",
                details={"dimensions": slideshow.get("dimensions", {})},
            )
        )
    threshold = similarity_threshold if similarity_threshold is not None else float(os.environ.get("HONCUT_STORYBOARD_QA_SIMILARITY", DEFAULT_SIMILARITY_THRESHOLD))
    l2_issues, l2 = run_l2_checks(storyboard, characters, images, threshold, embedder)
    character_reference_images = find_character_reference_images(
        output_dir,
        characters,
    )
    l3_issues, l3 = run_l3_review(
        storyboard,
        characters,
        visual_style,
        images,
        output_dir / "storyboard_qa_grid.jpg",
        multimodal_client,
        character_reference_images=character_reference_images,
    )
    capacity_issues = run_generation_capacity_checks(storyboard, events_data)
    artifact_issues = [
        _issue(
            "L1", "severe", "storyboard_beat_image_missing",
            f"{beat_id} has no valid Phase 2 storyboard image",
            [_parent_shot_id(beat_id)], beat_id=beat_id,
        )
        for beat_id in expected_beats
        if beat_id not in beat_images
    ]
    issues = l1_issues + structural_issues + artifact_issues + capacity_issues + l2_issues + l3_issues
    for index, shot in enumerate(storyboard.get("shots", [])):
        sid = _shot_id(shot, index)
        detail = per_shot.setdefault(sid, {"issues": []})
        detail["characters"] = _characters_in_shot(shot, characters.get("characters", []))
        shot_beat_images = {
            image_id: str(path)
            for image_id, path in beat_images.items()
            if _parent_shot_id(image_id) == sid
        }
        detail["image_path"] = str(images[sid]) if sid in images else None
        detail["storyboard_beat_images"] = shot_beat_images
        detail["issues"] = [issue for issue in issues if sid in issue.get("shot_ids", [])]
    grade = grade_issues(issues)
    failed_shots = sorted({
        sid
        for issue in blocking_issues(issues)
        for sid in issue.get("shot_ids", [])
    })
    report = {"status": "done" if grade in {"A", "B"} else "error", "grade": grade, "gate_passed": grade in {"A", "B"}, "issues": issues, "issue_counts": {severity: sum(item.get("severity") == severity for item in issues) for severity in ("severe", "moderate", "minor")}, "failed_shot_ids": failed_shots, "shots": per_shot, "variation_score": variation_quality, "slideshow_risk": slideshow_risk, "layers": {"L1": {"status": "completed"}, "L2": l2, "L3": l3}, "outputs": ["storyboard_qa_report.json", "variation_report.json", "slideshow_risk_report.json", *( ["storyboard_qa_grid.jpg"] if grid_path_exists(output_dir) else [])]}
    if not report["gate_passed"]:
        report["error"] = f"Storyboard QA grade {grade} blocks Phase 6; redraw only failed_shot_ids"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def grid_path_exists(output_dir: Path) -> bool:
    return (output_dir / "storyboard_qa_grid.jpg").is_file()
