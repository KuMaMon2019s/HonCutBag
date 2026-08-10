"""Phase 4.5: pre-generation storyboard quality gate.

Every judgment is derived from project artifacts.  The gate never repairs a
storyboard; it reports the shots that need to be redrawn before Phase 5.
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
        sid = _shot_id(shot, index)
        per_shot[sid] = {"issues": [], "characters": []}
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

        present = [key for key in dialogue_fields if key in shot]
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    if norm == 0:
        raise ValueError("embedding vectors must have non-zero norms")
    return dot / norm


def run_l2_checks(storyboard: dict, characters_data: dict, images: dict[str, Path], threshold: float = DEFAULT_SIMILARITY_THRESHOLD, embedder: Callable[[str], list[float] | None] | None = None) -> tuple[list[dict], dict]:
    """Embed shot images and compare every pair containing the same character."""
    if not 0 <= threshold <= 1:
        raise ValueError("similarity threshold must be between 0 and 1")
    if embedder is None:
        from utils.shot_embedder import embed_image
        embedder = embed_image
    characters = characters_data.get("characters", []) if isinstance(characters_data, dict) else []
    shot_characters: dict[str, list[str]] = {}
    for index, shot in enumerate(storyboard.get("shots", [])):
        shot_characters[_shot_id(shot, index)] = _characters_in_shot(shot, characters)
    vectors: dict[str, list[float]] = {}
    errors: list[str] = []
    for sid, path in images.items():
        try:
            vector = embedder(str(path))
            if vector:
                vectors[sid] = [float(value) for value in vector]
        except Exception as exc:  # preserve the failure in the report
            errors.append(f"{sid}: {exc}")

    issues: list[dict] = []
    matrices: dict[str, dict] = {}
    for character in characters:
        cid = str(character.get("id", ""))
        shot_ids = [sid for sid, ids in shot_characters.items() if cid in ids and sid in vectors]
        matrix: list[list[float]] = []
        for left in shot_ids:
            row = []
            for right in shot_ids:
                try:
                    score = cosine_similarity(vectors[left], vectors[right])
                except ValueError as exc:
                    errors.append(f"{left}/{right}: {exc}")
                    score = 0.0
                row.append(round(score, 6))
            matrix.append(row)
        matrices[cid] = {"shot_ids": shot_ids, "matrix": matrix}
        for i, left in enumerate(shot_ids):
            for j in range(i + 1, len(shot_ids)):
                score = matrix[i][j]
                if score < threshold:
                    issues.append(_issue("L2", "severe", "character_similarity_low", f"{cid} differs between {left} and {shot_ids[j]} (cosine={score:.3f})", [left, shot_ids[j]], character_id=cid, similarity=score, threshold=threshold))
    status = "completed" if vectors else "skipped"
    reason = None if vectors else ("ARK_AGENT_API_KEY missing or embedding service returned no vectors")
    return issues, {"status": status, "skipped_reason": reason, "threshold": threshold, "embedded_shots": sorted(vectors), "character_matrices": matrices, "errors": errors}


def create_storyboard_grid(image_paths: list[Path], output_path: Path, columns: int = 5) -> Path:
    """Create a labelled, artifact-derived contact sheet using Pillow."""
    if not image_paths:
        raise ValueError("at least one storyboard image is required")
    from PIL import Image, ImageDraw
    opened = [Image.open(path).convert("RGB") for path in image_paths]
    width = min(480, max(image.width for image in opened))
    height = max(1, int(max(image.height / image.width for image in opened) * width))
    label_height = 28
    rows = math.ceil(len(opened) / columns)
    grid = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    for index, (path, source) in enumerate(zip(image_paths, opened, strict=True)):
        source.thumbnail((width, height))
        x = (index % columns) * width + (width - source.width) // 2
        y = (index // columns) * (height + label_height)
        grid.paste(source, (x, y))
        draw.text(((index % columns) * width + 6, y + height + 6), path.stem, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def run_l3_review(storyboard: dict, characters_data: dict, visual_style: str, images: dict[str, Path], grid_path: Path, client: ArkMultimodalClient | None = None) -> tuple[list[dict], dict]:
    if not images:
        return [], {"status": "skipped", "skipped_reason": "no storyboard images available"}
    ordered = [images[_shot_id(shot, i)] for i, shot in enumerate(storyboard.get("shots", [])) if _shot_id(shot, i) in images]
    create_storyboard_grid(ordered, grid_path)
    if client is None and not (os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("ARK_API_KEY")):
        return [], {"status": "skipped", "grid_path": str(grid_path), "skipped_reason": "ARK multimodal API key missing"}
    prompt = f"""Review this storyboard grid against the supplied project artifacts. Apply red lines R1-R4: R1 character identity/gender/build continuity; R2 time-of-day and lighting continuity; R3 scene/action continuity; R4 storyboard-to-image semantic fidelity. Only identify problems; do not propose or perform edits. Return JSON: {{"issues":[{{"red_line":"R1","severity":"severe|moderate|minor","shot_ids":["S01"],"message":"..."}}]}}. Do not invent shot IDs.
STORYBOARD:
{json.dumps(storyboard, ensure_ascii=False)}
CHARACTERS:
{json.dumps(characters_data, ensure_ascii=False)}
VISUAL STYLE:
{visual_style}"""
    try:
        raw = (client or ArkMultimodalClient()).review([grid_path], prompt)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("issues"), list):
            raise ValueError("response must contain an issues array")
        valid_ids = set(images)
        issues = []
        for value in parsed["issues"]:
            if not isinstance(value, dict):
                continue
            severity = value.get("severity") if value.get("severity") in {"severe", "moderate", "minor"} else "moderate"
            shot_ids = [sid for sid in value.get("shot_ids", []) if sid in valid_ids]
            issues.append(_issue("L3", severity, str(value.get("red_line", "semantic_review")), str(value.get("message", "Multimodal review issue")), shot_ids))
        return issues, {"status": "completed", "grid_path": str(grid_path), "raw_issue_count": len(parsed["issues"])}
    except Exception as exc:
        return [], {"status": "skipped", "grid_path": str(grid_path), "skipped_reason": f"multimodal review unavailable: {exc}"}


def grade_issues(issues: list[dict]) -> str:
    severe = sum(issue.get("severity") == "severe" for issue in issues)
    moderate = sum(issue.get("severity") == "moderate" for issue in issues)
    if severe >= 3:
        return "D"
    if severe:
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
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "error", "grade": "D", "gate_passed": False, "error": f"required artifact unreadable: {exc}", "issues": [_issue("L1", "severe", "artifact_unreadable", str(exc))], "failed_shot_ids": []}
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    images = find_storyboard_images(output_dir, storyboard)
    l1_issues, per_shot = run_l1_checks(storyboard, visual_style)
    threshold = similarity_threshold if similarity_threshold is not None else float(os.environ.get("HONCUT_STORYBOARD_QA_SIMILARITY", DEFAULT_SIMILARITY_THRESHOLD))
    l2_issues, l2 = run_l2_checks(storyboard, characters, images, threshold, embedder)
    l3_issues, l3 = run_l3_review(storyboard, characters, visual_style, images, output_dir / "storyboard_qa_grid.jpg", multimodal_client)
    issues = l1_issues + l2_issues + l3_issues
    for index, shot in enumerate(storyboard.get("shots", [])):
        sid = _shot_id(shot, index)
        detail = per_shot.setdefault(sid, {"issues": []})
        detail["characters"] = _characters_in_shot(shot, characters.get("characters", []))
        detail["image_path"] = str(images[sid]) if sid in images else None
        detail["issues"] = [issue for issue in issues if sid in issue.get("shot_ids", [])]
    grade = grade_issues(issues)
    failed_shots = sorted({sid for issue in issues if issue.get("severity") == "severe" for sid in issue.get("shot_ids", [])})
    report = {"status": "done" if grade in {"A", "B"} else "error", "grade": grade, "gate_passed": grade in {"A", "B"}, "issues": issues, "issue_counts": {severity: sum(item.get("severity") == severity for item in issues) for severity in ("severe", "moderate", "minor")}, "failed_shot_ids": failed_shots, "shots": per_shot, "layers": {"L1": {"status": "completed"}, "L2": l2, "L3": l3}, "outputs": ["storyboard_qa_report.json", *( ["storyboard_qa_grid.jpg"] if grid_path_exists(output_dir) else [])]}
    if not report["gate_passed"]:
        report["error"] = f"Storyboard QA grade {grade} blocks Phase 5; redraw only failed_shot_ids"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def grid_path_exists(output_dir: Path) -> bool:
    return (output_dir / "storyboard_qa_grid.jpg").is_file()
