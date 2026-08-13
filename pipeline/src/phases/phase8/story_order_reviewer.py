"""Phase 8 storyboard narrative-order review."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from clients.ark_multimodal_client import ArkMultimodalClient


REVIEW_PROMPT = """You are a film storyboard continuity reviewer. Review all supplied
images together, in their supplied order, against the complete storyboard JSON below.
This phase decides chronological shot ordering only. If the supplied order tells the
story coherently, set narrative_consistent=true even when an individual still has a
pose, prop, costume-material, or action-fidelity issue; list such visual issues in
issues without turning an otherwise correct chronology into false.
Return one JSON object only with these fields:
- suggested_order: array containing every shot ID exactly once
- narrative_consistent: boolean
- issues: array of concise strings
Judge narrative chronology, character/action continuity, and whether the visual order
matches the written story. Do not invent or omit shot IDs.

STORYBOARD JSON:
{storyboard_json}
"""


def _review_payload(storyboard: dict) -> dict:
    """Strip generation-only prompt bulk from the narrative-order request."""
    fields = (
        "id",
        "shot_id",
        "who",
        "where",
        "what",
        "action_description",
        "visual",
        "dialogue",
        "shot_type",
        "shot_size",
        "camera_movement",
    )
    shots = []
    for shot in storyboard.get("shots", []):
        compact = {
            key: shot.get(key)
            for key in fields
            if shot.get(key) not in (None, "", [])
        }
        if isinstance(compact.get("visual"), str):
            compact["visual"] = compact["visual"][:800]
        shots.append(compact)
    return {"shots": shots}


def _shot_id(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    value = str(value).strip().upper()
    if value.startswith("S"):
        value = value[1:]
    if not value.isdigit():
        return None
    return f"S{int(value):02d}"


def storyboard_shot_ids(storyboard: dict) -> list[str]:
    ids: list[str] = []
    for shot in storyboard.get("shots", []):
        raw = shot.get("shot_id") or shot.get("id")
        normalized = _shot_id(str(raw)) if raw is not None else None
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def reorder_shots(
    clip_paths: list[str], shot_metas: list[dict], suggested_order: list[str]
) -> tuple[list[str], list[dict], bool]:
    """Reorder aligned clips/metas, preserving unmentioned clips at the end."""
    aligned = list(zip(clip_paths, shot_metas))
    by_id = {Path(path).parent.name.upper(): (path, meta) for path, meta in aligned}
    normalized = [_shot_id(value) for value in suggested_order]
    ordered = [by_id[shot_id] for shot_id in normalized if shot_id in by_id]
    used = {Path(path).parent.name.upper() for path, _ in ordered}
    ordered.extend(pair for pair in aligned if Path(pair[0]).parent.name.upper() not in used)
    new_paths = [path for path, _ in ordered]
    return new_paths, [meta for _, meta in ordered], new_paths != clip_paths


def review_with_multimodal_llm(
    storyboard: dict,
    image_paths: list[Path],
    client: ArkMultimodalClient | None = None,
) -> dict:
    """Review a complete storyboard with all images in one ARK request."""
    reviewer = client or ArkMultimodalClient()
    prompt = REVIEW_PROMPT.format(
        storyboard_json=json.dumps(_review_payload(storyboard), ensure_ascii=False, indent=2)
    )
    raw = reviewer.review(image_paths, prompt)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ARK multimodal review returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("ARK multimodal review must return a JSON object")

    expected = storyboard_shot_ids(storyboard)
    suggested = [_shot_id(item) for item in result.get("suggested_order", [])]
    if (
        any(item is None for item in suggested)
        or len(suggested) != len(expected)
        or set(suggested) != set(expected)
    ):
        raise RuntimeError("ARK multimodal review returned an invalid or incomplete shot order")
    result["suggested_order"] = suggested
    if not isinstance(result.get("narrative_consistent"), bool):
        raise RuntimeError("ARK multimodal review omitted boolean narrative_consistent")
    if not isinstance(result.get("issues"), list):
        raise RuntimeError("ARK multimodal review omitted issues array")
    return result


def review_story_order(output_dir: Path, current_order: list[str]) -> dict:
    """Review storyboard order in explicit ``real`` or ``mock`` mode."""
    output_dir = Path(output_dir)
    review_path = output_dir / "storyboard_order_review.json"
    storyboard_path = output_dir / "STORYBOARD.json"
    image_dir = output_dir / "storyboard_images"
    current = [_shot_id(value) for value in current_order]
    current = [value for value in current if value]

    missing: list[str] = []
    if not storyboard_path.is_file():
        missing.append("STORYBOARD.json missing")
        storyboard = {}
    else:
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            missing.append(f"STORYBOARD.json unreadable: {exc}")
            storyboard = {}

    expected = storyboard_shot_ids(storyboard) or current
    images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
    image_ids = {_shot_id(path.stem) for path in images}
    absent_images = [shot_id for shot_id in expected if shot_id not in image_ids]
    if absent_images:
        missing.append("storyboard images missing: " + ", ".join(absent_images))

    mode = os.environ.get("HONCUT_STORYBOARD_REVIEW", "real").strip().lower()
    # Preserve the old test-only switch while callers migrate to the named mode.
    if os.environ.get("HONCUT_ORDER_REVIEW_MOCK") == "1":
        mode = "mock"
    if mode not in {"mock", "real"}:
        raise ValueError("HONCUT_STORYBOARD_REVIEW must be 'mock' or 'real'")
    mock_enabled = mode == "mock"
    llm_review: dict | None = None
    if not mock_enabled and not missing:
        # Real mode is an explicit production contract. Falling back to ID
        # order would prove only sorting, not narrative continuity.
        llm_review = review_with_multimodal_llm(storyboard, images)

    if llm_review is not None:
        suggested = llm_review.get("suggested_order") or llm_review.get("ordered_shot_ids") or current
        suggested = [value for value in (_shot_id(item) for item in suggested) if value]
        issues = list(llm_review.get("issues") or [])
        consistent = bool(llm_review.get("narrative_consistent", True))
        skipped_reason = None
        source = "llm"
    elif mock_enabled and not missing:
        print("  [8.1] HONCUT_STORYBOARD_REVIEW=mock，使用确定性剧情顺序审稿", flush=True)
        suggested = expected
        issues: list[str] = []
        consistent = True
        skipped_reason = None
        source = "mock"
    else:
        if mock_enabled and missing:
            reason = "; ".join(missing)
        elif missing:
            reason = "; ".join(missing)
        else:
            reason = (
                "未配置可用的全剧本多图 LLM 客户端；设置 "
                "HONCUT_STORYBOARD_REVIEW=mock 可启用确定性测试路径"
            )
        print(f"  ⚠ [8.1] 剧情顺序校验跳过: {reason}；保持原顺序", flush=True)
        suggested = current
        issues = [reason]
        consistent = True
        skipped_reason = reason
        source = "skipped"

    review = {
        "suggested_order": suggested,
        "matches_current_order": suggested == current,
        "narrative_consistent": consistent,
        "issues": issues,
        "source": source,
    }
    if skipped_reason:
        review["skipped_reason"] = skipped_reason
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    return review
