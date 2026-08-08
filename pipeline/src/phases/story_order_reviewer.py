"""Phase 7.1 storyboard narrative-order review."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _shot_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    if not value.startswith("S") or not value[1:].isdigit():
        return None
    return f"S{int(value[1:]):02d}"


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


def review_with_multimodal_llm(storyboard: dict, image_paths: list[Path]) -> dict:
    """Production interface for a future multi-image narrative reviewer.

    The repository currently has single-asset vision/embedding helpers only;
    none accepts all storyboard images plus the complete script. Keeping this
    interface explicit prevents an image generator or transition embedder from
    being mistaken for a narrative reviewer.
    """
    # TODO: send storyboard + every image to the approved multimodal LLM and
    # return suggested_order, narrative_consistent, and issues.
    raise NotImplementedError("full-story multi-image LLM client is not configured")


def review_story_order(output_dir: Path, current_order: list[str]) -> dict:
    """Review storyboard order, with an explicit deterministic mock/skip path.

    TODO: wire a production multi-image LLM client once the pipeline has a
    supported narrative-review endpoint. Existing vision helpers analyze one
    image/video at a time and cannot reliably order a complete storyboard.
    """
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

    mock_enabled = os.environ.get("HONCUT_ORDER_REVIEW_MOCK") == "1"
    llm_review: dict | None = None
    if not mock_enabled and not missing:
        try:
            llm_review = review_with_multimodal_llm(storyboard, images)
        except Exception as exc:
            missing.append(f"multimodal LLM unavailable: {exc}")

    if llm_review is not None:
        suggested = llm_review.get("suggested_order") or llm_review.get("ordered_shot_ids") or current
        suggested = [value for value in (_shot_id(item) for item in suggested) if value]
        issues = list(llm_review.get("issues") or [])
        consistent = bool(llm_review.get("narrative_consistent", True))
        skipped_reason = None
        source = "llm"
    elif mock_enabled and not missing:
        print("  [7.1] HONCUT_ORDER_REVIEW_MOCK=1，使用确定性剧情顺序审稿", flush=True)
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
                "HONCUT_ORDER_REVIEW_MOCK=1 可启用确定性测试路径"
            )
        print(f"  ⚠ [7.1] 剧情顺序校验跳过: {reason}；保持原顺序", flush=True)
        suggested = current
        issues = [reason]
        consistent = True
        skipped_reason = reason
        source = "mock"

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
