"""Generate the Phase 1 director overview through the configured image model."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


DIRECTOR_STORYBOARD_SIZE = "2560x1440"
GROUP_MAX_SHOTS = 3
DIRECTOR_PANEL_SCHEMA = "honcut.director-panels.v1"


class ImageGenerationClient(Protocol):
    model: str

    def text_to_image(
        self,
        prompt: str,
        output_path: str,
        size: str,
        timeout: int,
    ) -> str: ...


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _compact(value: Any, limit: int = 100) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _preliminary_groups(shots: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    group_number = 0
    count = 0
    for index, shot in enumerate(shots):
        boundary = str(shot.get("boundary_before") or "").lower()
        if index == 0 or boundary == "cut" or count >= GROUP_MAX_SHOTS:
            group_number += 1
            count = 0
        count += 1
        result.append(f"DG{group_number:03d}")
    return result


def _layout(panel_count: int) -> tuple[int, int]:
    if panel_count <= 0:
        return 0, 0
    if panel_count <= 3:
        return panel_count, 1
    if panel_count <= 4:
        return 2, 2
    if panel_count <= 6:
        return 3, 2
    if panel_count <= 9:
        return 3, 3
    if panel_count <= 12:
        return 4, 3
    if panel_count <= 15:
        return 5, 3
    columns = 4 if panel_count == 16 else 5
    return columns, math.ceil(panel_count / columns)


def _character_lines(characters: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for character in characters:
        name = str(character.get("name") or character.get("id") or "角色").strip()
        appearance = character.get("appearance") or {}
        if isinstance(appearance, dict):
            description = appearance.get("summary") or appearance.get("description")
        else:
            description = appearance
        description = (
            description
            or character.get("description")
            or character.get("visual_description")
        )
        if description:
            lines.append(f"- {name}：{_compact(description, 180)}")
    return lines


def build_director_storyboard_prompt(
    storyboard: dict[str, Any],
    characters: list[dict[str, Any]] | None = None,
    aspect_ratio: str = "16:9",
) -> tuple[str, list[dict[str, Any]], tuple[int, int]]:
    """Build one image-model prompt and the exact panel contract behind it."""
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    groups = _preliminary_groups(shots)
    columns, rows = _layout(len(shots))
    panels: list[dict[str, Any]] = []
    panel_lines: list[str] = []
    for index, (shot, group_id) in enumerate(zip(shots, groups), 1):
        shot_id = _shot_id(shot, index)
        # Sxx is a director/editorial shot. Every Sxx starts from its own P01;
        # native extension is reserved for P02+ inside that same shot.
        generation_mode = "FRESH"
        actions = shot.get("generation_actions") or []
        if isinstance(actions, str):
            actions = [actions]
        action = " → ".join(
            _compact(value, 55) for value in actions if str(value).strip()
        )
        action = action or _compact(
            shot.get("action_description") or shot.get("what") or shot.get("visual"),
            90,
        )
        setting = _compact(
            shot.get("where") or shot.get("scene") or shot.get("visual"),
            90,
        )
        camera = _compact(
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or "固定镜头",
            45,
        )
        size = str(
            shot.get("shot_size") or shot.get("shot_type") or "medium"
        ).upper()
        duration = float(
            shot.get("duration") or shot.get("suggested_duration") or 0
        )
        who = shot.get("who") or []
        who_text = (
            "、".join(str(value) for value in who)
            if isinstance(who, list)
            else str(who)
        )
        storyboard_beats = shot.get("storyboard_beats") or []
        beat_count = (
            max(1, len(storyboard_beats))
            if isinstance(storyboard_beats, list)
            else 1
        )
        label = (
            f"{shot_id} · {duration:g}s · {size} · 内部{beat_count}格"
        )
        panel_lines.append(
            f"面板{index}【{label}】：地点={setting or '延续前镜'}；"
            f"人物={who_text or '环境'}；故事摘要={action or '环境建立'}；运镜={camera}；"
            f"后续 Phase 2 为本镜绘制 {beat_count} 个连续故事格。"
        )
        panels.append({
            "position": index,
            "shot_id": shot_id,
            "group_id": group_id,
            "generation_mode": generation_mode.lower(),
            "duration_s": duration,
            "shot_size": size.lower(),
            "camera_movement": camera,
            "setting": setting,
            "characters": who_text,
            "summary": action,
            "storyboard_beat_count": beat_count,
        })

    character_lines = _character_lines(characters or [])
    title = str(storyboard.get("title") or "导演故事板总览")
    prompt = f"""为影片《{title}》绘制一张专业 PREVIS 导演故事板总览图。

画布与版式：
- 单张 {aspect_ratio} 故事板纸，严格使用 {columns} 列 × {rows} 行，共 {len(shots)} 个面板。
- 阅读顺序必须从左到右、从上到下；不得合并、遗漏、重复或打乱面板。
- 每个面板顶部仅写清晰的大号编号 S01、S02……；底部可写不超过 8 个汉字的动作速记。
- 每个 Sxx 是导演级叙事镜头；在编号旁清楚标注“×N格”，N 来自逐格合同中的内部故事格数量。
- 用细黑线明确分隔每格，留出统一白边。

绘画风格：
- 真正的导演手绘分镜草图，不是成片剧照，不是彩色概念设计，不是漫画成稿。
- 白纸、黑色粗铅笔和炭笔线条，松弛快速的 gesture drawing，少量灰色阴影。
- 人物姿势和空间关系必须清楚，动作方向用红色手绘箭头，摄像机运动用蓝色手绘箭头。
- 保持粗糙、动态、未完成的工作稿质感；人物身份遵守项目角色合同，拒绝海报排版和装饰性大标题。

跨面板连续性：
- 同名角色在所有面板保持相同发型、服装、武器和身体比例。
- 每个 Sxx 都是 FRESH 导演镜头，由自己的 P01 建立构图；视频延长仅发生在该 Sxx 内部的 P02+。
- DG 编号相同的相邻 Sxx 虽然重新构图，仍须保持空间轴线、天气和叙事因果连续。

角色设计约束：
{chr(10).join(character_lines) if character_lines else '- 严格依据各面板人物描述，不自行增加或替换人物。'}

逐格内容合同：
{chr(10).join(panel_lines)}

最终检查：画面中必须恰好出现 {len(shots)} 格，镜头编号连续且唯一，构图和动作按上述合同逐格对应。"""
    return prompt, panels, (columns, rows)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _detect_dividers(
    gray: np.ndarray,
    count: int,
    *,
    axis: str,
) -> list[int]:
    """Detect long dark grid dividers close to the authored uniform layout."""
    if count <= 1:
        return []
    if axis == "vertical":
        dark_profile = (gray < 120).mean(axis=0)
        gradient = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean(axis=0) / 255.0
    elif axis == "horizontal":
        dark_profile = (gray < 120).mean(axis=1)
        gradient = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1) / 255.0
    else:
        raise ValueError(f"unknown director grid axis: {axis}")
    gradient = np.pad(gradient, (0, 1), mode="edge")
    length = int(dark_profile.shape[0])
    cell_size = length / count
    search_radius = max(4, round(cell_size * 0.2))
    dividers: list[int] = []
    for divider_index in range(1, count):
        expected = round(length * divider_index / count)
        start = max(1, expected - search_radius)
        end = min(length - 1, expected + search_radius + 1)
        combined = dark_profile[start:end] + gradient[start:end] * 0.75
        offset = int(np.argmax(combined))
        position = start + offset
        local_median = float(np.median(combined))
        if (
            float(dark_profile[position]) < 0.28
            and float(gradient[position]) < 0.14
        ) or float(combined[offset]) < local_median + 0.05:
            raise RuntimeError(
                f"director storyboard {axis} divider {divider_index}/{count - 1} "
                f"was not detected near pixel {expected}"
            )
        dividers.append(position)
    if dividers != sorted(set(dividers)):
        raise RuntimeError(f"director storyboard {axis} dividers overlap")
    return dividers


def materialize_director_panels(
    image_path: Path,
    panels: list[dict[str, Any]],
    columns: int,
    rows: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split the generated overview into exact Sxx assets and persist bboxes."""
    if columns <= 0 or rows <= 0 or len(panels) > columns * rows:
        raise ValueError(
            f"invalid director grid contract: {len(panels)} panels in {columns}x{rows}"
        )
    shot_ids = [str(panel.get("shot_id") or "") for panel in panels]
    if not all(shot_ids) or len(set(shot_ids)) != len(shot_ids):
        raise ValueError("director grid requires unique non-empty shot ids")

    panel_dir = output_dir / "director_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source_image:
        source = source_image.convert("RGB")
    width, height = source.size
    if width < columns * 64 or height < rows * 64:
        raise RuntimeError(
            f"director storyboard is too small for {columns}x{rows}: {width}x{height}"
        )
    gray = np.asarray(source.convert("L"))
    vertical = _detect_dividers(gray, columns, axis="vertical")
    horizontal = _detect_dividers(gray, rows, axis="horizontal")
    x_boundaries = [0, *vertical, width]
    y_boundaries = [0, *horizontal, height]

    enriched: list[dict[str, Any]] = []
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        left, right = x_boundaries[column], x_boundaries[column + 1]
        top, bottom = y_boundaries[row], y_boundaries[row + 1]
        inset = max(2, round(min(right - left, bottom - top) * 0.006))
        bbox = [left + inset, top + inset, right - inset, bottom - inset]
        if bbox[2] - bbox[0] < 64 or bbox[3] - bbox[1] < 64:
            raise RuntimeError(
                f"director panel {shot_ids[index]} has invalid detected bbox {bbox}"
            )
        crop_path = panel_dir / f"{shot_ids[index]}.png"
        temporary = crop_path.with_suffix(".png.tmp")
        source.crop(tuple(bbox)).save(temporary, format="PNG", optimize=True)
        temporary.replace(crop_path)
        enriched.append({
            **panel,
            "grid_row": row,
            "grid_column": column,
            "bbox_px": bbox,
            "bbox_norm": [
                round(bbox[0] / width, 6),
                round(bbox[1] / height, 6),
                round(bbox[2] / width, 6),
                round(bbox[3] / height, 6),
            ],
            "crop": str(crop_path.relative_to(output_dir)),
            "crop_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
        })
    extraction = {
        "schema": DIRECTOR_PANEL_SCHEMA,
        "method": "long-dark-divider-v1",
        "source_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "source_size": [width, height],
        "vertical_dividers_px": vertical,
        "horizontal_dividers_px": horizontal,
        "panel_count": len(enriched),
    }
    return enriched, extraction


def generate_director_storyboard(
    output_dir: Path,
    storyboard: dict[str, Any],
    characters: list[dict[str, Any]] | None = None,
    *,
    client: ImageGenerationClient | None = None,
    dry_run: bool = False,
    size: str = DIRECTOR_STORYBOARD_SIZE,
) -> dict[str, Any]:
    """Generate and audit one Phase 1 director board with Seedream."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aspect_ratio = str(storyboard.get("aspect_ratio") or "").strip()
    if not aspect_ratio:
        try:
            width, height = (int(value) for value in size.lower().split("x", 1))
            divisor = math.gcd(width, height)
            aspect_ratio = f"{width // divisor}:{height // divisor}"
        except (TypeError, ValueError):
            aspect_ratio = "16:9"
    prompt, panels, layout = build_director_storyboard_prompt(
        storyboard,
        characters,
        aspect_ratio,
    )
    image_path = output_dir / "director_storyboard.png"
    prompt_path = output_dir / "director_storyboard_prompt.txt"
    manifest_path = output_dir / "director_storyboard.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    requested_model = (
        getattr(client, "model", None) or "doubao-seedream-5.0-lite"
    )
    manifest: dict[str, Any] = {
        "kind": "honcut.director_storyboard.v3",
        "version": 3,
        "status": "planned",
        "provider": "seedream",
        "model": requested_model,
        "image": image_path.name,
        "prompt": prompt_path.name,
        "prompt_sha256": prompt_sha256,
        "size_requested": size,
        "aspect_ratio": aspect_ratio,
        "columns": layout[0],
        "rows": layout[1],
        "group_max_shots": GROUP_MAX_SHOTS,
        "panels": panels,
    }
    if not panels:
        manifest["status"] = "no_shots"
        manifest["image"] = None
        _write_manifest(manifest_path, manifest)
        return manifest
    if dry_run:
        manifest["status"] = "dry_run"
        manifest["image"] = None
        _write_manifest(manifest_path, manifest)
        return manifest

    # A completed image-model call is a paid artifact. Reuse it only when both
    # the exact prompt and requested model match and the image still decodes.
    if manifest_path.is_file() and image_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "done"
                and previous.get("prompt_sha256") == prompt_sha256
                and previous.get("model") == requested_model
            ):
                with Image.open(image_path) as generated:
                    generated.verify()
                cropped_panels, extraction = materialize_director_panels(
                    image_path,
                    panels,
                    layout[0],
                    layout[1],
                    output_dir,
                )
                previous.update({
                    "kind": manifest["kind"],
                    "version": manifest["version"],
                    "columns": layout[0],
                    "rows": layout[1],
                    "panels": cropped_panels,
                    "panel_extraction": extraction,
                    "cache_hit": True,
                })
                _write_manifest(manifest_path, previous)
                return previous
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    _write_manifest(manifest_path, manifest)
    try:
        if client is None:
            from clients.seedream_client import SeedreamClient

            client = SeedreamClient()
            manifest["model"] = client.model
        result_url = client.text_to_image(
            prompt=prompt,
            output_path=str(image_path),
            size=size,
            timeout=180,
        )
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(
                "image provider returned without director_storyboard.png"
            )
        with Image.open(image_path) as generated:
            generated.verify()
        with Image.open(image_path) as generated:
            manifest["size_actual"] = list(generated.size)
        manifest["panels"], manifest["panel_extraction"] = materialize_director_panels(
            image_path,
            panels,
            layout[0],
            layout[1],
            output_dir,
        )
        manifest["status"] = "done"
        manifest["result_url"] = result_url
        _write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = str(exc)
        _write_manifest(manifest_path, manifest)
        raise
