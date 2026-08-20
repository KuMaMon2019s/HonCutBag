"""Generate the Phase 1 director overview through the configured image model."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from utils.character_body_contracts import character_visual_description
from utils.body_action_contracts import body_action_prompt
from utils.camera_motion_contracts import (
    camera_motion_negative_prompt,
    camera_motion_prompt,
)

DIRECTOR_STORYBOARD_SIZE = "2560x1440"
DIRECTOR_STORYBOARD_MAX_ATTEMPTS = 2
GROUP_MAX_SHOTS = 3
DIRECTOR_PANEL_SCHEMA = "honcut.director-panels.v1"


class DirectorStoryboardLayoutError(RuntimeError):
    """The generated overview cannot be split into the authored panel grid."""


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
        description = character_visual_description(character)
        if description:
            lines.append(f"- {name}：{_compact(description, 1400)}")
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
    for index, (shot, group_id) in enumerate(zip(shots, groups, strict=True), 1):
        shot_id = _shot_id(shot, index)
        # Sxx is a director/editorial shot. Every Sxx starts from its own P01;
        # later Pxx are selected by content capacity and the next boundary.
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
        physical_camera = camera_motion_prompt(shot)
        camera_negative = camera_motion_negative_prompt(shot)
        choreography = body_action_prompt(shot)
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
            f"摄影物理合同={physical_camera}；"
            f"摄影禁止项={camera_negative}；"
            f"逐拍肢体动作谱={_compact(choreography, 1200) or '无专项舞蹈/格斗动作'}；"
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
            "body_action_contract": shot.get("body_action_contract") or {},
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
- 这是必须可被机器切分的固定网格合同，版式准确性优先于绘画装饰和构图自由。
- 所有列必须等宽，所有行必须等高；列边界从画布顶端贯通到底端，行边界从画布左端贯通到右端，不得错位或中断。
- 相邻面板之间保留 16–24 像素纯白留白槽，并沿每个面板四周绘制清晰、连续、深黑色矩形边框；留白槽内禁止出现人物、道具、箭头或文字。
- S01 至 S{len(shots):02d} 每个编号只能出现一次，并且必须位于对应面板内部；禁止在行间额外重复任何 Sxx 标题，禁止增加横幅、表头或第二组编号。
- 面板不得跨格、合并、重叠或越过留白槽；任何动作、字幕与运镜箭头都必须完整留在所属面板内。

绘画风格：
- 真正的导演手绘分镜草图，不是成片剧照，不是彩色概念设计，不是漫画成稿。
- 白纸、黑色粗铅笔和炭笔线条，松弛快速的 gesture drawing，少量灰色阴影。
- 人物姿势和空间关系必须清楚，动作方向用红色手绘箭头，摄像机运动用蓝色手绘箭头。
- 保持粗糙、动态、未完成的工作稿质感；人物身份遵守项目角色合同，拒绝海报排版和装饰性大标题。

跨面板连续性：
- 同名角色在所有面板保持相同发型、服装、武器和身体比例。
- 每个 Sxx 都由自己的 P01 建立构图；只有单段容量不足时才在该 Sxx 内增加视频延长格。跨 Sxx 桥接不占故事板面板：只有连续边界才在所有一级视频完成后用相邻成片真实尾帧/首帧生成，转场边界交给 Phase 8 添加特效。
- DG 编号相同的相邻 Sxx 虽然重新构图，仍须保持空间轴线、天气和叙事因果连续。

角色设计约束：
{chr(10).join(character_lines) if character_lines else '- 严格依据各面板人物描述，不自行增加或替换人物。'}

角色职责与道具合同：
- 每个具名角色只执行逐格内容合同明确分配给自己的动作；不得因邻近角色或群体正在行动，就让旁观、记录、驾驶、守卫或其他角色自动模仿、加入或交换职责。
- 严格保留源文本声明的道具类型、持有者、朝向和使用方式；不得把一种设备替换成另一种，也不得让角色无故放下、丢失或交换道具。

逐格内容合同：
{chr(10).join(panel_lines)}

最终版式自检：输出前逐项确认画面恰好为 {columns}×{rows} 等分网格、恰好 {len(shots)} 格，S01 至 S{len(shots):02d} 连续且各出现一次，只有一组编号，所有贯通边界和纯白留白槽清晰可见；任一条件不满足时必须先重新排版再输出。构图和动作按上述合同逐格对应。"""
    return prompt, panels, (columns, rows)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_layout_retry_prompt(
    prompt: str,
    *,
    attempt: int,
    max_attempts: int,
    previous_error: str,
) -> str:
    """Add explicit corrective layout instructions after a rejected image."""
    return f"""{prompt}

版式纠错重生成（第 {attempt}/{max_attempts} 次）：
- 上一张图片已被机器版式审计拒绝：{previous_error}
- 必须从空白画布重新排版，禁止沿用上一张的网格、标题带或面板边界。
- 先建立严格等宽等高的固定网格与贯通留白槽，再在每个面板内部绘制内容。
- 只允许一组 Sxx 编号；禁止重复行标题、额外表头、跨格画面和错位边界。
- 输出前再次执行最终版式自检，任何一项不满足都不得提交。"""


def _detect_dividers(
    gray: np.ndarray,
    count: int,
    *,
    axis: str,
) -> list[int]:
    """Detect aligned dark rules or white gutters near the uniform layout."""
    if count <= 1:
        return []
    if axis == "vertical":
        dark_profile = (gray < 120).mean(axis=0)
        white_profile = (gray > 248).mean(axis=0)
        gradient = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean(axis=0) / 255.0
    elif axis == "horizontal":
        dark_profile = (gray < 120).mean(axis=1)
        white_profile = (gray > 248).mean(axis=1)
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

        # Image models commonly draw a clean white gutter instead of the
        # requested black rule. Accept a narrow, high-whitespace band only
        # when meaningful non-white content exists on both sides; this keeps
        # an empty white canvas fail-closed.
        minimum_gutter_width = max(3, round(cell_size * 0.004))
        flank_width = max(8, round(cell_size * 0.04))
        gutter_mask = white_profile[start:end] >= 0.97
        gutter_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for offset, is_gutter in enumerate(gutter_mask):
            if bool(is_gutter) and run_start is None:
                run_start = offset
            elif not bool(is_gutter) and run_start is not None:
                gutter_runs.append((start + run_start, start + offset))
                run_start = None
        if run_start is not None:
            gutter_runs.append((start + run_start, end))

        valid_gutters: list[tuple[float, float, int]] = []
        for gutter_start, gutter_end in gutter_runs:
            if gutter_end - gutter_start < minimum_gutter_width:
                continue
            left = white_profile[max(start, gutter_start - flank_width):gutter_start]
            right = white_profile[gutter_end:min(end, gutter_end + flank_width)]
            if not left.size or not right.size:
                continue
            left_nonwhite = 1.0 - float(left.mean())
            right_nonwhite = 1.0 - float(right.mean())
            if min(left_nonwhite, right_nonwhite) < 0.05:
                continue
            center = round((gutter_start + gutter_end - 1) / 2)
            valid_gutters.append((
                abs(center - expected),
                -float(white_profile[gutter_start:gutter_end].mean()),
                center,
            ))
        if valid_gutters:
            dividers.append(min(valid_gutters)[2])
            continue

        combined = dark_profile[start:end] + gradient[start:end] * 0.75
        offset = int(np.argmax(combined))
        position = start + offset
        local_median = float(np.median(combined))
        if (
            float(dark_profile[position]) < 0.28
            and float(gradient[position]) < 0.14
        ) or float(combined[offset]) < local_median + 0.05:
            raise DirectorStoryboardLayoutError(
                f"director storyboard {axis} divider {divider_index}/{count - 1} "
                f"was not detected near pixel {expected}"
            )
        dividers.append(position)
    if dividers != sorted(set(dividers)):
        raise DirectorStoryboardLayoutError(
            f"director storyboard {axis} dividers overlap"
        )
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
        raise DirectorStoryboardLayoutError(
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
            raise DirectorStoryboardLayoutError(
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
        "method": "aligned-rule-or-gutter-v2",
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
    max_layout_attempts: int = DIRECTOR_STORYBOARD_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Generate and audit one Phase 1 director board with Seedream."""
    if max_layout_attempts < 1:
        raise ValueError("max_layout_attempts must be at least 1")
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
        "contract_prompt_sha256": prompt_sha256,
        "size_requested": size,
        "aspect_ratio": aspect_ratio,
        "columns": layout[0],
        "rows": layout[1],
        "group_max_shots": GROUP_MAX_SHOTS,
        "panels": panels,
    }
    if not panels:
        prompt_path.write_text(prompt, encoding="utf-8")
        manifest["status"] = "no_shots"
        manifest["image"] = None
        _write_manifest(manifest_path, manifest)
        return manifest
    if dry_run:
        prompt_path.write_text(prompt, encoding="utf-8")
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
                and previous.get(
                    "contract_prompt_sha256",
                    previous.get("prompt_sha256"),
                ) == prompt_sha256
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
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            DirectorStoryboardLayoutError,
        ):
            pass

    _write_manifest(manifest_path, manifest)
    try:
        if client is None:
            from clients.seedream_client import SeedreamClient

            client = SeedreamClient()
            manifest["model"] = client.model
        generation_attempts: list[dict[str, Any]] = []
        previous_layout_error = ""
        for attempt in range(1, max_layout_attempts + 1):
            attempt_path = output_dir / f".director_storyboard_attempt_{attempt:02d}.png"
            if attempt_path.exists():
                attempt_path.unlink()
            attempt_prompt = (
                prompt
                if not previous_layout_error
                else _build_layout_retry_prompt(
                    prompt,
                    attempt=attempt,
                    max_attempts=max_layout_attempts,
                    previous_error=previous_layout_error,
                )
            )
            attempt_prompt_sha256 = hashlib.sha256(
                attempt_prompt.encode("utf-8")
            ).hexdigest()
            prompt_path.write_text(attempt_prompt, encoding="utf-8")
            manifest["prompt_sha256"] = attempt_prompt_sha256
            result_url = client.text_to_image(
                prompt=attempt_prompt,
                output_path=str(attempt_path),
                size=size,
                timeout=180,
            )
            if not attempt_path.is_file() or attempt_path.stat().st_size == 0:
                raise RuntimeError(
                    "image provider returned without director_storyboard.png"
                )
            with Image.open(attempt_path) as generated:
                generated.verify()
            with Image.open(attempt_path) as generated:
                manifest["size_actual"] = list(generated.size)
            try:
                enriched_panels, extraction = materialize_director_panels(
                    attempt_path,
                    panels,
                    layout[0],
                    layout[1],
                    output_dir,
                )
            except DirectorStoryboardLayoutError as exc:
                rejected_dir = output_dir / "director_storyboard_attempts"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                rejected_path = rejected_dir / f"attempt_{attempt:02d}_rejected.png"
                rejected_prompt_path = (
                    rejected_dir / f"attempt_{attempt:02d}_prompt.txt"
                )
                attempt_path.replace(rejected_path)
                rejected_prompt_path.write_text(attempt_prompt, encoding="utf-8")
                generation_attempts.append({
                    "attempt": attempt,
                    "status": "rejected_layout",
                    "error": str(exc),
                    "image": str(rejected_path.relative_to(output_dir)),
                    "prompt": str(rejected_prompt_path.relative_to(output_dir)),
                    "prompt_sha256": attempt_prompt_sha256,
                })
                previous_layout_error = str(exc)
                manifest["generation_attempts"] = generation_attempts
                _write_manifest(manifest_path, manifest)
                if attempt < max_layout_attempts:
                    print(
                        "  ⚠️ 导演故事板版式审计失败，自动重新生成 "
                        f"({attempt + 1}/{max_layout_attempts}): {exc}"
                    )
                    continue
                raise

            shutil.copy2(attempt_path, image_path)
            attempt_path.unlink()
            generation_attempts.append({
                "attempt": attempt,
                "status": "accepted",
                "result_url": result_url,
                "prompt_sha256": attempt_prompt_sha256,
            })
            manifest["generation_attempts"] = generation_attempts
            manifest["panels"] = enriched_panels
            manifest["panel_extraction"] = extraction
            manifest["status"] = "done"
            manifest["result_url"] = result_url
            _write_manifest(manifest_path, manifest)
            return manifest
        raise RuntimeError("director storyboard generation exhausted without result")
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = str(exc)
        _write_manifest(manifest_path, manifest)
        raise
