"""Render style-authored video first frames from safe upstream composition guides."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from phases.phase2.shot_storyboards import _character_reference_paths, _shot_who
from prompt.seedream_image_prompt import (
    IMAGE_REQUEST_CONTRACT_ID,
    IMAGE_REQUEST_CONTRACT_VERSION,
    REFERENCE_CONTRACT_TEMPLATE_ID,
    REFERENCE_CONTRACT_TEMPLATE_VERSION,
    bind_reference_roles,
    image_request_fingerprint,
    prompt_guidance_metrics,
)
from utils.body_action_contracts import body_action_prompt
from utils.camera_motion_contracts import camera_motion_negative_prompt, camera_motion_prompt
from utils.character_body_contracts import character_visual_description
from utils.pixel_text_policy import (
    PIXEL_TEXT_METADATA_CONTRACT,
    strip_pixel_text_identity_markers,
)
from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    temporal_visual_negative_prompt,
    temporal_visual_prompt,
)
from utils.visual_style_contract import (
    BASE_STYLE_SCHEMA,
    build_visual_style_contract,
    style_reference_compatible,
)
from utils.visual_style_spec import VisualStyle, parse_visual_style

CINEMATIC_FIRST_FRAME_SCHEMA = "honcut.cinematic-first-frame.v1"
CINEMATIC_FIRST_FRAMES_SCHEMA = "honcut.cinematic-first-frames.v1"
PREVIS_PATH_PARTS = frozenset({
    "director_panels",
    "storyboard_beats",
    "shot_storyboards",
    "storyboard_groups",
    "storyboard_bridges",
    "phase5_reference_boards",
})


class ImageGenerationClient(Protocol):
    model: str

    def text_to_image(
        self,
        prompt: str,
        output_path: str,
        size: str,
        timeout: int,
    ) -> str: ...

    def image_to_image(
        self,
        prompt: str,
        ref_image: str | list[str],
        output_path: str,
        size: str,
    ) -> str: ...


class ImageStyleClassifier(Protocol):
    def classify(self, path: Path) -> dict[str, Any]: ...


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _portable_path(output_dir: Path, path: Path) -> str:
    return str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else str(path)


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _compact(value: Any, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _style_prompt(style: VisualStyle, style_contract: dict[str, Any]) -> str:
    parts = [
        "BASE VISUAL STYLE HARD CONTRACT: "
        f"{style_contract['base_style']}. {style_contract['positive_prompt']}. "
        f"{style_contract['negative_prompt']}.",
        style.style_prompt_full or style.style_prompt_short,
    ]
    if style.colors_primary or style.colors_accent:
        colors = [
            f"{entry.name} {entry.hex} ({entry.role})"
            for entry in [*style.colors_primary, *style.colors_accent]
            if entry.name or entry.hex
        ]
        if colors:
            parts.append("色彩合同：" + "；".join(colors))
    if style.mood_keywords:
        parts.append("情绪关键词：" + "、".join(style.mood_keywords))
    if style.mood_avoid:
        parts.append("风格禁止项：" + "、".join(style.mood_avoid))
    if style.tags:
        parts.append("风格标签：" + "、".join(style.tags))
    return "\n".join(part for part in parts if str(part).strip()).strip()


def load_cinematic_style_contract(
    output_dir: Path,
    visual_style_path: Path | None = None,
) -> dict[str, Any]:
    """Load the project style, falling back explicitly to the bundled contract."""
    output_dir = Path(output_dir)
    requested = Path(visual_style_path) if visual_style_path else output_dir / "visual-style.md"
    if requested.is_file():
        source = requested
        source_kind = "project_visual_style"
    else:
        source = Path(__file__).resolve().parents[3] / "prompts" / "default_visual_style.md"
        source_kind = "bundled_default_visual_style"
    raw = source.read_text(encoding="utf-8")
    parsed = parse_visual_style(raw)
    style_contract = build_visual_style_contract(parsed)
    prompt = _style_prompt(parsed, style_contract)
    if not prompt:
        raise RuntimeError(f"cinematic visual style has no prompt content: {source}")
    return {
        "source": _portable_path(output_dir, source),
        "source_kind": source_kind,
        "source_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "name": parsed.name,
        "style_contract": style_contract,
        "base_style": style_contract["base_style"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def _character_lines(
    characters: list[dict[str, Any]],
    who: list[Any] | None,
) -> list[str]:
    if who == []:
        return ["- 无人物镜头；禁止出现人物、人形剪影或群众。"]
    requested = {str(value).casefold() for value in who or []}
    lines: list[str] = []
    for character in characters:
        names = {
            str(character.get("id") or "").casefold(),
            str(character.get("name") or "").casefold(),
            *(str(value).casefold() for value in character.get("aliases", [])),
        }
        if requested and requested.isdisjoint(names):
            continue
        description = strip_pixel_text_identity_markers(
            character_visual_description(character)
        )
        if description:
            lines.append(
                f"- {character.get('name') or character.get('id')}："
                f"{_compact(description)}"
            )
    return lines or ["- 严格使用项目角色合同，不增加或替换人物。"]


def _requires_flat_shadow_material(
    shot: dict[str, Any],
    beat: dict[str, Any],
) -> bool:
    """Detect first frames whose subject must remain projected on the screen."""
    start_state = str(beat.get("start_state") or shot.get("start_state") or "")
    scene = " ".join(
        str(value or "")
        for value in (
            shot.get("where"),
            shot.get("visual"),
            beat.get("visual"),
        )
    )
    return (
        "幕布" in scene
        and any(marker in start_state for marker in ("投影贴合", "贴合于幕布", "幕布上的皮影"))
        and not any(marker in start_state for marker in ("完全脱离", "已脱离幕布"))
    )


def _stage_material_contract(
    shot: dict[str, Any],
    beat: dict[str, Any],
) -> str:
    if not _requires_flat_shadow_material(shot, beat):
        return ""
    return (
        "【首帧舞台介质硬合同｜零例外】第一帧中的主角只能是贴合半透明幕布平面的"
        "二维透光皮影/镂空皮影片或平面投影，身体不得站在幕前地面，不得成为有体积的"
        "真人、CG 人偶或实体木偶；头发、脸、服装和肢体身份特征必须转换为同一幕布平面"
        "内的剪影、镂空和彩绘透光纹理。操纵者只能位于幕布之后，以暗色剪影和手/操纵杆"
        "投影出现，不得在幕前露出实体面部或服装。画面中幕前实体角色数量必须为零。"
    )


def build_cinematic_first_frame_prompt(
    shot: dict[str, Any],
    beat: dict[str, Any],
    shot_id: str,
    characters: list[dict[str, Any]],
    style_contract: dict[str, Any],
    scene_contract: dict[str, Any] | None = None,
    *,
    aspect_ratio: str = "16:9",
    correction_context: list[dict[str, Any]] | None = None,
) -> str:
    """Build one finished-look first frame with the style header at byte zero."""
    scene_contract = scene_contract or {}
    temporal = apply_temporal_visual_contract({**shot, **beat})
    temporal_lines = (
        f"时间视觉合同：{temporal_visual_prompt(temporal)}。\n"
        f"时间视觉禁止项：{temporal_visual_negative_prompt(temporal)}。"
        if temporal
        else ""
    )
    beat_id = str(beat.get("beat_id") or f"{shot_id}_P01")
    correction_lines = []
    for correction in correction_context or []:
        expected = _compact(correction.get("expected"), 900)
        observed = _compact(correction.get("observed"), 900)
        message = _compact(correction.get("message"), 500)
        correction_lines.append(
            "- 必须修复上一轮 L4 拒绝"
            f"（{correction.get('code') or 'visual_mismatch'}）：{message}；"
            f"本轮必须呈现：{expected}；严禁再次呈现：{observed}。"
        )
    correction_block = (
        "\nL4 定向纠偏合同（高于导演单格和角色参考图）：\n"
        + "\n".join(correction_lines)
        if correction_lines
        else ""
    )
    stage_material_contract = _stage_material_contract(shot, beat)
    return f"""【美术风格｜最高优先级｜成片质感】
{style_contract['prompt']}

{stage_material_contract}

生成 {beat_id} 的单幅 {aspect_ratio} 视频第一帧。它是会直接送入视频模型的成片质感像素资产，不是 PREVIS、分镜草图、导演工作板、概念线稿或说明图。

上游参考图职责合同：
- 若提供该镜头的导演故事板单格，它只负责一级镜头的场景结构、主体站位、空间关系、视线与构图意图；必须按本提示开头的项目美术风格重新渲染，绝不继承其线稿媒介、白纸底色、箭头、轨迹线、文字、编号、边框或制作标注。
- 角色参考图只负责身份、面纱/妆造、服装材质与身体比例；不得复制参考姿势或参考图排版。
- 上一张成片质感帧只负责同一镜头内的连续状态；任何参考图都不是可直接交付的视频首帧。
- 导演单格或角色参考若把主体画成幕前实体，而剧情要求幕后剪影、幕布投影、平面皮影、镂空皮影片或其他舞台介质，必须改变主体的前后层级、遮挡关系、立体程度和材质，使其服从剧情与项目美术风格；参考图的实体感没有继承权。

第一帧叙事合同：
- 一级镜头：{shot_id}；二级剧情：{beat_id}。
- 第一帧只呈现动作开始前可继续运动的起始状态：{_compact(beat.get('start_state') or shot.get('start_state') or shot.get('what'))}。
- 下一时刻将执行：{_compact(beat.get('action') or shot.get('action_description') or shot.get('what'))}。用真实姿态、重心、视线、关节和空间关系蓄势，不得把动作结果提前画进第一帧。
- 本段目标终态只用于构图预留，不得提前出现：{_compact(beat.get('end_state') or shot.get('end_state'))}。
- 场景：{_compact(shot.get('where') or shot.get('visual'))}。
- 完整场景显影与舞台分层：{_compact(shot.get('visual') or shot.get('what'), 1800)}。
- 景别：{beat.get('shot_size') or shot.get('shot_size') or 'medium'}；运镜意图：{beat.get('camera_movement') or shot.get('camera_movement') or 'steadicam'}。
- 运镜物理合同：{camera_motion_prompt({**shot, **beat})}。
- 人体动作合同：{_compact(body_action_prompt({**shot, **beat}), 1800) or '自然、可继续运动的起始姿态'}。
- 光照：以本提示开头的项目美术风格光影与色彩合同为唯一权威；场景连续性只能维持同一风格内的方向与明暗关系，不得覆盖调色、饱和度、舞台介质或主辅光颜色；如有冲突，忽略场景连续性光照描述。
{temporal_lines}

角色合同：
{chr(10).join(_character_lines(characters, _shot_who(shot)))}

像素洁净硬闸门：
- 画面铺满单幅 {aspect_ratio}，必须呈现项目美术风格要求的材质、色彩、舞台/环境结构、光影与成片完成度。
- 禁止任何红色/蓝色/彩色动作箭头、运镜箭头、轨迹线、辅助线、编号、Pxx/Sxx/Gxx、字幕、文字、字母、数字、水印、边框、分格、拼贴、接触表、色卡或 UI。
- 禁止白纸铅笔稿、炭笔稿、漫画线稿、导演草图、故事板网格、双拼板、九宫格和制作标注。
- 每个已声明角色只能按剧情所需出现一次；除非剧本明确要求分身、镜像或屏幕影像，禁止同一已声明角色出现透明复写、双重曝光或重复实例。
- 参考图若存在，只能按上述职责锁定导演单格构图、角色身份或上一张成片质感帧；不得继承参考图媒介、姿态、排版或说明元素。
- {PIXEL_TEXT_METADATA_CONTRACT}。
- 禁止：{camera_motion_negative_prompt({**shot, **beat})}。
{correction_block}

输出前逐项自检：美术风格、场景结构、角色身份、起始状态、无文字、无箭头、无分格七项全部通过后才输出。"""


def _is_previs_reference(path: Path) -> bool:
    lowered_parts = {part.casefold() for part in path.parts}
    return bool(lowered_parts & PREVIS_PATH_PARTS) or path.name.casefold() in {
        "director_storyboard.png",
        "storyboard.png",
    }


def _director_composition_reference(output_dir: Path, shot_id: str) -> Path | None:
    """Return only the exact per-shot director cell, never a board/contact sheet."""
    candidate = output_dir / "director_panels" / f"{shot_id}.png"
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        return None
    _validate_image(candidate)
    return candidate


def _validate_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"cinematic first frame missing: {path}")
    with Image.open(path) as image:
        image.verify()


def _phase5_rejected_frame_ids(
    output_dir: Path,
) -> tuple[set[str], str | None, dict[str, list[dict[str, Any]]]]:
    """Load L4 frame rejections that must invalidate an otherwise valid cache."""
    report_path = output_dir / "storyboard_qa_report.json"
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return set(), None, {}
    if report.get("gate_passed") is True:
        return set(), None, {}
    rejected: set[str] = set()
    correction_context: dict[str, list[dict[str, Any]]] = {}
    for issue in report.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        if str(issue.get("layer") or "").upper() != "L4":
            continue
        if str(issue.get("severity") or "").casefold() not in {"severe", "moderate"}:
            continue
        code = str(issue.get("code") or "").casefold()
        if code == "first_frame_style_review_unavailable":
            # A provider/key outage did not reject the pixels themselves.
            continue
        details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
        candidates = [
            *(issue.get("frame_ids") or []),
            *(details.get("frame_ids") or []),
            issue.get("frame_id"),
            details.get("frame_id"),
            issue.get("resource_id"),
            details.get("resource_id"),
        ]
        frame_ids = {
            str(value).strip()
            for value in candidates
            if re.fullmatch(r"S[^\s]+_P\d+", str(value or "").strip(), re.IGNORECASE)
        }
        rejected.update(frame_ids)
        context = {
            "code": issue.get("code"),
            "message": issue.get("message"),
            "expected": issue.get("expected") or details.get("expected"),
            "observed": issue.get("observed") or details.get("observed"),
        }
        if any(str(context.get(key) or "").strip() for key in ("message", "expected", "observed")):
            for frame_id in frame_ids:
                correction_context.setdefault(frame_id, []).append(context)
    return (
        rejected,
        hashlib.sha256(raw).hexdigest() if rejected else None,
        correction_context,
    )


def generate_cinematic_first_frames(
    output_dir: Path,
    storyboard: dict[str, Any],
    characters: list[dict[str, Any]],
    scene_consistency: dict[str, Any] | None = None,
    *,
    client: ImageGenerationClient | None = None,
    size: str = "2K",
    visual_style_path: Path | None = None,
    aspect_ratio: str | None = None,
    style_classifier: ImageStyleClassifier | None = None,
) -> dict[str, Any]:
    """Generate every Pxx video frame and replace legacy Sxx preview aliases."""
    output_dir = Path(output_dir)
    frame_dir = output_dir / "video_first_frames"
    alias_dir = output_dir / "storyboard_images"
    frame_dir.mkdir(parents=True, exist_ok=True)
    alias_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "CINEMATIC_FIRST_FRAMES.json"
    style = load_cinematic_style_contract(output_dir, visual_style_path)
    if not aspect_ratio:
        aspect_ratio = str(storyboard.get("aspect_ratio") or "16:9")
    if client is None:
        from clients.seedream_client import SeedreamClient

        client = SeedreamClient()
    model = getattr(client, "model", None) or "doubao-seedream-5.0-lite"
    scene_shots = (scene_consistency or {}).get("shots") or {}
    (
        rejected_frame_ids,
        rejected_report_sha256,
        correction_context_by_frame,
    ) = _phase5_rejected_frame_ids(output_dir)
    manifest: dict[str, Any] = {
        "kind": CINEMATIC_FIRST_FRAMES_SCHEMA,
        "version": 1,
        "status": "running",
        "provider": "seedream",
        "model": model,
        "size_requested": size,
        "aspect_ratio": aspect_ratio,
        "style": style,
        "style_contract": style["style_contract"],
        "phase5_rejected_frame_ids": sorted(rejected_frame_ids),
        "phase5_rejection_report_sha256": rejected_report_sha256,
        "frames": [],
    }
    _write_json(manifest_path, manifest)
    try:
        for shot_index, shot in enumerate(storyboard.get("shots", []), 1):
            if not isinstance(shot, dict):
                continue
            shot_id = _shot_id(shot, shot_index)
            director_panel = _director_composition_reference(
                output_dir,
                shot_id,
            )
            director_composition = director_panel
            director_panel_style: dict[str, Any] | None = None
            if director_panel is not None and style_classifier is not None:
                director_panel_style = style_classifier.classify(director_panel)
                if not style_reference_compatible(
                    style["base_style"],
                    director_panel_style,
                ):
                    director_composition = None
            previous_cinematic: Path | None = None
            for beat_index, beat in enumerate(shot.get("storyboard_beats") or [], 1):
                if not isinstance(beat, dict):
                    continue
                beat_id = str(beat.get("beat_id") or f"{shot_id}_P{beat_index:02d}")
                correction_context = correction_context_by_frame.get(beat_id)
                prompt = build_cinematic_first_frame_prompt(
                    shot,
                    beat,
                    shot_id,
                    characters,
                    style,
                    scene_shots.get(shot_id) if isinstance(scene_shots, dict) else {},
                    aspect_ratio=aspect_ratio,
                    correction_context=correction_context,
                )
                baseline_prompt = (
                    build_cinematic_first_frame_prompt(
                        shot,
                        beat,
                        shot_id,
                        characters,
                        style,
                        scene_shots.get(shot_id)
                        if isinstance(scene_shots, dict)
                        else {},
                        aspect_ratio=aspect_ratio,
                        correction_context=None,
                    )
                    if correction_context
                    else prompt
                )
                prompt_path = frame_dir / f"{beat_id}_prompt.txt"
                image_path = frame_dir / f"{beat_id}.png"
                receipt_path = frame_dir / f"{beat_id}.json"
                identity_references = _character_reference_paths(
                    output_dir,
                    characters,
                    _shot_who(shot),
                )
                flat_shadow_material = _requires_flat_shadow_material(shot, beat)
                if flat_shadow_material:
                    # A photographic/full-body identity image strongly biases
                    # image-to-image toward a solid actor. The complete text
                    # identity contract remains in the prompt, while the exact
                    # director cell still supplies composition as requested.
                    identity_references = []
                reference_paths = (
                    [director_composition]
                    if director_composition is not None
                    else []
                )
                reference_paths.extend(identity_references)
                if previous_cinematic is not None:
                    reference_paths.append(previous_cinematic)
                reference_paths = list(dict.fromkeys(reference_paths))[:8]
                forbidden = [
                    path
                    for path in reference_paths
                    if _is_previs_reference(path) and path != director_composition
                ]
                if forbidden:
                    raise RuntimeError(
                        f"{beat_id} cinematic generation rejected PREVIS references: "
                        + ", ".join(str(path) for path in forbidden)
                    )
                reference_roles = [
                    "director_single_panel_composition_only"
                    if path == director_composition
                    else "prior_cinematic_state"
                    if path == previous_cinematic
                    else "character_identity_only"
                    for path in reference_paths
                ]
                prompt = bind_reference_roles(prompt, reference_roles)
                baseline_prompt = bind_reference_roles(
                    baseline_prompt,
                    reference_roles,
                )
                prompt_metrics = prompt_guidance_metrics(prompt)
                reference_hashes = [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in reference_paths
                ]
                input_sha = image_request_fingerprint(
                    prompt=prompt,
                    model=model,
                    size=size,
                    reference_image_sha256=reference_hashes,
                )
                baseline_input_sha = image_request_fingerprint(
                    prompt=baseline_prompt,
                    model=model,
                    size=size,
                    reference_image_sha256=reference_hashes,
                )
                record: dict[str, Any] = {
                    "kind": CINEMATIC_FIRST_FRAME_SCHEMA,
                    "version": 1,
                    "status": "planned",
                    "shot_id": shot_id,
                    "beat_id": beat_id,
                    "image": _portable_path(output_dir, image_path),
                    "prompt": _portable_path(output_dir, prompt_path),
                    "prompt_sha256": prompt_metrics["sha256"],
                    "prompt_guidance": prompt_metrics,
                    "reference_contract_template_id": REFERENCE_CONTRACT_TEMPLATE_ID,
                    "reference_contract_template_version": (
                        REFERENCE_CONTRACT_TEMPLATE_VERSION
                    ),
                    "input_sha256": input_sha,
                    "baseline_input_sha256": baseline_input_sha,
                    "model": model,
                    "size_requested": size,
                    "request_contract_id": IMAGE_REQUEST_CONTRACT_ID,
                    "request_contract_version": IMAGE_REQUEST_CONTRACT_VERSION,
                    "style_source": style["source"],
                    "style_source_sha256": style["source_sha256"],
                    "style_prompt_sha256": style["prompt_sha256"],
                    "style_contract": style["style_contract"],
                    "reference_images": [
                        _portable_path(output_dir, path) for path in reference_paths
                    ],
                    "reference_image_sha256": reference_hashes,
                    "reference_roles": reference_roles,
                    "upstream_director_panel": (
                        _portable_path(output_dir, director_panel)
                        if director_panel is not None
                        else None
                    ),
                    "upstream_director_panel_sha256": (
                        hashlib.sha256(director_panel.read_bytes()).hexdigest()
                        if director_panel is not None
                        else None
                    ),
                    "upstream_director_panel_included": (
                        director_composition is not None
                    ),
                    "upstream_director_panel_style": director_panel_style,
                    "upstream_director_panel_usage": (
                        "image_generation_composition_only_never_video_reference"
                        if director_panel is not None and director_composition is not None
                        else "style_incompatible_excluded_before_provider"
                        if director_panel is not None
                        else None
                    ),
                    # This field is the video-transport boundary: no PREVIS pixel
                    # is ever sent directly to Seedance. The separately named
                    # upstream field records the transformed generation lineage.
                    "previs_reference_images": [],
                    "usage": "video_model_cinematic_composition_reference",
                    "annotation_policy": "no_text_no_arrows_no_grid",
                    "character_identity_reference_mode": (
                        "text_contract_only_for_flat_shadow_material"
                        if flat_shadow_material
                        else "canonical_character_images"
                    ),
                    "phase5_correction_context": correction_context or [],
                }
                if beat_id in rejected_frame_ids and rejected_report_sha256:
                    record["supersedes_phase5_rejection_sha256"] = (
                        rejected_report_sha256
                    )
                cached = False
                if receipt_path.is_file() and image_path.is_file():
                    try:
                        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
                        same_input = previous.get("input_sha256") == input_sha
                        consumed_rejection = (
                            beat_id not in rejected_frame_ids
                            and bool(
                                previous.get(
                                    "supersedes_phase5_rejection_sha256"
                                )
                            )
                            and previous.get("baseline_input_sha256") == input_sha
                        )
                        if (
                            previous.get("kind") == CINEMATIC_FIRST_FRAME_SCHEMA
                            and previous.get("status") == "done"
                            and (same_input or consumed_rejection)
                            and previous.get("model") == model
                            and previous.get("size_requested") == size
                            and (
                                beat_id not in rejected_frame_ids
                                or previous.get(
                                    "supersedes_phase5_rejection_sha256"
                                )
                                == rejected_report_sha256
                            )
                        ):
                            _validate_image(image_path)
                            record = previous
                            record["cache_hit"] = True
                            cached = True
                    except (OSError, ValueError, json.JSONDecodeError):
                        cached = False
                if not cached:
                    prompt_path.write_text(prompt, encoding="utf-8")
                    if reference_paths and hasattr(client, "image_to_image"):
                        mode = "image_to_image"
                        result_url = client.image_to_image(
                            prompt=prompt,
                            ref_image=(
                                str(reference_paths[0])
                                if len(reference_paths) == 1
                                else [str(path) for path in reference_paths]
                            ),
                            output_path=str(image_path),
                            size=size,
                        )
                    else:
                        mode = "text_to_image"
                        result_url = client.text_to_image(
                            prompt=prompt,
                            output_path=str(image_path),
                            size=size,
                            timeout=180,
                        )
                    _validate_image(image_path)
                    record.update({
                        "status": "done",
                        "mode": mode,
                        "result_url": result_url,
                        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    })
                _write_json(receipt_path, record)
                beat["video_first_frame"] = record["image"]
                beat["video_first_frame_kind"] = CINEMATIC_FIRST_FRAME_SCHEMA
                beat["video_first_frame_receipt"] = _portable_path(
                    output_dir, receipt_path
                )
                manifest["frames"].append(record)
                previous_cinematic = image_path
                if beat_index == 1:
                    alias_path = alias_dir / f"{shot_id}.png"
                    shutil.copy2(image_path, alias_path)
                    alias_record = {
                        **record,
                        "image": _portable_path(output_dir, alias_path),
                        "image_sha256": hashlib.sha256(alias_path.read_bytes()).hexdigest(),
                        "canonical_source": record["image"],
                        "canonical_receipt": _portable_path(output_dir, receipt_path),
                    }
                    _write_json(alias_dir / f"{shot_id}.json", alias_record)
        manifest["status"] = "done"
        manifest["frame_count"] = len(manifest["frames"])
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = str(exc)
        _write_json(manifest_path, manifest)
        raise


def validate_cinematic_first_frame_artifacts(
    output_dir: Path,
    storyboard: dict[str, Any],
) -> list[str]:
    """Return provenance, style-injection, alias, and PREVIS-separation errors."""
    output_dir = Path(output_dir)
    errors: list[str] = []
    expected = 0
    for shot_index, shot in enumerate(storyboard.get("shots", []), 1):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, shot_index)
        for beat_index, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            expected += 1
            beat_id = str(beat.get("beat_id") or f"{shot_id}_P{beat_index:02d}")
            value = str(beat.get("video_first_frame") or "").strip()
            receipt_value = str(beat.get("video_first_frame_receipt") or "").strip()
            if beat.get("video_first_frame_kind") != CINEMATIC_FIRST_FRAME_SCHEMA:
                errors.append(f"{beat_id} has no cinematic first-frame kind")
            if not value or not receipt_value:
                errors.append(f"{beat_id} has no cinematic first-frame artifact")
                continue
            image_path = Path(value) if Path(value).is_absolute() else output_dir / value
            receipt_path = (
                Path(receipt_value)
                if Path(receipt_value).is_absolute()
                else output_dir / receipt_value
            )
            try:
                _validate_image(image_path)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if receipt.get("kind") != CINEMATIC_FIRST_FRAME_SCHEMA:
                    raise RuntimeError("receipt kind is not cinematic")
                if receipt.get("status") != "done" or receipt.get("image_sha256") != image_sha:
                    raise RuntimeError("receipt status/hash mismatch")
                if not receipt.get("style_source_sha256") or not receipt.get("style_prompt_sha256"):
                    raise RuntimeError("style injection receipt is missing")
                style_contract = receipt.get("style_contract")
                if (
                    not isinstance(style_contract, dict)
                    or style_contract.get("schema") != BASE_STYLE_SCHEMA
                    or not style_contract.get("base_style")
                ):
                    raise RuntimeError("controlled style contract is missing")
                if receipt.get("previs_reference_images") != []:
                    raise RuntimeError("direct-to-video PREVIS reference list is not empty")
                upstream_value = str(receipt.get("upstream_director_panel") or "")
                expected_upstream = output_dir / "director_panels" / f"{shot_id}.png"
                if expected_upstream.is_file():
                    if upstream_value != str(expected_upstream.relative_to(output_dir)):
                        raise RuntimeError("per-shot director composition input is missing")
                    if receipt.get("upstream_director_panel_sha256") != hashlib.sha256(
                        expected_upstream.read_bytes()
                    ).hexdigest():
                        raise RuntimeError("director panel lineage hash mismatch")
                    included = receipt.get("upstream_director_panel_included")
                    usage = receipt.get("upstream_director_panel_usage")
                    if included is True and usage != (
                        "image_generation_composition_only_never_video_reference"
                    ):
                        raise RuntimeError("included director panel usage is invalid")
                    if included is False and usage != (
                        "style_incompatible_excluded_before_provider"
                    ):
                        raise RuntimeError("excluded director panel usage is invalid")
                allowed_upstream = (
                    {upstream_value}
                    if upstream_value
                    and receipt.get("upstream_director_panel_included") is True
                    else set()
                )
                leaked = [
                    value
                    for value in receipt.get("reference_images", [])
                    if _is_previs_reference(output_dir / value)
                    and value not in allowed_upstream
                ]
                if leaked:
                    raise RuntimeError("non-director PREVIS leaked into cinematic generation")
                previs_value = str(beat.get("storyboard_image") or "").strip()
                if previs_value:
                    previs_path = (
                        Path(previs_value)
                        if Path(previs_value).is_absolute()
                        else output_dir / previs_value
                    )
                    if previs_path.is_file() and hashlib.sha256(
                        previs_path.read_bytes()
                    ).hexdigest() == image_sha:
                        raise RuntimeError("cinematic frame is byte-identical to PREVIS")
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                errors.append(f"{beat_id} invalid cinematic first frame: {exc}")
        first_beat = next(
            (
                beat
                for beat in (shot.get("storyboard_beats") or [])
                if isinstance(beat, dict)
            ),
            None,
        )
        if first_beat:
            alias = output_dir / "storyboard_images" / f"{shot_id}.png"
            source_value = str(first_beat.get("video_first_frame") or "")
            source = Path(source_value) if Path(source_value).is_absolute() else output_dir / source_value
            if not alias.is_file() or not source.is_file() or hashlib.sha256(
                alias.read_bytes()
            ).hexdigest() != hashlib.sha256(source.read_bytes()).hexdigest():
                errors.append(f"{shot_id} storyboard_images alias is not cinematic P01")
    try:
        manifest = json.loads(
            (output_dir / "CINEMATIC_FIRST_FRAMES.json").read_text(encoding="utf-8")
        )
        if manifest.get("kind") != CINEMATIC_FIRST_FRAMES_SCHEMA:
            errors.append("CINEMATIC_FIRST_FRAMES.json has invalid kind")
        if manifest.get("status") != "done":
            errors.append("CINEMATIC_FIRST_FRAMES.json is not complete")
        if int(manifest.get("frame_count") or 0) != expected:
            errors.append("CINEMATIC_FIRST_FRAMES.json frame count mismatch")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"CINEMATIC_FIRST_FRAMES.json unreadable: {exc}")
    return errors
