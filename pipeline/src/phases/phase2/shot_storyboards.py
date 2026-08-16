"""Generate one model-drawn hand storyboard for every director-level shot."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

SHOT_STORYBOARD_SIZE = "2560x1440"


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


def _shot_id(shot: dict[str, Any], index: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _compact(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _shot_who(shot: dict[str, Any]) -> list[Any] | None:
    """Preserve missing ``who`` separately from an explicit empty contract."""
    if "who" not in shot:
        return None
    raw = shot.get("who")
    if isinstance(raw, list):
        return raw
    return [raw] if raw else []


def _character_contract(
    characters: list[dict[str, Any]],
    who: list[Any] | None,
) -> str:
    if who == []:
        return "- 无人物镜头；禁止生成任何角色、群众、剪影或人形主体。"
    requested = {str(value).casefold() for value in who or []}
    lines = []
    for character in characters:
        names = {
            str(character.get("id") or "").casefold(),
            str(character.get("name") or "").casefold(),
            *(str(value).casefold() for value in character.get("aliases", [])),
        }
        if requested and requested.isdisjoint(names):
            continue
        appearance = character.get("appearance") or {}
        if isinstance(appearance, dict):
            description = (
                appearance.get("summary")
                or appearance.get("description")
                or json.dumps(appearance, ensure_ascii=False)
            )
        else:
            description = appearance
        description = (
            description
            or character.get("description")
            or character.get("visual_description")
        )
        if description:
            lines.append(
                f"- {character.get('name') or character.get('id')}："
                f"{_compact(description, 220)}"
            )
    contract = "\n".join(lines) or "- 严格使用 STORYBOARD.json 声明的角色设定，不自行增加人物。"
    from utils.privacy_visual_policy import (
        is_no_real_person_enabled,
        no_real_person_prompt_contract,
    )

    if is_no_real_person_enabled():
        contract = f"- {no_real_person_prompt_contract()}\n{contract}"
    return contract


def build_shot_storyboard_prompt(
    shot: dict[str, Any],
    shot_id: str,
    characters: list[dict[str, Any]],
    aspect_ratio: str = "16:9",
) -> tuple[str, list[dict[str, Any]]]:
    beats = [
        dict(beat)
        for beat in (shot.get("storyboard_beats") or [])
        if isinstance(beat, dict)
    ]
    if not beats:
        raise ValueError(f"{shot_id} has no storyboard_beats")
    who = _shot_who(shot)
    beat_lines = []
    for position, beat in enumerate(beats, 1):
        beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
        generation_mode = str(
            beat.get("generation_mode") or ""
        ).strip().lower()
        mode = {
            "multi_image": "MULTI_IMAGE",
            "tail_video_extend": "TAIL_VIDEO_EXTEND",
            "first_last_frame_bridge": "FIRST_LAST_FRAME_BRIDGE",
        }.get(generation_mode, "FRESH" if position == 1 else "EXTEND")
        beat_lines.append(
            f"故事格{position}【{beat_id} · {mode} · {float(beat.get('duration_s') or 5):g}秒】："
            f"起始状态={_compact(beat.get('start_state'))}；"
            f"本格只表现={_compact(beat.get('action'))}；"
            f"结束状态={_compact(beat.get('end_state'))}；"
            f"景别={beat.get('shot_size') or shot.get('shot_size') or 'medium'}；"
            f"运镜={beat.get('camera_movement') or shot.get('camera_movement') or 'steadicam'}。"
        )
    prompt = f"""为导演级镜头 {shot_id} 绘制一张内部动作故事板。

版式合同：
- 单张 {aspect_ratio} 故事板纸，严格按时间顺序排列 {len(beats)} 格，恰好对应 P01 到 P{len(beats):02d}。
- 每格顶部只写 {shot_id}_P01、{shot_id}_P02 这样的编号；不得增加、合并、重复或交换格子。
- 这是 {shot_id} 自己的故事板，不要画其他 Sxx 的内容。

绘画风格：
- 专业 PREVIS 手绘工作稿，黑色粗铅笔和炭笔，少量灰色阴影，快速 gesture drawing。
- 动作方向用红色手绘箭头，摄影机运动用蓝色手绘箭头。
- 不是剧照、不是完成度很高的漫画或概念图；人物身份与项目角色设定保持一致。
- 同一角色在所有格保持发型、服装、武器、受伤状态和左右站位连续。

二级分镜执行语义：
- P01 是“多图生成视频”的当前一级分镜起始构图；角色图锁身份，本格故事图锁场景、站位与当前剧情。
- 只有标记为 TAIL_VIDEO_EXTEND 的格才是容量延长格：它表示前一段最大叙事时长/动作容量不足以完整承载本 Sxx，必须从前段视频末态继续，禁止重新入场、回放或重复动作。
- 只有标记为 FIRST_LAST_FRAME_BRIDGE 的最后一格才是跨一级分镜桥接格：它只在下一 Sxx 与当前 Sxx 剧情连续时存在；Phase 6 用前段真实尾帧作首帧、下一 Sxx 的 P01 作尾帧。它不得承担当前 Sxx 尚未完成的动作，也不得提前执行下一 Sxx 的动作。
- 若下一一级分镜是换场、跳时、主体切换、回忆、梦境或其他 cut/fade/dissolve 转场，本 Sxx 不得绘制 FIRST_LAST_FRAME_BRIDGE 格。
- 每格只能细分当前 Sxx 已写明的动作与状态，不得新增角色、道具、冲突、伤亡或剧情结果。

场景：{_compact(shot.get('where') or shot.get('visual'), 260)}
角色：
{_character_contract(characters, who)}

逐格合同：
{chr(10).join(beat_lines)}

最终检查：恰好 {len(beats)} 格，严格服从每格标记的执行模式；全部当前 Sxx 动作只能由 MULTI_IMAGE/TAIL_VIDEO_EXTEND 格按原顺序覆盖，FIRST_LAST_FRAME_BRIDGE（若有）必须位于最后且只负责连续边界交接。"""
    return prompt, beats


def _build_panel_prompt(
    shot: dict[str, Any],
    beat: dict[str, Any],
    position: int,
    count: int,
    characters: list[dict[str, Any]],
    *,
    uses_director_board: bool = False,
    aspect_ratio: str = "16:9",
    correction_contract: str = "",
) -> str:
    who = _shot_who(shot)
    beat_id = str(beat.get("beat_id") or f"P{position:02d}")
    generation_mode = str(beat.get("generation_mode") or "").strip().lower()
    is_continuation = generation_mode in {
        "extend",
        "tail_video_extend",
        "first_last_frame_bridge",
    }
    continuation = (
        (
            "这是当前一级分镜内的后续剧情格。严格继承上一参考图的角色身份、服装、场景、"
            "机位轴线和动作方向，但姿态必须推进到本格的新状态；不得复制上一格。"
        )
        if is_continuation
        else (
            "这是当前一级分镜的 P01，将与角色图等多张参考图共同生成第一段视频；"
            "建立本 Sxx 的起始构图，但保持整部剧本的一镜到底空间连续性。"
        )
    )
    director_reference = (
        "参考图中包含整部影片的导演总览板。只读取其中标为本 Sxx 的面板来继承机位、"
        "人物站位、空间轴线和光影；输出必须是一张铺满画布的单格，绝不能复制总览板的网格、"
        "边框、编号或其他 Sxx 内容。"
        if uses_director_board
        else ""
    )
    previous_state_contract = (
        "上一参考图只用于继承空间轴线、机位、动作方向和上一格结束姿态；"
        "项目角色参考与下方角色合同始终优先于上一格。若上一格的发型、服装基础色、"
        "身份或武器归属有偏差，本格必须纠正，不得继续放大偏差。"
        if is_continuation
        else "项目角色参考与下方角色合同是人物身份、发型、服装基础色和装备的唯一准绳。"
    )
    bridge_contract = (
        f"这是首尾帧交接格：仍只完成 {beat.get('parent_shot_id') or '当前一级分镜'} "
        f"的剧情；结束构图必须能够无跳变地接到下一一级分镜 "
        f"{beat.get('bridge_target_beat_id') or 'P01'} 的起始状态。禁止提前执行下一镜动作。"
        if generation_mode == "first_last_frame_bridge"
        else ""
    )
    final_beat_contract = (
        "这是本镜最后一格：必须把“结束状态”作为画面最醒目的已完成事实。若结束状态包含"
        "稳定、停止、定格、落地、倒地、飞向或撞向等结果，必须清楚画出该结果；"
        "不得仍停留在搏斗、准备、过渡或前一动作中，也不得用运动线否定静止/定格。"
        if position == count
        else "这不是本镜最后一格：只推进到本格结束状态，不得抢先画后续格的结果。"
    )
    correction_section = (
        f"""

Phase 5 定向纠偏合同：
{correction_contract}
- 上述“已观察到的错误”是本轮禁止复现的负面约束，不是要继续画入画面的剧情。
- 纠偏时仍以本格起始状态、唯一动作、结束状态和角色合同为最高事实源；不得通过增加破坏、伤亡、道具或画外事件来规避问题。
- 输出前逐项确认：原偏差已经消失，且没有引入新的角色、动作、道具、环境结果或连续性跳变。
"""
        if correction_contract.strip()
        else ""
    )
    return f"""绘制一张单独的 {aspect_ratio} PREVIS 导演手绘故事格：{beat_id}（第 {position}/{count} 格）。

{continuation}
{director_reference}
{previous_state_contract}
{bridge_contract}
本格起始状态：{_compact(beat.get('start_state'))}
本格唯一可见动作：{_compact(beat.get('action'))}
本格必须到达的结束状态：{_compact(beat.get('end_state'))}
场景：{_compact(shot.get('where') or shot.get('visual'), 260)}
景别：{beat.get('shot_size') or shot.get('shot_size') or 'medium'}
运镜意图：{beat.get('camera_movement') or shot.get('camera_movement') or 'steadicam'}

角色：
{_character_contract(characters, who)}

角色与动作硬约束：
- 若角色合同启用“非真人视觉硬约束”，角色在每一格都必须保持全封闭不透明面甲，禁止露出人脸、皮肤、头发或真人肖像特征；旧剧情中的男性、脸、头发、真人写实描述不得覆盖该约束。
- 逐字遵守角色合同中的发型、服装基础色、制服类型、体型和装备；警示灯、阴影和炭笔风格只能改变受光，不得把服装基础色改成另一角色的颜色。
- 每个动作的执行者、承受者、左右位置、朝向以及武器持有者必须与“本格唯一可见动作”一致；禁止交换人物、攻守关系或武器归属。
- “解除武器/争夺武器”必须画出双方同时接触并控制同一武器的过程，不得替换为单方持枪瞄准、开枪或普通对打；其他动作也不得用相邻剧情或泛化搏斗代替。
- {final_beat_contract}
{correction_section}

风格要求：黑色粗铅笔与炭笔、少量灰色阴影、快速 gesture drawing、专业导演工作稿；
动作方向可用红色手绘箭头，摄像机运动可用蓝色箭头。人物外形严格遵守项目角色合同。
画面必须铺满 {aspect_ratio} 单格，禁止分格、拼贴、边框、大标题、字幕、对白气泡、编号和水印。
只画本格动作，不得提前表现 P{position + 1:02d} 或其他 Sxx 的剧情。生成前先核对角色→动作→对象→道具→结束状态五项；任一项冲突时必须重新构图后再输出。"""


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _compose_board(
    panel_paths: list[Path],
    labels: list[str],
    board_path: Path,
) -> None:
    with Image.open(panel_paths[0]) as first:
        ratio = first.width / max(first.height, 1)
    cell_height = 720
    cell_width = max(320, round(cell_height * ratio))
    label_height, gutter = 58, 12
    board = Image.new(
        "RGB",
        (
            len(panel_paths) * cell_width + (len(panel_paths) + 1) * gutter,
            cell_height + label_height + 2 * gutter,
        ),
        "white",
    )
    draw = ImageDraw.Draw(board)
    font = _font(30)
    for index, (path, label) in enumerate(zip(panel_paths, labels, strict=True)):
        with Image.open(path) as source:
            panel = ImageOps.fit(
                source.convert("RGB"),
                (cell_width, cell_height),
                method=Image.Resampling.LANCZOS,
            )
        x = gutter + index * (cell_width + gutter)
        board.paste(panel, (x, gutter + label_height))
        draw.text((x + 12, gutter + 10), label, fill="black", font=font)
        draw.rectangle(
            (x, gutter + label_height, x + cell_width, gutter + label_height + cell_height),
            outline=(55, 55, 55),
            width=3,
        )
    board.save(board_path, format="PNG", optimize=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_output_image_safety_rejection(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker.casefold() in message
        for marker in (
            "OutputImageSensitiveContentDetected",
            "output image may contain sensitive information",
        )
    )


def _is_transient_image_transport_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "write operation timed out",
            "read timed out",
            "connection aborted",
            "connection reset",
            "remote disconnected",
        )
    )


def _storyboard_safety_retry_prompt(prompt: str) -> str:
    """Preserve blocking/action semantics while removing graphic implications."""
    return f"""【自动安全重生成合同｜最高优先级】
- 这是完全虚构的非真人机械合成人特技预演图，不是真人打斗或现实暴力。
- 所有角色必须保持全封闭不透明机械面甲；禁止人脸、皮肤、头发或生物伤口。
- 用错开的预接触姿态、格挡姿态和红色动作箭头表达动作方向；不要画拳、肘、膝或武器真正击中身体的瞬间。
- 无血液、无伤口、无痛苦表情、无骨折、无身体损伤、无处决、无武器开火。
- 必须保留原合同的角色身份、攻守关系、空间轴线和结束状态，但把接触表现为安全的机械训练编排。

{prompt}"""


def _director_panel_references(
    output_dir: Path,
    director_board: Path,
    shot_ids: list[str],
) -> dict[str, Path]:
    """Load exact Sxx crops bound to the current director overview bytes."""
    manifest_path = director_board.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"director panel lookup requires readable {manifest_path.name}: {exc}"
        ) from exc
    extraction = manifest.get("panel_extraction")
    if (
        manifest.get("status") != "done"
        or not isinstance(extraction, dict)
        or extraction.get("schema") != "honcut.director-panels.v1"
    ):
        raise RuntimeError(
            "director storyboard has no completed exact-panel extraction receipt"
        )
    source_sha = hashlib.sha256(director_board.read_bytes()).hexdigest()
    if extraction.get("source_image_sha256") != source_sha:
        raise RuntimeError("director panel extraction belongs to different overview bytes")

    references: dict[str, Path] = {}
    for panel in manifest.get("panels", []):
        if not isinstance(panel, dict):
            continue
        shot_id = str(panel.get("shot_id") or "")
        crop_value = str(panel.get("crop") or "")
        if not shot_id or not crop_value:
            continue
        crop = Path(crop_value)
        if not crop.is_absolute():
            crop = output_dir / crop
        try:
            if hashlib.sha256(crop.read_bytes()).hexdigest() != panel.get("crop_sha256"):
                raise RuntimeError("crop hash mismatch")
            with Image.open(crop) as image:
                image.verify()
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"invalid director crop for {shot_id}: {exc}") from exc
        references[shot_id] = crop
    missing = [shot_id for shot_id in shot_ids if shot_id not in references]
    if missing:
        raise RuntimeError(
            "director storyboard is missing exact crops for: " + ", ".join(missing)
        )
    return references


def _character_reference_paths(
    output_dir: Path,
    characters: list[dict[str, Any]],
    who: list[Any] | None,
) -> list[Path]:
    """Resolve only the character references owned by this shot contract."""
    if who == []:
        return []
    requested = {str(value).casefold() for value in who or []}
    references: list[Path] = []
    for character in characters:
        names = {
            str(character.get("id") or "").casefold(),
            str(character.get("name") or "").casefold(),
            *(str(value).casefold() for value in character.get("aliases", [])),
        }
        if requested and requested.isdisjoint(names):
            continue
        character_id = str(character.get("id") or "").strip()
        if not character_id:
            continue
        character_references: list[Path] = []
        for character_dir in (
            output_dir / "characters" / character_id,
            output_dir / "characters" / "characters" / character_id,
        ):
            character_references = [
                path
                for path in (
                    character_dir / "face_closeup.png",
                    character_dir / "full_body.png",
                )
                if path.is_file() and path.stat().st_size > 0
            ]
            if not character_references:
                character_references = [
                    path
                    for path in (
                        character_dir / "closeup.png",
                        character_dir / "front.png",
                        character_dir / "side.png",
                        character_dir / "back.png",
                        *sorted(character_dir.glob("variant_*.png")),
                    )
                    if path.is_file() and path.stat().st_size > 0
                ][:2]
            if character_references:
                break
        for reference in character_references:
            if reference not in references:
                references.append(reference)
    return references


def validate_shot_storyboard_artifacts(
    output_dir: Path,
    storyboard: dict[str, Any],
) -> list[str]:
    """Return every missing/corrupt Pxx artifact in a two-level storyboard."""
    output_dir = Path(output_dir)
    errors: list[str] = []
    authored_count = 0
    for shot in storyboard.get("shots", []):
        if not isinstance(shot, dict):
            continue
        for beat in shot.get("storyboard_beats") or []:
            if not isinstance(beat, dict):
                continue
            authored_count += 1
            beat_id = str(beat.get("beat_id") or "<missing-beat-id>")
            image_value = str(beat.get("storyboard_image") or "").strip()
            if not image_value:
                errors.append(f"{beat_id} has no storyboard_image")
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = output_dir / image_path
            try:
                if not image_path.is_file() or image_path.stat().st_size <= 1024:
                    raise OSError("file missing or too small")
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                errors.append(f"{beat_id} invalid storyboard image {image_value}: {exc}")
    if authored_count:
        manifest = output_dir / "SHOT_STORYBOARDS.json"
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            if document.get("status") != "done":
                errors.append("SHOT_STORYBOARDS.json is not complete")
            if int(document.get("total_panels") or 0) != authored_count:
                errors.append(
                    "SHOT_STORYBOARDS.json panel count does not match STORYBOARD.json"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"SHOT_STORYBOARDS.json unreadable: {exc}")
    return errors


def generate_shot_storyboards(
    output_dir: Path,
    storyboard: dict[str, Any],
    characters: list[dict[str, Any]],
    *,
    client: ImageGenerationClient | None = None,
    size: str = SHOT_STORYBOARD_SIZE,
    director_storyboard_path: str | Path | None = None,
    aspect_ratio: str | None = None,
    correction_context_by_shot: dict[str, list[dict[str, Any]]] | None = None,
    correction_attempt: int = 0,
) -> dict[str, Any]:
    """Generate each Pxx as 16:9 model art, then compose the Sxx overview."""
    output_dir = Path(output_dir)
    boards_dir = output_dir / "shot_storyboards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    beats_dir = output_dir / "storyboard_beats"
    beats_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "storyboard_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(client, "model", None) or "doubao-seedream-5.0-lite"
    contract: dict[str, Any] = {
        "kind": "honcut.shot_storyboards.v1",
        "version": 1,
        "status": "running",
        "provider": "seedream",
        "model": model,
        "shots": [],
    }
    correction_context_by_shot = correction_context_by_shot or {}
    if correction_context_by_shot:
        contract["correction"] = {
            "attempt": int(correction_attempt),
            "shot_ids": sorted(correction_context_by_shot),
        }
    if aspect_ratio is None:
        aspect_ratio = str(storyboard.get("aspect_ratio") or "").strip()
    if not aspect_ratio:
        try:
            width, height = (int(value) for value in size.lower().split("x", 1))
            divisor = math.gcd(width, height)
            aspect_ratio = f"{width // divisor}:{height // divisor}"
        except (TypeError, ValueError):
            aspect_ratio = "16:9"
    contract["aspect_ratio"] = aspect_ratio
    contract["size_requested"] = size
    manifest_path = output_dir / "SHOT_STORYBOARDS.json"
    _write_json(manifest_path, contract)
    if client is None:
        from clients.seedream_client import SeedreamClient

        client = SeedreamClient()
        contract["model"] = client.model

    if director_storyboard_path is None:
        director_ref = storyboard.get("director_storyboard") or {}
        director_value = director_ref.get("image") if isinstance(director_ref, dict) else None
        director_storyboard_path = director_value or "director_storyboard.png"
    director_board = Path(director_storyboard_path)
    if not director_board.is_absolute():
        director_board = output_dir / director_board
    if not director_board.is_file() or director_board.stat().st_size == 0:
        director_board = None
    contract["director_storyboard"] = (
        str(director_board.relative_to(output_dir))
        if director_board is not None and director_board.is_relative_to(output_dir)
        else str(director_board) if director_board is not None else None
    )
    authored_shot_ids = [
        _shot_id(shot, index)
        for index, shot in enumerate(storyboard.get("shots", []), 1)
        if isinstance(shot, dict)
    ]
    director_panels = (
        _director_panel_references(
            output_dir,
            director_board,
            authored_shot_ids,
        )
        if director_board is not None and authored_shot_ids
        else {}
    )
    contract["director_panel_schema"] = (
        "honcut.director-panels.v1" if director_panels else None
    )

    previous_storyboard_panel: Path | None = None
    try:
        for index, shot in enumerate(storyboard.get("shots", []), 1):
            if not isinstance(shot, dict):
                continue
            shot_id = _shot_id(shot, index)
            prompt, beats = build_shot_storyboard_prompt(
                shot,
                shot_id,
                characters,
                aspect_ratio,
            )
            prompt_path = boards_dir / f"{shot_id}_prompt.txt"
            board_path = boards_dir / f"{shot_id}.png"
            prompt_path.write_text(prompt, encoding="utf-8")
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            director_panel = director_panels.get(shot_id)
            correction_issues = correction_context_by_shot.get(shot_id, [])
            correction_lines: list[str] = []
            for issue_index, issue in enumerate(correction_issues, 1):
                details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
                expected = _compact(details.get("expected"), 320)
                observed = _compact(details.get("observed"), 320)
                message = _compact(issue.get("message"), 420)
                code = str(issue.get("code") or "QA").upper()
                correction_lines.append(
                    f"- 纠偏项 {issue_index}（{code}）：必须满足="
                    f"{expected or '严格恢复本格动作与结束状态合同'}；"
                    f"已观察到且禁止复现={observed or message}。"
                )
            correction_contract = "\n".join(correction_lines)
            if correction_contract:
                correction_contract = (
                    f"这是第 {int(correction_attempt)} 轮自动纠偏，只修复 {shot_id}。\n"
                    + correction_contract
                )
            record = {
                "shot_id": shot_id,
                "board": str(board_path.relative_to(output_dir)),
                "prompt": str(prompt_path.relative_to(output_dir)),
                "prompt_sha256": prompt_sha,
                "panel_count": len(beats),
                "director_panel": (
                    str(director_panel.relative_to(output_dir))
                    if director_panel is not None and director_panel.is_relative_to(output_dir)
                    else str(director_panel) if director_panel is not None else None
                ),
                "status": "planned",
            }
            if correction_issues:
                record["correction"] = {
                    "attempt": int(correction_attempt),
                    "issues": correction_issues,
                }
            panel_records = []
            panel_paths: list[Path] = []
            previous_panel = (
                previous_storyboard_panel
                if str(shot.get("boundary_before") or "").strip().lower()
                == "continuous"
                else None
            )
            character_references = _character_reference_paths(
                output_dir,
                characters,
                _shot_who(shot),
            )
            for position, beat in enumerate(beats, 1):
                beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
                panel_prompt = _build_panel_prompt(
                    shot,
                    beat,
                    position,
                    len(beats),
                    characters,
                    uses_director_board=director_panel is not None,
                    aspect_ratio=aspect_ratio,
                    correction_contract=correction_contract,
                )
                panel_prompt_path = beats_dir / f"{beat_id}_prompt.txt"
                panel_path = beats_dir / f"{beat_id}.png"
                panel_sidecar = beats_dir / f"{beat_id}.json"
                panel_prompt_path.write_text(panel_prompt, encoding="utf-8")
                reference_paths: list[Path] = []
                reference_paths.extend(character_references)
                if previous_panel is not None:
                    reference_paths.append(previous_panel)
                if director_panel is not None:
                    reference_paths.append(director_panel)
                reference_hashes = [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in reference_paths
                ]
                panel_sha = hashlib.sha256(
                    f"{panel_prompt}\nreferences={','.join(reference_hashes)}".encode("utf-8")
                ).hexdigest()
                panel_record = {
                    "beat_id": beat_id,
                    "position": position,
                    "mode": "image_to_image" if reference_paths else "text_to_image",
                    "image": str(panel_path.relative_to(output_dir)),
                    "prompt": str(panel_prompt_path.relative_to(output_dir)),
                    "prompt_sha256": panel_sha,
                    "model": contract["model"],
                    "character_references": [
                        str(path.relative_to(output_dir))
                        if path.is_relative_to(output_dir)
                        else str(path)
                        for path in character_references
                    ],
                    "reference_images": [
                        str(path.relative_to(output_dir))
                        if path.is_relative_to(output_dir)
                        else str(path)
                        for path in reference_paths
                    ],
                    "status": "planned",
                }
                cached = False
                if panel_sidecar.is_file() and panel_path.is_file():
                    try:
                        previous_record = json.loads(
                            panel_sidecar.read_text(encoding="utf-8")
                        )
                        if (
                            previous_record.get("status") == "done"
                            and previous_record.get("prompt_sha256") == panel_sha
                            and previous_record.get("model") == contract["model"]
                        ):
                            with Image.open(panel_path) as image:
                                image.verify()
                            panel_record = previous_record
                            panel_record["cache_hit"] = True
                            cached = True
                    except (OSError, ValueError, json.JSONDecodeError):
                        cached = False
                if not cached:
                    def generate_panel(generation_prompt: str):
                        if reference_paths and hasattr(client, "image_to_image"):
                            return client.image_to_image(
                                prompt=generation_prompt,
                                ref_image=(
                                    str(reference_paths[0])
                                    if len(reference_paths) == 1
                                    else [str(path) for path in reference_paths]
                                ),
                                output_path=str(panel_path),
                                size=size,
                            )
                        panel_record["mode"] = "text_to_image"
                        return client.text_to_image(
                            prompt=generation_prompt,
                            output_path=str(panel_path),
                            size=size,
                            timeout=180,
                        )

                    transport_retries = 0

                    def generate_with_transport_retry(generation_prompt: str):
                        nonlocal transport_retries
                        max_transport_retries = 2
                        for attempt in range(max_transport_retries + 1):
                            try:
                                return generate_panel(generation_prompt)
                            except Exception as exc:
                                if (
                                    not _is_transient_image_transport_error(exc)
                                    or attempt >= max_transport_retries
                                ):
                                    raise
                                transport_retries += 1
                                print(
                                    f"  [seedream] {beat_id} 参考图传输中断；"
                                    f"同请求限次重试 {attempt + 1}/{max_transport_retries}",
                                    flush=True,
                                )

                    try:
                        result_url = generate_with_transport_retry(panel_prompt)
                    except Exception as exc:
                        if not _is_output_image_safety_rejection(exc):
                            raise
                        safety_prompt = _storyboard_safety_retry_prompt(panel_prompt)
                        safety_prompt_path = beats_dir / f"{beat_id}_safety_retry_prompt.txt"
                        safety_prompt_path.write_text(safety_prompt, encoding="utf-8")
                        print(
                            f"  [seedream] {beat_id} 输出安全拒绝；改为非接触机械特技预演，限次重试 1 次",
                            flush=True,
                        )
                        result_url = generate_with_transport_retry(safety_prompt)
                        panel_record["safety_retry"] = {
                            "reason": "output_image_sensitive_content",
                            "attempts": 1,
                            "policy": "synthetic_non_contact_stunt_v1",
                            "prompt": str(safety_prompt_path.relative_to(output_dir)),
                            "prompt_sha256": hashlib.sha256(
                                safety_prompt.encode("utf-8")
                            ).hexdigest(),
                        }
                    if transport_retries:
                        panel_record["transport_retry"] = {
                            "attempts": transport_retries,
                            "policy": "same_request_bounded_transport_retry_v1",
                        }
                    if not panel_path.is_file() or panel_path.stat().st_size == 0:
                        raise RuntimeError(f"Seedream returned without {panel_path.name}")
                    with Image.open(panel_path) as image:
                        image.verify()
                    panel_record.update({
                        "status": "done",
                        "result_url": result_url,
                        "reference_image_sha256": reference_hashes,
                    })
                _write_json(panel_sidecar, panel_record)
                beat["storyboard_image"] = panel_record["image"]
                panel_records.append(panel_record)
                panel_paths.append(panel_path)
                previous_panel = panel_path
            _compose_board(
                panel_paths,
                [str(beat.get("beat_id")) for beat in beats],
                board_path,
            )
            shutil.copy2(panel_paths[0], image_dir / f"{shot_id}.png")
            record.update({
                "status": "done",
                "model": contract["model"],
                "panels": panel_records,
            })
            shot["storyboard_board"] = record["board"]
            shot["storyboard_beats"] = beats
            _write_json(boards_dir / f"{shot_id}.json", record)
            contract["shots"].append(record)
            if panel_paths:
                previous_storyboard_panel = panel_paths[-1]
            _write_json(manifest_path, contract)
        contract["status"] = "done"
        contract["total_boards"] = len(contract["shots"])
        contract["total_panels"] = sum(
            int(item.get("panel_count") or 0) for item in contract["shots"]
        )
        _write_json(manifest_path, contract)
        return contract
    except Exception as exc:
        contract["status"] = "error"
        contract["error"] = str(exc)
        _write_json(manifest_path, contract)
        raise
