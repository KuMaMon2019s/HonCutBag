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
from utils.character_body_contracts import character_visual_description
from utils.body_action_contracts import body_action_prompt
from utils.character_reference_contracts import (
    character_identity_detail_items,
    identity_detail_prompt_items,
)
from utils.camera_motion_contracts import (
    camera_motion_negative_prompt,
    camera_motion_prompt,
)
from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    temporal_visual_negative_prompt,
    temporal_visual_prompt,
)

SHOT_STORYBOARD_SIZE = "2560x1440"
SHOT_STORYBOARD_GRID_COLUMNS = 3
SHOT_STORYBOARD_GRID_ROWS = 3
SHOT_STORYBOARD_GRID_CELLS = (
    SHOT_STORYBOARD_GRID_COLUMNS * SHOT_STORYBOARD_GRID_ROWS
)


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


def _edge_handle_contract(beat: dict[str, Any]) -> str:
    incoming = float(beat.get("incoming_bridge_handle_s") or 0)
    outgoing = float(beat.get("outgoing_bridge_handle_s") or 0)
    clauses = []
    if incoming > 0:
        clauses.append(
            f"开头{incoming:g}秒保持起始状态，仅自然微动，之后才开始新动作"
        )
    if outgoing > 0:
        clauses.append(
            f"结尾前完成全部剧情动作，最后{outgoing:g}秒稳定保持结束状态"
        )
    return "；".join(clauses) or "无跨一级分镜边界把手"


def _narrative_grid_contract(
    shot: dict[str, Any],
    shot_id: str,
    beats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand one primary Sxx plus its ordered Pxx beats into nine review cells.

    The grid is a director/LLM narrative artifact, not a provider frame ledger.
    Cells subdivide only already-authored start/action/end facts and therefore do
    not create new plot events when one Pxx needs more than three review cells.
    """
    base, remainder = divmod(SHOT_STORYBOARD_GRID_CELLS, len(beats))
    cells: list[dict[str, Any]] = []
    for beat_index, beat in enumerate(beats):
        cell_count = base + (1 if beat_index < remainder else 0)
        beat_id = str(beat.get("beat_id") or f"{shot_id}_P{beat_index + 1:02d}")
        start_state = _compact(beat.get("start_state"), 320)
        action = _compact(beat.get("action"), 520)
        end_state = _compact(beat.get("end_state"), 320)
        for local_index in range(cell_count):
            if local_index == 0:
                stage = "start"
                visible_fact = start_state or action
            elif local_index == cell_count - 1:
                stage = "end"
                visible_fact = end_state or action
            else:
                stage = "action_progress"
                progress = round(local_index / max(cell_count - 1, 1), 3)
                visible_fact = (
                    f"仅演绎本格既有动作的 {progress:.0%} 进度：{action}；"
                    "不得增加新事件、角色、道具或结果"
                )
            cells.append({
                "cell": len(cells) + 1,
                "label": f"{shot_id}_G{len(cells) + 1:02d}",
                "primary_shot_id": shot_id,
                "secondary_beat_id": beat_id,
                "secondary_beat_position": beat_index + 1,
                "stage": stage,
                "visible_fact": visible_fact,
                "camera_movement": beat.get("camera_movement")
                or shot.get("camera_movement")
                or "steadicam",
                "rendered_annotations": ["cell_label"],
            })
    if len(cells) != SHOT_STORYBOARD_GRID_CELLS:
        raise RuntimeError(
            f"{shot_id} narrative grid produced {len(cells)} cells, expected "
            f"{SHOT_STORYBOARD_GRID_CELLS}"
        )
    return cells


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
        description = character_visual_description(character)
        if description:
            lines.append(
                f"- {character.get('name') or character.get('id')}："
                f"{_compact(description, 1400)}"
            )
        identity_items = character_identity_detail_items(character)
        if identity_items:
            lines.append(
                f"  身份道具细节参考：{identity_detail_prompt_items(identity_items)}。"
                "body_attached 项保持在固定佩挂点；isolated_handheld 项只有本格动作明确调用时才出现，"
                "出现时几何、颜色、材质和标记必须与独立细节板一致。"
            )
    contract = "\n".join(lines) or "- 严格使用 STORYBOARD.json 声明的角色设定，不自行增加人物。"
    from utils.privacy_visual_policy import (
        is_no_real_person_enabled,
        is_synthetic_visual_identity_policy,
        no_real_person_prompt_contract,
    )

    if is_no_real_person_enabled() or any(
        is_synthetic_visual_identity_policy(character.get("visual_identity_policy"))
        for character in characters
        if isinstance(character, dict)
    ):
        contract = f"- {no_real_person_prompt_contract()}\n{contract}"
    return contract


def build_shot_storyboard_prompt(
    shot: dict[str, Any],
    shot_id: str,
    characters: list[dict[str, Any]],
    aspect_ratio: str = "16:9",
) -> tuple[str, list[dict[str, Any]]]:
    temporal_contract = apply_temporal_visual_contract(shot)
    temporal_section = (
        "\n时间段视觉硬合同：\n"
        f"- {temporal_visual_prompt(temporal_contract)}。\n"
        f"- 禁止：{temporal_visual_negative_prompt(temporal_contract)}。\n"
        if temporal_contract
        else ""
    )
    beats = [
        dict(beat)
        for beat in (shot.get("storyboard_beats") or [])
        if isinstance(beat, dict)
    ]
    if not beats:
        raise ValueError(f"{shot_id} has no storyboard_beats")
    narrative_grid = _narrative_grid_contract(shot, shot_id, beats)
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
        beat_camera_contract = camera_motion_prompt({**shot, **beat})
        beat_choreography = body_action_prompt({**shot, **beat})
        beat_lines.append(
            f"故事格{position}【{beat_id} · {mode} · {float(beat.get('duration_s') or 5):g}秒】："
            f"起始状态={_compact(beat.get('start_state'))}；"
            f"本格只表现={_compact(beat.get('action'))}；"
            f"结束状态={_compact(beat.get('end_state'))}；"
            f"景别={beat.get('shot_size') or shot.get('shot_size') or 'medium'}；"
            f"运镜={beat.get('camera_movement') or shot.get('camera_movement') or 'steadicam'}；"
            f"边界把手={_edge_handle_contract(beat)}；"
            f"物理合同={beat_camera_contract}；"
            f"逐拍肢体动作谱={_compact(beat_choreography, 1200) or '无专项舞蹈/格斗动作'}。"
        )
    grid_lines = [
        (
            f"格{cell['cell']}【{cell['label']}｜一级={shot_id}｜"
            f"二级={cell['secondary_beat_id']}｜阶段={cell['stage']}】："
            f"{cell['visible_fact']}；运镜={cell['camera_movement']}。"
        )
        for cell in narrative_grid
    ]
    prompt = f"""为导演级镜头 {shot_id} 绘制一张九宫格剧情演绎故事板。

版式合同：
- 单张 {aspect_ratio} 故事板纸，严格使用 3 列 × 3 行九宫格，恰好 9 格；格子等宽等高，阅读顺序从左到右、从上到下。
- 每格顶部只写 {shot_id}_G01 到 {shot_id}_G09；不得增加、合并、重复、交换、跨格或省略格子。
- 九宫格按一级镜头 {shot_id} 的剧情顺序，完整演绎二级 P01 到 P{len(beats):02d}；同一 Pxx 占多格时只能细分既有动作进度，不得发明新剧情。
- 这是 {shot_id} 自己的故事板，不要画其他 Sxx 的内容。

绘画风格：
- 专业 PREVIS 手绘工作稿，黑色粗铅笔和炭笔，少量灰色阴影，快速 gesture drawing。
- 动作方向用红色手绘箭头，摄影机运动用蓝色手绘箭头。
- 不是剧照、不是完成度很高的漫画或概念图；人物身份与项目角色设定保持一致。
- 同一角色在所有格保持发型、服装、武器、受伤状态和左右站位连续。

二级分镜执行语义：
- P01 是“多图生成视频”的当前一级分镜起始构图；角色图锁身份，本格故事图锁场景、站位与当前剧情。
- P01 视频必须为 8–15 秒；只有标记为 TAIL_VIDEO_EXTEND 的格才是 6–10 秒容量延长格。延长格表示前一段时长/动作容量不足，必须从前段视频末态继续，禁止重新入场、回放或重复动作。
- 跨一级分镜桥接不占 Pxx。全部 Sxx/Pxx 一级分镜完成后，另生成一张相邻 Sxx 之间的过渡分镜；它只规划连续动作与运镜路径，不新增剧情。
- 过渡分镜不是视频首尾帧替代品。所有一级视频完成后，Phase 6 仍只以相邻成片的真实尾帧和真实首帧生成 4–6 秒桥接视频。
- 换场、跳时、主体切换、回忆、梦境或其他 cut/fade/dissolve 边界不生成桥接视频，由 Phase 8 添加转场特效。
- 每格只能细分当前 Sxx 已写明的动作与状态，不得新增角色、道具、冲突、伤亡或剧情结果。

场景：{_compact(shot.get('where') or shot.get('visual'), 260)}
{temporal_section}
角色：
{_character_contract(characters, who)}

角色职责与道具合同：
- 每个角色只执行逐格合同明确分配给自己的动作；不得把其他角色或群体的动作复制给旁观者、记录者、驾驶者、守卫或任何未被指定的角色。
- 保留源文本中的道具类型、持有者和使用方式；不得替换设备、交换道具或让角色无故放下道具。
- 舞蹈、格斗、功夫和武术格必须画清左右侧、执行肢体、步法、躯干旋转、重心转移、方向、接触点和终态；不得用“复杂动作”“跳舞”“激烈格斗”或动作箭头代替身体姿态。

摄影与人体透视禁止项：
- {camera_motion_negative_prompt(shot)}。

二级 Pxx 执行合同：
{chr(10).join(beat_lines)}

九宫格演绎合同：
{chr(10).join(grid_lines)}

最终检查：画面必须是完整 3×3、恰好 9 格，G01 至 G09 连续且各出现一次；严格服从每个格子绑定的 Pxx 和阶段。全部当前 Sxx 动作只能按原顺序覆盖，不得绘制跨一级分镜桥接格，也不得把九宫格误作视频首帧或成片质感参考。"""
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
    is_last_content_beat: bool | None = None,
) -> str:
    temporal_contract = apply_temporal_visual_contract(shot)
    temporal_section = (
        f"时间段视觉硬合同：{temporal_visual_prompt(temporal_contract)}\n"
        f"时间段禁止项：{temporal_visual_negative_prompt(temporal_contract)}\n"
        if temporal_contract
        else ""
    )
    who = _shot_who(shot)
    beat_id = str(beat.get("beat_id") or f"P{position:02d}")
    choreography_section = body_action_prompt({**shot, **beat})
    generation_mode = str(beat.get("generation_mode") or "").strip().lower()
    is_bridge = generation_mode == "first_last_frame_bridge"
    if is_last_content_beat is None:
        is_last_content_beat = position == count and not is_bridge
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
        if is_bridge
        else ""
    )
    if is_bridge:
        final_beat_contract = (
            "这是桥接预览格，不是新的剧情动作格。只能保持上一剧情格已经完成的结束状态，"
            "并为下一 Sxx 的 P01 构图留出连续路径；不得回到动作进行中，不得新增、复制或"
            "删除人物，不得改变武器归属。多张参考图里的同名角色都是同一个角色，绝不能"
            "把不同参考图复制成多个实体。"
        )
    elif is_last_content_beat:
        final_beat_contract = (
            "这是本 Sxx 最后一个承载剧情的故事格；本格必须完整"
            "完成当前 Sxx 的全部剧情。画面是所有列出动作完成后的终态快照，不是中间过程拼贴。"
            "必须把“结束状态”作为最醒目的已完成事实；稳定、停止、定格、落地、倒地、飞向或"
            "撞向等结果必须清楚可见，不得仍停留在搏斗、争夺、准备或前一动作中，也不得用"
            "运动线否定静止或定格。"
        )
    else:
        final_beat_contract = (
            "本格仍必须画出本格动作完成后的结束状态快照；只是不准抢先执行后续剧情格的动作。"
        )
    action_text = _compact(beat.get("action"), 500)
    end_state_text = _compact(beat.get("end_state"), 500)
    semantic_text = f"{action_text} {end_state_text}".casefold()
    disarm_requested = bool(
        ("解除" in semantic_text and "武器" in semantic_text)
        or any(
            marker in semantic_text
            for marker in ("缴械", "disarm", "weapon removed")
        )
    )
    disarm_contract = (
        "- 本格要求的是解除武器的完成态：结束画面中敌方不得继续握持武器，双方不得仍共同争夺"
        "同一武器；武器只能已由 Agent 控制、固定在远离敌方的位置或独立失重漂浮。画面中恰好"
        "一件该武器，禁止复制武器。\n"
        if disarm_requested
        else (
            "- 只有剧情明确写成尚未完成的‘争夺武器’时，才可画双方共同控制同一武器的过程；"
            "不得替换为单方持枪瞄准、开枪或普通对打。\n"
        )
    )
    canonical_cast = list(dict.fromkeys(
        str(name) for name in (who or []) if str(name).strip()
    ))
    cast_contract = (
        f"- 画面中恰好出现 {len(canonical_cast)} 个具名角色实体："
        f"{'、'.join(canonical_cast) if canonical_cast else '零个角色'}。"
        "每个角色只出现一次；多张近景、全身、上一格和导演参考图只是同一身份的不同参考，"
        "绝不能复制成额外人物。\n"
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
本格逐拍肢体动作谱：{_compact(choreography_section, 1600) or '无专项舞蹈/格斗动作'}
本格必须到达的结束状态：{_compact(beat.get('end_state'))}
场景：{_compact(shot.get('where') or shot.get('visual'), 260)}
{temporal_section}景别：{beat.get('shot_size') or shot.get('shot_size') or 'medium'}
运镜意图：{beat.get('camera_movement') or shot.get('camera_movement') or 'steadicam'}
运镜物理硬合同：{camera_motion_prompt({**shot, **beat})}
边界把手合同：{_edge_handle_contract(beat)}

角色：
{_character_contract(characters, who)}

角色与动作硬约束：
- 若角色合同启用“非真人视觉硬约束”，每个角色必须逐格保持自己声明的面纱/遮罩、图形化妆、面部纹样、机械纹理、非人材质或其他合成妆造锚点；禁止退化为无妆造的自然真人，也禁止擅自给所有人套用同一种头盔。
- 逐字遵守角色合同中的发型、服装基础色、制服类型、体型和装备；警示灯、阴影和炭笔风格只能改变受光，不得把服装基础色改成另一角色的颜色。
- 每个动作的执行者、承受者、左右位置、朝向以及武器持有者必须与“本格唯一可见动作”一致；禁止交换人物、攻守关系或武器归属。
- 舞蹈/格斗/功夫/武术动作必须逐拍执行上述肢体动作谱，明确左挡、右闪、支撑侧、摆动侧、躯干旋转、重心转移、接触点和终态；原文点名的托马斯、铁山靠等招式不得泛化、镜像或换招。
- 每个角色只执行本格明确分配给自己的动作；不得把其他角色或群体的动作复制给旁观者、记录者、驾驶者、守卫或任何未被指定的角色。
- 严格保留本格声明的道具类型、持有者和使用方式；不得替换设备、交换道具或让角色无故放下道具。
- 静态故事格只能画一个时间点，不得把多个时间点或动作过程拼贴在一起；但若本格给主体分配了肢体或位移动作，必须选择达到结束状态时仍具动力学信息的瞬间，以关节弯曲、肢体伸展、重心偏移、接触关系和动作方向清楚表现该动作。不得把“终态”误画成人物中性站立；只有源合同明确要求静止、停止或定格时才画静止姿态。
- 主动作角色必须是画面的主要运动来源。不得只让背景人群、车辆、光影、粒子、衣物、头发或摄影机产生动感，而让被分配动作的主体保持参考图原姿势；背景与运镜只能辅助，不能替代主体动作。
- 人物构图必须遵守运镜合同中的 50–85mm 自然透视与稳定人物尺度；禁止：{camera_motion_negative_prompt({**shot, **beat})}。
{disarm_contract}{cast_contract}- 其他动作不得用相邻剧情或泛化搏斗代替。
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


def _normalize_nine_grid_board(board_path: Path, shot_id: str) -> dict[str, Any]:
    """Overlay an exact 3x3 machine-readable boundary on model-drawn board art."""
    with Image.open(board_path) as source:
        board = source.convert("RGB")
    width, height = board.size
    if width < 300 or height < 180:
        raise RuntimeError(
            f"{shot_id} nine-grid storyboard is too small: {width}x{height}"
        )
    draw = ImageDraw.Draw(board)
    gutter = max(8, round(min(width / 3, height / 3) * 0.025))
    border = max(3, gutter // 3)
    label_font = _font(max(18, round(height / 42)))
    cells: list[dict[str, Any]] = []
    for index in range(SHOT_STORYBOARD_GRID_CELLS):
        row, column = divmod(index, SHOT_STORYBOARD_GRID_COLUMNS)
        left = round(column * width / SHOT_STORYBOARD_GRID_COLUMNS)
        right = round((column + 1) * width / SHOT_STORYBOARD_GRID_COLUMNS)
        top = round(row * height / SHOT_STORYBOARD_GRID_ROWS)
        bottom = round((row + 1) * height / SHOT_STORYBOARD_GRID_ROWS)
        if column:
            draw.rectangle(
                (left - gutter // 2, 0, left + math.ceil(gutter / 2), height),
                fill="white",
            )
        if row:
            draw.rectangle(
                (0, top - gutter // 2, width, top + math.ceil(gutter / 2)),
                fill="white",
            )
        cells.append({
            "cell": index + 1,
            "label": f"{shot_id}_G{index + 1:02d}",
            "grid_row": row,
            "grid_column": column,
            "bbox_px": [left, top, right, bottom],
        })
    for cell in cells:
        left, top, right, bottom = cell["bbox_px"]
        inset = max(2, gutter // 2)
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=(20, 20, 20),
            width=border,
        )
        text_x = left + inset + border + 6
        text_y = top + inset + border + 4
        text_bbox = draw.textbbox((text_x, text_y), cell["label"], font=label_font)
        padding = max(4, border)
        draw.rectangle(
            (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding,
            ),
            fill="white",
        )
        draw.text((text_x, text_y), cell["label"], fill="black", font=label_font)
    temporary = board_path.with_suffix(".png.tmp")
    board.save(temporary, format="PNG", optimize=True)
    temporary.replace(board_path)
    return {
        "schema": "honcut.shot-storyboard-grid.v1",
        "columns": SHOT_STORYBOARD_GRID_COLUMNS,
        "rows": SHOT_STORYBOARD_GRID_ROWS,
        "cell_count": SHOT_STORYBOARD_GRID_CELLS,
        "reading_order": "left_to_right_top_to_bottom",
        "cells": cells,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_path(output_dir: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else output_dir / path


def _portable_path(output_dir: Path, path: Path) -> str:
    return (
        str(path.relative_to(output_dir))
        if path.is_relative_to(output_dir)
        else str(path)
    )


def _build_primary_bridge_storyboard_prompt(
    bridge: dict[str, Any],
    source_beat: dict[str, Any],
    target_beat: dict[str, Any],
    *,
    aspect_ratio: str,
) -> str:
    """Describe one non-narrative midpoint between adjacent Sxx boards."""
    bridge_id = str(bridge.get("bridge_id") or "Sxx__Sxx")
    source_shot_id = str(bridge.get("source_shot_id") or "上一一级分镜")
    target_shot_id = str(bridge.get("target_shot_id") or "下一一级分镜")
    return f"""绘制一张单独的 {aspect_ratio} PREVIS 手绘过渡分镜：{bridge_id}。

参考图合同：
- 图片1是 {source_shot_id} 最后一个 Pxx 故事格，代表上一一级分镜已经完成的结束状态。
- 图片2是 {target_shot_id} 的 P01 故事格，代表下一一级分镜尚未执行新动作时的起始状态。
- 两张参考图中的同名角色是同一实体；严格保持人物身份、数量、服装、道具归属、场景轴线、光影和屏幕方向连续。

过渡任务：
- 只画图片1到图片2之间一个时间点的连续过渡姿态与摄影机路径；不得照抄任一端点，也不得做拼贴、叠化、双重曝光或左右对比图。
- 上一状态：{_compact(bridge.get('start_state') or source_beat.get('end_state'), 420)}
- 过渡动作：{_compact(bridge.get('action_prompt'), 520)}
- 下一状态：{_compact(bridge.get('end_state') or target_beat.get('start_state'), 420)}
- 角色若处于连续身体动作中，必须通过躯干、四肢、关节、重心、接触关系与道具运动画出物理上可达的中间姿态；不得让角色保持图片1原姿势，只让背景、光影或摄影机移动。
- 不得提前执行 {target_shot_id} 的新剧情动作，不得重复 {source_shot_id} 已完成的动作，不得新增角色、道具、事件或剧情结果。

用途边界：这张图只用于导演检查跨一级分镜的连续路径，不会替代视频生成的首尾帧。Phase 6 必须继续使用两段已完成一级视频的真实尾帧和真实首帧。

风格要求：黑色粗铅笔与炭笔、少量灰色阴影、快速 gesture drawing、专业导演工作稿；主体动作方向可用红色手绘箭头，摄影机运动可用蓝色手绘箭头。画面铺满 {aspect_ratio} 单格，禁止分格、边框、标题、字幕、对白气泡、编号和水印。"""


def _generate_primary_bridge_storyboards(
    output_dir: Path,
    storyboard: dict[str, Any],
    *,
    client: ImageGenerationClient,
    model: str,
    size: str,
    aspect_ratio: str,
) -> list[dict[str, Any]]:
    """Generate bridge boards only after every ordinary Sxx/Pxx board exists."""
    bridges = [
        bridge
        for bridge in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(bridge, dict)
    ]
    if not bridges:
        return []
    if not hasattr(client, "image_to_image"):
        raise RuntimeError(
            "primary bridge storyboards require a two-reference image_to_image client"
        )

    shots = {
        _shot_id(shot, index): shot
        for index, shot in enumerate(storyboard.get("shots", []), 1)
        if isinstance(shot, dict)
    }
    bridge_dir = output_dir / "storyboard_bridges"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for bridge in bridges:
        bridge_id = str(
            bridge.get("bridge_id")
            or f"{bridge.get('source_shot_id')}__{bridge.get('target_shot_id')}"
        )
        source_shot = shots.get(str(bridge.get("source_shot_id") or ""))
        target_shot = shots.get(str(bridge.get("target_shot_id") or ""))
        source_beats = [
            beat
            for beat in ((source_shot or {}).get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        target_beats = [
            beat
            for beat in ((target_shot or {}).get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        if not source_beats or not target_beats:
            raise RuntimeError(
                f"{bridge_id} transition storyboard requires both adjacent Pxx chains"
            )
        source_beat = source_beats[-1]
        target_beat = target_beats[0]
        source_image = _artifact_path(
            output_dir, source_beat.get("storyboard_image")
        )
        target_image = _artifact_path(
            output_dir, target_beat.get("storyboard_image")
        )
        for label, path in (
            ("source final Pxx", source_image),
            ("target P01", target_image),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"{bridge_id} {label} is missing: {path}")

        prompt = _build_primary_bridge_storyboard_prompt(
            bridge,
            source_beat,
            target_beat,
            aspect_ratio=aspect_ratio,
        )
        image_path = bridge_dir / f"{bridge_id}.png"
        prompt_path = bridge_dir / f"{bridge_id}_prompt.txt"
        sidecar_path = bridge_dir / f"{bridge_id}.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        reference_paths = [source_image, target_image]
        reference_hashes = [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in reference_paths
        ]
        prompt_sha = hashlib.sha256(
            f"{prompt}\nreferences={','.join(reference_hashes)}".encode()
        ).hexdigest()
        record: dict[str, Any] = {
            "bridge_id": bridge_id,
            "source_shot_id": str(bridge.get("source_shot_id") or ""),
            "target_shot_id": str(bridge.get("target_shot_id") or ""),
            "image": _portable_path(output_dir, image_path),
            "prompt": _portable_path(output_dir, prompt_path),
            "prompt_sha256": prompt_sha,
            "model": model,
            "reference_images": [
                _portable_path(output_dir, path) for path in reference_paths
            ],
            "reference_image_sha256": reference_hashes,
            "generation_phase": "post_primary_storyboards",
            "usage": "visual_continuity_plan_not_video_endpoint",
            "status": "planned",
        }
        cached = False
        if sidecar_path.is_file() and image_path.is_file():
            try:
                previous = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if (
                    previous.get("status") == "done"
                    and previous.get("prompt_sha256") == prompt_sha
                    and previous.get("model") == model
                ):
                    with Image.open(image_path) as image:
                        image.verify()
                    record = previous
                    record["cache_hit"] = True
                    cached = True
            except (OSError, ValueError, json.JSONDecodeError):
                cached = False
        if not cached:
            for attempt in range(3):
                try:
                    result_url = client.image_to_image(
                        prompt=prompt,
                        ref_image=[str(path) for path in reference_paths],
                        output_path=str(image_path),
                        size=size,
                    )
                    break
                except Exception as exc:
                    if not _is_transient_image_transport_error(exc) or attempt == 2:
                        raise
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise RuntimeError(f"Seedream returned without {image_path.name}")
            with Image.open(image_path) as image:
                image.verify()
            record.update({"status": "done", "result_url": result_url})
        _write_json(sidecar_path, record)
        bridge["storyboard_transition"] = {
            "image": record["image"],
            "prompt": record["prompt"],
            "reference_images": record["reference_images"],
            "generation_phase": record["generation_phase"],
            "usage": record["usage"],
        }
        records.append(record)
    return records


def _is_output_image_safety_rejection(error: BaseException) -> bool:
    provider_code = str(getattr(error, "provider_code", "")).casefold()
    message = str(error).casefold()
    return any(
        marker.casefold() in provider_code or marker.casefold() in message
        for marker in (
            "OutputImageSensitiveContentDetected",
            "output image may contain sensitive information",
        )
    )


def _is_input_image_safety_rejection(error: BaseException) -> bool:
    provider_code = str(getattr(error, "provider_code", "")).casefold()
    message = str(error).casefold()
    return any(
        marker.casefold() in provider_code or marker.casefold() in message
        for marker in (
            "InputImageSensitiveContentDetected",
            "input image may contain sensitive information",
        )
    )


def _reference_reduction_candidates(
    reference_paths: list[Path],
    *,
    character_references: list[Path],
    previous_panel: Path | None,
    director_panel: Path | None,
    accepted_reference_hashes: set[str],
) -> list[list[Path]]:
    """Return at most three role-aware, progressively safer retry inputs."""
    original = tuple(reference_paths)
    seen = {original}
    nonempty: list[list[Path]] = []

    def add(paths: list[Path]) -> None:
        ordered: list[Path] = []
        for path in reference_paths:
            if path in paths and path not in ordered:
                ordered.append(path)
        candidate = tuple(ordered)
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        nonempty.append(list(candidate))

    if director_panel is not None:
        add([path for path in reference_paths if path != director_panel])

    known_accepted = [
        path
        for path in reference_paths
        if hashlib.sha256(path.read_bytes()).hexdigest() in accepted_reference_hashes
    ]
    add(known_accepted)

    body_identity = [
        path
        for path in character_references
        if path.stem.casefold() in {"full_body", "front", "side", "back"}
    ]
    add(body_identity)
    if previous_panel is not None:
        add([previous_panel])
    if director_panel is not None:
        add([director_panel])

    # Keep retries bounded: two reference-preserving attempts, then a final
    # text-only generation whose identity contract remains in the prompt.
    return [*nonempty[:2], []]


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
- 这是完全虚构的风格化 CGI 数字角色特技预演图，不是真人打斗或现实暴力。
- 每个角色必须保持自己声明的至少两个非真人妆造锚点（面纱/遮罩、图形化妆、面部纹样、机械纹理、非人材质等）；禁止未经妆造的自然真人脸、照片级人类皮肤或生物伤口，也禁止给全体套同款头盔。
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
    bridge_specs = [
        bridge
        for bridge in (storyboard.get("primary_shot_bridges") or [])
        if isinstance(bridge, dict)
    ]
    bridge_count = 0
    for bridge in bridge_specs:
        bridge_id = str(bridge.get("bridge_id") or "<missing-bridge-id>")
        transition = bridge.get("storyboard_transition")
        if not isinstance(transition, dict):
            errors.append(f"{bridge_id} has no post-primary storyboard transition")
            continue
        if transition.get("generation_phase") != "post_primary_storyboards":
            errors.append(
                f"{bridge_id} storyboard transition has invalid generation phase"
            )
        image_value = str(transition.get("image") or "").strip()
        if not image_value:
            errors.append(f"{bridge_id} storyboard transition has no image")
            continue
        image_path = _artifact_path(output_dir, image_value)
        try:
            if not image_path.is_file() or image_path.stat().st_size <= 1024:
                raise OSError("file missing or too small")
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            errors.append(
                f"{bridge_id} invalid storyboard transition {image_value}: {exc}"
            )
            continue
        bridge_count += 1
    if authored_count or bridge_specs:
        manifest = output_dir / "SHOT_STORYBOARDS.json"
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            if document.get("kind") != "honcut.shot_storyboards.v2":
                errors.append("SHOT_STORYBOARDS.json is not the nine-grid v2 contract")
            if document.get("status") != "done":
                errors.append("SHOT_STORYBOARDS.json is not complete")
            if int(document.get("total_panels") or 0) != authored_count:
                errors.append(
                    "SHOT_STORYBOARDS.json panel count does not match STORYBOARD.json"
                )
            if int(document.get("total_transition_panels") or 0) != len(
                bridge_specs
            ):
                errors.append(
                    "SHOT_STORYBOARDS.json transition count does not match STORYBOARD.json"
                )
            manifest_bridge_ids = {
                str(record.get("bridge_id") or "")
                for record in (document.get("bridges") or [])
                if isinstance(record, dict) and record.get("status") == "done"
            }
            expected_bridge_ids = {
                str(bridge.get("bridge_id") or "") for bridge in bridge_specs
            }
            if bridge_count != len(bridge_specs) or manifest_bridge_ids != expected_bridge_ids:
                errors.append(
                    "SHOT_STORYBOARDS.json has incomplete post-primary transitions"
                )
            manifest_shots = {
                str(record.get("shot_id") or ""): record
                for record in (document.get("shots") or [])
                if isinstance(record, dict) and record.get("shot_id")
            }
            for shot_index, shot in enumerate(storyboard.get("shots", []), 1):
                if not isinstance(shot, dict):
                    continue
                shot_id = _shot_id(shot, shot_index)
                record = manifest_shots.get(shot_id)
                if not record or record.get("status") != "done":
                    errors.append(f"{shot_id} has no completed nine-grid storyboard")
                    continue
                grid = record.get("grid_contract") or {}
                narrative = record.get("narrative_grid") or []
                if (
                    grid.get("columns") != SHOT_STORYBOARD_GRID_COLUMNS
                    or grid.get("rows") != SHOT_STORYBOARD_GRID_ROWS
                    or grid.get("cell_count") != SHOT_STORYBOARD_GRID_CELLS
                    or len(grid.get("cells") or []) != SHOT_STORYBOARD_GRID_CELLS
                    or len(narrative) != SHOT_STORYBOARD_GRID_CELLS
                ):
                    errors.append(f"{shot_id} has an invalid 3x3 narrative grid contract")
                if record.get("usage") != "director_llm_review_only_not_video_reference":
                    errors.append(f"{shot_id} nine-grid board has unsafe usage metadata")
                board_value = str(record.get("board") or "").strip()
                if not board_value:
                    errors.append(f"{shot_id} nine-grid board path is missing")
                    continue
                board_path = _artifact_path(output_dir, board_value)
                try:
                    if not board_path.is_file() or board_path.stat().st_size <= 1024:
                        raise OSError("file missing or too small")
                    with Image.open(board_path) as board_image:
                        board_image.verify()
                    observed_sha = hashlib.sha256(board_path.read_bytes()).hexdigest()
                    if observed_sha != str(record.get("board_sha256") or ""):
                        raise ValueError("board hash mismatch")
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"{shot_id} invalid nine-grid storyboard {board_value}: {exc}"
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
    target_shot_ids: set[str] | None = None,
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
    manifest_path = output_dir / "SHOT_STORYBOARDS.json"
    previous_manifest: dict[str, Any] = {}
    if target_shot_ids and manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "targeted storyboard regeneration requires a readable prior manifest"
            ) from exc
    normalized_targets = {
        str(shot_id).strip() for shot_id in (target_shot_ids or set())
        if str(shot_id).strip()
    }
    previous_records = {
        str(record.get("shot_id") or ""): record
        for record in (previous_manifest.get("shots") or [])
        if isinstance(record, dict) and record.get("shot_id")
    }
    preserved_records = [
        record for shot_id, record in previous_records.items()
        if shot_id not in normalized_targets
    ]
    contract: dict[str, Any] = {
        "kind": "honcut.shot_storyboards.v2",
        "version": 2,
        "status": "running",
        "provider": "seedream",
        "model": model,
        "shots": preserved_records,
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
    accepted_reference_hashes: set[str] = set()
    try:
        for index, shot in enumerate(storyboard.get("shots", []), 1):
            if not isinstance(shot, dict):
                continue
            shot_id = _shot_id(shot, index)
            if normalized_targets and shot_id not in normalized_targets:
                if shot_id not in previous_records:
                    raise RuntimeError(
                        f"targeted storyboard regeneration has no prior record for {shot_id}"
                    )
                authored_beats = [
                    beat for beat in (shot.get("storyboard_beats") or [])
                    if isinstance(beat, dict)
                ]
                if authored_beats:
                    last_beat_id = str(
                        authored_beats[-1].get("beat_id")
                        or f"{shot_id}_P{len(authored_beats):02d}"
                    )
                    candidate = beats_dir / f"{last_beat_id}.png"
                    if not candidate.is_file() or candidate.stat().st_size == 0:
                        raise RuntimeError(
                            f"targeted storyboard regeneration is missing prior panel {candidate}"
                        )
                    previous_storyboard_panel = candidate
                continue
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
            narrative_grid = _narrative_grid_contract(shot, shot_id, beats)
            director_panel = director_panels.get(shot_id)
            correction_issues = correction_context_by_shot.get(shot_id, [])
            record = {
                "shot_id": shot_id,
                "board": str(board_path.relative_to(output_dir)),
                "prompt": str(prompt_path.relative_to(output_dir)),
                "prompt_sha256": prompt_sha,
                "panel_count": len(beats),
                "narrative_grid": narrative_grid,
                "grid_columns": SHOT_STORYBOARD_GRID_COLUMNS,
                "grid_rows": SHOT_STORYBOARD_GRID_ROWS,
                "grid_cell_count": SHOT_STORYBOARD_GRID_CELLS,
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
            content_positions = [
                beat_position
                for beat_position, authored_beat in enumerate(beats, 1)
                if str(authored_beat.get("generation_mode") or "").strip().lower()
                != "first_last_frame_bridge"
            ]
            last_content_position = max(content_positions, default=0)
            for position, beat in enumerate(beats, 1):
                beat_id = str(beat.get("beat_id") or f"{shot_id}_P{position:02d}")
                beat_correction_issues = []
                for issue in correction_issues:
                    details = (
                        issue.get("details")
                        if isinstance(issue.get("details"), dict)
                        else {}
                    )
                    storyboard_ids = [
                        str(value) for value in (details.get("storyboard_ids") or [])
                    ]
                    if not storyboard_ids or beat_id in storyboard_ids:
                        beat_correction_issues.append(issue)
                correction_lines: list[str] = []
                for issue_index, issue in enumerate(beat_correction_issues, 1):
                    details = (
                        issue.get("details")
                        if isinstance(issue.get("details"), dict)
                        else {}
                    )
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
                        f"这是第 {int(correction_attempt)} 轮自动纠偏，只修复 "
                        f"{beat_id}。\n{correction_contract}"
                    )
                panel_prompt = _build_panel_prompt(
                    shot,
                    beat,
                    position,
                    len(beats),
                    characters,
                    uses_director_board=director_panel is not None,
                    aspect_ratio=aspect_ratio,
                    correction_contract=correction_contract,
                    is_last_content_beat=position == last_content_position,
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
                    f"{panel_prompt}\nreferences={','.join(reference_hashes)}".encode()
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
                    def reference_value(path: Path) -> str:
                        return (
                            str(path.relative_to(output_dir))
                            if path.is_relative_to(output_dir)
                            else str(path)
                        )

                    def reference_hash(path: Path) -> str:
                        return hashlib.sha256(path.read_bytes()).hexdigest()

                    def generate_panel(
                        generation_prompt: str,
                        selected_references: list[Path],
                        *,
                        _panel_record: dict[str, Any] = panel_record,
                        _panel_path: Path = panel_path,
                    ):
                        if selected_references and hasattr(client, "image_to_image"):
                            _panel_record["mode"] = "image_to_image"
                            return client.image_to_image(
                                prompt=generation_prompt,
                                ref_image=(
                                    str(selected_references[0])
                                    if len(selected_references) == 1
                                    else [str(path) for path in selected_references]
                                ),
                                output_path=str(_panel_path),
                                size=size,
                            )
                        _panel_record["mode"] = "text_to_image"
                        return client.text_to_image(
                            prompt=generation_prompt,
                            output_path=str(_panel_path),
                            size=size,
                            timeout=180,
                        )

                    transport_retries = 0

                    def generate_with_transport_retry(
                        generation_prompt: str,
                        selected_references: list[Path],
                        *,
                        _beat_id: str = beat_id,
                    ):
                        nonlocal transport_retries
                        max_transport_retries = 2
                        for attempt in range(max_transport_retries + 1):
                            try:
                                return generate_panel(
                                    generation_prompt,
                                    selected_references,
                                )
                            except Exception as exc:
                                if (
                                    not _is_transient_image_transport_error(exc)
                                    or attempt >= max_transport_retries
                                ):
                                    raise
                                transport_retries += 1
                                print(
                                    f"  [seedream] {_beat_id} 参考图传输中断；"
                                    f"同请求限次重试 {attempt + 1}/{max_transport_retries}",
                                    flush=True,
                                )

                    input_safety_trace: list[dict[str, Any]] = []
                    provider_accepted_references = list(reference_paths)

                    def generate_with_input_safety_fallback(
                        generation_prompt: str,
                        starting_references: list[Path],
                        *,
                        _character_references: tuple[Path, ...] = tuple(
                            character_references
                        ),
                        _previous_panel: Path | None = previous_panel,
                        _director_panel: Path | None = director_panel,
                        _trace: list[dict[str, Any]] = input_safety_trace,
                        _beat_id: str = beat_id,
                    ) -> tuple[str, list[Path]]:
                        nonlocal provider_accepted_references
                        fallback_references = _reference_reduction_candidates(
                            starting_references,
                            character_references=list(_character_references),
                            previous_panel=_previous_panel,
                            director_panel=_director_panel,
                            accepted_reference_hashes=accepted_reference_hashes,
                        )
                        candidates = [starting_references, *fallback_references]
                        last_input_error: BaseException | None = None
                        for attempt, selected_references in enumerate(candidates, 1):
                            try:
                                result = generate_with_transport_retry(
                                    generation_prompt,
                                    selected_references,
                                )
                            except Exception as exc:
                                if not _is_input_image_safety_rejection(exc):
                                    if _is_output_image_safety_rejection(exc):
                                        provider_accepted_references = list(
                                            selected_references
                                        )
                                        accepted_reference_hashes.update(
                                            reference_hash(path)
                                            for path in selected_references
                                        )
                                    raise
                                last_input_error = exc
                                _trace.append({
                                    "attempt": attempt,
                                    "status": "rejected",
                                    "mode": (
                                        "image_to_image"
                                        if selected_references
                                        else "text_to_image"
                                    ),
                                    "reference_images": [
                                        reference_value(path)
                                        for path in selected_references
                                    ],
                                    "provider_code": str(
                                        getattr(exc, "provider_code", "")
                                        or "InputImageSensitiveContentDetected"
                                    ),
                                    "request_id": (
                                        str(getattr(exc, "request_id", "") or "")
                                        or None
                                    ),
                                })
                                if attempt < len(candidates):
                                    print(
                                        f"  [seedream] {_beat_id} 输入参考图被服务商拒绝；"
                                        f"按身份与连续性优先级缩减参考图 "
                                        f"{attempt}/{len(candidates) - 1}",
                                        flush=True,
                                    )
                                continue
                            provider_accepted_references = list(selected_references)
                            accepted_reference_hashes.update(
                                reference_hash(path) for path in selected_references
                            )
                            if _trace:
                                _trace.append({
                                    "attempt": attempt,
                                    "status": "accepted",
                                    "mode": (
                                        "image_to_image"
                                        if selected_references
                                        else "text_to_image"
                                    ),
                                    "reference_images": [
                                        reference_value(path)
                                        for path in selected_references
                                    ],
                                })
                            return result, list(selected_references)
                        if last_input_error is not None:
                            raise last_input_error
                        raise RuntimeError(
                            f"{_beat_id} exhausted storyboard reference fallbacks"
                        )

                    try:
                        result_url, used_reference_paths = (
                            generate_with_input_safety_fallback(
                                panel_prompt,
                                list(reference_paths),
                            )
                        )
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
                        result_url, used_reference_paths = (
                            generate_with_input_safety_fallback(
                                safety_prompt,
                                provider_accepted_references,
                            )
                        )
                        panel_record["safety_retry"] = {
                            "reason": "output_image_sensitive_content",
                            "attempts": 1,
                            "policy": "synthetic_non_contact_stunt_v1",
                            "prompt": str(safety_prompt_path.relative_to(output_dir)),
                            "prompt_sha256": hashlib.sha256(
                                safety_prompt.encode("utf-8")
                            ).hexdigest(),
                        }
                    used_reference_hashes = [
                        reference_hash(path) for path in used_reference_paths
                    ]
                    used_reference_values = [
                        reference_value(path) for path in used_reference_paths
                    ]
                    if input_safety_trace:
                        panel_record["input_safety_fallback"] = {
                            "reason": "input_image_sensitive_content",
                            "attempts": sum(
                                item.get("status") == "rejected"
                                for item in input_safety_trace
                            ),
                            "policy": "role_preserving_reference_reduction_v1",
                            "trace": input_safety_trace,
                            "final_mode": panel_record["mode"],
                        }
                    panel_record["used_reference_images"] = used_reference_values
                    panel_record["dropped_reference_images"] = [
                        reference_value(path)
                        for path in reference_paths
                        if path not in used_reference_paths
                    ]
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
                        "requested_reference_image_sha256": reference_hashes,
                        "reference_image_sha256": used_reference_hashes,
                    })
                accepted_reference_hashes.update(
                    str(value)
                    for value in (panel_record.get("reference_image_sha256") or [])
                    if str(value)
                )
                _write_json(panel_sidecar, panel_record)
                beat["storyboard_image"] = panel_record["image"]
                panel_records.append(panel_record)
                panel_paths.append(panel_path)
                previous_panel = panel_path
            # The Sxx board is a director/LLM-only nine-grid narrative artifact.
            # It is generated independently from the per-Pxx images and is never
            # eligible as video-model media. Exact grid boundaries and cell IDs
            # are then overlaid deterministically so layout cannot silently
            # collapse into a linear strip.
            board_result_url = client.text_to_image(
                prompt=prompt,
                output_path=str(board_path),
                size=size,
                timeout=180,
            )
            if not board_path.is_file() or board_path.stat().st_size == 0:
                raise RuntimeError(f"Seedream returned without {board_path.name}")
            with Image.open(board_path) as board_image:
                board_image.verify()
            raw_board_sha256 = hashlib.sha256(board_path.read_bytes()).hexdigest()
            grid_contract = _normalize_nine_grid_board(board_path, shot_id)
            legacy_preview_path = image_dir / f"{shot_id}.png"
            shutil.copy2(panel_paths[0], legacy_preview_path)
            _write_json(
                image_dir / f"{shot_id}.json",
                {
                    "kind": "honcut.previs-placeholder.v1",
                    "status": "previs_only",
                    "usage": "phase2_review_placeholder_never_video_reference",
                    "image": str(legacy_preview_path.relative_to(output_dir)),
                    "image_sha256": hashlib.sha256(
                        legacy_preview_path.read_bytes()
                    ).hexdigest(),
                    "source": str(panel_paths[0].relative_to(output_dir)),
                    "replaced_by_phase": "phase4_cinematic_first_frames",
                },
            )
            record.update({
                "status": "done",
                "model": contract["model"],
                "panels": panel_records,
                "usage": "director_llm_review_only_not_video_reference",
                "result_url": board_result_url,
                "raw_board_sha256": raw_board_sha256,
                "board_sha256": hashlib.sha256(board_path.read_bytes()).hexdigest(),
                "grid_contract": grid_contract,
            })
            shot["storyboard_board"] = record["board"]
            shot["storyboard_beats"] = beats
            _write_json(boards_dir / f"{shot_id}.json", record)
            contract["shots"].append(record)
            contract["shots"].sort(
                key=lambda item: authored_shot_ids.index(str(item.get("shot_id")))
                if str(item.get("shot_id")) in authored_shot_ids
                else len(authored_shot_ids)
            )
            if panel_paths:
                previous_storyboard_panel = panel_paths[-1]
            _write_json(manifest_path, contract)
        bridge_records = _generate_primary_bridge_storyboards(
            output_dir,
            storyboard,
            client=client,
            model=str(contract["model"]),
            size=size,
            aspect_ratio=aspect_ratio,
        )
        contract["bridges"] = bridge_records
        contract["status"] = "done"
        contract["total_boards"] = len(contract["shots"])
        contract["total_panels"] = sum(
            int(item.get("panel_count") or 0) for item in contract["shots"]
        )
        contract["total_transition_panels"] = len(bridge_records)
        _write_json(manifest_path, contract)
        return contract
    except Exception as exc:
        contract["status"] = "error"
        contract["error"] = str(exc)
        _write_json(manifest_path, contract)
        raise
