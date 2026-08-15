import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients import tos_uploader
from phases.phase1.character_discoverer import _add_reference_contract
from phases.pipeline_core import (
    _extract_visual_style_text,
    _storyboard_keyframe_description,
    _write_project_visual_style,
)
from phases.phase1.storyboard_generator import (
    _build_shot_prompt,
    _load_default_visual_style,
    _render_system_prompt,
    _specific_lighting,
    generate_storyboard,
)
from phases.phase5.storyboard_qa_gate import run_generation_capacity_checks
from phases.phase6.video_generator import build_video_prompt
from phases.phase4.scene_consistency import build_scene_reference_prompt
from prompt.prompt_router import route_prompt
from tools.asset_packager import build_content_for_shot, inject_reference_instruction
from utils.visual_style_spec import VisualStyle


STYLE_DECLARED_SCRIPT = """\
《雾港来信》

美术风格：二维赛璐璐动画，冷青色夜景，手绘线条与克制的霓虹反光。

雨落在空旷站台上。信使收起伞，望向驶离的列车。
"""

PLOT_ONLY_SCRIPT = """\
《雾港来信》

雨落在空旷站台上。信使收起伞，望向驶离的列车。
"""


def test_f1_extracts_and_parses_v10_project_style(tmp_path):
    style_text = _extract_visual_style_text(STYLE_DECLARED_SCRIPT)
    assert style_text and "赛璐璐" in style_text

    style_path = _write_project_visual_style(tmp_path, style_text)
    parsed = _load_default_visual_style(str(style_path))
    assert "赛璐璐" in style_path.read_text(encoding="utf-8")
    assert parsed.style_prompt_short


def test_f1_plot_only_script_falls_back_without_crash():
    """纯剧情剧本（无美术风格段）正则提取应返回空值而非崩溃，交给 LLM 兜底。"""
    style_text = _extract_visual_style_text(PLOT_ONLY_SCRIPT)
    assert style_text is None or style_text == ""


def test_f2_system_prompt_uses_project_style_or_legacy_default():
    project = _render_system_prompt("赛璐璐二次元，冷色调雨夜，霓虹反射")
    assert "严格遵循项目美术风格" in project
    assert "禁止替换为真人写实摄影风格" in project
    assert "delicate skin texture" not in project

    legacy = _render_system_prompt()
    assert "Photorealistic cinematography" in legacy
    assert "delicate skin texture" in legacy


def _write_character_assets(root: Path, char_id: str, byte: bytes):
    char_dir = root / "characters" / char_id
    char_dir.mkdir(parents=True)
    (char_dir / "face_closeup.png").write_bytes(byte * 2048)
    (char_dir / "full_body.png").write_bytes(byte.upper() * 2048)


def test_f3_two_characters_bind_distinct_actual_reference_numbers(tmp_path, monkeypatch):
    characters = [
        {"id": "rin", "name": "凛", "appearance": {"clothing": "黑色短夹克", "face": "冷静眼神", "hair": "银灰短发"}},
        {"id": "jin", "name": "烬", "appearance": {"clothing": "深色战斗服", "face": "沉稳面容", "hair": "黑发"}},
    ]
    for character in characters:
        _add_reference_contract(character)
        assert character["distinguishing_features"]
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": characters}, ensure_ascii=False), encoding="utf-8"
    )
    _write_character_assets(tmp_path, "rin", b"r")
    _write_character_assets(tmp_path, "jin", b"j")
    monkeypatch.setattr(
        tos_uploader,
        "upload_image",
        lambda image_bytes, content_type: f"https://mock.invalid/{image_bytes[:1].decode()}.png",
    )

    content = build_content_for_shot(
        tmp_path,
        "S02",
        {
            "prompt": "参考{图片N}中的凛作为主体；参考{图片N}中的烬作为主体。",
            "gen_strategy": "phantom",
            "_char_ids": ["rin", "jin"],
        },
    )
    prompt = next(item["text"] for item in content if item["type"] == "text")
    refs = [item for item in content if item.get("role") == "reference_image"]
    assert len(refs) == 4
    assert "参考图片1中的凛" in prompt
    assert "参考图片3中的烬" in prompt
    assert "定义为<主体1>" in prompt and "定义为<主体2>" in prompt
    assert "{图片N}" not in prompt and "{主体N}" not in prompt


def test_f4_source_excerpt_survives_blueprint_and_storyboard(monkeypatch):
    key_action = "凛猛地后仰，刀锋从鼻尖掠过削断几缕银发，右手随即反转刀柄横扫烬的腰侧"
    shot = {
        "id": 1,
        "who": ["凛"],
        "where": "雨夜轨道",
        "source_excerpt": key_action,
        "what": "完成首轮攻防互换",
        "suggested_duration": 5,
    }
    prompt = _build_shot_prompt(shot, [])
    assert "刀锋从鼻尖掠过削断几缕银发" in prompt

    monkeypatch.setattr(
        "phases.phase1.storyboard_generator._call_llm",
        lambda user_prompt, visual_style_text=None: '{"prompt":"mock", "caption":"交锋"}',
    )
    storyboard = generate_storyboard([shot], [], visual_style_text="赛璐璐")
    assert key_action in storyboard["shots"][0]["action_description"]
    assert "刀锋从鼻尖掠过削断几缕银发" in storyboard["shots"][0]["prompt"]


def test_action_prompt_preserves_explicit_static_camera_and_bounded_actions():
    shot = {
        "id": 2,
        "who": ["凛", "烬"],
        "where": "雨夜高架",
        "suggested_duration": 4,
        "shot_intent": "action",
        "camera_movement": "static",
        "generation_actions": ["凛冲刺", "跃起劈刀", "烬举臂格挡", "火星炸开"],
        "source_excerpt": "不应进入模型提示的二十六步完整动作流水账",
        "what": "首轮交锋",
    }

    prompt = _build_shot_prompt(shot, [])

    assert "4.0秒" in prompt
    assert "固定(fixed/locked)" in prompt
    assert shot["camera_movement"] == "static"
    assert "凛冲刺 → 跃起劈刀 → 烬举臂格挡 → 火星炸开" in prompt
    assert "不应进入模型提示的二十六步完整动作流水账" not in prompt
    assert "补充：首轮交锋" not in prompt
    assert "不得在首帧姿态原地停留" in prompt
    assert "不得无动机切成特写" in prompt


def test_action_prompt_does_not_inject_unrelated_emotion_gesture():
    prompt = _build_shot_prompt({
        "id": 1,
        "who": ["凛", "烬"],
        "where": "废弃高架",
        "what": "凛攻击烬",
        "emotion": "激烈、凌厉、高度紧张",
        "shot_intent": "action",
        "camera_movement": "steadicam",
        "suggested_duration": 4,
        "generation_actions": [
            "凛踩碎积水冲出",
            "凛腾空举刀下劈",
            "烬以机械臂格挡",
        ],
    })

    assert "凛踩碎积水冲出" in prompt
    assert "频繁看手表" not in prompt
    assert "手指敲击桌面" not in prompt


def test_action_storyboard_keyframe_uses_start_state_not_final_impact():
    prompt = _storyboard_keyframe_description({
        "who": ["凛", "烬"],
        "where": "废弃高架",
        "shot_intent": "action",
        "start_state": "凛与烬相隔数米对峙，双方武器尚未接触",
        "generation_actions": ["凛踩水冲出", "刀锋撞上机械臂"],
        "action_description": "凛踩水冲出 → 刀锋撞上机械臂并炸出火星",
        "visual": "凛冲刺后与烬碰撞，火星炸开",
    })

    assert "凛与烬相隔数米对峙" in prompt
    assert "frame zero" in prompt
    assert "impact, and result have not happened" in prompt
    assert "Visual staging: 凛冲刺后与烬碰撞" not in prompt


def test_phase5_blocks_action_overload_static_hold_and_missing_coverage():
    storyboard = {
        "shots": [{
            "id": 1,
            "duration": 10,
            "source_action_unit_ids": ["AU001", "AU002"],
            "generation_actions": ["a", "b", "c", "d", "e"],
            "camera_movement": "static",
        }]
    }
    events = {"events": [
        {"action_unit_id": "AU001"},
        {"action_unit_id": "AU002"},
        {"action_unit_id": "AU003"},
    ]}

    issues = run_generation_capacity_checks(storyboard, events)
    codes = {issue["code"] for issue in issues}

    assert {
        "action_unit_overload",
        "generation_action_overload",
        "action_shot_too_long",
        "static_action_camera",
        "action_unit_coverage_missing",
    } <= codes


def test_phase5_uses_stricter_action_budget_for_four_second_clip():
    issues = run_generation_capacity_checks({
        "shots": [{
            "id": 1,
            "duration": 4,
            "source_action_unit_ids": ["AU001"],
            "generation_actions": ["a", "b", "c", "d"],
            "camera_movement": "steadicam",
        }]
    })

    overload = next(issue for issue in issues if issue["code"] == "generation_action_overload")
    assert overload["details"]["action_limit"] == 1


def test_action_phantom_references_are_motion_bounded(tmp_path, monkeypatch):
    characters = [
        {"id": "rin", "name": "凛", "aliases": []},
        {"id": "jin", "name": "烬", "aliases": []},
    ]
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": characters}, ensure_ascii=False), encoding="utf-8"
    )
    for character in characters:
        _write_character_assets(tmp_path, character["id"], character["id"][0].encode())
    storyboard_dir = tmp_path / "storyboard_images"
    storyboard_dir.mkdir()
    (storyboard_dir / "S01.png").write_bytes(b"s" * 2048)
    monkeypatch.setattr(
        tos_uploader,
        "upload_image",
        lambda image_bytes, content_type: f"https://mock.invalid/{image_bytes[:1].decode()}.png",
    )

    content = build_content_for_shot(tmp_path, "S01", {
        "prompt": "两人交锋",
        "gen_strategy": "phantom",
        "who": ["凛", "烬"],
        "generation_actions": ["冲刺", "格挡", "分开"],
    })

    images = [item for item in content if item["type"] == "image_url"]
    prompt = next(item["text"] for item in content if item["type"] == "text")
    assert len(images) == 3
    urls = [item["image_url"]["url"] for item in images]
    assert "https://mock.invalid/R.png" in urls
    assert "https://mock.invalid/J.png" in urls
    assert "motion-priority" in prompt


def test_f5_placeholder_rendering_ignores_first_frame_numbering():
    prompt = "参考{图片N}中的凛作为主体；参考{图片N}中的烬作为主体。"
    rendered = inject_reference_instruction(
        prompt,
        [
            {"char_id": "rin", "reference_description": "凛的面部特写"},
            {"char_id": "rin", "reference_description": "凛的全身照"},
            {"char_id": "jin", "reference_description": "烬的面部特写"},
            {"char_id": "jin", "reference_description": "烬的全身照"},
        ],
    )
    assert "参考图片1中的凛" in rendered
    assert "参考图片3中的烬" in rendered
    assert "{图片N}" not in rendered


def test_specific_lighting_inherits_film_wide_rainy_night_style():
    style = VisualStyle(
        name="rainy-night",
        style_prompt_full="暗黑冷峻硬核赛博朋克，暴雨夜高空废弃磁悬浮轨道，湿冷压抑",
    )

    lighting = _specific_lighting({}, "废弃高空磁悬浮轨道", style)

    assert "冷蓝雨夜光" in lighting
    assert "黄金时段" not in lighting


def test_specific_lighting_preserves_valid_shot_lighting():
    original = "冷色主光从镜头右上方侧光照入，色温5600K，雨雾氛围压抑"
    style = VisualStyle(name="unrelated", style_prompt_full="golden hour")

    assert _specific_lighting({"lighting_description": original}, "轨道", style) == original


def test_specific_lighting_uses_neutral_fallback_without_time_of_day():
    style = VisualStyle(name="neutral", style_prompt_full="粗粝金属质感，低饱和度")

    lighting = _specific_lighting({}, "废弃轨道", style)

    assert "与全片美术风格一致的自然光照" in lighting
    assert "黄金时段" not in lighting


def test_specific_lighting_does_not_turn_generic_outdoors_into_moonlight():
    style = VisualStyle(name="day-neutral", style_prompt_full="自然纪实，真实色彩")

    lighting = _specific_lighting({}, "学校室外操场", style)

    assert "月光" not in lighting
    assert "自然光照" in lighting


def test_video_prompt_preserves_scene_lighting_description():
    lighting = "环境光从开口处自然进入，保留柔和阴影"
    prompt = build_video_prompt(
        {"id": 1, "where": "室内", "visual": "场景主体"},
        {"characters": []},
        {
            "global_lighting": "跨镜头光照保持连续",
            "shots": {"S01": {"lighting_description": lighting}},
        },
        "seedance",
    )

    assert lighting in prompt


def test_video_prompt_uses_global_lighting_without_fixed_color_temperature():
    global_lighting = "跨镜头光照方向与明暗关系保持连续"
    prompt = build_video_prompt(
        {"id": 1, "where": "室内", "visual": "场景主体"},
        {"characters": []},
        {"global_lighting": global_lighting, "shots": {"S01": {}}},
        "seedance",
    )

    assert global_lighting in prompt
    assert "色温4800K" not in prompt


def test_video_prompt_uses_neutral_lighting_for_empty_scene_consistency():
    prompt = build_video_prompt(
        {"id": 1, "where": "室内", "visual": "场景主体"},
        {"characters": []},
        {},
        "seedance",
    )

    assert "与全片美术风格一致的自然光照，明暗关系真实克制" in prompt
    assert "色温4800K" not in prompt


def test_seedance_single_route_preserves_complete_rainy_night_prompt():
    assembled = "镜头6。场景与光影：废弃高架桥，冷蓝雨夜。全局收尾：冷暗雨夜赛博霓虹。"

    prompt = route_prompt(
        "doubao-seedance-2.0-mini",
        "single_shot",
        {
            "prompt": assembled,
            "visual": "两人对话",
            "where": "废弃高架桥",
            "lighting_description": "冷蓝雨夜光，整镜头保持深夜",
            "time_of_day": "夜间，雨天",
        },
    )

    assert assembled in prompt
    assert "Time and weather: 夜间，雨天" in prompt
    assert "Lighting continuity: 冷蓝雨夜光" in prompt


def test_video_prompt_adds_hard_night_lock_and_daylight_negative():
    prompt = build_video_prompt(
        {"id": 6, "where": "废弃高架桥", "visual": "两人在雨中对话"},
        {"characters": []},
        {
            "global_style_lock": "冷暗雨夜赛博霓虹",
            "shots": {"S06": {"lighting_description": "冷蓝雨夜光"}},
        },
        "seedance",
    )

    assert "从第一帧到最后一帧始终保持深夜" in prompt
    assert "白天(daytime)" in prompt
    assert "灰白日间天空(overcast daylight)" in prompt


def test_scene_reference_prompt_inherits_night_weather_and_excludes_daylight():
    prompt = build_scene_reference_prompt(
        "废弃高架桥",
        [{"where": "废弃高架桥", "time": "夜间，雨天", "visual": "霓虹在湿地反光"}],
        VisualStyle(name="rainy-night", style_prompt_full="冷暗雨夜赛博霓虹"),
    )

    assert "深夜雨天" in prompt
    assert "不得出现白天、日光" in prompt
    assert "冷蓝雨夜光" in prompt
