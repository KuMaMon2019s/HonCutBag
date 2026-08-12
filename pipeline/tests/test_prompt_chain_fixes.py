import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients import tos_uploader
from phases.phase1.character_discoverer import _add_reference_contract
from phases.pipeline_core import _extract_visual_style_text, _write_project_visual_style
from phases.phase1.storyboard_generator import (
    _build_shot_prompt,
    _load_default_visual_style,
    _render_system_prompt,
    _specific_lighting,
    generate_storyboard,
)
from phases.phase6.video_generator import build_video_prompt
from tools.asset_packager import build_content_for_shot, inject_reference_instruction
from utils.visual_style_spec import VisualStyle


V10_SCRIPT = Path("/Users/soda/knowledge-base/2026-08-09_01/input/source_text.with_style_header.bak.txt")
# 2026-08-09: 部长下令换成纯剧情版（无美术风格段）后，F1 正则提取测试
# 固定指向带风格段的备份版；纯剧情版在下方 fallback 测试覆盖。
V10_SCRIPT_PLOT_ONLY = Path("/Users/soda/knowledge-base/2026-08-09_01/input/source_text.txt")


def test_f1_extracts_and_parses_v10_project_style(tmp_path):
    script_text = V10_SCRIPT.read_text(encoding="utf-8")
    style_text = _extract_visual_style_text(script_text)
    assert style_text and "赛璐璐" in style_text

    style_path = _write_project_visual_style(tmp_path, style_text)
    parsed = _load_default_visual_style(str(style_path))
    assert "赛璐璐" in style_path.read_text(encoding="utf-8")
    assert parsed.style_prompt_short


def test_f1_plot_only_script_falls_back_without_crash():
    """纯剧情剧本（无美术风格段）正则提取应返回空值而非崩溃，交给 LLM 兜底。"""
    script_text = V10_SCRIPT_PLOT_ONLY.read_text(encoding="utf-8")
    style_text = _extract_visual_style_text(script_text)
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
