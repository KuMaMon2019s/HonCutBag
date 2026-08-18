from __future__ import annotations

import sys
from pathlib import Path
import json


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import pipeline_core
from phases.phase3.character_factory import build_model_reference_prompts
from phases.phase6.video_generator import build_video_prompt
from utils.style_slices import get_slice, split_visual_style


STYLE = """# Character
Character appearance uses a tailored red outfit and a lean body silhouette.

# Environment
Rainy environment lighting uses a restrained blue palette and somber mood.

# Storyboard
Camera composition favors wide framing and a low angle.

# Motion
Motion and action follow believable physics with a deliberate tempo.
"""


def test_split_visual_style_routes_task_sections():
    slices = split_visual_style(STYLE)

    assert "red outfit" in slices["character"]
    assert "Rainy environment" in slices["scene"]
    assert "Camera composition" in slices["storyboard"]
    assert "believable physics" in slices["video"]
    assert "blue palette" in slices["global"]
    assert "red outfit" not in slices["video"]


def test_get_slice_always_appends_global_guidance():
    result = get_slice(STYLE, "video")

    assert "believable physics" in result
    assert "blue palette" in result
    assert len(result) < len(STYLE)


def test_get_slice_empty_unknown_or_unmatched_type_falls_back_to_full_document():
    unmatched = "A medium with tactile handcrafted detail."

    assert get_slice(STYLE, "") == STYLE.strip()
    assert get_slice(STYLE, "prop") == STYLE.strip()
    assert get_slice(unmatched, "character") == unmatched


def test_phase3_character_builder_receives_sliced_style(tmp_path, monkeypatch):
    (tmp_path / "visual-style.md").write_text(STYLE, encoding="utf-8")
    captured = []

    def fake_batch(characters, output_dir, skip_images=False, **_kwargs):
        captured.append(characters[0]["style"])
        return ["characters/lead/"]

    monkeypatch.setattr("phases.phase3.character_factory.batch_generate", fake_batch)
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda *args, **kwargs: type("Report", (), {"passed": True})(),
    )

    result = pipeline_core.run_phase3(
        tmp_path,
        {"characters": [{"id": "lead", "name": "Lead", "appearance": {}}]},
        dry_run=True,
    )

    assert result["status"] == "done"
    assert "red outfit" in captured[0]
    assert "blue palette" in captured[0]
    assert "believable physics" not in captured[0]
    assert len(captured[0]) < len(STYLE)


def test_phase3_persists_and_generates_from_no_real_person_contract(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setenv("HONCUT_NO_REAL_PERSON", "1")

    def fake_batch(characters, output_dir, skip_images=False, **_kwargs):
        captured.extend(characters)
        return ["characters/agent/"]

    monkeypatch.setattr("phases.phase3.character_factory.batch_generate", fake_batch)
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda *args, **kwargs: type("Report", (), {"passed": True})(),
    )
    source = {
        "characters": [
            {
                "id": "agent",
                "name": "Agent",
                "role": "protagonist",
                "style": "真人写实",
                "appearance": {
                    "hair": "黑色短发",
                    "face": "高鼻梁",
                    "clothing": "深灰色战术服",
                    "summary": "男性真人特工",
                },
            }
        ]
    }

    result = pipeline_core.run_phase3(tmp_path, source, dry_run=True)

    persisted = json.loads((tmp_path / "CHARACTERS.json").read_text(encoding="utf-8"))
    assert result["status"] == "done"
    assert persisted["visual_identity_policy"] == "synthetic_faceless_android_v1"
    assert "全封闭机械头盔" in captured[0]["description"]
    assert "风格化三维 CGI" in captured[0]["style"]
    assert "photorealistic human" in captured[0]["negative"]


def test_cgi_character_style_never_falls_back_to_photoreal_skin():
    prompts = build_model_reference_prompts(
        "全封闭机械头盔的虚构合成人",
        "高成本风格化三维 CGI 科幻动画",
    )

    assert all("no visible human face" in prompt for prompt in prompts.values())
    assert all(
        "Photorealistic, natural skin texture" not in prompt
        for prompt in prompts.values()
    )


def test_storyboard_prompt_builder_does_not_receive_plot_bearing_global_style(tmp_path, monkeypatch):
    (tmp_path / "visual-style.md").write_text(STYLE, encoding="utf-8")
    captured = {}

    def fake_build_batch_prompts(scenes, style_context=None):
        captured.update(style_context or {})
        return []

    monkeypatch.setattr(pipeline_core, "build_batch_prompts", fake_build_batch_prompts)

    assert pipeline_core._generate_shot_images(tmp_path, {"shots": []}) == 0
    assert captured == {}


def test_video_prompt_builder_uses_plot_neutral_rendering_continuity():
    prompt = build_video_prompt(
        {"id": 1, "duration": 5},
        [],
        {"global_style_lock": STYLE},
        "seedance",
    )

    assert "believable physics" not in prompt
    assert "blue palette" not in prompt
    assert "red outfit" not in prompt
    assert "不得从项目级风格描述引入" in prompt
    assert STYLE not in prompt
