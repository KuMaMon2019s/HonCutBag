from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import pipeline_core
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

    def fake_batch(characters, output_dir, skip_images=False):
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


def test_storyboard_prompt_builder_receives_sliced_style(tmp_path, monkeypatch):
    (tmp_path / "visual-style.md").write_text(STYLE, encoding="utf-8")
    captured = {}

    def fake_build_batch_prompts(scenes, style_context=None):
        captured.update(style_context or {})
        return []

    monkeypatch.setattr(pipeline_core, "build_batch_prompts", fake_build_batch_prompts)

    assert pipeline_core._generate_shot_images(tmp_path, {"shots": []}) == 0
    assert "Camera composition" in captured["mood"]
    assert "blue palette" in captured["mood"]
    assert "red outfit" not in captured["mood"]
    assert len(captured["mood"]) < len(STYLE)


def test_video_prompt_builder_receives_sliced_style():
    prompt = build_video_prompt(
        {"id": 1, "duration": 5},
        [],
        {"global_style_lock": STYLE},
        "seedance",
    )

    assert "believable physics" in prompt
    assert "blue palette" in prompt
    assert "red outfit" not in prompt
    assert STYLE not in prompt
