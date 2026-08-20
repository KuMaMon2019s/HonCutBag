"""Regression coverage for time-window and identity-detail visual anchors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase1 import adaptation_engine, storyboard_generator
from phases.phase2.shot_storyboards import build_shot_storyboard_prompt
from phases.phase3 import character_factory
from phases.phase4 import scene_consistency
from phases.phase6.video_generator import build_video_prompt
from quality import video_qa
from quality.character_reference_qa import parse_identity_detail_qa
from quality.quality_gate import run_quality_check
from tools.asset_packager import collect_character_reference_assets
from utils.character_reference_contracts import (
    character_identity_detail_items,
    normalize_character_reference_assets,
)
from utils.temporal_visual_contracts import (
    apply_temporal_visual_contract,
    build_temporal_visual_contract,
    temporal_visual_negative_prompt,
    temporal_visual_prompt,
)
from utils.video_generation_contracts import render_video_generation_contract
from utils.visual_style_spec import VisualStyle


def test_generic_day_becomes_clock_window_with_observable_light_guards():
    contract = build_temporal_visual_contract(
        source_time="春季日间",
        visual_context="暴雨，冷色霓虹",
        lighting_context="neon night mood",
    )

    assert contract is not None
    assert contract["period"] == "day"
    assert contract["source_kind"] == "authored_time"
    assert contract["local_clock_window"] == {
        "start": "10:00",
        "end": "16:00",
        "basis": "local_scene_time_visual_lock",
    }
    assert "日光主导" in temporal_visual_prompt(contract)
    assert "霓虹独占主照明" in temporal_visual_negative_prompt(contract)


def test_source_event_time_is_inherited_into_every_adapted_shot():
    shots = [{"source_events": [1], "who": [], "where": "雨中的城市广场"}]
    events = [{"who": [], "time": "日间", "micro_actions": []}]

    adaptation_engine._inherit_event_semantics(shots, events)

    assert shots[0]["time"] == "日间"
    assert shots[0]["time_of_day"] == "day"
    assert shots[0]["time_window"] == "10:00-16:00"
    assert shots[0]["temporal_visual_contract"]["period"] == "day"


def test_day_contract_overrides_night_style_in_storyboard_scene_and_video(monkeypatch):
    style = VisualStyle(
        name="rainy-neon",
        style_prompt_short="cold neon rain",
        style_prompt_full="dramatic rainy night with moonlight",
    )
    shot = {
        "id": 1,
        "duration": 8,
        "time": "日间",
        "where": "雨中的高架桥",
        "visual": "车辆穿过冷色霓虹反光",
        "lighting_key": "neon",
        "who": [],
        "storyboard_beats": [
            {
                "beat_id": "S01_P01",
                "duration_s": 8,
                "start_state": "车辆在远处",
                "action": "车辆驶近",
                "end_state": "车辆抵达桥下",
            }
        ],
    }

    lighting = storyboard_generator._specific_lighting(shot, shot["where"], style)
    assert "日间" in lighting
    assert "绝非夜景" in lighting

    monkeypatch.setattr(scene_consistency, "_load_style", lambda _path: style)
    scene_contract = scene_consistency.generate_scene_consistency({"shots": [shot]})
    scene = scene_contract["shots"]["S01"]
    assert scene["temporal_visual_contract"]["period"] == "day"
    assert "日间" in scene["lighting_description"]
    assert "雨夜" not in scene["lighting_description"]

    video_prompt = build_video_prompt(shot, {"characters": []}, scene_contract, "seedance")
    assert isinstance(video_prompt, str)
    assert "10:00–16:00" in video_prompt
    assert "霓虹独占主照明" in video_prompt
    assert "不得因雨、冷色、霓虹" in video_prompt

    storyboard_prompt, _beats = build_shot_storyboard_prompt(shot, "S01", [])
    assert "时间段视觉硬合同" in storyboard_prompt
    assert "10:00–16:00" in storyboard_prompt


def test_final_qa_receives_exact_temporal_contract(tmp_path):
    frame_path = tmp_path / "S01_mid.jpg"
    frame_path.write_bytes(b"frame")
    captured: dict[str, str] = {}

    class Reviewer:
        def review(self, _paths, prompt):
            captured["prompt"] = prompt
            return json.dumps({"verdict": "pass", "issues": [], "confidence": 0.99})

    storyboard = {
        "shots": [
            {"id": "S01", "time": "日间", "where": "雨中的高架桥", "who": []}
        ]
    }
    frames = [
        video_qa.FrameSample(path=str(frame_path), timestamp=2.0, label="S01_mid")
    ]

    result = video_qa._vlm_semantic_check(Reviewer(), frames, storyboard)

    assert result["verdict"] == "pass"
    assert "temporal_visual_contract" in captured["prompt"]
    assert "10:00" in captured["prompt"]
    assert "rain, neon, cold color grading" in captured["prompt"]


def test_identity_props_are_visual_details_not_neutral_handheld_props():
    character = {
        "appearance": {
            "clothing": "藏蓝夹克 + 手持相机",
            "identity_props": [
                {
                    "id": "camera_a",
                    "name": "摄影机",
                    "description": "黑色方形机身、银色定焦镜头、红色腕带",
                    "attachment_mode": "isolated_handheld",
                    "persistence": "role_active",
                    "reference_required": True,
                }
            ],
        }
    }

    normalize_character_reference_assets(character)

    assert character["appearance"]["clothing"] == "藏蓝夹克"
    assert character["appearance"]["interaction_props"] == ["手持相机"]
    items = character_identity_detail_items(character)
    assert items[0]["id"] == "camera_a"
    assert items[0]["attachment_mode"] == "isolated_handheld"
    assert character["appearance"]["reference_asset_contract"][
        "identity_detail_assets"
    ] == "derived_board_body_attached_or_isolated"


def test_identity_detail_qa_recomputes_declared_item_failure():
    result = parse_identity_detail_qa(
        json.dumps({
            "passed": True,
            "character_identity_consistent": True,
            "declared_items_present": False,
            "item_geometry_consistent": True,
            "colors_materials_consistent": True,
            "attachment_modes_correct": True,
            "undeclared_items_absent": True,
            "issues": ["red wrist strap is missing"],
        })
    )

    assert result["passed"] is False
    assert result["declared_items_present"] is False


def test_phase3_derives_detail_and_variant_from_approved_four_views(
    monkeypatch, tmp_path
):
    calls: list[dict] = []

    class ImageClient:
        def __init__(self, model):
            self.model = model

        @staticmethod
        def _write(path):
            Image.effect_noise((512, 512), 96).convert("RGB").save(path)

        def text_to_image(self, **kwargs):
            calls.append({"kind": "text", **kwargs})
            self._write(kwargs["output_path"])

        def image_to_image(self, **kwargs):
            calls.append({"kind": "image", **kwargs})
            self._write(kwargs["output_path"])

    class Reviewer:
        def review(self, paths, _prompt):
            if len(paths) == 3:
                return json.dumps({
                    "passed": True,
                    "character_identity_consistent": True,
                    "declared_items_present": True,
                    "item_geometry_consistent": True,
                    "colors_materials_consistent": True,
                    "attachment_modes_correct": True,
                    "undeclared_items_absent": True,
                    "issues": [],
                })
            common = {
                "passed": True,
                "view_match": True,
                "framing_match": True,
                "neutral_pose": True,
                "hands_empty": True,
                "plain_background": True,
                "single_character": True,
                "face_visible": True,
                "both_eyes_visible": True,
                "issues": [],
            }
            return json.dumps({
                "views": {
                    "face_closeup": common,
                    "full_body": common,
                    "side": {**common, "both_eyes_visible": False},
                    "back": {
                        **common,
                        "face_visible": False,
                        "both_eyes_visible": False,
                    },
                },
                "cross_view": {
                    "passed": True,
                    "identity_consistent": True,
                    "outfit_consistent": True,
                    "body_proportions_consistent": True,
                    "issues": [],
                },
                "failed_views": [],
                "summary": "pass",
            })

    identity_props = [
        {
            "id": "camera_a",
            "name": "摄影机",
            "description": "黑色方形机身、银色定焦镜头、红色腕带",
            "attachment_mode": "isolated_handheld",
            "persistence": "role_active",
            "reference_required": True,
        }
    ]
    monkeypatch.setattr(character_factory, "SeedreamClient", ImageClient)
    result = character_factory.generate_character(
        char_id="photographer",
        name="摄影师",
        description="藏蓝头盔、米色摄影背心、深色长裤",
        output_dir=str(tmp_path),
        identity_props=identity_props,
        variants=[{"state_name": "rain", "description": "背心和头盔被雨水打湿"}],
        review_client=Reviewer(),
    )

    detail_path = tmp_path / "characters/photographer/identity_detail.png"
    variant_path = tmp_path / "characters/photographer/variant_rain.png"
    assert detail_path.is_file()
    assert variant_path.is_file()
    detail_call = next(
        call for call in calls if Path(call["output_path"]).name == ".identity_detail.generating.png"
    )
    variant_call = next(
        call for call in calls if Path(call["output_path"]).name == "variant_rain.png"
    )
    assert detail_call["kind"] == "image"
    assert [Path(path).name for path in detail_call["ref_image"]] == [
        "face_closeup.png",
        "full_body.png",
    ]
    assert variant_call["kind"] == "image"
    assert [Path(path).name for path in variant_call["ref_image"]] == [
        "face_closeup.png",
        "full_body.png",
    ]
    assert result["identity_detail"] == str(detail_path)
    assert run_quality_check("phase3", tmp_path).passed is True

    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({
            "characters": [
                {
                    "id": "photographer",
                    "name": "摄影师",
                    "prompt_definition": "将{图片N}中的摄影师定义为{主体N}",
                    "appearance": {"identity_props": identity_props},
                }
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    references = collect_character_reference_assets(
        tmp_path,
        {
            "who": ["摄影师"],
            "generation_actions": ["摄影师持续跟拍前方人物"],
        },
    )
    detail_asset = next(
        asset for asset in references if asset["path"].name == "identity_detail.png"
    )
    assert detail_asset["bind_subject"] is False
    assert detail_asset["identity_detail_active"] is True
    generation_contract = render_video_generation_contract(
        {
            "who": ["摄影师"],
            "generation_actions": ["摄影师持续跟拍前方人物"],
        },
        json.loads((tmp_path / "CHARACTERS.json").read_text(encoding="utf-8")),
    )
    assert "[identity-prop-reference-lock]" in generation_contract
    assert "camera_a" in generation_contract
    assert "exactly one matching instance owned by 摄影师" in generation_contract

    Image.new("RGB", (512, 512), "red").save(detail_path)
    assert run_quality_check("phase3", tmp_path).passed is False
