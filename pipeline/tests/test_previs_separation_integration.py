# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients import seedream_client, tos_uploader
from phases import pipeline_core
from phases.phase1.storyboard_beats import plan_storyboard_beats
from phases.phase2.shot_storyboards import (
    generate_shot_storyboards,
    validate_shot_storyboard_artifacts,
)
from phases.phase4.cinematic_first_frames import (
    CINEMATIC_FIRST_FRAME_SCHEMA,
    validate_cinematic_first_frame_artifacts,
)
from phases.phase5.storyboard_qa_gate import run_l4_first_frame_review
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import _provider_content
from schemas.continuity import ContinuityPlan


def _signed_tos_url(monkeypatch, object_key: str) -> str:
    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_BUCKET", "honcut-fixtures")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    monkeypatch.setenv("TOS_REGION", "cn-beijing")
    return tos_uploader.get_signed_url(object_key)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DeterministicImageClient:
    """Draw visibly different PREVIS and cinematic assets without network I/O."""

    model = "offline-seedream-contract-fixture"

    def __init__(self):
        self.image_to_image_calls: list[dict[str, object]] = []

    @staticmethod
    def _render(output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if "video_first_frames" in path.parts:
            image = Image.new("RGB", (1280, 720), (8, 20, 64))
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 80, 1200, 640), outline=(255, 116, 24), width=28)
            draw.ellipse((520, 190, 760, 590), fill=(18, 24, 44), outline=(255, 142, 36), width=18)
        else:
            image = Image.new("RGB", (1280, 720), "white")
            draw = ImageDraw.Draw(image)
            draw.line((120, 520, 950, 180), fill="red", width=24)
            draw.polygon(((950, 180), (890, 182), (930, 240)), fill="red")
            draw.line((180, 140, 1080, 140), fill="blue", width=20)
            draw.text((90, 40), "S02_P01 subtle zoom in", fill="black")
        image.save(path, format="PNG")

    def text_to_image(self, prompt, output_path, size, timeout=180):
        self._render(output_path)
        return "offline://text-to-image"

    def image_to_image(self, prompt, ref_image, output_path, size):
        refs = [ref_image] if isinstance(ref_image, str) else list(ref_image)
        self.image_to_image_calls.append({
            "output_path": output_path,
            "references": refs,
        })
        self._render(output_path)
        return "offline://image-to-image"


class _CleanFirstFrameReviewer:
    def review(self, paths, prompt):
        assert paths
        assert all("video_first_frames" in path.parts for path in paths)
        assert "ANNOTATION_CONTAMINATION" in prompt
        assert "STYLE_MISMATCH" in prompt
        return json.dumps({"issues": []})


class _CompatibleStyleClassifier:
    def classify(self, path: Path) -> dict:
        return {
            "schema": "honcut.clip-style-classification.v1",
            "status": "done",
            "model": "offline-style-fixture",
            "source_sha256": _sha256(path),
            "top_style": "shadow_puppet",
            "rankings": [
                {"base_style": "shadow_puppet", "score": 0.3},
                {"base_style": "concept_art", "score": 0.2},
            ],
        }


def test_previs_pixels_are_separated_end_to_end_before_video_transport(
    tmp_path,
    monkeypatch,
    canonical_run_contract,
):
    client = _DeterministicImageClient()
    pipeline_core._write_project_visual_style(
        tmp_path,
        "皮影戏台，靛蓝幕布，炽橙轮廓光，平面剪影材质",
    )
    character_dir = tmp_path / "characters" / "puppet"
    character_dir.mkdir(parents=True)
    Image.effect_noise((640, 640), 60).convert("RGB").save(
        character_dir / "face_closeup.png"
    )
    Image.effect_noise((640, 960), 80).convert("RGB").save(
        character_dir / "full_body.png"
    )
    Image.effect_noise((640, 960), 70).convert("RGB").save(
        character_dir / "side.png"
    )
    Image.effect_noise((640, 960), 75).convert("RGB").save(
        character_dir / "back.png"
    )
    characters = [{
        "id": "puppet",
        "name": "银色傀儡",
        "appearance": {"summary": "平面银色皮影剪影"},
    }]
    storyboard = {
        "aspect_ratio": "16:9",
        "shots": [{
            "id": 2,
            "name": "银色傀儡抬手",
            "duration": 15,
            "who": ["puppet"],
            "_char_ids": ["puppet"],
            "where": "靛蓝幕布皮影戏台",
            "what": "银色傀儡在幕布中央抬手",
            "visual": "炽橙轮廓光勾勒银色傀儡",
            "prompt": "银色傀儡在靛蓝幕布皮影戏台中央缓慢抬手。",
            "gen_strategy": "phantom",
            "micro_actions": ["银色傀儡缓慢抬起右手"],
            "camera_movement": "slow dolly in",
            "start_state": "银色傀儡垂手立在幕布中央",
            "end_state": "右手停在肩部高度",
        }],
    }
    plan_storyboard_beats(storyboard)
    characters_data, _contract = canonical_run_contract(
        tmp_path,
        {"characters": characters},
    )
    characters = characters_data["characters"]

    # This deliberately annotated director cell is a legal Phase 4 composition
    # input, but it must be transformed into a clean cinematic frame before the
    # video transport boundary.
    director_panel = tmp_path / "director_panels" / "S02.png"
    client._render(str(director_panel))
    director_sha256 = _sha256(director_panel)

    generate_shot_storyboards(tmp_path, storyboard, characters, client=client)
    assert validate_shot_storyboard_artifacts(tmp_path, storyboard) == []
    previs_path = tmp_path / "storyboard_beats" / "S02_P01.png"
    nine_grid_path = tmp_path / "shot_storyboards" / "S02.png"
    narrative_guide_path = tmp_path / "storyboard_guides" / "S02_P01.png"
    previs_sha256 = _sha256(previs_path)
    nine_grid_sha256 = _sha256(nine_grid_path)
    narrative_guide_sha256 = _sha256(narrative_guide_path)
    recovered_provider_calls = len(client.image_to_image_calls)
    generate_shot_storyboards(tmp_path, storyboard, characters, client=client)
    recovered_manifest_sha256 = _sha256(tmp_path / "SHOT_STORYBOARDS.json")
    for _ in range(10):
        generate_shot_storyboards(tmp_path, storyboard, characters, client=client)
        assert validate_shot_storyboard_artifacts(tmp_path, storyboard) == []
        assert _sha256(narrative_guide_path) == narrative_guide_sha256
        assert _sha256(tmp_path / "SHOT_STORYBOARDS.json") == recovered_manifest_sha256
    assert len(client.image_to_image_calls) == recovered_provider_calls
    phase2_alias_receipt = json.loads(
        (tmp_path / "storyboard_images" / "S02.json").read_text(encoding="utf-8")
    )
    assert phase2_alias_receipt["status"] == "previs_only"

    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(
            {
                **storyboard,
                "canonical_visual_contract": "CANONICAL_VISUAL_CONTRACT.json",
                "canonical_visual_contract_sha256": _contract[
                    "contract_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(seedream_client, "SeedreamClient", lambda: client)
    monkeypatch.setattr(
        "utils.clip_style_classifier.ClipStyleClassifier",
        _CompatibleStyleClassifier,
    )
    phase4 = pipeline_core.run_phase4(tmp_path, dry_run=False)
    assert phase4["status"] == "done"
    assert phase4["constraint_review"]["status"] == "passed"

    storyboard = json.loads(
        (tmp_path / "STORYBOARD.json").read_text(encoding="utf-8")
    )
    beat = storyboard["shots"][0]["storyboard_beats"][0]
    assert beat["video_first_frame_kind"] == CINEMATIC_FIRST_FRAME_SCHEMA
    assert validate_cinematic_first_frame_artifacts(tmp_path, storyboard) == []
    cinematic_path = tmp_path / beat["video_first_frame"]
    cinematic_sha256 = _sha256(cinematic_path)
    assert cinematic_sha256 not in {
        director_sha256,
        previs_sha256,
        nine_grid_sha256,
    }
    assert _sha256(tmp_path / "storyboard_images" / "S02.png") == cinematic_sha256
    cinematic_call = next(
        call
        for call in client.image_to_image_calls
        if Path(str(call["output_path"])) == cinematic_path
    )
    assert str(director_panel) in cinematic_call["references"]
    cinematic_receipt = json.loads(
        (tmp_path / beat["video_first_frame_receipt"]).read_text(encoding="utf-8")
    )
    assert cinematic_receipt["upstream_director_panel"] == "director_panels/S02.png"
    assert cinematic_receipt["upstream_director_panel_sha256"] == director_sha256
    assert cinematic_receipt["previs_reference_images"] == []

    continuity = json.loads(
        (tmp_path / "CONTINUITY_PLAN.json").read_text(encoding="utf-8")
    )
    chunk = continuity["shots"][0]["chunks"][0]
    assert chunk["storyboard_image"] == beat["video_first_frame"]
    assert chunk["storyboard_image_kind"] == CINEMATIC_FIRST_FRAME_SCHEMA
    assert chunk["storyboard_narrative_guide"] == (
        "storyboard_guides/S02_P01.png"
    )
    assert chunk["storyboard_narrative_guide_cell_ids"] == [
        f"S02_G{index:02d}" for index in range(1, 10)
    ]

    assert chunk["storyboard_narrative_guide_pose_contract_schema"] == (
        "honcut.storyboard-guide-pose-contract.v3"
    )
    assert len(chunk["storyboard_narrative_guide_pose_fingerprints"]) == 9
    l4_issues, l4 = run_l4_first_frame_review(
        storyboard,
        (tmp_path / "visual-style.md").read_text(encoding="utf-8"),
        {"S02_P01": cinematic_path},
        tmp_path,
        _CleanFirstFrameReviewer(),
    )
    assert l4_issues == []
    assert l4["status"] == "completed"

    uploaded_sha256: list[str] = []

    def _record_upload(image_data, content_type):
        digest = hashlib.sha256(image_data).hexdigest()
        uploaded_sha256.append(digest)
        return _signed_tos_url(monkeypatch, f"fixture/{digest}.png")

    monkeypatch.setattr(
        tos_uploader,
        "upload_image_required",
        _record_upload,
    )
    continuity_plan = ContinuityPlan.model_validate(continuity)
    continuity_chunk = continuity_plan.shots[0].chunks[0]
    content, *_ = _provider_content(
        tmp_path,
        ChunkExecutionRequest(
            resource_id=continuity_chunk.chunk_id,
            shot_id="S02",
            chunk=continuity_chunk,
            anchors={},
            output_path=tmp_path / "shots/S02/chunks/S02_C01.mp4",
            previous_output_path=None,
            input_fingerprint="previs-separation-fixture",
            memory_context="",
        ),
    )
    prompt = next(item["text"] for item in content if item["type"] == "text")
    image_roles = [
        item.get("role") for item in content if item["type"] == "image_url"
    ]
    assert image_roles
    assert set(image_roles) == {"reference_image"}
    identity_board_sha256 = _sha256(
        tmp_path / "characters" / "puppet" / "reference_board.png"
    )
    pose_atlas_candidates = chunk["storyboard_pose_atlas_candidates"]
    selected_pose_atlas = next(
        candidate
        for candidate in pose_atlas_candidates
        if candidate["preferred"] is True
    )
    pose_atlas_sha256 = [
        page["image_sha256"] for page in selected_pose_atlas["pages"]
    ]
    assert uploaded_sha256 == [
        identity_board_sha256,
        cinematic_sha256,
        *pose_atlas_sha256,
    ]
    assert cinematic_sha256 in uploaded_sha256
    assert narrative_guide_sha256 not in uploaded_sha256
    assert director_sha256 not in uploaded_sha256
    assert previs_sha256 not in uploaded_sha256
    assert nine_grid_sha256 not in uploaded_sha256
    assert chunk["storyboard_narrative_guide_source_board_sha256"] == (
        nine_grid_sha256
    )
    assert "图片2只锁定t=0构图，不得冻结动作" in prompt
    assert "成片质感第一帧" in prompt
    assert "图片3是当前动作顺序、根位移和重心轨迹权威" in prompt
    assert "零时长初始锚点=G01" in prompt
    assert "视频时间从紧随其后的动态姿态开始" in prompt
    assert "唯一允许动作组数量=" in prompt
    assert "不得把多姿态画成克隆" in prompt
    assert "honcut.phase6-media-role-isolation.v1" in prompt
    assert "非权威占位像素" in prompt
