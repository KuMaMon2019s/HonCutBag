import json
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "legacy"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(VENDOR_DIR))

import adaptation_engine
import asset_packager
import local_video_client
import pipeline_runner


def test_orchestrator_writes_gen_strategy_to_shot_meta(monkeypatch, tmp_path):
    import orchestrator

    monkeypatch.setattr(orchestrator, "SHOTS_DIR", tmp_path / "shots")
    parsed = orchestrator.parse_shots({"shots": [{
        "id": 1,
        "name": "action",
        "duration": 5,
        "prompt": "walks over",
        "caption": "",
        "caption_frames": "",
        "gen_strategy": "flf2v",
    }]})
    parsed = [orchestrator.route_shot(shot, {}) for shot in parsed]
    orchestrator.setup_shot_dirs(parsed)
    meta = json.loads((tmp_path / "shots" / "S01" / "SHOT_META.json").read_text())
    assert meta["gen_strategy"] == "flf2v"


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        ({"visual": "林晓抬手拂发", "who": ["林晓"], "shot_intent": "action"}, "flf2v"),
        ({"what": "陈阳走来坐下", "who": ["陈阳"]}, "flf2v"),
        ({"visual": "两人面对面低声交谈", "who": ["林晓", "陈阳"]}, "phantom"),
        ({"visual": "林晓眼含泪水的表情特写", "who": ["林晓"]}, "phantom"),
        ({"visual": "夕阳映照湖面，柳枝轻拂", "who": []}, "i2v"),
        ({"visual": "抽象而不确定的画面", "who": ["林晓"]}, "i2v"),
    ],
)
def test_deterministic_generation_strategy(shot, expected):
    assert adaptation_engine.determine_gen_strategy(shot) == expected


def test_end_frame_prompt_describes_completed_action_and_keeps_source_prompt():
    prompt = pipeline_runner.build_end_frame_prompt({
        "prompt": "Golden-hour medium shot, photorealistic cinematography.",
        "visual": "林晓抬手拂发",
    })
    assert "Golden-hour medium shot" in prompt
    assert "after the action has fully completed" in prompt
    assert "林晓抬手拂发" in prompt
    assert "Preserve the exact visual style" in prompt


def test_end_frame_generation_skips_existing_large_file(monkeypatch, tmp_path):
    first = tmp_path / "S01.png"
    end = tmp_path / "S01_end.png"
    first.write_bytes(b"x" * 2048)
    end.write_bytes(b"x" * (10 * 1024 + 1))

    class ForbiddenClient:
        def __init__(self):
            pytest.fail("Seedream must not be constructed for an existing end frame")

    import seedream_client
    monkeypatch.setattr(seedream_client, "SeedreamClient", ForbiddenClient)
    assert pipeline_runner._generate_flf2v_end_frame(
        {"gen_strategy": "flf2v", "prompt": "shot"}, "S01", first, None
    ) is False


@pytest.mark.parametrize(
    ("meta", "expected_model"),
    [
        ({"gen_strategy": "flf2v"}, "flf2v"),
        ({"gen_strategy": "phantom"}, "phantom"),
        ({"gen_strategy": "i2v"}, "wan22"),
        ({}, "wan22"),
    ],
)
def test_phase5_routes_model_and_defaults_old_metadata(
    monkeypatch, tmp_path, meta, expected_model
):
    shot_dir = tmp_path / "shots" / "S01"
    shot_dir.mkdir(parents=True)
    shot_meta = {"prompt": "test prompt", "duration": 5, **meta}
    (shot_dir / "SHOT_META.json").write_text(json.dumps(shot_meta))

    captured = {}
    monkeypatch.setattr(local_video_client, "is_available", lambda timeout: True)
    monkeypatch.setattr(
        asset_packager,
        "build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": "test prompt"},
                          {"type": "image_url", "role": "first_frame"}],
    )

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"x" * (10 * 1024 + 1))

    monkeypatch.setattr(local_video_client, "generate_video", fake_generate_video)
    result = pipeline_runner._run_phase5_fallback(tmp_path)

    assert result["status"] == "done"
    assert captured["model"] == expected_model


def test_phantom_content_has_first_frame_and_character_three_views(monkeypatch, tmp_path):
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    (image_dir / "S02.png").write_bytes(b"x" * 2048)
    char_dir = tmp_path / "characters" / "lin_xiao"
    char_dir.mkdir(parents=True)
    for view in ("front", "side", "back"):
        (char_dir / f"{view}.png").write_bytes(b"x" * 2048)

    import tos_uploader
    monkeypatch.setattr(tos_uploader, "upload_image", lambda *_: "https://example.test/image")
    content = asset_packager.build_content_for_shot(
        tmp_path,
        "S02",
        {"prompt": "dialogue", "gen_strategy": "phantom", "_char_ids": ["lin_xiao"]},
    )
    roles = [item.get("role") for item in content if item["type"] == "image_url"]
    assert roles == ["first_frame", "reference_image", "reference_image", "reference_image"]


def test_flf2v_content_has_first_and_last_frames(monkeypatch, tmp_path):
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    (image_dir / "S03.png").write_bytes(b"x" * 2048)
    (image_dir / "S03_end.png").write_bytes(b"x" * 2048)

    import tos_uploader
    monkeypatch.setattr(tos_uploader, "upload_image", lambda *_: "https://example.test/image")
    content = asset_packager.build_content_for_shot(
        tmp_path, "S03", {"prompt": "action", "gen_strategy": "flf2v"}
    )
    roles = [item.get("role") for item in content if item["type"] == "image_url"]
    assert roles == ["first_frame", "last_frame"]


def test_submit_only_includes_model_when_provided(monkeypatch):
    class Response:
        status_code = 200
        text = ""
        def json(self):
            return {"task_id": "task-1"}

    class Session:
        def __init__(self):
            self.payloads = []
        def post(self, *args, **kwargs):
            self.payloads.append(kwargs["json"])
            return Response()

    session = Session()
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    local_video_client.submit("prompt", model="phantom")
    local_video_client.submit("prompt")
    assert session.payloads[0]["model"] == "phantom"
    assert "model" not in session.payloads[1]
