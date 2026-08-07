import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clients import local_video_client
from phases.character_discoverer import _filter_descriptive_phrases
from phases.character_discoverer import _add_reference_contract
from phases.character_factory import build_model_reference_prompts
from phases.scene_consistency import generate_scene_consistency
from phases.storyboard_generator import _build_shot_prompt
from phases.video_generator import BASE_NEGATIVE_PROMPT, build_video_prompt


class _DownloadResponse:
    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"downloaded-video"


def test_download_keeps_file_when_ffprobe_metadata_is_unreadable(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "output.mp4"
    session = SimpleNamespace(get=lambda *args, **kwargs: _DownloadResponse())
    monkeypatch.setattr(local_video_client, "_request_session", lambda: session)
    monkeypatch.setattr(
        local_video_client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid data"
        ),
    )

    local_video_client.download(
        "task-123",
        str(output_path),
        expected_duration=10.0,
        expected_width=1280,
        expected_height=720,
    )

    assert output_path.read_bytes() == b"downloaded-video"
    assert "WARNING: metadata unavailable" in capsys.readouterr().err


def test_compound_robot_name_passes_character_filter():
    name = "白色金属AI巡检机器人"
    stats = {name: {"count": 3, "events": ["E01", "E02", "E03"]}}

    filtered = _filter_descriptive_phrases(stats)

    assert name in filtered


def test_storyboard_prompt_uses_eight_layers_without_fast_motion():
    shot = {
        "id": 1,
        "who": ["林夏"],
        "shot_size": "close_up",
        "what": "转身看向窗外",
        "emotion": "紧张",
        "where": "夜间办公室",
        "duration": 6,
    }
    characters = [{
        "id": "lin_xia", "name": "林夏", "aliases": [],
        "appearance": {"hair": "黑色短发", "face": "圆脸杏眼", "clothing": "白衬衫"},
    }]

    prompt = _build_shot_prompt(shot, characters)

    for layer in ("元素参考声明", "镜头1", "动作：", "运镜：", "场景与光影：", "音效：", "全局收尾："):
        assert layer in prompt
    assert "fast" not in prompt.lower()
    assert "快速" not in prompt


def test_character_reference_contract_and_model_routes():
    character = {
        "name": "林夏",
        "appearance": {"hair": "黑色短发", "face": "圆脸", "clothing": "白衬衫"},
    }
    _add_reference_contract(character)
    assert character["face_reference"] == "face_closeup.png"
    assert character["body_reference"] == "full_body.png"
    assert "<主体1>" in character["prompt_definition"]
    assert set(build_model_reference_prompts("角色", target_model="seedance")) == {"face_closeup", "full_body"}
    assert set(build_model_reference_prompts("角色", target_model="kling")) == {"front", "side", "three_quarter", "detail"}


def test_scene_contract_and_video_prompt_model_routing():
    storyboard = {"shots": [{
        "id": 1, "who": ["林夏"], "where": "夜间工厂", "shot_size": "medium",
        "what": "左手攥紧衣角", "camera_movement": "push_in", "duration": 8,
    }]}
    characters = {"characters": [{
        "id": "lin_xia", "name": "林夏", "aliases": [],
        "appearance": {"hair": "黑色短发", "face": "圆脸", "clothing": "白衬衫"},
        "prompt_definition": "将图片1中的[白衬衫、圆脸、黑色短发]定义为<主体1>",
        "negative_guardrails": "更换服装, 颜色偏移",
    }]}
    contract = generate_scene_consistency(storyboard, characters)
    scene = contract["shots"]["S01"]
    assert "5600K" in scene["lighting_description"]
    assert scene["spatial_layout"]["subject"]

    seedance = build_video_prompt(storyboard["shots"][0], characters, contract, "seedance")
    kling = build_video_prompt(storyboard["shots"][0], characters, contract, "kling")
    assert isinstance(seedance, str) and "identity-lock" in seedance and BASE_NEGATIVE_PROMPT in seedance
    assert isinstance(kling, dict) and BASE_NEGATIVE_PROMPT in kling["negative_prompt"]
    assert "推入(push in)" in kling["prompt"]
