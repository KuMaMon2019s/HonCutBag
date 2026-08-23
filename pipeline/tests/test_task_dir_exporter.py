import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients import tos_uploader
from tools import asset_packager, task_dir_exporter


def _signed_tos_url(monkeypatch, object_key: str) -> str:
    monkeypatch.setenv("TOS_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("TOS_SECRET_KEY", "test-sk")
    monkeypatch.setenv("TOS_BUCKET", "honcut-fixtures")
    monkeypatch.setenv("TOS_ENDPOINT", "tos-cn-beijing.volces.com")
    monkeypatch.setenv("TOS_REGION", "cn-beijing")
    return tos_uploader.get_signed_url(object_key)


def _image(path: Path, marker: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(marker * 2048)


def _multi_character_project(tmp_path: Path) -> dict:
    frame = tmp_path / "video_first_frames" / "S02_P01.png"
    _image(frame, b"s")
    frame.with_suffix(".json").write_text(
        json.dumps({
            "kind": "honcut.cinematic-first-frame.v1",
            "status": "done",
            "image_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            "previs_reference_images": [],
        }),
        encoding="utf-8",
    )
    for char_id, marker in (("black_merc", b"b"), ("silver_tech", b"t")):
        char_dir = tmp_path / "characters" / char_id
        _image(char_dir / "face_closeup.png", marker)
        _image(char_dir / "full_body.png", marker.upper())
    _image(tmp_path / "characters" / "black_merc" / "variant_b.png", b"2")
    _image(tmp_path / "characters" / "black_merc" / "variant_a.png", b"1")
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({"characters": [
            {"id": "black_merc", "name": "黑色佣兵"},
            {"id": "silver_tech", "name": "银色技师"},
        ]}),
        encoding="utf-8",
    )
    return {
        "shot_id": "S02",
        "prompt": "两名角色在机库交谈。",
        "gen_strategy": "phantom",
        "_char_ids": ["black_merc", "silver_tech"],
        "_storyboard_frame_path": "video_first_frames/S02_P01.png",
        "_storyboard_frame_kind": "honcut.cinematic-first-frame.v1",
        "duration": 12,
        "width": 1280,
        "height": 720,
        "generate_audio": True,
    }


def test_task_dir_and_content_promote_phantom_to_strict_first_frame(tmp_path, monkeypatch):
    shot_meta = _multi_character_project(tmp_path)
    monkeypatch.setattr(
        tos_uploader,
        "upload_image",
        lambda image_data, content_type: _signed_tos_url(
            monkeypatch, f"fixture/{image_data[:1].hex()}.png"
        ),
    )
    content = asset_packager.build_content_for_shot(tmp_path, "S02", shot_meta)
    task_dir = task_dir_exporter.build_task_dir(tmp_path, ["S02"], shot_meta)
    manifest = json.loads((task_dir / "S02" / "manifest.json").read_text(encoding="utf-8"))
    prompt = (task_dir / "S02" / "提示词" / "提示词.txt").read_text(encoding="utf-8")

    content_prompt = next(item["text"] for item in content if item["type"] == "text")
    assert [item.get("role") for item in content if item["type"] == "image_url"] == [
        "first_frame"
    ]
    assert manifest["strategy"] == "i2v"
    assert manifest["references"] == []
    assert "分镜/分镜图.png是S02成片质感首帧" in prompt
    assert "构图、角色站位、场景结构、项目美术风格、时间天气和光影" in prompt
    assert "图片1为S02成片质感第一帧" in content_prompt
    assert not (task_dir / "S02" / "black_merc").exists()
    assert not (task_dir / "S02" / "silver_tech").exists()


def test_task_directory_matches_contract_structure(tmp_path):
    shot_meta = _multi_character_project(tmp_path)
    task_dir = task_dir_exporter.build_task_dir(tmp_path, ["S02"], shot_meta)
    required = [
        "task_manifest.json",
        "S02/manifest.json",
        "S02/分镜/分镜图.png",
        "S02/提示词/提示词.txt",
    ]
    assert all((task_dir / relative).is_file() for relative in required)
    manifest = json.loads((task_dir / "S02" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "shot_id": "S02",
        "strategy": "i2v",
        "first_frame": "分镜/分镜图.png",
        "last_frame": None,
        "references": manifest["references"],
        "prompt_file": "提示词/提示词.txt",
        "duration": 12,
        "width": 1280,
        "height": 720,
        "generate_audio": True,
    }


def test_upload_preserves_utf8_object_keys_without_network(tmp_path, monkeypatch):
    shot_meta = _multi_character_project(tmp_path)
    task_dir = task_dir_exporter.build_task_dir(tmp_path, ["S02"], shot_meta)
    uploads = []

    def fake_upload(image_data, object_key, content_type="image/png"):
        uploads.append((object_key, content_type))
        return f"https://example.test/{object_key}"

    monkeypatch.setattr(tos_uploader, "upload_file", fake_upload)
    task_id = task_dir_exporter.upload_task_dir(task_dir, "tasks")

    assert task_id == task_dir.name
    keys = [key for key, _ in uploads]
    assert f"tasks/{task_id}/S02/提示词/提示词.txt" in keys
    assert f"tasks/{task_id}/S02/分镜/分镜图.png" in keys
    assert not any("大头照" in key or "全身照" in key for key in keys)
    assert all("%" not in key for key in keys)


def test_task_dir_mode_default_keeps_legacy_content_payload(monkeypatch):
    monkeypatch.delenv("HONCUT_TASK_DIR_MODE", raising=False)
    posted = {}

    class FakeSession:
        def post(self, url, json, timeout):
            posted.update(json)
            return SimpleNamespace(
                status_code=200,
                text="",
                json=lambda: {"task_id": "legacy-task", "images_used": 1},
            )

    monkeypatch.setattr("clients.local_video_client._request_session", FakeSession)
    from clients import local_video_client

    task_id = local_video_client.submit(
        prompt="legacy prompt",
        content=[{"type": "text", "text": "legacy prompt"}],
    )

    assert task_id == "legacy-task"
    assert "content" in posted
    assert "task_dir" not in posted
