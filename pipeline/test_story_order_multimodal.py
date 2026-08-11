import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients.ark_multimodal_client import ArkMultimodalClient
import phases.phase8.story_order_reviewer as story_order_reviewer


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_ark_client_constructs_one_request_with_multiple_images(tmp_path):
    first = tmp_path / "S01.png"
    second = tmp_path / "S02.jpg"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    completions = FakeCompletions('{"ok": true}')
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    client = ArkMultimodalClient(client=transport, model="doubao-seed-2.1-turbo")
    assert client.review([first, second], "review these") == '{"ok": true}'

    request = completions.kwargs
    assert request["model"] == "doubao-seed-2.1-turbo"
    assert request["response_format"] == {"type": "json_object"}
    content = request["messages"][0]["content"]
    urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(urls) == 2
    assert base64.b64decode(urls[0].split(",", 1)[1]) == b"first-image"
    assert urls[1].startswith("data:image/jpeg;base64,")


def test_review_parses_and_validates_ark_json(tmp_path):
    images = [tmp_path / "S01.png", tmp_path / "S02.png"]
    for image in images:
        image.write_bytes(b"image")
    reviewer = SimpleNamespace(
        review=lambda paths, prompt: json.dumps(
            {
                "suggested_order": ["S02", "S01"],
                "narrative_consistent": False,
                "issues": ["chronology"],
            }
        )
    )
    result = story_order_reviewer.review_with_multimodal_llm(
        {"script": "complete story", "shots": [{"shot_id": "S01"}, {"shot_id": "S02"}]},
        images,
        client=reviewer,
    )
    assert result["suggested_order"] == ["S02", "S01"]


def test_real_mode_propagates_client_failure(tmp_path, monkeypatch):
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"shot_id": "S01"}]}), encoding="utf-8"
    )
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    (image_dir / "S01.png").write_bytes(b"image")
    monkeypatch.setenv("HONCUT_STORYBOARD_REVIEW", "real")
    monkeypatch.setattr(
        story_order_reviewer,
        "review_with_multimodal_llm",
        lambda *args: (_ for _ in ()).throw(RuntimeError("ARK failed")),
    )

    with pytest.raises(RuntimeError, match="ARK failed"):
        story_order_reviewer.review_story_order(tmp_path, ["S01"])
    assert not (tmp_path / "storyboard_order_review.json").exists()


def test_mock_mode_never_calls_client(tmp_path, monkeypatch):
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps({"shots": [{"shot_id": "S01"}]}), encoding="utf-8"
    )
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    (image_dir / "S01.png").write_bytes(b"image")
    monkeypatch.setenv("HONCUT_STORYBOARD_REVIEW", "mock")
    monkeypatch.setattr(
        story_order_reviewer,
        "review_with_multimodal_llm",
        lambda *args: pytest.fail("client should not be called in mock mode"),
    )

    result = story_order_reviewer.review_story_order(tmp_path, ["S01"])
    assert result["source"] == "mock"
