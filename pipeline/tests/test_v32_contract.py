"""Unit tests for the HonCut Bridge v3.2 contract adaptation."""

import asyncio

from clients.invokeai_client import InvokeAIClient
from clients.video_client import VideoClient, VideoResult
from phases.video_generator import VideoGenerator
from utils.config import VideoModel


class Response:
    def __init__(self, data):
        self.data = data
        self.checked = False

    def raise_for_status(self):
        self.checked = True

    def json(self):
        return self.data


class Session:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.data)


def test_video_model_supports_all_v32_routes():
    assert {model.value for model in VideoModel} == {"wan22", "phantom", "flf2v", "seedance"}
    assert VideoModel("seedance") is VideoModel.SEEDANCE


def test_generate_with_assets_posts_v32_payload():
    session = Session({"task_id": "task-32", "status": "queued"})
    client = VideoClient("wan", session=session, bridge_url="http://bridge/")

    result = asyncio.run(
        client.generate_with_assets("char-1", model="seedance", prompt="wave")
    )

    assert result.task_id == "task-32"
    assert session.calls == [("http://bridge/generate_with_assets", {"json": {
        "asset_id": "char-1", "asset_type": "character", "image_index": 0,
        "model": "seedance", "prompt": "wave", "num_frames": 49, "steps": 20,
        "cfg": 5.0, "seed": -1, "width": 1280, "height": 720, "fps": 24,
    }})]


def test_import_from_invokeai_posts_v32_payload():
    session = Session({"success": True, "imported": 2})
    client = InvokeAIClient("http://bridge/", session=session)

    result = asyncio.run(
        client.import_from_invokeai("char-1", ["a.png", "b.png"], target_view="side")
    )

    assert result == {"success": True, "imported": 2}
    assert session.calls[0] == ("http://bridge/import_from_invokeai", {"json": {
        "asset_id": "char-1", "asset_type": "character",
        "image_names": ["a.png", "b.png"], "target_view": "side",
        "invokeai_url": "http://127.0.0.1:9090",
    }})


class GeneratorClient:
    def __init__(self):
        self.calls = []

    async def generate_with_assets(self, **kwargs):
        self.calls.append(("assets", kwargs))
        return VideoResult("wan22", "bridge", task_id="asset-task")

    def generate(self, **kwargs):
        self.calls.append(("legacy", kwargs))
        return VideoResult("wan22", "bridge", task_id="legacy-task")


def test_video_generator_uses_assets_when_asset_id_is_present():
    client = GeneratorClient()
    state = {"character_asset_id": "char-7", "model": "phantom", "prompt": "turn"}
    result = asyncio.run(VideoGenerator(client).run(state))
    assert client.calls == [("assets", {
        "asset_id": "char-7", "asset_type": "character", "image_index": 0,
        "model": "phantom", "prompt": "turn",
    })]
    assert result["video_result"].task_id == "asset-task"


def test_video_generator_keeps_legacy_path_without_asset_id():
    client = GeneratorClient()
    state = {"prompt": "turn", "reference_images": ["front.png"]}
    result = asyncio.run(VideoGenerator(client).run(state))
    assert client.calls == [("legacy", {
        "prompt": "turn", "reference_images": ["front.png"],
    })]
    assert result["video_result"].task_id == "legacy-task"
