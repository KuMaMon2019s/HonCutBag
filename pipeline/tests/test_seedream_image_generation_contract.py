import base64
import hashlib
import io

import pytest
from PIL import Image

from clients import seedream_client
from clients.seedream_client import SeedreamClient
from prompt.seedream_image_prompt import (
    bind_reference_roles,
    image_request_fingerprint,
    prompt_guidance_metrics,
)


def _capture_payload(client: SeedreamClient, monkeypatch):
    observed = {}

    def fake_call(payload, output_path, timeout=180):
        observed.update(payload)
        observed["output_path"] = output_path
        observed["timeout"] = timeout
        return "https://image.invalid/result.png"

    monkeypatch.setattr(client, "_call_and_save", fake_call)
    return observed


def test_seedream_text_request_uses_agent_plan_quality_contract(monkeypatch, tmp_path):
    client = SeedreamClient(api_key="agent-plan-test-key")
    observed = _capture_payload(client, monkeypatch)

    client.text_to_image("一名特工在雨夜车站回头", output_path=str(tmp_path / "out.png"))

    assert client.model == "doubao-seedream-5.0-lite"
    assert observed == {
        "model": "doubao-seedream-5.0-lite",
        "prompt": "一名特工在雨夜车站回头",
        "size": "2K",
        "response_format": "url",
        "output_format": "png",
        "watermark": False,
        "sequential_image_generation": "disabled",
        "stream": False,
        "optimize_prompt_options": {"mode": "standard"},
        "output_path": str(tmp_path / "out.png"),
        "timeout": 60,
    }


def test_seedream_reference_request_uses_documented_single_image_contract(
    monkeypatch,
    tmp_path,
):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"not-a-real-png-but-valid-for-data-url-encoding")
    client = SeedreamClient(api_key="agent-plan-test-key")
    observed = _capture_payload(client, monkeypatch)

    client.image_to_image(
        "保持图1人物身份，改为侧身闪避",
        str(reference),
        output_path=str(tmp_path / "out.png"),
    )

    assert observed["image"].startswith("data:image/png;base64,")
    assert base64.b64decode(observed["image"].split(",", 1)[1]) == reference.read_bytes()
    assert "n" not in observed
    assert observed["size"] == "2K"
    assert observed["response_format"] == "url"
    assert observed["output_format"] == "png"
    assert observed["sequential_image_generation"] == "disabled"
    assert observed["stream"] is False
    assert observed["optimize_prompt_options"] == {"mode": "standard"}


@pytest.mark.parametrize("size", ["1500x1500", "2Kx2K", "5K"])
def test_seedream_rejects_unsupported_size_before_provider_call(
    size,
    monkeypatch,
    tmp_path,
):
    client = SeedreamClient(api_key="agent-plan-test-key")
    monkeypatch.setattr(
        client,
        "_call_and_save",
        lambda *_args, **_kwargs: pytest.fail("invalid size reached provider boundary"),
    )

    with pytest.raises(ValueError, match="Seedream 5.0 lite size"):
        client.text_to_image("test", output_path=str(tmp_path / "out.png"), size=size)


def test_seedream_rejects_more_than_fourteen_references_before_encoding(
    monkeypatch,
    tmp_path,
):
    client = SeedreamClient(api_key="agent-plan-test-key")
    references = [str(tmp_path / f"missing-{index}.png") for index in range(15)]
    monkeypatch.setattr(
        client,
        "_call_and_save",
        lambda *_args, **_kwargs: pytest.fail("invalid references reached provider boundary"),
    )

    with pytest.raises(ValueError, match="at most 14 reference images"):
        client.image_to_image("test", references, output_path=str(tmp_path / "out.png"))


def test_seedream_agent_plan_rejects_pay_as_you_go_model_ids():
    with pytest.raises(ValueError, match="Agent Plan image generation only supports"):
        SeedreamClient(
            api_key="agent-plan-test-key",
            model="doubao-seedream-5-0-lite-260128",
        )


def test_seedream_rejects_invalid_success_envelope_without_writing(
    monkeypatch,
    tmp_path,
):
    class InvalidSuccessResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"data": []}

    output = tmp_path / "out.png"
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "0")
    monkeypatch.setattr(
        seedream_client.requests,
        "post",
        lambda *_args, **_kwargs: InvalidSuccessResponse(),
    )

    with pytest.raises(RuntimeError, match="invalid non-streaming image envelope"):
        SeedreamClient(api_key="agent-plan-test-key").text_to_image(
            "test",
            output_path=str(output),
        )

    assert not output.exists()
    assert not output.with_suffix(".png.part").exists()


def test_reference_binding_names_every_input_in_provider_order():
    prompt = bind_reference_roles(
        "主体后仰闪避，刀锋掠过风衣；雨夜车站，16:9电影构图。",
        [
            "character_identity_only",
            "prior_storyboard_state",
            "director_single_panel_composition_only",
        ],
    )

    assert prompt.startswith("[honcut-seedream-reference-contract-v2]")
    assert "Image 1: character identity only" in prompt
    assert "Image 2: previous storyboard state" in prompt
    assert "Image 3: director single panel" in prompt
    assert prompt.index("Image 1") < prompt.index("Image 2") < prompt.index("Image 3")
    assert prompt.endswith("主体后仰闪避，刀锋掠过风衣；雨夜车站，16:9电影构图。")


def test_prior_storyboard_binding_preserves_space_but_never_locks_pose():
    prompt = bind_reference_roles(
        "推进到踢腕后的新终态。",
        ["prior_storyboard_state"],
    )

    assert "preserve camera axis, screen direction and relative spatial continuity" in prompt
    assert "do not copy its pose or action progress" in prompt
    assert "preserve camera axis, screen direction and the completed pose" not in prompt


def test_prompt_guidance_metrics_never_mutate_or_truncate_contract():
    prompt = "主体向前跨步。" * 80
    original = prompt

    metrics = prompt_guidance_metrics(prompt)

    assert metrics["sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert metrics["cjk_characters"] > 300
    assert metrics["over_recommended_length"] is True
    assert "prompt" not in metrics
    assert prompt == original


def test_image_request_fingerprint_binds_size_and_reference_order():
    reference_a = "a" * 64
    reference_b = "b" * 64
    baseline = image_request_fingerprint(
        prompt="same prompt",
        model="doubao-seedream-5.0-lite",
        size="2K",
        reference_image_sha256=[reference_a, reference_b],
    )

    assert baseline != image_request_fingerprint(
        prompt="same prompt",
        model="doubao-seedream-5.0-lite",
        size="3K",
        reference_image_sha256=[reference_a, reference_b],
    )
    assert baseline != image_request_fingerprint(
        prompt="same prompt",
        model="doubao-seedream-5.0-lite",
        size="2K",
        reference_image_sha256=[reference_b, reference_a],
    )


def test_seedream_download_rejects_invalid_image_without_overwriting(
    monkeypatch,
    tmp_path,
):
    class InvalidImageResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def iter_content(chunk_size=8192):
            del chunk_size
            yield b"not-an-image"

    destination = tmp_path / "existing.png"
    valid = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(valid, format="PNG")
    destination.write_bytes(valid.getvalue())
    original_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    monkeypatch.setattr(
        seedream_client.requests,
        "get",
        lambda *_args, **_kwargs: InvalidImageResponse(),
    )

    with pytest.raises(Exception):
        SeedreamClient(api_key="agent-plan-test-key")._download(
            "https://image.invalid/result.png",
            str(destination),
        )

    assert hashlib.sha256(destination.read_bytes()).hexdigest() == original_sha256
    assert not destination.with_suffix(".png.part").exists()
