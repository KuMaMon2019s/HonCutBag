from __future__ import annotations

import json

import pytest

from runtime.generation_fingerprint import build_generation_fingerprint


def _fingerprint(**overrides):
    values = {
        "prompt_text": "A subject crosses the frame.",
        "prompt_template_id": "honcut.phase6.video",
        "prompt_template_version": "1",
        "provider_id": "seedance",
        "provider_version": "ark-agent-plan-v3",
        "model_id": "doubao-seedance-2.0-fast",
        "model_version": "doubao-seedance-2.0-fast",
        "parameters": {"duration": 8, "ratio": "16:9", "seed": 7},
        "input_artifact_hashes": {"storyboard": "a" * 64},
    }
    values.update(overrides)
    return build_generation_fingerprint(**values)


def test_generation_fingerprint_is_deterministic_for_semantic_inputs():
    first = _fingerprint(
        parameters={"seed": 7, "duration": 8, "nested": {"b": 2, "a": 1}}
    )
    second = _fingerprint(
        parameters={"nested": {"a": 1, "b": 2}, "duration": 8, "seed": 7}
    )
    assert first.value == second.value
    assert first.material == second.material
    assert len(first.value) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_text", "A different authored action."),
        ("prompt_template_version", "2"),
        ("provider_version", "ark-agent-plan-v4"),
        ("model_version", "seedance-revision-2"),
        ("parameters", {"duration": 12, "ratio": "16:9", "seed": 7}),
        ("input_artifact_hashes", {"storyboard": "b" * 64}),
    ],
)
def test_generation_fingerprint_changes_for_every_semantic_dimension(field, value):
    assert _fingerprint(**{field: value}).value != _fingerprint().value


def test_generation_fingerprint_excludes_credentials_recursively():
    first = _fingerprint(
        parameters={
            "duration": 8,
            "api_key": "secret-one",
            "transport": {"Authorization": "Bearer first", "timeout": 30},
        }
    )
    second = _fingerprint(
        parameters={
            "duration": 8,
            "api_key": "secret-two",
            "transport": {"Authorization": "Bearer second", "timeout": 30},
        }
    )
    serialized = json.dumps(first.material, sort_keys=True)
    assert first.value == second.value
    assert "secret-one" not in serialized
    assert "Authorization" not in serialized


def test_semantic_token_limits_are_not_mistaken_for_credentials():
    assert _fingerprint(parameters={"max_tokens": 100}).value != _fingerprint(
        parameters={"max_tokens": 200}
    ).value


def test_generation_fingerprint_rejects_invalid_asset_hash():
    with pytest.raises(ValueError, match="SHA-256"):
        _fingerprint(input_artifact_hashes={"storyboard": "not-a-hash"})
