"""Regression tests for M2 per-shot storyboard image naming."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline_runner import _generate_shot_images, _normalize_shot_id


def test_id_only_shots_create_distinct_zero_padded_filenames(tmp_path, monkeypatch):
    calls = []

    def fake_text_to_image(*, prompt, output_path, size=None):
        calls.append(output_path)
        # Match the real client's successful side effect.
        Path(output_path).write_bytes(b"mock image")

    monkeypatch.setitem(
        sys.modules,
        "clients.seedream_client",
        types.SimpleNamespace(text_to_image=fake_text_to_image),
    )
    storyboard = {
        "shots": [{"id": shot_id, "prompt": f"shot {shot_id}"} for shot_id in range(1, 9)]
    }

    assert _generate_shot_images(tmp_path, storyboard) == 8
    assert sorted(path.name for path in (tmp_path / "storyboard_images").iterdir()) == [
        f"S{shot_id:02d}.png" for shot_id in range(1, 9)
    ]
    assert len(set(calls)) == 8


def test_legacy_shot_ids_and_order_remain_supported():
    assert _normalize_shot_id({"shot_id": "S01"}) == "S01"
    assert _normalize_shot_id({"shot_id": 2}) == "S02"
    assert _normalize_shot_id({"shot_order": 3}) == "S03"
    assert _normalize_shot_id({}) is None
