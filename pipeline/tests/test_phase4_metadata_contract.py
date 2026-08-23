"""Phase 4 must preserve structured storyboard fields for Phase 6."""

import json

from phases.phase4.shot_setup import materialize_shot_directories, normalize_shots
from phases.phase6.video_generator import _shot_number


def test_parse_shots_preserves_generation_contract_fields():
    source = {
        "id": 2,
        "name": "attack",
        "duration": 5,
        "prompt": "legacy prompt",
        "what": "complete action",
        "action_description": "complete action description",
        "shot_type": "medium",
        "shot_size": "medium",
        "camera_movement": "tracking_right",
        "where": "rainy overpass",
        "dialogue": {"speaker": "烬", "line": "台词"},
        "speech_duration_s": 2,
        "generation_actions": ["冲刺", "格挡"],
        "generation_load": 2,
        "source_action_unit_ids": ["AU001", "AU002"],
        "start_state": "相隔数米",
        "end_state": "兵刃相接",
        "causal_link": "冲刺导致碰撞",
    }

    parsed = normalize_shots({"shots": [source]}, storyboard_dir=None)[0]

    for key in (
        "action_description",
        "shot_type",
        "shot_size",
        "camera_movement",
        "where",
        "dialogue",
        "speech_duration_s",
        "generation_actions",
        "generation_load",
        "source_action_unit_ids",
        "start_state",
        "end_state",
        "causal_link",
    ):
        assert parsed[key] == source[key]


def test_phase6_reads_normalized_shot_id_from_phase4_metadata():
    assert _shot_number({"shot_id": "S02"}) == 2


def test_setup_shot_dirs_writes_generation_contract(tmp_path):
    source = {
        "id": 1,
        "shot_id": "S01",
        "name": "attack",
        "duration": 4,
        "prompt": "attack prompt",
        "route": "img2vid",
        "route_reason": "reference",
        "caption": "",
        "caption_frames": "",
        "generation_actions": ["冲刺", "格挡"],
        "generation_load": 2,
        "source_action_unit_ids": ["AU001", "AU002"],
        "start_state": "相隔数米",
        "end_state": "兵刃相接",
        "causal_link": "冲刺导致碰撞",
    }

    [meta_path] = materialize_shot_directories(tmp_path / "shots", [source])
    written = json.loads(meta_path.read_text(encoding="utf-8"))

    for key in (
        "generation_actions",
        "generation_load",
        "source_action_unit_ids",
        "start_state",
        "end_state",
        "causal_link",
    ):
        assert written[key] == source[key]
