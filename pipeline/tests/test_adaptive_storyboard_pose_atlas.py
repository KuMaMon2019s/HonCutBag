from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from PIL import Image, ImageChops, ImageFont

from phases.phase2 import shot_storyboards as storyboard_owner
from phases.phase2 import storyboard_guide_pose as pose_owner
from phases.phase2.storyboard_pose_atlas import (
    build_pose_atlas_plan,
    render_pose_atlas_candidates,
    select_pose_atlas_candidate,
)
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import _bind_final_media_index_prompt
from schemas.continuity import GenerationChunk
from utils.camera_motion_contracts import (
    apply_camera_motion_contract,
    build_camera_motion_contract,
    camera_motion_minimum_duration_s,
    validate_camera_motion_duration,
)
from utils.video_capabilities import SEEDANCE_2_CAPABILITIES


def _unit(index: int, action: str) -> dict:
    return {
        "unit_id": f"GAU{index:03d}",
        "source_action_unit_id": f"AU{index:03d}",
        "source_event_id": index,
        "source_generation_unit_indexes": [index],
        "source_micro_action_indexes": [index],
        "ledger_indexes": [index - 1],
        "actions": [action],
        "performers": ["actor-alpha"],
        "targets": ["actor-beta"],
    }


def test_phase2_maps_instance_name_and_source_mention_to_one_actor_role() -> None:
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 4,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            {
                **_unit(1, "澜璃快速向右闪避"),
                "performers": ["澜璃"],
                "targets": ["蓝色短棍"],
            }
        ],
        "character_ids": ["lead_I01"],
    }
    shot = {
        "id": "S01",
        "character_ids": ["lead_I01"],
        "who": ["Lan Li"],
        "participant_refs": [
            {
                "mention": "澜璃",
                "character_id": "lead_I01",
                "instance_id": "lead_I01",
            }
        ],
    }
    characters = [
        {
            "id": "lead_I01",
            "instance_id": "lead_I01",
            "name": "Lan Li",
            "aliases": ["L. Li"],
            "source_mentions": ["澜璃"],
        }
    ]

    aliases = storyboard_owner._actor_role_aliases(shot, beat, characters)
    review_grid = storyboard_owner._narrative_grid_contract(
        shot,
        "S01",
        [beat],
        characters,
    )
    plan = build_pose_atlas_plan(beat, actor_role_aliases=aliases)

    assert aliases == {
        "lead_I01": "lead_I01",
        "Lan Li": "lead_I01",
        "L. Li": "lead_I01",
        "澜璃": "lead_I01",
    }
    assert all(
        sample["pose_contract"]["actor_roles"] == ["lead_I01"] for sample in plan["pose_samples"]
    )
    assert all(cell["pose_contract"]["actor_roles"] == ["lead_I01"] for cell in review_grid)
    assert all(sample["pose_contract"]["geometry"]["actors"] for sample in plan["pose_samples"])


def test_phase2_rejects_ambiguous_source_mention_actor_mapping() -> None:
    beat = {"beat_id": "S01_P01", "character_ids": ["lead_I01", "lead_I02"]}
    shot = {
        "id": "S01",
        "character_ids": ["lead_I01", "lead_I02"],
        "participant_refs": [
            {"mention": "队员", "instance_id": "lead_I01"},
            {"mention": "队员", "instance_id": "lead_I02"},
        ],
    }

    with pytest.raises(ValueError, match="ambiguous canonical actor alias"):
        storyboard_owner._actor_role_aliases(shot, beat, [])


def test_canonical_performer_changes_actual_body_raster_not_only_arrows() -> None:
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 4,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            {
                **_unit(1, "澜璃大幅向右闪避并后仰"),
                "performers": ["澜璃"],
                "targets": [],
            }
        ],
        "character_ids": ["lead_I01"],
    }
    plan = build_pose_atlas_plan(
        beat,
        actor_role_aliases={"lead_I01": "lead_I01", "澜璃": "lead_I01"},
    )
    samples = plan["pose_samples"]

    def body_crop(sample: dict) -> Image.Image:
        rendered = pose_owner.render_pose_cell(
            {
                "label": sample["cell_id"],
                "secondary_beat_id": beat["beat_id"],
                "pose_contract": sample["pose_contract"],
            },
            font_factory=lambda _size: ImageFont.load_default(),
        )
        # Excludes the label and both annotation-arrow lanes; only actor pixels remain.
        return rendered.crop((150, 82, 305, 188))

    first = body_crop(samples[0])
    last = body_crop(samples[-1])
    difference = ImageChops.difference(first, last)

    assert samples[0]["pose_contract"]["actor_roles"] == ["lead_I01"]
    assert samples[-1]["pose_contract"]["actor_roles"] == ["lead_I01"]
    assert difference.getbbox() is not None
    assert sum(difference.convert("L").histogram()[1:]) >= 80


@pytest.mark.parametrize(
    ("duration_s", "pose_samples", "action_groups"),
    ((4, 9, 6), (7, 18, 10), (10, 27, 15), (15, 36, 20)),
)
def test_seedance_pose_atlas_capacity_is_duration_bound(
    duration_s: float,
    pose_samples: int,
    action_groups: int,
) -> None:
    capacity = SEEDANCE_2_CAPABILITIES.storyboard_pose_capacity(duration_s)
    assert capacity == {
        "pose_sample_count": pose_samples,
        "reliable_action_group_limit": action_groups,
        "atlas_page_cell_count": 9,
        "single_atlas_max_cells": 36,
        "single_atlas_high_fidelity_group_limit": 6,
    }


@pytest.mark.parametrize(
    "change",
    (
        {"pose_sample_steps": ((7, 18), (4, 9))},
        {"atlas_page_cell_count": 0},
        {"single_atlas_max_cells": 10},
        {"terminal_hold_ratio": 1.0},
        {"terminal_hold_min_s": 2.0, "terminal_hold_max_s": 1.0},
    ),
)
def test_pose_atlas_capability_rejects_invalid_profiles(change) -> None:
    with pytest.raises(ValueError):
        replace(SEEDANCE_2_CAPABILITIES, **change)


def test_seven_second_timing_contract_excludes_anchor_and_allows_terminal_hold() -> None:
    timing = SEEDANCE_2_CAPABILITIES.storyboard_timing_contract(
        7,
        has_initial_anchor=True,
    )

    assert timing["schema"] == "honcut.storyboard-action-timing.v1"
    assert timing["initial_anchor"] == {
        "present": True,
        "story_time_s": 0.0,
        "execution": "start_immediately_after_reference_frame",
    }
    assert timing["story_action"]["target_completion_s"] == 5.95
    assert timing["story_action"]["completion_window_s"] == [5.5, 6.2]
    assert timing["terminal_hold"] == {
        "mode": "semantic_hold",
        "target_duration_s": 1.05,
        "allowed_duration_s": [0.8, 1.5],
    }


def test_pose_plan_separates_action_groups_from_pose_samples() -> None:
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            _unit(1, "actor-alpha advances"),
            _unit(2, "actor-alpha blocks actor-beta"),
            _unit(3, "actor-alpha rotates and controls actor-beta"),
        ],
        "character_ids": ["actor-alpha", "actor-beta"],
    }

    first = build_pose_atlas_plan(beat, known_actor_roles=("actor-alpha", "actor-beta"))
    second = build_pose_atlas_plan(beat, known_actor_roles=("actor-alpha", "actor-beta"))

    assert first == second
    assert first["schema"] == "honcut.storyboard-pose-atlas-plan.v1"
    assert len(first["action_groups"]) == 3
    assert len(first["pose_samples"]) == 18
    assert first["pose_samples"][0]["timing_role"] == "initial_anchor"
    assert first["pose_samples"][0]["story_time_weight"] == 0.0
    assert first["pose_samples"][-1]["timing_role"] == "terminal_hold"
    assert (
        first["pose_samples"][-1]["action_group_id"]
        == first["action_groups"][-1]["action_group_id"]
    )
    assert [sample["sample_index"] for sample in first["pose_samples"]] == list(range(1, 19))
    assert len({sample["pose_fingerprint"] for sample in first["pose_samples"]}) > 9
    assert first["plan_sha256"] == second["plan_sha256"]


def test_pose_plan_rejects_action_groups_above_capability() -> None:
    beat = {
        "beat_id": "S01_P02",
        "duration_s": 4,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            _unit(index, f"actor-alpha performs distinct movement {index}") for index in range(1, 8)
        ],
        "character_ids": ["actor-alpha"],
    }

    with pytest.raises(ValueError, match="reliable action-group limit"):
        build_pose_atlas_plan(beat, known_actor_roles=("actor-alpha",))


def test_phase6_candidate_selection_prefers_pages_then_dense_atlas() -> None:
    candidates = [
        {
            "strategy": "single_atlas",
            "page_count": 1,
            "preferred": False,
            "pages": [{"image": "dense.png"}],
        },
        {
            "strategy": "paged_atlas",
            "page_count": 2,
            "preferred": True,
            "pages": [{"image": "page-01.png"}, {"image": "page-02.png"}],
        },
    ]

    assert (
        select_pose_atlas_candidate(candidates, available_image_slots=2)["strategy"]
        == "paged_atlas"
    )
    assert (
        select_pose_atlas_candidate(candidates, available_image_slots=1)["strategy"]
        == "single_atlas"
    )
    with pytest.raises(ValueError, match="no image slot"):
        select_pose_atlas_candidate(candidates, available_image_slots=0)


def test_low_density_plan_prefers_single_atlas_even_when_pages_fit() -> None:
    plan = build_pose_atlas_plan(
        {
            "beat_id": "S01_P01",
            "duration_s": 7,
            "planner_version": "honcut.secondary-storyboard.v16",
            "generation_action_units": [_unit(1, "actor-alpha advances")],
            "character_ids": ["actor-alpha"],
        }
    )

    assert (
        select_pose_atlas_candidate(
            plan["atlas_candidates"],
            available_image_slots=2,
        )["strategy"]
        == "single_atlas"
    )


def test_camera_contract_rejects_infeasible_pan_without_mutating_contract() -> None:
    shot = {
        "camera_movement": "pan_left",
        "camera_motion_parameters": {
            "pan_degrees": -90.0,
            "pan_speed_degrees_per_s": 10.0,
        },
    }
    contract = build_camera_motion_contract(shot, has_human=False)
    snapshot = dict(contract)

    assert camera_motion_minimum_duration_s(contract) == 9.0
    with pytest.raises(ValueError, match="requires at least 9"):
        validate_camera_motion_duration(contract, 7, resource_id="S01_P01")
    assert contract == snapshot


def test_adaptation_persists_camera_hash_and_phase2_samples_same_path() -> None:
    beat = {
        "beat_id": "S04_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [_unit(1, "actor-alpha advances")],
        "character_ids": ["actor-alpha"],
        "camera_movement": "pan_left",
        "camera_motion_parameters": {
            "pan_degrees": -150.0,
            "pan_speed_degrees_per_s": 30.0,
        },
    }
    apply_camera_motion_contract(beat)
    authored_hash = beat["camera_motion_contract_sha256"]
    plan = build_pose_atlas_plan(beat, known_actor_roles=("actor-alpha",))

    assert plan["camera_motion_contract_sha256"] == authored_hash
    assert plan["camera_motion_contract"] == beat["camera_motion_contract"]
    assert plan["pose_samples"][0]["camera_projection"]["view"] == "front"
    assert plan["pose_samples"][-1]["camera_projection"]["view"] == ("rear_three_quarter")
    first_joints = plan["pose_samples"][0]["pose_contract"]["geometry"]["actors"][0]["joints"]
    last_joints = plan["pose_samples"][-1]["pose_contract"]["geometry"]["actors"][0]["joints"]
    assert first_joints != last_joints


def test_phase2_renders_deterministic_single_and_paged_atlas_candidates(tmp_path) -> None:
    beat = {
        "beat_id": "S03_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            _unit(index, f"actor-alpha performs movement {index}") for index in range(1, 8)
        ],
        "character_ids": ["actor-alpha", "actor-beta"],
        "camera_movement": "orbital",
    }
    plan = build_pose_atlas_plan(
        beat,
        known_actor_roles=("actor-alpha", "actor-beta"),
    )

    first = render_pose_atlas_candidates(
        tmp_path,
        plan,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    recovery_receipts = [
        render_pose_atlas_candidates(
            tmp_path,
            plan,
            font_factory=lambda _size: ImageFont.load_default(),
        )
        for _ in range(10)
    ]

    assert all(
        receipt["receipt_sha256"] == first["receipt_sha256"] for receipt in recovery_receipts
    )
    assert [candidate["strategy"] for candidate in first["candidates"]] == [
        "single_atlas",
        "paged_atlas",
    ]
    assert [candidate["page_count"] for candidate in first["candidates"]] == [1, 2]
    for candidate in first["candidates"]:
        covered = [sample_id for page in candidate["pages"] for sample_id in page["sample_ids"]]
        assert covered == [f"G{index:02d}" for index in range(1, 19)]
        for page in candidate["pages"]:
            with Image.open(tmp_path / page["image"]) as image:
                assert image.width <= 6000
                assert image.height <= 6000
                assert 0.4 <= image.width / image.height <= 2.5


def test_generation_chunk_rejects_tampered_atlas_coverage(tmp_path) -> None:
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 4,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [_unit(1, "actor-alpha blocks actor-beta")],
        "character_ids": ["actor-alpha", "actor-beta"],
    }
    plan = build_pose_atlas_plan(beat)
    receipt = render_pose_atlas_candidates(
        tmp_path,
        plan,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    candidates = receipt["candidates"]
    candidates[0]["pages"][0]["sample_ids"] = candidates[0]["pages"][0]["sample_ids"][:-1]

    with pytest.raises(ValueError, match="cover every sample"):
        GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=4,
            mode="fresh",
            storyboard_beat_id="S01_P01",
            storyboard_pose_atlas_plan_schema=plan["schema"],
            storyboard_pose_atlas_plan_sha256=plan["plan_sha256"],
            storyboard_pose_atlas_timing_contract=plan["timing_contract"],
            storyboard_pose_atlas_camera_motion_contract_sha256=plan[
                "camera_motion_contract_sha256"
            ],
            storyboard_pose_atlas_action_groups=plan["action_groups"],
            storyboard_pose_atlas_pose_samples=plan["pose_samples"],
            storyboard_pose_atlas_candidates=candidates,
            storyboard_pose_atlas_receipt=receipt["receipt"],
            storyboard_pose_atlas_receipt_sha256=receipt["receipt_sha256"],
        )


def test_phase6_binds_two_page_atlas_after_authoritative_media(tmp_path) -> None:
    beat = {
        "beat_id": "S01_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [
            _unit(index, f"actor-alpha performs movement {index}") for index in range(1, 8)
        ],
        "character_ids": ["actor-alpha", "actor-beta"],
    }
    plan = build_pose_atlas_plan(beat)
    receipt = render_pose_atlas_candidates(
        tmp_path,
        plan,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    paged = next(
        candidate for candidate in receipt["candidates"] if candidate["strategy"] == "paged_atlas"
    )
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=7,
        mode="fresh",
        storyboard_beat_id="S01_P01",
        storyboard_pose_atlas_plan_schema=plan["schema"],
        storyboard_pose_atlas_plan_sha256=plan["plan_sha256"],
        storyboard_pose_atlas_timing_contract=plan["timing_contract"],
        storyboard_pose_atlas_camera_motion_contract_sha256=plan["camera_motion_contract_sha256"],
        storyboard_pose_atlas_action_groups=plan["action_groups"],
        storyboard_pose_atlas_pose_samples=plan["pose_samples"],
        storyboard_pose_atlas_candidates=receipt["candidates"],
        storyboard_pose_atlas_receipt=receipt["receipt"],
        storyboard_pose_atlas_receipt_sha256=receipt["receipt_sha256"],
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=chunk,
        anchors={},
        output_path=tmp_path / "S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint="atlas-media-order",
        memory_context="",
    )
    atlas_media = [
        {
            "type": "image_url",
            "image_url": {"url": f"https://example.invalid/page-{index}.png"},
            "role": "reference_image",
            "_reference_kind": "storyboard_pose_atlas",
            "_reference_description": "current pose atlas",
            "_narrative_beat_id": "S01_P01",
            "_narrative_cell_ids": list(page["sample_ids"]),
            "_narrative_zero_time_anchor_cell_ids": ["G01"] if index == 1 else [],
            "_pose_atlas_strategy": "paged_atlas",
            "_pose_atlas_page_index": index,
            "_pose_atlas_page_count": 2,
            "_pose_atlas_plan_sha256": plan["plan_sha256"],
            "_pose_atlas_timing_contract": plan["timing_contract"],
            "_pose_atlas_camera_motion_contract_sha256": plan["camera_motion_contract_sha256"],
            "_mandatory_reference": True,
        }
        for index, page in enumerate(paged["pages"], 1)
    ]

    content, manifest = _bind_final_media_index_prompt(
        [
            {"type": "text", "text": "execute current action"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/identity.png"},
                "role": "reference_image",
                "_reference_kind": "character_identity_board",
                "_reference_description": "identity board",
                "_mandatory_reference": True,
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/frame.png"},
                "role": "reference_image",
                "_reference_kind": "cinematic_composition",
                "_reference_description": "first frame",
                "_mandatory_reference": True,
            },
            *atlas_media,
        ],
        request,
    )

    assert [item["responsibility"] for item in manifest] == [
        "character_identity_board",
        "cinematic_composition",
        "storyboard_pose_atlas",
        "storyboard_pose_atlas",
    ]
    assert "当前动作姿态图集是图片3、图片4" in content[0]["text"]
    assert "5.5～6.2秒完成" in content[0]["text"]
    assert "首帧后立即从G02开始运动" in content[0]["text"]


def test_exact_terminal_reference_is_hashed_and_prompt_bound(tmp_path) -> None:
    terminal_path = tmp_path / "terminal.png"
    terminal_path.write_bytes(b"exact-terminal-pose")
    terminal_sha256 = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    beat = {
        "beat_id": "S05_P01",
        "duration_s": 7,
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": [_unit(1, "actor-alpha blocks actor-beta")],
        "character_ids": ["actor-alpha", "actor-beta"],
        "terminal_reference_mode": "exact_pose",
    }
    plan = build_pose_atlas_plan(beat)
    receipt = render_pose_atlas_candidates(
        tmp_path,
        plan,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    selected = next(candidate for candidate in receipt["candidates"] if candidate["preferred"])
    chunk = GenerationChunk(
        chunk_id="S05_C01",
        sequence=1,
        target_duration_s=7,
        mode="fresh",
        storyboard_beat_id="S05_P01",
        storyboard_pose_atlas_plan_schema=plan["schema"],
        storyboard_pose_atlas_plan_sha256=plan["plan_sha256"],
        storyboard_pose_atlas_timing_contract=plan["timing_contract"],
        storyboard_pose_atlas_camera_motion_contract_sha256=plan["camera_motion_contract_sha256"],
        storyboard_pose_atlas_action_groups=plan["action_groups"],
        storyboard_pose_atlas_pose_samples=plan["pose_samples"],
        storyboard_pose_atlas_candidates=receipt["candidates"],
        storyboard_pose_atlas_receipt=receipt["receipt"],
        storyboard_pose_atlas_receipt_sha256=receipt["receipt_sha256"],
        terminal_reference_mode="exact_pose",
        terminal_pose_reference=str(terminal_path),
        terminal_pose_reference_sha256=terminal_sha256,
    )
    request = ChunkExecutionRequest(
        resource_id="S05_C01",
        shot_id="S05",
        chunk=chunk,
        anchors={},
        output_path=tmp_path / "S05_C01.mp4",
        previous_output_path=None,
        input_fingerprint="exact-terminal",
        memory_context="",
    )
    atlas_page = selected["pages"][0]
    media = [
        {"type": "text", "text": "execute current action"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/identity.png"},
            "role": "reference_image",
            "_reference_kind": "character_identity_board",
            "_reference_description": "identity board",
            "_mandatory_reference": True,
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/frame.png"},
            "role": "reference_image",
            "_reference_kind": "cinematic_composition",
            "_reference_description": "first frame",
            "_mandatory_reference": True,
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/terminal.png"},
            "role": "reference_image",
            "_reference_kind": "terminal_pose_reference",
            "_reference_description": "terminal pose",
            "_reference_sha256": terminal_sha256,
            "_mandatory_reference": True,
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/atlas.png"},
            "role": "reference_image",
            "_reference_kind": "storyboard_pose_atlas",
            "_reference_description": "current pose atlas",
            "_narrative_beat_id": "S05_P01",
            "_narrative_cell_ids": list(atlas_page["sample_ids"]),
            "_narrative_zero_time_anchor_cell_ids": ["G01"],
            "_pose_atlas_strategy": selected["strategy"],
            "_pose_atlas_page_index": 1,
            "_pose_atlas_page_count": 1,
            "_pose_atlas_plan_sha256": plan["plan_sha256"],
            "_pose_atlas_timing_contract": plan["timing_contract"],
            "_pose_atlas_camera_motion_contract_sha256": plan["camera_motion_contract_sha256"],
            "_mandatory_reference": True,
        },
    ]

    content, manifest = _bind_final_media_index_prompt(media, request)

    assert [item["responsibility"] for item in manifest] == [
        "character_identity_board",
        "cinematic_composition",
        "terminal_pose_reference",
        "storyboard_pose_atlas",
    ]
    assert "图片3只约束当前 Pxx 的精确结束姿态" in content[0]["text"]
    with pytest.raises(ValueError, match="exact terminal pose evidence"):
        _bind_final_media_index_prompt(
            [item for item in media if item.get("_reference_kind") != "terminal_pose_reference"],
            request,
        )
