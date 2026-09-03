from __future__ import annotations

import copy
import hashlib
import inspect
from io import BytesIO

import pytest
from PIL import ImageFont

from phases.phase2 import storyboard_guide_pose as pose_owner
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import _task_payload
from schemas.continuity import GenerationChunk


def _cell(label: str = "S01_G01", stage: str = "action_progress") -> dict:
    return {
        "cell": int(label.rsplit("G", 1)[-1]),
        "label": label,
        "primary_shot_id": "S01",
        "secondary_beat_id": "S01_P01",
        "stage": stage,
        "camera_movement": "camera moves right",
    }


def _unit(
    unit_id: str,
    action: str,
    *,
    performers: tuple[str, ...] = ("actor-alpha",),
    targets: tuple[str, ...] = (),
    ledger_index: int = 0,
) -> dict:
    return {
        "unit_id": unit_id,
        "actions": [action],
        "performers": list(performers),
        "targets": list(targets),
        "ledger_indexes": [ledger_index],
        "source_event_id": ledger_index + 1,
        "source_generation_unit_indexes": [ledger_index + 1],
    }


def _beat(*units: dict, character_ids: tuple[str, ...] = ()) -> dict:
    return {
        "beat_id": "S01_P01",
        "planner_version": "honcut.secondary-storyboard.v16",
        "generation_action_units": list(units),
        "character_ids": list(character_ids),
        "action": " then ".join(unit["actions"][0] for unit in units),
    }


@pytest.fixture
def anonymized_action_catalog() -> tuple[dict, ...]:
    return (
        _unit(
            "GAU001",
            "automatic door moves left while light shifts",
            performers=("automatic door",),
            targets=("light",),
            ledger_index=0,
        ),
        _unit(
            "GAU002",
            "actor-alpha appears as three guards emerge",
            performers=("actor-alpha", "guard-1", "guard-2", "guard-3"),
            ledger_index=1,
        ),
        _unit("GAU003", "actor-alpha runs forward", ledger_index=2),
        _unit("GAU004", "actor-alpha dodges backward", ledger_index=3),
        _unit(
            "GAU005",
            "actor-alpha blocks guard-1",
            targets=("guard-1",),
            ledger_index=4,
        ),
        _unit(
            "GAU006",
            "actor-alpha strikes guard-1",
            targets=("guard-1",),
            ledger_index=5,
        ),
        _unit(
            "GAU007",
            "actor-alpha grabs guard-1 and throws left",
            targets=("guard-1",),
            ledger_index=6,
        ),
        _unit(
            "GAU008",
            "actor-alpha activates handheld device",
            targets=("handheld device",),
            ledger_index=7,
        ),
    )


def test_anonymized_catalog_covers_required_motion_families(
    anonymized_action_catalog,
):
    observed = []
    for unit in anonymized_action_catalog:
        contract = pose_owner.compile_pose_contracts(_beat(unit), [_cell()])[0]["pose_contract"]
        observed.append(contract["pose_family"])
        assert contract["action_bindings"][0]["unit_id"] == unit["unit_id"]
        assert contract["lineage_status"] == "canonical"
    assert observed == [
        "locomotion",
        "reveal",
        "locomotion",
        "evade",
        "block",
        "strike",
        "throw",
        "prop_use",
    ]


def test_pose_progress_changes_joint_geometry_and_fingerprint():
    unit = _unit("GAU001", "actor-alpha kicks right")
    cells = pose_owner.compile_pose_contracts(
        _beat(unit),
        [
            _cell("S01_G01", "start"),
            _cell("S01_G02", "action_progress"),
            _cell("S01_G03", "end"),
        ],
    )
    fingerprints = pose_owner.pose_fingerprints(cells)
    assert len(set(fingerprints)) == 3
    feet = [cell["pose_contract"]["geometry"]["actors"][0]["joints"]["left_foot"] for cell in cells]
    assert len({tuple(point) for point in feet}) == 3
    pose_owner.validate_pose_sequence(cells, beat_id="S01_P01")


def test_action_units_partition_in_source_order_without_loss():
    units = (
        _unit("GAU001", "actor-alpha runs forward", ledger_index=0),
        _unit("GAU002", "actor-alpha blocks right", ledger_index=1),
        _unit("GAU003", "actor-alpha strikes right", ledger_index=2),
    )
    cells = pose_owner.compile_pose_contracts(
        _beat(*units),
        [_cell("S01_G01", "start"), _cell("S01_G02", "end")],
    )
    ordered_ids = [
        binding["unit_id"]
        for cell in cells
        for binding in cell["pose_contract"]["action_bindings"]
    ]
    assert ordered_ids == ["GAU001", "GAU002", "GAU003"]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("actor-alpha dodges left", "evade"),
        ("actor-alpha blocks forward", "block"),
        ("actor-alpha strikes right", "strike"),
        ("actor-alpha kicks left", "kick"),
        ("actor-alpha grabs guard-1", "grab_control"),
        ("actor-alpha throws guard-1 right", "throw"),
        ("actor-alpha uses handheld device", "prop_use"),
        ("actor-alpha makes an unfamiliar transition", "spatial"),
    ],
)
def test_multilingual_fallback_is_controlled_and_auditable(action, expected):
    targets = ("guard-1",) if "guard" in action else ()
    contract = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", action, targets=targets)), [_cell()]
    )[0]["pose_contract"]
    assert contract["pose_family"] == expected
    assert contract["classification_evidence"]
    assert contract["pose_policy_sha256"] == pose_owner.POSE_POLICY_SHA256


def test_multi_subject_interaction_uses_distinct_role_slots():
    cell = pose_owner.compile_pose_contracts(
        _beat(
            _unit(
                "GAU001",
                "actor-alpha strikes guard-1 right",
                targets=("guard-1",),
            )
        ),
        [_cell()],
    )[0]
    contract = cell["pose_contract"]
    assert contract["actor_roles"] == ["actor-alpha", "guard-1"]
    assert [actor["slot"] for actor in contract["geometry"]["actors"]] == [1, 2]
    assert contract["geometry"]["actors"][0]["pose_family"] == "strike"
    assert contract["geometry"]["actors"][1]["pose_family"] == "evade"
    assert all(
        len(actor["pose_fingerprint"]) == 64
        for actor in contract["geometry"]["actors"]
    )
    assert (
        contract["geometry"]["actors"][0]["pose_fingerprint"]
        != contract["geometry"]["actors"][1]["pose_fingerprint"]
    )


def test_environment_only_motion_uses_object_glyph_without_fabricated_actor():
    cell = pose_owner.compile_pose_contracts(
        _beat(
            _unit(
                "GAU001",
                "automatic door moves left",
                performers=("automatic door",),
            ),
            character_ids=("actor-not-performing",),
        ),
        [_cell()],
    )[0]
    contract = cell["pose_contract"]
    assert contract["actor_roles"] == []
    assert contract["geometry"]["actors"] == []
    assert contract["object_roles"] == ["automatic door"]
    assert contract["geometry"]["objects"]


def test_action_and_camera_arrows_follow_resolved_vectors(monkeypatch):
    rendered_arrows = []
    original = pose_owner._draw_arrow

    def record(draw, start, end, *, fill, width):
        rendered_arrows.append((start, end, fill))
        return original(draw, start, end, fill=fill, width=width)

    monkeypatch.setattr(pose_owner, "_draw_arrow", record)
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha runs left")), [_cell()]
    )[0]
    pose_owner.render_pose_cell(
        cell,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    action_arrow = next(item for item in rendered_arrows if item[2] == (205, 48, 54))
    camera_arrow = next(item for item in rendered_arrows if item[2] == (42, 104, 190))
    assert action_arrow[1][0] < action_arrow[0][0]
    assert camera_arrow[1][0] > camera_arrow[0][0]


def test_identical_inputs_render_identical_bytes():
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha blocks right")), [_cell()]
    )[0]

    def png_sha256() -> str:
        image = pose_owner.render_pose_cell(
            copy.deepcopy(cell),
            font_factory=lambda _size: ImageFont.load_default(),
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    assert png_sha256() == png_sha256()


def test_tampered_geometry_and_missing_lineage_fail_closed():
    cell = pose_owner.compile_pose_contracts(
        _beat(_unit("GAU001", "actor-alpha blocks right")), [_cell()]
    )[0]
    tampered = copy.deepcopy(cell)
    tampered["pose_contract"]["geometry"]["actors"][0]["joints"]["head"] = [0, 0]
    with pytest.raises(ValueError, match="contract hash mismatch"):
        pose_owner.validate_pose_sequence([tampered], beat_id="S01_P01")

    no_lineage = _unit("GAU001", "actor-alpha blocks right")
    no_lineage.pop("ledger_indexes")
    no_lineage.pop("source_event_id")
    no_lineage.pop("source_generation_unit_indexes")
    with pytest.raises(ValueError, match="no source action/event lineage"):
        pose_owner.compile_pose_contracts(_beat(no_lineage), [_cell()])


def test_current_planner_without_action_units_fails_closed():
    beat = {
        "beat_id": "S01_P01",
        "planner_version": "honcut.secondary-storyboard.v16",
        "action": "actor-alpha moves",
    }
    with pytest.raises(ValueError, match="missing generation action units"):
        pose_owner.compile_pose_contracts(beat, [_cell()])


def test_pose_owner_has_no_provider_phase3_or_pipeline_core_dependency():
    source = inspect.getsource(pose_owner)
    assert "clients." not in source
    assert "phase3" not in source.casefold()
    assert "pipeline_core" not in source


def test_phase6_task_fingerprint_binds_ordered_pose_fingerprints(tmp_path):
    guide_fields = {
        "storyboard_beat_id": "S01_P01",
        "storyboard_narrative_guide": "storyboard_guides/S01_P01.png",
        "storyboard_narrative_guide_kind": "honcut.storyboard-narrative-guide.v3",
        "storyboard_narrative_guide_usage": ("phase6_story_narrative_guide_not_output_pixels"),
        "storyboard_narrative_guide_cell_ids": ["S01_G01", "S01_G02"],
        "storyboard_narrative_guide_sha256": "a" * 64,
        "storyboard_narrative_guide_renderer": ("honcut.identity-neutral-story-guide-renderer.v2"),
        "storyboard_narrative_guide_pose_contract_schema": (
            "honcut.storyboard-guide-pose-contract.v1"
        ),
        "storyboard_narrative_guide_pose_policy_sha256": "b" * 64,
        "storyboard_narrative_guide_pose_contracts_sha256": "c" * 64,
        "storyboard_narrative_guide_pose_fingerprints": ["d" * 64, "e" * 64],
        "storyboard_narrative_guide_source_pixel_usage": "none",
        "storyboard_narrative_guide_semantic_payload_sha256": "f" * 64,
        "storyboard_narrative_guide_source_board": "shot_storyboards/S01.png",
        "storyboard_narrative_guide_source_board_sha256": "1" * 64,
        "storyboard_narrative_guide_receipt": "storyboard_guides/S01_P01.json",
        "storyboard_narrative_guide_authority_roles": [
            "narrative_order",
            "action_direction",
            "camera_motion",
            "spatial_relationship",
        ],
        "storyboard_narrative_guide_non_authority_roles": ["character_identity"],
    }

    def payload(fingerprints: list[str]) -> dict:
        chunk = GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=5,
            mode="fresh",
            execution_strategy="multi_image",
            action_prompt="actor-alpha moves",
            **{
                **guide_fields,
                "storyboard_narrative_guide_pose_fingerprints": fingerprints,
            },
        )
        request = ChunkExecutionRequest(
            resource_id="S01_C01",
            shot_id="S01",
            chunk=chunk,
            anchors={},
            output_path=tmp_path / "S01_C01.mp4",
            previous_output_path=None,
            input_fingerprint="2" * 64,
            memory_context="",
        )
        return _task_payload(
            request,
            model="test-model",
            provider_id="test-provider",
            provider_version="1",
            project_id="project",
            run_id="run",
            duration=5,
            seed=7,
        )

    first = payload(["d" * 64, "e" * 64])
    second = payload(["d" * 64, "9" * 64])
    assert first["storyboard_narrative_guide_pose_fingerprints"] == [
        "d" * 64,
        "e" * 64,
    ]
    assert first["input_fingerprint"] != second["input_fingerprint"]


def test_continuity_chunk_rejects_incomplete_pose_fingerprint_binding():
    with pytest.raises(ValueError, match="pose fingerprints must bind every"):
        GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=5,
            mode="fresh",
            execution_strategy="multi_image",
            storyboard_beat_id="S01_P01",
            storyboard_narrative_guide="storyboard_guides/S01_P01.png",
            storyboard_narrative_guide_kind="honcut.storyboard-narrative-guide.v3",
            storyboard_narrative_guide_usage=("phase6_story_narrative_guide_not_output_pixels"),
            storyboard_narrative_guide_cell_ids=["S01_G01", "S01_G02"],
            storyboard_narrative_guide_sha256="a" * 64,
            storyboard_narrative_guide_renderer=("honcut.identity-neutral-story-guide-renderer.v2"),
            storyboard_narrative_guide_pose_contract_schema=(
                "honcut.storyboard-guide-pose-contract.v1"
            ),
            storyboard_narrative_guide_pose_policy_sha256="b" * 64,
            storyboard_narrative_guide_pose_contracts_sha256="c" * 64,
            storyboard_narrative_guide_pose_fingerprints=["d" * 64],
            storyboard_narrative_guide_source_pixel_usage="none",
            storyboard_narrative_guide_semantic_payload_sha256="e" * 64,
            storyboard_narrative_guide_source_board="shot_storyboards/S01.png",
            storyboard_narrative_guide_source_board_sha256="f" * 64,
            storyboard_narrative_guide_receipt="storyboard_guides/S01_P01.json",
            storyboard_narrative_guide_authority_roles=[
                "narrative_order",
                "action_direction",
                "camera_motion",
                "spatial_relationship",
            ],
            storyboard_narrative_guide_non_authority_roles=["character_identity"],
        )
