from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from acceptance.motion_blueprint import (
    CameraTrack,
    MotionBlueprintInput,
    MotionBlueprintManifest,
    MotionEvent,
    SourceLineage,
    assess_legacy_blueprint_manifest,
    combination_eligibility,
    compile_canonical_motion_blueprint,
    compile_motion_blueprint,
    compile_semantic_frames,
    inspect_identity_neutral_pixels,
    measure_output_motion,
    measure_rendered_blueprint_motion,
    measure_semantic_frames,
    technique_registry_sha256,
)
from utils.canonical_visual_contracts import build_canonical_visual_contract
from utils.action_kinematics import (
    apply_generation_kinematics_projection,
    compile_source_kinematics,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "seedance_motion_blueprint_acceptance.py"
SPEC = importlib.util.spec_from_file_location("seedance_motion_blueprint_acceptance", SCRIPT_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage() -> SourceLineage:
    digest = "a" * 64
    return SourceLineage(
        canonical_visual_contract_path="CANONICAL_VISUAL_CONTRACT.json",
        canonical_visual_contract_sha256=digest,
        continuity_plan_path="CONTINUITY_PLAN.json",
        continuity_plan_sha256=digest,
        source_receipt_path="source.json",
        source_receipt_sha256=digest,
    )


def _contract(*primitives: str) -> MotionBlueprintInput:
    return MotionBlueprintInput(
        beat_id="S01_P01",
        duration_s=4.0,
        actor_ids=("actor_01",),
        events=tuple(
            MotionEvent(
                event_id=f"S01_P01_M{index:02d}",
                order=index,
                actor_ids=("actor_01",),
                primitive=primitive,
                direction="right",
                source_action_group_id=f"S01_P01_A{index:02d}",
                source_action_unit_ids=(f"AU{index:03d}",),
                prop_contact=primitive in {"strike", "prop_use"},
            )
            for index, primitive in enumerate(primitives, 1)
        ),
        camera=CameraTrack(primitive="push_in", magnitude=0.3),
        lineage=_lineage(),
    )


def test_compiler_is_deterministic_and_seedance_compatible(tmp_path: Path) -> None:
    contract = _contract("locomotion", "strike", "evade", "block")
    first = compile_motion_blueprint(contract, tmp_path / "first.mp4")
    second = compile_motion_blueprint(contract, tmp_path / "second.mp4")
    assert first["semantic_frames_sha256"] == second["semantic_frames_sha256"]
    assert first["media_sha256"] == second["media_sha256"]
    assert first["technical_probe"]["streams"][0]["codec_name"] == "h264"
    assert first["measurements"]["fps"] == 24
    assert first["measurements"]["duration_s"] == 4.0
    assert first["identity_authority"] is False
    assert (
        inspect_identity_neutral_pixels(tmp_path / "first.mp4")["forbidden_annotation_pixels"] == 0
    )
    assert measure_output_motion(tmp_path / "first.mp4")["deterministic_motion_pass"] is True


def test_compiler_records_order_large_motion_contact_and_camera() -> None:
    contract = _contract("locomotion", "kick", "prop_use", "evade")
    frames, intervals = compile_semantic_frames(contract)
    result = measure_semantic_frames(contract, frames, intervals)
    assert result["ordered_event_ids"] == [event.event_id for event in contract.events]
    assert result["action_onset_s"] <= 0.5
    assert result["terminal_hold_fraction"] <= 0.15
    dynamic = [event for event in result["events"] if event["admission_role"] == "dynamic_action"]
    assert all(event["passes_kinetics"] for event in dynamic)
    assert all(event["major_joint_participants"] >= 4 for event in dynamic)
    assert all(event["apex_fraction"] <= 0.72 for event in dynamic)
    assert any(frame["prop_contact"] for frame in frames)
    assert frames[-1]["camera"]["zoom"] > frames[0]["camera"]["zoom"]


def test_combination_tempo_uses_distinct_contiguous_bounded_actions() -> None:
    contract = _contract("ready", "strike", "evade", "block")
    frames, intervals = compile_semantic_frames(contract)
    result = measure_semantic_frames(contract, frames, intervals)
    combination = result["combination"]
    assert combination["eligible"] is True
    assert combination["tempo_pass"] is True
    assert combination["dynamic_action_count"] == 3
    assert combination["distinct_dynamic_primitive_count"] == 3
    assert combination["dynamic_action_density_per_s"] == 0.75
    assert combination["maximum_dynamic_event_duration_s"] <= 1.25 + (1 / 24)
    assert combination["inter_action_gap_frames"] == [0, 0]
    assert combination["ordered_source_action_group_ids"] == [
        "S01_P01_A02",
        "S01_P01_A03",
        "S01_P01_A04",
    ]


def test_rendered_combination_has_visibly_distinct_successive_techniques(
    tmp_path: Path,
) -> None:
    import cv2
    import numpy as np

    manifest = compile_motion_blueprint(
        _contract("ready", "strike", "evade", "block"),
        tmp_path / "combination.mp4",
    )
    dynamic = [
        event
        for event in manifest["measurements"]["events"]
        if event["admission_role"] == "dynamic_action"
    ]
    capture = cv2.VideoCapture(str(tmp_path / "combination.mp4"))
    masks = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 70)
    capture.release()
    apex_masks = [masks[event["apex_frame"]] for event in dynamic]
    visible_changes = []
    for first, second in itertools.pairwise(apex_masks):
        union = int(np.logical_or(first, second).sum())
        visible_changes.append(float(np.logical_xor(first, second).sum() / max(1, union)))
    assert min(visible_changes) >= 0.65
    assert [event["technique_id"] for event in dynamic] == [
        "linear_hand_strike_v1",
        "lateral_slip_v1",
        "two_arm_high_block_v1",
    ]


def test_single_action_is_explicitly_ineligible_for_combination_gate() -> None:
    eligibility = combination_eligibility(_contract("ready", "evade"))
    assert eligibility["eligible"] is False
    assert eligibility["dynamic_action_count"] == 1
    assert "insufficient_dynamic_actions" in eligibility["reasons"]
    assert "insufficient_action_density" in eligibility["reasons"]
    two_action = combination_eligibility(_contract("ready", "strike", "block"))
    assert two_action["eligible"] is False
    assert two_action["dynamic_action_count"] == 2
    assert two_action["distinct_dynamic_primitive_count"] == 2
    assert two_action["dynamic_action_density_per_s"] == 0.5
    assert "insufficient_dynamic_actions" in two_action["reasons"]
    assert "insufficient_action_density" in two_action["reasons"]
    repeated = combination_eligibility(_contract("strike", "strike", "strike"))
    assert repeated["eligible"] is False
    assert repeated["dynamic_action_count"] == 3
    assert "insufficient_distinct_techniques" in repeated["reasons"]


@pytest.mark.parametrize(
    ("primitive", "expected_phases"),
    [
        ("locomotion", ["push_off", "flight", "lead_plant", "weight_pass", "terminal_stride"]),
        ("evade", ["threat_read", "weight_load", "escape_peak", "counterbalance", "recovery"]),
        ("strike", ["chamber", "drive", "contact_extension", "recoil", "guard_return"]),
        (
            "attack",
            ["weapon_chamber", "body_drive", "attack_contact", "follow_through", "combat_recovery"],
        ),
        (
            "kick",
            ["support_load", "knee_chamber", "foot_contact", "leg_retraction", "landing_guard"],
        ),
        ("block", ["guard_raise", "brace", "contact_absorb", "redirect", "guard_settle"]),
        ("grapple", ["reach", "grip_contact", "base_drop", "control_pull", "control_hold"]),
        ("throw", ["load", "pivot", "release_contact", "follow_through", "balanced_finish"]),
        ("jump", ["crouch_load", "takeoff", "airborne_apex", "landing_absorb", "settle"]),
        ("crouch", ["level_drop", "compression", "lateral_ready", "rise_control", "low_finish"]),
        ("lean_back", ["base_set", "torso_arc", "maximum_lean", "counterbalance", "recenter"]),
        (
            "lean_forward",
            ["rear_foot_load", "torso_hinge", "forward_peak", "base_catch", "recenter"],
        ),
        (
            "prop_use",
            ["acquire_line", "aim", "activation_contact", "follow_through", "safe_finish"],
        ),
        ("spatial", ["orient", "reach", "indicate_apex", "retract", "settle"]),
    ],
)
def test_dynamic_primitives_have_code_owned_technique_phases(
    primitive: str,
    expected_phases: list[str],
) -> None:
    contract = _contract(primitive)
    frames, intervals = compile_semantic_frames(contract)
    measurements = measure_semantic_frames(contract, frames, intervals)
    interval = intervals[0]
    assert interval["technique_phase_ids"] == expected_phases
    assert measurements["events"][0]["observed_technique_phase_ids"] == expected_phases
    assert measurements["events"][0]["technique_phase_order_valid"] is True
    assert len(interval["technique_keyframes_sha256"]) == 64


def _frame_for_phase(
    frames: list[dict[str, Any]],
    phase_id: str,
) -> dict[str, Any]:
    return next(frame for frame in frames if frame["technique_phase_id"] == phase_id)


def test_representative_techniques_have_distinct_biomechanics() -> None:
    fingerprints: set[str] = set()
    for primitive in ("evade", "kick", "block", "strike"):
        frames, intervals = compile_semantic_frames(_contract(primitive))
        fingerprints.add(intervals[0]["technique_keyframes_sha256"])
        if primitive == "evade":
            loaded = _frame_for_phase(frames, "weight_load")
            peak = _frame_for_phase(frames, "escape_peak")
            assert peak["root"][0] - loaded["root"][0] > 0.25
            assert peak["joints"]["head"][0] < loaded["joints"]["head"][0]
        elif primitive == "kick":
            chamber = _frame_for_phase(frames, "knee_chamber")
            contact = _frame_for_phase(frames, "foot_contact")
            assert chamber["joints"]["right_knee"][1] < 0.08
            assert contact["joints"]["right_ankle"][0] > chamber["joints"]["right_ankle"][0] + 0.30
        elif primitive == "block":
            brace = _frame_for_phase(frames, "brace")
            contact = _frame_for_phase(frames, "contact_absorb")
            assert brace["joints"]["left_wrist"][1] < -0.20
            assert contact["joints"]["right_wrist"][1] < -0.30
        else:
            chamber = _frame_for_phase(frames, "chamber")
            contact = _frame_for_phase(frames, "contact_extension")
            assert contact["joints"]["left_wrist"][0] > chamber["joints"]["left_wrist"][0] + 0.50
    assert len(fingerprints) == 4


def test_contact_geometry_is_limited_to_declared_technique_phase() -> None:
    contract = _contract("strike")
    frames, intervals = compile_semantic_frames(contract)
    event_frames = [frame for frame in frames if frame["event_id"] == "S01_P01_M01"]
    contact_frames = [frame for frame in event_frames if frame["prop_contact"]]
    assert 0 < len(contact_frames) < len(event_frames)
    assert {frame["technique_phase_id"] for frame in contact_frames} == {"contact_extension"}
    assert intervals[0]["technique_contact_phase_ids"] == ["contact_extension"]


def test_manifest_binds_technique_registry_and_keyframes(tmp_path: Path) -> None:
    manifest = compile_motion_blueprint(_contract("evade"), tmp_path / "motion.mp4")
    assert manifest["technique_registry_sha256"] == technique_registry_sha256()
    event = manifest["measurements"]["events"][0]
    assert event["technique_id"] == "lateral_slip_v1"
    assert event["technique_phase_ids"][2] == "escape_peak"
    assert len(event["technique_keyframes_sha256"]) == 64


def test_rendered_evade_visibly_changes_at_each_technique_phase(
    tmp_path: Path,
) -> None:
    import cv2
    import numpy as np

    contract = _contract("evade")
    semantic_frames, _intervals = compile_semantic_frames(contract)
    video_path = tmp_path / "evade-technique.mp4"
    compile_motion_blueprint(contract, video_path)
    capture = cv2.VideoCapture(str(video_path))
    rendered_masks = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rendered_masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 70)
    capture.release()

    phase_centres = []
    for phase_id in dict.fromkeys(frame["technique_phase_id"] for frame in semantic_frames):
        indexes = [
            index
            for index, frame in enumerate(semantic_frames)
            if frame["technique_phase_id"] == phase_id
        ]
        phase_centres.append((phase_id, indexes[len(indexes) // 2]))
    assert [phase_id for phase_id, _index in phase_centres] == [
        "threat_read",
        "weight_load",
        "escape_peak",
        "counterbalance",
        "recovery",
    ]
    phase_changes = []
    for (_previous_phase, previous_index), (_phase, current_index) in itertools.pairwise(
        phase_centres
    ):
        previous = rendered_masks[previous_index]
        current = rendered_masks[current_index]
        union = int(np.logical_or(previous, current).sum())
        phase_changes.append(float(np.logical_xor(previous, current).sum() / max(1, union)))
    assert min(phase_changes) >= 0.20


def test_setup_pose_cannot_consume_the_dynamic_action_window() -> None:
    contract = _contract("ready", "evade")
    _frames, intervals = compile_semantic_frames(contract)
    setup_frames = intervals[0]["end_frame"] - intervals[0]["start_frame"] + 1
    dynamic_frames = intervals[1]["end_frame"] - intervals[1]["start_frame"] + 1
    assert setup_frames <= int(0.15 * contract.fps)
    assert dynamic_frames > setup_frames * 20

    crowded = _contract("ready", "prop_hold", "evade")
    _frames, crowded_intervals = compile_semantic_frames(crowded)
    total_setup_frames = sum(
        interval["end_frame"] - interval["start_frame"] + 1
        for interval in crowded_intervals
        if interval["admission_role"] == "setup_anchor"
    )
    assert total_setup_frames <= int(0.15 * crowded.fps)


def test_unknown_primitive_fails_before_render(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unsupported motion primitives.*S01_P01_A01"):
        compile_motion_blueprint(_contract("teleport_spin"), tmp_path / "never.mp4")
    assert not (tmp_path / "never.mp4").exists()


def test_zero_or_multiple_actor_scope_fails_closed() -> None:
    payload = _contract("evade").model_dump(mode="json")
    payload["actor_ids"] = []
    payload["events"][0]["actor_ids"] = []
    with pytest.raises(ValueError, match=r"canonical actors|exactly one"):
        MotionBlueprintInput.model_validate(payload)
    payload = _contract("evade").model_dump(mode="json")
    payload["actor_ids"] = ["a", "b"]
    payload["events"][0]["actor_ids"] = ["a", "b"]
    with pytest.raises(ValueError, match="exactly one"):
        MotionBlueprintInput.model_validate(payload)


def test_future_schema_and_incomplete_lineage_fail_closed(tmp_path: Path) -> None:
    payload = _contract("evade").model_dump(mode="json")
    payload["schema"] = "honcut.seedance-motion-blueprint.v99"
    with pytest.raises(ValueError):
        MotionBlueprintInput.model_validate(payload)
    payload = _contract("evade").model_dump(mode="json")
    payload["duration_s"] = 7
    with pytest.raises(ValueError):
        MotionBlueprintInput.model_validate(payload)
    manifest = compile_motion_blueprint(_contract("evade"), tmp_path / "schema-test.mp4")
    manifest["schema"] = "honcut.seedance-motion-blueprint.v99"
    with pytest.raises(ValueError):
        MotionBlueprintManifest.model_validate(manifest)
    payload = _contract("evade").model_dump(mode="json")
    payload["lineage"].pop("source_receipt_sha256")
    with pytest.raises(ValueError):
        MotionBlueprintInput.model_validate(payload)


def test_static_and_low_amplitude_semantics_fail_measurement() -> None:
    contract = _contract("evade")
    frames, intervals = compile_semantic_frames(contract)
    for frame in frames:
        frame["root"] = list(frames[0]["root"])
        frame["joints"] = json.loads(json.dumps(frames[0]["joints"]))
    with pytest.raises(ValueError, match=r"action onset|sub-threshold kinetics"):
        measure_semantic_frames(contract, frames, intervals)


def test_slow_endpoint_drift_and_single_joint_motion_fail_kinetics() -> None:
    contract = _contract("evade")
    frames, intervals = compile_semantic_frames(contract)
    start = frames[0]
    total = len(frames) - 1
    for index, frame in enumerate(frames):
        progress = index / total
        frame["root"] = [start["root"][0] + 0.21 * progress, start["root"][1]]
        frame["joints"] = json.loads(json.dumps(start["joints"]))
    with pytest.raises(ValueError, match=r"action onset|sub-threshold kinetics"):
        measure_semantic_frames(contract, frames, intervals)

    frames, intervals = compile_semantic_frames(contract)
    start = frames[0]
    for frame in frames:
        frame["root"] = list(start["root"])
        frame["joints"] = json.loads(json.dumps(start["joints"]))
    peak = intervals[0]["start_frame"] + 2
    frames[peak]["joints"]["right_wrist"][0] += 0.8
    with pytest.raises(ValueError, match=r"action onset|sub-threshold kinetics"):
        measure_semantic_frames(contract, frames, intervals)


def test_rendered_blueprint_requires_visible_full_scale_motion(tmp_path: Path) -> None:
    path = tmp_path / "visible.mp4"
    manifest = compile_motion_blueprint(_contract("ready", "evade"), path)
    rendered = measure_rendered_blueprint_motion(path)
    assert rendered == manifest["measurements"]["rendered_motion"]
    assert rendered["deterministic_motion_pass"] is True
    assert rendered["median_actor_height_fraction"] >= 0.46

    tiny = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=214:120,pad=854:480:(ow-iw)/2:(oh-ih)/2:color=0x121418",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tiny),
        ],
        check=True,
    )
    assert measure_rendered_blueprint_motion(tiny)["deterministic_motion_pass"] is False


def test_legacy_slow_drift_is_audit_only_and_rejected_without_upload(tmp_path: Path) -> None:
    media = tmp_path / "legacy.mp4"
    current = compile_motion_blueprint(_contract("ready", "evade"), media)
    legacy = {
        **current,
        "schema": "honcut.seedance-motion-blueprint.v1",
        "policy_schema": "honcut.motion-blueprint-policy.v1",
        "renderer_id": "honcut.identity-neutral-motion-renderer.v1",
        "measurements": {
            "fps": 24,
            "events": [
                {
                    "event_id": "S01_P01_M01",
                    "primitive": "ready",
                    "start_frame": 5,
                    "end_frame": 16,
                    "root_displacement": 0.08,
                    "max_joint_displacement": 0.20,
                },
                {
                    "event_id": "S01_P01_M02",
                    "primitive": "evade",
                    "start_frame": 17,
                    "end_frame": 198,
                    "root_displacement": 0.316,
                    "max_joint_displacement": 0.49,
                },
            ],
        },
    }
    path = tmp_path / "legacy-manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assessment = assess_legacy_blueprint_manifest(path)
    assert assessment["audit_only"] is True
    assert assessment["admission_status"] == "paid_admission_blocked"
    assert assessment["slow_dynamic_events"][0]["event_id"] == "S01_P01_M02"
    with pytest.raises(ValueError):
        MotionBlueprintManifest.model_validate(legacy)


def test_v2_common_curve_blueprint_is_audit_only(tmp_path: Path) -> None:
    media = tmp_path / "v2.mp4"
    legacy = compile_motion_blueprint(_contract("evade"), media)
    legacy["schema"] = "honcut.seedance-motion-blueprint.v2"
    legacy["policy_schema"] = "honcut.motion-blueprint-policy.v2"
    legacy["renderer_id"] = "honcut.identity-neutral-motion-renderer.v2"
    legacy.pop("technique_registry_sha256")
    for event in legacy["measurements"]["events"]:
        for field in (
            "technique_id",
            "technique_phase_ids",
            "technique_contact_phase_ids",
            "technique_keyframes_sha256",
            "observed_technique_phase_ids",
            "technique_phase_order_valid",
            "contact_frame_count",
            "observed_contact_phase_ids",
        ):
            event.pop(field, None)
    path = tmp_path / "v2-manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assessment = assess_legacy_blueprint_manifest(path)
    assert assessment["source_schema"] == "honcut.seedance-motion-blueprint.v2"
    assert assessment["audit_only"] is True
    assert assessment["admission_status"] == "paid_admission_blocked"
    assert "v3 technique choreography" in assessment["reason"]


def test_v3_single_action_policy_is_audit_only(tmp_path: Path) -> None:
    media = tmp_path / "v3.mp4"
    legacy = compile_motion_blueprint(_contract("ready", "strike", "block"), media)
    legacy["schema"] = "honcut.seedance-motion-blueprint.v3"
    legacy["policy_schema"] = "honcut.motion-blueprint-policy.v3"
    legacy["renderer_id"] = "honcut.identity-neutral-motion-renderer.v3"
    path = tmp_path / "v3-manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assessment = assess_legacy_blueprint_manifest(path)
    assert assessment["source_schema"] == "honcut.seedance-motion-blueprint.v3"
    assert assessment["audit_only"] is True
    assert assessment["admission_status"] == "paid_admission_blocked"
    assert "v4 combination density" in assessment["reason"]


def test_v4_two_action_policy_is_audit_only(tmp_path: Path) -> None:
    media = tmp_path / "v4.mp4"
    legacy = compile_motion_blueprint(_contract("ready", "strike", "block"), media)
    legacy["schema"] = "honcut.seedance-motion-blueprint.v4"
    legacy["policy_schema"] = "honcut.motion-blueprint-policy.v4"
    legacy["renderer_id"] = "honcut.identity-neutral-motion-renderer.v4"
    path = tmp_path / "v4-manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assessment = assess_legacy_blueprint_manifest(path)
    assert assessment["source_schema"] == "honcut.seedance-motion-blueprint.v4"
    assert assessment["audit_only"] is True
    assert assessment["admission_status"] == "paid_admission_blocked"
    assert "v5 fast-combination cadence" in assessment["reason"]


def _fixture_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "source"
    run.mkdir()
    canonical = build_canonical_visual_contract(
        {
            "characters": [
                {
                    "id": "actor_01",
                    "name": "Actor",
                    "visual_identity_policy": "fictional_cinematic_human_v1",
                    "appearance": {
                        "hair": "short black",
                        "build": "athletic",
                        "face": "fictional",
                        "clothing": "neutral",
                    },
                }
            ]
        },
        requested_policy="fictional_cinematic_human_v1",
    )
    (run / "CANONICAL_VISUAL_CONTRACT.json").write_text(json.dumps(canonical), encoding="utf-8")
    media = []
    for name, responsibility in (
        ("identity.png", "character_identity_board"),
        ("first.png", "cinematic_composition"),
        ("atlas.png", "storyboard_pose_atlas"),
    ):
        path = run / name
        path.write_bytes((name * 20).encode())
        media.append(
            {
                "content_index": len(media) + 1,
                "media_type": "image_url",
                "prompt_index": f"图片{len(media) + 1}",
                "role": "reference_image",
                "responsibility": responsibility,
                "path": name,
                "sha256": _sha(path),
            }
        )
    prompt = "图片1锁定身份；图片2锁定开场；图片3负责动作。"
    (run / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    action_specs = (
        ("ready", "actor_01 holds ready stance", "ready guard", ""),
        ("strike", "actor_01 drives a right-hand strike", "right-hand strike", "right hand contacts"),
        ("evade", "actor_01 lunges left to evade", "leftward evade", ""),
        ("block", "actor_01 raises both arms to block", "two-arm block", "arms absorb contact"),
    )
    mechanics_beats = []
    units = []
    for index, (_family, action, technique, contact) in enumerate(action_specs, 1):
        mechanics = {
            "micro_action_index": index,
            "micro_action": action,
            "performer": "actor_01",
            "technique": technique,
            "side": "right" if index == 2 else "bilateral",
            "limbs": ["right arm", "right hand", "left leg", "left foot", "waist", "head"],
            "footwork": "left foot supports while right foot advances",
            "torso": "waist coordinates the movement",
            "weight_shift": "weight moves through the stance",
            "direction": "left" if index == 3 else "forward",
            "contact": contact,
            "end_pose": "balanced completion",
        }
        mechanics_beats.append(
            {**mechanics, "kinematics": compile_source_kinematics(mechanics)}
        )
        units.append(
            {
                "unit_id": f"GAU{index:03d}",
                "source_action_unit_id": f"AU{index:03d}",
                "source_micro_action_indexes": [index],
                "actions": [action],
            }
        )
    kinematics_record = {
        "beat_id": "S01_P01",
        "body_action_contract": {
            "schema": "honcut.body-action-choreography.v2",
            "required": True,
            "valid": True,
            "beats": mechanics_beats,
        },
        "generation_action_units": units,
    }
    apply_generation_kinematics_projection(kinematics_record)
    projected_units = {
        unit["unit_id"]: unit for unit in kinematics_record["generation_action_units"]
    }
    samples = []
    for index, family in enumerate(("ready", "strike", "evade", "block"), 1):
        samples.append(
            {
                "sample_id": f"G{index:02d}",
                "action_group_id": f"S01_P01_A{index:02d}",
                "pose_contract": {
                    "pose_family": family,
                    "direction": "right",
                    "actor_roles": ["actor_01"],
                    "object_roles": ["prop"] if index == 2 else [],
                    "camera_vector": {"x": 1, "y": 0},
                    "timing_role": "initial_anchor" if index == 1 else "story_action",
                },
            }
        )
    plan = {
        "shots": [
            {
                "chunks": [
                    {
                        "storyboard_beat_id": "S01_P01",
                        "target_duration_s": 7,
                        "storyboard_pose_atlas_pose_samples": samples,
                        "storyboard_pose_atlas_action_groups": [
                            {
                                "action_group_id": f"S01_P01_A{index:02d}",
                                "order": index,
                        "lineage": {
                            "unit_ids": [f"GAU{index:03d}"],
                            "source_action_unit_ids": [f"AU{index:03d}"],
                        },
                        "kinematics_projection": projected_units[f"GAU{index:03d}"][
                            "kinematics_projection"
                        ],
                        "kinematics_projection_sha256": projected_units[
                            f"GAU{index:03d}"
                        ]["kinematics_projection_sha256"],
                    }
                            for index in (1, 2, 3, 4)
                        ],
                    }
                ]
            }
        ]
    }
    (run / "CONTINUITY_PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    receipt = {
        "schema": gate.SOURCE_RECEIPT_SCHEMA,
        "preflight": {
            "beat_id": "S01_P01",
            "duration": 7,
            "ratio": "16:9",
            "resolution": "480p",
            "prompt_path": "prompt.txt",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "media_index_manifest": media,
            "synthetic_identity": {
                "canonical_visual_contract_sha256": canonical["contract_sha256"]
            },
        },
        "task_payload": {
            "model": "doubao-seedance-2.0-fast",
            "duration": 7,
            "ratio": "16:9",
            "resolution": "480p",
            "phase6_prompt_projection_schema": "honcut.phase6-prompt-projection.v1",
            "phase6_prompt_projection_sha256": "b" * 64,
            "generation_fingerprint": "c" * 64,
            "input_fingerprint": "d" * 64,
            "media_index_manifest": media,
        },
    }
    receipt_path = run / "phase6_storyboard_pose_atlas_live_acceptance.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return run, receipt_path


def test_no_submit_projection_is_single_variable_seedance_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, receipt = _fixture_run(tmp_path)
    monkeypatch.setattr(gate, "SEEDANCE_MODEL", "doubao-seedance-2.0-fast")
    result = gate.prepare_gate(run, tmp_path / "gate", source_receipt_path=receipt)
    assert result["status"] == "pending_live_acceptance"
    assert result["provider_request_count"] == result["tos_put_count"] == 0
    assert result["budgets"] == {
        "tos_put_ceiling": 3,
        "video_submission_ceiling": 1,
        "automatic_retry_ceiling": 0,
        "alternate_provider_submission_ceiling": 0,
    }
    projected = result["request_projection"]["generation"]
    assert projected["model"] == "doubao-seedance-2.0-fast"
    assert projected["duration"] == 4
    assert result["request_projection"]["equivalence"]["source_duration"] == 7
    assert result["request_projection"]["equivalence"]["control_duration"] == 4
    assert result["request_projection"]["equivalence"]["candidate_duration"] == 4
    assert [item["responsibility"] for item in projected["media"]] == [
        "character_identity_board",
        "cinematic_composition",
        "motion_blueprint",
    ]
    assert projected["media"][-1]["media_type"] == "video_url"
    assert projected["media"][-1]["prompt_index"] == "视频1"
    prompt = (tmp_path / "gate" / "seedance_prompt.txt").read_text()
    assert "[honcut.motion-blueprint-reference.v6]" in prompt
    assert "视频1仅负责当前Pxx的动作时序" in prompt
    for tuning_phrase in ("0.15秒", "爆发峰值", "预备反向", "越位与回收", "慢速匀速漂移"):
        assert tuning_phrase not in prompt
    assert "图片3负责动作" not in (tmp_path / "gate" / "seedance_prompt.txt").read_text()
    assert projected["motion_blueprint_schema"] == "honcut.seedance-motion-blueprint.v6"
    assert len(projected["motion_kinematics_projection_sha256s"]) == 3
    assert all(
        item["technique_id"].startswith("canonical:")
        for item in projected["motion_techniques"]
    )
    assert all(
        item["technique_phase_ids"]
        for item in projected["motion_techniques"]
    )


def test_gate_selects_canonical_combo_but_never_borrows_another_pxx_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, receipt_path = _fixture_run(tmp_path)
    plan_path = run / "CONTINUITY_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    p01 = plan["shots"][0]["chunks"][0]
    p02 = json.loads(json.dumps(p01))
    p01["storyboard_pose_atlas_pose_samples"] = p01[
        "storyboard_pose_atlas_pose_samples"
    ][:3]
    p01["storyboard_pose_atlas_action_groups"] = p01[
        "storyboard_pose_atlas_action_groups"
    ][:3]
    p02["storyboard_beat_id"] = "S01_P02"
    for index, sample in enumerate(p02["storyboard_pose_atlas_pose_samples"], 1):
        sample["action_group_id"] = f"S01_P02_A{index:02d}"
    for index, group in enumerate(p02["storyboard_pose_atlas_action_groups"], 1):
        group["action_group_id"] = f"S01_P02_A{index:02d}"
        group["lineage"] = {
            "unit_ids": [f"GAU{index:03d}"],
            "source_action_unit_ids": [f"AU{index + 10:03d}"],
        }
        projection = group["kinematics_projection"]
        projection["beat_id"] = "S01_P02"
        projection.pop("projection_sha256")
        projection["projection_sha256"] = hashlib.sha256(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        group["kinematics_projection_sha256"] = projection["projection_sha256"]
    plan["shots"][0]["chunks"].append(p02)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(gate, "SEEDANCE_MODEL", "doubao-seedance-2.0-fast")

    result = gate.prepare_gate(run, tmp_path / "gate", source_receipt_path=receipt_path)

    assert result["status"] == "pending_source_request_projection"
    assert result["call_chain_verdict"] == "not_submittable"
    assert result["business_motion_verdict"] == {"status": "not_admitted"}
    assert result["candidate_selection"] == {
        "selected_beat_id": "S01_P02",
        "source_receipt_beat_id": "S01_P01",
        "combination": result["blueprint"]["measurements"]["combination"],
        "exact_production_request_available": False,
    }
    assert result["request_projection"]["status"] == "missing_exact_pxx_receipt"
    assert result["budgets"]["video_submission_ceiling"] == 0
    unavailable = gate._read_object(tmp_path / "gate" / "request_projection.runtime.json")
    assert unavailable == {
        "schema": "honcut.seedance-motion-blueprint-request-unavailable.v1",
        "status": "missing_exact_pxx_receipt",
        "selected_beat_id": "S01_P02",
        "provider_request_count": 0,
    }
    with pytest.raises(gate.GateEvidenceError, match="already submitted or is not current"):
        gate.submit_gate(
            tmp_path / "gate",
            fee_authorization="authorized-seedance-motion-blueprint-once",
        )


def test_no_canonical_combination_fails_before_projection(tmp_path: Path) -> None:
    run, receipt_path = _fixture_run(tmp_path)
    plan_path = run / "CONTINUITY_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    chunk = plan["shots"][0]["chunks"][0]
    chunk["storyboard_pose_atlas_pose_samples"] = chunk[
        "storyboard_pose_atlas_pose_samples"
    ][:2]
    chunk["storyboard_pose_atlas_action_groups"] = chunk[
        "storyboard_pose_atlas_action_groups"
    ][:2]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(gate.GateEvidenceError, match="three canonical dynamic actions"):
        gate.prepare_gate(run, tmp_path / "gate", source_receipt_path=receipt_path)
    assert not (tmp_path / "gate" / "request_projection.runtime.json").exists()


@pytest.mark.parametrize("model", ["wan2.2", "kling-v2", "doubao-seedance-1.5-pro"])
def test_alternate_or_unsupported_model_is_rejected(model: str) -> None:
    with pytest.raises(gate.GateEvidenceError, match=r"Seedance 2.0"):
        gate._assert_seedance_only(model)


def test_changed_identity_or_duration_breaks_equivalence(tmp_path: Path) -> None:
    run, receipt_path = _fixture_run(tmp_path)
    contract, receipt, _ = gate._build_contract(run, receipt_path)
    blueprint = compile_canonical_motion_blueprint(contract, tmp_path / "motion.mp4")
    gate._project_request(run, receipt, blueprint)
    receipt["preflight"]["duration"] = 8
    with pytest.raises(gate.GateEvidenceError, match="frozen model/output profile"):
        gate._project_request(run, receipt, blueprint)
    receipt["preflight"]["duration"] = 7
    identity = run / "identity.png"
    identity.write_bytes(b"changed")
    with pytest.raises(gate.GateEvidenceError, match="sha256 mismatch"):
        gate._project_request(run, receipt, blueprint)


def test_submit_requires_authorization_and_passing_regression(tmp_path: Path) -> None:
    run, receipt = _fixture_run(tmp_path)
    gate.prepare_gate(run, tmp_path / "gate", source_receipt_path=receipt)
    with pytest.raises(gate.GateEvidenceError, match="fee authorization"):
        gate.submit_gate(tmp_path / "gate", fee_authorization="")
    with pytest.raises(gate.GateEvidenceError, match="passing bound regression"):
        gate.submit_gate(
            tmp_path / "gate", fee_authorization="authorized-seedance-motion-blueprint-once"
        )


def test_provider_success_alone_cannot_pass_capability(tmp_path: Path) -> None:
    contract = _contract("locomotion", "strike", "evade")
    output = tmp_path / "gate" / "seedance_output.mp4"
    manifest = compile_motion_blueprint(contract, output)
    receipt = {
        "schema": gate.RECEIPT_SCHEMA,
        "status": "pending_human_verdict",
        "submitted": True,
        "call_chain_verdict": "passed",
        "output_path": str(output),
        "output_sha256": manifest["media_sha256"],
    }
    gate._write_object(tmp_path / "gate" / "seedance_motion_blueprint_gate.json", receipt)
    assert (
        gate._read_object(tmp_path / "gate" / "seedance_motion_blueprint_gate.json")["status"]
        != "capability_gate_passed"
    )
    finalized = gate.record_human_verdict(
        tmp_path / "gate", verdict="pass", notes="ordered motion observed"
    )
    assert finalized["status"] == "capability_gate_passed"
    assert finalized["business_motion_verdict"]["production_activation_authorized"] is False


def test_conclusive_human_failure_pauses_route_without_retry(tmp_path: Path) -> None:
    contract = _contract("locomotion", "strike")
    output = tmp_path / "gate" / "seedance_output.mp4"
    manifest = compile_motion_blueprint(contract, output)
    gate._write_object(
        tmp_path / "gate" / "seedance_motion_blueprint_gate.json",
        {
            "schema": gate.RECEIPT_SCHEMA,
            "status": "pending_human_verdict",
            "submitted": True,
            "call_chain_verdict": "passed",
            "output_path": str(output),
            "output_sha256": manifest["media_sha256"],
        },
    )
    finalized = gate.record_human_verdict(
        tmp_path / "gate", verdict="fail", notes="motion did not transfer"
    )
    assert finalized["status"] == "capability_route_paused"
    assert all(value is False for value in finalized["enforcement"].values())


def test_ten_no_submit_resumes_are_hash_stable_and_never_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, receipt = _fixture_run(tmp_path)
    monkeypatch.setattr(
        gate, "upload_media_file_required", lambda *_a, **_k: pytest.fail("upload attempted")
    )
    hashes = []
    results = []
    for _ in range(10):
        result = gate.prepare_gate(run, tmp_path / "gate", source_receipt_path=receipt)
        results.append(result)
        hashes.append(
            (
                result["blueprint"]["media_sha256"],
                result["request_projection"]["generation"]["task_fingerprint"],
            )
        )
    assert len(set(hashes)) == 1
    assert results[-1]["provider_request_count"] == 0


def test_optional_submit_uses_existing_task_ledger_and_blocks_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, receipt_path = _fixture_run(tmp_path)
    output_dir = tmp_path / "gate"
    gate.prepare_gate(run, output_dir, source_receipt_path=receipt_path)
    receipt_file = output_dir / "seedance_motion_blueprint_gate.json"
    receipt = gate._read_object(receipt_file)
    receipt["regression"] = {"status": "passed"}
    gate._write_object(receipt_file, receipt)
    monkeypatch.setattr(
        gate, "upload_media_file_required", lambda *_a, **_k: "https://invalid.local/signed"
    )
    monkeypatch.setattr(gate, "get_api_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr(
        gate.seedance_client, "_validate_content_media_roles", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        gate.seedance_client, "poll", lambda *_a, **_k: "https://download.invalid/video.mp4"
    )

    def fake_download(_url: str, destination: str) -> str:
        shutil.copy2(output_dir / "motion_blueprint.mp4", destination)
        return destination

    monkeypatch.setattr(gate.seedance_client, "download", fake_download)

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, str]:
            events = gate.GenerationTaskStore(output_dir / "runtime.db")
            rows = (
                events._connect()
                .execute(
                    "SELECT event_type FROM generation_task_events ORDER BY event_sequence"
                )
                .fetchall()
            )
            assert [row["event_type"] for row in rows][-1] == "SubmissionAttempted"
            return {"id": "seedance-job-1"}

    raw_calls = []

    def fake_post(url: str, *args: object, **kwargs: object) -> Response:
        raw_calls.append(url)
        return Response()

    monkeypatch.setattr(gate.seedance_client.requests, "post", fake_post)
    result = gate.submit_gate(
        output_dir,
        fee_authorization="authorized-seedance-motion-blueprint-once",
    )
    assert result["provider_request_count"] == 1
    assert result["call_chain_verdict"] == "passed"
    assert len(raw_calls) == 1
    events = gate.GenerationTaskStore(output_dir / "runtime.db").events(
        result["generation_task_id"]
    )
    assert [event.event_type for event in events] == [
        "TaskQueued",
        "TaskClaimed",
        "SubmissionAttempted",
        "ProviderAccepted",
        "TaskSucceeded",
    ]
    with pytest.raises(gate.GateEvidenceError, match="already submitted"):
        gate.submit_gate(
            output_dir,
            fee_authorization="authorized-seedance-motion-blueprint-once",
        )
    assert len(raw_calls) == 1


def test_acceptance_module_is_not_imported_by_production_entrypoints() -> None:
    root = Path(__file__).parents[1] / "src"
    forbidden = [
        root / "pipeline_runner.py",
        root / "runtime" / "pipeline_execution.py",
        root / "graph" / "workflow.py",
    ]
    for path in forbidden:
        assert "acceptance.motion_blueprint" not in path.read_text(encoding="utf-8")
