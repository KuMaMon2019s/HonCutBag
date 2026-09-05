from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import ImageFont

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase6_action_execution_replay
from graph import composition as graph_composition
from runtime import continuity_provider
from runtime import pipeline_execution
from phases.phase2.storyboard_pose_atlas import (
    build_pose_atlas_plan,
    render_pose_atlas_candidates,
)
from phases.phase6 import action_execution_prompt as action_prompt_owner
from phases.phase6 import phase6_video_gen
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import (
    _bind_final_media_index_prompt,
    _provider_prompt_metadata,
    _task_payload,
)
from runtime.execution_errors import ProviderPreparationError
from runtime.generation_tasks import GenerationTaskStore
from schemas.continuity import GenerationChunk
from utils.prompt_budget import PromptBudget, PromptBudgetExceededError


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


def _request_and_content(tmp_path, *, duration: float = 7) -> tuple:
    actions = ["actor-alpha shifts weight and evades", "actor-alpha blocks contact"]
    beat = {
        "beat_id": "S01_P01",
        "duration_s": duration,
        "planner_version": "honcut.secondary-storyboard.v17",
        "generation_action_units": [
            _unit(index, action) for index, action in enumerate(actions, 1)
        ],
        "character_ids": ["actor-alpha", "actor-beta"],
    }
    plan = build_pose_atlas_plan(beat)
    receipt = render_pose_atlas_candidates(
        tmp_path,
        plan,
        font_factory=lambda _size: ImageFont.load_default(),
    )
    selected = next(candidate for candidate in receipt["candidates"] if candidate["preferred"])
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=duration,
        mode="fresh",
        storyboard_beat_id="S01_P01",
        action_prompt="→".join(actions),
        start_state="actor-alpha is already moving",
        end_state="actor-alpha completes the block with stable balance",
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
        input_fingerprint="action-first-fixture",
        memory_context="",
    )
    page = selected["pages"][0]
    content = [
        {
            "type": "text",
            "text": (
                "18秒慢节奏真人皮肤；先保持戒备姿态，再按旧动作尾部执行。"
                "[honcut-video-generation-contract-v2]"
            ),
            "_canonical_identity_projection": (
                "[honcut.phase6-identity-projection.v1]\n"
                '{"canonical_visual_contract_sha256":"'
                + "a"
                * 64
                + '","characters":[{"instance_count":1,"face":"stable"}],'
                '"required_character_count":1,'
                '"schema":"honcut.phase6-identity-projection.v1"}'
            ),
            "_canonical_visual_contract_sha256": "a" * 64,
            "_phase6_prompt_context": {
                "where": "interior",
                "visual": "cinematic lighting",
                "camera_movement": "track right",
                "emotion": "controlled urgency",
            },
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/identity.png"},
            "role": "reference_image",
            "_reference_kind": "character_identity_board",
            "_reference_description": "identity board",
            "_reference_sha256": "1" * 64,
            "_mandatory_reference": True,
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/frame.png"},
            "role": "reference_image",
            "_reference_kind": "cinematic_composition",
            "_reference_description": "first frame",
            "_reference_sha256": "2" * 64,
            "_mandatory_reference": True,
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/atlas.png"},
            "role": "reference_image",
            "_reference_kind": "storyboard_pose_atlas",
            "_reference_description": "current pose atlas",
            "_reference_sha256": page["image_sha256"],
            "_narrative_beat_id": "S01_P01",
            "_narrative_cell_ids": list(page["sample_ids"]),
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
    return request, content


def test_action_first_projection_replaces_conflicting_legacy_prompt(tmp_path) -> None:
    request, content = _request_and_content(tmp_path)

    first, manifest = _bind_final_media_index_prompt(content, request)
    second, second_manifest = _bind_final_media_index_prompt(content, request)

    assert first[0]["text"] == second[0]["text"]
    assert manifest == second_manifest
    prompt = first[0]["text"]
    assert "18秒" not in prompt
    assert "真人皮肤" not in prompt
    assert "先保持戒备姿态" not in prompt
    assert "[honcut-video-generation-contract-v2]" not in prompt
    assert prompt.count("[honcut.action-execution-brief.v2]") == 1
    assert prompt.count("唯一主运镜") == 1
    assert prompt.index("[honcut.action-execution-brief.v2]") < prompt.index(
        "[honcut.phase6-identity-projection.v1]"
    )
    assert first[0]["_action_execution_group_ids"] == [
        group["action_group_id"] for group in request.chunk.storyboard_pose_atlas_action_groups
    ]
    assert all(prompt.count(group_id) == 1 for group_id in first[0]["_action_execution_group_ids"])
    assert [item["sha256"] for item in manifest] == [
        "1" * 64,
        "2" * 64,
        content[-1]["_reference_sha256"],
    ]


def test_graph_and_sequential_execution_share_phase6_owner() -> None:
    """Both orchestration paths must reach the same production Phase owner."""

    assert pipeline_execution.run_phase6 is phase6_video_gen.run_phase6
    assert graph_composition._resolve_phase_owner(None).run_phase6 is (phase6_video_gen.run_phase6)


def test_action_brief_metadata_changes_with_action_lineage(tmp_path) -> None:
    request, content = _request_and_content(tmp_path)
    first, _manifest = _bind_final_media_index_prompt(content, request)
    metadata = _provider_prompt_metadata(first)

    changed_groups = copy.deepcopy(request.chunk.storyboard_pose_atlas_action_groups)
    changed_groups[0]["lineage"]["source_action_unit_ids"] = ["AU999"]
    changed_chunk = request.chunk.model_copy(
        update={"storyboard_pose_atlas_action_groups": changed_groups}
    )
    changed_request = replace(request, chunk=changed_chunk)
    changed, _manifest = _bind_final_media_index_prompt(content, changed_request)
    changed_metadata = _provider_prompt_metadata(changed)

    assert (
        metadata["action_execution_brief_sha256"]
        != changed_metadata["action_execution_brief_sha256"]
    )
    assert metadata["provider_prompt_sha256"] == changed_metadata["provider_prompt_sha256"]
    assert metadata["action_execution_source_action_unit_ids"] == ["AU001", "AU002"]
    assert changed_metadata["action_execution_source_action_unit_ids"] == [
        "AU999",
        "AU002",
    ]
    base_payload = _task_payload(
        request,
        model="seedance-test",
        provider_id="seedance",
        provider_version="test",
        project_id="test",
        run_id="test",
        duration=7,
        seed=1,
        generation_parameters=metadata,
    )
    changed_payload = _task_payload(
        changed_request,
        model="seedance-test",
        provider_id="seedance",
        provider_version="test",
        project_id="test",
        run_id="test",
        duration=7,
        seed=1,
        generation_parameters=changed_metadata,
    )
    assert base_payload["generation_fingerprint"] != changed_payload["generation_fingerprint"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duration", "conflicts with chunk duration"),
        ("group_order", "group order"),
        ("future_beat", "future or foreign"),
        ("media_hash", "hash is missing or invalid"),
    ],
)
def test_action_brief_rejects_invalid_deterministic_inputs(
    tmp_path,
    mutation,
    message,
) -> None:
    request, content = _request_and_content(tmp_path)
    if mutation == "duration":
        timing = copy.deepcopy(request.chunk.storyboard_pose_atlas_timing_contract)
        timing["duration_s"] = 6
        request = replace(
            request,
            chunk=request.chunk.model_copy(
                update={"storyboard_pose_atlas_timing_contract": timing}
            ),
        )
    elif mutation == "group_order":
        groups = copy.deepcopy(request.chunk.storyboard_pose_atlas_action_groups)
        groups[0]["order"] = 2
        request = replace(
            request,
            chunk=request.chunk.model_copy(update={"storyboard_pose_atlas_action_groups": groups}),
        )
    elif mutation == "future_beat":
        samples = copy.deepcopy(request.chunk.storyboard_pose_atlas_pose_samples)
        samples[0]["pose_contract"]["secondary_beat_id"] = "S01_P02"
        request = replace(
            request,
            chunk=request.chunk.model_copy(update={"storyboard_pose_atlas_pose_samples": samples}),
        )
    else:
        content[-1].pop("_reference_sha256")

    with pytest.raises(ValueError, match=message):
        _bind_final_media_index_prompt(content, request)


def test_prompt_budget_omits_whole_optional_sections(monkeypatch) -> None:
    budget = PromptBudget(
        provider="seedance",
        model="test",
        purpose="video_generation",
        soft_chars=500,
        hard_chars=620,
    )
    monkeypatch.setattr(action_prompt_owner, "resolve_prompt_budget", lambda **_kwargs: budget)

    prompt, metadata = action_prompt_owner.project_action_first_prompt(
        media_index_preamble="media",
        media_role_isolation="roles",
        action_brief_text="action",
        identity_projection="identity",
        prompt_context={"where": "x" * 1_000, "emotion": "y" * 1_000},
        provider="seedance",
        model="test",
    )

    assert "x" * 100 not in prompt
    assert metadata["omitted_optional_sections"] == ["scene", "emotion"]
    assert metadata["total_chars"] == len(prompt)


def test_prompt_budget_fails_before_optional_truncation(monkeypatch) -> None:
    budget = PromptBudget(
        provider="seedance",
        model="test",
        purpose="video_generation",
        soft_chars=20,
        hard_chars=40,
    )
    monkeypatch.setattr(action_prompt_owner, "resolve_prompt_budget", lambda **_kwargs: budget)

    with pytest.raises(PromptBudgetExceededError, match="mandatory"):
        action_prompt_owner.project_action_first_prompt(
            media_index_preamble="media" * 20,
            media_role_isolation="roles",
            action_brief_text="action",
            identity_projection="identity",
            prompt_context={},
            provider="seedance",
            model="test",
        )


def test_mandatory_prompt_overflow_creates_no_submission_event(
    monkeypatch,
    tmp_path,
) -> None:
    request, _content = _request_and_content(tmp_path)
    store = GenerationTaskStore(tmp_path / "runtime.db")
    monkeypatch.setattr(
        "utils.config.get_api_key_or_raise",
        lambda _provider: "test-key",
    )
    monkeypatch.setattr(
        continuity_provider,
        "_read_shot_meta",
        lambda *_args: {"aspect_ratio": "16:9"},
    )
    monkeypatch.setattr(
        continuity_provider,
        "_provider_content",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PromptBudgetExceededError("mandatory action brief exceeds hard limit")
        ),
    )
    execute = continuity_provider._direct_seedance_executor(
        tmp_path,
        store,
        allow_policy_repairs=False,
    )

    with pytest.raises(ProviderPreparationError, match="mandatory action brief"):
        execute(request)

    assert store.submission_attempt_count() == 0


@pytest.mark.parametrize("duration", [4, 7, 10, 15])
@pytest.mark.parametrize("beat_id", ["S04_P01", "S04_P02"])
def test_action_brief_supports_provider_durations_and_later_beats(
    duration,
    beat_id,
) -> None:
    action = "actor-alpha steps through contact and transfers weight"
    plan = build_pose_atlas_plan(
        {
            "beat_id": beat_id,
            "duration_s": duration,
            "planner_version": "honcut.secondary-storyboard.v17",
            "generation_action_units": [_unit(1, action)],
            "character_ids": ["actor-alpha"],
        }
    )
    brief = action_prompt_owner.compile_action_execution_brief(
        beat_id=beat_id,
        action_prompt=action,
        start_state="moving",
        end_state="balanced",
        target_duration_s=duration,
        action_groups=plan["action_groups"],
        pose_samples=plan["pose_samples"],
        timing_contract=plan["timing_contract"],
        media_manifest=[
            {
                "prompt_index": "图片1",
                "responsibility": "character_identity_board",
                "sha256": "1" * 64,
            },
            {
                "prompt_index": "图片2",
                "responsibility": "storyboard_pose_atlas",
                "narrative_cell_ids": [sample["sample_id"] for sample in plan["pose_samples"]],
                "sha256": "2" * 64,
            },
        ],
        prompt_context={"camera_movement": "track right"},
        canonical_visual_contract_sha256="a" * 64,
    )

    assert brief["duration_s"] == duration
    assert brief["ordered_action_group_ids"] == [f"{beat_id}_A01"]
    assert bool(brief["initial_anchor_sample_ids"]) is beat_id.endswith("_P01")
    assert brief["target_completion_s"] + brief["terminal_hold"][
        "target_duration_s"
    ] == pytest.approx(duration)


def test_run02_provider_deny_replay_matches_sanitized_evidence(tmp_path) -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "phase6_action_execution"
    failure = json.loads((fixture_dir / "run02_evidence.json").read_text())
    expected = json.loads((fixture_dir / "run02_replay_expected.json").read_text())
    assert failure["action_priority_first_char"] > failure["prompt_chars"] * 0.9
    assert failure["timing_window_first_char"] > failure["camera_execution_first_char"]
    assert failure["business_verdict"] == "failed_static_opening_pose"
    assert failure["contains_prompt_or_media_body"] is False
    assert failure["contains_provider_url_or_secret"] is False
    assert failure["source_receipt_sha256"] == expected["source_receipt_sha256"]
    assert failure["source_continuity_plan_sha256"] == expected["continuity_plan_sha256"]
    assert expected["action_brief_position"] < expected["identity_projection_position"]
    assert expected["prompt_chars"] < failure["prompt_chars"]
    assert expected["provider_request_count"] == 0

    source_root = os.environ.get("HONCUT_PHASE6_RUN02_EVIDENCE_DIR", "").strip()
    if not source_root:
        pytest.skip("immutable run-02 evidence path was not supplied")
    source_run = Path(source_root)
    source_receipt = source_run / phase6_action_execution_replay.DEFAULT_SOURCE_RECEIPT
    if not source_receipt.is_file():
        pytest.skip("immutable run-02 evidence is not installed on this host")
    before = hashlib.sha256(source_receipt.read_bytes()).hexdigest()
    receipt = phase6_action_execution_replay.replay_persisted_action_request(
        source_run,
        source_receipt_path=source_receipt,
        output_receipt_path=tmp_path / "run02-replay.json",
    )
    after = hashlib.sha256(source_receipt.read_bytes()).hexdigest()

    assert before == after == expected["source_receipt_sha256"]
    assert receipt["continuity_plan_sha256"] == expected["continuity_plan_sha256"]
    assert receipt["prompt_chars"] == expected["prompt_chars"]
    assert receipt["action_brief_position"] == expected["action_brief_position"]
    assert receipt["identity_projection_position"] == expected["identity_projection_position"]
    assert (
        receipt["prompt_metadata"]["provider_prompt_sha256"] == expected["provider_prompt_sha256"]
    )
    assert (
        receipt["prompt_metadata"]["action_execution_brief_sha256"]
        == expected["action_execution_brief_sha256"]
    )
    assert (
        receipt["prompt_metadata"]["action_execution_group_ids"]
        == expected["canonical_action_group_ids"]
    )
    checks = receipt["prompt_contract_checks"]
    assert all(
        receipt["action_brief_position"] <= position < receipt["identity_projection_position"]
        for position in checks["action_group_positions"].values()
    )
    assert checks["marker_counts"] == {
        "action_execution_brief": 1,
        "identity_projection": 1,
        "primary_camera": 1,
        "legacy_live_pacing": 0,
        "legacy_video_contract": 0,
    }
    assert receipt["media_sha256"] == expected["media_sha256"]
    assert receipt["provider_request_count"] == 0
