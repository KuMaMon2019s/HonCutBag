from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_visual_ledger_36s_acceptance as acceptance
from phases.phase3.phase3_character import run_phase3
from phases.phase1.character_roster import (
    CHARACTER_ROSTER_FILENAME,
    compile_character_roster,
    persist_character_roster,
    reconcile_character_observations,
)
from utils.canonical_visual_contracts import (
    expand_character_instances,
    persist_canonical_visual_contract,
)
from utils.semantic_contracts import bind_story_semantics


def _write_inputs(workspace):
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True)
    story = input_dir / "story.txt"
    story.write_text("原创人物进入车站。人物观察手中装置。", encoding="utf-8")
    expectations = workspace / "acceptance_expectations.json"
    expectations.write_text(
        json.dumps({
            "schema": "honcut.full-chain-acceptance-expectations.v2",
            "expected_duration_s": 36,
            "expected_character_entities": 1,
            "expected_character_instances": 1,
            "entity_expectations": [{
                "expectation_id": "lead",
                "source_mentions_any": ["原创人物"],
                "instance_count": 1,
                "visual_facts": {},
            }],
            "required_events": ["进入车站", "观察手中装置"],
            "visual_facts": {
                "character_visual_policy": "fictional_cinematic_human_v1"
            },
        }),
        encoding="utf-8",
    )
    return story, expectations


def _persist_phase1_identity_artifacts(workspace, events):
    roster = persist_character_roster(
        workspace / CHARACTER_ROSTER_FILENAME,
        compile_character_roster(events),
    )
    entities, _diagnostics = reconcile_character_observations(
        [],
        roster,
        semantic_qa_enabled=False,
    )
    projected, contract = persist_canonical_visual_contract(
        workspace,
        {
            "characters": entities,
            "character_roster": roster,
            "character_roster_sha256": roster["roster_sha256"],
        },
        requested_policy="fictional_cinematic_human_v1",
    )
    semantic = bind_story_semantics(events, projected["characters"])
    projected = expand_character_instances(projected, contract)
    (workspace / "CHARACTERS.json").write_text(
        json.dumps(projected),
        encoding="utf-8",
    )
    (workspace / "SEMANTIC_LEDGER.json").write_text(
        json.dumps(semantic),
        encoding="utf-8",
    )
    (workspace / "phase1_events.json").write_text(
        json.dumps({"events": events}),
        encoding="utf-8",
    )
    return roster, contract


def test_stage0_preflight_is_zero_request_and_has_finite_hard_limits(
    tmp_path, monkeypatch
):
    story, expectations = _write_inputs(tmp_path)
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "configured")
    monkeypatch.setattr(acceptance, "is_media_upload_configured", lambda: True)
    monkeypatch.setattr(
        acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance,
        "_regression_evidence",
        lambda _workspace, _commit: {
            "status": "passed",
            "path": acceptance.REGRESSION_RECEIPT_NAME,
            "sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _required: {"valid": True, "missing": []},
    )

    receipt = acceptance.build_stage0_preflight(
        tmp_path,
        story,
        expectations,
    )

    assert receipt["status"] == "preflight_passed"
    assert receipt["provider_request_count"] == 0
    assert all(
        isinstance(value, int) and value > 0
        for value in receipt["authorized_hard_limits"].values()
    )
    hard_limits = receipt["authorized_hard_limits"]
    assert hard_limits["phase1_director_storyboard_image_requests"] == 1
    assert hard_limits["phase1_provider_requests"] == (
        hard_limits["phase1_text_requests"]
        + hard_limits["phase1_director_storyboard_image_requests"]
    )
    assert hard_limits["tos_put_attempts"] == (
        hard_limits["multimodal_observation_requests"]
        * hard_limits["max_tos_inputs_per_multimodal_request"]
        + hard_limits["seedance_video_submissions"]
        * hard_limits["max_tos_inputs_per_video_submission"]
    )
    assert receipt["configuration"]["automatic_reshoot"] is False
    assert receipt["configuration"]["phase1_semantic_qa"] is False
    assert receipt["configuration"]["character_library_configured"] is False


def test_acceptance_project_identity_is_derived_from_fresh_workspace(tmp_path):
    workspace = tmp_path / "honcut-canonical-visual-ledger-36s-run-03"
    arguments = acceptance._pipeline_arguments(
        workspace,
        workspace / "input" / "story.txt",
    )

    assert arguments["project_id"] == workspace.name


def test_stage0_preflight_rejects_input_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "story.txt"
    outside.write_text("story", encoding="utf-8")
    expectations = workspace / "acceptance_expectations.json"
    expectations.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes workspace"):
        acceptance.build_stage0_preflight(workspace, outside, expectations)


def test_stage0_rejects_expectation_anchors_absent_from_source(
    tmp_path,
    monkeypatch,
):
    story, expectations = _write_inputs(tmp_path)
    payload = json.loads(expectations.read_text(encoding="utf-8"))
    payload["required_events"] = ["不存在的来源事件"]
    expectations.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source anchors absent"):
        acceptance.build_stage0_preflight(tmp_path, story, expectations)


def test_source_mentions_any_accepts_one_matching_alias(tmp_path, monkeypatch):
    story, expectations = _write_inputs(tmp_path)
    payload = json.loads(expectations.read_text(encoding="utf-8"))
    payload["entity_expectations"][0]["source_mentions_any"] = [
        "absent alias",
        "原创人物",
    ]
    expectations.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "configured")
    monkeypatch.setattr(acceptance, "is_media_upload_configured", lambda: True)
    monkeypatch.setattr(
        acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance,
        "_regression_evidence",
        lambda _workspace, _commit: {"status": "passed"},
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _required: {"valid": True, "missing": []},
    )

    receipt = acceptance.build_stage0_preflight(
        tmp_path,
        story,
        expectations,
    )

    assert receipt["status"] == "preflight_passed"


def test_post_phase1_expectations_match_source_roster_and_semantic_ledger(
    tmp_path,
):
    events = [{
        "id": 1,
        "sequence_id": "SEQ001",
        "who": ["原创人物"],
        "source_excerpt": "原创人物进入车站并观察手中装置。",
        "what": "原创人物进入车站并观察手中装置",
    }]
    _roster, contract = _persist_phase1_identity_artifacts(tmp_path, events)
    _story, expectations = _write_inputs(tmp_path)

    evidence = acceptance.validate_post_phase1_expectations(
        tmp_path,
        {"source": {"expectations_path": expectations.name}},
    )

    assert evidence["status"] == "passed"
    assert evidence["character_entities"] == 1
    assert evidence["character_instances"] == 1
    assert evidence["entity_matches"]["lead"] == contract["characters"][0][
        "entity_id"
    ]


def test_entity_anchors_ignore_cross_character_narrative_evidence(tmp_path):
    events = [
        {
            "id": 1,
            "sequence_id": "SEQ001",
            "who": ["林夏"],
            "source_excerpt": "林夏看到顾北并停下脚步。",
            "what": "林夏看到顾北",
        },
        {
            "id": 2,
            "sequence_id": "SEQ001",
            "who": ["顾北"],
            "source_excerpt": "顾北阻挡林夏继续前进。",
            "what": "顾北阻挡林夏",
        },
    ]
    _persist_phase1_identity_artifacts(tmp_path, events)
    expectations = tmp_path / "acceptance_expectations.json"
    expectations.write_text(
        json.dumps({
            "schema": "honcut.full-chain-acceptance-expectations.v2",
            "expected_duration_s": 36,
            "expected_character_entities": 2,
            "expected_character_instances": 2,
            "entity_expectations": [
                {
                    "expectation_id": "lead",
                    "source_mentions_any": ["林夏"],
                    "instance_count": 1,
                    "visual_facts": {},
                },
                {
                    "expectation_id": "guard",
                    "source_mentions_any": ["顾北"],
                    "instance_count": 1,
                    "visual_facts": {},
                },
            ],
            "required_events": ["林夏看到顾北", "顾北阻挡林夏"],
            "visual_facts": {
                "character_visual_policy": "fictional_cinematic_human_v1",
            },
        }),
        encoding="utf-8",
    )

    evidence = acceptance.validate_post_phase1_expectations(
        tmp_path,
        {"source": {"expectations_path": expectations.name}},
    )

    assert evidence["status"] == "passed"
    assert evidence["entity_source_anchor_authority"] == (
        "character_roster_display_and_instance_mentions"
    )
    assert set(evidence["entity_matches"]) == {"lead", "guard"}
    assert len(set(evidence["entity_matches"].values())) == 2


def test_entity_anchor_still_fails_closed_when_roster_labels_are_ambiguous(
    tmp_path,
):
    events = [
        {
            "id": 1,
            "sequence_id": "SEQ001",
            "who": ["年轻男子甲"],
            "source_excerpt": "年轻男子甲独自进入房间。",
            "what": "年轻男子甲进入房间",
        },
        {
            "id": 2,
            "sequence_id": "SEQ001",
            "who": ["年轻男子乙"],
            "source_excerpt": "年轻男子乙随后进入房间。",
            "what": "年轻男子乙进入房间",
        },
    ]
    _persist_phase1_identity_artifacts(tmp_path, events)
    expectations = tmp_path / "acceptance_expectations.json"
    expectations.write_text(
        json.dumps({
            "schema": "honcut.full-chain-acceptance-expectations.v2",
            "expected_duration_s": 36,
            "expected_character_entities": 2,
            "expected_character_instances": 2,
            "entity_expectations": [
                {
                    "expectation_id": "underspecified_lead",
                    "source_mentions_any": ["年轻男子"],
                    "instance_count": 1,
                    "visual_facts": {},
                },
                {
                    "expectation_id": "second_lead",
                    "source_mentions_any": ["年轻男子乙"],
                    "instance_count": 1,
                    "visual_facts": {},
                },
            ],
            "required_events": [],
            "visual_facts": {
                "character_visual_policy": "fictional_cinematic_human_v1",
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="do not resolve uniquely"):
        acceptance.validate_post_phase1_expectations(
            tmp_path,
            {"source": {"expectations_path": expectations.name}},
        )


def test_stage0_preflight_rejects_source_reduced_to_zero_events(
    tmp_path,
    monkeypatch,
):
    story, expectations = _write_inputs(tmp_path)
    monkeypatch.setattr(acceptance, "get_api_key", lambda _name: "configured")
    monkeypatch.setattr(acceptance, "is_media_upload_configured", lambda: True)
    monkeypatch.setattr(
        acceptance,
        "_repo_source_identity",
        lambda: {"git_commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        acceptance,
        "_regression_evidence",
        lambda _workspace, _commit: {"status": "passed"},
    )
    monkeypatch.setattr(
        acceptance,
        "validate_config",
        lambda _required: {"valid": True, "missing": []},
    )
    zero_event_receipt = {
        "receipt": {
            "status": "passed",
            "source_derived_event_count": 0,
        }
    }
    monkeypatch.setattr(
        acceptance,
        "build_dry_run_capacity_preflight",
        lambda *_args, **_kwargs: zero_event_receipt,
    )

    receipt = acceptance.build_stage0_preflight(
        tmp_path,
        story,
        expectations,
    )

    assert receipt["status"] == "preflight_blocked"
    assert "source_structure_has_events" in receipt["missing_configuration"]
    assert receipt["provider_request_count"] == 0


def test_phase1_paid_ledger_allows_only_current_process_in_flight_requests(
    tmp_path,
):
    ledger = acceptance._BoundedPaidRequestLedger(
        tmp_path,
        "phase1_provider",
        3,
    )
    first = ledger.before({"provider_family": "ark_text", "prompt_sha256": "a"})
    second = ledger.before({"provider_family": "ark_text", "prompt_sha256": "b"})
    ledger.after(first, {"transport_status": "stream_accepted"})
    ledger.after(second, {"transport_status": "stream_accepted"})

    receipt = ledger.settled_receipt()

    assert receipt["status"] == "provider_completed"
    assert receipt["provider_request_count"] == 2
    assert [attempt["status"] for attempt in receipt["attempts"]] == [
        "provider_completed",
        "provider_completed",
    ]


def test_phase1_paid_ledger_refuses_uncertain_request_after_process_restart(
    tmp_path,
):
    ledger = acceptance._BoundedPaidRequestLedger(
        tmp_path,
        "phase1_provider",
        2,
    )
    ledger.before({"provider_family": "ark_text", "prompt_sha256": "a"})

    restored = acceptance._BoundedPaidRequestLedger(
        tmp_path,
        "phase1_provider",
        2,
    )
    with pytest.raises(RuntimeError, match="automatic resubmission is forbidden"):
        restored.before({"provider_family": "ark_text", "prompt_sha256": "b"})
    with pytest.raises(RuntimeError, match="failed or unresolved"):
        restored.settled_receipt()


def test_paid_request_summary_counts_families_without_copying_payloads(tmp_path):
    ledger = acceptance._BoundedPaidRequestLedger(
        tmp_path,
        "phase1_provider",
        3,
    )
    first = ledger.before({
        "provider_family": "ark_text",
        "prompt_sha256": "a" * 64,
        "secret": "must-not-be-copied",
    })
    second = ledger.before({
        "provider_family": "seedream_image",
        "prompt_sha256": "b" * 64,
    })
    ledger.after(first, {"transport_status": "response_completed"})
    ledger.after(second, {"transport_status": "response_validated"})

    summary = acceptance._paid_request_summary(tmp_path)

    assert summary["provider_request_count"] == 2
    assert summary["provider_family_counts"] == {
        "ark_text": 1,
        "seedream_image": 1,
    }
    assert len(summary["paid_request_receipts"]) == 1
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "must-not-be-copied" not in serialized
    assert "prompt_sha256" not in serialized


def test_full_chain_failure_persists_aggregated_paid_request_count(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HONCUT_PHASE5_MAX_CORRECTIONS", "3")
    story, _expectations = _write_inputs(tmp_path)
    preflight = {
        "schema": acceptance.RECEIPT_SCHEMA,
        "status": "preflight_passed",
        "provider_request_count": 0,
        "source": {
            "git_commit": "a" * 40,
            "story_sha256": acceptance._sha256(story),
            "expectations_sha256": "b" * 64,
        },
        "authorized_hard_limits": {
            "phase1_provider_requests": 2,
            "tos_put_attempts": 1,
        },
    }

    def fail_phase1(workspace, *_args, **_kwargs):
        ledger = acceptance._BoundedPaidRequestLedger(
            workspace,
            "phase1_provider",
            2,
        )
        token = ledger.before({
            "provider_family": "ark_text",
            "messages_sha256": "c" * 64,
        })
        ledger.failed(token, {
            "submission_outcome": "known_rejected",
            "error_type": "FixtureProviderError",
        })
        raise RuntimeError("fixture Provider rejection")

    monkeypatch.setattr(acceptance, "_run_selected_phases", fail_phase1)
    with pytest.raises(RuntimeError, match="fixture Provider rejection"):
        acceptance.execute_paid_full_chain(tmp_path, story, preflight)

    receipt = json.loads(
        (tmp_path / acceptance.RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "live_acceptance_failed"
    assert receipt["provider_request_count"] == 1
    assert receipt["provider_family_counts"] == {"ark_text": 1}
    assert receipt["paid_request_receipts"][0]["provider_request_count"] == 1
    assert "messages_sha256" not in json.dumps(
        receipt["paid_request_receipts"]
    )
    assert os.environ["HONCUT_PHASE5_MAX_CORRECTIONS"] == "3"
    assert "HONCUT_CONTINUITY_MODE" not in os.environ
    assert "HONCUT_CONTINUITY_MAX_REPAIRS" not in os.environ
    assert "VIDEO_GEN_CONCURRENCY" not in os.environ


def test_phase3_live_gate_stops_after_one_image_and_persists_partial_resume(
    tmp_path,
    monkeypatch,
    canonical_run_contract,
):
    projected, _contract = canonical_run_contract(
        tmp_path,
        {
            "characters": [
                {
                    "id": "agent",
                    "name": "Agent",
                    "description": "fictional adult",
                },
                {
                    "id": "guard",
                    "name": "Guard",
                    "description": "fictional adult guard",
                },
            ]
        },
    )
    image_calls = []
    review_calls = []
    guarded_requests = []

    class ImageClient:
        def __init__(self, model):
            self.model = model

        def text_to_image(self, *, prompt, output_path, size):
            image_calls.append(("text", prompt, Path(output_path).name, size))
            Image.effect_noise((512, 512), 64).convert("RGB").save(output_path)

        def image_to_image(self, **_kwargs):
            output_path = _kwargs["output_path"]
            image_calls.append(
                (
                    "image",
                    _kwargs["prompt"],
                    Path(output_path).name,
                    _kwargs["size"],
                )
            )
            Image.effect_noise((512, 512), 64).convert("RGB").save(output_path)

    class ReviewClient:
        model = "fixture-vlm"

        def review(self, paths, _prompt):
            review_calls.append(list(paths))
            views = {}
            for path in paths:
                view_name = Path(path).stem
                views[view_name] = {
                    "passed": True,
                    "view_match": True,
                    "framing_match": True,
                    "neutral_pose": True,
                    "hands_empty": True,
                    "plain_background": True,
                    "single_character": True,
                    "face_visible": view_name != "back",
                    "both_eyes_visible": view_name not in {"side", "back"},
                    "declared_identity_match": True,
                    "declared_outfit_match": True,
                    "semantic_confidence": 0.9,
                    "semantic_evidence": [
                        "visible fictional identity reference"
                    ],
                    "issues": [],
                }
            return json.dumps({
                "views": views,
                "cross_view": {
                    "passed": True,
                    "identity_consistent": True,
                    "outfit_consistent": True,
                    "body_proportions_consistent": True,
                    "semantic_confidence": 0.9,
                    "semantic_evidence": ["single-view bootstrap gate"],
                    "issues": [],
                },
                "failed_views": [],
                "summary": "bootstrap view passed",
            })

    monkeypatch.setattr(
        "phases.phase3.character_factory.SeedreamClient",
        ImageClient,
    )
    monkeypatch.setattr(
        "clients.ark_multimodal_client.ArkMultimodalClient",
        ReviewClient,
    )

    result = run_phase3(
        tmp_path,
        projected,
        dry_run=False,
        _acceptance_max_new_image_requests=1,
        _acceptance_disable_provider_retries=True,
        _acceptance_before_provider_request=guarded_requests.append,
    )

    assert result["status"] == "acceptance_gate_passed"
    assert result["gate"] == "first_character_identity_image"
    assert result["image_provider_request_count"] == 1
    assert result["qa_provider_request_count"] == 1
    assert result["qa_verdict"] in {"pass", "acceptable_deviation"}
    assert len(image_calls) == 1
    assert len(review_calls) == 1
    assert [item["provider_family"] for item in guarded_requests] == [
        "seedream_image",
        "multimodal_observation",
    ]
    assert all(item.get("prompt_sha256") for item in guarded_requests)
    pending = json.loads(
        (tmp_path / "characters/agent/character_reference_qa.json").read_text()
    )
    assert pending["status"] == "pending"
    assert list(pending["inputs"]) == ["face_closeup"]
    assert not (tmp_path / "characters/guard").exists()

    resumed = run_phase3(
        tmp_path,
        projected,
        dry_run=False,
        _acceptance_disable_provider_retries=True,
    )

    assert resumed["status"] == "done"
    assert sum(call[0] == "text" for call in image_calls) == 2
    assert len(image_calls) == 8


def test_single_paid_request_guard_persists_uncertain_before_completion(
    tmp_path,
):
    guard = acceptance._SinglePaidRequestGuard(tmp_path, "phase3_identity")
    payload = {
        "provider_family": "seedream_image",
        "prompt_sha256": "a" * 64,
    }

    guard(payload)

    uncertain = json.loads(guard.path.read_text(encoding="utf-8"))
    assert uncertain["status"] == "submission_uncertain"
    assert uncertain["events"][-1]["event"] == "SubmissionAttempted"
    assert uncertain["zero_submit_preflight"]["status"] == "passed"
    with pytest.raises(RuntimeError, match="one-request hard limit"):
        guard(payload)

    completed = guard.complete(outcome={"asset_sha256": "b" * 64})
    assert completed["status"] == "provider_completed"
    assert completed["provider_request_count"] == 1


def test_recovery_matrix_uses_completed_resume_without_provider_calls(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "run"
    (workspace / "input").mkdir(parents=True)
    (workspace / "input/story.txt").write_text("story", encoding="utf-8")
    (workspace / "polished.mp4").write_bytes(b"final-video")
    final_sha256 = acceptance._sha256(workspace / "polished.mp4")
    lifecycle_calls = []

    monkeypatch.setattr(
        "utils.artifact_chain.can_resume_from",
        lambda _phase, _workspace: True,
    )

    def fake_resume(**kwargs):
        lifecycle_calls.append(kwargs)
        assert kwargs["resume"] is True
        assert kwargs["_force_sequential"] is True
        return {"status": "completed"}

    monkeypatch.setattr(acceptance.pipeline_lifecycle, "_run_pipeline", fake_resume)

    receipt = acceptance.verify_completed_recovery_matrix(
        workspace,
        final_video_sha256=final_sha256,
    )

    assert receipt["status"] == "passed"
    assert receipt["provider_request_count"] == 0
    assert [row["boundary"] for row in receipt["boundaries"]] == [
        "phase1",
        "phase3",
        "phase5",
        "phase6",
        "phase8",
    ]
    assert len(lifecycle_calls) == 5
    snapshot = Path(receipt["source_snapshot"])
    assert snapshot.stat().st_mode & stat.S_IWUSR == 0
