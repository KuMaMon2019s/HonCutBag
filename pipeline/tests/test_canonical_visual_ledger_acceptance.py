from __future__ import annotations

import json
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


def _write_inputs(workspace):
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True)
    story = input_dir / "story.txt"
    story.write_text("原创人物进入车站。人物观察手中装置。", encoding="utf-8")
    expectations = workspace / "acceptance_expectations.json"
    expectations.write_text(
        json.dumps({
            "schema": "honcut.full-chain-acceptance-expectations.v1",
            "expected_duration_s": 36,
            "expected_character_entities": 1,
            "expected_character_instances": 1,
            "required_events": ["enter", "inspect"],
            "visual_facts": {"character": ["fictional"]},
        }),
        encoding="utf-8",
    )
    return story, expectations


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
    assert receipt["configuration"]["automatic_reshoot"] is False
    assert receipt["configuration"]["character_library_configured"] is False


def test_stage0_preflight_rejects_input_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "story.txt"
    outside.write_text("story", encoding="utf-8")
    expectations = workspace / "acceptance_expectations.json"
    expectations.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes workspace"):
        acceptance.build_stage0_preflight(workspace, outside, expectations)


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
