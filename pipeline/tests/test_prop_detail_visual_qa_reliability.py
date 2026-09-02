"""Stop-loss coverage for Phase 3 logical prop-detail QA."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase3 import character_factory
from quality.character_reference_qa import (
    CharacterReferenceQAError,
    PROP_DETAIL_OBSERVATION_SCHEMA,
    build_identity_detail_input_contract,
    file_sha256,
    parse_identity_detail_qa,
    resolve_identity_detail_logical_items,
    review_identity_detail_reference,
)
from quality.visual_qa_policy import POLICY_ID, policy_sha256
from runtime.qa_ledger import QALedger
from pipeline.scripts.phase3_prop_detail_replay_acceptance import (
    MANIFEST_SCHEMA,
    REGRESSION_SCHEMA,
    _ProviderDeny,
    _deny_timeouts,
    run_replay,
)
from runtime.provider_attempt_policy import provider_attempt_scope
from utils.provider_request_guard import (
    media_upload_guard_scope,
    media_upload_prepared,
    provider_request_started,
)


def _identity_props() -> list[dict]:
    return [{
        "id": "camera_a",
        "name": "camera",
        "description": "black rectangular body, one silver lens, red wrist strap",
        "attachment_mode": "isolated_handheld",
        "persistence": "role_active",
        "reference_required": True,
    }]


def _observation(
    *,
    item_id: str = "camera_a",
    aggregate_passed: bool = True,
    item_confidence: float = 0.95,
    topology_consistent: bool = True,
    undeclared: bool = False,
    undeclared_confidence: float = 0.95,
) -> dict:
    return {
        "schema": PROP_DETAIL_OBSERVATION_SCHEMA,
        "passed": aggregate_passed,
        "character_identity_consistent": True,
        "character_identity_confidence": 0.95,
        "character_identity_evidence": [
            "face and outfit match the two canonical references"
        ],
        "items": [{
            "logical_item_id": item_id,
            "logical_identity_present": True,
            "depiction_count": 3,
            "depictions_mutually_consistent": True,
            "topology_consistent": topology_consistent,
            "colors_materials_consistent": True,
            "attachment_mode_correct": True,
            "undeclared_logical_item_evidence": [],
            "semantic_confidence": item_confidence,
            "semantic_evidence": [
                "front, side and three-quarter depictions share one body, lens and strap"
            ],
            "issues": [],
        }],
        "no_undeclared_logical_items": not undeclared,
        "undeclared_items_confidence": undeclared_confidence,
        "undeclared_items_evidence": (
            ["a separate blue cylindrical tool appears beside the declared camera"]
            if undeclared
            else ["only the declared camera identity is depicted"]
        ),
        "issues": [],
    }


def _run_inputs(tmp_path: Path, canonical_run_contract):
    props = _identity_props()
    canonical_run_contract(
        tmp_path,
        {"characters": [{
            "id": "photographer",
            "name": "photographer",
            "description": "navy helmet and beige vest",
            "appearance": {"identity_props": props},
        }]},
    )
    character_dir = tmp_path / "characters" / "photographer"
    character_dir.mkdir(parents=True)
    canonical_paths = [
        character_dir / "face_closeup.png",
        character_dir / "full_body.png",
    ]
    for index, path in enumerate(canonical_paths, start=1):
        Image.new("RGB", (32, 32), (index * 30, 50, 70)).save(path)
    detail_path = character_dir / "prop_detail_board.png"
    Image.new("RGB", (32, 32), (80, 100, 120)).save(detail_path)
    canonical_hash, logical_items = resolve_identity_detail_logical_items(
        tmp_path,
        "photographer",
        props,
    )
    input_contract = build_identity_detail_input_contract(
        char_id="photographer",
        character_description="navy helmet and beige vest",
        identity_props=props,
        canonical_contract_sha256=canonical_hash,
        logical_items=logical_items,
        prompt_sha256="a" * 64,
        canonical_paths=canonical_paths,
    )
    return props, character_dir, canonical_paths, detail_path, input_contract


class _Reviewer:
    model = "fixture-vlm"

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def review(self, _paths, _prompt):
        self.calls += 1
        return json.dumps(self.payload)


def test_three_depictions_remain_one_logical_item_even_if_aggregate_fails():
    parsed = parse_identity_detail_qa(
        json.dumps(_observation(aggregate_passed=False)),
        [{"logical_item_id": "camera_a"}],
    )

    assert parsed["model_passed_diagnostic"] is False
    assert parsed["qa_verdict"] == "pass"
    assert parsed["items"][0]["depiction_count"] == 3


def test_distinct_undeclared_prop_requires_high_confidence_evidence():
    blocked = parse_identity_detail_qa(
        json.dumps(_observation(undeclared=True, undeclared_confidence=0.91)),
        [{"logical_item_id": "camera_a"}],
    )
    tolerated = parse_identity_detail_qa(
        json.dumps(_observation(undeclared=True, undeclared_confidence=0.72)),
        [{"logical_item_id": "camera_a"}],
    )

    assert blocked["qa_verdict"] == "block"
    assert blocked["policy_decision"]["blocking_categories"] == [
        "undeclared_logical_item"
    ]
    assert tolerated["qa_verdict"] == "acceptable_deviation"


@pytest.mark.parametrize("item_id", ["", "invented_prop"])
def test_missing_or_invented_logical_item_id_blocks_deterministically(item_id):
    parsed = parse_identity_detail_qa(
        json.dumps(_observation(item_id=item_id)),
        [{"logical_item_id": "camera_a"}],
    )

    assert parsed["qa_verdict"] == "block"
    assert parsed["policy_decision"]["blocking_categories"] == [
        "canonical_contract"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("items"),
        lambda value: value.__setitem__(
            "schema", "honcut.prop-detail-observation.v999"
        ),
    ],
)
def test_malformed_or_future_observation_schema_fails_closed(mutation):
    payload = _observation()
    mutation(payload)

    with pytest.raises(ValidationError):
        parse_identity_detail_qa(
            json.dumps(payload),
            [{"logical_item_id": "camera_a"}],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: contract.__setitem__("logical_items_sha256", "0" * 64),
        lambda contract: contract["canonical_references"][0].__setitem__(
            "media_role", "prop_geometry"
        ),
        lambda contract: contract.__setitem__("parent_lineage_sha256", "0" * 64),
        lambda contract: contract.__setitem__(
            "canonical_visual_contract_sha256", "0" * 64
        ),
    ],
)
def test_contract_hash_role_and_lineage_errors_block_before_review(
    tmp_path,
    canonical_run_contract,
    mutate,
):
    _, _, canonical_paths, detail_path, input_contract = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )
    reviewer = _Reviewer(_observation())
    corrupted = copy.deepcopy(input_contract)
    mutate(corrupted)

    with pytest.raises(CharacterReferenceQAError):
        review_identity_detail_reference(
            reviewer,
            canonical_paths,
            detail_path,
            corrupted,
        )
    assert reviewer.calls == 0


def test_ledger_reuses_observation_and_policy_change_only_adds_decision(
    tmp_path,
    canonical_run_contract,
):
    _, _, canonical_paths, detail_path, input_contract = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )
    reviewer = _Reviewer(_observation())

    first = review_identity_detail_reference(
        reviewer,
        canonical_paths,
        detail_path,
        input_contract,
    )
    recovered = [
        review_identity_detail_reference(
            reviewer,
            canonical_paths,
            detail_path,
            input_contract,
        )
        for _ in range(10)
    ]

    assert reviewer.calls == 1
    assert {value["qa_observation_id"] for value in recovered} == {
        first["qa_observation_id"]
    }
    assert {value["qa_decision_id"] for value in recovered} == {
        first["qa_decision_id"]
    }
    ledger = QALedger(tmp_path / "runtime.db")
    superseding, reused = ledger.record_decision(
        observation_id=first["qa_observation_id"],
        phase_owner="phase3.prop_detail_qa",
        policy_id=f"{POLICY_ID}.fixture",
        policy_sha256="f" * 64,
        verdict="pass",
        semantic_score=0.95,
        decision={"fixture_policy": True},
    )
    assert reused is False
    assert superseding.supersedes == first["qa_decision_id"]
    assert reviewer.calls == 1

    with sqlite3.connect(tmp_path / "runtime.db") as connection:
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM qa_observations"
        ).fetchone()[0]
        decision_count = connection.execute(
            "SELECT COUNT(*) FROM qa_decisions"
        ).fetchone()[0]
    assert observation_count == 1
    assert decision_count == 2


def test_manual_review_stops_without_redraw_or_re_review(
    tmp_path,
    canonical_run_contract,
):
    props, character_dir, canonical_paths, detail_path, _ = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )

    class ImageClient:
        def __init__(self):
            self.calls = 0

        def image_to_image(self, **kwargs):
            self.calls += 1
            Image.new("RGB", (32, 32), (20, 40, 60)).save(kwargs["output_path"])

    image_client = ImageClient()
    reviewer = _Reviewer(_observation(item_confidence=0.40))

    with pytest.raises(CharacterReferenceQAError, match="manual review"):
        character_factory._quality_control_identity_detail(
            char_id="photographer",
            character_description="navy helmet and beige vest",
            identity_props=props,
            style="",
            char_dir=character_dir,
            canonical_paths=canonical_paths,
            detail_path=detail_path,
            image_client=image_client,
            review_client=reviewer,
            max_retries=2,
        )
    assert image_client.calls == 1
    assert reviewer.calls == 1
    receipt_path = character_dir / "prop_detail_board_qa_v2.json"
    receipt_hash = file_sha256(receipt_path)

    with pytest.raises(CharacterReferenceQAError, match="automatic re-review"):
        character_factory._quality_control_identity_detail(
            char_id="photographer",
            character_description="navy helmet and beige vest",
            identity_props=props,
            style="",
            char_dir=character_dir,
            canonical_paths=canonical_paths,
            detail_path=detail_path,
            image_client=image_client,
            review_client=reviewer,
            max_retries=2,
        )
    assert image_client.calls == 1
    assert reviewer.calls == 1
    assert file_sha256(receipt_path) == receipt_hash


def test_prop_detail_retry_budget_overflow_fails_before_any_provider(
    tmp_path,
    canonical_run_contract,
):
    props, character_dir, canonical_paths, detail_path, _ = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )
    reviewer = _Reviewer(_observation())

    class ImageClient:
        calls = 0

        def image_to_image(self, **_kwargs):
            self.calls += 1

    image_client = ImageClient()
    with pytest.raises(ValueError, match="retries must be between 0 and 2"):
        character_factory._quality_control_identity_detail(
            char_id="photographer",
            character_description="navy helmet and beige vest",
            identity_props=props,
            style="",
            char_dir=character_dir,
            canonical_paths=canonical_paths,
            detail_path=detail_path,
            image_client=image_client,
            review_client=reviewer,
            max_retries=3,
        )
    assert image_client.calls == 0
    assert reviewer.calls == 0


def test_future_prop_detail_receipt_fails_before_any_provider(
    tmp_path,
    canonical_run_contract,
):
    props, character_dir, canonical_paths, detail_path, _ = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )
    (character_dir / "prop_detail_board_qa_v2.json").write_text(
        json.dumps({"schema": "honcut.prop-detail-board-qa.v999"}),
        encoding="utf-8",
    )
    reviewer = _Reviewer(_observation())

    class ImageClient:
        calls = 0

        def image_to_image(self, **_kwargs):
            self.calls += 1

    image_client = ImageClient()
    with pytest.raises(CharacterReferenceQAError, match="schema is unsupported"):
        character_factory._quality_control_identity_detail(
            char_id="photographer",
            character_description="navy helmet and beige vest",
            identity_props=props,
            style="",
            char_dir=character_dir,
            canonical_paths=canonical_paths,
            detail_path=detail_path,
            image_client=image_client,
            review_client=reviewer,
            max_retries=0,
        )
    assert image_client.calls == 0
    assert reviewer.calls == 0


def test_legacy_receipt_is_immutable_audit_evidence(
    tmp_path,
    canonical_run_contract,
):
    props, character_dir, canonical_paths, detail_path, _ = _run_inputs(
        tmp_path,
        canonical_run_contract,
    )
    legacy_path = character_dir / "prop_detail_board_qa.json"
    _write_json(legacy_path, {
        "schema": "honcut.prop-detail-board-qa.v1",
        "status": "failed",
        "inputs": {
            "prop_detail_board": {
                "path": detail_path.name,
                "sha256": file_sha256(detail_path),
            },
        },
    })
    legacy_hash = file_sha256(legacy_path)
    reviewer = _Reviewer(_observation())

    class ImageClient:
        calls = 0

        def image_to_image(self, **_kwargs):
            self.calls += 1

    image_client = ImageClient()
    receipt = character_factory._quality_control_identity_detail(
        char_id="photographer",
        character_description="navy helmet and beige vest",
        identity_props=props,
        style="",
        char_dir=character_dir,
        canonical_paths=canonical_paths,
        detail_path=detail_path,
        image_client=image_client,
        review_client=reviewer,
        max_retries=0,
    )

    assert receipt["status"] == "passed"
    assert receipt["legacy_evidence"]["status"] == "audit_only"
    assert image_client.calls == 0
    assert reviewer.calls == 1
    assert file_sha256(legacy_path) == legacy_hash


def test_policy_identity_is_stable():
    assert POLICY_ID == "honcut.visual-qa-policy.v1"
    assert len(policy_sha256()) == 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _replay_source(tmp_path: Path, canonical_run_contract):
    source = tmp_path / "source-run"
    props = _identity_props()
    _, contract = canonical_run_contract(
        source,
        {"characters": [{
            "id": "photographer",
            "name": "photographer",
            "description": "navy helmet and beige vest",
            "appearance": {"identity_props": props},
        }]},
    )
    character_dir = source / "characters" / "photographer"
    character_dir.mkdir(parents=True)
    for index, name in enumerate(
        ("face_closeup.png", "full_body.png", "prop_detail_board.png"),
        start=1,
    ):
        Image.new("RGB", (32, 32), (index * 30, 50, 70)).save(
            character_dir / name
        )
    run_fingerprint = "1" * 64
    source_commit = "2" * 40
    _write_json(source / "RUN_MANIFEST.json", {
        "run_fingerprint": run_fingerprint,
    })
    _write_json(source / "canonical_visual_ledger_36s_acceptance.json", {
        "status": "live_acceptance_failed",
        "source": {"git_commit": source_commit},
    })
    references = [
        {"path": name, "sha256": file_sha256(character_dir / name)}
        for name in ("face_closeup.png", "full_body.png")
    ]
    _write_json(character_dir / "prop_detail_board_qa.json", {
        "schema": "honcut.prop-detail-board-qa.v1",
        "status": "failed",
        "character_id": "photographer",
        "identity_props": props,
        "inputs": {
            "canonical_references": references,
            "prop_detail_board": {
                "path": "prop_detail_board.png",
                "sha256": file_sha256(character_dir / "prop_detail_board.png"),
            },
        },
    })
    role_paths = {
        "run_manifest": source / "RUN_MANIFEST.json",
        "failed_acceptance": source / "canonical_visual_ledger_36s_acceptance.json",
        "canonical_contract": source / "CANONICAL_VISUAL_CONTRACT.json",
        "face_reference": character_dir / "face_closeup.png",
        "body_reference": character_dir / "full_body.png",
        "prop_detail_board": character_dir / "prop_detail_board.png",
        "legacy_qa_receipt": character_dir / "prop_detail_board_qa.json",
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_run_id": source.name,
        "source_run_fingerprint": run_fingerprint,
        "source_git_commit": source_commit,
        "historical_status": "live_acceptance_failed",
        "character_id": "photographer",
        "canonical_contract_sha256": contract["contract_sha256"],
        "artifacts": {
            role: {
                "path": path.relative_to(source).as_posix(),
                "sha256": file_sha256(path),
            }
            for role, path in role_paths.items()
        },
    }
    manifest_path = tmp_path / "manifest.json"
    observation_path = tmp_path / "observation.json"
    regression_path = tmp_path / "regression.json"
    candidate_commit = "3" * 40
    _write_json(manifest_path, manifest)
    _write_json(observation_path, _observation(aggregate_passed=False))
    _write_json(regression_path, {
        "schema": REGRESSION_SCHEMA,
        "status": "passed",
        "candidate_commit": candidate_commit,
    })
    return (
        source,
        role_paths,
        manifest_path,
        observation_path,
        regression_path,
        candidate_commit,
    )


def test_provider_deny_replay_is_stable_for_ten_recoveries(
    tmp_path,
    canonical_run_contract,
):
    (
        source,
        role_paths,
        manifest_path,
        observation_path,
        regression_path,
        candidate_commit,
    ) = _replay_source(tmp_path, canonical_run_contract)
    output_dir = tmp_path / "replay"
    before = {role: file_sha256(path) for role, path in role_paths.items()}

    receipts = [
        run_replay(
            source_run=source,
            evidence_manifest_path=manifest_path,
            observation_fixture_path=observation_path,
            output_dir=output_dir,
            candidate_commit=candidate_commit,
            regression_receipt_path=regression_path,
        )
        for _ in range(10)
    ]

    assert {receipt["qa_observation_id"] for receipt in receipts} == {
        receipts[0]["qa_observation_id"]
    }
    assert {receipt["qa_decision_id"] for receipt in receipts} == {
        receipts[0]["qa_decision_id"]
    }
    assert {receipt["provider_request_count"] for receipt in receipts} == {0}
    assert receipts[0]["qa_verdict"] == "pass"
    assert receipts[0]["source"]["historical_status"] == "live_acceptance_failed"
    assert {role: file_sha256(path) for role, path in role_paths.items()} == before
    with sqlite3.connect(output_dir / "runtime.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM qa_observations").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM qa_decisions").fetchone()[0] == 1


def test_replay_fails_closed_when_preserved_media_hash_changes(
    tmp_path,
    canonical_run_contract,
):
    (
        source,
        role_paths,
        manifest_path,
        observation_path,
        regression_path,
        candidate_commit,
    ) = _replay_source(tmp_path, canonical_run_contract)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(role_paths["prop_detail_board"])

    with pytest.raises(RuntimeError, match="prop_detail_board hash mismatch"):
        run_replay(
            source_run=source,
            evidence_manifest_path=manifest_path,
            observation_fixture_path=observation_path,
            output_dir=tmp_path / "replay",
            candidate_commit=candidate_commit,
            regression_receipt_path=regression_path,
        )


def test_provider_deny_guards_reject_transport_and_tos_boundaries():
    deny = _ProviderDeny()
    with provider_attempt_scope(
        max_retries=0,
        before_provider_request=deny.before_request,
    ), media_upload_guard_scope(
        timeout_resolver=_deny_timeouts,
        prepare_upload=deny.prepare_upload,
    ):
        with pytest.raises(RuntimeError, match="ark_text"):
            provider_request_started({"provider_family": "ark_text"})
        with pytest.raises(RuntimeError, match="tos_media_upload"):
            media_upload_prepared({"path_sha256": "a" * 64})
    assert deny.attempted_families == ["ark_text", "tos_media_upload"]
