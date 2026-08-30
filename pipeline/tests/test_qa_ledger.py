from __future__ import annotations

import sqlite3

import pytest
from PIL import Image

from phases.phase8.frame_analysis import _automatic_semantic_reviewer
from quality.visual_qa_policy import decide_visual_qa, policy_sha256
from runtime.qa_ledger import QALedger, observation_fingerprint


def _fingerprint() -> str:
    return observation_fingerprint(
        evidence=[{"path": "board.png", "sha256": "a" * 64}],
        canonical_contract_sha256="b" * 64,
        evaluator_model="vlm-test",
        prompt_sha256="c" * 64,
        observation_schema="honcut.qa.test.v1",
    )


def test_identical_observation_is_recorded_once_across_ten_resumes(tmp_path):
    ledger = QALedger(tmp_path / "runtime.db")
    observations = []
    for _ in range(10):
        observation, reused = ledger.record_observation(
            run_id="run-1",
            phase="phase5",
            resource_id="S01",
            evidence_fingerprint=_fingerprint(),
            canonical_contract_sha256="b" * 64,
            evaluator_model="vlm-test",
            prompt_sha256="c" * 64,
            observation_schema="honcut.qa.test.v1",
            observation={"semantic_score": 0.72, "findings": []},
        )
        observations.append(observation.observation_id)
        assert reused is (_ > 0)
    assert len(set(observations)) == 1
    assert observations[0] == observations[-1]
    assert len(observations[0]) == 64
    assert ledger.counts() == {"observations": 1, "decisions": 0}


def test_policy_change_reuses_observation_and_appends_superseding_decision(tmp_path):
    ledger = QALedger(tmp_path / "runtime.db")
    observation, _ = ledger.record_observation(
        run_id="run-1",
        phase="phase5",
        resource_id="S01",
        evidence_fingerprint=_fingerprint(),
        canonical_contract_sha256="b" * 64,
        evaluator_model="vlm-test",
        prompt_sha256="c" * 64,
        observation_schema="honcut.qa.test.v1",
        observation={"semantic_score": 0.72, "findings": []},
    )
    first, _ = ledger.record_decision(
        observation_id=observation.observation_id,
        phase_owner="phase5.storyboard_qa",
        policy_id="policy-v1",
        policy_sha256="d" * 64,
        verdict="pass",
        semantic_score=0.72,
        decision={"reason": "first"},
    )
    second, reused = ledger.record_decision(
        observation_id=observation.observation_id,
        phase_owner="phase5.storyboard_qa",
        policy_id="policy-v2",
        policy_sha256="e" * 64,
        verdict="acceptable_deviation",
        semantic_score=0.72,
        decision={"reason": "updated tolerance"},
    )
    assert reused is False
    assert second.supersedes == first.decision_id
    assert len(first.decision_id) == 64
    assert len(second.decision_id) == 64
    assert ledger.counts() == {"observations": 1, "decisions": 2}


def test_ledger_tables_are_append_only(tmp_path):
    database = tmp_path / "runtime.db"
    ledger = QALedger(database)
    observation, _ = ledger.record_observation(
        run_id="run-1",
        phase="phase5",
        resource_id="S01",
        evidence_fingerprint=_fingerprint(),
        canonical_contract_sha256="b" * 64,
        evaluator_model="vlm-test",
        prompt_sha256="c" * 64,
        observation_schema="honcut.qa.test.v1",
        observation={"semantic_score": 0.72},
    )
    with sqlite3.connect(database) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE qa_observations SET phase = 'other' WHERE observation_id = ?",
            (observation.observation_id,),
        )


def test_visual_policy_allows_65_percent_and_blocks_only_strong_evidence():
    accepted = decide_visual_qa(
        semantic_score=0.65,
        findings=[{
            "blocking_category": "identity",
            "confidence": 0.84,
            "evidence": "slight face variation",
        }],
    )
    assert accepted.verdict == "acceptable_deviation"

    blocked = decide_visual_qa(
        semantic_score=0.9,
        findings=[{
            "blocking_category": "identity",
            "confidence": 0.85,
            "evidence": "different character in G04",
        }],
    )
    assert blocked.verdict == "block"

    strict = decide_visual_qa(
        semantic_score=1.0,
        findings=[],
        deterministic_errors=[{"category": "artifact_hash", "evidence": "mismatch"}],
    )
    assert strict.verdict == "block"
    assert policy_sha256() == policy_sha256()


def test_phase8_reuses_observation_and_accepts_low_confidence_negative(
    tmp_path,
    monkeypatch,
    canonical_run_contract,
):
    canonical_run_contract(tmp_path, {"characters": []})
    frames = []
    for index in range(2):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (640, 360), (40 + index, 50, 60)).save(path)
        frames.append(path)

    class Reviewer:
        model = "vlm-ledger-test"
        calls = 0

        def review(self, _paths, _prompt):
            self.calls += 1
            return (
                '{"verdict":"reshoot","issues":["minor pose ambiguity"],'
                '"confidence":0.80}'
            )

    client = Reviewer()
    monkeypatch.setattr(
        "clients.ark_multimodal_client.ArkMultimodalClient",
        lambda: client,
    )
    reviewer = _automatic_semantic_reviewer(tmp_path)
    assert reviewer is not None
    first = reviewer(frames, {"shot_id": "S01"})
    second = reviewer(frames, {"shot_id": "S01"})

    assert first["verdict"] == "pass"
    assert first["qa_verdict"] == "acceptable_deviation"
    assert second["qa_observation_reused"] is True
    assert client.calls == 1
    assert QALedger(tmp_path / "runtime.db").counts() == {
        "observations": 1,
        "decisions": 1,
    }
