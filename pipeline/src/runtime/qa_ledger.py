"""Append-only persistence for model observations and policy decisions.

The ledger deliberately separates probabilistic model output from the deterministic
policy decision owned by a HonCut phase.  It shares ``runtime.db`` with generation
tasks, but owns only its namespaced tables and never changes SQLite ``user_version``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


QAVerdict = Literal["pass", "acceptable_deviation", "block", "manual_review"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("QA ledger JSON must contain an object")
    return decoded


def _stable_id(namespace: str, *parts: str) -> str:
    material = _json({"namespace": namespace, "parts": list(parts)})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def observation_fingerprint(
    *,
    evidence: list[dict[str, Any]],
    canonical_contract_sha256: str,
    evaluator_model: str,
    prompt_sha256: str,
    observation_schema: str,
) -> str:
    """Hash every input that can change one probabilistic observation."""
    canonical = {
        "canonical_contract_sha256": canonical_contract_sha256,
        "evaluator_model": evaluator_model,
        "evidence": evidence,
        "observation_schema": observation_schema,
        "prompt_sha256": prompt_sha256,
    }
    return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QAObservation:
    observation_id: str
    run_id: str
    phase: str
    resource_id: str
    evidence_fingerprint: str
    canonical_contract_sha256: str
    evaluator_model: str
    prompt_sha256: str
    observation_schema: str
    observation: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> QAObservation:
        return cls(
            observation_id=row["observation_id"],
            run_id=row["run_id"],
            phase=row["phase"],
            resource_id=row["resource_id"],
            evidence_fingerprint=row["evidence_fingerprint"],
            canonical_contract_sha256=row["canonical_contract_sha256"],
            evaluator_model=row["evaluator_model"],
            prompt_sha256=row["prompt_sha256"],
            observation_schema=row["observation_schema"],
            observation=_object(row["observation_json"]),
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class QADecision:
    decision_id: str
    observation_id: str
    phase_owner: str
    policy_id: str
    policy_sha256: str
    verdict: QAVerdict
    semantic_score: float | None
    decision: dict[str, Any]
    supersedes: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> QADecision:
        return cls(
            decision_id=row["decision_id"],
            observation_id=row["observation_id"],
            phase_owner=row["phase_owner"],
            policy_id=row["policy_id"],
            policy_sha256=row["policy_sha256"],
            verdict=row["verdict"],
            semantic_score=row["semantic_score"],
            decision=_object(row["decision_json"]),
            supersedes=row["supersedes"],
            created_at=row["created_at"],
        )


class QALedger:
    """Own append-only QA observation and decision tables in one runtime DB."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS qa_observations (
                    observation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    evidence_fingerprint TEXT NOT NULL UNIQUE,
                    canonical_contract_sha256 TEXT NOT NULL,
                    evaluator_model TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    observation_schema TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_qa_observations_resource
                ON qa_observations(run_id, phase, resource_id, created_at);

                CREATE TABLE IF NOT EXISTS qa_decisions (
                    decision_id TEXT PRIMARY KEY,
                    observation_id TEXT NOT NULL,
                    phase_owner TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    verdict TEXT NOT NULL CHECK(verdict IN (
                        'pass', 'acceptable_deviation', 'block', 'manual_review'
                    )),
                    semantic_score REAL,
                    decision_json TEXT NOT NULL,
                    supersedes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(observation_id) REFERENCES qa_observations(observation_id),
                    FOREIGN KEY(supersedes) REFERENCES qa_decisions(decision_id),
                    UNIQUE(observation_id, phase_owner, policy_sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_qa_decisions_observation
                ON qa_decisions(observation_id, created_at);

                CREATE TRIGGER IF NOT EXISTS qa_observations_no_update
                BEFORE UPDATE ON qa_observations BEGIN
                    SELECT RAISE(ABORT, 'qa_observations is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS qa_observations_no_delete
                BEFORE DELETE ON qa_observations BEGIN
                    SELECT RAISE(ABORT, 'qa_observations is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS qa_decisions_no_update
                BEFORE UPDATE ON qa_decisions BEGIN
                    SELECT RAISE(ABORT, 'qa_decisions is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS qa_decisions_no_delete
                BEFORE DELETE ON qa_decisions BEGIN
                    SELECT RAISE(ABORT, 'qa_decisions is append-only');
                END;
                """
            )

    def find_observation(self, evidence_fingerprint: str) -> QAObservation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM qa_observations WHERE evidence_fingerprint = ?",
                (evidence_fingerprint,),
            ).fetchone()
        return QAObservation.from_row(row) if row is not None else None

    def record_observation(
        self,
        *,
        run_id: str,
        phase: str,
        resource_id: str,
        evidence_fingerprint: str,
        canonical_contract_sha256: str,
        evaluator_model: str,
        prompt_sha256: str,
        observation_schema: str,
        observation: dict[str, Any],
    ) -> tuple[QAObservation, bool]:
        """Insert once; return ``(observation, reused)`` for identical evidence."""
        existing = self.find_observation(evidence_fingerprint)
        if existing is not None:
            return existing, True
        observation_id = _stable_id("qa-observation", evidence_fingerprint)
        now = _utc_now()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO qa_observations (
                        observation_id, run_id, phase, resource_id,
                        evidence_fingerprint, canonical_contract_sha256,
                        evaluator_model, prompt_sha256, observation_schema,
                        observation_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        run_id,
                        phase,
                        resource_id,
                        evidence_fingerprint,
                        canonical_contract_sha256,
                        evaluator_model,
                        prompt_sha256,
                        observation_schema,
                        _json(observation),
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM qa_observations WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone()
                connection.commit()
        except sqlite3.IntegrityError:
            raced = self.find_observation(evidence_fingerprint)
            if raced is None:
                raise
            return raced, True
        if row is None:
            raise RuntimeError("QA observation disappeared after insert")
        return QAObservation.from_row(row), False

    def record_decision(
        self,
        *,
        observation_id: str,
        phase_owner: str,
        policy_id: str,
        policy_sha256: str,
        verdict: QAVerdict,
        decision: dict[str, Any],
        semantic_score: float | None = None,
    ) -> tuple[QADecision, bool]:
        """Append a policy decision; a new policy supersedes the prior decision."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM qa_decisions
                WHERE observation_id = ? AND phase_owner = ? AND policy_sha256 = ?
                """,
                (observation_id, phase_owner, policy_sha256),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return QADecision.from_row(existing), True
            previous = connection.execute(
                """
                SELECT decision_id FROM qa_decisions
                WHERE observation_id = ? AND phase_owner = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (observation_id, phase_owner),
            ).fetchone()
            supersedes = previous["decision_id"] if previous is not None else None
            decision_id = _stable_id(
                "qa-decision",
                observation_id,
                phase_owner,
                policy_sha256,
            )
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO qa_decisions (
                    decision_id, observation_id, phase_owner, policy_id,
                    policy_sha256, verdict, semantic_score, decision_json,
                    supersedes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    observation_id,
                    phase_owner,
                    policy_id,
                    policy_sha256,
                    verdict,
                    semantic_score,
                    _json(decision),
                    supersedes,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM qa_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("QA decision disappeared after insert")
        return QADecision.from_row(row), False

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            observations = int(
                connection.execute("SELECT COUNT(*) FROM qa_observations").fetchone()[0]
            )
            decisions = int(
                connection.execute("SELECT COUNT(*) FROM qa_decisions").fetchone()[0]
            )
        return {"observations": observations, "decisions": decisions}
