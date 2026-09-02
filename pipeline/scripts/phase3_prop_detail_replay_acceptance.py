#!/usr/bin/env python3
"""Provider-deny replay for preserved Phase 3 prop-detail evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from quality.character_reference_qa import (
    PROP_DETAIL_OBSERVATION_SCHEMA,
    build_identity_detail_qa_prompt,
    evaluate_identity_detail_observation,
    file_sha256,
    resolve_identity_detail_logical_items,
)
from quality.visual_qa_policy import policy_sha256
from runtime.provider_attempt_policy import provider_attempt_scope
from utils.canonical_visual_contracts import load_canonical_visual_contract
from utils.provider_request_guard import (
    MediaUploadTimeouts,
    media_upload_guard_scope,
)


MANIFEST_SCHEMA = "honcut.prop-detail-preserved-evidence-manifest.v1"
REGRESSION_SCHEMA = "honcut.phase3-visual-qa-regression.v1"
REPLAY_SCHEMA = "honcut.prop-detail-replay-acceptance.v1"
REPLAY_RECEIPT = "phase3_prop_detail_replay_acceptance.json"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence must be an object: {path.name}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("preserved evidence path escapes the source run") from exc
    return resolved


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _deny_timeouts(_payload_bytes: int) -> MediaUploadTimeouts:
    return MediaUploadTimeouts(1, 1, 1, 1, 1)


class _ProviderDeny:
    """Reject every authoritative Provider family before network submission."""

    def __init__(self) -> None:
        self.attempted_families: list[str] = []

    def before_request(self, metadata: dict[str, Any]) -> None:
        family = str(metadata.get("provider_family") or "unknown_provider")
        self.attempted_families.append(family)
        raise RuntimeError(f"provider-deny replay reached {family}")

    def prepare_upload(self, _metadata: dict[str, Any]) -> tuple[None, str]:
        self.attempted_families.append("tos_media_upload")
        raise RuntimeError("provider-deny replay reached tos_media_upload")


def _validated_artifacts(
    source_run: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("preserved evidence manifest has no artifacts")
    expected = {
        "run_manifest",
        "failed_acceptance",
        "canonical_contract",
        "face_reference",
        "body_reference",
        "prop_detail_board",
        "legacy_qa_receipt",
    }
    if set(artifacts) != expected:
        raise RuntimeError("preserved evidence manifest artifact set is invalid")
    paths: dict[str, Path] = {}
    for role, record in artifacts.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"preserved {role} record is invalid")
        path = _inside(source_run / str(record.get("path") or ""), source_run)
        if not path.is_file() or record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"preserved {role} hash mismatch")
        paths[role] = path
    return paths


def run_replay(
    *,
    source_run: Path,
    evidence_manifest_path: Path,
    observation_fixture_path: Path,
    output_dir: Path,
    candidate_commit: str,
    regression_receipt_path: Path,
) -> dict[str, Any]:
    """Evaluate preserved evidence repeatedly without touching the source run."""
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_run or source_run in output_dir.parents:
        raise RuntimeError("replay output must be outside the historical source run")
    manifest = _read_object(evidence_manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported preserved evidence manifest schema")
    artifacts = _validated_artifacts(source_run, manifest)

    source_manifest = _read_object(artifacts["run_manifest"])
    failed_acceptance = _read_object(artifacts["failed_acceptance"])
    legacy_qa = _read_object(artifacts["legacy_qa_receipt"])
    if (
        source_run.name != manifest.get("source_run_id")
        or source_manifest.get("run_fingerprint")
        != manifest.get("source_run_fingerprint")
        or failed_acceptance.get("status") != "live_acceptance_failed"
        or manifest.get("historical_status") != "live_acceptance_failed"
        or failed_acceptance.get("source", {}).get("git_commit")
        != manifest.get("source_git_commit")
        or legacy_qa.get("schema") != "honcut.prop-detail-board-qa.v1"
        or legacy_qa.get("status") != "failed"
        or legacy_qa.get("character_id") != manifest.get("character_id")
    ):
        raise RuntimeError("historical run identity or failure status mismatch")

    contract = load_canonical_visual_contract(source_run)
    if contract.get("contract_sha256") != manifest.get("canonical_contract_sha256"):
        raise RuntimeError("preserved canonical contract identity mismatch")
    legacy_inputs = legacy_qa.get("inputs") or {}
    legacy_references = legacy_inputs.get("canonical_references")
    if not isinstance(legacy_references, list) or len(legacy_references) != 2:
        raise RuntimeError("legacy canonical reference lineage is incomplete")
    expected_legacy_hashes = {
        "face_reference": legacy_references[0],
        "body_reference": legacy_references[1],
        "prop_detail_board": legacy_inputs.get("prop_detail_board") or {},
    }
    for role, record in expected_legacy_hashes.items():
        if record.get("sha256") != manifest["artifacts"][role]["sha256"]:
            raise RuntimeError(f"legacy {role} lineage mismatch")

    canonical_hash, logical_items = resolve_identity_detail_logical_items(
        source_run,
        str(manifest["character_id"]),
        legacy_qa.get("identity_props") or [],
    )
    if canonical_hash != contract["contract_sha256"]:
        raise RuntimeError("resolved canonical contract hash mismatch")
    observation_fixture = _read_object(observation_fixture_path)
    if observation_fixture.get("schema") != PROP_DETAIL_OBSERVATION_SCHEMA:
        raise RuntimeError("replay Observation fixture schema is not current")

    regression = _read_object(regression_receipt_path)
    if (
        regression.get("schema") != REGRESSION_SCHEMA
        or regression.get("status") != "passed"
        or regression.get("candidate_commit") != candidate_commit
    ):
        raise RuntimeError("replay regression receipt is not current")

    prompt = build_identity_detail_qa_prompt(logical_items)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    evidence = [
        {
            "role": role,
            "path": manifest["artifacts"][role]["path"],
            "sha256": manifest["artifacts"][role]["sha256"],
        }
        for role in ("face_reference", "body_reference", "prop_detail_board")
    ]
    deny = _ProviderDeny()
    with ExitStack() as stack:
        stack.enter_context(provider_attempt_scope(
            max_retries=0,
            before_provider_request=deny.before_request,
        ))
        stack.enter_context(media_upload_guard_scope(
            timeout_resolver=_deny_timeouts,
            prepare_upload=deny.prepare_upload,
        ))
        result = evaluate_identity_detail_observation(
            output_dir=output_dir,
            character_id=str(manifest["character_id"]),
            evidence=evidence,
            canonical_contract_sha256=contract["contract_sha256"],
            evaluator_model="stored-typed-evidence-fixture-v1",
            prompt_sha256=prompt_hash,
            logical_items=logical_items,
            observation_payload=observation_fixture,
            run_id=f"{manifest['source_run_fingerprint']}:provider-deny-replay",
        )
    if deny.attempted_families:
        raise RuntimeError(
            "provider-deny replay attempted submission: "
            + ",".join(deny.attempted_families)
        )

    stable = {
        "schema": REPLAY_SCHEMA,
        "status": "passed" if result["passed"] else result["qa_verdict"],
        "candidate_commit": candidate_commit,
        "regression_receipt": {
            "path": regression_receipt_path.name,
            "sha256": file_sha256(regression_receipt_path),
        },
        "source": {
            "run_id": manifest["source_run_id"],
            "run_fingerprint": manifest["source_run_fingerprint"],
            "historical_status": "live_acceptance_failed",
            "source_git_commit": manifest["source_git_commit"],
            "manifest_sha256": file_sha256(evidence_manifest_path),
            "observation_fixture_sha256": file_sha256(observation_fixture_path),
            "artifacts": manifest["artifacts"],
        },
        "canonical_contract_sha256": contract["contract_sha256"],
        "logical_item_ids": [item["logical_item_id"] for item in logical_items],
        "policy_sha256": policy_sha256(),
        "qa_verdict": result["qa_verdict"],
        "qa_observation_id": result["qa_observation_id"],
        "qa_decision_id": result["qa_decision_id"],
        "provider_request_count": 0,
        "provider_submission_count": 0,
        "automatic_provider_corrections": False,
        "historical_run_mutated": False,
    }
    receipt_path = output_dir / REPLAY_RECEIPT
    if receipt_path.exists():
        existing = _read_object(receipt_path)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        if comparable != stable:
            raise RuntimeError("existing replay receipt disagrees with stable evidence")
        return existing
    receipt = {**stable, "created_at": datetime.now(UTC).isoformat()}
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Phase 3 prop-detail evidence with Provider submissions denied"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--observation-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_replay(
        source_run=args.source_run,
        evidence_manifest_path=args.evidence_manifest,
        observation_fixture_path=args.observation_fixture,
        output_dir=args.output_dir,
        candidate_commit=args.candidate_commit,
        regression_receipt_path=args.regression_receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
