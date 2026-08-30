#!/usr/bin/env python3
"""One-request live acceptance for the Phase 1 Character Roster owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import canonical_visual_ledger_36s_acceptance as full_chain_acceptance
from phases.phase1.character_discoverer import discover_characters
from phases.phase1.character_roster import (
    CHARACTER_ROSTER_FILENAME,
    compile_character_roster,
    validate_character_roster,
)
from runtime.provider_attempt_policy import provider_attempt_scope
from utils.config import get_api_key, validate_config


INPUT_SCHEMA = "honcut.character-roster-live-input.v1"
RECEIPT_SCHEMA = "honcut.character-roster-live-acceptance.v1"
RECEIPT_NAME = "character_roster_live_acceptance.json"
LIVE_DIRECTORY = Path("live_acceptance") / "character_roster"
REQUEST_RECEIPT_NAME = "character_roster_observation_request.json"
_SAFE_TRANSPORT_FIELDS = frozenset({
    "error_type",
    "max_tokens",
    "messages_sha256",
    "model",
    "provider_family",
    "response_format_sha256",
    "response_sha256",
    "stream",
    "submission_outcome",
    "transport_status",
})


def _read_object(path: Path) -> dict[str, Any]:
    return full_chain_acceptance._read_object(path)


def _canonical_sha256(value: Any) -> str:
    return full_chain_acceptance._canonical_sha256(value)


def _safe_transport_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Allow only bounded transport metadata in persistent paid-call receipts."""
    return {
        key: value
        for key, value in payload.items()
        if key in _SAFE_TRANSPORT_FIELDS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }


def _load_events_contract(path: Path) -> tuple[list[dict[str, Any]], str]:
    document = _read_object(path)
    if document.get("schema") != INPUT_SCHEMA:
        raise RuntimeError("unsupported Character Roster live input schema")
    events = document.get("events")
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        raise RuntimeError("Character Roster live input events are invalid")
    unsigned = {"schema": INPUT_SCHEMA, "events": events}
    expected_hash = _canonical_sha256(unsigned)
    if document.get("events_sha256") != expected_hash:
        raise RuntimeError("Character Roster live input hash mismatch")
    return events, expected_hash


def _match_roster_expectations(
    roster: dict[str, Any],
    expectations: dict[str, Any],
) -> dict[str, str]:
    expected_entities, expected_instances = full_chain_acceptance._expectation_counts(
        expectations
    )
    entities = roster["entities"]
    actual_instances = sum(entity["instance_count"] for entity in entities)
    if len(entities) != expected_entities or actual_instances != expected_instances:
        raise RuntimeError("Character Roster live cardinality mismatch")
    matches: dict[str, str] = {}
    used: set[str] = set()
    for item in expectations["entity_expectations"]:
        anchors = {
            full_chain_acceptance._normalized_source_text(value)
            for value in item["source_mentions_any"]
        }
        candidates = []
        for entity in entities:
            source_text = " ".join([
                *(
                    str(mention)
                    for instance in entity["instances"]
                    for mention in instance["source_mentions"]
                ),
                *(
                    evidence["source_excerpt"]
                    for evidence in entity["source_visual_evidence"]
                ),
            ])
            normalized = full_chain_acceptance._normalized_source_text(source_text)
            if any(anchor in normalized for anchor in anchors):
                candidates.append(entity)
        if len(candidates) != 1:
            raise RuntimeError(
                "Character Roster live source anchors do not resolve uniquely: "
                + item["expectation_id"]
            )
        entity = candidates[0]
        if entity["entity_id"] in used:
            raise RuntimeError("Character Roster live expectations overlap")
        if entity["instance_count"] != item["instance_count"]:
            raise RuntimeError("Character Roster live instance count mismatch")
        used.add(entity["entity_id"])
        matches[item["expectation_id"]] = entity["entity_id"]
    return matches


def _validate_reconciled_characters(
    result: dict[str, Any],
    roster: dict[str, Any],
) -> None:
    characters = result.get("characters")
    if not isinstance(characters, list) or any(
        not isinstance(character, dict) for character in characters
    ):
        raise RuntimeError("Character Roster live result characters are invalid")
    expected = {
        entity["entity_id"]: int(entity["instance_count"])
        for entity in roster["entities"]
    }
    actual: dict[str, int] = {}
    for character in characters:
        entity_id = str(character.get("entity_id") or "")
        count = character.get("instance_count")
        if (
            not entity_id
            or entity_id in actual
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise RuntimeError(
                "Character Roster live reconciled character lineage is invalid"
            )
        actual[entity_id] = count
    if actual != expected:
        raise RuntimeError(
            "Character Roster live reconciled characters disagree with the roster"
        )


class CharacterRosterRequestReceipt:
    """Persist the only allowed Ark request before the transport sends it."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.path = workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME
        self.token: str | None = None

    def before(self, payload: dict[str, Any]) -> str:
        if self.path.exists() or self.token is not None:
            raise RuntimeError(
                "Character Roster live request already exists; resubmission forbidden"
            )
        portable = _safe_transport_metadata(payload)
        if portable.get("provider_family") != "ark_text":
            raise RuntimeError("Character Roster live request is not Ark text")
        token = _canonical_sha256(portable)
        full_chain_acceptance._atomic_write_json(self.path, {
            "schema": RECEIPT_SCHEMA,
            "status": "submission_uncertain",
            "request_fingerprint": token,
            "provider_request_count": 1,
            "safe_payload": portable,
            "automatic_resubmission_forbidden": True,
        })
        self.token = token
        return token

    def after(self, token: str, outcome: dict[str, Any]) -> None:
        if token != self.token or not self.path.is_file():
            raise RuntimeError("Character Roster live completion token is invalid")
        receipt = _read_object(self.path)
        receipt.update({
            "status": "provider_completed",
            "safe_outcome": _safe_transport_metadata(outcome),
        })
        full_chain_acceptance._atomic_write_json(self.path, receipt)

    def failed(self, token: str, outcome: dict[str, Any]) -> None:
        if token != self.token or not self.path.is_file():
            raise RuntimeError("Character Roster live failure token is invalid")
        receipt = _read_object(self.path)
        known_rejected = outcome.get("submission_outcome") == "known_rejected"
        receipt.update({
            "status": "provider_failed" if known_rejected else "submission_uncertain",
            "safe_failure": _safe_transport_metadata(outcome),
            "automatic_resubmission_forbidden": True,
        })
        full_chain_acceptance._atomic_write_json(self.path, receipt)

    def completed(self) -> dict[str, Any]:
        receipt = _read_object(self.path)
        if receipt.get("status") != "provider_completed":
            raise RuntimeError("Character Roster live request did not settle")
        return receipt


def build_preflight(
    workspace: Path,
    events_path: Path,
    expectations_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    events_path = full_chain_acceptance._inside(events_path, workspace)
    expectations_path = full_chain_acceptance._inside(
        expectations_path,
        workspace,
    )
    events, events_sha256 = _load_events_contract(events_path)
    expectations = _read_object(expectations_path)
    full_chain_acceptance._expectation_counts(expectations)
    source_text = "\n".join(
        str(event.get("source_excerpt") or event.get("what") or "")
        for event in events
    )
    full_chain_acceptance._validate_expectation_source_anchors(
        source_text,
        expectations,
    )
    roster = compile_character_roster(events)
    entity_matches = _match_roster_expectations(roster, expectations)
    source_identity = full_chain_acceptance._repo_source_identity()
    regression = full_chain_acceptance._regression_evidence(
        workspace,
        source_identity["git_commit"],
    )
    formal_config = validate_config(["ARK_AGENT"])
    checks = {
        "worktree_clean": source_identity["worktree_clean"],
        "regression_receipt_current": regression["status"] == "passed",
        "ark_agent_configured": bool(
            formal_config.get("valid") and get_api_key("ARK_AGENT_API_KEY")
        ),
        "no_previous_request": not (
            workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME
        ).exists(),
    }
    missing = sorted(key for key, passed in checks.items() if not passed)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "preflight_passed" if not missing else "preflight_blocked",
        "call_chain_verdict": "not_submitted",
        "business_verdict": "not_evaluated",
        "provider_request_count": 0,
        "exact_request_limit": 1,
        "source": {
            **source_identity,
            "events_path": events_path.relative_to(workspace).as_posix(),
            "events_sha256": events_sha256,
            "expectations_path": expectations_path.relative_to(workspace).as_posix(),
            "expectations_sha256": hashlib.sha256(
                expectations_path.read_bytes()
            ).hexdigest(),
        },
        "character_roster_sha256": roster["roster_sha256"],
        "entity_matches": entity_matches,
        "checks": checks,
        "missing_configuration": missing,
        "regression": regression,
        "next_stage": "paid_single_observation" if not missing else "stop_zero_request",
    }


def execute_single_observation(
    workspace: Path,
    events_path: Path,
    expectations_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if preflight.get("status") != "preflight_passed":
        raise RuntimeError("Character Roster paid gate refused before preflight passed")
    workspace = workspace.resolve()
    events_path = full_chain_acceptance._inside(events_path, workspace)
    expectations_path = full_chain_acceptance._inside(
        expectations_path,
        workspace,
    )
    events, events_sha256 = _load_events_contract(events_path)
    if events_sha256 != (preflight.get("source") or {}).get("events_sha256"):
        raise RuntimeError("Character Roster live input changed after preflight")
    expectations = _read_object(expectations_path)
    request_receipt = CharacterRosterRequestReceipt(workspace)
    roster_path = workspace / LIVE_DIRECTORY / CHARACTER_ROSTER_FILENAME
    with provider_attempt_scope(
        max_retries=0,
        before_provider_request=request_receipt.before,
        after_provider_request=request_receipt.after,
        failed_provider_request=request_receipt.failed,
    ):
        result = discover_characters(
            events,
            semantic_qa_enabled=False,
            roster_output_path=roster_path,
        )
    transport = request_receipt.completed()
    roster = validate_character_roster(result["character_roster"])
    _validate_reconciled_characters(result, roster)
    matches = _match_roster_expectations(roster, expectations)
    evidence_path = workspace / LIVE_DIRECTORY / "CHARACTER_ROSTER_RESULT.json"
    full_chain_acceptance._atomic_write_json(evidence_path, {
        "schema": RECEIPT_SCHEMA,
        "character_roster_sha256": roster["roster_sha256"],
        "entity_matches": matches,
        "character_entities": len(roster["entities"]),
        "character_instances": sum(
            entity["instance_count"] for entity in roster["entities"]
        ),
        "semantic_diagnostic_codes": sorted({
            str(item.get("code") or "")
            for item in result.get("semantic_diagnostics") or []
            if str(item.get("code") or "")
        }),
        "provider_request_count": 1,
    })
    return {
        **preflight,
        "status": "passed",
        "call_chain_verdict": "passed",
        "business_verdict": "passed",
        "provider_request_count": 1,
        "request_receipt_sha256": hashlib.sha256(
            request_receipt.path.read_bytes()
        ).hexdigest(),
        "evidence_path": evidence_path.relative_to(workspace).as_posix(),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "transport_status": transport["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    receipt_path = workspace / RECEIPT_NAME
    existing = _read_object(receipt_path) if receipt_path.is_file() else None
    if args.submit and existing and existing.get("status") in {
        "passed",
        "live_acceptance_failed",
        "submission_uncertain",
    }:
        raise RuntimeError(
            "Character Roster paid gate is terminal; resubmission is forbidden"
        )
    preflight = build_preflight(
        workspace,
        args.events,
        args.expectations,
    )
    full_chain_acceptance._atomic_write_json(receipt_path, preflight)
    if not args.submit:
        return 0 if preflight["status"] == "preflight_passed" else 2
    try:
        receipt = execute_single_observation(
            workspace,
            args.events,
            args.expectations,
            preflight,
        )
    except BaseException as error:
        failed = {
            **preflight,
            "status": "live_acceptance_failed",
            "call_chain_verdict": "failed",
            "business_verdict": "not_evaluated",
            "safe_error_type": type(error).__name__,
            "provider_request_count": int(
                (workspace / LIVE_DIRECTORY / REQUEST_RECEIPT_NAME).is_file()
            ),
        }
        full_chain_acceptance._atomic_write_json(receipt_path, failed)
        raise
    full_chain_acceptance._atomic_write_json(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
