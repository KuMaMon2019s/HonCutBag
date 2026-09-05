"""Strict sidecar migration for legacy parent action artifacts.

The parent payload is immutable evidence.  This owner emits a separate JSON-safe
sidecar and audit disposition; it never rewrites source media or calls a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from utils.action_kinematics import (
    KINEMATICS_POLICY_SHA256,
    apply_generation_kinematics_projection,
    compile_source_kinematics,
)

MIGRATION_SCHEMA = "honcut.action-kinematics-sidecar-migration.v1"
LEGACY_BODY_ACTION_SCHEMA = "honcut.body-action-choreography.v1"
CURRENT_BODY_ACTION_SCHEMA = "honcut.body-action-choreography.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_BEAT_FIELDS = frozenset(
    {
        "micro_action_index",
        "micro_action",
        "performer",
        "technique",
        "side",
        "limbs",
        "footwork",
        "torso",
        "weight_shift",
        "direction",
        "contact",
        "end_pose",
    }
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_records(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    if "body_action_contract" in payload:
        candidates.append(("$", copy.deepcopy(dict(payload))))
    for shot_index, shot in enumerate(payload.get("shots") or []):
        if not isinstance(shot, Mapping):
            continue
        if "body_action_contract" in shot:
            candidates.append((f"shots[{shot_index}]", copy.deepcopy(dict(shot))))
        for beat_index, beat in enumerate(shot.get("storyboard_beats") or []):
            if isinstance(beat, Mapping) and "body_action_contract" in beat:
                candidates.append(
                    (
                        f"shots[{shot_index}].storyboard_beats[{beat_index}]",
                        copy.deepcopy(dict(beat)),
                    )
                )
    for event_index, event in enumerate(payload.get("events") or []):
        if isinstance(event, Mapping) and "body_action_contract" in event:
            candidates.append((f"events[{event_index}]", copy.deepcopy(dict(event))))
    return candidates


def _audit_only_receipt(
    *,
    parent_artifact_id: str,
    parent_content_sha256: str,
    parent_payload_sha256: str,
    reason: str,
    downstream_artifact_ids: Sequence[str],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": MIGRATION_SCHEMA,
        "status": "audit_only",
        "reason": reason,
        "parent_artifact_id": parent_artifact_id,
        "parent_content_sha256": parent_content_sha256,
        "parent_payload_sha256": parent_payload_sha256,
        "from_schema": LEGACY_BODY_ACTION_SCHEMA,
        "to_schema": CURRENT_BODY_ACTION_SCHEMA,
        "kinematics_policy_sha256": KINEMATICS_POLICY_SHA256,
        "sidecar": None,
        "downstream_disposition": [
            {"artifact_id": value, "status": "stale", "usage": "audit_only"}
            for value in downstream_artifact_ids
        ],
        "provider_request_count": 0,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def migrate_parent_action_kinematics(
    parent_payload: Mapping[str, Any],
    *,
    parent_artifact_id: str,
    parent_content_sha256: str,
    downstream_artifact_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile an evidence-complete v1 parent into an immutable sidecar."""
    if not parent_artifact_id.strip() or not _SHA256_RE.fullmatch(parent_content_sha256):
        raise ValueError("action kinematics migration parent identity is invalid")
    parent_copy = copy.deepcopy(dict(parent_payload))
    parent_payload_sha256 = _canonical_sha256(parent_copy)
    candidates = _candidate_records(parent_copy)
    if not candidates:
        return _audit_only_receipt(
            parent_artifact_id=parent_artifact_id,
            parent_content_sha256=parent_content_sha256,
            parent_payload_sha256=parent_payload_sha256,
            reason="parent_has_no_body_action_contract",
            downstream_artifact_ids=downstream_artifact_ids,
        )
    migrated_records: list[dict[str, Any]] = []
    try:
        for source_path, record in candidates:
            contract = record.get("body_action_contract")
            if not isinstance(contract, Mapping):
                raise ValueError("body_action_contract_not_object")
            schema = str(contract.get("schema") or "")
            if schema != LEGACY_BODY_ACTION_SCHEMA:
                version = re.fullmatch(r"honcut\.body-action-choreography\.v(\d+)", schema)
                reason = (
                    "future_body_action_schema"
                    if version and int(version.group(1)) > 2
                    else "non_migratable_body_action_schema"
                )
                raise ValueError(reason)
            beats = contract.get("beats")
            if not isinstance(beats, list) or not beats:
                raise ValueError("body_action_evidence_incomplete")
            compiled_beats = []
            for beat in beats:
                if not isinstance(beat, Mapping) or not _REQUIRED_BEAT_FIELDS.issubset(beat):
                    raise ValueError("body_action_evidence_incomplete")
                item = copy.deepcopy(dict(beat))
                item["kinematics"] = compile_source_kinematics(item)
                compiled_beats.append(item)
            units = record.get("generation_action_units")
            if not isinstance(units, list) or not units:
                raise ValueError("final_generation_lineage_missing")
            record["body_action_contract"] = {
                **copy.deepcopy(dict(contract)),
                "schema": CURRENT_BODY_ACTION_SCHEMA,
                "beats": compiled_beats,
            }
            projection = apply_generation_kinematics_projection(record)
            if projection is None:
                raise ValueError("body_action_kinematics_missing")
            migrated_records.append(
                {
                    "source_path": source_path,
                    "beat_id": str(record.get("beat_id") or ""),
                    "generation_action_unit_ids": [
                        str(unit.get("unit_id") or "")
                        for unit in record["generation_action_units"]
                    ],
                    "source_micro_action_indexes": projection[
                        "source_micro_action_indexes"
                    ],
                    "body_action_contract": record["body_action_contract"],
                    "generation_action_units": record["generation_action_units"],
                    "kinematics_projection": projection,
                }
            )
    except (KeyError, TypeError, ValueError) as exc:
        return _audit_only_receipt(
            parent_artifact_id=parent_artifact_id,
            parent_content_sha256=parent_content_sha256,
            parent_payload_sha256=parent_payload_sha256,
            reason=str(exc),
            downstream_artifact_ids=downstream_artifact_ids,
        )
    sidecar = {
        "schema": "honcut.action-kinematics-sidecar.v1",
        "parent_artifact_id": parent_artifact_id,
        "parent_content_sha256": parent_content_sha256,
        "parent_payload_sha256": parent_payload_sha256,
        "kinematics_policy_sha256": KINEMATICS_POLICY_SHA256,
        "records": migrated_records,
    }
    sidecar["sidecar_sha256"] = _canonical_sha256(sidecar)
    receipt = {
        "schema": MIGRATION_SCHEMA,
        "status": "migrated_sidecar",
        "parent_artifact_id": parent_artifact_id,
        "parent_content_sha256": parent_content_sha256,
        "parent_payload_sha256": parent_payload_sha256,
        "from_schema": LEGACY_BODY_ACTION_SCHEMA,
        "to_schema": CURRENT_BODY_ACTION_SCHEMA,
        "kinematics_policy_sha256": KINEMATICS_POLICY_SHA256,
        "sidecar": sidecar,
        "downstream_disposition": [
            {"artifact_id": value, "status": "stale", "usage": "audit_only"}
            for value in downstream_artifact_ids
        ],
        "provider_request_count": 0,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = [
    "CURRENT_BODY_ACTION_SCHEMA",
    "LEGACY_BODY_ACTION_SCHEMA",
    "MIGRATION_SCHEMA",
    "migrate_parent_action_kinematics",
]
