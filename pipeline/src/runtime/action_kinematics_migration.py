"""Strict sidecar migration for legacy parent action artifacts.

The parent payload is immutable evidence.  This owner emits a separate JSON-safe
sidecar and audit disposition; it never rewrites source media or calls a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runtime.artifact_manifest import ArtifactManifestStore
from runtime.security_boundaries import resolve_within_workspace
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_downstream_lineage(
    store: ArtifactManifestStore,
    *,
    parent_artifact_id: str,
    downstream_artifact_ids: Sequence[str],
) -> None:
    manifest = store.load()
    by_id = {artifact.artifact_id: artifact for artifact in manifest.artifacts}

    def descends_from_parent(artifact_id: str) -> bool:
        pending = list(by_id[artifact_id].parent_artifact_ids)
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == parent_artifact_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(by_id[current].parent_artifact_ids)
        return False

    for artifact_id in downstream_artifact_ids:
        if artifact_id == parent_artifact_id:
            raise RuntimeError("migration downstream artifact cannot be its parent")
        store.resolve(artifact_id, verify_content=True)
        if artifact_id not in by_id or not descends_from_parent(artifact_id):
            raise RuntimeError(
                f"migration downstream artifact does not descend from parent: {artifact_id}"
            )


def migrate_action_kinematics_artifact(
    store: ArtifactManifestStore,
    *,
    parent_artifact_id: str,
    downstream_artifact_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve, compile and register a zero-provider migration through Artifact v2."""
    if len(set(downstream_artifact_ids)) != len(downstream_artifact_ids):
        raise ValueError("migration downstream artifact IDs must be unique")
    parent = store.resolve(
        parent_artifact_id,
        verify_content=True,
        required_authority_roles=("story_action",),
    )
    _validate_downstream_lineage(
        store,
        parent_artifact_id=parent_artifact_id,
        downstream_artifact_ids=downstream_artifact_ids,
    )
    parent_path = resolve_within_workspace(
        store.run_directory,
        parent.relative_path,
        must_exist=True,
    )
    try:
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("action kinematics migration parent is not valid JSON") from exc
    if not isinstance(parent_payload, Mapping):
        raise RuntimeError("action kinematics migration parent must be a JSON object")
    receipt = migrate_parent_action_kinematics(
        parent_payload,
        parent_artifact_id=parent_artifact_id,
        parent_content_sha256=parent.content_sha256,
        downstream_artifact_ids=downstream_artifact_ids,
    )
    migration_key = _canonical_sha256(
        {
            "parent_artifact_id": parent_artifact_id,
            "parent_content_sha256": parent.content_sha256,
            "downstream_artifact_ids": list(downstream_artifact_ids),
            "policy_sha256": KINEMATICS_POLICY_SHA256,
        }
    )[:20]
    output_directory = resolve_within_workspace(
        store.run_directory,
        Path("migrations") / "action_kinematics" / migration_key,
    )
    registered_parent_ids = [parent_artifact_id]
    sidecar_artifact_id: str | None = None
    sidecar = receipt.get("sidecar")
    if isinstance(sidecar, Mapping):
        sidecar_path = output_directory / "sidecar.json"
        _write_json_atomic(sidecar_path, sidecar)
        sidecar_ref = store.register_file(
            sidecar_path,
            artifact_type="action_kinematics_sidecar",
            producer_node="runtime.action_kinematics_migration",
            parent_artifact_ids=(parent_artifact_id,),
            semantic_fingerprint=str(sidecar["sidecar_sha256"]),
            canonical_contract_sha256=parent.canonical_contract_sha256,
            authority_roles=("story_action",),
        )
        sidecar_artifact_id = sidecar_ref.artifact_id
        registered_parent_ids.append(sidecar_artifact_id)
    receipt_path = output_directory / "receipt.json"
    _write_json_atomic(receipt_path, receipt)
    receipt_ref = store.register_file(
        receipt_path,
        artifact_type="action_kinematics_migration_receipt",
        producer_node="runtime.action_kinematics_migration",
        parent_artifact_ids=registered_parent_ids,
        semantic_fingerprint=str(receipt["receipt_sha256"]),
        canonical_contract_sha256=parent.canonical_contract_sha256,
    )
    return {
        "receipt": receipt,
        "sidecar_artifact_id": sidecar_artifact_id,
        "receipt_artifact_id": receipt_ref.artifact_id,
        "provider_request_count": 0,
    }


__all__ = [
    "CURRENT_BODY_ACTION_SCHEMA",
    "LEGACY_BODY_ACTION_SCHEMA",
    "MIGRATION_SCHEMA",
    "migrate_action_kinematics_artifact",
    "migrate_parent_action_kinematics",
]
