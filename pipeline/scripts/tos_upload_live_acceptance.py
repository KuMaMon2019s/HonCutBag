#!/usr/bin/env python3
"""One-PUT live acceptance for HonCut's required TOS media upload path."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from clients.tos_uploader import (
    is_media_upload_configured,
    upload_multimodal_media_file_required,
    validate_multimodal_media_file,
)
from runtime.tos_uploads import (
    TOS_UPLOAD_LEDGER_NAME,
    TOS_UPLOAD_LEDGER_SCHEMA,
    tos_upload_execution_scope,
)


ACCEPTANCE_SCHEMA = "honcut.tos-upload-live-acceptance.v1"
REGRESSION_SCHEMA = "honcut.tos-upload-regression.v1"
RECEIPT_NAME = "tos_upload_live_acceptance.json"
MIN_FIXTURE_BYTES = 1024 * 1024
MAX_FIXTURE_BYTES = 10 * 1024 * 1024 - 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"acceptance JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"acceptance JSON must be an object: {path.name}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=True, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repo_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return {"git_commit": commit, "tracked_worktree_clean": not status}


def _inside(path: Path, workspace: Path) -> Path:
    resolved = path.resolve(strict=True)
    root = workspace.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("acceptance input must remain inside its workspace") from exc
    return resolved


def build_preflight(
    workspace: Path,
    input_path: Path,
    regression_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    input_path = _inside(input_path, workspace)
    regression_path = _inside(regression_path, workspace)
    media_kind = validate_multimodal_media_file(input_path)
    size = input_path.stat().st_size
    if media_kind != "image" or not MIN_FIXTURE_BYTES <= size <= MAX_FIXTURE_BYTES:
        raise RuntimeError("TOS live fixture must be a valid 1-10 MB multimodal image")
    source = _repo_identity()
    regression = _read_object(regression_path)
    if (
        regression.get("schema") != REGRESSION_SCHEMA
        or regression.get("status") != "passed"
        or regression.get("git_commit") != source["git_commit"]
    ):
        raise RuntimeError("TOS regression receipt is missing, failed, or stale")
    input_sha256 = _sha256(input_path)
    acceptance_id = hashlib.sha256(
        f"{source['git_commit']}:{input_sha256}".encode("utf-8")
    ).hexdigest()[:24]
    checks = {
        "tracked_worktree_clean": source["tracked_worktree_clean"],
        "tos_media_upload_configured": is_media_upload_configured(),
        "regression_receipt_current": True,
        "fixture_media_valid": True,
    }
    missing = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "preflight_passed" if not missing else "preflight_blocked",
        "stage": "zero_submit_preflight",
        "created_at": _utc_now(),
        "source": source,
        "fixture": {
            "path": input_path.relative_to(workspace).as_posix(),
            "sha256": input_sha256,
            "bytes": size,
            "media_kind": media_kind,
        },
        "regression": {
            "path": regression_path.relative_to(workspace).as_posix(),
            "sha256": _sha256(regression_path),
        },
        "acceptance_id": acceptance_id,
        "object_prefix": f"volcengine/live-acceptance/{acceptance_id}",
        "hard_limits": {
            "authoritative_puts": 1,
            "read_only_head_checks": 2,
            "other_provider_requests": 0,
        },
        "provider_request_count": 0,
        "checks": checks,
        "missing_requirements": missing,
        "next_stage": "live_submit" if not missing else "stop_zero_request",
    }


def _ledger_summary(workspace: Path) -> dict[str, Any]:
    ledger_path = workspace / TOS_UPLOAD_LEDGER_NAME
    if not ledger_path.is_file():
        return {
            "path": TOS_UPLOAD_LEDGER_NAME,
            "exists": False,
            "submission_attempt_count": 0,
        }
    ledger = _read_object(ledger_path)
    if ledger.get("schema") != TOS_UPLOAD_LEDGER_SCHEMA:
        raise RuntimeError("TOS live ledger schema is incompatible")
    uploads = ledger.get("uploads")
    if not isinstance(uploads, list):
        raise RuntimeError("TOS live ledger uploads are invalid")
    return {
        "path": TOS_UPLOAD_LEDGER_NAME,
        "exists": True,
        "sha256": _sha256(ledger_path),
        "submission_attempt_count": int(ledger.get("submission_attempt_count") or 0),
        "statuses": [
            str(upload.get("status") or "") for upload in uploads if isinstance(upload, dict)
        ],
    }


def submit_once(workspace: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    receipt_path = workspace / RECEIPT_NAME
    existing = _read_object(receipt_path) if receipt_path.is_file() else None
    if existing and existing.get("status") not in {"preflight_passed"}:
        raise RuntimeError("TOS live acceptance is terminal and cannot be replayed")
    current = build_preflight(
        workspace,
        workspace / preflight["fixture"]["path"],
        workspace / preflight["regression"]["path"],
    )
    immutable = ("source", "fixture", "regression", "acceptance_id", "object_prefix")
    if any(current.get(key) != preflight.get(key) for key in immutable):
        raise RuntimeError("TOS live acceptance preflight identity changed")
    if current["status"] != "preflight_passed":
        raise RuntimeError("TOS live acceptance preflight no longer passes")
    pending = {
        **preflight,
        "status": "submission_uncertain",
        "stage": "live_submit",
        "submission_attempted_at": _utc_now(),
    }
    _atomic_write_json(receipt_path, pending)
    try:
        with tos_upload_execution_scope(workspace, max_submissions=1):
            upload_multimodal_media_file_required(
                workspace / preflight["fixture"]["path"],
                prefix=preflight["object_prefix"],
                label="TOS live acceptance fixture",
            )
        ledger = _ledger_summary(workspace)
        if (
            ledger["submission_attempt_count"] != 1
            or len(ledger["statuses"]) != 1
            or ledger["statuses"][0] not in {"provider_completed", "reconciled_completed"}
        ):
            raise RuntimeError("TOS live gate did not perform and verify exactly one new PUT")
    except BaseException as exc:
        failed = {
            **pending,
            "status": "live_acceptance_failed",
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "ledger": _ledger_summary(workspace),
            "automatic_retry_forbidden": True,
        }
        _atomic_write_json(receipt_path, failed)
        raise
    accepted = {
        **pending,
        "status": "accepted",
        "completed_at": _utc_now(),
        "provider_request_count": 1,
        "ledger": ledger,
        "call_chain_verdict": "passed",
        "business_verdict": "payload_hash_and_length_verified",
        "automatic_retry_forbidden": True,
    }
    _atomic_write_json(receipt_path, accepted)
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    receipt_path = workspace / RECEIPT_NAME
    if args.submit:
        if not receipt_path.is_file():
            raise RuntimeError("run zero-submit preflight before --submit")
        receipt = submit_once(workspace, _read_object(receipt_path))
    else:
        if receipt_path.is_file():
            existing = _read_object(receipt_path)
            if existing.get("status") not in {
                "preflight_passed",
                "preflight_blocked",
            }:
                raise RuntimeError("TOS live acceptance already crossed its boundary")
        receipt = build_preflight(
            workspace,
            args.input,
            args.regression_receipt,
        )
        _atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"preflight_passed", "accepted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
