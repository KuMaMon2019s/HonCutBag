"""Runtime policy and durable transitions for content-addressed TOS uploads."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator

from runtime.provider_policy import TOSUploadExecutionPolicy
from utils.provider_request_guard import media_upload_guard_scope


TOS_UPLOAD_LEDGER_SCHEMA = "honcut.tos-upload-ledger.v1"
TOS_UPLOAD_LEDGER_NAME = "TOS_UPLOAD_LEDGER.json"
_PAYLOAD_KEYS = frozenset(
    {
        "provider_family",
        "object_key",
        "payload_sha256",
        "payload_bytes",
        "content_type",
    }
)
_COMPLETED_STATUSES = frozenset(
    {
        "provider_completed",
        "reconciled_completed",
        "reused_completed",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("TOS upload payload must use the canonical safe fields")
    safe = {key: payload[key] for key in sorted(_PAYLOAD_KEYS)}
    if safe["provider_family"] != "tos_media_upload":
        raise ValueError("TOS upload payload has an invalid Provider family")
    if not isinstance(safe["object_key"], str) or not safe["object_key"].strip():
        raise ValueError("TOS upload object key must not be empty")
    digest = safe["payload_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("TOS upload payload SHA-256 is invalid")
    size = safe["payload_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("TOS upload payload size is invalid")
    if not isinstance(safe["content_type"], str) or not safe["content_type"].strip():
        raise ValueError("TOS upload content type must not be empty")
    return safe


class TOSUploadLedger:
    """One run-local append-only transition ledger for authoritative PUTs."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        max_submissions: int | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / TOS_UPLOAD_LEDGER_NAME
        self._lock = threading.RLock()

        if max_submissions is not None and (
            isinstance(max_submissions, bool)
            or not isinstance(max_submissions, int)
            or max_submissions < 0
        ):
            raise ValueError("TOS upload hard limit must be a non-negative integer")
        self.max_submissions = max_submissions

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": TOS_UPLOAD_LEDGER_SCHEMA,
            "hard_limit": self.max_submissions,
            "submission_attempt_count": 0,
            "uploads": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("TOS upload ledger is unreadable") from exc
        if not isinstance(value, dict) or value.get("schema") != TOS_UPLOAD_LEDGER_SCHEMA:
            raise RuntimeError("TOS upload ledger schema is unsupported")
        if value.get("hard_limit") != self.max_submissions:
            raise RuntimeError("TOS upload ledger hard limit changed")
        if not isinstance(value.get("submission_attempt_count"), int):
            raise RuntimeError("TOS upload ledger attempt count is invalid")
        uploads = value.get("uploads")
        if not isinstance(uploads, list) or any(not isinstance(upload, dict) for upload in uploads):
            raise RuntimeError("TOS upload ledger records are invalid")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(value, output, ensure_ascii=True, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _find(value: dict[str, Any], upload_id: str) -> dict[str, Any] | None:
        matches = [upload for upload in value["uploads"] if upload.get("upload_id") == upload_id]
        if len(matches) > 1:
            raise RuntimeError("TOS upload ledger contains duplicate upload IDs")
        return matches[0] if matches else None

    def prepare(self, payload: dict[str, Any]) -> tuple[dict[str, str], str]:
        safe = _safe_payload(payload)
        upload_id = _canonical_sha256(safe)
        with self._lock:
            value = self._read()
            record = self._find(value, upload_id)
            if record is None:
                record = {
                    "upload_id": upload_id,
                    "payload": safe,
                    "status": "prepared",
                    "transitions": [
                        {
                            "event": "UploadPrepared",
                            "at": _utc_now(),
                        }
                    ],
                }
                value["uploads"].append(record)
                self._write(value)
            elif record.get("payload") != safe:
                raise RuntimeError("TOS upload ledger fingerprint payload changed")
            status = str(record.get("status") or "")
            if status not in {
                "prepared",
                "submission_uncertain",
                "provider_rejected",
                *_COMPLETED_STATUSES,
            }:
                raise RuntimeError("TOS upload ledger status is unsupported")
            return {"upload_id": upload_id}, status

    def submission_started(
        self,
        token: dict[str, str],
        _payload: dict[str, Any],
    ) -> None:
        upload_id = str(token.get("upload_id") or "")
        with self._lock:
            value = self._read()
            record = self._find(value, upload_id)
            if record is None or record.get("status") != "prepared":
                raise RuntimeError("TOS upload is not eligible for a new submission")
            if (
                self.max_submissions is not None
                and value["submission_attempt_count"] >= self.max_submissions
            ):
                raise RuntimeError("TOS upload crossed its authoritative PUT hard limit")
            record["status"] = "submission_uncertain"
            record["transitions"].append(
                {
                    "event": "SubmissionAttempted",
                    "at": _utc_now(),
                }
            )
            value["submission_attempt_count"] += 1
            self._write(value)

    def completed(self, token: dict[str, str], outcome: dict[str, Any]) -> None:
        upload_id = str(token.get("upload_id") or "")
        completion_kind = str(outcome.get("completion_kind") or "")
        status_and_event = {
            "provider": ("provider_completed", "UploadCompleted"),
            "reconciled": ("reconciled_completed", "UploadReconciled"),
            "reused": ("reused_completed", "UploadReused"),
        }
        if completion_kind not in status_and_event:
            raise ValueError("TOS upload completion kind is invalid")
        with self._lock:
            value = self._read()
            record = self._find(value, upload_id)
            if record is None:
                raise RuntimeError("TOS upload completion has no prepared record")
            prior_status = str(record.get("status") or "")
            if prior_status in _COMPLETED_STATUSES:
                return
            if prior_status not in {"prepared", "submission_uncertain"}:
                raise RuntimeError("TOS upload completion has an invalid prior status")
            status, event = status_and_event[completion_kind]
            record["status"] = status
            record["transitions"].append(
                {
                    "event": event,
                    "at": _utc_now(),
                    "http_status": outcome.get("http_status"),
                    "verification": outcome.get("verification"),
                }
            )
            self._write(value)

    def failed(self, token: dict[str, str], outcome: dict[str, Any]) -> None:
        upload_id = str(token.get("upload_id") or "")
        known_rejected = outcome.get("submission_outcome") == "known_rejected"
        with self._lock:
            value = self._read()
            record = self._find(value, upload_id)
            if record is None or record.get("status") != "submission_uncertain":
                raise RuntimeError("TOS upload failure has an invalid prior status")
            record["status"] = "provider_rejected" if known_rejected else "submission_uncertain"
            record["transitions"].append(
                {
                    "event": "UploadRejected" if known_rejected else "UploadUncertain",
                    "at": _utc_now(),
                    "error_type": str(outcome.get("error_type") or "unknown"),
                    "http_status": outcome.get("http_status"),
                    "automatic_resubmission_forbidden": True,
                }
            )
            self._write(value)


@contextmanager
def tos_upload_execution_scope(
    workspace: str | Path,
    *,
    policy: TOSUploadExecutionPolicy | None = None,
    max_submissions: int | None = None,
) -> Iterator[TOSUploadLedger]:
    """Install the Runtime policy and ledger without exposing Runtime to clients."""

    upload_policy = policy or TOSUploadExecutionPolicy.from_environment()
    ledger = TOSUploadLedger(workspace, max_submissions=max_submissions)
    with media_upload_guard_scope(
        timeout_resolver=upload_policy.timeouts_for_payload,
        prepare_upload=ledger.prepare,
        submission_started=ledger.submission_started,
        upload_completed=ledger.completed,
        upload_failed=ledger.failed,
    ):
        yield ledger


__all__ = [
    "TOS_UPLOAD_LEDGER_NAME",
    "TOS_UPLOAD_LEDGER_SCHEMA",
    "TOSUploadLedger",
    "tos_upload_execution_scope",
]
