#!/usr/bin/env python3
# ruff: noqa: E402
"""Run one explicitly authorized live Phase 5 adjudication acceptance.

Without ``--submit`` this command performs a zero-request preflight.  A
submitted acceptance is permanently capped at one Ark Responses boundary
invocation for the selected run and writes a crash-safe receipt before the
request can start.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from clients.ark_multimodal_client import ArkMultimodalClient
from clients.tos_uploader import is_media_upload_configured
from phases.phase5 import storyboard_qa_gate
from runtime.security_boundaries import redact_text
from utils.config import ARK_AGENT_CREDENTIAL_SOURCE, get_api_key

RECEIPT_SCHEMA = "honcut.phase5-adjudication-live-acceptance.v1"
RECEIPT_NAME = "phase5_adjudication_live_acceptance.json"
MAX_PAID_PROVIDER_REQUESTS = 1
REQUIRED_ACCEPTANCE_GATES = ["regression", "live_paid_provider"]


class ProviderRequestLimitError(RuntimeError):
    """The acceptance tried to cross its one-request Provider boundary."""


class _SingleRequestResponses:
    def __init__(self, owner: SinglePaidRequestReviewClient, delegate: Any) -> None:
        self._owner = owner
        self._delegate = delegate

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._owner.provider_request_attempt_count += 1
        if self._owner.provider_request_count >= MAX_PAID_PROVIDER_REQUESTS:
            self._owner.blocked_provider_request_count += 1
            raise ProviderRequestLimitError(
                "Phase 5 live acceptance permits exactly one paid Provider request"
            )
        self._owner.provider_request_count += 1
        return self._delegate.create(*args, **kwargs)


class _SingleRequestTransport:
    def __init__(self, owner: SinglePaidRequestReviewClient, delegate: Any) -> None:
        self._delegate = delegate
        self.responses = _SingleRequestResponses(owner, delegate.responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class SinglePaidRequestReviewClient:
    """Count review operations and guard the raw Responses transport boundary."""

    def __init__(self, delegate: ArkMultimodalClient) -> None:
        self._delegate = delegate
        self.review_operation_attempt_count = 0
        self.provider_request_attempt_count = 0
        self.provider_request_count = 0
        self.blocked_provider_request_count = 0
        delegate.client = _SingleRequestTransport(self, delegate.client)

    def review_structured(
        self,
        media_paths: list[Path],
        prompt: str,
        response_model: Any,
    ) -> Any:
        self.review_operation_attempt_count += 1
        return self._delegate.review_structured(
            media_paths,
            prompt,
            response_model,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _credential_readiness() -> dict[str, Any]:
    if not get_api_key("ARK_AGENT_API_KEY"):
        raise RuntimeError("ARK_AGENT_API_KEY is not configured")
    if not is_media_upload_configured():
        raise RuntimeError("TOS media upload credentials are not configured")
    return {
        "ark_agent_credential_source": ARK_AGENT_CREDENTIAL_SOURCE,
        "tos_media_upload_configured": True,
    }


def _pending_preflight(
    output_dir: Path,
    *,
    expected_storyboard_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Phase 5 output directory not found: {output_dir}")
    loaded = storyboard_qa_gate._load_pending_adjudication(output_dir)
    if loaded is None:
        raise RuntimeError("Phase 5 has no pending review adjudication")
    pending, _receipts = loaded
    if pending.get("status") != "blocked_unavailable":
        raise RuntimeError("only a blocked_unavailable Phase 5 adjudication is live-resumable")
    raw_targets = pending.get("storyboard_ids")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RuntimeError("pending Phase 5 adjudication has no storyboard IDs")
    targets = sorted(dict.fromkeys(str(value) for value in raw_targets if value))
    expected = sorted(dict.fromkeys(expected_storyboard_ids or []))
    if expected and expected != targets:
        raise RuntimeError("pending Phase 5 adjudication IDs do not match the explicit expectation")
    parent_shot_ids = sorted({storyboard_qa_gate._parent_shot_id(value) for value in targets})
    if len(parent_shot_ids) != 1:
        raise RuntimeError(
            "one-request live acceptance requires all disputed panels to share one Sxx"
        )
    observed_hashes = storyboard_qa_gate._storyboard_panel_hashes(output_dir)
    expected_hashes = pending.get("asset_sha256")
    if not isinstance(expected_hashes, dict) or any(
        observed_hashes.get(storyboard_id) != expected_hashes.get(storyboard_id)
        for storyboard_id in targets
    ):
        raise RuntimeError("pending Phase 5 adjudication pixels changed before preflight")
    return {
        "output_dir": str(output_dir),
        "storyboard_ids": targets,
        "parent_shot_ids": parent_shot_ids,
        "asset_sha256": {
            storyboard_id: observed_hashes[storyboard_id] for storyboard_id in targets
        },
        "pending_receipt_schema": pending.get("schema"),
        "pending_receipt_status": pending.get("status"),
        "legacy_reconstructed": pending.get("legacy_reconstructed") is True,
        "credentials": _credential_readiness(),
    }


def _acceptance_summary(result: dict[str, Any]) -> dict[str, Any]:
    receipts = result.get("review_adjudications")
    if not isinstance(receipts, list):
        correction = result.get("correction")
        receipts = correction.get("review_adjudications") if isinstance(correction, dict) else []
    final_receipt = receipts[-1] if isinstance(receipts, list) and receipts else {}
    return {
        "status": result.get("status"),
        "grade": result.get("grade"),
        "gate_passed": result.get("gate_passed") is True,
        "error": redact_text(str(result.get("error") or "")),
        "adjudication_status": (
            final_receipt.get("status") if isinstance(final_receipt, dict) else None
        ),
        "decisions": (final_receipt.get("decisions") if isinstance(final_receipt, dict) else {}),
    }


def run_acceptance(
    output_dir: Path,
    *,
    submit: bool,
    expected_storyboard_ids: list[str] | None = None,
    client_factory: Callable[[], ArkMultimodalClient] = ArkMultimodalClient,
    resume_runner: Callable[..., dict[str, Any]] = (
        storyboard_qa_gate.run_storyboard_qa_with_correction
    ),
    adjudication_review: Callable[..., dict[str, Any]] = (
        storyboard_qa_gate._run_storyboard_adjudication_review
    ),
) -> dict[str, Any]:
    """Preflight or execute one live, shot-scoped adjudication resume."""
    output_dir = Path(output_dir).resolve()
    receipt_path = output_dir / RECEIPT_NAME
    if receipt_path.is_file():
        existing = _read_json_object(receipt_path)
        if existing.get("schema") != RECEIPT_SCHEMA:
            raise RuntimeError("unknown Phase 5 live acceptance receipt schema")
        if existing.get("submitted") is True:
            if submit:
                raise RuntimeError(
                    "this Phase 5 run already consumed or attempted its live request"
                )
            return existing

    preflight = _pending_preflight(
        output_dir,
        expected_storyboard_ids=expected_storyboard_ids,
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "preflight_passed",
        "submitted": False,
        "created_at": _utc_now(),
        "acceptance_gate": "live_paid_provider",
        "required_acceptance_gates": REQUIRED_ACCEPTANCE_GATES,
        "provider_request_limit": MAX_PAID_PROVIDER_REQUESTS,
        "provider_request_count": 0,
        "preflight": preflight,
    }
    _atomic_write_json(receipt_path, receipt)
    if not submit:
        return receipt

    receipt.update(
        status="submission_uncertain",
        submitted=True,
        submission_started_at=_utc_now(),
    )
    _atomic_write_json(receipt_path, receipt)

    limited_client: SinglePaidRequestReviewClient | None = None
    try:
        limited_client = SinglePaidRequestReviewClient(client_factory())
        targets = list(preflight["storyboard_ids"])

        def confirmation_runner(
            target_output_dir: Path,
            storyboard_ids: list[str],
        ) -> dict[str, Any]:
            if Path(target_output_dir).resolve() != output_dir:
                raise RuntimeError("Phase 5 resume changed the acceptance output directory")
            if sorted(storyboard_ids) != targets:
                raise RuntimeError("Phase 5 resume changed the preflighted adjudication scope")
            return adjudication_review(
                output_dir,
                targets,
                multimodal_client=limited_client,
                structured_understanding_max_attempts=1,
            )

        result = resume_runner(
            output_dir,
            resume_pending_adjudication=True,
            adjudication_runner=confirmation_runner,
        )
        summary = _acceptance_summary(result)
        receipt.update(
            provider_request_count=limited_client.provider_request_count,
            provider_request_attempt_count=(limited_client.provider_request_attempt_count),
            review_operation_attempt_count=(limited_client.review_operation_attempt_count),
            blocked_provider_request_count=(limited_client.blocked_provider_request_count),
            phase5=summary,
            completed_at=_utc_now(),
        )
        accepted = (
            limited_client.provider_request_count == MAX_PAID_PROVIDER_REQUESTS
            and summary.get("adjudication_status") == "completed"
        )
        receipt["status"] = "passed" if accepted else "failed"
        if not accepted:
            receipt["error"] = (
                "live acceptance did not complete one schema-valid Phase 5 adjudication"
            )
    except Exception as exc:
        receipt.update(
            status="failed",
            error=f"{type(exc).__name__}: {redact_text(str(exc))}",
            completed_at=_utc_now(),
        )
        if limited_client is not None:
            receipt.update(
                provider_request_count=limited_client.provider_request_count,
                provider_request_attempt_count=(limited_client.provider_request_attempt_count),
                review_operation_attempt_count=(limited_client.review_operation_attempt_count),
                blocked_provider_request_count=(limited_client.blocked_provider_request_count),
            )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-storyboard-id",
        action="append",
        dest="expected_storyboard_ids",
        default=[],
        help="pin one expected disputed Pxx ID; repeat for multiple Pxx in one Sxx",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "authorize at most one paid Ark Responses request; without it only "
            "zero-request preflight runs"
        ),
    )
    args = parser.parse_args()
    try:
        result = run_acceptance(
            args.output_dir,
            submit=args.submit,
            expected_storyboard_ids=args.expected_storyboard_ids,
        )
    except Exception as exc:
        result = {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "submitted": False,
            "acceptance_gate": "live_paid_provider",
            "required_acceptance_gates": REQUIRED_ACCEPTANCE_GATES,
            "provider_request_count": 0,
            "error": f"{type(exc).__name__}: {redact_text(str(exc))}",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"preflight_passed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
