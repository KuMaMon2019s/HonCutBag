#!/usr/bin/env python3
# ruff: noqa: E402
"""Accept one live Phase 5 post-adjudication PREVIS correction request.

Run this only against an isolated copy of a production run.  Without
``--submit`` it performs a zero-request preflight.  A submitted receipt is
permanently capped at one Seedream operation and one raw image-generation
POST for one explicitly pinned Pxx.
"""

from __future__ import annotations

import argparse
import hashlib
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

from clients import seedream_client
from clients.seedream_client import SeedreamClient
from phases.phase5 import storyboard_qa_gate
from runtime.security_boundaries import redact_text
from utils.config import ARK_AGENT_CREDENTIAL_SOURCE, get_api_key

RECEIPT_SCHEMA = "honcut.phase5-correction-resume-live-acceptance.v1"
RECEIPT_NAME = "phase5_correction_resume_live_acceptance.json"
MAX_PAID_PROVIDER_REQUESTS = 1
REQUIRED_ACCEPTANCE_GATES = ["regression", "live_paid_provider"]


class ProviderRequestLimitError(RuntimeError):
    """The acceptance tried to cross its one-request Provider boundary."""


class SinglePaidRequestImageClient:
    """Guard one logical Seedream image operation."""

    def __init__(self, delegate: SeedreamClient) -> None:
        self._delegate = delegate
        self.image_operation_attempt_count = 0
        self.image_operation_count = 0
        self.blocked_image_operation_count = 0

    def _invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        self.image_operation_attempt_count += 1
        if self.image_operation_count >= MAX_PAID_PROVIDER_REQUESTS:
            self.blocked_image_operation_count += 1
            raise ProviderRequestLimitError(
                "Phase 5 correction live acceptance permits exactly one "
                "Seedream image operation"
            )
        self.image_operation_count += 1
        return getattr(self._delegate, method_name)(*args, **kwargs)

    def text_to_image(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("text_to_image", *args, **kwargs)

    def image_to_image(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("image_to_image", *args, **kwargs)


class SinglePaidRequestTransport:
    """Guard the raw Seedream POST so internal retry cannot spend twice."""

    def __init__(self) -> None:
        self.provider_request_attempt_count = 0
        self.provider_request_count = 0
        self.blocked_provider_request_count = 0
        self._original_post: Callable[..., Any] | None = None

    def __enter__(self) -> SinglePaidRequestTransport:
        self._original_post = seedream_client.requests.post
        seedream_client.requests.post = self._post
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._original_post is not None:
            seedream_client.requests.post = self._original_post

    def _post(self, *args: Any, **kwargs: Any) -> Any:
        self.provider_request_attempt_count += 1
        if self.provider_request_count >= MAX_PAID_PROVIDER_REQUESTS:
            self.blocked_provider_request_count += 1
            raise ProviderRequestLimitError(
                "Phase 5 correction live acceptance permits exactly one "
                "paid Provider request"
            )
        if self._original_post is None:
            raise RuntimeError("Seedream transport guard is not active")
        self.provider_request_count += 1
        return self._original_post(*args, **kwargs)


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel_sidecar_hashes(
    output_dir: Path,
    storyboard_ids: list[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for storyboard_id in storyboard_ids:
        for suffix in (".png", ".json", "_prompt.txt"):
            path = output_dir / "storyboard_beats" / f"{storyboard_id}{suffix}"
            if path.is_file():
                hashes[str(path.relative_to(output_dir))] = _sha256_file(path)
    return hashes


def _credential_readiness() -> dict[str, Any]:
    if not get_api_key("ARK_AGENT_API_KEY"):
        raise RuntimeError("ARK_AGENT_API_KEY is not configured")
    return {"ark_agent_credential_source": ARK_AGENT_CREDENTIAL_SOURCE}


def _continuation_preflight(
    output_dir: Path,
    *,
    expected_storyboard_id: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Phase 5 output directory not found: {output_dir}")
    state = storyboard_qa_gate._load_completed_adjudication_correction(output_dir)
    if state is None or state.get("continuable") is not True:
        raise RuntimeError("Phase 5 has no continuable completed adjudication")
    correction = state["correction"]
    if correction.get("correction_family") != "storyboard_previs":
        raise RuntimeError("live acceptance only supports storyboard_previs continuation")
    attempts_used = int(correction.get("attempts_used") or 0)
    max_attempts = int(correction.get("max_attempts") or 0)
    if attempts_used < 1 or attempts_used >= max_attempts:
        raise RuntimeError("Phase 5 has no remaining PREVIS correction attempt")
    issues = storyboard_qa_gate._correctable_issues(state["result"])
    targets = storyboard_qa_gate._correctable_storyboard_ids(issues)
    expected_storyboard_id = str(expected_storyboard_id or "").strip()
    if expected_storyboard_id not in targets:
        raise RuntimeError(
            "expected storyboard ID is not in the evidence-backed correction scope"
        )
    panel_hashes = storyboard_qa_gate._storyboard_panel_hashes(output_dir)
    protected_ids = sorted(set(panel_hashes) - {expected_storyboard_id})
    return {
        "output_dir": str(output_dir),
        "storyboard_id": expected_storyboard_id,
        "available_storyboard_ids": targets,
        "parent_shot_id": storyboard_qa_gate._parent_shot_id(
            expected_storyboard_id
        ),
        "correction_attempt": attempts_used + 1,
        "asset_sha256": panel_hashes[expected_storyboard_id],
        "protected_panel_sha256": {
            storyboard_id: panel_hashes[storyboard_id]
            for storyboard_id in protected_ids
        },
        "protected_sidecar_sha256": _panel_sidecar_hashes(
            output_dir, protected_ids
        ),
        "credentials": _credential_readiness(),
    }


def _issues_for_storyboard_id(
    report: dict[str, Any],
    storyboard_id: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for issue in storyboard_qa_gate._correctable_issues(report):
        narrowed = storyboard_qa_gate._issue_for_storyboard_id(
            issue, storyboard_id
        )
        if narrowed is not None:
            issues.append(narrowed)
    if not issues:
        raise RuntimeError("live acceptance target lost its correction evidence")
    return issues


def run_acceptance(
    output_dir: Path,
    *,
    submit: bool,
    expected_storyboard_id: str,
    client_factory: Callable[[], SeedreamClient] = SeedreamClient,
    redraw_runner: Callable[..., dict[str, Any]] = (
        storyboard_qa_gate._redraw_failed_storyboards
    ),
) -> dict[str, Any]:
    """Preflight or execute one live, exact-Pxx PREVIS correction."""
    output_dir = Path(output_dir).resolve()
    receipt_path = output_dir / RECEIPT_NAME
    if receipt_path.is_file():
        existing = _read_json_object(receipt_path)
        if existing.get("schema") != RECEIPT_SCHEMA:
            raise RuntimeError("unknown Phase 5 correction acceptance receipt schema")
        if existing.get("submitted") is True:
            if submit:
                raise RuntimeError(
                    "this Phase 5 correction acceptance already consumed or "
                    "attempted its live request"
                )
            return existing

    preflight = _continuation_preflight(
        output_dir,
        expected_storyboard_id=expected_storyboard_id,
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "pending_live_acceptance",
        "preflight_status": "passed",
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

    current_preflight = _continuation_preflight(
        output_dir,
        expected_storyboard_id=expected_storyboard_id,
    )
    if current_preflight != preflight:
        raise RuntimeError("Phase 5 correction evidence changed after preflight")
    receipt.update(
        status="submission_uncertain",
        submitted=True,
        submission_started_at=_utc_now(),
    )
    _atomic_write_json(receipt_path, receipt)

    limited_client: SinglePaidRequestImageClient | None = None
    transport: SinglePaidRequestTransport | None = None
    try:
        state = storyboard_qa_gate._load_completed_adjudication_correction(
            output_dir
        )
        if state is None or state.get("continuable") is not True:
            raise RuntimeError("Phase 5 correction continuation disappeared")
        target = str(preflight["storyboard_id"])
        issues = _issues_for_storyboard_id(state["result"], target)
        limited_client = SinglePaidRequestImageClient(client_factory())
        transport = SinglePaidRequestTransport()
        with transport:
            redraw = redraw_runner(
                output_dir,
                [str(preflight["parent_shot_id"])],
                issues,
                int(preflight["correction_attempt"]),
                image_client=limited_client,
            )

        panel_hashes = storyboard_qa_gate._storyboard_panel_hashes(output_dir)
        protected_panel_hashes = preflight["protected_panel_sha256"]
        protected_sidecars = preflight["protected_sidecar_sha256"]
        current_sidecars = {
            relative_path: _sha256_file(output_dir / relative_path)
            for relative_path in protected_sidecars
            if (output_dir / relative_path).is_file()
        }
        business_assertions = {
            "exact_storyboard_scope": redraw.get("storyboard_ids") == [target],
            "single_panel_regenerated": redraw.get("regenerated_panel_count") == 1,
            "target_pixel_changed": (
                panel_hashes.get(target) != preflight["asset_sha256"]
            ),
            "protected_panel_pixels_unchanged": all(
                panel_hashes.get(storyboard_id) == digest
                for storyboard_id, digest in protected_panel_hashes.items()
            ),
            "protected_panel_sidecars_unchanged": (
                current_sidecars == protected_sidecars
            ),
        }
        receipt.update(
            provider_request_count=transport.provider_request_count,
            provider_request_attempt_count=transport.provider_request_attempt_count,
            blocked_provider_request_count=transport.blocked_provider_request_count,
            image_operation_count=limited_client.image_operation_count,
            image_operation_attempt_count=(
                limited_client.image_operation_attempt_count
            ),
            blocked_image_operation_count=(
                limited_client.blocked_image_operation_count
            ),
            redraw=redraw,
            business_assertions=business_assertions,
            completed_at=_utc_now(),
        )
        accepted = (
            transport.provider_request_count == MAX_PAID_PROVIDER_REQUESTS
            and limited_client.image_operation_count
            == MAX_PAID_PROVIDER_REQUESTS
            and all(business_assertions.values())
        )
        receipt["status"] = "passed" if accepted else "live_acceptance_failed"
        if not accepted:
            receipt["error"] = (
                "live acceptance did not complete one exact-Pxx correction"
            )
    except Exception as exc:
        receipt.update(
            status="live_acceptance_failed",
            error=f"{type(exc).__name__}: {redact_text(str(exc))}",
            completed_at=_utc_now(),
        )
        if transport is not None:
            receipt.update(
                provider_request_count=transport.provider_request_count,
                provider_request_attempt_count=(
                    transport.provider_request_attempt_count
                ),
                blocked_provider_request_count=(
                    transport.blocked_provider_request_count
                ),
            )
        if limited_client is not None:
            receipt.update(
                image_operation_count=limited_client.image_operation_count,
                image_operation_attempt_count=(
                    limited_client.image_operation_attempt_count
                ),
                blocked_image_operation_count=(
                    limited_client.blocked_image_operation_count
                ),
            )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="isolated copy of a Phase 5 run; submitted acceptance mutates it",
    )
    parser.add_argument("--expected-storyboard-id", required=True)
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "authorize at most one paid Seedream request; without it only "
            "zero-request preflight runs"
        ),
    )
    args = parser.parse_args()
    try:
        result = run_acceptance(
            args.output_dir,
            submit=args.submit,
            expected_storyboard_id=args.expected_storyboard_id,
        )
    except Exception as exc:
        result = {
            "schema": RECEIPT_SCHEMA,
            "status": "live_acceptance_failed",
            "submitted": False,
            "acceptance_gate": "live_paid_provider",
            "required_acceptance_gates": REQUIRED_ACCEPTANCE_GATES,
            "provider_request_count": 0,
            "error": f"{type(exc).__name__}: {redact_text(str(exc))}",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"pending_live_acceptance", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
