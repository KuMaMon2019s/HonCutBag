#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a resumable, source-generic 36-second Canonical Visual Ledger acceptance.

The default mode is a zero-request Stage 0 preflight.  ``--submit`` is an
explicit paid boundary and uses the normal Lifecycle/Phase owners; this script
does not contain a fixture Provider, story-specific branch, or alternate
production topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from clients.seedream_client import DEFAULT_MODEL as SEEDREAM_MODEL
from clients.seedream_client import AGENT_PLAN_IMAGE_MODELS
from clients import seedance_client
from clients.tos_uploader import is_media_upload_configured
from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from phases.phase1.dry_run_capacity import build_dry_run_capacity_preflight
from phases.phase3.phase3_character import run_phase3 as run_phase3_owner
from phases.phase5.storyboard_qa_gate import (
    QAObservationGatePaused,
    run_storyboard_qa_gate,
    run_storyboard_qa_with_correction,
)
from phases.phase6.phase6_video_gen import run_phase6 as run_phase6_owner
from prompt.event_extractor import MAX_RETRIES as EVENT_EXTRACTOR_MAX_RETRIES
from prompt.text_parser import parse_text
import runtime.pipeline_execution as pipeline_lifecycle
from runtime.continuity_chunks import (
    ChunkExecutionRequest,
    load_continuity_plan,
    write_shadow_runtime_report,
)
from runtime.continuity_provider import (
    _media_index_manifest,
    _provider_content,
    _task_payload,
    _video_geometry,
)
from runtime.generation_tasks import GenerationTaskStore
from runtime.pipeline_execution import run_pipeline
from runtime.provider_attempt_policy import provider_attempt_scope
from runtime.tos_uploads import (
    TOS_UPLOAD_LEDGER_NAME,
    TOS_UPLOAD_LEDGER_SCHEMA,
    tos_upload_execution_scope,
)
from utils.config import (
    ARK_AGENT_CREDENTIAL_SOURCE,
    DEFAULT_MULTIMODAL_MODEL,
    SEEDANCE_MODEL,
    get_api_key,
    validate_config,
)
from utils.canonical_visual_contracts import load_canonical_visual_contract
from phases.phase1.character_roster import validate_character_roster
from utils.video_capabilities import get_video_capabilities


RECEIPT_SCHEMA = "honcut.canonical-visual-ledger-full-chain-acceptance.v1"
EXPECTATIONS_SCHEMA = "honcut.full-chain-acceptance-expectations.v2"
RECEIPT_NAME = "canonical_visual_ledger_36s_acceptance.json"
REGRESSION_SCHEMA = "honcut.canonical-visual-ledger-regression.v1"
REGRESSION_RECEIPT_NAME = "canonical_visual_ledger_regression.json"
LIVE_GATES_DIRECTORY = Path("live_acceptance") / "canonical_visual_ledger"
PAID_REQUEST_GUARD_SCHEMA = "honcut.live-paid-request-guard.v1"
RECOVERY_MATRIX_SCHEMA = "honcut.provider-deny-recovery-matrix.v1"
TARGET_DURATION_S = 36
MEDIA_PROFILE = "480p"
VISUAL_POLICY = "fictional_cinematic_human_v1"
ALL_PHASES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required JSON is missing or invalid: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON must contain an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _portable_payload(value: Any, workspace: Path) -> Any:
    """Keep acceptance receipts JSON-safe and free of workspace absolutes."""
    if isinstance(value, dict):
        return {
            str(key): _portable_payload(item, workspace)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_payload(item, workspace) for item in value]
    if isinstance(value, tuple):
        return [_portable_payload(item, workspace) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(
                    workspace.resolve()
                ).as_posix()
            except ValueError:
                return value
    return value


class _SinglePaidRequestGuard:
    """Durably freeze one safe payload before its one permitted request."""

    def __init__(self, workspace: Path, name: str) -> None:
        self.workspace = workspace
        self.name = name
        self.path = workspace / LIVE_GATES_DIRECTORY / f"{name}_request.json"
        self._called = False

    def __call__(self, payload: dict[str, Any]) -> None:
        if self._called:
            raise RuntimeError(f"{self.name} crossed its one-request hard limit")
        if self.path.exists():
            previous = _read_object(self.path)
            raise RuntimeError(
                f"{self.name} already has a durable {previous.get('status')} "
                "attempt; automatic resubmission is forbidden"
            )
        portable = _portable_payload(payload, self.workspace)
        fingerprint = _canonical_sha256(portable)
        now = _utc_now()
        _atomic_write_json(self.path, {
            "schema": PAID_REQUEST_GUARD_SCHEMA,
            "status": "submission_uncertain",
            "request_name": self.name,
            "request_fingerprint": fingerprint,
            "provider_request_count": 1,
            "zero_submit_preflight": {
                "status": "passed",
                "payload": portable,
                "payload_sha256": fingerprint,
            },
            "events": [
                {"event": "PayloadFrozen", "at": now},
                {"event": "SubmissionAttempted", "at": now},
            ],
        })
        self._called = True

    def complete(self, *, outcome: dict[str, Any]) -> dict[str, Any]:
        if not self._called or not self.path.is_file():
            raise RuntimeError(f"{self.name} did not submit its guarded request")
        receipt = _read_object(self.path)
        receipt.update({
            "status": "provider_completed",
            "completed_at": _utc_now(),
            "outcome": _portable_payload(outcome, self.workspace),
        })
        receipt.setdefault("events", []).append({
            "event": "ProviderCompleted",
            "at": receipt["completed_at"],
        })
        _atomic_write_json(self.path, receipt)
        return receipt


class _BoundedPaidRequestLedger:
    """Append-only acceptance ledger for a finite family of sync requests."""

    def __init__(self, workspace: Path, name: str, hard_limit: int) -> None:
        if isinstance(hard_limit, bool) or not isinstance(hard_limit, int) or hard_limit < 1:
            raise ValueError("paid request ledger hard_limit must be positive")
        self.workspace = workspace
        self.name = name
        self.hard_limit = hard_limit
        self.path = workspace / LIVE_GATES_DIRECTORY / f"{name}_requests.json"
        self._lock = threading.Lock()
        self._in_flight: set[str] = set()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema": PAID_REQUEST_GUARD_SCHEMA,
                "status": "active",
                "request_name": self.name,
                "hard_limit": self.hard_limit,
                "provider_request_count": 0,
                "attempts": [],
            }
        receipt = _read_object(self.path)
        if (
            receipt.get("schema") != PAID_REQUEST_GUARD_SCHEMA
            or receipt.get("request_name") != self.name
            or receipt.get("hard_limit") != self.hard_limit
        ):
            raise RuntimeError(f"{self.name} paid request ledger is incompatible")
        return receipt

    def before(self, payload: dict[str, Any]) -> str:
        with self._lock:
            receipt = self._read()
            blocked_failures = [
                attempt
                for attempt in receipt["attempts"]
                if attempt.get("status") == "provider_failed"
            ]
            unresolved = [
                attempt
                for attempt in receipt["attempts"]
                if (
                    attempt.get("status") == "submission_uncertain"
                    and attempt.get("request_id") not in self._in_flight
                )
            ]
            if blocked_failures or unresolved:
                raise RuntimeError(
                    f"{self.name} has a failed or unresolved request; "
                    "automatic resubmission is forbidden"
                )
            if receipt["provider_request_count"] >= self.hard_limit:
                raise RuntimeError(f"{self.name} crossed its paid hard limit")
            portable = _portable_payload(payload, self.workspace)
            request_index = receipt["provider_request_count"] + 1
            token = hashlib.sha256(
                f"{self.name}:{request_index}:{_canonical_sha256(portable)}".encode(
                    "utf-8"
                )
            ).hexdigest()
            receipt["attempts"].append({
                "request_id": token,
                "request_index": request_index,
                "request_fingerprint": _canonical_sha256(portable),
                "payload": portable,
                "status": "submission_uncertain",
                "submission_attempted_at": _utc_now(),
            })
            receipt["provider_request_count"] = request_index
            receipt["status"] = "submission_uncertain"
            _atomic_write_json(self.path, receipt)
            self._in_flight.add(token)
            return token

    def after(self, token: str, outcome: dict[str, Any]) -> None:
        with self._lock:
            receipt = self._read()
            matching = [
                attempt
                for attempt in receipt["attempts"]
                if attempt.get("request_id") == token
            ]
            if len(matching) != 1 or matching[0].get("status") != "submission_uncertain":
                raise RuntimeError(f"{self.name} completion token is invalid")
            matching[0].update({
                "status": "provider_completed",
                "provider_completed_at": _utc_now(),
                "outcome": _portable_payload(outcome, self.workspace),
            })
            self._in_flight.discard(token)
            statuses = {
                attempt.get("status") for attempt in receipt["attempts"]
            }
            receipt["status"] = (
                "provider_failed"
                if "provider_failed" in statuses
                else "submission_uncertain"
                if "submission_uncertain" in statuses
                else "provider_completed"
            )
            _atomic_write_json(self.path, receipt)

    def failed(self, token: str, outcome: dict[str, Any]) -> None:
        with self._lock:
            receipt = self._read()
            matching = [
                attempt
                for attempt in receipt["attempts"]
                if attempt.get("request_id") == token
            ]
            if len(matching) != 1 or matching[0].get("status") != "submission_uncertain":
                raise RuntimeError(f"{self.name} failure token is invalid")
            known_rejected = outcome.get("submission_outcome") == "known_rejected"
            matching[0].update({
                "status": "provider_failed" if known_rejected else "submission_uncertain",
                "failed_at": _utc_now(),
                "failure": _portable_payload(outcome, self.workspace),
                "automatic_resubmission_forbidden": True,
            })
            self._in_flight.discard(token)
            receipt["status"] = matching[0]["status"]
            _atomic_write_json(self.path, receipt)

    def settled_receipt(self) -> dict[str, Any]:
        with self._lock:
            receipt = self._read()
            if any(
                attempt.get("status") in {"submission_uncertain", "provider_failed"}
                for attempt in receipt["attempts"]
            ):
                raise RuntimeError(
                    f"{self.name} has a failed or unresolved Provider request"
                )
            return receipt


def _repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _repo_source_identity() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return {
        "git_commit": _repo_commit(),
        "worktree_clean": not bool(status),
    }


def _regression_evidence(workspace: Path, git_commit: str) -> dict[str, Any]:
    path = workspace / REGRESSION_RECEIPT_NAME
    try:
        receipt = _read_object(path)
    except RuntimeError:
        return {
            "status": "missing",
            "path": REGRESSION_RECEIPT_NAME,
        }
    valid = bool(
        receipt.get("schema") == REGRESSION_SCHEMA
        and receipt.get("status") == "passed"
        and (receipt.get("source") or {}).get("git_commit") == git_commit
        and int(receipt.get("provider_request_count") or 0) == 0
    )
    return {
        "status": "passed" if valid else "invalid",
        "path": REGRESSION_RECEIPT_NAME,
        "sha256": _sha256(path),
    }


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"acceptance input escapes workspace: {path}") from error
    return resolved


def _expectation_counts(expectations: dict[str, Any]) -> tuple[int, int]:
    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise RuntimeError("unsupported acceptance expectations schema")
    if expectations.get("expected_duration_s") != TARGET_DURATION_S:
        raise RuntimeError("acceptance expectations must require exactly 36 seconds")
    entity_count = expectations.get("expected_character_entities")
    instance_count = expectations.get("expected_character_instances")
    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or entity_count < 0
        or isinstance(instance_count, bool)
        or not isinstance(instance_count, int)
        or instance_count < entity_count
    ):
        raise RuntimeError("acceptance character counts are invalid")
    if not isinstance(expectations.get("required_events"), list):
        raise RuntimeError("acceptance required_events must be a list")
    if not isinstance(expectations.get("visual_facts"), dict):
        raise RuntimeError("acceptance visual_facts must be an object")
    entity_expectations = expectations.get("entity_expectations")
    if (
        not isinstance(entity_expectations, list)
        or len(entity_expectations) != entity_count
    ):
        raise RuntimeError(
            "acceptance entity_expectations must match the expected entity count"
        )
    expected_instances = 0
    expectation_ids: set[str] = set()
    for item in entity_expectations:
        if not isinstance(item, dict):
            raise RuntimeError("acceptance entity expectation must be an object")
        expectation_id = str(item.get("expectation_id") or "").strip()
        source_mentions = item.get("source_mentions_any")
        count = item.get("instance_count")
        if (
            not expectation_id
            or expectation_id in expectation_ids
            or not isinstance(source_mentions, list)
            or not source_mentions
            or any(not str(value or "").strip() for value in source_mentions)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(item.get("visual_facts", {}), dict)
        ):
            raise RuntimeError("acceptance entity expectation is invalid")
        expectation_ids.add(expectation_id)
        expected_instances += count
    if expected_instances != instance_count:
        raise RuntimeError(
            "acceptance entity expectations do not sum to the instance count"
        )
    return entity_count, instance_count


def _normalized_source_text(value: Any) -> str:
    return "".join(str(value or "").casefold().split())


def _validate_expectation_source_anchors(
    story: str,
    expectations: dict[str, Any],
) -> None:
    normalized_story = _normalized_source_text(story)
    required_anchors = [
        str(value or "").strip()
        for value in expectations.get("required_events") or []
    ]
    missing = [
        anchor
        for anchor in required_anchors
        if not anchor or _normalized_source_text(anchor) not in normalized_story
    ]
    for item in expectations.get("entity_expectations") or []:
        if not isinstance(item, dict):
            continue
        aliases = [
            str(value or "").strip()
            for value in item.get("source_mentions_any") or []
        ]
        if not any(
            alias and _normalized_source_text(alias) in normalized_story
            for alias in aliases
        ):
            missing.append(str(item.get("expectation_id") or "entity"))
    if missing:
        raise RuntimeError(
            "acceptance expectations contain source anchors absent from the story: "
            + ", ".join(sorted(set(missing)))
        )


def build_stage0_preflight(
    workspace: Path,
    story_path: Path,
    expectations_path: Path,
) -> dict[str, Any]:
    """Build a finite, zero-network upper bound from source and capabilities."""
    workspace = workspace.resolve()
    story_path = _inside(story_path, workspace)
    expectations_path = _inside(expectations_path, workspace)
    story = story_path.read_text(encoding="utf-8").strip()
    if not story:
        raise RuntimeError("acceptance story is empty")
    expectations = _read_object(expectations_path)
    character_entities, character_instances = _expectation_counts(expectations)
    _validate_expectation_source_anchors(story, expectations)
    parsed = parse_text(story)
    segments = list(parsed.get("segments") or [])
    if not segments:
        raise RuntimeError("acceptance story produced no source segments")

    structural = build_dry_run_capacity_preflight(
        story,
        segments,
        duration=TARGET_DURATION_S,
        shot_duration=AVG_SHOT_DURATION,
        shot_policy="continuity",
        max_material_padding_ratio=0.25,
        delivery_overrun_ratio=0.0,
    )["receipt"]
    caps = get_video_capabilities(
        model=SEEDANCE_MODEL,
        provider="seedance",
    )
    max_primary_shots = math.ceil(
        TARGET_DURATION_S / float(caps.min_primary_story_duration_s or 1)
    )
    max_pxx = max_primary_shots * 4
    # These are deliberately conservative pre-Phase-1 ceilings.  Phase 1 and
    # Phase 5 must replace them with exact frozen lists before later families.
    phase1_text_request_limit = (
        len(segments) * (EVENT_EXTRACTOR_MAX_RETRIES + 1)
        + 2 * math.ceil(max_primary_shots / 3)
        + 16
    )
    phase1_director_storyboard_image_limit = 1
    seedream_image_request_limit = (
        1
        + max_primary_shots * 6
        + character_instances * 6
    )
    multimodal_observation_request_limit = (
        character_instances * 8
        + max_primary_shots * 2
        + max_pxx
    )
    seedance_video_submission_limit = max_pxx + max(0, max_primary_shots - 1)
    max_tos_inputs_per_multimodal_request = 9
    max_tos_inputs_per_video_submission = caps.max_reference_images + 1
    hard_limits = {
        "phase1_text_requests": phase1_text_request_limit,
        "phase1_director_storyboard_image_requests": (
            phase1_director_storyboard_image_limit
        ),
        "phase1_provider_requests": (
            phase1_text_request_limit
            + phase1_director_storyboard_image_limit
        ),
        "seedream_image_requests": seedream_image_request_limit,
        "multimodal_observation_requests": multimodal_observation_request_limit,
        "seedance_video_submissions": seedance_video_submission_limit,
        "tos_put_attempts": (
            multimodal_observation_request_limit
            * max_tos_inputs_per_multimodal_request
            + seedance_video_submission_limit
            * max_tos_inputs_per_video_submission
        ),
        "max_tos_inputs_per_multimodal_request": (
            max_tos_inputs_per_multimodal_request
        ),
        "max_tos_inputs_per_video_submission": (
            max_tos_inputs_per_video_submission
        ),
    }
    source_identity = _repo_source_identity()
    regression = _regression_evidence(
        workspace,
        source_identity["git_commit"],
    )
    formal_config = validate_config(["ARK_AGENT"])
    credentials = {
        "ark_agent_configured": bool(
            formal_config.get("valid")
            and get_api_key("ARK_AGENT_API_KEY")
        ),
        "ark_agent_credential_source": ARK_AGENT_CREDENTIAL_SOURCE,
        "tos_media_upload_configured": is_media_upload_configured(),
    }
    missing = [name for name, configured in credentials.items() if not configured]
    # The credential source is descriptive, not a boolean requirement.
    missing = [name for name in missing if name != "ark_agent_credential_source"]
    capability_checks = {
        "seedream_agent_plan_model_supported": (
            SEEDREAM_MODEL in AGENT_PLAN_IMAGE_MODELS
        ),
        "seedance_2_model_selected": (
            "seedance-2.0" in SEEDANCE_MODEL.casefold()
        ),
        "seedance_reference_budget_at_least_nine": (
            isinstance(caps.max_reference_images, int)
            and caps.max_reference_images >= 9
        ),
        "multimodal_model_configured": bool(DEFAULT_MULTIMODAL_MODEL),
        "source_worktree_clean": source_identity["worktree_clean"],
        "regression_receipt_current": regression["status"] == "passed",
        "source_structure_fits_36_seconds": structural.get("status") == "passed",
        "source_structure_has_events": (
            int(structural.get("source_derived_event_count") or 0) > 0
        ),
    }
    missing.extend(
        name for name, passed in capability_checks.items() if not passed
    )
    missing = sorted(set(missing))
    status = "preflight_passed" if not missing else "preflight_blocked"
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "stage": "stage0_zero_request",
        "created_at": _utc_now(),
        "provider_request_count": 0,
        "source": {
            **source_identity,
            "story_path": story_path.relative_to(workspace).as_posix(),
            "story_sha256": _sha256(story_path),
            "expectations_path": expectations_path.relative_to(workspace).as_posix(),
            "expectations_sha256": _sha256(expectations_path),
        },
        "configuration": {
            **credentials,
            "seedream_model": SEEDREAM_MODEL,
            "seedance_model": SEEDANCE_MODEL,
            "multimodal_model": DEFAULT_MULTIMODAL_MODEL,
            "media_profile": MEDIA_PROFILE,
            "character_visual_policy": VISUAL_POLICY,
            "phase1_semantic_qa": False,
            "delivery_overrun_ratio": 0.0,
            "automatic_reshoot": False,
            "character_library_configured": False,
        },
        "capabilities": {
            "video_min_duration_s": caps.min_shot_duration_s,
            "video_max_duration_s": caps.max_shot_duration_s,
            "max_reference_images": caps.max_reference_images,
            "max_primary_shots": max_primary_shots,
            "max_pxx": max_pxx,
            "checks": capability_checks,
        },
        "regression": regression,
        "expectation_counts": {
            "character_entities": character_entities,
            "character_instances": character_instances,
            "required_event_count": len(expectations["required_events"]),
        },
        "source_structure": structural,
        "authorized_hard_limits": hard_limits,
        "missing_configuration": missing,
        "next_stage": "paid_full_chain" if not missing else "stop_zero_request",
    }


class _AcceptancePhaseOwner:
    """Narrow test-only injection that keeps every production Phase owner."""

    def __init__(
        self,
        workspace: Path,
        *,
        phase1_request_limit: int | None = None,
    ) -> None:
        self.phase1_request_ledger = (
            _BoundedPaidRequestLedger(
                workspace,
                "phase1_provider",
                phase1_request_limit,
            )
            if phase1_request_limit is not None
            else None
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(pipeline_lifecycle, name)

    def run_phase1(self, *args: Any, **kwargs: Any) -> dict:
        if self.phase1_request_ledger is not None:
            with provider_attempt_scope(
                max_retries=0,
                before_provider_request=self.phase1_request_ledger.before,
                after_provider_request=self.phase1_request_ledger.after,
                failed_provider_request=self.phase1_request_ledger.failed,
            ):
                return pipeline_lifecycle.run_phase1(*args, **kwargs)
        return pipeline_lifecycle.run_phase1(*args, **kwargs)

    @staticmethod
    def run_phase3(output_dir: Path, characters_data: dict, dry_run: bool) -> dict:
        return run_phase3_owner(
            output_dir,
            characters_data,
            dry_run,
            _acceptance_disable_provider_retries=True,
        )

    @staticmethod
    def run_storyboard_qa_gate(*args: Any, **kwargs: Any) -> dict:
        kwargs["structured_understanding_max_attempts"] = 1
        return run_storyboard_qa_gate(*args, **kwargs)

    @staticmethod
    def run_storyboard_qa_with_correction(
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        kwargs["max_correction_attempts"] = 0
        return run_storyboard_qa_with_correction(*args, **kwargs)

    @staticmethod
    def run_phase6(*args: Any, **kwargs: Any) -> dict:
        kwargs.update(
            _acceptance_disable_provider_repairs=True,
            _acceptance_disable_continuity_repairs=True,
        )
        return run_phase6_owner(*args, **kwargs)


def _pipeline_arguments(workspace: Path, story_path: Path) -> dict[str, Any]:
    return {
        "input_file": str(story_path),
        "duration": TARGET_DURATION_S,
        "shot_duration": AVG_SHOT_DURATION,
        "shot_policy": "continuity",
        "max_material_padding_ratio": 0.25,
        "delivery_overrun_ratio": 0.0,
        "chain_mode": True,
        "dry_run": False,
        "output_dir": str(workspace),
        "media_profile": MEDIA_PROFILE,
        "enable_reshoot": False,
        "character_visual_policy": VISUAL_POLICY,
        "phase1_semantic_qa": False,
        "auto_approve": True,
        "project_id": workspace.resolve().name,
        "character_library_dir": None,
    }


def _run_selected_phases(
    workspace: Path,
    story_path: Path,
    phases: set[float | int],
    *,
    resume: bool,
    phase1_request_limit: int | None = None,
    tos_upload_hard_limit: int | None = None,
) -> dict[str, Any]:
    owner = _AcceptancePhaseOwner(
        workspace,
        phase1_request_limit=phase1_request_limit,
    )
    result = run_pipeline(
        **_pipeline_arguments(workspace, story_path),
        skip_phase=[phase for phase in ALL_PHASES if phase not in phases],
        resume=resume,
        _phase_owner=owner,
        _tos_upload_hard_limit=tos_upload_hard_limit,
    )
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise RuntimeError(
            "acceptance lifecycle stage failed: "
            f"{result.get('error') if isinstance(result, dict) else result}"
        )
    receipts = result.get("phases") or {}
    for phase in phases:
        phase_name = "phase9_5" if phase == 9.5 else f"phase{int(phase)}"
        phase_receipt = receipts.get(phase_name) or {}
        if phase_receipt.get("status") not in {
            "done",
            "completed",
            "skipped",
        }:
            raise RuntimeError(
                f"acceptance lifecycle {phase_name} did not complete"
            )
    if owner.phase1_request_ledger is not None:
        result["acceptance_provider_ledger"] = (
            owner.phase1_request_ledger.settled_receipt()
        )
    return result


def _write_live_gate(
    workspace: Path,
    name: str,
    payload: dict[str, Any],
) -> Path:
    path = workspace / LIVE_GATES_DIRECTORY / f"{name}.json"
    _atomic_write_json(path, payload)
    return path


def _paid_request_summary(workspace: Path) -> dict[str, Any]:
    """Summarize durable paid attempts without copying payloads or secrets."""

    provider_request_count = 0
    provider_family_counts: dict[str, int] = {}
    request_receipts: list[dict[str, Any]] = []
    gate_dir = workspace / LIVE_GATES_DIRECTORY
    guard_paths = sorted({
        *gate_dir.glob("*_request.json"),
        *gate_dir.glob("*_requests.json"),
    })
    for path in guard_paths:
        receipt = _read_object(path)
        if receipt.get("schema") != PAID_REQUEST_GUARD_SCHEMA:
            continue
        count = int(receipt.get("provider_request_count") or 0)
        provider_request_count += count
        attempts = receipt.get("attempts")
        payloads = [
            attempt.get("payload") or {}
            for attempt in attempts
            if isinstance(attempt, dict)
        ] if isinstance(attempts, list) else []
        if not payloads:
            preflight_payload = (
                receipt.get("zero_submit_preflight") or {}
            ).get("payload")
            if isinstance(preflight_payload, dict):
                payloads = [preflight_payload]
        counted_families = 0
        for payload in payloads:
            family = str(payload.get("provider_family") or "").strip()
            if not family:
                continue
            provider_family_counts[family] = (
                provider_family_counts.get(family, 0) + 1
            )
            counted_families += 1
        if counted_families < count:
            family = str(receipt.get("request_name") or "unknown_provider")
            provider_family_counts[family] = (
                provider_family_counts.get(family, 0)
                + count
                - counted_families
            )
        request_receipts.append({
            "path": path.relative_to(workspace).as_posix(),
            "sha256": _sha256(path),
            "status": receipt.get("status"),
            "provider_request_count": count,
        })

    task_db = workspace / "runtime.db"
    if task_db.is_file():
        video_submissions = GenerationTaskStore(
            task_db
        ).submission_attempt_count()
        provider_request_count += video_submissions
        provider_family_counts["seedance_video"] = (
            provider_family_counts.get("seedance_video", 0)
            + video_submissions
        )
        request_receipts.append({
            "path": task_db.relative_to(workspace).as_posix(),
            "sha256": _sha256(task_db),
            "status": "generation_task_store",
            "provider_request_count": video_submissions,
        })

    tos_ledger_path = workspace / TOS_UPLOAD_LEDGER_NAME
    if tos_ledger_path.is_file():
        tos_ledger = _read_object(tos_ledger_path)
        if tos_ledger.get("schema") != TOS_UPLOAD_LEDGER_SCHEMA:
            raise RuntimeError("TOS upload ledger schema is incompatible")
        tos_submissions = int(tos_ledger.get("submission_attempt_count") or 0)
        provider_request_count += tos_submissions
        provider_family_counts["tos_media_upload"] = (
            provider_family_counts.get("tos_media_upload", 0) + tos_submissions
        )
        request_receipts.append({
            "path": tos_ledger_path.relative_to(workspace).as_posix(),
            "sha256": _sha256(tos_ledger_path),
            "status": "tos_upload_ledger",
            "provider_request_count": tos_submissions,
        })

    return {
        "provider_request_count": provider_request_count,
        "provider_family_counts": dict(sorted(provider_family_counts.items())),
        "paid_request_receipts": request_receipts,
    }


def _fact_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) >= {"value", "origin", "source_refs"}:
        return value["value"]
    if isinstance(value, dict):
        return {key: _fact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fact_value(item) for item in value]
    return value


def _expectation_matches(actual: Any, expected: Any) -> bool:
    actual = _fact_value(actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _expectation_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_expectation_matches(candidate, value) for candidate in actual)
            for value in expected
        )
    if isinstance(expected, str):
        return _normalized_source_text(expected) in _normalized_source_text(
            json.dumps(actual, ensure_ascii=False, sort_keys=True)
            if not isinstance(actual, str)
            else actual
        )
    return actual == expected


def validate_post_phase1_expectations(
    workspace: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Verify the external source anchors against current Phase 1 artifacts."""

    source = preflight.get("source") or {}
    expectations_path = _inside(
        workspace / str(source.get("expectations_path") or ""),
        workspace,
    )
    expectations = _read_object(expectations_path)
    expected_entities, expected_instances = _expectation_counts(expectations)
    characters = _read_object(workspace / "CHARACTERS.json")
    canonical = load_canonical_visual_contract(
        workspace,
        characters_data=characters,
    )
    roster = validate_character_roster(
        _read_object(workspace / "CHARACTER_ROSTER.json")
    )
    if canonical["character_roster_sha256"] != roster["roster_sha256"]:
        raise RuntimeError("Phase 1 canonical contract and roster hash disagree")
    records = canonical.get("characters") or []
    actual_instances = sum(
        len(record.get("instances") or [])
        for record in records
        if isinstance(record, dict)
    )
    if len(records) != expected_entities or actual_instances != expected_instances:
        raise RuntimeError(
            "Phase 1 character cardinality disagrees with acceptance expectations"
        )
    policy = (expectations.get("visual_facts") or {}).get(
        "character_visual_policy"
    )
    if policy and canonical.get("requested_policy") != policy:
        raise RuntimeError("Phase 1 character visual policy is not the expected policy")

    entity_matches: dict[str, str] = {}
    matched_entity_ids: set[str] = set()
    for item in expectations.get("entity_expectations") or []:
        anchors = {
            _normalized_source_text(value)
            for value in item["source_mentions_any"]
        }
        candidates = []
        for record in records:
            roster_record = next(
                (
                    entity
                    for entity in roster["entities"]
                    if entity["entity_id"] == record["entity_id"]
                ),
                None,
            )
            if roster_record is None:
                raise RuntimeError(
                    "canonical character is absent from the validated roster: "
                    + str(record.get("entity_id") or "<missing>")
                )
            # Identity anchors may only consume labels that Character Roster
            # bound to this entity. Narrative excerpts commonly describe
            # several co-present characters and are visual-fact evidence,
            # never an alias pool.
            record_source = [
                str(roster_record.get("display_name") or ""),
                *(
                    str(value or "")
                    for instance in roster_record.get("instances") or []
                    for value in instance.get("source_mentions") or []
                ),
            ]
            normalized_references = {
                _normalized_source_text(value)
                for value in record_source
                if str(value or "").strip()
            }
            if any(
                anchor == reference
                or anchor in reference
                or reference in anchor
                for anchor in anchors
                for reference in normalized_references
            ):
                candidates.append(record)
        if len(candidates) != 1:
            raise RuntimeError(
                "acceptance entity source anchors do not resolve uniquely: "
                + item["expectation_id"]
            )
        record = candidates[0]
        if record["entity_id"] in matched_entity_ids:
            raise RuntimeError("acceptance entity expectations overlap")
        if _fact_value(record["instance_count"]) != item["instance_count"]:
            raise RuntimeError(
                "acceptance entity instance count mismatch: "
                + item["expectation_id"]
            )
        if not _expectation_matches(record, item.get("visual_facts") or {}):
            raise RuntimeError(
                "acceptance entity visual facts mismatch: "
                + item["expectation_id"]
            )
        matched_entity_ids.add(record["entity_id"])
        entity_matches[item["expectation_id"]] = record["entity_id"]

    events_document = _read_object(workspace / "phase1_events.json")
    semantic = _read_object(workspace / "SEMANTIC_LEDGER.json")
    semantic_event_ids = {
        int(event["event_id"])
        for event in semantic.get("events") or []
        if isinstance(event, dict) and isinstance(event.get("event_id"), int)
    }
    required_event_matches: dict[str, int] = {}
    for required in expectations.get("required_events") or []:
        normalized_required = _normalized_source_text(required)
        candidates = []
        for position, event in enumerate(events_document.get("events") or [], 1):
            if not isinstance(event, dict):
                continue
            source_text = " ".join(
                str(event.get(field) or "")
                for field in ("source_excerpt", "what", "visual")
            )
            event_id = int(event.get("event_id") or event.get("id") or position)
            if (
                normalized_required in _normalized_source_text(source_text)
                and event_id in semantic_event_ids
            ):
                candidates.append(event_id)
        if not candidates:
            raise RuntimeError(
                "required source event is absent from the semantic ledger: "
                + str(required)
            )
        required_event_matches[str(required)] = min(candidates)

    return {
        "status": "passed",
        "canonical_visual_contract_sha256": canonical["contract_sha256"],
        "character_roster_sha256": roster["roster_sha256"],
        "character_entities": len(records),
        "character_instances": actual_instances,
        "entity_matches": entity_matches,
        "entity_source_anchor_authority": (
            "character_roster_display_and_instance_mentions"
        ),
        "required_event_matches": required_event_matches,
    }


def _freeze_post_phase1_budget(
    workspace: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    expectation_validation = validate_post_phase1_expectations(
        workspace,
        preflight,
    )
    canonical = _read_object(workspace / "CANONICAL_VISUAL_CONTRACT.json")
    storyboard = _read_object(workspace / "STORYBOARD.json")
    characters = canonical.get("characters") or []
    character_instances = sum(
        len(character.get("instances") or [])
        for character in characters
        if isinstance(character, dict)
    )
    shots = [value for value in storyboard.get("shots") or [] if isinstance(value, dict)]
    pxx = sum(len(shot.get("storyboard_beats") or []) for shot in shots)
    remaining_multimodal_requests = character_instances * 8 + len(shots) * 2 + pxx
    video_submission_limit = preflight["authorized_hard_limits"][
        "seedance_video_submissions"
    ]
    exact_upper_bounds = {
        "character_entities": len(characters),
        "character_instances": character_instances,
        "primary_shots": len(shots),
        "pxx": pxx,
        "remaining_seedream_image_requests": character_instances * 6 + len(shots) * 6,
        "remaining_multimodal_observation_requests": remaining_multimodal_requests,
        "remaining_tos_put_attempts": (
            remaining_multimodal_requests
            * preflight["authorized_hard_limits"][
                "max_tos_inputs_per_multimodal_request"
            ]
            + video_submission_limit
            * preflight["authorized_hard_limits"][
                "max_tos_inputs_per_video_submission"
            ]
        ),
    }
    initial = preflight["authorized_hard_limits"]
    if (
        exact_upper_bounds["remaining_seedream_image_requests"]
        > initial["seedream_image_requests"]
        or exact_upper_bounds["remaining_multimodal_observation_requests"]
        > initial["multimodal_observation_requests"]
        or exact_upper_bounds["remaining_tos_put_attempts"]
        > initial["tos_put_attempts"]
    ):
        raise RuntimeError("post-Phase-1 exact workload exceeds the authorized hard limit")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "frozen",
        "stage": "post_phase1_budget",
        "created_at": _utc_now(),
        "canonical_visual_contract_sha256": canonical.get("contract_sha256"),
        "expectation_validation": expectation_validation,
        "exact_upper_bounds": exact_upper_bounds,
        "provider_request_count": 0,
    }
    _write_live_gate(workspace, "post_phase1_budget", receipt)
    return receipt


def _content_prompt(content: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(item.get("text") or "")
        for item in content
        if item.get("type") == "text"
    )


def _freeze_phase6_tasks(
    workspace: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    plan = load_continuity_plan(workspace / "CONTINUITY_PLAN.json")
    chunk_ids = [chunk.chunk_id for shot in plan.shots for chunk in shot.chunks]
    bridge_ids = [bridge.bridge_id for bridge in plan.bridges]
    exact_submissions = len(chunk_ids) + len(bridge_ids)
    if exact_submissions < 1:
        raise RuntimeError("Phase 6 plan contains no video task")
    if exact_submissions > preflight["authorized_hard_limits"]["seedance_video_submissions"]:
        raise RuntimeError("Phase 6 exact task list exceeds the authorized hard limit")

    first_shot = plan.shots[0]
    first_chunk = first_shot.chunks[0]
    output_path = (
        workspace
        / "shots"
        / first_shot.shot_id
        / "chunks"
        / f"{first_chunk.chunk_id}.mp4"
    )
    request = ChunkExecutionRequest(
        resource_id=first_chunk.chunk_id,
        shot_id=first_shot.shot_id,
        chunk=first_chunk,
        anchors=first_shot.anchors.model_dump(mode="json"),
        output_path=output_path,
        previous_output_path=None,
        input_fingerprint=hashlib.sha256(
            json.dumps(
                first_chunk.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        memory_context="",
    )
    content, shot_meta, seed, duration = _provider_content(workspace, request)
    prompt = _content_prompt(content)
    prompt_path = workspace / LIVE_GATES_DIRECTORY / "phase6_p01_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    media_manifest = _media_index_manifest(content)
    ratio, _width, _height = _video_geometry(shot_meta)
    resolution = seedance_client.resolution_for_media_profile(
        MEDIA_PROFILE,
        SEEDANCE_MODEL,
    )
    task_payload = _task_payload(
        request,
        model=SEEDANCE_MODEL,
        provider_id="seedance",
        provider_version="ark-agent-plan-v3",
        project_id=workspace.resolve().name,
        run_id=workspace.name,
        duration=duration,
        seed=seed,
        generation_parameters={
            "ratio": ratio,
            "resolution": resolution,
            "return_last_frame": True,
            "media_index_manifest": media_manifest,
            "provider_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        },
    )
    shadow = write_shadow_runtime_report(workspace)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "frozen",
        "stage": "phase6_zero_submit_preflight",
        "created_at": _utc_now(),
        "provider_request_count": 0,
        "task_ids": chunk_ids,
        "bridge_ids": bridge_ids,
        "exact_video_submission_limit": exact_submissions,
        "first_task": {
            "chunk_id": first_chunk.chunk_id,
            "prompt_path": prompt_path.relative_to(workspace).as_posix(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "media_index_manifest": media_manifest,
            "generation_fingerprint": task_payload["generation_fingerprint"],
            "duration": duration,
            "ratio": ratio,
            "resolution": resolution,
        },
        "shadow_runtime": shadow,
        "automatic_provider_repairs": False,
        "automatic_continuity_repairs": False,
        "automatic_duration_topups": False,
    }
    _write_live_gate(workspace, "phase6_zero_submit_preflight", receipt)
    return receipt


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return float(completed.stdout.strip())


class _ProviderDenyPhaseOwner:
    """Fail immediately if a completed recovery tries to re-enter a Phase owner."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("run_"):
            def _deny(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    f"provider-deny recovery re-entered completed owner {name}"
                )

            return _deny
        return getattr(pipeline_lifecycle, name)


def _set_tree_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError as error:
            raise RuntimeError(f"cannot prepare recovery clone: {path}") from error


def _set_tree_readonly(root: Path) -> None:
    for path in [*root.rglob("*"), root]:
        try:
            path.chmod(
                path.stat().st_mode
                & ~stat.S_IWUSR
                & ~stat.S_IWGRP
                & ~stat.S_IWOTH
            )
        except OSError as error:
            raise RuntimeError(f"cannot seal immutable snapshot: {path}") from error


def verify_completed_recovery_matrix(
    workspace: Path,
    *,
    final_video_sha256: str,
) -> dict[str, Any]:
    """Clone a completed run and prove five resume boundaries are zero-request.

    ``resume_from`` is intentionally not used: that option invalidates downstream
    artifacts and authorizes recomputation.  Recovery instead validates each
    boundary's artifacts and asks Lifecycle to resume the completed checkpoint;
    any Phase owner call is denied.
    """
    from utils.artifact_chain import can_resume_from

    root = workspace.parent / f"{workspace.name}-recovery-matrix"
    if root.exists():
        raise RuntimeError(
            f"recovery matrix destination already exists and will not be overwritten: {root}"
        )
    root.mkdir(parents=True)
    source = root / "immutable-source"
    shutil.copytree(workspace, source)
    boundaries = ("phase1", "phase3", "phase5", "phase6", "phase8")
    rows: list[dict[str, Any]] = []
    try:
        for boundary in boundaries:
            clone = root / f"recovery-{boundary}"
            shutil.copytree(source, clone)
            _set_tree_writable(clone)
            if not can_resume_from(boundary, clone):
                raise RuntimeError(
                    f"{boundary} recovery prerequisite validation failed"
                )
            task_store = GenerationTaskStore(clone / "runtime.db")
            submissions_before = task_store.submission_attempt_count()
            story_path = clone / "input" / "story.txt"
            result = pipeline_lifecycle._run_pipeline(
                **_pipeline_arguments(clone, story_path),
                resume=True,
                _phase_owner=_ProviderDenyPhaseOwner(),
                _force_sequential=True,
            )
            if result.get("status") != "completed":
                raise RuntimeError(
                    f"{boundary} provider-deny resume did not complete"
                )
            submissions_after = GenerationTaskStore(
                clone / "runtime.db"
            ).submission_attempt_count()
            clone_video = clone / "polished.mp4"
            clone_sha256 = _sha256(clone_video)
            if submissions_after != submissions_before:
                raise RuntimeError(
                    f"{boundary} recovery added a Provider submission"
                )
            if clone_sha256 != final_video_sha256:
                raise RuntimeError(
                    f"{boundary} recovery changed the final video hash"
                )
            rows.append({
                "boundary": boundary,
                "status": "passed",
                "provider_request_count": 0,
                "submission_attempt_count_before": submissions_before,
                "submission_attempt_count_after": submissions_after,
                "final_video_sha256": clone_sha256,
            })
        _set_tree_readonly(source)
    except BaseException:
        # Keep every partial clone as audit evidence; never erase a failed matrix.
        raise
    receipt = {
        "schema": RECOVERY_MATRIX_SCHEMA,
        "status": "passed",
        "created_at": _utc_now(),
        "source_snapshot": str(source),
        "source_snapshot_readonly": True,
        "provider_request_count": 0,
        "final_video_sha256": final_video_sha256,
        "boundaries": rows,
    }
    _write_live_gate(workspace, "recovery_matrix", receipt)
    return receipt


def execute_paid_full_chain(
    workspace: Path,
    story_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Execute paid work through three durable gates with no acceptance retry."""
    if preflight.get("status") != "preflight_passed":
        raise RuntimeError("paid execution refused because Stage 0 did not pass")
    receipt_path = workspace / RECEIPT_NAME
    existing = _read_object(receipt_path) if receipt_path.is_file() else None
    if existing and existing.get("status") == "pending_business_verdict":
        return existing
    if existing and existing.get("status") in {
        "full_chain_failed",
        "live_acceptance_failed",
    }:
        raise RuntimeError(
            "failed paid acceptance is audit-only; automatic paid retry is forbidden"
        )
    if existing and existing.get("status") == "full_chain_running":
        expected_source = preflight.get("source") or {}
        actual_source = existing.get("source") or {}
        immutable_keys = (
            "git_commit",
            "story_sha256",
            "expectations_sha256",
        )
        if any(
            actual_source.get(key) != expected_source.get(key)
            for key in immutable_keys
        ):
            raise RuntimeError(
                "running paid acceptance cannot resume against changed source"
            )
        started = existing
    else:
        started = {
            **preflight,
            "status": "full_chain_running",
            "stage": "paid_full_chain",
            "paid_execution_started_at": _utc_now(),
            "retry_policy": (
                "no image-pack replay, no Phase 5 correction, no Seedance "
                "policy/seam/duration repair, no automatic reshoot"
            ),
            "gates": {},
        }
    for guard_path in sorted({
        *(workspace / LIVE_GATES_DIRECTORY).glob("*_request.json"),
        *(workspace / LIVE_GATES_DIRECTORY).glob("*_requests.json"),
    }):
        guard = _read_object(guard_path)
        if guard.get("status") in {
            "submission_uncertain",
            "provider_failed",
        }:
            raise RuntimeError(
                "paid acceptance has an unresolved submission_uncertain "
                f"request: {guard_path.name}"
            )
    tos_ledger_path = workspace / TOS_UPLOAD_LEDGER_NAME
    if tos_ledger_path.is_file():
        tos_ledger = _read_object(tos_ledger_path)
        if tos_ledger.get("schema") != TOS_UPLOAD_LEDGER_SCHEMA:
            raise RuntimeError("paid acceptance TOS upload ledger is incompatible")
        uploads = tos_ledger.get("uploads")
        if not isinstance(uploads, list):
            raise RuntimeError("paid acceptance TOS upload records are invalid")
        unresolved = [
            upload
            for upload in uploads
            if isinstance(upload, dict)
            and upload.get("status") in {
                "submission_uncertain",
                "provider_rejected",
            }
        ]
        if unresolved:
            raise RuntimeError(
                "paid acceptance has an unresolved or rejected TOS upload; "
                "automatic resubmission is forbidden"
            )
        if int(tos_ledger.get("submission_attempt_count") or 0) > int(
            preflight["authorized_hard_limits"]["tos_put_attempts"]
        ):
            raise RuntimeError("paid acceptance crossed its TOS PUT hard limit")
    _atomic_write_json(receipt_path, started)
    acceptance_environment = {
        "HONCUT_CONTINUITY_MODE": "auto",
        "HONCUT_CONTINUITY_MAX_REPAIRS": "0",
        "HONCUT_PHASE5_MAX_CORRECTIONS": "0",
        "VIDEO_GEN_CONCURRENCY": "1",
    }
    previous_environment = {
        name: os.environ.get(name) for name in acceptance_environment
    }
    os.environ.update(acceptance_environment)
    try:
        tos_upload_hard_limit = preflight["authorized_hard_limits"][
            "tos_put_attempts"
        ]
        phase1_result = _run_selected_phases(
            workspace,
            story_path,
            {1},
            resume=(workspace / "CANONICAL_VISUAL_CONTRACT.json").is_file(),
            phase1_request_limit=preflight["authorized_hard_limits"][
                "phase1_provider_requests"
            ],
            tos_upload_hard_limit=tos_upload_hard_limit,
        )
        phase1_ledger = phase1_result.get("acceptance_provider_ledger") or {}
        if (
            phase1_ledger.get("status") != "provider_completed"
            or int(phase1_ledger.get("provider_request_count") or 0) < 1
        ):
            raise RuntimeError("Phase 1 Provider ledger did not settle")
        _write_live_gate(workspace, "phase1_provider", phase1_ledger)
        post_phase1 = _freeze_post_phase1_budget(workspace, preflight)

        characters = _read_object(workspace / "CHARACTERS.json")
        if not (characters.get("characters") or []):
            raise RuntimeError("Phase 3 live gate requires at least one canonical character")
        phase3_gate = started["gates"].get("phase3")
        if not isinstance(phase3_gate, dict):
            phase3_image_guard = _SinglePaidRequestGuard(
                workspace,
                "phase3_identity_image",
            )
            phase3_qa_guard = _SinglePaidRequestGuard(
                workspace,
                "phase3_identity_qa",
            )

            def _phase3_request_hook(payload: dict[str, Any]) -> None:
                family = payload.get("provider_family")
                if family == "seedream_image":
                    phase3_image_guard(payload)
                elif family == "multimodal_observation":
                    phase3_qa_guard(payload)
                else:
                    raise RuntimeError(
                        f"unexpected Phase 3 paid Provider family: {family}"
                    )

            with tos_upload_execution_scope(
                workspace,
                max_submissions=tos_upload_hard_limit,
            ):
                phase3_gate = run_phase3_owner(
                    workspace,
                    characters,
                    False,
                    _acceptance_max_new_image_requests=1,
                    _acceptance_disable_provider_retries=True,
                    _acceptance_before_provider_request=_phase3_request_hook,
                )
            if (
                phase3_gate.get("status") != "acceptance_gate_passed"
                or phase3_gate.get("image_provider_request_count") != 1
                or phase3_gate.get("qa_provider_request_count") != 1
            ):
                raise RuntimeError("Phase 3 single-image live gate did not pass")
            image_guard_receipt = phase3_image_guard.complete(
                outcome={
                    "view_path": phase3_gate["view_path"],
                    "view_sha256": phase3_gate["view_sha256"],
                }
            )
            qa_guard_receipt = phase3_qa_guard.complete(
                outcome={
                    "qa_observation_id": phase3_gate["qa_observation_id"],
                    "qa_decision_id": phase3_gate["qa_decision_id"],
                    "qa_verdict": phase3_gate["qa_verdict"],
                }
            )
            phase3_gate = {
                **phase3_gate,
                "image_request_receipt_sha256": _sha256(
                    phase3_image_guard.path
                ),
                "qa_request_receipt_sha256": _sha256(
                    phase3_qa_guard.path
                ),
                "guarded_provider_request_count": (
                    image_guard_receipt["provider_request_count"]
                    + qa_guard_receipt["provider_request_count"]
                ),
            }
            _write_live_gate(workspace, "phase3_identity", phase3_gate)
            started["gates"]["phase3"] = phase3_gate
            _atomic_write_json(receipt_path, started)

        _run_selected_phases(
            workspace,
            story_path,
            {2},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )
        _run_selected_phases(
            workspace,
            story_path,
            {3},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )
        _run_selected_phases(
            workspace,
            story_path,
            {4},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )

        phase5_gate = started["gates"].get("phase5")
        if not isinstance(phase5_gate, dict):
            phase5_guard = _SinglePaidRequestGuard(
                workspace,
                "phase5_observation",
            )
            try:
                with tos_upload_execution_scope(
                    workspace,
                    max_submissions=tos_upload_hard_limit,
                ):
                    run_storyboard_qa_gate(
                        workspace,
                        structured_understanding_max_attempts=1,
                        _acceptance_max_new_observations=1,
                        _acceptance_before_provider_request=phase5_guard,
                    )
            except QAObservationGatePaused as paused:
                if (
                    paused.provider_request_count != 1
                    or paused.verdict not in {"pass", "acceptable_deviation"}
                ):
                    raise RuntimeError(
                        "Phase 5 single-observation live gate blocked"
                    ) from paused
                guard_receipt = phase5_guard.complete(outcome={
                    "qa_observation_id": paused.observation_id,
                    "qa_decision_id": paused.decision_id,
                    "qa_verdict": paused.verdict,
                })
                phase5_gate = {
                    "status": "passed",
                    "provider_request_count": paused.provider_request_count,
                    "qa_observation_id": paused.observation_id,
                    "qa_decision_id": paused.decision_id,
                    "qa_verdict": paused.verdict,
                    "request_receipt_sha256": _sha256(phase5_guard.path),
                    "request_fingerprint": guard_receipt[
                        "request_fingerprint"
                    ],
                }
            else:
                raise RuntimeError(
                    "Phase 5 live gate did not pause after exactly one new observation"
                )
            _write_live_gate(workspace, "phase5_observation", phase5_gate)
            started["gates"]["phase5"] = phase5_gate
            _atomic_write_json(receipt_path, started)

        _run_selected_phases(
            workspace,
            story_path,
            {5},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )
        phase6_preflight = _freeze_phase6_tasks(workspace, preflight)
        phase6_gate = started["gates"].get("phase6")
        if not isinstance(phase6_gate, dict):
            with tos_upload_execution_scope(
                workspace,
                max_submissions=tos_upload_hard_limit,
            ):
                phase6_gate = run_phase6_owner(
                    _read_object(workspace / "STORYBOARD.json"),
                    workspace,
                    False,
                    chain_mode=True,
                    media_profile=MEDIA_PROFILE,
                    _acceptance_max_new_chunks=1,
                    _acceptance_disable_provider_repairs=True,
                    _acceptance_disable_continuity_repairs=True,
                )
            if phase6_gate.get("status") != "acceptance_gate_passed":
                raise RuntimeError("Phase 6 P01 single-submit live gate did not pass")
            submission_count = GenerationTaskStore(
                workspace / "runtime.db"
            ).submission_attempt_count()
            if submission_count != 1:
                raise RuntimeError(
                    "Phase 6 P01 gate did not persist exactly one submission attempt"
                )
            phase6_gate = {
                **phase6_gate,
                "status": "passed",
                "submission_attempt_count": submission_count,
                "preflight_sha256": _sha256(
                    workspace
                    / LIVE_GATES_DIRECTORY
                    / "phase6_zero_submit_preflight.json"
                ),
            }
            _write_live_gate(workspace, "phase6_p01", phase6_gate)
            started["gates"]["phase6"] = phase6_gate
            _atomic_write_json(receipt_path, started)

        _run_selected_phases(
            workspace,
            story_path,
            {6},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )
        total_submissions = GenerationTaskStore(
            workspace / "runtime.db"
        ).submission_attempt_count()
        if total_submissions != phase6_preflight["exact_video_submission_limit"]:
            raise RuntimeError(
                "Phase 6 submission ledger differs from the frozen task list"
            )
        _run_selected_phases(
            workspace,
            story_path,
            {7, 8, 9, 9.5},
            resume=True,
            tos_upload_hard_limit=tos_upload_hard_limit,
        )

        final_video = workspace / "polished.mp4"
        if not final_video.is_file():
            raise RuntimeError("full chain produced no polished.mp4")
        actual_duration = _probe_duration(final_video)
        if not math.isclose(actual_duration, TARGET_DURATION_S, abs_tol=0.05):
            raise RuntimeError(
                f"final duration is {actual_duration:.3f}s, expected exactly 36s"
            )
        final_sha256 = _sha256(final_video)
        recovery_matrix = verify_completed_recovery_matrix(
            workspace,
            final_video_sha256=final_sha256,
        )
        completed = {
            **started,
            **_paid_request_summary(workspace),
            "status": "pending_business_verdict",
            "finished_at": _utc_now(),
            "pipeline_status": "completed",
            "call_chain_verdict": "pass",
            "business_verdict": "pending_manual_and_phase9_5_evidence",
            "post_phase1_budget": post_phase1,
            "phase6_frozen_tasks": phase6_preflight,
            "video_submission_attempt_count": total_submissions,
            "final_video": {
                "path": "polished.mp4",
                "sha256": final_sha256,
                "duration_s": actual_duration,
            },
            "recovery_matrix": recovery_matrix,
        }
        _atomic_write_json(receipt_path, completed)
        return completed
    except BaseException as error:
        failed = {
            **started,
            **_paid_request_summary(workspace),
            "status": "live_acceptance_failed",
            "finished_at": _utc_now(),
            "call_chain_verdict": "failed",
            "business_verdict": "not_evaluated",
            "safe_error": f"{type(error).__name__}: {error}",
        }
        _atomic_write_json(receipt_path, failed)
        raise
    finally:
        for name, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--story", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    preflight = build_stage0_preflight(workspace, args.story, args.expectations)
    if not args.submit:
        _atomic_write_json(workspace / RECEIPT_NAME, preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["status"] == "preflight_passed" else 2
    result = execute_paid_full_chain(workspace, args.story.resolve(), preflight)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pending_business_verdict" else 1


if __name__ == "__main__":
    raise SystemExit(main())
