#!/usr/bin/env python3
# ruff: noqa: E402
"""Accept one live Phase 5 post-adjudication PREVIS correction request.

Run this only against an isolated copy of a production run.  Without
``--submit`` it performs a zero-request preflight.  A submitted receipt is
permanently capped at one Seedream operation and one raw image-generation
POST for one explicitly pinned Pxx.  The live artifact is written outside
the canonical Phase 2/5 artifact tree: a complete correction also regenerates
its model-authored Sxx board and therefore cannot fit inside this one-request
acceptance boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from clients import seedream_client
from clients.seedream_client import SeedreamClient
from phases.phase2 import shot_storyboards
from phases.phase5 import storyboard_qa_gate
from prompt.seedream_image_prompt import (
    bind_reference_roles,
    prompt_guidance_metrics,
    single_image_request_parameters,
)
from runtime.security_boundaries import redact_text
from tools.character_reference_board import character_reference_role
from utils.config import ARK_AGENT_CREDENTIAL_SOURCE, get_api_key

RECEIPT_SCHEMA = "honcut.phase5-correction-resume-live-acceptance.v3"
RECEIPT_NAME = "phase5_correction_resume_live_acceptance.json"
ARTIFACT_SCHEMA = "honcut.phase5-correction-live-artifact.v1"
ARTIFACT_DIRECTORY = Path("live_acceptance") / "phase5_correction"
MAX_PAID_PROVIDER_REQUESTS = 1
REQUIRED_ACCEPTANCE_GATES = ["regression", "live_paid_provider"]


class ProviderRequestLimitError(RuntimeError):
    """The acceptance tried to cross its one-request Provider boundary."""


def _provider_error_summary(error: BaseException) -> dict[str, Any]:
    """Persist stable Provider evidence without response bodies or Prompt text."""
    safe_message = f"{type(error).__name__}: {redact_text(str(error))}"
    return {
        "type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "provider_code": str(getattr(error, "provider_code", "") or "") or None,
        "request_id": str(getattr(error, "request_id", "") or "") or None,
        "message_sha256": hashlib.sha256(safe_message.encode("utf-8")).hexdigest(),
    }


class SinglePaidRequestImageClient:
    """Guard one logical Seedream image operation."""

    def __init__(self, delegate: SeedreamClient) -> None:
        self._delegate = delegate
        self.image_operation_attempt_count = 0
        self.image_operation_count = 0
        self.blocked_image_operation_count = 0
        self.first_provider_error: dict[str, Any] | None = None

    def _invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        self.image_operation_attempt_count += 1
        if self.image_operation_count >= MAX_PAID_PROVIDER_REQUESTS:
            self.blocked_image_operation_count += 1
            raise ProviderRequestLimitError(
                "Phase 5 correction live acceptance permits exactly one "
                "Seedream image operation"
            )
        self.image_operation_count += 1
        try:
            return getattr(self._delegate, method_name)(*args, **kwargs)
        except Exception as exc:
            if self.first_provider_error is None:
                self.first_provider_error = _provider_error_summary(exc)
            raise

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
        self.request_contract: dict[str, Any] | None = None
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
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            prompt = str(payload.get("prompt") or "")
            self.request_contract = {
                "model": payload.get("model"),
                "size": payload.get("size"),
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "optimize_prompt_options": payload.get(
                    "optimize_prompt_options"
                ),
                "response_format": payload.get("response_format"),
                "sequential_image_generation": payload.get(
                    "sequential_image_generation"
                ),
                "output_format": payload.get("output_format"),
                "stream": payload.get("stream"),
                "watermark": payload.get("watermark"),
                "reference_count": (
                    len(payload.get("image"))
                    if isinstance(payload.get("image"), list)
                    else 1 if payload.get("image") else 0
                ),
            }
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


def _canonical_asset_hashes(output_dir: Path) -> dict[str, str]:
    """Hash the Phase 2/5 artifact surface that a live probe must not mutate."""
    paths: set[Path] = set()
    for name in ("STORYBOARD.json", "CHARACTERS.json", "SHOT_STORYBOARDS.json"):
        candidate = output_dir / name
        if candidate.is_file():
            paths.add(candidate)
    for directory_name in (
        "storyboard_beats",
        "shot_storyboards",
        "storyboard_images",
    ):
        directory = output_dir / directory_name
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    return {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(paths)
    }


def _acceptance_artifact_paths(
    output_dir: Path,
    storyboard_id: str,
) -> tuple[Path, Path]:
    artifact_dir = output_dir / ARTIFACT_DIRECTORY
    return (
        artifact_dir / f"{storyboard_id}.png",
        artifact_dir / f"{storyboard_id}.json",
    )


def _credential_readiness() -> dict[str, Any]:
    if not get_api_key("ARK_AGENT_API_KEY"):
        raise RuntimeError("ARK_AGENT_API_KEY is not configured")
    return {"ark_agent_credential_source": ARK_AGENT_CREDENTIAL_SOURCE}


def _source_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "live acceptance requires a committed, clean source worktree"
        )
    return {"git_commit": revision, "worktree_clean": True}


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
    preflight = {
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
        "canonical_asset_sha256": _canonical_asset_hashes(output_dir),
        "acceptance_artifact": str(
            _acceptance_artifact_paths(
                output_dir, expected_storyboard_id
            )[0].relative_to(output_dir)
        ),
        "source": _source_identity(),
        "credentials": _credential_readiness(),
    }
    preflight["prompt_projection"] = _prompt_projection_preflight(
        output_dir,
        state=state,
        storyboard_id=expected_storyboard_id,
        correction_attempt=attempts_used + 1,
    )
    return preflight


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


def _build_exact_pxx_request(
    output_dir: Path,
    *,
    state: dict[str, Any],
    storyboard_id: str,
    correction_attempt: int,
) -> dict[str, Any]:
    """Build the exact next Pxx request without entering the multi-asset Phase."""
    storyboard = _read_json_object(output_dir / "STORYBOARD.json")
    characters_path = output_dir / "CHARACTERS.json"
    characters_payload = (
        _read_json_object(characters_path)
        if characters_path.is_file()
        else {"characters": []}
    )
    characters = characters_payload.get("characters") or []
    if not isinstance(characters, list):
        raise ValueError("CHARACTERS.json characters must be a list")

    matches: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for shot in storyboard.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for position, beat in enumerate(shot.get("storyboard_beats") or [], 1):
            if (
                isinstance(beat, dict)
                and str(beat.get("beat_id") or "") == storyboard_id
            ):
                matches.append((shot, position, beat))
    if len(matches) != 1:
        raise RuntimeError("live acceptance target must resolve to exactly one Pxx")
    shot, position, beat = matches[0]
    beats = [
        value
        for value in (shot.get("storyboard_beats") or [])
        if isinstance(value, dict)
    ]
    issues = _issues_for_storyboard_id(state["result"], storyboard_id)
    directives = shot_storyboards._panel_correction_directives(
        issues,
        beat,
        storyboard_id,
    )
    if not directives:
        raise RuntimeError("live acceptance target has no correction DTO")
    beat_cast = shot_storyboards._beat_cast_contract(shot, beat, characters)
    character_references = shot_storyboards._character_reference_paths(
        output_dir,
        characters,
        beat_cast["who"],
    )
    content_positions = [
        index
        for index, value in enumerate(beats, 1)
        if str(value.get("generation_mode") or "").strip().lower()
        != "first_last_frame_bridge"
    ]
    manifest_path = output_dir / "SHOT_STORYBOARDS.json"
    manifest = (
        _read_json_object(manifest_path)
        if manifest_path.is_file()
        else {}
    )
    aspect_ratio = (
        str(manifest.get("aspect_ratio") or storyboard.get("aspect_ratio") or "")
        .strip()
        or "16:9"
    )
    panel_prompt = shot_storyboards._build_panel_prompt(
        shot,
        beat,
        position,
        len(beats),
        characters,
        uses_director_board=False,
        aspect_ratio=aspect_ratio,
        correction_contract=shot_storyboards._render_correction_contract(
            directives,
            attempt=correction_attempt,
        ),
        is_last_content_beat=position == max(content_positions, default=0),
        referenced_character_ids={
            path.parent.name for path in character_references
        },
    )
    provider_prompt = bind_reference_roles(
        panel_prompt,
        [character_reference_role(path) for path in character_references],
    )
    action_projection = shot_storyboards._compact(beat.get("action"), 500)
    observed_evidence = [
        str(value.get("observed_error") or "")
        for value in directives
        if str(value.get("observed_error") or "")
    ]
    checks = {
        "canonical_action_once": (
            bool(action_projection)
            and provider_prompt.count(action_projection) == 1
        ),
        "raw_observed_excluded": all(
            value not in provider_prompt for value in observed_evidence
        ),
        "correction_policy_current": (
            shot_storyboards.PANEL_CORRECTION_PROMPT_POLICY
            == "canonical-positive-projection-v2"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("live acceptance Provider Prompt projection failed checks")
    request_parameters = single_image_request_parameters(
        str(manifest.get("size_requested") or shot_storyboards.SHOT_STORYBOARD_SIZE)
    )
    transport_prompt, filtered_terms = seedream_client.sanitize_prompt(
        provider_prompt
    )
    if not transport_prompt.strip():
        raise RuntimeError("live acceptance Prompt is empty after IP filtering")
    safety_policy = (
        "non_graphic_staged_conflict_v1"
        if shot_storyboards._first_request_safety_contract(beat)
        else None
    )
    projection = {
        "panel_prompt_template_id": shot_storyboards.PANEL_PROMPT_TEMPLATE_ID,
        "panel_prompt_template_version": (
            shot_storyboards.PANEL_PROMPT_TEMPLATE_VERSION
        ),
        "correction_prompt_policy": (
            shot_storyboards.PANEL_CORRECTION_PROMPT_POLICY
        ),
        "first_request_safety_policy": safety_policy,
        "prompt_optimization": request_parameters["optimize_prompt_options"],
        "image_request_parameters": request_parameters,
        "provider_prompt_guidance": prompt_guidance_metrics(provider_prompt),
        "transport_prompt_guidance": prompt_guidance_metrics(transport_prompt),
        "ip_filtered_term_count": len(filtered_terms),
        "reference_count": len(character_references),
        "reference_roles": [
            character_reference_role(path) for path in character_references
        ],
        "reference_image_sha256": [
            _sha256_file(path) for path in character_references
        ],
        "action_sha256": hashlib.sha256(
            action_projection.encode("utf-8")
        ).hexdigest(),
        "observed_evidence_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in observed_evidence
        ],
        "checks": checks,
    }
    return {
        "provider_prompt": provider_prompt,
        "reference_paths": character_references,
        "size": str(
            manifest.get("size_requested")
            or shot_storyboards.SHOT_STORYBOARD_SIZE
        ),
        "model": str(
            manifest.get("model")
            or seedream_client.DEFAULT_MODEL
        ),
        "request_parameters": request_parameters,
        "projection": projection,
    }


def _prompt_projection_preflight(
    output_dir: Path,
    *,
    state: dict[str, Any],
    storyboard_id: str,
    correction_attempt: int,
) -> dict[str, Any]:
    """Project the exact next Provider request into receipt-safe metadata."""
    return _build_exact_pxx_request(
        output_dir,
        state=state,
        storyboard_id=storyboard_id,
        correction_attempt=correction_attempt,
    )["projection"]


def _image_artifact_contract(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Seedream live acceptance returned no image artifact")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "").upper()
    return {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "decodable": True,
    }


def run_acceptance(
    output_dir: Path,
    *,
    submit: bool,
    expected_storyboard_id: str,
    client_factory: Callable[[], SeedreamClient] = SeedreamClient,
) -> dict[str, Any]:
    """Preflight or execute one isolated live exact-Pxx Provider request."""
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
    target = str(preflight["storyboard_id"])
    artifact_path, artifact_sidecar_path = _acceptance_artifact_paths(
        output_dir, target
    )
    if artifact_path.exists() or artifact_sidecar_path.exists():
        raise RuntimeError("live acceptance artifact already exists")
    state = storyboard_qa_gate._load_completed_adjudication_correction(
        output_dir
    )
    if state is None or state.get("continuable") is not True:
        raise RuntimeError("Phase 5 correction continuation disappeared")
    exact_request = _build_exact_pxx_request(
        output_dir,
        state=state,
        storyboard_id=target,
        correction_attempt=int(preflight["correction_attempt"]),
    )
    if exact_request["projection"] != preflight.get("prompt_projection"):
        raise RuntimeError("exact Pxx request changed after preflight")
    receipt.update(
        status="submission_uncertain",
        submitted=True,
        submission_started_at=_utc_now(),
    )
    _atomic_write_json(receipt_path, receipt)

    limited_client: SinglePaidRequestImageClient | None = None
    transport: SinglePaidRequestTransport | None = None
    try:
        delegate = client_factory()
        observed_model = str(getattr(delegate, "model", "") or "")
        if observed_model != exact_request["model"]:
            raise RuntimeError(
                "live acceptance client model does not match the preflighted model"
            )
        limited_client = SinglePaidRequestImageClient(delegate)
        transport = SinglePaidRequestTransport()
        reference_paths = list(exact_request["reference_paths"])
        with transport:
            if reference_paths:
                result_url = limited_client.image_to_image(
                    prompt=str(exact_request["provider_prompt"]),
                    ref_image=(
                        str(reference_paths[0])
                        if len(reference_paths) == 1
                        else [str(path) for path in reference_paths]
                    ),
                    output_path=str(artifact_path),
                    size=str(exact_request["size"]),
                )
            else:
                result_url = limited_client.text_to_image(
                    prompt=str(exact_request["provider_prompt"]),
                    output_path=str(artifact_path),
                    size=str(exact_request["size"]),
                    timeout=180,
                )

        artifact = _image_artifact_contract(artifact_path)
        prompt_projection = preflight.get("prompt_projection") or {}
        result_source = "base64" if str(result_url).startswith("b64://") else "url"
        artifact_sidecar = {
            "schema": ARTIFACT_SCHEMA,
            "storyboard_id": target,
            "correction_attempt": int(preflight["correction_attempt"]),
            "image": str(artifact_path.relative_to(output_dir)),
            "image_sha256": artifact["sha256"],
            "image_bytes": artifact["bytes"],
            "image_width": artifact["width"],
            "image_height": artifact["height"],
            "image_format": artifact["format"],
            "provider_result_source": result_source,
            "provider_result_locator_sha256": hashlib.sha256(
                str(result_url).encode("utf-8")
            ).hexdigest(),
            "model": exact_request["model"],
            "size_requested": exact_request["size"],
            "provider_prompt_sha256": (
                prompt_projection.get("provider_prompt_guidance") or {}
            ).get("sha256"),
            "transport_prompt_sha256": (
                prompt_projection.get("transport_prompt_guidance") or {}
            ).get("sha256"),
            "prompt_optimization": prompt_projection.get(
                "prompt_optimization"
            ),
            "reference_roles": prompt_projection.get("reference_roles"),
            "reference_image_sha256": prompt_projection.get(
                "reference_image_sha256"
            ),
            "created_at": _utc_now(),
        }
        _atomic_write_json(artifact_sidecar_path, artifact_sidecar)
        canonical_assets_after = _canonical_asset_hashes(output_dir)
        transport_contract = transport.request_contract or {}
        business_assertions = {
            "exact_storyboard_scope": artifact_sidecar["storyboard_id"] == target,
            "generated_artifact_valid": artifact["decodable"] is True,
            "generated_pixel_differs_from_canonical": (
                artifact["sha256"] != preflight["asset_sha256"]
            ),
            "canonical_assets_unchanged": (
                canonical_assets_after == preflight["canonical_asset_sha256"]
            ),
            "prompt_projection_preflight_passed": bool(
                prompt_projection.get("checks")
            ) and all((prompt_projection.get("checks") or {}).values()),
            "transport_prompt_matches_preflight": (
                transport_contract.get("prompt_sha256")
                == (prompt_projection.get("transport_prompt_guidance") or {}).get(
                    "sha256"
                )
            ),
            "transport_contract_matches_preflight": (
                transport_contract.get("model") == exact_request["model"]
                and all(
                    transport_contract.get(field)
                    == exact_request["request_parameters"].get(field)
                    for field in (
                        "size",
                        "response_format",
                        "output_format",
                        "watermark",
                        "sequential_image_generation",
                        "stream",
                        "optimize_prompt_options",
                    )
                )
                and transport_contract.get("reference_count")
                == prompt_projection.get("reference_count")
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
            first_provider_error=limited_client.first_provider_error,
            provider_call_chain={
                "status": "passed",
                "strict_response_envelope": "validated_by_seedream_client",
                "artifact_download": "completed",
                "transport_contract": transport_contract,
            },
            model_business_verdict={
                "status": "not_provided",
                "reason": (
                    "Seedream generation returns an artifact, not a semantic "
                    "pass/block verdict"
                ),
            },
            artifact=artifact_sidecar,
            business_assertions=business_assertions,
            completed_at=_utc_now(),
        )
        accepted = (
            transport.provider_request_count == MAX_PAID_PROVIDER_REQUESTS
            and transport.provider_request_attempt_count
            == MAX_PAID_PROVIDER_REQUESTS
            and transport.blocked_provider_request_count == 0
            and limited_client.image_operation_count
            == MAX_PAID_PROVIDER_REQUESTS
            and limited_client.image_operation_attempt_count
            == MAX_PAID_PROVIDER_REQUESTS
            and limited_client.blocked_image_operation_count == 0
            and limited_client.first_provider_error is None
            and all(business_assertions.values())
        )
        receipt["status"] = "passed" if accepted else "live_acceptance_failed"
        if not accepted:
            receipt["error"] = (
                "live acceptance did not complete one isolated exact-Pxx request"
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
                first_provider_error=limited_client.first_provider_error,
            )
        receipt["provider_call_chain"] = {
            "status": "failed",
            "transport_contract": (
                transport.request_contract if transport is not None else None
            ),
        }
        receipt["model_business_verdict"] = {
            "status": "not_provided",
            "reason": "live Provider request did not complete acceptance",
        }
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
