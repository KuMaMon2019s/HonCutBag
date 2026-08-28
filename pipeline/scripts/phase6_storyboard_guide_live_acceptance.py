#!/usr/bin/env python3
# ruff: noqa: E402
"""Accept one Phase 6 P01 narrative-guide request against live Seedance.

Without ``--submit`` this command performs the required no-video-submit
preflight and persists the exact media index, prompt hash, input hashes, and
generation fingerprint. A submitted acceptance is permanently capped at one
logical submit and one raw Seedance POST. Poll recovery never resubmits.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from clients import seedance_client
from clients.tos_uploader import is_media_upload_configured
from phases.phase2.shot_storyboards import (
    migrate_shot_storyboard_narrative_guides,
    validate_shot_storyboard_artifacts,
)
from phases.phase4.cinematic_first_frames import (
    migrate_cinematic_first_frames,
    validate_cinematic_first_frame_artifacts,
)
from phases.phase4.continuity_plan import write_continuity_plan
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import (
    _media_index_manifest,
    _provider_content,
    _provider_ready_content,
    _task_payload,
    _video_geometry,
)
from runtime.execution_errors import ProviderJobFailedError
from runtime.generation_tasks import GenerationTaskStore
from runtime.security_boundaries import redact_text
from runtime.seedance_execution import execute_seedance_video_task
from schemas.continuity import ContinuityPlan, GenerationChunk
from utils.config import ARK_AGENT_CREDENTIAL_SOURCE, SEEDANCE_MODEL, get_api_key
from utils.video_validation import is_valid_video

RECEIPT_SCHEMA = "honcut.phase6-storyboard-guide-live-acceptance.v1"
REGRESSION_SCHEMA = "honcut.phase6-storyboard-guide-regression.v1"
RECEIPT_NAME = "phase6_storyboard_guide_live_acceptance.json"
ACCEPTANCE_DIRECTORY = Path("live_acceptance") / "phase6_storyboard_guide"
MAX_PAID_PROVIDER_REQUESTS = 1
REQUIRED_ACCEPTANCE_GATES = ["regression", "live_paid_provider"]


class ProviderRequestLimitError(RuntimeError):
    """The acceptance tried to cross its one-submit boundary again."""


class SinglePaidRequestTransport:
    """Hard-limit the raw Seedance submission POST while leaving polling alone."""

    def __init__(self) -> None:
        self.provider_request_attempt_count = 0
        self.provider_request_count = 0
        self.blocked_provider_request_count = 0
        self.request_contract: dict[str, Any] | None = None
        self._original_post: Callable[..., Any] | None = None

    def __enter__(self) -> SinglePaidRequestTransport:
        self._original_post = seedance_client.requests.post
        seedance_client.requests.post = self._post
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._original_post is not None:
            seedance_client.requests.post = self._original_post

    def _post(self, *args: Any, **kwargs: Any) -> Any:
        self.provider_request_attempt_count += 1
        if self.provider_request_count >= MAX_PAID_PROVIDER_REQUESTS:
            self.blocked_provider_request_count += 1
            raise ProviderRequestLimitError(
                "Phase 6 live acceptance permits exactly one paid Provider request"
            )
        if self._original_post is None:
            raise RuntimeError("Seedance transport guard is not active")
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            content = payload.get("content") or []
            prompt = "\n".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            self.request_contract = {
                "model": payload.get("model"),
                "duration": payload.get("duration"),
                "ratio": payload.get("ratio"),
                "resolution": payload.get("resolution"),
                "return_last_frame": payload.get("return_last_frame") is True,
                "watermark": payload.get("watermark"),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "media_roles": [
                    {
                        "type": item.get("type"),
                        "role": item.get("role"),
                    }
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") in {"image_url", "video_url"}
                ],
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


def _source_identity(*, require_clean: bool) -> dict[str, Any]:
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
    if require_clean and status:
        raise RuntimeError("live acceptance requires a committed, clean source worktree")
    return {"git_commit": revision, "worktree_clean": not bool(status)}


def _credential_readiness() -> dict[str, Any]:
    if not get_api_key("ARK_AGENT_API_KEY"):
        raise RuntimeError("ARK_AGENT_API_KEY is not configured")
    if not is_media_upload_configured():
        raise RuntimeError("TOS media upload credentials are not configured")
    return {
        "ark_agent_credential_source": ARK_AGENT_CREDENTIAL_SOURCE,
        "tos_media_upload_configured": True,
    }


def _shot_id(shot: dict[str, Any], position: int) -> str:
    raw = shot.get("shot_id") or shot.get("id") or position
    text = str(raw).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return f"S{int(text):02d}" if text.isdigit() else str(raw)


def _select_candidate(
    storyboard: dict[str, Any],
    plan: ContinuityPlan,
    requested_shot_id: str | None,
) -> tuple[dict[str, Any], GenerationChunk, list[str]]:
    plan_by_shot = {shot.shot_id: shot for shot in plan.shots}
    candidates = []
    for position, shot in enumerate(storyboard.get("shots") or [], 1):
        if not isinstance(shot, dict):
            continue
        shot_id = _shot_id(shot, position)
        if requested_shot_id and shot_id != requested_shot_id:
            continue
        beats = [
            beat
            for beat in (shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        if len(beats) < 2:
            continue
        first_beat = beats[0]
        visible = first_beat.get("character_ids")
        if visible is None:
            visible = shot.get("who") or shot.get("characters") or []
        if not isinstance(visible, list) or not 1 <= len(visible) <= 2:
            continue
        continuity_shot = plan_by_shot.get(shot_id)
        if continuity_shot is None or not continuity_shot.chunks:
            continue
        p01 = continuity_shot.chunks[0]
        if (
            p01.sequence != 1
            or p01.execution_strategy != "multi_image"
            or not p01.storyboard_beat_id
        ):
            continue
        candidates.append((shot, p01, [str(value) for value in visible]))
    if not candidates:
        scope = f" {requested_shot_id}" if requested_shot_id else ""
        raise RuntimeError(
            "no Phase 6 live candidate" + scope + " has >=2 Pxx and 1-2 visible characters"
        )
    if requested_shot_id and len(candidates) != 1:
        raise RuntimeError("explicit live acceptance shot did not resolve uniquely")
    return candidates[0]


def _content_prompt(content: list[dict[str, Any]]) -> str:
    return next(
        (
            str(item.get("text") or "")
            for item in content
            if item.get("type") == "text"
        ),
        "",
    )


def _preflight_contract(
    output_dir: Path,
    *,
    shot_id: str | None,
    require_clean_source: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = output_dir.resolve()
    storyboard = _read_json_object(output_dir / "STORYBOARD.json")
    phase2_manifest = migrate_shot_storyboard_narrative_guides(
        output_dir,
        storyboard,
    )
    phase4_manifest = migrate_cinematic_first_frames(output_dir, storyboard)
    scene_path = output_dir / "SCENE_CONSISTENCY.json"
    scene_consistency = (
        _read_json_object(scene_path) if scene_path.is_file() else {}
    )
    plan = write_continuity_plan(
        output_dir / "CONTINUITY_PLAN.json",
        storyboard,
        scene_consistency,
    )
    artifact_migration = {
        "phase2_kind": phase2_manifest.get("kind"),
        "phase2_migration": phase2_manifest.get("migration"),
        "phase4_kind": phase4_manifest.get("kind"),
        "phase4_migration": phase4_manifest.get("migration"),
        "continuity_plan_version": plan.version,
        "provider_request_count": 0,
    }
    storyboard_errors = validate_shot_storyboard_artifacts(output_dir, storyboard)
    if storyboard_errors:
        raise RuntimeError("Phase 2 preflight failed: " + "; ".join(storyboard_errors[:8]))
    cinematic_errors = validate_cinematic_first_frame_artifacts(
        output_dir,
        storyboard,
    )
    if cinematic_errors:
        raise RuntimeError("Phase 4 preflight failed: " + "; ".join(cinematic_errors[:8]))
    shot, chunk, visible_character_ids = _select_candidate(
        storyboard,
        plan,
        shot_id,
    )
    selected_shot_id = str(chunk.storyboard_beat_id).split("_P", 1)[0]
    if not chunk.storyboard_image or not chunk.storyboard_narrative_guide:
        raise RuntimeError("selected P01 lacks cinematic or narrative-guide provenance")
    if any(
        later.storyboard_image
        for continuity_shot in plan.shots
        if continuity_shot.shot_id == selected_shot_id
        for later in continuity_shot.chunks[1:]
    ):
        raise RuntimeError("selected Sxx still declares a P02+ cinematic frame")
    acceptance_dir = output_dir / ACCEPTANCE_DIRECTORY
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = acceptance_dir / f"{chunk.storyboard_beat_id}_prompt.txt"
    output_path = acceptance_dir / f"{chunk.storyboard_beat_id}.mp4"
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "chunk": chunk.model_dump(mode="json"),
                "visible_character_ids": visible_character_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    request = ChunkExecutionRequest(
        resource_id=f"LIVE_{chunk.chunk_id}",
        shot_id=selected_shot_id,
        chunk=chunk,
        anchors={},
        output_path=output_path,
        previous_output_path=None,
        input_fingerprint=input_fingerprint,
        memory_context="",
    )
    content, shot_meta, seed, duration = _provider_content(output_dir, request)
    prompt = _content_prompt(content)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    media_manifest = _media_index_manifest(content)
    image_media = [
        item for item in media_manifest if item.get("media_type") == "image_url"
    ]
    video_media = [
        item for item in media_manifest if item.get("media_type") == "video_url"
    ]
    guides = [
        item
        for item in image_media
        if item.get("responsibility") == "storyboard_narrative_guide"
    ]
    character_media = [
        item
        for item in image_media
        if item.get("responsibility") == "character_identity_board"
    ]
    required_prompt_fragments = (
        "当前剧情导航图是图片",
        "红色箭头表示主体或物体运动方向",
        "蓝色箭头表示摄影机运动",
        "不得提前演绎其他 Gxx 或后续 Pxx",
        "严禁渲染进视频画面",
    )
    if (
        not image_media
        or image_media[0].get("responsibility") != "cinematic_composition"
        or len(image_media) > 9
        or video_media
        or len(guides) != 1
        or len(character_media) != len(visible_character_ids)
        or any(fragment not in prompt for fragment in required_prompt_fragments)
    ):
        raise RuntimeError("selected P01 does not satisfy the final Phase 6 media contract")
    ratio, _width, _height = _video_geometry(shot_meta)
    resolution = seedance_client.resolution_for_media_profile("480p", SEEDANCE_MODEL)
    generation_parameters = {
        "ratio": ratio,
        "resolution": resolution,
        "return_last_frame": True,
        "seedance_prompt_contract": "all_modal_reference_with_narrative_guide_v2",
        "media_index_manifest": media_manifest,
        "provider_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    run_id = f"{output_dir.name}:phase6-storyboard-guide-live-v1"
    payload = _task_payload(
        request,
        model=SEEDANCE_MODEL,
        provider_id="seedance",
        provider_version="ark-agent-plan-v3",
        project_id="live-acceptance",
        run_id=run_id,
        duration=duration,
        seed=seed,
        generation_parameters=generation_parameters,
    )
    preflight = {
        "output_dir": str(output_dir),
        "shot_id": selected_shot_id,
        "beat_id": chunk.storyboard_beat_id,
        "p_count": len(shot.get("storyboard_beats") or []),
        "visible_character_ids": visible_character_ids,
        "narrative_cell_ids": list(chunk.storyboard_narrative_guide_cell_ids),
        "narrative_guide_sha256": chunk.storyboard_narrative_guide_sha256,
        "narrative_source_board_sha256": (
            chunk.storyboard_narrative_guide_source_board_sha256
        ),
        "cinematic_sha256": _sha256_file(output_dir / chunk.storyboard_image),
        "prompt_path": str(prompt_path.relative_to(output_dir)),
        "prompt_sha256": generation_parameters["provider_prompt_sha256"],
        "media_index_manifest": media_manifest,
        "image_count": len(image_media),
        "video_count": len(video_media),
        "duration": duration,
        "ratio": ratio,
        "resolution": resolution,
        "generation_fingerprint": payload["generation_fingerprint"],
        "provider_request_count": 0,
        "artifact_migration": artifact_migration,
        "source": _source_identity(require_clean=require_clean_source),
        "credentials": _credential_readiness(),
    }
    runtime = {
        "request": request,
        "content": content,
        "payload": payload,
        "run_id": run_id,
        "output_path": output_path,
        "task_store_path": acceptance_dir / "runtime.db",
        "duration": duration,
        "ratio": ratio,
        "resolution": resolution,
        "seed": seed,
    }
    return preflight, runtime


def _error_summary(error: BaseException) -> dict[str, Any]:
    safe = f"{type(error).__name__}: {redact_text(str(error))}"
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(safe.encode("utf-8")).hexdigest(),
    }


def _final_status(receipt: dict[str, Any]) -> str:
    gates = receipt.get("gates") or {}
    live = gates.get("live_paid_provider") or {}
    regression = gates.get("regression") or {}
    if live.get("status") == "failed" or live.get("business_verdict") == "fail":
        return "live_acceptance_failed"
    if (
        live.get("status") == "passed"
        and live.get("business_verdict") == "pass"
        and regression.get("status") == "passed"
    ):
        return "accepted"
    if live.get("status") == "submission_uncertain":
        return "submission_uncertain"
    if live.get("status") == "passed":
        return "pending_business_verdict"
    return "pending_live_acceptance"


def _write_contact_sheet(video_path: Path) -> Path | None:
    contact_sheet = video_path.with_name(f"{video_path.stem}_review_grid.jpg")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1,scale=480:-2,tile=3x3:padding=4:margin=4",
            "-frames:v",
            "1",
            str(contact_sheet),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return (
        contact_sheet
        if completed.returncode == 0 and contact_sheet.is_file()
        else None
    )


def _apply_regression_evidence(
    receipt: dict[str, Any],
    evidence_path: Path,
) -> None:
    evidence = _read_json_object(evidence_path)
    if evidence.get("schema") != REGRESSION_SCHEMA or evidence.get("status") != "passed":
        raise RuntimeError("regression evidence is not a passing current schema")
    expected_commit = ((receipt.get("preflight") or {}).get("source") or {}).get(
        "git_commit"
    )
    if (evidence.get("source") or {}).get("git_commit") != expected_commit:
        raise RuntimeError("regression evidence git commit differs from live preflight")
    receipt["gates"]["regression"] = {
        "status": "passed",
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256_file(evidence_path),
    }


def _preflight_replay_contract(preflight: dict[str, Any]) -> dict[str, Any]:
    """Select immutable fields that must match before the one live submit."""
    fields = (
        "shot_id",
        "beat_id",
        "p_count",
        "visible_character_ids",
        "narrative_cell_ids",
        "narrative_guide_sha256",
        "narrative_source_board_sha256",
        "cinematic_sha256",
        "prompt_sha256",
        "media_index_manifest",
        "image_count",
        "video_count",
        "duration",
        "ratio",
        "resolution",
        "generation_fingerprint",
        "artifact_migration",
    )
    contract = {field: preflight.get(field) for field in fields}
    contract["source_git_commit"] = (preflight.get("source") or {}).get(
        "git_commit"
    )
    return contract


def _reconcile_zero_request_preflight_failure(
    output_dir: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Release only a proven local failure that never reached raw transport."""
    live = (receipt.get("gates") or {}).get("live_paid_provider") or {}
    if not (
        receipt.get("status") == "submission_uncertain"
        and receipt.get("submitted") is True
        and int(receipt.get("provider_request_count") or 0) == 0
        and int(receipt.get("provider_request_attempt_count") or 0) == 0
        and not live.get("provider_job_id")
        and (live.get("error") or {}).get("type")
        in {"PromptBudgetExceededError", "DuplicatePromptContractError"}
    ):
        return receipt
    store = GenerationTaskStore(output_dir / receipt["task_store_path"])
    active = store.find_active(
        run_id=receipt["run_id"],
        task_type="video.generate",
        resource_id=f"LIVE_{receipt['preflight']['beat_id'].replace('_P', '_C')}",
        provider_id="seedance",
    )
    if active is None or active.status != "submission_uncertain" or active.provider_job_id:
        raise RuntimeError(
            "zero-request preflight failure cannot be reconciled against its task ledger"
        )
    store.resolve_unsubmitted_uncertain_as_failed(
        active.task_id,
        "confirmed local preflight failure before raw Seedance transport",
    )
    failure = {
        "status": "preflight_failed",
        "error": live.get("error"),
        "generation_task_id": active.task_id,
        "provider_request_count": 0,
        "provider_request_attempt_count": 0,
        "reconciled_at": _utc_now(),
    }
    receipt.setdefault("preflight_failures", []).append(failure)
    receipt.update({
        "status": "preflight_failed",
        "submitted": False,
        "provider_request_count": 0,
        "logical_submit_count": 0,
        "updated_at": _utc_now(),
    })
    live.update({
        "status": "preflight_failed",
        "call_chain_status": "preflight_failed",
    })
    _atomic_write_json(receipt_path, receipt)
    return receipt


def run_acceptance(
    output_dir: Path,
    *,
    submit: bool = False,
    resume_poll: bool = False,
    shot_id: str | None = None,
    business_verdict: str | None = None,
    verdict_notes: str = "",
    regression_evidence: Path | None = None,
    poll_attempts: int = 80,
    poll_interval: int = 15,
    preflight_builder: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        _preflight_contract
    ),
) -> dict[str, Any]:
    """Preflight, submit once, recover polling, or finalize the two gates."""
    output_dir = Path(output_dir).resolve()
    receipt_path = output_dir / RECEIPT_NAME
    existing = _read_json_object(receipt_path) if receipt_path.is_file() else None
    if existing is not None and existing.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError("unknown Phase 6 live acceptance receipt schema")
    if existing is not None:
        existing = _reconcile_zero_request_preflight_failure(
            output_dir,
            receipt_path,
            existing,
        )

    if business_verdict or regression_evidence is not None:
        if existing is None or existing.get("submitted") is not True:
            raise RuntimeError("cannot finalize an unsubmitted live acceptance")
        if regression_evidence is not None:
            _apply_regression_evidence(existing, regression_evidence)
        if business_verdict:
            live = existing["gates"]["live_paid_provider"]
            if live.get("call_chain_status") != "passed":
                raise RuntimeError("business verdict requires a passed live call chain")
            live["business_verdict"] = business_verdict
            live["business_verdict_notes"] = verdict_notes
            live["business_verdict_recorded_at"] = _utc_now()
            live["status"] = "passed" if business_verdict == "pass" else "failed"
        existing["status"] = _final_status(existing)
        existing["updated_at"] = _utc_now()
        _atomic_write_json(receipt_path, existing)
        return existing

    if existing is not None and existing.get("submitted") is True and not resume_poll:
        if submit:
            raise RuntimeError("this Phase 6 acceptance already consumed its live request")
        return existing

    if resume_poll:
        if existing is None or existing.get("submitted") is not True:
            raise RuntimeError("there is no submitted Phase 6 acceptance to resume")
        live = existing["gates"]["live_paid_provider"]
        provider_job_id = str(live.get("provider_job_id") or "")
        if not provider_job_id:
            raise RuntimeError("submission is uncertain and has no Provider task ID")
        runtime = {
            "payload": existing["task_payload"],
            "run_id": existing["run_id"],
            "output_path": output_dir / existing["output_path"],
            "task_store_path": output_dir / existing["task_store_path"],
        }
    else:
        preflight, runtime = preflight_builder(
            output_dir,
            shot_id=shot_id,
            require_clean_source=submit,
        )
        replace_failed_preflight = bool(
            existing is not None and existing.get("status") == "preflight_failed"
        )
        preflight_changed = bool(
            existing is not None
            and (
                _preflight_replay_contract(existing.get("preflight") or {})
                != _preflight_replay_contract(preflight)
                or existing.get("task_payload") != runtime["payload"]
            )
        )
        refresh_unsubmitted_preflight = bool(
            existing is not None
            and existing.get("submitted") is False
            and not submit
            and preflight_changed
        )
        if refresh_unsubmitted_preflight and int(
            existing.get("provider_request_count") or 0
        ) != 0:
            raise RuntimeError(
                "cannot refresh an unsubmitted preflight with Provider requests"
            )
        replace_preflight = (
            replace_failed_preflight or refresh_unsubmitted_preflight
        )
        if existing is not None and not replace_preflight:
            if preflight_changed:
                raise RuntimeError(
                    "live submit refused because the saved no-submit preflight changed"
                )
            existing["preflight_revalidated_at"] = _utc_now()
            existing["preflight"]["source"] = preflight["source"]
            if "credentials" in preflight:
                existing["preflight"]["credentials"] = preflight["credentials"]
            _atomic_write_json(receipt_path, existing)
            receipt = existing
        else:
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "status": "pending_live_acceptance",
                "submitted": False,
                "created_at": _utc_now(),
                "acceptance_scope": "Phase 1～Phase 9",
                "required_acceptance_gates": REQUIRED_ACCEPTANCE_GATES,
                "provider_request_limit": MAX_PAID_PROVIDER_REQUESTS,
                "provider_request_count": 0,
                "preflight": preflight,
                "run_id": runtime["run_id"],
                "output_path": str(runtime["output_path"].relative_to(output_dir)),
                "task_store_path": str(
                    runtime["task_store_path"].relative_to(output_dir)
                ),
                "task_payload": runtime["payload"],
                "gates": {
                    "regression": {"status": "pending"},
                    "live_paid_provider": {
                        "status": "pending",
                        "call_chain_status": "pending",
                        "business_verdict": "pending",
                    },
                },
            }
            if replace_preflight:
                receipt["preflight_failures"] = list(
                    existing.get("preflight_failures") or []
                )
            if refresh_unsubmitted_preflight:
                revisions = list(existing.get("preflight_revisions") or [])
                previous_preflight = existing.get("preflight") or {}
                revisions.append({
                    "reason": "unsubmitted_zero_request_preflight_refresh",
                    "source_git_commit": (
                        previous_preflight.get("source") or {}
                    ).get("git_commit"),
                    "prompt_sha256": previous_preflight.get("prompt_sha256"),
                    "provider_request_count": 0,
                    "refreshed_at": _utc_now(),
                })
                receipt["preflight_revisions"] = revisions
            _atomic_write_json(receipt_path, receipt)
        if not submit:
            return receipt
        existing = receipt

    assert existing is not None
    if not resume_poll:
        existing["submitted"] = True
        existing["status"] = "submission_uncertain"
        existing["submission_started_at"] = _utc_now()
        existing["gates"]["live_paid_provider"].update({
            "status": "submission_uncertain",
            "call_chain_status": "submission_uncertain",
        })
        _atomic_write_json(receipt_path, existing)

    store = GenerationTaskStore(runtime["task_store_path"])
    active = store.find_active(
        run_id=runtime["run_id"],
        task_type="video.generate",
        resource_id=f"LIVE_{existing['preflight']['beat_id'].replace('_P', '_C')}",
        provider_id="seedance",
    )
    known_job_id = str(
        existing["gates"]["live_paid_provider"].get("provider_job_id") or ""
    )
    if resume_poll and active is not None and not active.provider_job_id:
        store.persist_provider_job(
            active.task_id,
            provider_job_id=known_job_id,
            provider_endpoint=seedance_client.SUBMIT_ENDPOINT,
        )

    api_key = get_api_key("ARK_AGENT_API_KEY")
    if not api_key:
        raise RuntimeError("ARK_AGENT_API_KEY is not configured")
    logical_submit_attempt_count = 0
    logical_submit_count = 0
    transport: SinglePaidRequestTransport | None = None

    def submit_once() -> str:
        nonlocal logical_submit_attempt_count, logical_submit_count
        logical_submit_attempt_count += 1
        if resume_poll or logical_submit_count >= MAX_PAID_PROVIDER_REQUESTS:
            raise ProviderRequestLimitError(
                "Phase 6 live acceptance permits exactly one logical submit"
            )
        task_id = seedance_client.submit_content(
            _provider_ready_content(runtime["content"]),
            api_key=api_key,
            model=SEEDANCE_MODEL,
            duration=runtime["duration"],
            ratio=runtime["ratio"],
            resolution=runtime["resolution"],
            seed=runtime["seed"],
            return_last_frame=True,
            timeout=30,
        )
        logical_submit_count += 1
        existing["gates"]["live_paid_provider"]["provider_job_id"] = task_id
        existing["provider_request_count"] = logical_submit_count
        _atomic_write_json(receipt_path, existing)
        return task_id

    try:
        if resume_poll:
            execution = execute_seedance_video_task(
                store,
                run_id=runtime["run_id"],
                resource_id=f"LIVE_{existing['preflight']['beat_id'].replace('_P', '_C')}",
                payload=runtime["payload"],
                provider_endpoint=seedance_client.SUBMIT_ENDPOINT,
                output_path=runtime["output_path"],
                submit=lambda: (_ for _ in ()).throw(
                    ProviderRequestLimitError("poll recovery must never resubmit")
                ),
                poll=lambda task_id: seedance_client.poll(
                    task_id,
                    api_key,
                    max_attempts=poll_attempts,
                    interval=poll_interval,
                ),
                download=seedance_client.download,
                validate_output=is_valid_video,
            )
        else:
            with SinglePaidRequestTransport() as transport:
                execution = execute_seedance_video_task(
                    store,
                    run_id=runtime["run_id"],
                    resource_id=f"LIVE_{existing['preflight']['beat_id'].replace('_P', '_C')}",
                    payload=runtime["payload"],
                    provider_endpoint=seedance_client.SUBMIT_ENDPOINT,
                    output_path=runtime["output_path"],
                    submit=submit_once,
                    poll=lambda task_id: seedance_client.poll(
                        task_id,
                        api_key,
                        max_attempts=poll_attempts,
                        interval=poll_interval,
                    ),
                    download=seedance_client.download,
                    validate_output=is_valid_video,
                )
        output_path = Path(execution.output_path)
        contact_sheet = _write_contact_sheet(output_path)
        live = existing["gates"]["live_paid_provider"]
        live.update({
            "status": "passed",
            "call_chain_status": "passed",
            "business_verdict": live.get("business_verdict", "pending"),
            "provider_job_id": execution.provider_job_id,
            "generation_task_id": execution.task_id,
            "output_sha256": _sha256_file(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "contact_sheet": (
                str(contact_sheet.relative_to(output_dir)) if contact_sheet else None
            ),
            "completed_at": _utc_now(),
        })
        if transport is not None:
            existing.update({
                "provider_request_count": transport.provider_request_count,
                "provider_request_attempt_count": (
                    transport.provider_request_attempt_count
                ),
                "blocked_provider_request_count": (
                    transport.blocked_provider_request_count
                ),
                "logical_submit_count": logical_submit_count,
                "logical_submit_attempt_count": logical_submit_attempt_count,
            })
            live["transport_request_contract"] = transport.request_contract
            if (
                transport.provider_request_count != MAX_PAID_PROVIDER_REQUESTS
                or logical_submit_count != MAX_PAID_PROVIDER_REQUESTS
            ):
                raise RuntimeError("live acceptance did not cross exactly one submit boundary")
        existing["status"] = _final_status(existing)
        existing["updated_at"] = _utc_now()
        _atomic_write_json(receipt_path, existing)
        return existing
    except Exception as error:
        live = existing["gates"]["live_paid_provider"]
        task_id = str(live.get("provider_job_id") or "")
        terminal = isinstance(error, ProviderJobFailedError)
        zero_request_preflight_failure = bool(
            transport is not None
            and transport.provider_request_attempt_count == 0
            and not task_id
        )
        live.update({
            "status": (
                "preflight_failed"
                if zero_request_preflight_failure
                else "failed"
                if terminal
                else "submission_uncertain"
            ),
            "call_chain_status": (
                "preflight_failed"
                if zero_request_preflight_failure
                else "failed"
                if terminal
                else "submission_uncertain"
            ),
            "provider_job_id": task_id or None,
            "error": _error_summary(error),
            "failed_at": _utc_now(),
        })
        if transport is not None:
            existing.update({
                "provider_request_count": transport.provider_request_count,
                "provider_request_attempt_count": (
                    transport.provider_request_attempt_count
                ),
                "blocked_provider_request_count": (
                    transport.blocked_provider_request_count
                ),
                "logical_submit_count": logical_submit_count,
                "logical_submit_attempt_count": logical_submit_attempt_count,
            })
            live["transport_request_contract"] = transport.request_contract
        if zero_request_preflight_failure:
            active = store.find_active(
                run_id=runtime["run_id"],
                task_type="video.generate",
                resource_id=(
                    f"LIVE_{existing['preflight']['beat_id'].replace('_P', '_C')}"
                ),
                provider_id="seedance",
            )
            if active is None or active.provider_job_id:
                raise RuntimeError(
                    "local preflight failure has no safely releasable task ledger entry"
                ) from error
            store.resolve_unsubmitted_uncertain_as_failed(
                active.task_id,
                "confirmed local preflight failure before raw Seedance transport",
            )
            existing.setdefault("preflight_failures", []).append({
                "status": "preflight_failed",
                "error": live.get("error"),
                "generation_task_id": active.task_id,
                "provider_request_count": 0,
                "provider_request_attempt_count": 0,
                "reconciled_at": _utc_now(),
            })
            existing["submitted"] = False
            existing["status"] = "preflight_failed"
        else:
            existing["status"] = _final_status(existing)
        existing["updated_at"] = _utc_now()
        _atomic_write_json(receipt_path, existing)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shot-id")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume-poll", action="store_true")
    parser.add_argument("--business-verdict", choices=("pass", "fail"))
    parser.add_argument("--verdict-notes", default="")
    parser.add_argument("--regression-evidence", type=Path)
    parser.add_argument("--poll-attempts", type=int, default=80)
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()
    if args.submit and args.resume_poll:
        parser.error("--submit and --resume-poll are mutually exclusive")
    result = run_acceptance(
        args.output_dir,
        submit=args.submit,
        resume_poll=args.resume_poll,
        shot_id=args.shot_id,
        business_verdict=args.business_verdict,
        verdict_notes=args.verdict_notes,
        regression_evidence=args.regression_evidence,
        poll_attempts=args.poll_attempts,
        poll_interval=args.poll_interval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
