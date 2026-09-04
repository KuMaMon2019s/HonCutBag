#!/usr/bin/env python3
# ruff: noqa: E402
"""Accept one Phase 6 P01 narrative/performance-guide request against Seedance.

Without ``--submit`` this command performs the required no-video-submit
preflight and persists the exact media index, prompt hash, input hashes, and
generation fingerprint. A submitted acceptance is permanently capped at one
logical submit and one raw Seedance POST. Poll recovery never resubmits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from phases.phase3.performance_reference_board import (
    performance_prompt_optimization_contract,
    validate_character_performance_board,
    validate_character_performance_guide,
)
from phases.phase4.cinematic_first_frames import (
    migrate_cinematic_first_frames,
    validate_cinematic_first_frame_artifacts,
)
from phases.phase4.continuity_plan import write_continuity_plan
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import (
    MEDIA_ROLE_ISOLATION_CONTRACT,
    SEEDANCE_ALL_MODAL_PROMPT_CONTRACT,
    _media_index_manifest,
    _provider_content,
    _provider_prompt_metadata,
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
from utils.privacy_visual_policy import (
    SYNTHETIC_MAKEUP_PROFILE_ID,
    SYNTHETIC_STYLIZED_CHARACTER_POLICY,
    synthetic_character_review_evidence,
    synthetic_makeup_profile_sha256,
)
from utils.video_validation import is_valid_video

RECEIPT_SCHEMA = "honcut.phase6-storyboard-pose-atlas-live-acceptance.v1"
REGRESSION_SCHEMA = "honcut.phase6-storyboard-pose-atlas-regression.v1"
RECEIPT_NAME = "phase6_storyboard_pose_atlas_live_acceptance.json"
ACCEPTANCE_DIRECTORY = Path("live_acceptance") / "phase3_performance_board"
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
                "Phase 3 performance live acceptance permits exactly one paid Provider request"
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
            or not p01.character_performance_required
            or not p01.character_performance_guides
            or not any(guide.prop_ids for guide in p01.character_performance_guides)
        ):
            continue
        candidates.append((shot, p01, [str(value) for value in visible]))
    if not candidates:
        scope = f" {requested_shot_id}" if requested_shot_id else ""
        raise RuntimeError(
            "no Phase 3 performance live candidate"
            + scope
            + " has >=2 Pxx, 1-2 visible characters, and a P01 prop-bound action guide"
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


def _apply_action_window_contract(
    content: list[dict[str, Any]],
    *,
    action_window_seconds: float | None,
    duration_seconds: float,
    beat_id: str,
    cell_ids: list[str],
) -> dict[str, float] | None:
    """Time-box current-guide motion for an isolated live pacing experiment."""
    duration = float(duration_seconds)
    text_items = [item for item in content if item.get("type") == "text"]
    if len(text_items) != 1:
        raise ValueError("paced live acceptance requires exactly one text prompt")
    action_brief = text_items[0].get("_action_execution_brief")
    if isinstance(action_brief, dict):
        if not math.isclose(float(action_brief.get("duration_s") or 0), duration):
            raise ValueError("action brief duration differs from live output duration")
        completion = action_brief.get("completion_window_s") or []
        terminal = action_brief.get("terminal_hold") or {}
        if len(completion) != 2:
            raise ValueError("action brief completion window is invalid")
        target = float(terminal.get("target_duration_s") or 0)
        canonical_window = float(action_brief.get("target_completion_s") or 0)
        if not math.isclose(canonical_window + target, duration, abs_tol=0.05):
            raise ValueError("action brief timing budget differs from live output duration")
        if action_window_seconds is not None:
            requested = float(action_window_seconds)
            if (
                not math.isfinite(requested)
                or requested < float(completion[0])
                or requested > float(completion[1])
            ):
                raise ValueError(
                    "action window conflicts with the canonical action brief completion window"
                )
        return {
            "action_window_seconds": canonical_window,
            "end_state_hold_seconds": target,
        }
    if action_window_seconds is None:
        return None
    window = float(action_window_seconds)
    if not math.isfinite(window) or window <= 0 or window >= duration:
        raise ValueError(
            "action window must be finite, positive, and shorter than output duration"
        )
    ordered_cells = [str(value).strip() for value in cell_ids if str(value).strip()]
    if not ordered_cells:
        raise ValueError("paced live acceptance requires ordered narrative cells")
    hold_seconds = duration - window
    pacing_contract = (
        f"[honcut.live-paced-action-window.v1] 在输出的前{window:g}秒内，严格按"
        + "→".join(ordered_cells)
        + f"完成{beat_id}当前动作全过程；节奏紧凑但符合人体惯性，不得慢动作、"
        f"拖长单拍或用静止持姿代替动作。完成后约{hold_seconds:g}秒保持当前Pxx"
        "要求的终态，只允许自然呼吸、衣物余势、光影和环境运动，不得重播动作、"
        "提前演绎后续Pxx或新增剧情。"
    )
    prompt = str(text_items[0].get("text") or "").strip()
    if "[honcut.live-paced-action-window.v1]" in prompt:
        raise ValueError("paced action-window contract was injected more than once")
    text_items[0]["text"] = f"{prompt}\n{pacing_contract}".strip()
    return {
        "action_window_seconds": window,
        "end_state_hold_seconds": hold_seconds,
    }


def _is_current_synthetic_identity_contract(
    characters_payload: dict[str, Any],
    synthetic_evidence: dict[str, Any],
) -> bool:
    """Validate the current canonical policy projection without legacy aliases."""
    character_records = characters_payload.get("characters")
    resolved_policies = characters_payload.get("resolved_character_visual_policies")
    canonical_hash = str(
        characters_payload.get("canonical_visual_contract_sha256") or ""
    ).strip()
    character_ids = [
        str(character.get("id") or "").strip()
        for character in character_records or []
        if isinstance(character, dict)
    ]
    evidence_ids = [
        str(value).strip()
        for value in synthetic_evidence.get("synthetic_character_ids") or []
        if str(value).strip()
    ]
    return bool(
        characters_payload.get("character_visual_policy")
        == SYNTHETIC_STYLIZED_CHARACTER_POLICY
        and resolved_policies == [SYNTHETIC_STYLIZED_CHARACTER_POLICY]
        and len(canonical_hash) == 64
        and all(character in "0123456789abcdef" for character in canonical_hash)
        and isinstance(character_records, list)
        and character_records
        and len(character_ids) == len(character_records)
        and all(character_ids)
        and all(
            isinstance(character, dict)
            and character.get("visual_identity_policy")
            == SYNTHETIC_STYLIZED_CHARACTER_POLICY
            for character in character_records
        )
        and synthetic_evidence.get("all_characters_policy_tagged") is True
        and synthetic_evidence.get("identity_contract_complete") is True
        and evidence_ids == character_ids
    )


def _expected_p01_media_responsibilities(
    *,
    visible_character_count: int,
    performance_guide_count: int,
    guide_responsibilities: list[str] | None = None,
) -> list[str]:
    """Mirror the current Phase 6 P01 authority order in the live gate."""
    return [
        *(["character_identity_board"] * visible_character_count),
        "cinematic_composition",
        *(["character_performance_guide"] * performance_guide_count),
        *(guide_responsibilities or ["storyboard_narrative_guide"]),
    ]


def _preflight_contract(
    output_dir: Path,
    *,
    shot_id: str | None,
    require_clean_source: bool,
    action_window_seconds: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = output_dir.resolve()
    storyboard = _read_json_object(output_dir / "STORYBOARD.json")
    characters_payload = _read_json_object(output_dir / "CHARACTERS.json")
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
    synthetic_evidence = synthetic_character_review_evidence(output_dir)
    if not _is_current_synthetic_identity_contract(
        characters_payload,
        synthetic_evidence,
    ):
        raise RuntimeError(
            "selected run does not satisfy the current synthetic porcelain identity contract"
        )
    selected_shot_id = str(chunk.storyboard_beat_id).split("_P", 1)[0]
    if not chunk.storyboard_image or not (
        chunk.storyboard_pose_atlas_plan_schema
        or chunk.storyboard_narrative_guide
    ):
        raise RuntimeError("selected P01 lacks cinematic or pose-guide provenance")
    if any(
        later.storyboard_image
        for continuity_shot in plan.shots
        if continuity_shot.shot_id == selected_shot_id
        for later in continuity_shot.chunks[1:]
    ):
        raise RuntimeError("selected Sxx still declares a P02+ cinematic frame")
    performance_guides = []
    for declared in chunk.character_performance_guides:
        if declared.character_id not in visible_character_ids:
            raise RuntimeError(
                "selected P01 performance guide belongs to a non-visible character"
            )
        if not validate_character_performance_board(
            output_dir,
            declared.character_id,
        ):
            raise RuntimeError(
                f"{declared.character_id} performance board failed current validation"
            )
        validated = validate_character_performance_guide(
            output_dir,
            declared.character_id,
            str(chunk.storyboard_beat_id),
        )
        if validated is None:
            raise RuntimeError(
                f"{declared.character_id} current-Pxx performance guide is invalid"
            )
        expected = declared.model_dump(mode="json")
        if any(
            validated.get(field) != expected.get(field)
            for field in (
                "character_id",
                "beat_id",
                "image",
                "image_sha256",
                "cell_ids",
                "source_action_unit_ids",
                "prop_ids",
                "source_board",
                "source_board_sha256",
                "source_board_receipt",
                "source_board_receipt_sha256",
            )
        ):
            raise RuntimeError("selected performance guide differs from continuity provenance")
        performance_guides.append({
            "character_id": declared.character_id,
            "cell_ids": list(declared.cell_ids),
            "source_action_unit_ids": list(declared.source_action_unit_ids),
            "prop_ids": list(declared.prop_ids),
            "guide_sha256": declared.image_sha256,
            "source_board_sha256": declared.source_board_sha256,
        })
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
    pacing = _apply_action_window_contract(
        content,
        action_window_seconds=action_window_seconds,
        duration_seconds=duration,
        beat_id=str(chunk.storyboard_beat_id),
        cell_ids=(
            [
                str(sample.get("sample_id") or "")
                for sample in chunk.storyboard_pose_atlas_pose_samples
            ]
            if chunk.storyboard_pose_atlas_plan_schema
            else list(chunk.storyboard_narrative_guide_cell_ids)
        ),
    )
    prompt = _content_prompt(content)
    from utils.prompt_budget import enforce_prompt_budget

    enforce_prompt_budget(
        prompt,
        provider="seedance",
        model=SEEDANCE_MODEL,
        purpose="video_generation",
    )
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
        if item.get("responsibility")
        in {"storyboard_narrative_guide", "storyboard_pose_atlas"}
    ]
    character_media = [
        item
        for item in image_media
        if item.get("responsibility") == "character_identity_board"
    ]
    performance_media = [
        item
        for item in image_media
        if item.get("responsibility") == "character_performance_guide"
    ]
    expected_responsibilities = _expected_p01_media_responsibilities(
        visible_character_count=len(visible_character_ids),
        performance_guide_count=len(performance_guides),
        guide_responsibilities=[
            str(item.get("responsibility") or "") for item in guides
        ],
    )
    required_prompt_fragments = [MEDIA_ROLE_ISOLATION_CONTRACT]
    if chunk.storyboard_pose_atlas_plan_schema:
        required_prompt_fragments.extend((
            "[honcut.action-execution-brief.v1]",
            "[honcut.phase6-identity-projection.v1]",
            "媒体执行职责",
            "是当前动作顺序、根位移和重心轨迹权威",
            "零时长初始锚点",
            "所有动态动作必须在",
            "唯一主运镜",
            "不得把多姿态画成克隆、分栏、拼贴、网格、文字、序号、箭头或边框",
        ))
    else:
        required_prompt_fragments.extend((
            "珍珠生体瓷妆",
            "严禁渲染进视频画面",
            "当前动作姿态图中的多个人形是同一个角色的不同参考姿态",
            "只执行本次明确列出的 Axx",
            "当前剧情导航图是图片",
            "红色箭头表示主体或物体运动方向",
            "蓝色箭头表示摄影机运动",
            "不得提前演绎其他 Gxx 或后续 Pxx",
        ))
    if (
        not image_media
        or len(image_media) > 9
        or video_media
        or not guides
        or len(character_media) != len(visible_character_ids)
        or len(performance_media) != len(performance_guides)
        or [item.get("responsibility") for item in image_media]
        != expected_responsibilities
        or any(fragment not in prompt for fragment in required_prompt_fragments)
    ):
        raise RuntimeError("selected P01 does not satisfy the final Phase 6 media contract")
    ratio, _width, _height = _video_geometry(shot_meta)
    resolution = seedance_client.resolution_for_media_profile("480p", SEEDANCE_MODEL)
    generation_parameters = {
        "ratio": ratio,
        "resolution": resolution,
        "return_last_frame": True,
        "seedance_prompt_contract": SEEDANCE_ALL_MODAL_PROMPT_CONTRACT,
        "media_index_manifest": media_manifest,
        **_provider_prompt_metadata(content),
    }
    run_id = f"{output_dir.name}:phase6-storyboard-pose-atlas-live-v1"
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
        "synthetic_identity": {
            "character_visual_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
            "canonical_visual_contract_sha256": characters_payload[
                "canonical_visual_contract_sha256"
            ],
            "aesthetic_profile_id": SYNTHETIC_MAKEUP_PROFILE_ID,
            "aesthetic_profile_sha256": synthetic_makeup_profile_sha256(),
            "identity_contract_complete": True,
            "character_ids": [
                character["id"]
                for character in synthetic_evidence.get("characters") or []
            ],
        },
        "narrative_cell_ids": (
            [
                str(sample.get("sample_id") or "")
                for sample in chunk.storyboard_pose_atlas_pose_samples
            ]
            if chunk.storyboard_pose_atlas_plan_schema
            else list(chunk.storyboard_narrative_guide_cell_ids)
        ),
        "narrative_guide_sha256": chunk.storyboard_narrative_guide_sha256,
        "narrative_source_board_sha256": (
            chunk.storyboard_narrative_guide_source_board_sha256
        ),
        "pose_atlas": {
            "plan_schema": chunk.storyboard_pose_atlas_plan_schema,
            "plan_sha256": chunk.storyboard_pose_atlas_plan_sha256,
            "timing_contract": chunk.storyboard_pose_atlas_timing_contract,
            "camera_motion_contract_sha256": (
                chunk.storyboard_pose_atlas_camera_motion_contract_sha256
            ),
            "selected_strategy": (
                guides[0].get("pose_atlas_strategy")
                if chunk.storyboard_pose_atlas_plan_schema
                else None
            ),
            "page_count": (
                len(guides) if chunk.storyboard_pose_atlas_plan_schema else 0
            ),
            "page_sha256": (
                [item.get("sha256") for item in guides]
                if chunk.storyboard_pose_atlas_plan_schema
                else []
            ),
        },
        "cinematic_sha256": _sha256_file(output_dir / chunk.storyboard_image),
        "performance_guides": performance_guides,
        "performance_prompt_optimization": (
            performance_prompt_optimization_contract()
        ),
        "prompt_path": str(prompt_path.relative_to(output_dir)),
        "prompt_sha256": generation_parameters["provider_prompt_sha256"],
        "media_index_manifest": media_manifest,
        "image_count": len(image_media),
        "video_count": len(video_media),
        "duration": duration,
        "pacing": pacing,
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
    if ((receipt.get("preflight") or {}).get("source") or {}).get(
        "worktree_clean"
    ) is not True:
        raise RuntimeError("regression evidence requires a clean preflight source")
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
        "synthetic_identity",
        "narrative_cell_ids",
        "narrative_guide_sha256",
        "narrative_source_board_sha256",
        "pose_atlas",
        "cinematic_sha256",
        "performance_guides",
        "performance_prompt_optimization",
        "prompt_sha256",
        "media_index_manifest",
        "image_count",
        "video_count",
        "duration",
        "pacing",
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
    action_window_seconds: float | None = None,
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
        raise RuntimeError("unknown Phase 3 performance live acceptance receipt schema")
    if existing is not None:
        existing = _reconcile_zero_request_preflight_failure(
            output_dir,
            receipt_path,
            existing,
        )

    if business_verdict or regression_evidence is not None:
        if existing is None:
            raise RuntimeError("cannot update live acceptance before no-submit preflight")
        if business_verdict and existing.get("submitted") is not True:
            raise RuntimeError("cannot record a business verdict before live submission")
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
            raise RuntimeError(
                "this Phase 3 performance acceptance already consumed its live request"
            )
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
            action_window_seconds=action_window_seconds,
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
        store.confirm_provider_job(
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
                "Phase 3 performance live acceptance permits exactly one logical submit"
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
    parser.add_argument(
        "--action-window-seconds",
        type=float,
        help=(
            "Live-test-only time box for completing the current Gxx sequence; "
            "the remaining output holds the current Pxx end state"
        ),
    )
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
        action_window_seconds=args.action_window_seconds,
        poll_attempts=args.poll_attempts,
        poll_interval=args.poll_interval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
