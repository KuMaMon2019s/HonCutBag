#!/usr/bin/env python3
"""Prepare or execute one falsifiable Seedance 2.0 motion-blueprint gate.

The default mode is strictly local: it compiles and validates the blueprint and
writes a no-submit receipt.  ``--submit`` is intentionally separate and may only
be used after a current, explicit fee authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from acceptance.motion_blueprint import (  # noqa: E402
    CameraTrack,
    MotionBlueprintInput,
    MotionEvent,
    SourceLineage,
    assess_legacy_blueprint_manifest,
    combination_eligibility,
    compile_motion_blueprint,
    inspect_identity_neutral_pixels,
    measure_output_motion,
    sha256_file,
    technique_supports_contact,
)
from clients import seedance_client  # noqa: E402
from clients.tos_uploader import upload_media_file_required  # noqa: E402
from runtime.generation_tasks import GenerationTaskStore  # noqa: E402
from runtime.provider_attempt_policy import provider_attempt_scope  # noqa: E402
from runtime.seedance_execution import execute_seedance_video_task  # noqa: E402
from utils.canonical_visual_contracts import validate_canonical_visual_contract  # noqa: E402
from utils.config import SEEDANCE_MODEL, get_api_key  # noqa: E402
from utils.prompt_budget import enforce_prompt_budget  # noqa: E402
from utils.video_validation import is_valid_video  # noqa: E402

RECEIPT_SCHEMA = "honcut.seedance-motion-blueprint-gate.v5"
SOURCE_RECEIPT_SCHEMA = "honcut.phase6-storyboard-pose-atlas-live-acceptance.v1"
REGRESSION_SCHEMA = "honcut.seedance-motion-blueprint-regression.v1"
MAX_VIDEO_SUBMISSIONS = 1
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_IMAGES = 9
MAX_TOTAL_REFERENCE_BYTES = 64 * 1024 * 1024


class SingleSeedancePost:
    """Process-local raw POST guard for the one authorized Seedance submit."""

    def __init__(self) -> None:
        self.count = 0
        self._original = None

    def __enter__(self) -> SingleSeedancePost:
        self._original = seedance_client.requests.post

        def guarded(url: str, *args: Any, **kwargs: Any) -> Any:
            if url != seedance_client.SUBMIT_ENDPOINT:
                raise GateEvidenceError("capability gate attempted a non-Seedance POST")
            if self.count >= MAX_VIDEO_SUBMISSIONS:
                raise GateEvidenceError("capability gate blocked a second Seedance POST")
            self.count += 1
            assert self._original is not None
            return self._original(url, *args, **kwargs)

        seedance_client.requests.post = guarded
        return self

    def __exit__(self, *_args: object) -> None:
        if self._original is not None:
            seedance_client.requests.post = self._original


class GateEvidenceError(RuntimeError):
    """Input evidence cannot support a safe capability request."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateEvidenceError(f"cannot read JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise GateEvidenceError(f"expected JSON object: {path}")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _contained_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise GateEvidenceError(f"missing {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GateEvidenceError(f"{label} escapes source run") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateEvidenceError(f"missing {label}: {relative}")
    return path


def _validate_hash(path: Path, expected: object, *, label: str) -> str:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise GateEvidenceError(f"invalid {label} sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise GateEvidenceError(f"{label} sha256 mismatch")
    return actual


def _source_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"git_commit": commit, "tracked_worktree_clean": not bool(dirty)}


def _primitive_for_family(family: str) -> str:
    aliases = {"walk": "locomotion", "run": "locomotion", "neutral": "ready"}
    return aliases.get(family, family)


def _camera_from_samples(samples: list[dict[str, Any]]) -> CameraTrack:
    vectors = [((item.get("pose_contract") or {}).get("camera_vector") or {}) for item in samples]
    x = sum(float(vector.get("x") or 0) for vector in vectors)
    y = sum(float(vector.get("y") or 0) for vector in vectors)
    if abs(x) >= abs(y) and x:
        primitive = "pan_right" if x > 0 else "pan_left"
    elif y:
        primitive = "tilt_down" if y > 0 else "tilt_up"
    else:
        primitive = "static"
    return CameraTrack(primitive=primitive, magnitude=0.35 if primitive != "static" else 0.0)


def _build_contract(
    source_run: Path, source_receipt_path: Path
) -> tuple[MotionBlueprintInput, dict[str, Any], dict[str, Any]]:
    receipt = _read_object(source_receipt_path)
    if receipt.get("schema") != SOURCE_RECEIPT_SCHEMA:
        raise GateEvidenceError("unsupported source live-receipt schema")
    preflight = receipt.get("preflight") or {}
    receipt_beat_id = str(preflight.get("beat_id") or "")
    if not receipt_beat_id:
        raise GateEvidenceError("source receipt lacks canonical beat id")
    canonical_path = _contained_file(
        source_run, "CANONICAL_VISUAL_CONTRACT.json", label="canonical visual contract"
    )
    canonical = validate_canonical_visual_contract(_read_object(canonical_path))
    receipt_contract_hash = (preflight.get("synthetic_identity") or {}).get(
        "canonical_visual_contract_sha256"
    )
    if receipt_contract_hash and canonical["contract_sha256"] != receipt_contract_hash:
        raise GateEvidenceError("source receipt canonical contract hash mismatch")
    plan_path = _contained_file(source_run, "CONTINUITY_PLAN.json", label="continuity plan")
    plan = _read_object(plan_path)
    chunks = [
        chunk
        for shot in plan.get("shots") or []
        for chunk in shot.get("chunks") or []
        if isinstance(chunk, dict)
    ]
    if sum(chunk.get("storyboard_beat_id") == receipt_beat_id for chunk in chunks) != 1:
        raise GateEvidenceError("source beat does not resolve uniquely in continuity plan")
    lineage = SourceLineage(
        canonical_visual_contract_path=str(canonical_path.relative_to(source_run)),
        canonical_visual_contract_sha256=sha256_file(canonical_path),
        continuity_plan_path=str(plan_path.relative_to(source_run)),
        continuity_plan_sha256=sha256_file(plan_path),
        source_receipt_path=str(source_receipt_path.relative_to(source_run)),
        source_receipt_sha256=sha256_file(source_receipt_path),
    )
    candidates: list[tuple[MotionBlueprintInput, dict[str, Any], dict[str, Any]]] = []
    for chunk in chunks:
        beat_id = str(chunk.get("storyboard_beat_id") or "")
        samples = chunk.get("storyboard_pose_atlas_pose_samples") or []
        groups = chunk.get("storyboard_pose_atlas_action_groups") or []
        if not beat_id or not samples or not groups:
            continue
        actors = sorted(
            {
                str(actor)
                for sample in samples
                for actor in ((sample.get("pose_contract") or {}).get("actor_roles") or [])
                if str(actor)
            }
        )
        if len(actors) != 1:
            continue
        events: list[MotionEvent] = []
        for order, group in enumerate(
            sorted(groups, key=lambda item: int(item.get("order") or 0)), start=1
        ):
            group_id = str(group.get("action_group_id") or "")
            group_samples = [
                sample for sample in samples if sample.get("action_group_id") == group_id
            ]
            if not group_samples:
                raise GateEvidenceError(
                    f"action group {group_id} lacks pose classification evidence"
                )
            pose_contract = group_samples[-1].get("pose_contract") or {}
            group_lineage = group.get("lineage") or {}
            primitive = _primitive_for_family(str(pose_contract.get("pose_family") or ""))
            events.append(
                MotionEvent(
                    event_id=f"{beat_id}_M{order:02d}",
                    order=order,
                    actor_ids=tuple(actors),
                    primitive=primitive,
                    direction=str(pose_contract.get("direction") or "right"),
                    source_action_group_id=group_id,
                    source_action_unit_ids=tuple(
                        str(value)
                        for value in group_lineage.get("source_action_unit_ids") or []
                    ),
                    prop_contact=bool(pose_contract.get("object_roles"))
                    and technique_supports_contact(primitive),
                )
            )
        candidate = MotionBlueprintInput(
            beat_id=beat_id,
            duration_s=4.0,
            actor_ids=tuple(actors),
            events=tuple(events),
            camera=_camera_from_samples(samples),
            lineage=lineage,
        )
        eligibility = combination_eligibility(candidate)
        if eligibility["eligible"]:
            candidates.append((candidate, chunk, eligibility))
    if not candidates:
        raise GateEvidenceError(
            "verified continuity plan has no single-actor Pxx with three canonical dynamic actions"
        )
    candidates.sort(
        key=lambda item: (
            item[0].beat_id != receipt_beat_id,
            -int(item[2]["dynamic_action_count"]),
            -int(item[2]["distinct_dynamic_primitive_count"]),
            item[0].beat_id,
        )
    )
    contract, chunk, _eligibility = candidates[0]
    return contract, receipt, chunk


def _verified_source_media(source_run: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = (receipt.get("preflight") or {}).get("media_index_manifest") or []
    if not manifest:
        raise GateEvidenceError("source receipt lacks final media manifest")
    verified = []
    for item in manifest:
        if not isinstance(item, dict):
            raise GateEvidenceError("source media manifest is malformed")
        path = _contained_file(source_run, item.get("path"), label="source media")
        _validate_hash(path, item.get("sha256"), label="source media")
        verified.append({**item, "absolute_path": str(path)})
    return verified


def _project_request(
    source_run: Path, receipt: dict[str, Any], blueprint: dict[str, Any]
) -> dict[str, Any]:
    preflight = receipt.get("preflight") or {}
    if str(preflight.get("beat_id") or "") != str(
        (blueprint.get("measurements") or {}).get("beat_id") or ""
    ):
        raise GateEvidenceError(
            "selected combination lacks an exact production-equivalent request receipt"
        )
    source_task = receipt.get("task_payload") or {}
    if not source_task:
        raise GateEvidenceError("source receipt lacks a persisted production Phase 6 task")
    if source_task:
        if (
            source_task.get("phase6_prompt_projection_schema")
            != "honcut.phase6-prompt-projection.v1"
        ):
            raise GateEvidenceError("source request lacks the supported Phase 6 prompt projection")
        if source_task.get("media_index_manifest") != preflight.get("media_index_manifest"):
            raise GateEvidenceError("source Phase 6 media projection changed after persistence")
        expected_source_fields = {
            "model": source_task.get("model"),
            "duration": source_task.get("duration"),
            "ratio": source_task.get("ratio"),
            "resolution": source_task.get("resolution"),
        }
        actual_source_fields = {
            "model": SEEDANCE_MODEL,
            "duration": round(float(preflight.get("duration") or 0)),
            "ratio": preflight.get("ratio"),
            "resolution": preflight.get("resolution"),
        }
        if expected_source_fields != actual_source_fields:
            raise GateEvidenceError(
                "source production request differs from the frozen model/output profile"
            )
    source_media = _verified_source_media(source_run, receipt)
    atlas = [item for item in source_media if item.get("responsibility") == "storyboard_pose_atlas"]
    if len(atlas) != 1:
        raise GateEvidenceError(
            "single-variable gate requires exactly one failed pose-atlas control"
        )
    prompt_path = _contained_file(source_run, preflight.get("prompt_path"), label="source prompt")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    expected_prompt_hash = preflight.get("prompt_sha256")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != expected_prompt_hash:
        raise GateEvidenceError("source prompt hash mismatch")
    kept = [item for item in source_media if item.get("responsibility") != "storyboard_pose_atlas"]
    images = [item for item in kept if item.get("media_type") == "image_url"]
    videos = [item for item in kept if item.get("media_type") == "video_url"]
    if len(images) > MAX_REFERENCE_IMAGES or len(videos) + 1 > MAX_REFERENCE_VIDEOS:
        raise GateEvidenceError("projected Seedance reference-media budget exceeded")
    total_reference_bytes = sum(Path(item["absolute_path"]).stat().st_size for item in kept) + int(
        blueprint["media_size_bytes"]
    )
    if total_reference_bytes > MAX_TOTAL_REFERENCE_BYTES:
        raise GateEvidenceError("projected Seedance request exceeds the 64 MB media ceiling")
    old_index = str(atlas[0].get("prompt_index") or "")
    new_index = f"视频{len(videos) + 1}"
    prompt = prompt.replace(old_index, new_index)
    prompt += (
        "\n[honcut.motion-blueprint-reference.v5] "
        f"{new_index}仅负责当前Pxx的动作时序、身体运动学、接触时点与运镜轨迹；"
        "其中中性骨架和道具线条不具有角色身份、服装、脸、发型、材质或成片像素权威；"
        "不得把蓝图画面、骨架、控制标记或背景复制进成片。"
    )
    enforce_prompt_budget(
        prompt, provider="seedance", model=SEEDANCE_MODEL, purpose="video_generation"
    )
    media = []
    image_index = video_index = 0
    for item in kept:
        if item.get("media_type") == "image_url":
            image_index += 1
            index = f"图片{image_index}"
        else:
            video_index += 1
            index = f"视频{video_index}"
        media.append(
            {
                "media_type": item.get("media_type"),
                "role": item.get("role"),
                "responsibility": item.get("responsibility"),
                "prompt_index": index,
                "path": item.get("path"),
                "absolute_path": item["absolute_path"],
                "sha256": item.get("sha256"),
            }
        )
    video_index += 1
    media.append(
        {
            "media_type": "video_url",
            "role": "reference_video",
            "responsibility": "motion_blueprint",
            "prompt_index": f"视频{video_index}",
            "path": "motion_blueprint.mp4",
            "absolute_path": blueprint["media_path"],
            "sha256": blueprint["media_sha256"],
        }
    )
    if new_index != f"视频{video_index}":
        raise GateEvidenceError("projected prompt/video index mismatch")
    generation = {
        "model": SEEDANCE_MODEL,
        "duration": round(float(blueprint["measurements"]["duration_s"])),
        "ratio": preflight.get("ratio"),
        "resolution": preflight.get("resolution"),
        "return_last_frame": True,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "production_projection_schema": source_task.get(
            "phase6_prompt_projection_schema",
            "honcut.phase6-prompt-projection.v1",
        ),
        "source_generation_fingerprint": source_task.get("generation_fingerprint"),
        "source_prompt_projection_sha256": source_task.get("phase6_prompt_projection_sha256"),
        "motion_blueprint_schema": blueprint["schema"],
        "motion_technique_registry_sha256": blueprint["technique_registry_sha256"],
        "motion_semantic_frames_sha256": blueprint["semantic_frames_sha256"],
        "motion_techniques": [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "technique_id",
                    "technique_phase_ids",
                    "technique_contact_phase_ids",
                    "technique_keyframes_sha256",
                )
            }
            for event in blueprint["measurements"]["events"]
        ],
        "media": [
            {
                key: item[key]
                for key in (
                    "media_type",
                    "role",
                    "responsibility",
                    "prompt_index",
                    "path",
                    "sha256",
                )
            }
            for item in media
        ],
    }
    generation["task_fingerprint"] = _hash(generation)
    equivalence = {
        "identity_media_sha256": [
            item["sha256"] for item in media if item["responsibility"] == "character_identity_board"
        ],
        "start_frame_sha256": [
            item["sha256"] for item in media if item["responsibility"] == "cinematic_composition"
        ],
        "source_duration": round(float(preflight.get("duration") or 0)),
        "control_duration": generation["duration"],
        "candidate_duration": generation["duration"],
        "ratio": generation["ratio"],
        "resolution": generation["resolution"],
        "removed_control": {
            "responsibility": "storyboard_pose_atlas",
            "sha256": atlas[0]["sha256"],
        },
        "added_control": {
            "responsibility": "motion_blueprint",
            "sha256": blueprint["media_sha256"],
        },
        "source_task_input_fingerprint": source_task.get("input_fingerprint"),
    }
    return {
        "prompt": prompt,
        "media_runtime": media,
        "generation": generation,
        "equivalence": equivalence,
        "capabilities": {
            "output_duration_s": [4, 15],
            "reference_video_duration_s": [2, 15],
            "max_reference_images": MAX_REFERENCE_IMAGES,
            "max_reference_videos": MAX_REFERENCE_VIDEOS,
            "max_total_reference_bytes": MAX_TOTAL_REFERENCE_BYTES,
            "total_reference_bytes": total_reference_bytes,
        },
    }


def _assert_seedance_only(model: str) -> None:
    if not re.fullmatch(r"doubao-seedance-2\.0(?:-[a-z0-9._-]+)?", model):
        raise GateEvidenceError("capability gate requires the configured Seedance 2.0 model")


def _validate_regression(path: Path | None, commit: str) -> dict[str, Any]:
    if path is None:
        return {"status": "pending", "reason": "regression receipt not supplied"}
    evidence = _read_object(path.resolve())
    if evidence.get("schema") != REGRESSION_SCHEMA or evidence.get("status") != "passed":
        raise GateEvidenceError("regression receipt is not passing")
    if (evidence.get("source") or {}).get("git_commit") != commit:
        raise GateEvidenceError("regression receipt commit mismatch")
    return {"status": "passed", "path": str(path.resolve()), "sha256": sha256_file(path.resolve())}


def prepare_gate(
    source_run: Path,
    output_dir: Path,
    *,
    source_receipt_path: Path | None = None,
    regression_receipt: Path | None = None,
    legacy_manifest_path: Path | None = None,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_receipt_path = (
        source_receipt_path or source_run / "phase6_storyboard_pose_atlas_live_acceptance.json"
    ).resolve()
    try:
        source_receipt_path.relative_to(source_run)
    except ValueError as error:
        raise GateEvidenceError("source receipt must remain inside source run") from error
    _assert_seedance_only(SEEDANCE_MODEL)
    contract, source_receipt, _chunk = _build_contract(source_run, source_receipt_path)
    contract_path = output_dir / "motion_blueprint_contract.json"
    _write_object(contract_path, contract.model_dump(mode="json"))
    blueprint = compile_motion_blueprint(contract, output_dir / "motion_blueprint.mp4")
    if blueprint["measurements"]["combination"]["tempo_pass"] is not True:
        raise GateEvidenceError("selected canonical combination does not satisfy tempo policy")
    blueprint["pixel_guard"] = inspect_identity_neutral_pixels(Path(blueprint["media_path"]))
    manifest_path = output_dir / "motion_blueprint_manifest.json"
    _write_object(manifest_path, blueprint)
    source = _source_identity()
    regression = _validate_regression(regression_receipt, source["git_commit"])
    legacy_assessment = (
        assess_legacy_blueprint_manifest(legacy_manifest_path)
        if legacy_manifest_path is not None
        else None
    )
    receipt_beat_id = str((source_receipt.get("preflight") or {}).get("beat_id") or "")
    exact_request_projection = contract.beat_id == receipt_beat_id
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "pending_live_acceptance"
            if exact_request_projection
            else "pending_source_request_projection"
        ),
        "submitted": False,
        "provider": "seedance",
        "model": SEEDANCE_MODEL,
        "evidence_scope": "single_actor_combination_choreography_only",
        "source": {
            **source,
            "run_dir": str(source_run),
            "receipt_path": str(source_receipt_path),
            "receipt_sha256": sha256_file(source_receipt_path),
        },
        "blueprint": {
            **blueprint,
            "contract_path": str(contract_path),
            "manifest_path": str(manifest_path),
        },
        "candidate_selection": {
            "selected_beat_id": contract.beat_id,
            "source_receipt_beat_id": receipt_beat_id,
            "combination": blueprint["measurements"]["combination"],
            "exact_production_request_available": exact_request_projection,
        },
        "regression": regression,
        "call_chain_verdict": "pending" if exact_request_projection else "not_submittable",
        "business_motion_verdict": {
            "status": "pending_human_verdict" if exact_request_projection else "not_admitted"
        },
        "provider_request_count": 0,
        "tos_put_count": 0,
    }
    if exact_request_projection:
        projection = _project_request(source_run, source_receipt, blueprint)
        prompt_path = output_dir / "seedance_prompt.txt"
        prompt_path.write_text(projection["prompt"] + "\n", encoding="utf-8")
        receipt["request_projection"] = {
            "generation": projection["generation"],
            "equivalence": projection["equivalence"],
            "capabilities": projection["capabilities"],
            "prompt_path": str(prompt_path),
            "prompt_sha256": projection["generation"]["prompt_sha256"],
        }
        receipt["budgets"] = {
            "tos_put_ceiling": len(projection["media_runtime"]),
            "video_submission_ceiling": 1,
            "automatic_retry_ceiling": 0,
            "alternate_provider_submission_ceiling": 0,
        }
        _write_object(
            output_dir / "request_projection.runtime.json",
            {
                "prompt": projection["prompt"],
                "media": projection["media_runtime"],
                "generation": projection["generation"],
            },
        )
    else:
        receipt["request_projection"] = {
            "status": "missing_exact_pxx_receipt",
            "provider_request_count": 0,
        }
        receipt["budgets"] = {
            "tos_put_ceiling": 0,
            "video_submission_ceiling": 0,
            "automatic_retry_ceiling": 0,
            "alternate_provider_submission_ceiling": 0,
        }
        _write_object(
            output_dir / "request_projection.runtime.json",
            {
                "schema": "honcut.seedance-motion-blueprint-request-unavailable.v1",
                "status": "missing_exact_pxx_receipt",
                "selected_beat_id": contract.beat_id,
                "provider_request_count": 0,
            },
        )
    if legacy_assessment is not None:
        receipt["legacy_blueprint_assessment"] = legacy_assessment
    receipt["preflight_fingerprint"] = _hash(
        {key: value for key, value in receipt.items() if key not in {"preflight_fingerprint"}}
    )
    _write_object(output_dir / "seedance_motion_blueprint_gate.json", receipt)
    return receipt


def submit_gate(output_dir: Path, *, fee_authorization: str) -> dict[str, Any]:
    """Submit the already-frozen projection exactly once through existing owners."""
    if fee_authorization != "authorized-seedance-motion-blueprint-once":
        raise GateEvidenceError(
            "current explicit Seedance motion-blueprint fee authorization is required"
        )
    output_dir = output_dir.resolve()
    receipt_path = output_dir / "seedance_motion_blueprint_gate.json"
    receipt = _read_object(receipt_path)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "pending_live_acceptance"
        or receipt.get("submitted")
        or receipt.get("provider_request_count")
    ):
        raise GateEvidenceError("gate was already submitted or is not current")
    runtime = _read_object(output_dir / "request_projection.runtime.json")
    if (receipt.get("regression") or {}).get("status") != "passed":
        raise GateEvidenceError("live submission requires a passing bound regression receipt")
    generation = runtime["generation"]
    _assert_seedance_only(str(generation["model"]))
    content: list[dict[str, Any]] = [{"type": "text", "text": runtime["prompt"]}]
    for item in runtime["media"]:
        url = upload_media_file_required(
            item["absolute_path"],
            prefix=f"honcut/motion-blueprint/{generation['task_fingerprint']}",
            label=item["responsibility"],
        )
        content.append(
            {"type": item["media_type"], item["media_type"]: {"url": url}, "role": item["role"]}
        )
    payload = {**generation, "media": generation["media"]}
    store = GenerationTaskStore(output_dir / "runtime.db")
    api_key = get_api_key("ARK_AGENT")
    receipt.update({"status": "submission_uncertain", "submitted": True})
    _write_object(receipt_path, receipt)
    with provider_attempt_scope(max_retries=0), SingleSeedancePost() as transport:
        execution = execute_seedance_video_task(
            store,
            run_id=f"{output_dir.name}:motion-blueprint-v5",
            resource_id=f"MOTION_{receipt['blueprint']['schema']}",
            payload=payload,
            provider_endpoint=seedance_client.SUBMIT_ENDPOINT,
            output_path=output_dir / "seedance_output.mp4",
            submit=lambda: seedance_client.submit_content(
                content,
                api_key=api_key,
                model=generation["model"],
                duration=generation["duration"],
                ratio=generation["ratio"],
                resolution=generation["resolution"],
                return_last_frame=True,
                timeout=30,
            ),
            poll=lambda job_id: seedance_client.poll(
                job_id, api_key=api_key, max_attempts=80, interval=15
            ),
            download=seedance_client.download,
            validate_output=is_valid_video,
        )
    if transport.count != 1:
        raise GateEvidenceError("live gate did not perform exactly one Seedance POST")
    receipt.update(
        {
            "status": "pending_human_verdict",
            "provider_request_count": 1,
            "call_chain_verdict": "passed",
            "generation_task_id": execution.task_id,
            "provider_job_id": execution.provider_job_id,
            "output_path": execution.output_path,
            "output_sha256": sha256_file(Path(execution.output_path)),
        }
    )
    _write_object(receipt_path, receipt)
    return receipt


def record_human_verdict(
    output_dir: Path,
    *,
    verdict: str,
    notes: str,
) -> dict[str, Any]:
    """Finalize the business gate without calling any Provider."""
    if verdict not in {"pass", "fail"}:
        raise GateEvidenceError("human verdict must be pass or fail")
    output_dir = output_dir.resolve()
    receipt_path = output_dir / "seedance_motion_blueprint_gate.json"
    receipt = _read_object(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise GateEvidenceError("unsupported motion-blueprint gate receipt")
    if receipt.get("submitted") is not True or receipt.get("call_chain_verdict") != "passed":
        raise GateEvidenceError("human verdict requires a completed live call chain")
    output_path = Path(str(receipt.get("output_path") or "")).resolve()
    if not output_path.is_file() or sha256_file(output_path) != receipt.get("output_sha256"):
        raise GateEvidenceError("live output evidence is missing or changed")
    measurements = measure_output_motion(output_path)
    motion_pass = measurements["deterministic_motion_pass"] is True
    passed = verdict == "pass" and motion_pass
    receipt["status"] = "capability_gate_passed" if passed else "capability_route_paused"
    receipt["business_motion_verdict"] = {
        "status": verdict,
        "deterministic_motion_pass": motion_pass,
        "measurements": measurements,
        "notes_sha256": hashlib.sha256(notes.encode("utf-8")).hexdigest(),
        "production_activation_authorized": False,
        "evidence_scope": "single_actor_combination_choreography_only",
    }
    receipt["enforcement"] = {
        "automatic_retry_allowed": False,
        "automatic_redraw_allowed": False,
        "automatic_reshoot_allowed": False,
        "budget_expansion_allowed": False,
        "alternate_provider_allowed": False,
        "production_activation_allowed": False,
    }
    _write_object(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path, nargs="?")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--regression-receipt", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--fee-authorization", default="")
    parser.add_argument("--business-verdict", choices=("pass", "fail"))
    parser.add_argument("--verdict-notes", default="")
    args = parser.parse_args()
    if args.business_verdict:
        result = record_human_verdict(
            args.output_dir,
            verdict=args.business_verdict,
            notes=args.verdict_notes,
        )
    elif args.submit:
        result = submit_gate(args.output_dir, fee_authorization=args.fee_authorization)
    else:
        if args.source_run is None:
            parser.error("source_run is required for no-submit preflight")
        result = prepare_gate(
            args.source_run,
            args.output_dir,
            source_receipt_path=args.source_receipt,
            regression_receipt=args.regression_receipt,
            legacy_manifest_path=args.legacy_manifest,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
