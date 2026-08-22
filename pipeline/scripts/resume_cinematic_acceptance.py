#!/usr/bin/env python3
"""Resume one L4-approved cinematic-frame acceptance through Seedance and Phase 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
DEFAULT_PROVIDER_COOLDOWN_SECONDS = 15 * 60
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_project_api_key(repo_root: Path) -> None:
    """Fail before imports can hide a stale exported key behind dotenv precedence."""
    project_key = str(
        dotenv_values(repo_root / ".env").get("ARK_AGENT_API_KEY") or ""
    )
    inherited_key = os.environ.get("ARK_AGENT_API_KEY", "")
    if not project_key:
        raise RuntimeError("project .env has no ARK_AGENT_API_KEY")
    if inherited_key and inherited_key != project_key:
        raise RuntimeError(
            "stale exported ARK_AGENT_API_KEY overrides project .env; "
            "run with `env -u ARK_AGENT_API_KEY ...` or restart the worker"
        )


def _find_shot_and_beat(
    storyboard: dict[str, Any],
    shot_id: str,
    beat_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_shot = shot_id.upper()
    for index, shot in enumerate(storyboard.get("shots") or [], 1):
        if not isinstance(shot, dict):
            continue
        raw_id = str(shot.get("id") or shot.get("shot_id") or index).upper()
        candidate = raw_id if raw_id.startswith("S") else f"S{int(raw_id):02d}"
        if candidate != normalized_shot:
            continue
        for beat in shot.get("storyboard_beats") or []:
            if isinstance(beat, dict) and beat.get("beat_id") == beat_id:
                return shot, beat
    raise RuntimeError(f"{beat_id} is not declared under {shot_id}")


def _assert_acceptance_prerequisites(
    output_dir: Path,
    storyboard: dict[str, Any],
    beat: dict[str, Any],
    beat_id: str,
) -> tuple[Path, dict[str, Any]]:
    from phases.phase4.cinematic_first_frames import (
        CINEMATIC_FIRST_FRAME_SCHEMA,
        validate_cinematic_first_frame_artifacts,
    )

    errors = validate_cinematic_first_frame_artifacts(output_dir, storyboard)
    if errors:
        raise RuntimeError("invalid cinematic first-frame chain: " + "; ".join(errors))
    if beat.get("video_first_frame_kind") != CINEMATIC_FIRST_FRAME_SCHEMA:
        raise RuntimeError(f"{beat_id} is not a cinematic Phase 4 frame")
    frame = output_dir / str(beat.get("video_first_frame") or "")
    frame_receipt = _read_json(
        output_dir / str(beat.get("video_first_frame_receipt") or "")
    )
    if frame_receipt.get("previs_reference_images") != []:
        raise RuntimeError("video-bound frame has non-empty PREVIS lineage")
    if frame_receipt.get("upstream_director_panel_usage") != (
        "image_generation_composition_only_never_video_reference"
    ):
        raise RuntimeError("director panel is not declared generation-only")

    l4 = _read_json(output_dir / "L4_FIRST_FRAME_RESULT.json")
    if l4.get("gate_passed") is not True or l4.get("issues"):
        raise RuntimeError("Phase 5 L4 has not approved the cinematic frame")
    qa_inputs = _read_json(output_dir / "first_frame_qa_inputs.json")
    records = qa_inputs.get("inputs") or qa_inputs.get("records") or []
    matching = [record for record in records if record.get("frame_id") == beat_id]
    if not matching or matching[0].get("sha256") != _sha256(frame):
        raise RuntimeError("L4 approval does not belong to the current frame bytes")
    return frame, frame_receipt


def _queue_totals(api_key: str) -> dict[str, int]:
    from clients.seedance_client import list_tasks

    return {
        status: int(
            list_tasks(
                api_key=api_key,
                page_size=50,
                filter_status=status,
            ).get("total")
            or 0
        )
        for status in ("queued", "running")
    }


def _provider_cooldown_state(
    database_path: Path,
    resource_id: str,
    *,
    cooldown_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    from runtime.provider_policy import provider_cooldown_state

    return provider_cooldown_state(
        database_path,
        resource_id,
        provider_id="seedance",
        cooldown_seconds=cooldown_seconds,
        now=now,
    )


def _update_acceptance_receipt(
    output_dir: Path,
    *,
    phase6: dict[str, Any] | None = None,
    phase8: dict[str, Any] | None = None,
    status: str | None = None,
) -> None:
    path = output_dir / "PIT54_ACCEPTANCE.json"
    receipt = _read_json(path) if path.is_file() else {
        "schema": "honcut.pit54-s02-acceptance.v1"
    }
    if status:
        receipt["status"] = status
    if phase6:
        receipt.setdefault("phase6", {}).update(phase6)
    if phase8:
        receipt.setdefault("phase8", {}).update(phase8)
    _write_json(path, receipt)


def run(
    output_dir: Path,
    *,
    shot_id: str,
    beat_id: str,
    duration: int,
    seed: int,
    submit: bool,
    force_submit: bool = False,
    cooldown_seconds: int = DEFAULT_PROVIDER_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    _assert_project_api_key(REPO_ROOT)

    # Import after the explicit-key provenance check. utils.config then loads
    # the project .env only when no matching process value is already present.
    from clients import seedance_client
    from phases.phase8.frame_analysis import analyze_shot_frames
    from phases.pipeline_core import _prepare_phase6_prompt
    from runtime.generation_tasks import GenerationTaskStore
    from runtime.seedance_execution import execute_seedance_video_task
    from tools.asset_packager import _detect_shot_characters, build_content_for_shot
    from utils.config import SEEDANCE_MODEL
    from utils.video_validation import is_valid_video

    output_dir = output_dir.resolve()
    storyboard = _read_json(output_dir / "STORYBOARD.json")
    characters = _read_json(output_dir / "CHARACTERS.json")
    scene = _read_json(output_dir / "SCENE_CONSISTENCY.json")
    shot, beat = _find_shot_and_beat(storyboard, shot_id, beat_id)
    frame, _frame_receipt = _assert_acceptance_prerequisites(
        output_dir,
        storyboard,
        beat,
        beat_id,
    )
    api_key = os.environ.get("ARK_AGENT_API_KEY", "")
    if not api_key:
        raise RuntimeError("project ARK_AGENT_API_KEY did not load")

    shot_dir = output_dir / "shots" / beat_id
    shot_dir.mkdir(parents=True, exist_ok=True)
    video_path = shot_dir / "output.mp4"
    meta = {
        **shot,
        **beat,
        "id": beat_id,
        "shot_id": beat_id,
        "gen_strategy": (
            beat.get("gen_strategy")
            or shot.get("gen_strategy")
            or "phantom"
        ),
        "duration": duration,
        "width": 2560,
        "height": 1440,
        "_storyboard_frame_path": beat["video_first_frame"],
        "_storyboard_frame_kind": beat["video_first_frame_kind"],
        "_storyboard_beat_id": beat_id,
    }
    meta["_char_ids"] = sorted(_detect_shot_characters(output_dir, meta))
    prompt, route_applied = _prepare_phase6_prompt(
        shot_id,
        meta,
        characters,
        scene,
        video_model=SEEDANCE_MODEL,
        route_model=SEEDANCE_MODEL,
    )
    meta["prompt"] = prompt
    forbidden = ("3C91", "A857", "阴雨天漫射冷光", "低饱和度")
    leaked = [marker for marker in forbidden if marker in prompt]
    if leaked:
        raise RuntimeError("Phase 6 prompt leaked forbidden markers: " + ", ".join(leaked))
    (shot_dir / "PHASE6_PROMPT.txt").write_text(prompt, encoding="utf-8")
    meta["first_frame_sha256"] = _sha256(frame)
    _write_json(shot_dir / "SHOT_META.json", meta)

    if video_path.is_file() and is_valid_video(video_path):
        execution_result = {"status": "resumed_local_video", "output_path": str(video_path)}
    elif not submit:
        return {
            "status": "ready_to_submit",
            "shot_id": beat_id,
            "model": SEEDANCE_MODEL,
            "first_frame_sha256": _sha256(frame),
            "route_applied": route_applied,
        }
    else:
        if not force_submit:
            cooldown = _provider_cooldown_state(
                output_dir / "runtime.db",
                beat_id,
                cooldown_seconds=cooldown_seconds,
            )
            if cooldown:
                return {
                    **cooldown,
                    "shot_id": beat_id,
                    "model": SEEDANCE_MODEL,
                    "first_frame_sha256": _sha256(frame),
                }
        queue = _queue_totals(api_key)
        if queue["queued"] or queue["running"]:
            raise RuntimeError(f"Seedance queue is not empty: {queue}")
        content = build_content_for_shot(output_dir, shot_id, meta)
        transport_prompt = next(
            item["text"] for item in content if item.get("type") == "text"
        )
        roles = [
            item.get("role")
            for item in content
            if item.get("type") == "image_url"
        ]
        if roles != ["first_frame"]:
            raise RuntimeError(f"expected one exact first_frame transport, got {roles}")
        leaked = [marker for marker in forbidden if marker in transport_prompt]
        if leaked:
            raise RuntimeError(
                "transport prompt leaked forbidden markers: " + ", ".join(leaked)
            )
        payload = {
            "shot_id": beat_id,
            "output_path": f"shots/{beat_id}/output.mp4",
            "model": SEEDANCE_MODEL,
            "duration": duration,
            "ratio": "16:9",
            "seed": seed,
            "prompt_sha256": hashlib.sha256(transport_prompt.encode()).hexdigest(),
            "first_frame_sha256": _sha256(frame),
            "transport_roles": roles,
        }
        _write_json(
            shot_dir / "PHASE6_REQUEST.json",
            {**payload, "route_applied": route_applied, "api_key_recorded": False},
        )
        try:
            execution = execute_seedance_video_task(
                GenerationTaskStore(output_dir / "runtime.db"),
                run_id=str(output_dir),
                resource_id=beat_id,
                payload=payload,
                provider_endpoint=seedance_client.BASE_URL,
                output_path=video_path,
                submit=partial(
                    seedance_client.submit_content,
                    content,
                    api_key=api_key,
                    model=SEEDANCE_MODEL,
                    duration=duration,
                    ratio="16:9",
                    seed=seed,
                ),
                poll=partial(
                    seedance_client.poll,
                    api_key=api_key,
                    max_attempts=40,
                    interval=15,
                ),
                download=seedance_client.download,
                validate_output=is_valid_video,
            )
        except Exception as exc:
            previous = _read_json(output_dir / "PIT54_ACCEPTANCE.json").get(
                "phase6", {}
            )
            _update_acceptance_receipt(
                output_dir,
                status="video_submission_blocked",
                phase6={
                    "status": "blocked_before_provider_job_creation",
                    "submission_attempts": int(previous.get("submission_attempts") or 0) + 1,
                    "last_error": str(exc),
                    "queued_tasks_observed": queue["queued"],
                    "running_tasks_observed": queue["running"],
                    "last_provider_attempt_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
        execution_result = {
            "status": "done",
            "generation_task_id": execution.task_id,
            "provider_job_id": execution.provider_job_id,
            "output_path": execution.output_path,
            "resumed": execution.resumed,
        }
        _update_acceptance_receipt(
            output_dir,
            status="video_generated_phase8_pending",
            phase6={**execution_result, "transport_roles": roles},
        )

    frame_report = analyze_shot_frames(
        output_dir / "shots",
        output_dir / "frame_analysis.json",
        semantic_reviewer=True,
        max_frames=10,
        interval_s=1.0,
    )
    action = (
        frame_report.get("shots", {}).get(beat_id, {}).get("action")
    )
    passed = action == "keep"
    _update_acceptance_receipt(
        output_dir,
        status="passed" if passed else "phase8_failed",
        phase8={
            "status": "passed" if passed else "failed",
            "action": action,
            "summary": frame_report.get("summary"),
            "report": "frame_analysis.json",
        },
    )
    if not passed:
        raise RuntimeError(f"Phase 8 rejected {beat_id}: action={action}")
    return {
        "status": "passed",
        "phase6": execution_result,
        "phase8": frame_report.get("summary"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "pipeline/output/pit54_s02_acceptance_20260821",
    )
    parser.add_argument("--shot-id", default="S02")
    parser.add_argument("--beat-id", default="S02_P01")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--seed", type=int, default=54202)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="authorize the paid Seedance submission; without it only preflight runs",
    )
    parser.add_argument(
        "--force-submit",
        action="store_true",
        help=(
            "bypass the persisted provider cooldown only after an observed external "
            "state change; still requires --submit"
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=DEFAULT_PROVIDER_COOLDOWN_SECONDS,
        help="minimum delay after a pre-job Seedance quota failure (default: 900)",
    )
    args = parser.parse_args()
    result = run(
        args.output_dir,
        shot_id=args.shot_id,
        beat_id=args.beat_id,
        duration=args.duration,
        seed=args.seed,
        submit=args.submit,
        force_submit=args.force_submit,
        cooldown_seconds=args.cooldown_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
