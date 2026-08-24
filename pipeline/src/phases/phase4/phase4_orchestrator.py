"""Phase 4 deterministic shot and continuity orchestration."""

from __future__ import annotations

import json
import os
import traceback
from copy import deepcopy
from pathlib import Path

from phases.phase4.shot_setup import materialize_shot_directories, normalize_shots
from runtime.phase_timing import _banner, _elapsed, _now
from tools.provider_scoring import rank_providers
from tools.video_composer import lock_runtime
from utils.storyboard_geometry import _storyboard_canvas, _storyboard_image_size
from utils.timing_estimator import estimate_phase_duration


def _director_pacing_by_sequence(plan: object) -> dict[str, dict]:
    """Validate the v1 Director artifact and index deterministic pacing."""
    if not isinstance(plan, dict):
        raise ValueError("director_plan.json must contain an object")
    if plan.get("schema") != "honcut.director-plan.v1":
        raise ValueError(
            "director_plan.json schema must be honcut.director-plan.v1"
        )
    sequences = plan.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("director_plan.json sequences must be an array")
    pacing_by_sequence: dict[str, dict] = {}
    for index, sequence in enumerate(sequences, 1):
        if not isinstance(sequence, dict):
            raise ValueError(f"director sequence {index} must be an object")
        sequence_id = str(sequence.get("sequence_id") or "").strip()
        if not sequence_id or sequence_id in pacing_by_sequence:
            raise ValueError(
                "director plan has empty or duplicate sequence_id: "
                f"{sequence_id!r}"
            )
        pacing = sequence.get("speech_pacing")
        if not isinstance(pacing, dict):
            raise ValueError(
                f"director sequence {sequence_id} has no speech_pacing"
            )
        duration = pacing.get("duration_s")
        emotion = pacing.get("emotion")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
            or not isinstance(emotion, str)
        ):
            raise ValueError(
                f"director sequence {sequence_id} has invalid speech_pacing"
            )
        pacing_by_sequence[sequence_id] = deepcopy(pacing)
    return pacing_by_sequence


def _bind_director_pacing(
    meta: dict,
    pacing_by_sequence: dict[str, dict],
) -> None:
    """Bind one shot by semantic sequence identity, never list position."""
    raw_sequence_ids = meta.get("source_sequence_ids")
    if not isinstance(raw_sequence_ids, list):
        raise ValueError("SHOT_META source_sequence_ids must be an array")
    sequence_ids = list(
        dict.fromkeys(
            str(value).strip() for value in raw_sequence_ids if str(value).strip()
        )
    )
    if len(sequence_ids) != 1:
        raise ValueError(
            "SHOT_META must bind exactly one Director sequence; "
            f"source_sequence_ids={sequence_ids}"
        )
    sequence_id = sequence_ids[0]
    if sequence_id not in pacing_by_sequence:
        raise ValueError(
            f"SHOT_META references unknown Director sequence: {sequence_id}"
        )
    meta["speech_pacing"] = deepcopy(pacing_by_sequence[sequence_id])


def run_phase4(output_dir: Path, dry_run: bool) -> dict:
    """Phase 4: deterministic orchestration and code-constraint review."""
    _banner(4, 9, "编排器 (Orchestrator)", dry_run)
    start = _now()
    _p4_est = estimate_phase_duration("phase4")
    print(f"  ⏱ Phase 4 开始 (预估 ~{int(_p4_est)}s)")
    print("  → 代码约束复查：无人工审批、无审核模型调用")
    outputs = []
    output_dir = Path(output_dir)

    storyboard_path = output_dir / "STORYBOARD.json"
    if not storyboard_path.exists():
        return {"status": "error", "error": "STORYBOARD.json not found", "duration_s": _elapsed(start)}

    try:
        from phases.phase4.continuity_plan import write_continuity_plan, write_storyboard_groups
        from phases.phase4.scene_consistency import write_scene_consistency

        storyboard_for_consistency = json.loads(storyboard_path.read_text(encoding="utf-8"))
        from phases.phase2.shot_storyboards import validate_shot_storyboard_artifacts

        storyboard_artifact_errors = (
            validate_shot_storyboard_artifacts(output_dir, storyboard_for_consistency)
            if not dry_run
            else []
        )
        if storyboard_artifact_errors:
            return {
                "status": "error",
                "error": "Phase 4 requires complete Phase 2 Pxx storyboards",
                "artifact_errors": storyboard_artifact_errors,
                "duration_s": _elapsed(start),
            }
        characters_path = output_dir / "CHARACTERS.json"
        characters_for_consistency = (
            json.loads(characters_path.read_text(encoding="utf-8"))
            if characters_path.exists() else {"characters": []}
        )
        visual_style_path = next(
            (
                candidate for candidate in (
                    output_dir / "visual-style.md",
                    output_dir / "visual_style_spec.md",
                ) if candidate.exists()
            ),
            None,
        )
        scene_consistency = write_scene_consistency(
            output_dir / "SCENE_CONSISTENCY.json",
            storyboard_for_consistency,
            characters_for_consistency,
            visual_style_path,
        )
        outputs.append("SCENE_CONSISTENCY.json")
        print("  ✓ 场景一致性契约: SCENE_CONSISTENCY.json")
        cinematic_errors: list[str] = []
        if not dry_run:
            from phases.phase4.cinematic_first_frames import (
                generate_cinematic_first_frames,
                validate_cinematic_first_frame_artifacts,
            )

            video_width, video_height, cinematic_aspect_ratio = _storyboard_canvas(
                storyboard_for_consistency
            )
            cinematic_frames = generate_cinematic_first_frames(
                output_dir,
                storyboard_for_consistency,
                characters_for_consistency.get("characters", []),
                scene_consistency,
                size=_storyboard_image_size(
                    video_width=video_width,
                    video_height=video_height,
                ),
                visual_style_path=visual_style_path,
                aspect_ratio=cinematic_aspect_ratio,
            )
            cinematic_errors = validate_cinematic_first_frame_artifacts(
                output_dir,
                storyboard_for_consistency,
            )
            if cinematic_errors:
                return {
                    "status": "error",
                    "error": "Phase 4 cinematic first-frame validation failed",
                    "artifact_errors": cinematic_errors,
                    "duration_s": _elapsed(start),
                }
            storyboard_path.write_text(
                json.dumps(
                    storyboard_for_consistency,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            outputs.extend(
                [
                    "CINEMATIC_FIRST_FRAMES.json",
                    "video_first_frames/",
                    "storyboard_images/",
                ]
            )
            print(
                "  ✓ 成片质感首帧: "
                f"{cinematic_frames.get('frame_count', 0)} 个（零 PREVIS 像素参考）"
            )
        continuity_plan = write_continuity_plan(
            output_dir / "CONTINUITY_PLAN.json",
            storyboard_for_consistency,
            scene_consistency,
            # Every native extension reserves replay context. Phase 8 removes
            # the planned prefix (and any detected extra rollback) by frame.
            continuation_overlap_s=float(
                os.environ.get("HONCUT_CONTINUITY_OVERLAP_SECONDS", "2.0")
            ),
            continuity_group_max_shots=int(
                os.environ.get("HONCUT_CONTINUITY_GROUP_MAX_SHOTS", "3")
            ),
        )
        outputs.append("CONTINUITY_PLAN.json")
        print("  ✓ 连续性计划: CONTINUITY_PLAN.json")
        storyboard_groups = write_storyboard_groups(
            output_dir,
            storyboard_for_consistency,
            continuity_plan,
        )
        outputs.append("STORYBOARD_GROUPS.json")
        outputs.extend(
            str(group["storyboard_board"])
            for group in storyboard_groups.get("groups", [])
            if group.get("storyboard_board")
        )
        print("  ✓ 组级故事板: STORYBOARD_GROUPS.json + storyboard_groups/")
        from runtime.continuity_memory import initialize_continuity_memory

        initialize_continuity_memory(output_dir, continuity_plan)
        outputs.append("CONTINUITY_MEMORY.json")
        print("  ✓ 不可变连续性锚点: CONTINUITY_MEMORY.json")

        shots_dir = output_dir / "shots"
        normalized_shots = normalize_shots(
            storyboard_for_consistency,
            storyboard_dir=storyboard_path.parent,
        )
        materialize_shot_directories(shots_dir, normalized_shots)
        outputs.extend(f"shots/{shot['shot_id']}/" for shot in normalized_shots)
        print(f"  ✓ 原生镜头元数据: {len(normalized_shots)} 个 SHOT_META.json")

        storyboard = storyboard_for_consistency
        expected_shot_ids = {shot["shot_id"] for shot in normalized_shots}
        actual_shot_ids = {
            directory.name
            for directory in shots_dir.iterdir()
            if directory.is_dir()
            and directory.name.startswith("S")
            and (directory / "SHOT_META.json").is_file()
        } if shots_dir.is_dir() else set()
        if actual_shot_ids != expected_shot_ids:
            return {
                "status": "error",
                "error": "Phase 4 shot directory invariant failed",
                "expected_shot_ids": sorted(expected_shot_ids),
                "actual_shot_ids": sorted(actual_shot_ids),
                "missing_shot_ids": sorted(expected_shot_ids - actual_shot_ids),
                "unexpected_shot_ids": sorted(actual_shot_ids - expected_shot_ids),
                "duration_s": _elapsed(start),
            }
        provider_candidates = [
            {"name": "local_video", "provider": "local", "capabilities": ["i2v", "flf2v"], "quality": .8, "control": .9, "reliability": .8, "cost": 0, "latency_score": .7},
            {"name": "seedance", "provider": "volcengine", "capabilities": ["i2v", "t2v", "reference_image"], "quality": .9, "control": .7, "reliability": .75, "cost": .4, "latency_score": .6},
        ]
        required_capabilities = sorted({
            shot.get("gen_strategy", "i2v") for shot in storyboard.get("shots", [])
        })
        rankings = rank_providers(provider_candidates, {"capabilities": required_capabilities})
        selected_provider = rankings[0].tool_name if rankings else "local_video"
        composition = {
            "cuts": [{"id": shot.get("id"), "type": shot.get("type", "video")} for shot in storyboard.get("shots", [])],
            "provider": selected_provider,
            "provider_rankings": [score.to_dict() for score in rankings],
        }
        locked_composition = lock_runtime(composition, available={"ffmpeg", "remotion"})

        director_pacing = None
        director_plan_path = output_dir / "director_plan.json"
        if director_plan_path.exists():
            director_pacing = _director_pacing_by_sequence(
                json.loads(director_plan_path.read_text(encoding="utf-8"))
            )
        for shot_dir in sorted(shots_dir.glob("S*")) if shots_dir.exists() else []:
            meta_path = shot_dir / "SHOT_META.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["provider"] = selected_provider
            meta["provider_rankings"] = locked_composition["provider_rankings"]
            meta["render_runtime"] = locked_composition["render_runtime"]
            if director_pacing is not None:
                _bind_director_pacing(meta, director_pacing)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        shot_output_count = sum(item.startswith("shots/") for item in outputs)
        print(f"  ✓ Phase 4 完成: {shot_output_count} 镜头目录")
        status = "done" if shot_output_count or dry_run else "error"
        constraint_checks = [
            {
                "id": "storyboard_artifacts_complete",
                "status": "passed",
                "detail": "all authored Pxx storyboard assets are present",
            },
            {
                "id": "scene_and_continuity_contracts_written",
                "status": "passed",
                "detail": "scene, continuity, group, and memory contracts were materialized",
            },
            {
                "id": "cinematic_first_frames_previs_isolated",
                "status": "passed" if dry_run or not cinematic_errors else "failed",
                "detail": (
                    "every video first frame has a style-injection receipt and zero "
                    "PREVIS pixel references"
                ),
            },
            {
                "id": "native_shot_metadata_only",
                "status": "passed",
                "detail": "Phase 4 materialized SHOT_META without a subprocess or Provider call",
            },
            {
                "id": "shot_meta_ids_exact",
                "status": "passed" if actual_shot_ids == expected_shot_ids else "failed",
                "detail": "SHOT_META directories exactly match STORYBOARD shot IDs",
            },
        ]
        if status == "done":
            print("  ✓ Phase 4 代码约束复查通过（人工复查：禁用）")
        else:
            print("  ✗ Phase 4 代码约束复查失败（人工复查：禁用）")
        return {
            "status": status,
            "duration_s": _elapsed(start),
            "outputs": outputs or ["shots/"],
            "provider": selected_provider,
            "render_runtime": locked_composition["render_runtime"],
            "constraint_review": {
                "status": "passed" if status == "done" else "failed",
                "mode": "deterministic_code",
                "human_review_required": False,
                "model_review_used": False,
                "checks": constraint_checks,
            },
            **(
                {"error": "native shot setup produced no shot directories"}
                if status == "error"
                else {}
            ),
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
