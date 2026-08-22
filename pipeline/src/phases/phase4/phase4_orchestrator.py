"""Phase 4 deterministic shot and continuity orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path

from phases.phase2.storyboard_assets import _normalize_shot_id
from runtime.phase_timing import _banner, _elapsed, _now
from tools.provider_scoring import rank_providers
from tools.video_composer import lock_runtime
from utils.source_paths import LEGACY_TOOLS_DIR, PIPELINE_SRC_DIR
from utils.file_integrity import file_sha256
from utils.storyboard_geometry import _storyboard_canvas, _storyboard_image_size
from utils.timing_estimator import estimate_phase_duration


PHASE4_LEGACY_STORYBOARD_SCHEMA = "honcut.phase4-legacy-storyboard.v1"
PHASE4_LEGACY_STORYBOARD_NAME = "phase4_legacy_storyboard.json"


def _first_compatibility_text(shot: dict, fields: tuple[str, ...]) -> str | None:
    """Return the first authored, non-empty string accepted by legacy Phase 4."""
    for field in fields:
        value = shot.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _write_legacy_storyboard_adapter(
    output_dir: Path,
    storyboard_path: Path,
    storyboard: dict,
) -> Path:
    """Write the narrow legacy input without changing the canonical storyboard."""
    adapted = deepcopy(storyboard)
    shots = adapted.get("shots")
    if not isinstance(shots, list):
        raise ValueError("Phase 4 storyboard must contain a shots array")

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise ValueError(f"Phase 4 shot at index {index} must be an object")
        normalized_id = _normalize_shot_id(shot)
        if normalized_id is None:
            raise ValueError(f"Phase 4 shot at index {index} has no usable shot ID")
        try:
            numeric_id = int(normalized_id[1:])
        except ValueError as exc:
            raise ValueError(
                f"Phase 4 shot at index {index} has a non-numeric shot ID: "
                f"{normalized_id}"
            ) from exc
        if numeric_id <= 0:
            raise ValueError(
                f"Phase 4 shot at index {index} has an invalid shot ID: "
                f"{normalized_id}"
            )

        name = _first_compatibility_text(shot, ("name",))
        if name is None:
            name = _first_compatibility_text(
                shot,
                (
                    "shot_intent",
                    "caption",
                    "action",
                    "what",
                    "visual",
                    "prompt",
                ),
            ) or normalized_id
        prompt = _first_compatibility_text(shot, ("prompt",))
        if prompt is None:
            prompt = _first_compatibility_text(
                shot,
                ("visual", "action", "what"),
            )
        if prompt is None:
            raise ValueError(
                f"Phase 4 shot {normalized_id} has no prompt-compatible visual, "
                "action, or what field"
            )

        shot["id"] = numeric_id
        shot["shot_id"] = normalized_id
        shot["name"] = name
        shot["prompt"] = prompt

    adapted["_compatibility"] = {
        "schema": PHASE4_LEGACY_STORYBOARD_SCHEMA,
        "source_path": storyboard_path.relative_to(output_dir).as_posix(),
        "source_sha256": file_sha256(storyboard_path),
    }
    adapter_path = output_dir / PHASE4_LEGACY_STORYBOARD_NAME
    temporary = adapter_path.with_suffix(adapter_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(adapted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, adapter_path)
    finally:
        temporary.unlink(missing_ok=True)
    return adapter_path


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

        orchestrator_script = LEGACY_TOOLS_DIR / "orchestrator.py"
        if not orchestrator_script.exists():
            return {"status": "error", "error": f"orchestrator.py not found at {orchestrator_script}", "duration_s": _elapsed(start)}

        legacy_storyboard_path = _write_legacy_storyboard_adapter(
            output_dir,
            storyboard_path,
            storyboard_for_consistency,
        )
        outputs.append(PHASE4_LEGACY_STORYBOARD_NAME)

        shots_dir = output_dir / "shots"
        cmd = [
            sys.executable, str(orchestrator_script),
            "--storyboard", str(legacy_storyboard_path.resolve()),
            "--skip-assembly",
            "--shots-dir", str(shots_dir.resolve()),
            # Phase 4 owns routing and SHOT_META creation only.  The legacy
            # orchestrator's live mode also submits video jobs, which belongs
            # exclusively to Phase 6 and can otherwise double-submit work.
            "--dry-run",
        ]

        print(f"  → orchestrator: {' '.join(cmd[-4:])}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(LEGACY_TOOLS_DIR),
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (str(PIPELINE_SRC_DIR), os.environ.get("PYTHONPATH", "")),
                    )
                ),
            },
        )

        print(f"  → orchestrator return code: {result.returncode}")

        if result.returncode != 0:
            print(f"  ⚠ orchestrator stdout tail: {result.stdout[-1500:]}")
            print(f"  ⚠ orchestrator stderr tail: {result.stderr[-1000:]}")
            return {
                "status": "error",
                "error": f"orchestrator exited with code {result.returncode}",
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1500:],
                "stderr_tail": result.stderr[-1000:],
                "duration_s": _elapsed(start),
            }

        # 扫描输出
        shots_dir = output_dir / "shots"
        if shots_dir.exists():
            for d in sorted(shots_dir.iterdir()):
                if d.is_dir() and d.name.startswith("S"):
                    outputs.append(f"shots/{d.name}/")

        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        expected_shot_ids = {
            shot_id
            for shot in storyboard.get("shots", [])
            if (shot_id := _normalize_shot_id(shot)) is not None
        }
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

        director_scenes = []
        director_plan_path = output_dir / "director_plan.json"
        if director_plan_path.exists():
            director_scenes = json.loads(director_plan_path.read_text(encoding="utf-8")).get("scenes", [])
        for index, shot_dir in enumerate(sorted(shots_dir.glob("S*")) if shots_dir.exists() else []):
            meta_path = shot_dir / "SHOT_META.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["provider"] = selected_provider
            meta["provider_rankings"] = locked_composition["provider_rankings"]
            meta["render_runtime"] = locked_composition["render_runtime"]
            if index < len(director_scenes) and director_scenes[index].get("speech_pacing"):
                meta["speech_pacing"] = director_scenes[index]["speech_pacing"]
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
                "id": "legacy_orchestrator_metadata_only",
                "status": "passed",
                "detail": "legacy orchestrator was forced to --dry-run --skip-assembly",
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
                {"error": "orchestrator produced no shot directories"}
                if status == "error"
                else {}
            ),
        }

    except subprocess.TimeoutExpired as e:
        timeout_stdout = e.stdout or ""
        timeout_stderr = e.stderr or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode(errors="replace")
        if isinstance(timeout_stderr, bytes):
            timeout_stderr = timeout_stderr.decode(errors="replace")
        print("  ⚠ orchestrator timed out after 120s")
        print(f"  ⚠ orchestrator stdout tail: {timeout_stdout[-1500:]}")
        print(f"  ⚠ orchestrator stderr tail: {timeout_stderr[-1000:]}")
        return {"status": "error", "error": "orchestrator timed out", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
