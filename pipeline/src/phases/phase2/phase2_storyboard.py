"""Phase 2 storyboard image generation."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from phases.phase2.storyboard_assets import (
    _generate_shot_images,
    _storyboard_canvas,
    _storyboard_image_size,
    _validate_storyboard_image_composition,
    fill_storyboard_template,
)
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from runtime.retry_execution import _retry_with_policy
from utils.source_paths import PIPELINE_SRC_DIR as SCRIPT_DIR
from utils.timing_estimator import estimate_phase_duration


def run_phase2(storyboard_data: dict, characters_data: dict, output_dir: Path, dry_run: bool) -> dict:
    """Phase 2: 使用 OM image_selector 生成故事板图片，不可用时降级到 Seedream API"""
    import shutil

    _banner("2", 9, "故事板图片生成 (ImageSelector / Seedream)", dry_run)
    start = _now()
    _p25_est = estimate_phase_duration("phase2")
    print(f"  ⏱ Phase 2 开始 (预估 ~{int(_p25_est)}s)")
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过故事板图片生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    video_width, video_height, aspect_ratio = _storyboard_canvas(storyboard_data)

    # Phase 1 now owns the model-generated director overview. Reuse that exact
    # image here so Phase 2 can focus on per-shot reference frames instead of
    # paying for a second, semantically competing overview board.
    director_ref = storyboard_data.get("director_storyboard") or {}
    director_image = output_dir / str(
        director_ref.get("image") or "director_storyboard.png"
    )
    if director_ref.get("status") == "done" and director_image.is_file():
        storyboard_path = output_dir / "storyboard.png"
        shutil.copy2(director_image, storyboard_path)
        print("  ↻ 复用 Phase 1 模型生成的导演故事板总览")
        qg_report = run_quality_check("phase2", output_dir)
        if not qg_report.passed:
            return {
                "status": "error",
                "error": f"Phase 2 质检未通过: {qg_report.grade}",
                "quality_report": qg_report,
                "duration_s": _elapsed(start),
            }
        from phases.phase2.shot_storyboards import (
            _character_reference_paths,
            generate_shot_storyboards,
            validate_shot_storyboard_artifacts,
        )

        characters = characters_data.get("characters", [])
        deferred_shot_ids: list[str] = []
        for shot_index, shot in enumerate(storyboard_data.get("shots", []), 1):
            if not isinstance(shot, dict):
                continue
            who = shot.get("who") or shot.get("character_ids") or []
            if isinstance(who, str):
                who = [who]
            if not who:
                continue
            references_ready = all(
                len(_character_reference_paths(output_dir, characters, [identity])) >= 2
                for identity in who
            )
            if not references_ready:
                deferred_shot_ids.append(
                    str(shot.get("id") or shot.get("shot_id") or f"S{shot_index:02d}")
                )
        if deferred_shot_ids:
            print(
                "  ↷ 角色参考尚未由 Phase 3 建立；延后含角色的 Pxx 生成，避免重复付费"
            )
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["storyboard.png"],
                "provider": "deferred_to_phase3",
                "shot_storyboards_generated": 0,
                "storyboard_panels_generated": 0,
                "deferred_shot_ids": deferred_shot_ids,
            }

        print("  → Seedream: 按 Sxx 生成内部手绘故事板...")
        shot_storyboards = generate_shot_storyboards(
            output_dir,
            storyboard_data,
            characters,
            size=_storyboard_image_size(
                video_width=video_width,
                video_height=video_height,
            ),
            director_storyboard_path=director_image,
            aspect_ratio=aspect_ratio,
        )
        artifact_errors = validate_shot_storyboard_artifacts(
            output_dir,
            storyboard_data,
        )
        if artifact_errors:
            return {
                "status": "error",
                "error": "Phase 2 Pxx artifact validation failed",
                "artifact_errors": artifact_errors,
                "duration_s": _elapsed(start),
            }
        # Persist the provider-returned board and model-generated Pxx references back
        # into the canonical storyboard consumed by Phase 4.
        (output_dir / "STORYBOARD.json").write_text(
            json.dumps(storyboard_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": ["storyboard.png", "SHOT_STORYBOARDS.json"],
            "provider": "seedream_shot_storyboards",
            "shot_storyboards_generated": shot_storyboards["total_boards"],
            "storyboard_panels_generated": shot_storyboards["total_panels"],
            "storyboard_transition_panels_generated": shot_storyboards.get(
                "total_transition_panels", 0
            ),
        }

    print("[cooldown] 等待 120s 让 Agent Plan 限流窗口重置...", flush=True)
    time.sleep(120)

    # 1. 加载模板
    template_path = SCRIPT_DIR.parent / "prompts" / "storyboard_template.md"
    if not template_path.exists():
        return {"status": "error", "error": f"storyboard_template.md not found at {template_path}", "duration_s": _elapsed(start)}

    template = template_path.read_text(encoding="utf-8")

    # 2. 填充模板
    prompt = fill_storyboard_template(template, storyboard_data, characters_data)
    (output_dir / "storyboard_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"  → 提示词已生成 ({len(prompt)} 字符)")

    storyboard_path = output_dir / "storyboard.png"
    om_error = None

    # 3. 尝试调用 OM image_selector
    try:
        from vendor.video_tools.tools.graphics.image_selector import ImageSelector
        selector = ImageSelector()

        print(f"  → image_selector: 生成故事板图片...")

        result = selector.execute({
            "prompt": prompt,
            "width": video_width,
            "height": video_height,
            "aspect_ratio": aspect_ratio,
            "output_path": str(storyboard_path),
        })

        if result.success:
            # 从 result.data 中提取输出路径
            out_path = result.data.get("output_path") or result.data.get("image_path")
            if out_path and Path(out_path).exists():
                # 如果输出不在目标位置，复制过去
                if Path(out_path) != storyboard_path:
                    import shutil
                    shutil.copy2(out_path, storyboard_path)
                print(f"  ✓ Phase 2 完成: storyboard.png (provider: OM)")

                # Quality gate: Phase 2
                qg_report = run_quality_check("phase2", output_dir)
                if not qg_report.passed:
                    return {"status": "error", "error": f"Phase 2 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}

                # --- M2: 分镜图序列（每镜头一张）---
                generated = _generate_shot_images(output_dir, storyboard_data)
                if generated:
                    composition_report = _validate_storyboard_image_composition(output_dir, storyboard_data)
                    if not composition_report["valid"]:
                        return {"status": "error", "error": "Storyboard composition validation failed", "composition_report": composition_report, "duration_s": _elapsed(start)}

                # --- P0-A: HonCut 场景参考图生成 ---
                try:
                    scenes_dir = output_dir / "scenes"
                    scenes_dir.mkdir(exist_ok=True)
                    from phases.phase4.scene_consistency import (
                        _load_style as _load_scene_visual_style,
                        build_scene_reference_prompt,
                    )
                    scene_visual_style = _load_scene_visual_style(
                        output_dir / "visual-style.md"
                    )
                    # 提取所有唯一场景
                    unique_wheres = list(set(
                        shot.get("where", "") for shot in storyboard_data.get("shots", [])
                        if shot.get("where")
                    ))
                    scene_count = 0
                    for where in unique_wheres:
                        scene_id = where.replace(" ", "_").replace("/", "_")[:30]
                        scene_dir = scenes_dir / scene_id
                        scene_dir.mkdir(exist_ok=True)
                        ref_path = scene_dir / "reference.png"
                        if ref_path.exists():
                            scene_count += 1
                            continue
                        try:
                            scene_prompt = build_scene_reference_prompt(
                                where,
                                list(storyboard_data.get("shots", [])),
                                scene_visual_style,
                            )
                            from clients.seedream_client import text_to_image
                            text_to_image(prompt=scene_prompt, output_path=str(ref_path))
                            scene_count += 1
                            print(f"    [P0-A] 场景参考图 {scene_id}/reference.png ✓")
                        except Exception as e:
                            print(f"    [P0-A] 场景参考图 {scene_id} 失败（降级跳过）: {e}")
                    print(f"  → [P0-A] 场景参考图: {scene_count}/{len(unique_wheres)} 个")
                except Exception as e:
                    print(f"  ⚠ [P0-A] 场景参考图生成失败（降级跳过）: {e}")

                return {
                    "status": "done",
                    "duration_s": _elapsed(start),
                    "outputs": ["storyboard.png"],
                    "provider": result.data.get("provider", "unknown"),
                }
            else:
                # result 成功但没有文件路径 — 可能有 URL
                image_url = result.data.get("image_url") or result.data.get("url")
                if image_url:
                    import urllib.request
                    print(f"  → 下载图片: {image_url[:80]}...")
                    urllib.request.urlretrieve(image_url, str(storyboard_path))
                    print(f"  ✓ Phase 2 完成: storyboard.png (provider: OM)")
                    return {"status": "done", "duration_s": _elapsed(start), "outputs": ["storyboard.png"], "provider": "om"}
                else:
                    om_error = "No output file or URL in result"
                    print(f"  ⚠ OM 生成成功但无输出文件/URL: {om_error}")
        else:
            om_error = result.error or "image_selector returned failure"
            print(f"  ⚠ OM image_selector 失败: {om_error}")

    except ImportError as e:
        om_error = f"OM tools unavailable: {e}"
        print(f"  ⚠ OM image_selector 不可用: {e}")
    except Exception as e:
        om_error = str(e)
        print(f"  ⚠ OM image_selector 异常: {e}")

    # 4. 降级到 Seedream API
    print(f"  → 降级到 Seedream API (ARK_AGENT_API_KEY)...")
    try:
        from clients.seedream_client import SeedreamClient
        client = SeedreamClient()
        seedream_size = _storyboard_image_size(
            video_width=video_width,
            video_height=video_height,
        )
        print(f"  → seedream: 生成故事板图片 ({seedream_size}, timeout=180s, retry=3)...")

        # Use retry policy for API call
        def _call_seedream():
            client.text_to_image(
                prompt=prompt,
                output_path=str(storyboard_path),
                size=seedream_size,
                timeout=180,
            )
            if not storyboard_path.exists():
                raise RuntimeError("Seedream 调用成功但未生成文件")

        _retry_with_policy(_call_seedream, max_attempts=3, backoff_factor=2.0)

        if storyboard_path.exists():
            print(f"  ✓ Phase 2 完成: storyboard.png (provider: Seedream, fallback from OM: {om_error})")

            # Quality gate: Phase 2
            qg_report = run_quality_check("phase2", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 2 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}

            # --- M2: 分镜图序列（每镜头一张）---
            generated = _generate_shot_images(output_dir, storyboard_data)
            if generated:
                composition_report = _validate_storyboard_image_composition(output_dir, storyboard_data)
                if not composition_report["valid"]:
                    return {"status": "error", "error": "Storyboard composition validation failed", "composition_report": composition_report, "duration_s": _elapsed(start)}

            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["storyboard.png"],
                "provider": "seedream",
                "fallback_reason": om_error,
            }
        else:
            print(f"  ✗ Seedream 调用成功但未生成文件")
            return {"status": "error", "error": f"OM failed ({om_error}), Seedream succeeded but no file produced", "duration_s": _elapsed(start)}

    except ImportError as e:
        print(f"  ✗ seedream_client 也不可用: {e}")
        return {"status": "error", "error": f"OM failed ({om_error}), Seedream import failed: {e}", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": f"OM failed ({om_error}), Seedream failed: {e}", "duration_s": _elapsed(start)}
