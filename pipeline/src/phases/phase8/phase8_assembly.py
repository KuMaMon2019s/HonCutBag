"""Phase 8 pixel QA, reshoot transactions, and reviewed assembly."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from phases.phase6.phase6_video_gen import run_phase6
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from tools.video_stitcher import build_stitch_plan
from utils.media_profiles import _get_profile_dict
from utils.timing_estimator import estimate_phase_duration


TransitionEmbeddingRunner = Callable[..., dict]


def _select_transition(shot_meta: dict, default_transition: str = "dissolve") -> str:
    """Select transition type based on shot emotion and context."""
    # Check if shot already has a transition_to_next field (from adaptation_engine)
    explicit = shot_meta.get("transition_to_next", "")
    if explicit in ("cut", "dissolve", "fade"):
        return explicit

    # Emotion-based selection
    emotion = shot_meta.get("emotion", "").lower()

    # Gentle emotions → dissolve
    gentle = ["温柔", "深情", "心动", "欣喜", "喜悦", "暧昧", "羞涩"]
    if any(e in emotion for e in gentle):
        return "dissolve"

    # Intense emotions → cut
    intense = ["紧张", "愤怒", "惊讶", "震惊", "慌乱", "压迫"]
    if any(e in emotion for e in intense):
        return "cut"

    # Scene change indicators → fade
    # (detected by comparing 'where' fields between consecutive shots)

    return default_transition


def _finish_phase8(
    phase_result: dict,
    output_dir: Path,
    target_duration: Optional[float],
    enable_reshoot: bool,
    transition: str,
    transition_duration: float,
    media_profile: str,
    reshoot_round: int,
    reshoot_history: list[dict],
    chain_mode: bool,
    transition_embedding_runner: Optional[TransitionEmbeddingRunner] = None,
) -> dict:
    """Apply the duration gate and fail closed when required footage is missing."""
    from phases.phase8.duration_gate import evaluate_duration_gate, trim_excess_to_target
    from phases.phase8.reshoot_transaction import (
        ReshootTransaction,
        mark_cycle_completed,
    )

    outputs = list(phase_result.get("outputs", []))
    for artifact in (
        "storyboard_order_review.json",
        "frame_analysis.json",
        "duration_gate.json",
    ):
        if artifact not in outputs:
            outputs.append(artifact)
    phase_result["outputs"] = outputs

    try:
        gate, reshoot_plan = evaluate_duration_gate(
            output_dir,
            target_duration,
            round_number=reshoot_round,
            reshoots=reshoot_history,
        )
        if gate.get("status") != "OVERLONG":
            duration_trim = trim_excess_to_target(output_dir, target_duration)
            if duration_trim:
                phase_result["duration_trim"] = duration_trim
                if "duration_trim.json" not in phase_result["outputs"]:
                    phase_result["outputs"].append("duration_trim.json")
                print(
                    "  ✂ [8.3] 组装时长归一化: "
                    f"{duration_trim['original_s']:.2f}s → {duration_trim['trimmed_s']:.2f}s",
                    flush=True,
                )
                gate, reshoot_plan = evaluate_duration_gate(
                    output_dir,
                    target_duration,
                    round_number=reshoot_round,
                    reshoots=reshoot_history,
                )
    except Exception as exc:
        print(f"  ⚠⚠ [8.3] 时长闸门执行失败: {exc}；阻止交付", flush=True)
        phase_result["duration_gate_error"] = str(exc)
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration gate failed: {exc}"
        return phase_result

    phase_result["duration_gate"] = gate
    if target_duration is None:
        print("  ⊘ [8.3] target_duration=None，跳过时长闸门", flush=True)
        mark_cycle_completed(output_dir)
        return phase_result
    if gate["passed"]:
        print(
            f"  ✓ [8.3] 时长闸门通过: {gate['actual_s']:.2f}s / {gate['target_s']:.2f}s",
            flush=True,
        )
        mark_cycle_completed(output_dir)
        return phase_result

    if gate.get("status") == "OVERLONG":
        print(
            f"  ⚠⚠ [8.3] 成片过长: 实际 {gate['actual_s']:.2f}s，"
            f"目标 {gate['target_s']:.2f}s，超出 {gate['excess_s']:.2f}s；"
            "必须重新剪辑，禁止生成补拍计划",
            flush=True,
        )
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration gate requires re-edit for overlong assembly: "
            f"excess {gate['excess_s']:.2f}s"
        )
        return phase_result

    print(
        f"  ⚠⚠ [8.3] 时长不足: 实际 {gate['actual_s']:.2f}s，"
        f"目标 {gate['target_s']:.2f}s，缺口 {gate['gap_s']:.2f}s",
        flush=True,
    )
    if not enable_reshoot:
        print("  ⊘ [8.3] enable_reshoot=false，时长缺口未修复，阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration gate requires reshoot but enable_reshoot=false: "
            f"missing {gate['gap_s']:.2f}s"
        )
        return phase_result
    if reshoot_round >= 2:
        print("  ⚠⚠ [8.3] 已达补录上限 2 轮；阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration still fails after 2 reshoot rounds: "
            f"missing {gate['gap_s']:.2f}s"
        )
        return phase_result
    selected = (reshoot_plan or {}).get("shots", [])
    if not selected:
        print("  ⚠⚠ [8.3] 未找到 requested > actual 的短板镜头，无法自动补录", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = "Phase 8 duration gate failed and no reshoot candidates were found"
        return phase_result

    selected_ids = [str(shot["shot_id"]) for shot in selected]
    try:
        transaction = ReshootTransaction.begin(
            output_dir,
            kind="duration_shortfall",
            shot_ids=selected_ids,
        )
        transaction.remove_sources()
    except Exception as exc:
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 could not prepare recoverable duration reshoot: {exc}"
        return phase_result
    print(
        f"  🔄 [8.3] 补录第 {reshoot_round + 1}/2 轮: {', '.join(selected_ids)}；"
        "其余镜头由 Phase 6 自动跳过",
        flush=True,
    )
    storyboard_path = output_dir / "STORYBOARD.json"
    try:
        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        generation = run_phase6(storyboard, output_dir, dry_run=False, chain_mode=chain_mode)
    except Exception as exc:
        transaction.rollback(str(exc))
        print(f"  ⚠⚠ [8.3] 补录调用 Phase 6 失败: {exc}；阻止交付", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration reshoot could not run Phase 6: {exc}"
        return phase_result
    if generation.get("status") == "error":
        failure = str(generation.get("error") or generation.get("errors"))
        transaction.rollback(failure)
        print(f"  ⚠⚠ [8.3] Phase 6 补录失败: {generation.get('error') or generation.get('errors')}", flush=True)
        phase_result["status"] = "error"
        phase_result["error"] = (
            "Phase 8 duration reshoot failed in Phase 6: "
            f"{generation.get('error') or generation.get('errors')}"
        )
        return phase_result

    try:
        transaction.commit()
    except Exception as exc:
        transaction.rollback(str(exc))
        phase_result["status"] = "error"
        phase_result["error"] = f"Phase 8 duration reshoot validation failed: {exc}"
        return phase_result

    history = reshoot_history + [{**reshoot_plan, "phase6_status": generation.get("status")}]
    return run_phase8(
        output_dir,
        dry_run=False,
        transition=transition,
        transition_duration=transition_duration,
        media_profile=media_profile,
        target_duration=target_duration,
        enable_reshoot=enable_reshoot,
        _reshoot_round=reshoot_round + 1,
        _reshoot_history=history,
        chain_mode=chain_mode,
        _transition_embedding_runner=transition_embedding_runner,
    )


def run_phase8(output_dir: Path, dry_run: bool,
               transition: str = "crossfade",
               transition_duration: float = 0.5,
               media_profile: str = "1080p",
               target_duration: Optional[float] = None,
               enable_reshoot: bool = True,
               chain_mode: bool = False,
               _reshoot_round: int = 0,
               _reshoot_history: Optional[list[dict]] = None,
               _continuity_round: int = 0,
               _transition_embedding_runner: Optional[
                   TransitionEmbeddingRunner
               ] = None) -> dict:
    """Phase 8: 逐镜质检、裁切/补录闭环与受审组装。"""
    _banner(8, 9, f"组装引擎 (Assembly) — {transition}", dry_run)
    start = _now()
    phase8_estimate = estimate_phase_duration("phase8")
    print(f"  ⏱ Phase 8 开始 (预估 ~{int(phase8_estimate)}s)")
    output_dir = Path(output_dir)
    reshoot_history = list(_reshoot_history or [])
    exhausted_reshoot_policy = os.environ.get(
        "HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY", "fail"
    ).strip().lower()
    if exhausted_reshoot_policy not in {"fail", "assemble_best"}:
        raise ValueError(
            "HONCUT_PHASE8_EXHAUSTED_RESHOOT_POLICY must be 'fail' or "
            "'assemble_best'"
        )

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频组装")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    # Prove the complete storyboard/video/metadata identity before any pixel
    # analysis or paid reshoot can begin. Resume and manual artifact repair can
    # bypass Phase 4, so Phase 8 owns this invariant too.
    from phases.phase8.inventory import Phase8InventoryError, load_phase8_inventory
    from phases.phase8.reshoot_transaction import durable_attempt_count

    try:
        clip_paths, shot_metas = load_phase8_inventory(output_dir)
        _reshoot_round = max(_reshoot_round, durable_attempt_count(output_dir))
    except (Phase8InventoryError, RuntimeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "duration_s": _elapsed(start),
        }

    shots_dir = output_dir / "shots"

    # Step 8.1: compare storyboard narrative order with the current clip order.
    from phases.phase8.story_order_reviewer import reorder_shots, review_story_order

    current_order = [Path(path).parent.name for path in clip_paths]
    order_review = review_story_order(output_dir, current_order)
    if not order_review["matches_current_order"]:
        clip_paths, shot_metas, changed = reorder_shots(
            clip_paths, shot_metas, order_review["suggested_order"]
        )
        if changed:
            print(
                "  🔀 [8.1] 按剧情审稿建议重排镜头: "
                + " → ".join(Path(path).parent.name for path in clip_paths),
                flush=True,
            )
    if not order_review["narrative_consistent"]:
        print(f"  ⚠ [8.1] 剧情连贯性问题: {order_review['issues']}", flush=True)

    # Step 8.15: the complete chunk trajectory is now available.  Revisit
    # provisional Phase 6 boundaries before per-shot QA or formal assembly.
    from phases.phase8.continuity_adjudication import adjudicate_continuity_seams
    from quality.sam3_sidecar import phase8_sam3_endpoint

    try:
        with phase8_sam3_endpoint(output_dir) as sam3_url:
            continuity_adjudication = adjudicate_continuity_seams(
                output_dir,
                sam3_base_url=sam3_url,
            )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Phase 8 continuity adjudication failed: {exc}",
            "duration_s": _elapsed(start),
        }
    if continuity_adjudication.get("requires_human_review"):
        review_boundaries = [
            boundary["boundary_id"]
            for shot in continuity_adjudication.get("shots", [])
            for boundary in shot.get("boundaries", [])
            if boundary.get("action") == "human_review"
        ]
        return {
            "status": "error",
            "error": (
                "Phase 8 found appearance-level rollback evidence but requires "
                "object-trajectory or human corroboration: "
                + ", ".join(review_boundaries)
            ),
            "duration_s": _elapsed(start),
            "continuity_adjudication": continuity_adjudication,
            "review_artifact": "CONTINUITY_ADJUDICATION.json",
        }
    if continuity_adjudication.get("requires_phase6"):
        requests = [
            request
            for request in json.loads(
                (output_dir / "CONTINUITY_TOPUP_REQUESTS.json").read_text(encoding="utf-8")
            ).get("requests", [])
        ]
        summary = ", ".join(
            f"{item['shot_id']} 缺 {item['deficit_frames']} 帧" for item in requests
        )
        print(
            f"  ↩ [8.15] 检出内部回退，已写入硬裁剪裁决；{summary}，回流 Phase 6",
            flush=True,
        )
        if not enable_reshoot:
            return {
                "status": "error",
                "error": (
                    "Phase 8 temporal seam adjudication requires continuation top-up "
                    f"but enable_reshoot=false: {summary}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        if _continuity_round >= 2:
            return {
                "status": "error",
                "error": (
                    "Phase 8 continuity still requires top-up after 2 feedback rounds: "
                    f"{summary}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        storyboard_path = output_dir / "STORYBOARD.json"
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            generation = run_phase6(
                storyboard,
                output_dir,
                dry_run=False,
                chain_mode=chain_mode,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Phase 8 continuity top-up could not run Phase 6: {exc}",
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
            }
        if generation.get("status") == "error":
            return {
                "status": "error",
                "error": (
                    "Phase 8 continuity top-up failed in Phase 6: "
                    f"{generation.get('error') or generation.get('errors')}"
                ),
                "duration_s": _elapsed(start),
                "continuity_adjudication": continuity_adjudication,
                "phase6_topup": generation,
            }
        history = reshoot_history + [
            {
                "kind": "continuity_topup",
                "round": _continuity_round + 1,
                "requests": requests,
                "phase6_status": generation.get("status"),
            }
        ]
        return run_phase8(
            output_dir,
            dry_run=False,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            target_duration=target_duration,
            enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
            _reshoot_round=_reshoot_round,
            _reshoot_history=history,
            _continuity_round=_continuity_round + 1,
            _transition_embedding_runner=_transition_embedding_runner,
        )

    # Step 8.2: dense per-shot review with actionable keep/trim/reshoot decisions.
    from phases.phase8.frame_analysis import analyze_shot_frames

    frame_report = analyze_shot_frames(shots_dir, output_dir / "frame_analysis.json")
    reshoot_shots = list(frame_report.get("summary", {}).get("reshoot", []))
    if (
        reshoot_shots
        and _reshoot_round >= 2
        and exhausted_reshoot_policy == "assemble_best"
    ):
        unresolved = list(reshoot_shots)
        print(
            "  ⚠ [8.2] 已达补录上限；按显式 assemble_best 策略组装最佳现有素材: "
            + ", ".join(unresolved),
            flush=True,
        )
        frame_report.setdefault("summary", {})["delivery_policy"] = "assemble_best"
        frame_report["summary"]["unresolved_after_reshoot_limit"] = unresolved
        reshoot_history.append(
            {
                "kind": "visual_quality_limit",
                "round": _reshoot_round,
                "shots": unresolved,
                "policy": "assemble_best",
            }
        )
        reshoot_shots = []
    if reshoot_shots:
        if not enable_reshoot:
            return {
                "status": "error",
                "error": (
                    "Phase 8 visual QA requires reshoot but enable_reshoot=false: "
                    + ", ".join(reshoot_shots)
                ),
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
            }
        if _reshoot_round >= 2:
            return {
                "status": "error",
                "error": f"Phase 8 visual QA still fails after 2 reshoot rounds: {', '.join(reshoot_shots)}",
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
                "continuity_adjudication": continuity_adjudication,
                "reshoot_history": reshoot_history,
            }

        from phases.phase8.reshoot_transaction import ReshootTransaction

        try:
            transaction = ReshootTransaction.begin(
                output_dir,
                kind="visual_quality",
                shot_ids=reshoot_shots,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Phase 8 could not prepare recoverable visual reshoot: {exc}",
                "duration_s": _elapsed(start),
            }

        for shot_id in reshoot_shots:
            video_path = shots_dir / shot_id / "output.mp4"
            # A rejected FLF2V result can be caused by an inconsistent
            # generated endpoint. Change the route only after its metadata and
            # source clip have both been backed up by the transaction.
            meta_path = shots_dir / shot_id / "SHOT_META.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            review_entry = frame_report.get("shots", {}).get(shot_id, {})
            semantic_review = review_entry.get("semantic_review") or {}
            reasons = review_entry.get("reasons") or semantic_review.get("issues") or []
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            meta["phase8_reshoot"] = {
                "round": _reshoot_round + 1,
                "qa_contract": semantic_review.get("qa_contract"),
                "issues": [str(reason) for reason in reasons if str(reason).strip()],
            }
            from utils.camera_motion_contracts import apply_camera_motion_contract

            apply_camera_motion_contract(meta)
            if meta.get("gen_strategy") == "flf2v":
                meta["gen_strategy"] = "phantom"
                meta["phase8_reshoot_route_reason"] = (
                    "FLF2V visual QA failure; avoid reusing a possibly "
                    "inconsistent generated endpoint"
                )
                print(
                    f"  ↪ [8.2] {shot_id}: FLF2V 补录改用 Phantom 角色参考路由",
                    flush=True,
                )
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        transaction.remove_sources()

        print(
            f"  🔄 [8.2] 视觉质检补录第 {_reshoot_round + 1}/2 轮: {', '.join(reshoot_shots)}",
            flush=True,
        )
        storyboard_path = output_dir / "STORYBOARD.json"
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            generation = run_phase6(
                storyboard, output_dir, dry_run=False, chain_mode=chain_mode
            )
        except Exception as exc:
            transaction.rollback(str(exc))
            return {
                "status": "error",
                "error": f"Phase 8 visual reshoot could not run Phase 6: {exc}",
                "duration_s": _elapsed(start),
            }
        if generation.get("status") == "error":
            failure = str(generation.get("error") or generation.get("errors"))
            transaction.rollback(failure)
            return {
                "status": "error",
                "error": (
                    "Phase 8 visual reshoot failed in Phase 6: "
                    f"{generation.get('error') or generation.get('errors')}"
                ),
                "duration_s": _elapsed(start),
            }
        missing_outputs = [
            shot_id for shot_id in reshoot_shots
            if not (shots_dir / shot_id / "output.mp4").is_file()
        ]
        if missing_outputs:
            failure = "Phase 6 reported success without regenerated clips: " + ", ".join(missing_outputs)
            transaction.rollback(failure)
            return {
                "status": "error",
                "error": failure,
                "duration_s": _elapsed(start),
            }
        try:
            transaction.commit()
        except Exception as exc:
            transaction.rollback(str(exc))
            return {
                "status": "error",
                "error": f"Phase 8 visual reshoot validation failed: {exc}",
                "duration_s": _elapsed(start),
            }
        history = reshoot_history + [{
            "kind": "visual_quality",
            "round": _reshoot_round + 1,
            "shots": reshoot_shots,
            "phase6_status": generation.get("status"),
        }]
        return run_phase8(
            output_dir,
            dry_run=False,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            target_duration=target_duration,
            enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
            _reshoot_round=_reshoot_round + 1,
            _reshoot_history=history,
            _transition_embedding_runner=_transition_embedding_runner,
        )

    reviewed_order = [Path(path).parent.name for path in clip_paths]

    # Intelligent transition selection based on shot emotions
    print(f"  → 发现 {len(clip_paths)} 个视频片段")

    # ── Smart transition: visual similarity + three-layer voting ──
    smart_decisions = None
    try:
        from utils.shot_embedder import embed_all_shots, compute_transition_similarity
        from tools.smart_transition import decide_all_transitions

        print("  → 智能转场: 抽帧 + 向量化 + 三层决策...")
        embedding_runner = (
            _transition_embedding_runner
            if _transition_embedding_runner is not None
            else embed_all_shots
        )
        embeddings = embedding_runner(str(shots_dir), run_id=str(output_dir.name))
        if embeddings:
            similarities = compute_transition_similarity(embeddings, reviewed_order)
            smart_decisions = decide_all_transitions(
                shot_metas,
                similarities,
                shot_ids=reviewed_order,
            )

            # Log decisions
            for d in smart_decisions:
                sim_str = f"{d['layers']['visual']['similarity']:.2f}" if d['layers']['visual']['similarity'] >= 0 else "N/A"
                print(f"    • {d['pair']}: {d['decision']} "
                      f"(语义={d['layers']['semantic']['choice']}, "
                      f"视觉={d['layers']['visual']['choice']}[{sim_str}], "
                      f"节奏={d['layers']['rhythm']['choice']})")
    except Exception as e:
        print(f"  ⚠ 智能转场不可用: {e}，降级为情绪映射")
        smart_decisions = None

    # Select transition for each shot (except last, which has no "next")
    selected_transitions = []
    for i, shot_meta in enumerate(shot_metas[:-1]):  # Last shot doesn't need a transition
        if smart_decisions and i < len(smart_decisions):
            sel_transition = smart_decisions[i]["decision"]
        else:
            sel_transition = _select_transition(shot_meta, default_transition=transition)
        selected_transitions.append(sel_transition)
        shot_name = reviewed_order[i]
        emotion = shot_meta.get("emotion", "N/A")
        source = "智能" if (smart_decisions and i < len(smart_decisions)) else "情绪"
        print(f"    • {shot_name} → {sel_transition} ({source}, emotion: {emotion})")

    # Determine the most common transition type for batch processing
    if selected_transitions:
        from collections import Counter
        transition_counts = Counter(selected_transitions)
        batch_transition = transition_counts.most_common(1)[0][0]

        # Check if all transitions are the same
        all_same = len(transition_counts) == 1

        if all_same:
            print(f"  → 拼接模式: {batch_transition} (所有镜头统一)")
        else:
            print(f"  → 拼接模式: {batch_transition} (混合模式，使用最常用类型)")
            print(f"    分布: {dict(transition_counts)}")
    else:
        batch_transition = transition
        print(f"  → 拼接模式: {batch_transition} (duration={transition_duration}s)")

    stitch_transition = {
        "dissolve": "crossfade",
        "fade": "fade_through_black",
    }.get(batch_transition, batch_transition)
    if stitch_transition not in {"cut", "crossfade", "fade_through_black"}:
        stitch_transition = "crossfade"
    stitch_plan = build_stitch_plan(
        [
            {"path": path, "duration": shot_metas[index].get("duration", 0) if index < len(shot_metas) else 0}
            for index, path in enumerate(clip_paths)
        ],
        stitch_transition,
        transition_duration,
    )
    clip_paths = stitch_plan.clips

    # The reviewed edit-decision path is primary: this is where per-shot trim
    # decisions become real frame-accurate cuts. Generic concat is fallback.
    transition_dicts = (
        [{"decision": value} for value in selected_transitions]
        if selected_transitions else None
    )
    continuity_plan_path = output_dir / "CONTINUITY_PLAN.json"
    continuity_plan_for_edit = (
        json.loads(continuity_plan_path.read_text(encoding="utf-8"))
        if continuity_plan_path.is_file()
        else None
    )
    has_continuity_transition_lock = any(
        shot.get("boundary_before") == "continuous"
        for shot in (continuity_plan_for_edit or {}).get("shots", [])
    )
    reviewed_edit_error = "reviewed edit path did not complete"
    reviewed_edit_execution_started = False
    try:
        from phases.phase8.edit_decisions import build_edit_decisions, execute_edit_decisions

        print("  → 构建 reviewed edit_decisions（质检裁切 + 音频归一化）...")
        assembly_profile = _get_profile_dict(media_profile)
        edit_decisions = build_edit_decisions(
            shots_dir=shots_dir,
            target_width=int(assembly_profile["width"]),
            target_height=int(assembly_profile["height"]),
            transition_decisions=transition_dicts,
            quality_report=frame_report,
            shot_order=reviewed_order,
            target_duration=target_duration,
            transition_duration=transition_duration,
            fit_mode="cover",
            continuity_plan=continuity_plan_for_edit,
            allow_unresolved_reshoots=(
                exhausted_reshoot_policy == "assemble_best" and _reshoot_round >= 2
            ),
        )
        print(f"  → 执行 reviewed edit_decisions（{len(edit_decisions['cuts'])} 个片段）...")
        reviewed_edit_execution_started = True
        reviewed_edit = execute_edit_decisions(
            edit_decisions,
            output_path=str(output_dir / "raw_assembly.mp4"),
        )
        if reviewed_edit.get("success"):
            print("  ✓ Phase 8 完成: raw_assembly.mp4 (reviewed_edit_decisions)")
            from phases.phase9.audio_mixer import prepare_phase9_audio_assets

            audio_receipt = prepare_phase9_audio_assets(output_dir)
            qg_report = run_quality_check("phase8", output_dir)
            if not qg_report.passed:
                return {
                    "status": "error",
                    "error": f"Phase 8 质检未通过: {qg_report.grade}",
                    "quality_report": qg_report,
                    "duration_s": _elapsed(start),
                }
            return _finish_phase8({
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4", "edit_timeline.json"],
                "method": "reviewed_edit_decisions",
                "transition": batch_transition,
                "transition_duration": transition_duration,
                "clip_count": len(edit_decisions["cuts"]),
                "transition_selections": selected_transitions or None,
                "edit_decisions_segments": reviewed_edit.get("segments"),
                "audio_transition_policy": edit_decisions.get("metadata", {}).get(
                    "audio_transition_policy"
                ),
                "transition_locks": edit_decisions.get("metadata", {}).get(
                    "transition_locks", []
                ),
                "audio_transition_counts": {
                    kind: sum(
                        item.get("audio_transition") == kind
                        for item in edit_decisions.get("transitions", [])
                    )
                    for kind in ("edge_fade", "crossfade")
                },
                "frame_analysis": frame_report.get("summary", {}),
                "reshoot_history": reshoot_history,
                "audio_layer": audio_receipt,
            }, output_dir, target_duration, enable_reshoot, transition,
               transition_duration, media_profile, _reshoot_round, reshoot_history,
               chain_mode, _transition_embedding_runner)
        reviewed_edit_error = str(reviewed_edit.get("error", "unknown error"))
        return {
            "status": "error",
            "error": f"Phase 8 reviewed edit execution failed: {reviewed_edit_error}",
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }
    except Exception as exc:
        reviewed_edit_error = str(exc)
        if reviewed_edit_execution_started:
            return {
                "status": "error",
                "error": f"Phase 8 reviewed edit execution failed: {reviewed_edit_error}",
                "duration_s": _elapsed(start),
                "frame_analysis": frame_report.get("summary", {}),
            }
        print(f"  ⚠ reviewed edit_decisions 构建异常: {exc}；降级为 VideoEdit", flush=True)

    if frame_report.get("summary", {}).get("trim"):
        return {
            "status": "error",
            "error": (
                "Phase 8 cannot safely fall back to raw concat because reviewed trims are required: "
                f"{reviewed_edit_error}"
            ),
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }
    if has_continuity_transition_lock:
        return {
            "status": "error",
            "error": (
                "Phase 8 cannot safely fall back to batch transitions because "
                f"continuous boundary locks are required: {reviewed_edit_error}"
            ),
            "duration_s": _elapsed(start),
            "frame_analysis": frame_report.get("summary", {}),
        }

    # Generic concat cannot apply per-shot decisions, so it is only reached
    # when the reviewed editor itself is unavailable or fails technically.
    try:
        from tools.video.video_edit import VideoEdit

        editor = VideoEdit()
        concat_output = output_dir / ".video_edit_concat.mp4"
        final_output = output_dir / "raw_assembly.mp4"
        video_edit_transition = "cut" if stitch_plan.transition == "cut" else "crossfade"
        call_started = time.time()
        print(
            f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] VideoEdit.concat: "
            f"{len(clip_paths)} clips, transition={video_edit_transition}, "
            f"crossfade={stitch_plan.duration}s"
        )
        concat_result = editor.execute({
            "operation": "concat",
            "input_paths": clip_paths,
            "output_path": str(concat_output),
            "transition": video_edit_transition,
            "crossfade_duration": stitch_plan.duration,
        })
        print(
            f"    VideoEdit.concat result: success={concat_result.success}, "
            f"elapsed={time.time() - call_started:.1f}s, "
            f"output={concat_output if concat_result.success else None}, "
            f"error={concat_result.error}"
        )
        if not concat_result.success:
            raise RuntimeError(concat_result.error or "VideoEdit.concat failed")

        # Trim the assembled container to its exact computed timeline.  Using
        # actual probed clip durations avoids stale SHOT_META duration values.
        clip_durations = []
        for clip_path in clip_paths:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=30, check=True,
            )
            clip_durations.append(float(probe.stdout.strip().splitlines()[0]))
        trim_end = sum(clip_durations)
        if video_edit_transition == "crossfade":
            trim_end -= stitch_plan.duration * (len(clip_paths) - 1)

        trim_started = time.time()
        print(
            f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] VideoEdit.trim: "
            f"start=0.0s, end={trim_end:.3f}s"
        )
        trim_result = editor.execute({
            "operation": "trim",
            "input_path": str(concat_output),
            "output_path": str(final_output),
            "start_time": 0.0,
            "end_time": trim_end,
        })
        print(
            f"    VideoEdit.trim result: success={trim_result.success}, "
            f"elapsed={time.time() - trim_started:.1f}s, output={final_output}, "
            f"error={trim_result.error}"
        )
        if not trim_result.success:
            raise RuntimeError(trim_result.error or "VideoEdit.trim failed")
        concat_output.unlink(missing_ok=True)

        print(f"  ✓ Phase 8 完成: raw_assembly.mp4 (VideoEdit)")
        from phases.phase9.audio_mixer import prepare_phase9_audio_assets
        audio_receipt = prepare_phase9_audio_assets(output_dir)
        qg_report = run_quality_check("phase8", output_dir)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 8 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}
        return _finish_phase8({
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": ["raw_assembly.mp4"],
            "method": "VideoEdit",
            "transition": video_edit_transition,
            "transition_duration": stitch_plan.duration,
            "clip_count": len(clip_paths),
            "transition_selections": selected_transitions if selected_transitions else None,
            "trim_end_s": round(trim_end, 3),
            "frame_analysis": frame_report.get("summary", {}),
            "reshoot_history": reshoot_history,
            "audio_layer": audio_receipt,
        }, output_dir, target_duration, enable_reshoot, transition,
           transition_duration, media_profile, _reshoot_round, reshoot_history,
           chain_mode, _transition_embedding_runner)
    except Exception as e:
        try:
            (output_dir / ".video_edit_concat.mp4").unlink(missing_ok=True)
        except OSError:
            pass
        print(f"  ⚠ VideoEdit 失败: {e}，降级为 VideoStitch")

    # Final fallback: OM VideoStitch for keep-only projects.
    try:
        from vendor.video_tools.tools.video.video_stitch import VideoStitch
        stitcher = VideoStitch()
        result = stitcher.execute({
            "operation": "stitch",
            "clips": clip_paths,
            "output_path": str(output_dir / "raw_assembly.mp4"),
            "transition": stitch_plan.transition,
            "transition_duration": stitch_plan.duration,
            "auto_normalize": True,
            "profile": media_profile,
        })

        if result.success:
            print(f"  ✓ Phase 8 完成: raw_assembly.mp4 (VideoStitch fallback)")
            from phases.phase9.audio_mixer import prepare_phase9_audio_assets
            audio_receipt = prepare_phase9_audio_assets(output_dir)

            # Quality gate: Phase 8
            qg_report = run_quality_check("phase8", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 8 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}

            return _finish_phase8({
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": ["raw_assembly.mp4"],
                "method": f"VideoStitch_{batch_transition}_fallback",
                "transition": stitch_plan.transition,
                "transition_duration": stitch_plan.duration,
                "clip_count": len(clip_paths),
                "transition_selections": selected_transitions if selected_transitions else None,
                "stitch_offsets": stitch_plan.offsets,
                "frame_analysis": frame_report.get("summary", {}),
                "reshoot_history": reshoot_history,
                "audio_layer": audio_receipt,
            }, output_dir, target_duration, enable_reshoot, transition,
               transition_duration, media_profile, _reshoot_round, reshoot_history,
               chain_mode, _transition_embedding_runner)
        else:
            return {"status": "error", "error": result.error, "duration_s": _elapsed(start)}

    except ImportError as e:
        return {"status": "error", "error": f"VideoStitch unavailable: {e}", "duration_s": _elapsed(start)}


__all__ = ["run_phase8"]
