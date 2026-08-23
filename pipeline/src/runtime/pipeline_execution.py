"""Complete HonCut pipeline lifecycle and phase orchestration."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

try:
    from langgraph.errors import GraphInterrupt
    LANGGRAPH_AVAILABLE = True
except ImportError:
    GraphInterrupt = None
    LANGGRAPH_AVAILABLE = False

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from phases.phase1.phase1_pipeline import run_phase1
from phases.phase2.phase2_storyboard import run_phase2
from phases.phase3.phase3_character import run_phase3
from phases.phase4.phase4_orchestrator import run_phase4
from phases.phase5.supervision import _run_storyboard_supervision
from phases.phase6.phase6_video_gen import run_phase6
from phases.phase7.phase7_consistency import run_phase7
from phases.phase8.phase8_assembly import run_phase8
from phases.phase9.phase9_post import run_phase9
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _elapsed, _ensure_dir, _now
from runtime.pipeline_checkpoints import (
    PHASE_ORDER,
    _checkpoint_path,
    _read_checkpoint,
    _record_stage_checkpoint,
    _resume_skip_phases,
    get_sqlite_checkpointer,
    load_state_from_sqlite,
)
from runtime.pipeline_reports import _write_report
from tools.checkpoint import invalidate_checkpoint_from as invalidate_stage_checkpoint
from utils.media_profiles import (
    DEFAULT_MEDIA_PROFILE,
    _get_profile_dict,
    _project_video_spec,
)
from utils.progress_reporter import ProgressReporter
from utils.source_paths import PROJECT_ROOT
from utils.timing_estimator import estimate_total


def run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    shot_duration: int = AVG_SHOT_DURATION,
    chain_mode: bool = False,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = DEFAULT_MEDIA_PROFILE,
    enable_reshoot: bool = True,
    no_real_person: bool = False,
    resume: bool = False,
    auto_approve: bool = True,
    resume_from: str = None,
    accept_code_change_from: str = None,
    project_id: str = "local",
    *,
    _phase_owner=None,
) -> dict:
    """Run the pipeline without leaking its privacy mode into later runs."""
    previous_no_real_person = os.environ.get("HONCUT_NO_REAL_PERSON")
    try:
        return _run_pipeline(
            text=text,
            input_file=input_file,
            duration=duration,
            shot_duration=shot_duration,
            chain_mode=chain_mode,
            dry_run=dry_run,
            skip_phase=skip_phase,
            output_dir=output_dir,
            project_id=project_id,
            transition=transition,
            transition_duration=transition_duration,
            media_profile=media_profile,
            enable_reshoot=enable_reshoot,
            no_real_person=no_real_person,
            resume=resume,
            auto_approve=auto_approve,
            resume_from=resume_from,
            accept_code_change_from=accept_code_change_from,
            _phase_owner=_phase_owner,
        )
    finally:
        if previous_no_real_person is None:
            os.environ.pop("HONCUT_NO_REAL_PERSON", None)
        else:
            os.environ["HONCUT_NO_REAL_PERSON"] = previous_no_real_person


def _run_pipeline(
    text: str = None,
    input_file: str = None,
    duration: int = 60,
    shot_duration: int = AVG_SHOT_DURATION,
    chain_mode: bool = False,
    dry_run: bool = False,
    skip_phase: list = None,
    output_dir: str = ".",
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    media_profile: str = DEFAULT_MEDIA_PROFILE,
    enable_reshoot: bool = True,
    no_real_person: bool = False,
    resume: bool = False,
    auto_approve: bool = True,
    resume_from: str = None,
    accept_code_change_from: str = None,
    project_id: str = "local",
    *,
    _phase_owner=None,
) -> dict:
    """
    主入口：端到端管线

    Args:
        text: 故事文本（直接传入）
        input_file: 故事文本文件路径（与 text 二选一）
        duration: 目标视频时长（秒）
        shot_duration: 每镜平均时长（秒）
        chain_mode: Seedance 尾帧接力模式
        dry_run: dry-run 模式（Phase 1 实际调 LLM，Phase 3 skip-images，Phase 4 dry-run，Phase 6-8 跳过）
        skip_phase: 跳过指定 phase 列表，如 [3, 8]
        output_dir: 输出目录
        project_id: 项目隔离标识；默认 `local`
        transition: Phase 8 转场模式 ("crossfade" | "fade" | "cut")
        transition_duration: Phase 8 转场时长（秒），默认 0.5
        media_profile: 编码配置名称，从 MEDIA_PROFILES 中选择（默认 "480p"）
        enable_reshoot: 视觉缺陷或时长不足时是否允许调用 Phase 6 补录（默认 True，最多两轮）
        no_real_person: 将所有角色锁定为带多样化可见妆造锚点的虚构 CGI 设计
        resume: 从检查点恢复，跳过已完成的 Phase
        accept_code_change_from: 显式接受代码变更并从指定 Phase 继续；其他身份变化仍拒绝

    Returns:
        pipeline_report dict
    """
    phase_owner = _phase_owner or sys.modules[__name__]
    run_phase1 = phase_owner.run_phase1
    run_phase2 = phase_owner.run_phase2
    run_phase3 = phase_owner.run_phase3
    run_phase4 = phase_owner.run_phase4
    run_phase6 = phase_owner.run_phase6
    run_phase7 = phase_owner.run_phase7
    run_phase8 = phase_owner.run_phase8
    run_phase9 = phase_owner.run_phase9
    supervision_runner = getattr(
        phase_owner,
        "_run_storyboard_supervision",
        _run_storyboard_supervision,
    )

    skip_phase = list(skip_phase or [])
    output_path = Path(output_dir).resolve()
    _ensure_dir(output_path)
    os.environ["HONCUT_NO_REAL_PERSON"] = "1" if no_real_person else "0"

    if accept_code_change_from is not None:
        from utils.artifact_chain import (
            PHASE_SEQUENCE,
            can_resume_from,
            invalidate_checkpoints_from,
        )

        if not resume:
            raise ValueError("code change acceptance requires resume mode")
        if accept_code_change_from not in PHASE_SEQUENCE:
            raise ValueError(
                f"code change acceptance has unknown Phase: {accept_code_change_from}"
            )
        if resume_from and accept_code_change_from != resume_from:
            raise ValueError(
                "code change acceptance Phase must match --resume-from"
            )
        if not can_resume_from(accept_code_change_from, output_path):
            raise RuntimeError(
                "code change acceptance refused: prerequisite artifacts are "
                f"incomplete for {accept_code_change_from}"
            )

    # Resolve source and run identity before consulting any checkpoint. This
    # prevents an old "all phases complete" record from short-circuiting a new
    # script, model, provider, geometry, or code version.
    if text is None and input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    if not text and not resume:
        raise ValueError("必须提供 --text 或 --input 参数")
    text = text or ""
    project_video_spec = _project_video_spec(media_profile)
    from runtime.run_manifest import prepare_run_manifest
    from utils.config import get_video_route

    configured_video_provider = os.environ.get("VIDEO_PROVIDER", "seedance").lower()
    effective_video_provider = (
        "seedance"
        if configured_video_provider in {"bridge", "ark"}
        else configured_video_provider
    )
    effective_video_route = get_video_route(configured_video_provider)

    run_manifest = prepare_run_manifest(
        output_path,
        source_text=text,
        resolved_config={
            "project_id": project_id,
            "duration": duration,
            "shot_duration": shot_duration,
            "chain_mode": chain_mode,
            "transition": transition,
            "transition_duration": transition_duration,
            "media_profile": media_profile,
            "enable_reshoot": enable_reshoot,
            "no_real_person": no_real_person,
            "dry_run": dry_run,
            "video_provider": effective_video_provider,
            "video_generation_mode": effective_video_route,
            "video_model": os.environ.get(
                "SEEDANCE_MODEL",
                os.environ.get("VIDEO_MODEL", "doubao-seedance-2.0-fast"),
            ),
            "project_video_spec": project_video_spec,
        },
        repo_root=PROJECT_ROOT,
        resume=resume,
        accepted_code_change_from=accept_code_change_from,
    )
    spec_path = output_path / "PROJECT_VIDEO_SPEC.json"
    spec_temporary = spec_path.with_suffix(".json.tmp")
    spec_temporary.write_text(
        json.dumps(project_video_spec, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(spec_temporary, spec_path)

    # --- M6: --resume-from 支持 ---
    if resume_from:
        from utils.artifact_chain import PHASE_SEQUENCE, can_resume_from

        if resume_from not in PHASE_SEQUENCE:
            raise ValueError(f"未知 Phase: {resume_from}")
        if not can_resume_from(resume_from, output_path):
            raise RuntimeError(
                f"Resume-from {resume_from} refused: prerequisite artifacts are incomplete"
            )
        invalidated = invalidate_stage_checkpoint(
            _checkpoint_path(output_path),
            resume_from,
            PHASE_SEQUENCE,
        )
        invalidated_artifact_receipts = invalidate_checkpoints_from(
            resume_from,
            output_path,
        )
        skip_phase = _resume_skip_phases(skip_phase, resume_from)
        print(f"  🔄 [M6] Resume-from {resume_from}: 跳过 {skip_phase}")
        stale_phases = list(dict.fromkeys([*invalidated, *invalidated_artifact_receipts]))
        if stale_phases:
            print(
                "  ♻ [M6] 已将目标阶段及下游 checkpoint 标记为 stale: "
                + ", ".join(stale_phases)
            )

    # ---- 进度报告系统初始化 ----
    # 编排器为每个 Phase 子进程设置 HONCUT_APPEND_EVENTS=1，跨阶段 events 历史保留。
    reporter = ProgressReporter(
        str(output_path),
        total_phases=len(PHASE_ORDER),
        clear_events=not os.environ.get("HONCUT_APPEND_EVENTS"),
    )

    # --- M6: 产物链（增量）---
    try:
        from utils.artifact_chain import save_checkpoint as save_artifact_checkpoint, can_resume_from
        M6_AVAILABLE = True
    except ImportError:
        M6_AVAILABLE = False

    # ---- Resume: 读取检查点 ----
    completed_phases = set()
    resume_snapshot = None
    resume_uses_graph = False
    if resume:
        from runtime.checkpoint_resolution import resolve_resume_snapshot

        graph_states = []
        if not resume_from:
            graph_states = [
                (
                    "graph",
                    load_state_from_sqlite(
                        output_path,
                        thread_id=run_manifest["run_fingerprint"],
                    ),
                ),
                (
                    "sqlite-stage",
                    load_state_from_sqlite(output_path, thread_id="pipeline_run"),
                ),
            ]
        resume_snapshot = resolve_resume_snapshot(
            output_path,
            run_fingerprint=run_manifest["run_fingerprint"],
            project_id=project_id,
            graph_states=graph_states,
        )
        completed_phases = set(resume_snapshot.completed_phases)
        resume_uses_graph = resume_snapshot.source == "graph"
        if completed_phases:
            print(
                f"\n  🔄 Resume 模式 ({resume_snapshot.source}): "
                f"跳过已完成的 Phase: {sorted(completed_phases)}"
            )
        else:
            print("\n  🔄 Resume 模式: 无可信检查点，从头开始")

        if len(completed_phases) == len(PHASE_ORDER):
            print("  ✓ 所有 Phase 已完成，无需重新运行")
            cp = _read_checkpoint(output_path)
            reporter.mark_completed()
            return {
                "status": "completed",
                "resumed": True,
                "completed_phases": sorted(completed_phases),
                "output_dir": str(output_dir),
                "timestamp": cp.get("timestamp", "") if cp else "",
            }

    total_start = _now()
    report = {
        "status": "completed",
        "input_text_length": len(text),
        "duration_target_s": duration,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "resumed": resume,
        "phases": {},
    }

    print(f"\n{'#'*60}")
    print(f"  Honcut AI Video Pipeline")
    print(f"  文本长度: {len(text)} 字 | 目标时长: {duration}s | dry-run: {dry_run}")
    print(f"  输出目录: {output_dir}")
    if resume and completed_phases:
        print(f"  🔄 Resume: 已完成 {len(completed_phases)}/{len(PHASE_ORDER)} Phase")
    print("  ⏭️ 人工故事板复查已禁用（100% 跳过）")
    
    # 打印预估总耗时
    if not dry_run:
        _est = estimate_total(num_characters=3, num_shots=10)  # 默认值，实际运行时会根据数据调整
        print(f"  ⏱ 预估总耗时: {_est['total_human']} (基于历史数据)")
    
    print(f"{'#'*60}")

    # --- LangGraph StateGraph execution path ---
    if LANGGRAPH_AVAILABLE and not skip_phase and (
        not resume or not completed_phases or resume_uses_graph
    ):
        print(f"\n  🚀 Using LangGraph StateGraph for pipeline execution")
        try:
            # Build the graph through the production composition root.
            from graph.composition import build_pipeline_graph as build_composed_graph

            graph = build_composed_graph(
                auto_approve=auto_approve,
                reporter=reporter,
                phase_owner=phase_owner,
            )
            
            if graph is None:
                raise RuntimeError("Failed to build pipeline graph")
            
            # Create SQLite checkpointer
            saver = get_sqlite_checkpointer(output_path)
            checkpointer = None
            if saver:
                try:
                    checkpointer = saver.__enter__()
                    app = graph.compile(checkpointer=checkpointer)
                except Exception as e:
                    print(f"  ⚠ SQLite checkpointer failed: {e}; compiling without checkpointer")
                    checkpointer = None
                    app = graph.compile()
            else:
                app = graph.compile()
            
            # Seed the live graph through the validated, checkpoint-safe contract.
            from graph.context import initial_state_from_config
            from graph.migrations import latest_error_message, migrate_state
            from schemas.workflow import GraphRunConfig

            run_config = GraphRunConfig(
                run_id=run_manifest["run_fingerprint"],
                project_id=project_id,
                input_text=text,
                output_dir=str(output_path),
                target_duration_s=duration,
                shot_duration_s=shot_duration,
                dry_run=dry_run,
                chain_mode=chain_mode,
                auto_approve=auto_approve,
                transition=transition,
                transition_duration_s=transition_duration,
                media_profile=media_profile,
                project_video_spec=project_video_spec,
                enable_reshoot=enable_reshoot,
                resume=resume,
                resume_from=resume_from,
                skip_phase=skip_phase,
            )
            initial_state = initial_state_from_config(
                run_config,
                include_legacy_aliases=False,
            )
            
            # Config for threading
            config = {
                "configurable": {
                    "thread_id": run_manifest["run_fingerprint"],
                }
            }
            
            # Handle resume: if resuming, try to get existing state
            invocation_input = initial_state
            if resume and checkpointer and resume_uses_graph:
                try:
                    existing_state = app.get_state(config)
                    if existing_state:
                        # Safely check for values attribute
                        state_values = getattr(existing_state, 'values', None)
                        if state_values and isinstance(state_values, dict):
                            print(f"  🔄 Resuming from LangGraph checkpoint")
                            migrate_state(state_values)
                            invocation_input = None
                except Exception as e:
                    raise RuntimeError(
                        f"failed to load trusted graph checkpoint: {e}"
                    ) from e
            
            # Execute the graph
            try:
                final_state = app.invoke(invocation_input, config=config)

                pending_interrupts = final_state.get("__interrupt__", ())
                if pending_interrupts:
                    raise RuntimeError(
                        "unexpected graph interrupt: human review is disabled"
                    )

                final_status = final_state.get("status", "completed")
                if final_status == "running":
                    final_status = "failed"
                
                # Build report from final state
                report = {
                    "status": final_status,
                    "input_text_length": len(text),
                    "duration_target_s": duration,
                    "dry_run": dry_run,
                    "output_dir": str(output_dir),
                    "resumed": resume,
                    "phases": final_state.get("phase_results", {}),
                    "total_duration_s": _elapsed(total_start),
                    "final_video": final_state.get("final_video", ""),
                    "langgraph": True,
                }
                
                if report["status"] == "completed":
                    reporter.mark_completed()
                else:
                    reporter.mark_failed(
                        latest_error_message(final_state)
                        or f"Pipeline ended with status: {report['status']}"
                    )
                
                # Write report
                _write_report(report, output_dir)
                
                # Print summary
                print(f"\n{'#'*60}")
                print(f"  Pipeline {report['status'].upper()} (LangGraph)")
                print(f"  总耗时: {report['total_duration_s']}s")
                for pid, pdata in report["phases"].items():
                    status_icon = {"done": "✓", "skipped": "⊘", "error": "✗"}.get(pdata.get("status", ""), "?")
                    dur = pdata.get("duration_s", "-")
                    print(f"    {status_icon} Phase {pid}: {pdata.get('status', '?')} ({dur}s)")
                print(f"{'#'*60}\n")
                
                return report
                
            except GraphInterrupt as e:
                raise RuntimeError(
                    "unexpected graph interrupt: human review is disabled"
                ) from e
                
        except Exception as e:
            print(f"\n  ⚠ LangGraph execution failed: {e}")
            traceback.print_exc()
            reporter.mark_failed(f"LangGraph execution failed: {e}")
            report.update(
                status="failed",
                error=f"LangGraph execution failed: {e}",
                total_duration_s=_elapsed(total_start),
            )
            _write_report(report, output_dir)
            return report
    
    # --- Sequential execution (fallback or when skip_phase is used) ---
    if LANGGRAPH_AVAILABLE and not skip_phase and (
        not resume or not completed_phases or resume_uses_graph
    ):
        pass  # Already tried above
    else:
        print(f"\n  📋 Using sequential execution mode")

    # ---- Phase 1: 导演拆解 + 编剧引擎 (必须成功) ----
    storyboard_data = None
    characters_data = None
    if 1 in skip_phase:
        report["phases"]["phase1"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase1" in completed_phases:
        # Resume: 从 checkpoint 加载 Phase 1 结果
        cp = _read_checkpoint(output_path)
        p2 = dict(cp["results"].get("phase1", {"status": "done"}))
        p2.setdefault("status", "done")
        storyboard_data = None
        characters_data = None
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())
        report["phases"]["phase1"] = {**p2, "resumed": True}
        print(f"  🔄 Phase 1: 从 checkpoint 恢复 (已跳过)")
    else:
        reporter.phase_start("phase1", "导演拆解 + 编剧引擎")
        p2 = run_phase1(
            text,
            output_path,
            duration,
            dry_run,
            reporter=reporter,
            shot_duration=shot_duration,
            project_video_spec=project_video_spec,
        )
        # 提取内部数据（不写入 report）
        storyboard_data = p2.pop("_storyboard", None)
        characters_data = p2.pop("_characters", None)
        report["phases"]["phase1"] = p2

        if p2["status"] == "error":
            reporter.mark_failed(f"Phase 1 failed: {p2.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 1 failed: {p2.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        reporter.phase_done("phase1", "导演拆解 + 编剧引擎完成", duration_s=p2.get("duration_s"))
        # 写入 checkpoint
        _record_stage_checkpoint(output_path, "phase1", p2)

    # --- P2-5b: HonCut 质检阻断 ---
    try:
        p2_result = report["phases"].get("phase1", {})
        review = p2_result.get("storyboard_review", {})
        if review.get("grade") == "D":
            print(f"  🚫 [P2-5b] 分镜审核 D 级，管线中止（节省后续 token）")
            report["status"] = "aborted_quality"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        elif review.get("grade") == "C":
            print(f"  ⚠ [P2-5b] 分镜审核 C 级，继续但需注意质量")
    except Exception as e:
        print(f"  ⚠ [P2-5b] 质检阻断检查失败（降级跳过）: {e}")

    # 如果 Phase 1 被跳过或数据为空，尝试从文件读
    if 1 in skip_phase or storyboard_data is None:
        sb_path = output_path / "STORYBOARD.json"
        ch_path = output_path / "CHARACTERS.json"
        if sb_path.exists():
            storyboard_data = json.loads(sb_path.read_text())
        if ch_path.exists():
            characters_data = json.loads(ch_path.read_text())

    # ---- Phase 2: 故事板图片生成 (OM image_selector) ----
    if 2 in skip_phase:
        report["phases"]["phase2"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase2" in completed_phases:
        cp = _read_checkpoint(output_path)
        p2_5 = dict(cp["results"].get("phase2", {"status": "done"}))
        p2_5.setdefault("status", "done")
        report["phases"]["phase2"] = {**p2_5, "resumed": True}
        print(f"  🔄 Phase 2: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None or characters_data is None:
        report["phases"]["phase2"] = {"status": "skipped", "reason": "no storyboard/characters data"}
    else:
        reporter.phase_start("phase2", "故事板图片生成")
        p2_5 = run_phase2(storyboard_data, characters_data, Path(output_dir), dry_run)
        report["phases"]["phase2"] = p2_5
        if p2_5["status"] == "error":
            reporter.phase_done("phase2", f"故事板图片生成失败: {p2_5.get('error')}", duration_s=p2_5.get("duration_s"))
            reporter.mark_failed(f"Phase 2 failed: {p2_5.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 2 failed: {p2_5.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase2", "故事板图片生成完成", duration_s=p2_5.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase2", p2_5)

    # ---- Phase 3: 角色工厂 ----
    # Checkpoints created before the semantic four-view contract may contain
    # four existing files but no valid angle/background/identity evidence.
    # Never let resume bypass the current blocking gate.
    phase3_resume_quality = None
    if 3 not in skip_phase and resume and "phase3" in completed_phases:
        phase3_resume_quality = run_quality_check("phase3", output_path)
        if not phase3_resume_quality.passed:
            print(
                "  ⚠ Phase 3 checkpoint 四视图审核凭证缺失、失败或已过期；"
                "本次恢复将重新执行 Phase 3"
            )
    if 3 in skip_phase:
        report["phases"]["phase3"] = {"status": "skipped", "reason": "user-specified"}
    elif (
        resume
        and "phase3" in completed_phases
        and phase3_resume_quality is not None
        and phase3_resume_quality.passed
    ):
        cp = _read_checkpoint(output_path)
        p3 = dict(cp["results"].get("phase3", {"status": "done"}))
        p3.setdefault("status", "done")
        report["phases"]["phase3"] = {**p3, "resumed": True}
        print(f"  🔄 Phase 3: 从 checkpoint 恢复 (已跳过)")
    elif characters_data is None:
        report["phases"]["phase3"] = {"status": "skipped", "reason": "no characters data"}
    else:
        reporter.phase_start("phase3", "角色工厂")
        p3 = run_phase3(output_dir, characters_data, dry_run)
        report["phases"]["phase3"] = p3
        if p3["status"] == "error":
            reporter.phase_done("phase3", f"角色工厂失败: {p3.get('error')}", duration_s=p3.get("duration_s"))
            reporter.mark_failed(f"Phase 3 failed: {p3.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 3 failed: {p3.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase3", "角色工厂完成", duration_s=p3.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase3", p3)

    # ---- Phase 4: 编排器 ----
    if 4 in skip_phase:
        report["phases"]["phase4"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase4" in completed_phases:
        cp = _read_checkpoint(output_path)
        p4 = dict(cp["results"].get("phase4", {"status": "done"}))
        p4.setdefault("status", "done")
        report["phases"]["phase4"] = {**p4, "resumed": True}
        print(f"  🔄 Phase 4: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["phase4"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        reporter.phase_start("phase4", "编排器")
        p4 = run_phase4(output_dir, dry_run)
        report["phases"]["phase4"] = p4
        if p4["status"] == "error":
            reporter.phase_done("phase4", f"编排器失败: {p4.get('error')}", duration_s=p4.get("duration_s"))
            reporter.mark_failed(f"Phase 4 failed: {p4.get('error')}")
            report["status"] = "failed"
            report["error"] = f"Phase 4 failed: {p4.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        else:
            reporter.phase_done("phase4", "编排器完成", duration_s=p4.get("duration_s"))
            _record_stage_checkpoint(output_path, "phase4", p4)

    # ---- Phase 5: 分镜质检闸门 ----
    # This is deliberately immediately before Phase 6: a resumed or partially
    # selected run must not bypass the last zero/video-cost checkpoint.
    if 5 in skip_phase:
        if 6 not in skip_phase:
            from utils.artifact_chain import can_resume_from

            checkpoint = _read_checkpoint(output_path)
            phase5_receipt = (
                checkpoint.get("results", {}).get("phase5")
                if isinstance(checkpoint, dict)
                else None
            )
            try:
                gate_report = json.loads(
                    (output_path / "storyboard_qa_report.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                gate_report = None
            gate_is_current_and_passing = bool(
                isinstance(phase5_receipt, dict)
                and phase5_receipt.get("status") == "done"
                and isinstance(gate_report, dict)
                and gate_report.get("gate_passed") is True
                and can_resume_from("phase6", output_path)
            )
            if not gate_is_current_and_passing:
                error = (
                    "Phase 6 refused: the current run has no passing Phase 5 "
                    "checkpoint and storyboard QA receipt"
                )
                report["phases"]["phase5"] = {
                    "status": "error",
                    "error": error,
                }
                report["status"] = "failed"
                report["error"] = error
                report["total_duration_s"] = _elapsed(total_start)
                reporter.mark_failed(error)
                _write_report(report, output_dir)
                return report
            report["phases"]["phase5"] = {
                **phase5_receipt,
                "resumed": True,
                "gate_validation": "current-run checkpoint",
            }
        else:
            report["phases"]["phase5"] = {
                "status": "skipped",
                "reason": "user-specified",
            }
    elif storyboard_data is None:
        report["phases"]["phase5"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        from phases.phase5 import storyboard_qa_gate

        reporter.phase_start("phase5", "分镜质检闸门")
        p4_5 = storyboard_qa_gate.run_storyboard_qa_with_correction(
            output_path,
            qa_runner=storyboard_qa_gate.run_storyboard_qa_gate,
            dry_run=dry_run,
        )
        report["phases"]["phase5"] = p4_5
        reporter.phase_done("phase5", f"分镜质检 {p4_5.get('grade', '?')} 级", duration_s=p4_5.get("duration_s"))
        if p4_5["status"] == "error":
            reporter.mark_failed(p4_5.get("error", "Phase 5 blocked Phase 6"))
            report["status"] = "failed"
            report["error"] = p4_5.get("error", "Phase 5 blocked Phase 6")
            report["total_duration_s"] = _elapsed(total_start)
            _write_report(report, output_dir)
            return report
        if dry_run:
            report["phases"]["phase5"]["supervision"] = {
                "status": "skipped",
                "reason": "dry-run",
            }
        else:
            from quality.supervision_agent import SupervisionBlockedError
            try:
                supervision = supervision_runner(storyboard_data, output_path)
                report["phases"]["phase5"]["supervision"] = supervision
            except SupervisionBlockedError as exc:
                reporter.mark_failed(str(exc))
                report["status"] = "failed"
                report["error"] = str(exc)
                report["total_duration_s"] = _elapsed(total_start)
                _write_report(report, output_dir)
                return report
        _record_stage_checkpoint(output_path, "phase5", p4_5)

    # ---- Phase 6: 视频生成 ----
    if 6 in skip_phase:
        report["phases"]["phase6"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase6" in completed_phases:
        cp = _read_checkpoint(output_path)
        p5 = dict(cp["results"].get("phase6", {"status": "done"}))
        p5.setdefault("status", "done")
        report["phases"]["phase6"] = {**p5, "resumed": True}
        print(f"  🔄 Phase 6: 从 checkpoint 恢复 (已跳过)")
    elif storyboard_data is None:
        report["phases"]["phase6"] = {"status": "skipped", "reason": "no storyboard data"}
    else:
        p5 = run_phase6(
            storyboard_data,
            output_dir,
            dry_run,
            chain_mode=chain_mode,
            media_profile=media_profile,
        )
        report["phases"]["phase6"] = p5
        if p5["status"] == "error":
            report["status"] = "failed"
            report["error"] = p5.get("error", "Phase 6 video generation failed")
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase6", p5)

    # ---- Phase 7: handoff into Phase 8 pixel-level QA ----
    if 7 in skip_phase:
        report["phases"]["phase7"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase7" in completed_phases:
        cp = _read_checkpoint(output_path)
        p6 = dict(cp["results"].get("phase7", {"status": "done"}))
        p6.setdefault("status", "done")
        report["phases"]["phase7"] = {**p6, "resumed": True}
        print(f"  🔄 Phase 7: 从 checkpoint 恢复 (已跳过)")
    else:
        # Ensure storyboard_data is available (may be None if Phase 1 was skipped)
        if storyboard_data is None:
            sb_path = output_path / "STORYBOARD.json"
            if sb_path.exists():
                storyboard_data = json.loads(sb_path.read_text())
        
        p6 = run_phase7(Path(output_dir), dry_run, storyboard_data=storyboard_data)
        report["phases"]["phase7"] = p6
        if p6["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 7 failed: {p6.get('error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase7", p6)

        report["quality_gate"] = {
            "passed": True,
            "video_quality_owner": "phase8",
        }

    # ---- Phase 8: 组装引擎 ----
    if 8 in skip_phase:
        report["phases"]["phase8"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase8" in completed_phases:
        cp = _read_checkpoint(output_path)
        p7 = dict(cp["results"].get("phase8", {"status": "done"}))
        p7.setdefault("status", "done")
        report["phases"]["phase8"] = {**p7, "resumed": True}
        print(f"  🔄 Phase 8: 从 checkpoint 恢复 (已跳过)")
    else:
        p7 = run_phase8(
            output_dir, dry_run, transition=transition,
            transition_duration=transition_duration, media_profile=media_profile,
            target_duration=duration, enable_reshoot=enable_reshoot,
            chain_mode=chain_mode,
        )
        report["phases"]["phase8"] = p7
        if p7["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 8 failed: {p7.get('error', 'unknown assembly error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase8", p7)

    # ---- Phase 9: 后期处理 ----
    if 9 in skip_phase:
        report["phases"]["phase9"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase9" in completed_phases:
        cp = _read_checkpoint(output_path)
        p8 = dict(cp["results"].get("phase9", {"status": "done"}))
        p8.setdefault("status", "done")
        report["phases"]["phase9"] = {**p8, "resumed": True}
        print(f"  🔄 Phase 9: 从 checkpoint 恢复 (已跳过)")
    else:
        p8 = run_phase9(
            output_dir, dry_run, media_profile=media_profile,
            target_duration=duration,
        )
        report["phases"]["phase9"] = p8
        if p8["status"] == "error":
            report["status"] = "failed"
            report["error"] = f"Phase 9 failed: {p8.get('error', 'unknown post-processing error')}"
            report["total_duration_s"] = _elapsed(total_start)
            reporter.mark_failed(report["error"])
            _write_report(report, output_dir)
            return report
        else:
            _record_stage_checkpoint(output_path, "phase9", p8)

    # ---- Phase 9.5: Video QA 硬性质检 ----
    if 9.5 in skip_phase:
        report["phases"]["phase9_5"] = {"status": "skipped", "reason": "user-specified"}
    elif resume and "phase9_5" in completed_phases:
        cp = _read_checkpoint(output_path)
        p9_5 = dict(cp["results"].get("phase9_5", {"status": "done"}))
        p9_5.setdefault("status", "done")
        report["phases"]["phase9_5"] = {**p9_5, "resumed": True}
        print(f"  🔄 Phase 9.5: 从 checkpoint 恢复 (已跳过)")
    else:
        try:
            from quality.video_qa import run_video_qa
            delivery_profile = _get_profile_dict(media_profile)
            qa_report = run_video_qa(
                output_dir,
                storyboard_data=storyboard_data,
                expected_width=int(delivery_profile["width"]),
                expected_height=int(delivery_profile["height"]),
                expected_min_duration=float(duration) - 1.0,
                expected_max_duration=float(duration) + 1.0,
            )
            qa_passed = qa_report.verdict == "pass"
            p9_5 = {
                "status": "done" if qa_passed else "error",
                "verdict": qa_report.verdict,
                "grade": qa_report.grade,
                "issues_count": len(qa_report.issues),
                "duration_s": 0,
            }
            report["phases"]["phase9_5"] = p9_5
            if not qa_passed:
                report["status"] = "failed"
                report["quality_gate"] = {
                    "passed": False,
                    "reason": f"Phase 9.5 delivery QA requires revision: {qa_report.grade} grade",
                    "issues": [i.message for i in qa_report.issues],
                }
            else:
                _record_stage_checkpoint(output_path, "phase9_5", p9_5)
        except ImportError:
            report["phases"]["phase9_5"] = {"status": "error", "reason": "video_qa module not available"}
            report["status"] = "failed"
            report["quality_gate"] = {"passed": False, "reason": "Phase 9.5 delivery QA is unavailable"}
        except Exception as e:
            report["phases"]["phase9_5"] = {"status": "error", "error": str(e)}
            report["status"] = "failed"
            report["quality_gate"] = {"passed": False, "reason": f"Phase 9.5 delivery QA failed to run: {e}"}

    report["total_duration_s"] = _elapsed(total_start)

    if report["status"] == "completed":
        report.pop("error", None)
        reporter.mark_completed()
    else:
        reporter.mark_failed(
            report.get("quality_gate", {}).get("reason")
            or report.get("error")
            or f"Pipeline ended with status: {report['status']}"
        )

    # --- M6: 产物链验证 ---
    if M6_AVAILABLE:
        try:
            from utils.artifact_chain import verify_artifacts, save_checkpoint as save_artifact_checkpoint
            for phase_name in PHASE_ORDER:
                phase_report = report.get("phases", {}).get(phase_name, {})
                if phase_report.get("status") != "done":
                    continue
                va = verify_artifacts(phase_name, output_path)
                if va["exists"]:
                    save_artifact_checkpoint(phase_name, output_path, va)
        except Exception as e:
            print(f"  ⚠ [M6] 产物链验证跳过: {e}")

    # 写报告
    _write_report(report, output_dir)

    # 打印总结
    print(f"\n{'#'*60}")
    print(f"  Pipeline {report['status'].upper()}")
    print(f"  总耗时: {report['total_duration_s']}s")
    for pid, pdata in report["phases"].items():
        status_icon = {"done": "✓", "skipped": "⊘", "error": "✗"}.get(pdata["status"], "?")
        dur = pdata.get("duration_s", "-")
        print(f"    {status_icon} Phase {pid}: {pdata['status']} ({dur}s)")
    print(f"{'#'*60}\n")

    return report
