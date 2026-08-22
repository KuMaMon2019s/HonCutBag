"""Phase 6 provider selection and continuity-runtime entry point."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from phases.phase6.direct_generation import _run_phase6_fallback
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.vendor_adapter import VendorAdapter, VendorModel
from utils.timing_estimator import estimate_phase_duration


class _PipelineVideoTool(BaseTool):
    """BaseTool-conforming wrapper around the pipeline's video generator."""
    name = "pipeline_video_generation"
    runtime = ToolRuntime.API
    capabilities = ["i2v", "flf2v"]
    input_schema = {"output_dir": "path"}

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = _now()
        try:
            data = _run_phase6_fallback(
                Path(inputs["output_dir"]),
                chain_mode=bool(inputs.get("chain_mode", False)),
            )
            return ToolResult(data.get("status") == "done", data=data, error=data.get("error"), duration_seconds=_elapsed(started))
        except Exception as exc:
            return ToolResult(False, error=str(exc), duration_seconds=_elapsed(started))


class _LocalVideoVendorAdapter(VendorAdapter):
    id = "honcut-local-video"
    name = "HonCut Local Video"

    def video_request(self, config: dict[str, Any], model: VendorModel) -> Any:
        return _PipelineVideoTool().execute(config)


def run_phase6(
    storyboard_data: dict,
    output_dir: str | Path,
    dry_run: bool,
    chain_mode: bool = False,
    *,
    _adapter_cls=None,
    _quality_runner=None,
    _test_continuity_executor_factory=None,
) -> dict:
    """Phase 6: video generation through the configured provider route."""
    # The phase orchestrator persists paths as strings when resuming.  Normalize
    # at the public phase boundary so both the legacy provider route and the
    # continuity runtime can safely compose child paths with ``/``.
    output_dir = Path(output_dir)
    adapter_cls = _adapter_cls or _LocalVideoVendorAdapter
    quality_runner = _quality_runner or run_quality_check
    _banner(6, 9, "视频生成 (Seedance — reference_to_video)", dry_run)
    start = _now()

    # Estimate based on shot count
    _num_shots = len(storyboard_data.get("shots", [])) if storyboard_data else 0
    if _num_shots == 0:
        # 如果没有 storyboard_data，使用默认值（但不写死 10）
        _num_shots = 5  # 合理的默认值，实际会根据剧本长度计算
    _p5_est = estimate_phase_duration("phase6", num_shots=_num_shots)
    print(f"  ⏱ Phase 6 开始 (预估 ~{int(_p5_est)}s, {_num_shots} 镜头)")

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频生成")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    try:
        from runtime.continuity_chunks import write_shadow_runtime_report

        continuity_runtime = write_shadow_runtime_report(output_dir)
        if continuity_runtime["mode"] == "shadow":
            print(
                "  [continuity] shadow: "
                f"{continuity_runtime.get('shot_count', 0)} shots / "
                f"{continuity_runtime.get('chunk_count', 0)} chunks; provider route unchanged",
                flush=True,
            )
    except (OSError, ValueError, RuntimeError) as error:
        return {
            "status": "error",
            "error": f"Continuity runtime configuration failed: {error}",
            "duration_s": _elapsed(start),
        }

    if continuity_runtime["mode"] == "auto":
        try:
            from quality.seam_calibration import load_seam_calibration
            from runtime.continuity_chunks import load_continuity_plan
            from runtime.continuity_provider import execute_phase6_auto_continuity

            print(
                "  [continuity] auto: grouped generation; Phase 8 owns final seam trim",
                flush=True,
            )
            execution_kwargs = {}
            if _test_continuity_executor_factory is not None:
                execution_kwargs["_test_executor_factory"] = (
                    _test_continuity_executor_factory
                )
            result = execute_phase6_auto_continuity(
                output_dir,
                load_continuity_plan(output_dir / "CONTINUITY_PLAN.json"),
                (
                    load_seam_calibration(output_dir / "CONTINUITY_CALIBRATION.json")
                    if (output_dir / "CONTINUITY_CALIBRATION.json").is_file()
                    else None
                ),
                **execution_kwargs,
            )
            result["duration_s"] = _elapsed(start)
            result["continuity_runtime"] = continuity_runtime
            if result["status"] == "done":
                qg_report = quality_runner("phase6", output_dir)
                if not qg_report.passed:
                    return {
                        "status": "error",
                        "error": f"Phase 6 质检未通过: {qg_report.grade}",
                        "quality_report": qg_report,
                        "duration_s": _elapsed(start),
                        "continuity_runtime": continuity_runtime,
                    }
            return result
        except Exception as error:
            return {
                "status": "error",
                "error": f"Continuity auto execution failed: {error}",
                "duration_s": _elapsed(start),
                "continuity_runtime": continuity_runtime,
            }

    print("  → Phase 6 使用当前配置的视频提供方路由", flush=True)
    try:
        adapter = adapter_cls([
            VendorModel("Local Bridge", "local-video-bridge", "video", ("i2v", "flf2v"))
        ])
        tool_result = adapter.request(
            "local-video-bridge",
            {"output_dir": str(output_dir), "chain_mode": chain_mode},
        )
        result = tool_result.data or {"status": "error", "error": tool_result.error}
        result["duration_s"] = _elapsed(start)
        result["continuity_runtime"] = continuity_runtime
        provider = result.get("provider", "unknown_provider")
        if result["status"] == "done":
            print(f"  ✓ Phase 6 完成: {len(result['outputs'])} 视频 ({provider})")

            # Quality gate: Phase 6
            qg_report = quality_runner("phase6", output_dir)
            if not qg_report.passed:
                return {"status": "error", "error": f"Phase 6 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}

        else:
            print(f"  ✗ Phase 6 失败 ({provider})")
        return result

    except ImportError as e:
        return {"status": "error", "error": f"All video generation methods unavailable: {e}", "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
