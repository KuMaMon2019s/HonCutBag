"""Production composition root for the canonical HonCut workflow.

This module injects the current runtime Phase owners into the pure topology in
``graph.workflow``. The optional ``phase_owner`` argument exists only for
explicit compatibility tests and alternate compositions.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable

from graph.migrations import canonicalize_node_result
from graph.routing import quality_gate_router, route_phase5
from graph.state import HonCutState


def _resolve_phase_owner(phase_owner: ModuleType | Any | None) -> Any:
    return (
        phase_owner
        if phase_owner is not None
        else import_module("runtime.pipeline_execution")
    )


def _invoke(
    state: HonCutState,
    node: Callable[[], Any],
    *,
    legacy_compat: bool,
    quality_target: str = "consistency",
):
    result = node()
    if legacy_compat:
        return result
    return canonicalize_node_result(
        result,
        state,
        quality_target=quality_target,
    )


def node_phase1(
    state: HonCutState,
    *,
    reporter: Any | None = None,
    phase_owner: ModuleType | Any | None = None,
) -> dict[str, Any]:
    from graph.nodes.phase1 import phase1_node

    owner = _resolve_phase_owner(phase_owner)
    return phase1_node(
        state,
        runner=owner.run_phase1,
        reporter=reporter,
        default_shot_duration=owner.AVG_SHOT_DURATION,
    )


def node_phase2(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase2 import phase2_node

    return phase2_node(state, runner=_resolve_phase_owner(phase_owner).run_phase2)


def node_phase3(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase3 import phase3_node

    return phase3_node(state, runner=_resolve_phase_owner(phase_owner).run_phase3)


def node_phase4(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase4 import phase4_node

    return phase4_node(state, runner=_resolve_phase_owner(phase_owner).run_phase4)


def node_phase5_quality(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase5 import phase5_node
    from phases.phase5 import storyboard_qa_gate
    from quality.supervision_agent import SupervisionBlockedError

    owner = _resolve_phase_owner(phase_owner)
    return phase5_node(
        state,
        qa_runner=lambda output_dir: (
            storyboard_qa_gate.run_storyboard_qa_with_correction(
                output_dir,
                qa_runner=storyboard_qa_gate.run_storyboard_qa_gate,
            )
        ),
        supervision_runner=owner._run_storyboard_supervision,
        supervision_blocked_error=SupervisionBlockedError,
    )


def node_phase6_txt2vid(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase6 import phase6_txt2vid_node

    return phase6_txt2vid_node(
        state, runner=_resolve_phase_owner(phase_owner).run_phase6
    )


def node_phase6_img2vid(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase6 import phase6_img2vid_node

    return phase6_img2vid_node(
        state, runner=_resolve_phase_owner(phase_owner).run_phase6
    )


def node_phase6_reference(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase6 import phase6_reference_node

    return phase6_reference_node(
        state, runner=_resolve_phase_owner(phase_owner).run_phase6
    )


def node_phase7(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase7 import phase7_node

    return phase7_node(state, runner=_resolve_phase_owner(phase_owner).run_phase7)


def node_phase8(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase8 import phase8_node

    return phase8_node(state, runner=_resolve_phase_owner(phase_owner).run_phase8)


def node_phase9(
    state: HonCutState, *, phase_owner: ModuleType | Any | None = None
) -> dict[str, Any]:
    from graph.nodes.phase9 import phase9_node

    return phase9_node(state, runner=_resolve_phase_owner(phase_owner).run_phase9)


def node_phase9_5(state: HonCutState) -> dict[str, Any]:
    from graph.nodes.final_qa import final_qa_node

    try:
        from quality.video_qa import run_video_qa
    except ImportError:
        run_video_qa = None
    return final_qa_node(state, runner=run_video_qa)


def build_pipeline_graph(
    *,
    auto_approve: bool = True,
    reporter: Any | None = None,
    phase_owner: ModuleType | Any | None = None,
    legacy_compat: bool = False,
):
    """Build the one production topology with concrete Phase dependencies."""

    from graph import workflow

    owner = _resolve_phase_owner(phase_owner)
    return workflow.build_workflow(
        state_schema=HonCutState,
        nodes={
            "phase1": lambda state: _invoke(
                state,
                lambda: node_phase1(state, reporter=reporter, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase2": lambda state: _invoke(
                state,
                lambda: node_phase2(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase3": lambda state: _invoke(
                state,
                lambda: node_phase3(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase4": lambda state: _invoke(
                state,
                lambda: node_phase4(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase5": lambda state: _invoke(
                state,
                lambda: node_phase5_quality(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase6_txt2vid": lambda state: _invoke(
                state,
                lambda: node_phase6_txt2vid(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase6_img2vid": lambda state: _invoke(
                state,
                lambda: node_phase6_img2vid(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase6_reference": lambda state: _invoke(
                state,
                lambda: node_phase6_reference(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase7": lambda state: _invoke(
                state,
                lambda: node_phase7(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase8": lambda state: _invoke(
                state,
                lambda: node_phase8(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase9": lambda state: _invoke(
                state,
                lambda: node_phase9(state, phase_owner=owner),
                legacy_compat=legacy_compat,
            ),
            "phase9_5": lambda state: _invoke(
                state,
                lambda: node_phase9_5(state),
                legacy_compat=legacy_compat,
                quality_target="final_qa",
            ),
        },
        route_phase5=route_phase5,
        quality_gate_router=quality_gate_router,
        route_after_phase8=import_module("graph.nodes.phase8").route_after_phase8,
        route_after_phase9=import_module("graph.nodes.phase9").route_after_phase9,
        auto_approve=auto_approve,
    )


__all__ = [
    "build_pipeline_graph",
    "node_phase1",
    "node_phase2",
    "node_phase3",
    "node_phase4",
    "node_phase5_quality",
    "node_phase6_img2vid",
    "node_phase6_reference",
    "node_phase6_txt2vid",
    "node_phase7",
    "node_phase8",
    "node_phase9",
    "node_phase9_5",
    "quality_gate_router",
    "route_phase5",
]
