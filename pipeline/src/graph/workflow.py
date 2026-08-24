"""StateGraph registration for the behavior-compatible HonCut workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

NodeCallable = Callable[[Any], Any]
RouterCallable = Callable[[Any], str]

PHASE_NODE_IDS = (
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6_txt2vid",
    "phase6_img2vid",
    "phase6_reference",
    "phase7",
    "phase8",
    "phase9",
    "phase9_5",
)


def route_after_phase(state: Any) -> Literal["continue", "end"]:
    """Stop every ordinary phase edge when its node recorded failure."""

    if isinstance(state, Mapping) and str(state.get("status") or "") in {
        "failed",
        "error",
        "blocked",
    }:
        return "end"
    return "continue"


def build_workflow(
    *,
    state_schema: type[Any],
    nodes: Mapping[str, NodeCallable],
    route_phase5: RouterCallable,
    quality_gate_router: RouterCallable,
    route_after_phase8: RouterCallable,
    route_after_phase9: RouterCallable,
    auto_approve: bool = True,
) -> StateGraph:
    """Build the canonical workflow with human storyboard review disabled.

    ``auto_approve`` remains a compatibility argument for older callers, but
    both values now mean the same deterministic policy: the workflow always
    advances directly from Phase 2 to Phase 3 and can never pause for a human
    storyboard decision.
    """

    graph = StateGraph(state_schema)
    for node_id in PHASE_NODE_IDS:
        graph.add_node(node_id, nodes[node_id])

    graph.add_edge(START, "phase1")
    for phase, target in (
        ("phase1", "phase2"),
        ("phase2", "phase3"),
        ("phase3", "phase4"),
        ("phase4", "phase5"),
    ):
        graph.add_conditional_edges(
            phase,
            route_after_phase,
            {"continue": target, "end": END},
        )

    graph.add_conditional_edges(
        "phase5",
        route_phase5,
        {
            "txt2vid": "phase6_txt2vid",
            "img2vid": "phase6_img2vid",
            "reference": "phase6_reference",
            "block": END,
        },
    )

    for phase in ("phase6_txt2vid", "phase6_img2vid", "phase6_reference"):
        graph.add_conditional_edges(
            phase,
            route_after_phase,
            {"continue": "phase7", "end": END},
        )

    graph.add_conditional_edges(
        "phase7",
        quality_gate_router,
        {
            "pass": "phase8",
            "block": END,
        },
    )
    graph.add_conditional_edges(
        "phase8",
        route_after_phase8,
        {
            "continue": "phase9",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "phase9",
        route_after_phase9,
        {
            "continue": "phase9_5",
            "end": END,
        },
    )
    graph.add_edge("phase9_5", END)
    return graph


__all__ = ["PHASE_NODE_IDS", "build_workflow", "route_after_phase"]
