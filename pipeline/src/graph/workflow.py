"""StateGraph registration for the behavior-compatible HonCut workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

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


def build_workflow(
    *,
    state_schema: type[Any],
    nodes: Mapping[str, NodeCallable],
    review_storyboard_node: NodeCallable,
    route_phase5: RouterCallable,
    quality_gate_router: RouterCallable,
    route_after_phase8: RouterCallable,
    route_after_phase9: RouterCallable,
    auto_approve: bool = False,
) -> StateGraph:
    """Build the uncompiled canonical workflow from injected node callables."""

    graph = StateGraph(state_schema)
    for node_id in PHASE_NODE_IDS:
        graph.add_node(node_id, nodes[node_id])

    if not auto_approve:
        graph.add_node("review_storyboard", review_storyboard_node)

    graph.add_edge(START, "phase1")
    graph.add_edge("phase1", "phase2")
    if auto_approve:
        graph.add_edge("phase2", "phase3")
    else:
        graph.add_edge("phase2", "review_storyboard")
        graph.add_edge("review_storyboard", "phase3")
    graph.add_edge("phase3", "phase4")
    graph.add_edge("phase4", "phase5")

    graph.add_conditional_edges(
        "phase5",
        route_phase5,
        {
            "txt2vid": "phase6_txt2vid",
            "img2vid": "phase6_img2vid",
            "reference": "phase6_reference",
        },
    )

    graph.add_edge("phase6_txt2vid", "phase7")
    graph.add_edge("phase6_img2vid", "phase7")
    graph.add_edge("phase6_reference", "phase7")

    graph.add_conditional_edges(
        "phase7",
        quality_gate_router,
        {
            "pass": "phase8",
            "retry": "phase6_txt2vid",
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


__all__ = ["PHASE_NODE_IDS", "build_workflow"]
