"""LangGraph adapters for the existing Phase 6 generation variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from graph.state import HonCutState


class Phase6Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase6``."""

    def __call__(
        self,
        storyboard_data: dict[str, Any] | None,
        output_dir: Path,
        dry_run: bool,
        chain_mode: bool = False,
    ) -> dict[str, Any]: ...


class Phase6NodeRunner(Protocol):
    """Callable contract for the live txt2vid compatibility facade."""

    def __call__(self, state: HonCutState) -> dict[str, Any]: ...


def phase6_txt2vid_node(
    state: HonCutState,
    *,
    runner: Phase6Runner,
) -> dict[str, Any]:
    """Run the shared Phase 6 implementation and preserve retry bookkeeping."""

    phase_receipt = runner(
        storyboard_data=state.get("storyboard"),
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
        chain_mode=state.get("chain_mode", False),
    )
    return {
        "videos": phase_receipt.get("outputs", []),
        "phase_results": {
            **state.get("phase_results", {}),
            "phase6": phase_receipt,
        },
        "completed_phases": [*state.get("completed_phases", []), "phase6"],
        "retry_count": state.get("retry_count", 0) + 1,
        "skip_phase": state.get("skip_phase", []),
    }


def phase6_img2vid_node(
    state: HonCutState,
    *,
    txt2vid_node: Phase6NodeRunner,
) -> dict[str, Any]:
    """Preserve the current image-to-video delegation to txt2vid."""

    return txt2vid_node(state)


def phase6_reference_node(
    state: HonCutState,
    *,
    txt2vid_node: Phase6NodeRunner,
) -> dict[str, Any]:
    """Preserve the current reference-to-video delegation to txt2vid."""

    return txt2vid_node(state)
