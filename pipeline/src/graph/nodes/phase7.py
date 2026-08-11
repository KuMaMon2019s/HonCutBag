"""LangGraph adapter for the existing Phase 7 consistency runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from graph.state import HonCutState


class Phase7Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase7``."""

    def __call__(
        self,
        output_dir: Path,
        dry_run: bool,
        storyboard_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def phase7_node(
    state: HonCutState,
    *,
    runner: Phase7Runner,
) -> dict[str, Any]:
    """Run consistency checks and expose the live quality-router metrics."""

    phase_receipt = runner(
        output_dir=Path(state["output_dir"]),
        dry_run=state["dry_run"],
        storyboard_data=state.get("storyboard"),
    )
    quality_report = {
        "slideshow_risk": phase_receipt.get("slideshow_risk", 0.0),
        "variation_score": phase_receipt.get("variation_score", 5.0),
    }
    return {
        "quality_report": quality_report,
        "phase_results": {
            **state.get("phase_results", {}),
            "phase7": phase_receipt,
        },
        "completed_phases": [*state.get("completed_phases", []), "phase7"],
        "skip_phase": state.get("skip_phase", []),
    }
