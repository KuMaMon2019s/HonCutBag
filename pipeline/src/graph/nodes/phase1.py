"""LangGraph adapter for the existing combined Phase 1 runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

from graph.state import HonCutState


class Phase1Runner(Protocol):
    """Callable contract implemented by the existing ``run_phase1``."""

    def __call__(
        self,
        text: str,
        output_dir: Path,
        duration: int,
        dry_run: bool,
        reporter: Any | None = None,
        shot_duration: int = ...,
        project_video_spec: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...


def phase1_node(
    state: HonCutState,
    *,
    runner: Phase1Runner,
    reporter: Any | None,
    default_shot_duration: int,
) -> dict[str, Any] | Command:
    """Call Phase 1 and return its existing compatibility State patch."""

    phase_receipt = runner(
        text=state["text"],
        output_dir=Path(state["output_dir"]),
        duration=state["duration"],
        dry_run=state["dry_run"],
        reporter=reporter,
        shot_duration=state.get("shot_duration", default_shot_duration),
        project_video_spec=state.get("project_video_spec"),
    )

    storyboard = phase_receipt.pop("_storyboard", None)
    characters = phase_receipt.pop("_characters", None)
    update: dict[str, Any] = {
        "storyboard": storyboard or {},
        "characters": characters.get("characters", []) if characters else [],
        "phase_results": {
            **state.get("phase_results", {}),
            "phase1": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase1"] if phase_receipt.get("status") != "error" else []),
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        update.update(
            status="failed",
            error=f"Phase 1 failed: {phase_receipt.get('error')}",
        )
        return Command(goto=END, update=update)
    return update
