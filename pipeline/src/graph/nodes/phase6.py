"""LangGraph adapters for the existing Phase 6 generation variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END
from langgraph.types import Command

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
) -> dict[str, Any] | Command:
    """Run the shared Phase 6 implementation and preserve retry bookkeeping."""

    output_dir = Path(state["output_dir"])
    replacement = None
    if state.get("retry_count", 0) > 0 and not state.get("dry_run", False):
        from phases.phase8.reshoot_transaction import ReshootTransaction

        shots_dir = output_dir / "shots"
        shot_ids = [
            item.name
            for item in sorted(shots_dir.iterdir()) if item.is_dir()
            and item.name.startswith("S") and (item / "output.mp4").is_file()
        ] if shots_dir.is_dir() else []
        if shot_ids:
            replacement = ReshootTransaction.begin(
                output_dir,
                kind="phase7_quality_retry",
                shot_ids=shot_ids,
                track_budget=False,
            )
            replacement.remove_sources()

    try:
        phase_receipt = runner(
            storyboard_data=state.get("storyboard"),
            output_dir=output_dir,
            dry_run=state["dry_run"],
            chain_mode=state.get("chain_mode", False),
        )
    except Exception as exc:
        if replacement:
            replacement.rollback(str(exc))
        raise

    update: dict[str, Any] = {
        "videos": phase_receipt.get("outputs", []),
        "phase_results": {
            **state.get("phase_results", {}),
            "phase6": phase_receipt,
        },
        "completed_phases": state.get("completed_phases", [])
        + (["phase6"] if phase_receipt.get("status") != "error" else []),
        "retry_count": state.get("retry_count", 0) + 1,
        "skip_phase": state.get("skip_phase", []),
    }
    if phase_receipt.get("status") == "error":
        if replacement:
            replacement.rollback(str(phase_receipt.get("error") or phase_receipt.get("errors")))
        update.update(status="failed", error=f"Phase 6 failed: {phase_receipt.get('error')}")
        return Command(goto=END, update=update)
    if replacement:
        replacement.commit()
    return update


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
