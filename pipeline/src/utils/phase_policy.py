#!/usr/bin/env python3
"""Phase-specific monitoring policies for the HonCut orchestrator.

Each phase declares the artifacts that prove it is making progress plus the
soft/hard stall thresholds the orchestrator monitor thread uses. A heartbeat
younger than ``soft_stall_s`` is healthy; between soft and hard the monitor
only records an alert; past ``hard_stall_s`` the phase subprocess is stopped
so the pipeline fails closed at a resumable checkpoint instead of burning
paid APIs forever.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PhasePolicy:
    """Monitoring contract for a single pipeline phase."""

    phase: str
    artifacts: List[str] = field(default_factory=list)
    soft_stall_s: int = 300
    hard_stall_s: int = 900

    def __post_init__(self) -> None:
        if self.soft_stall_s <= 0:
            raise ValueError("soft_stall_s must be positive")
        if self.hard_stall_s <= self.soft_stall_s:
            raise ValueError("hard_stall_s must be greater than soft_stall_s")


# Thresholds are calibrated from the 2026-08-12_01 live run:
# - phase1: adaptation wall timeout is 900s, so hard must exceed it.
# - phase2/3: image generation lands roughly every 2 minutes under rate limits
#   (phase3 also has a legitimate 120s cooldown), so 600s soft keeps headroom.
# - phase4: metadata only, completes in seconds.
# - phase5: bounded L3 wall clock of 240s plus deterministic fallback.
# - phase6: each shot costs ~2-3 minutes of provider time.
# - phase8: pairwise assembly writes an intermediate file per transition.
PHASE_POLICIES: Dict[str, PhasePolicy] = {
    "phase1": PhasePolicy(
        phase="phase1",
        artifacts=["director_plan.json", "phase1_events.json", "phase1_characters.json"],
        soft_stall_s=300,
        hard_stall_s=900,
    ),
    "phase2": PhasePolicy(
        phase="phase2",
        artifacts=["storyboard*.png", "shots/*/first_frame.png", "shots/*/last_frame.png"],
        soft_stall_s=600,
        hard_stall_s=1500,
    ),
    "phase3": PhasePolicy(
        phase="phase3",
        artifacts=["characters/*/*.png", "characters/*/*.json"],
        soft_stall_s=600,
        hard_stall_s=1500,
    ),
    "phase4": PhasePolicy(
        phase="phase4",
        artifacts=["shots/S*/SHOT_META.json"],
        soft_stall_s=60,
        hard_stall_s=300,
    ),
    "phase5": PhasePolicy(
        phase="phase5",
        artifacts=["storyboard_qa_report.json", "storyboard_qa_grid.jpg"],
        soft_stall_s=360,
        hard_stall_s=720,
    ),
    "phase6": PhasePolicy(
        phase="phase6",
        artifacts=["shots/S*/output.mp4", "runtime.db"],
        soft_stall_s=900,
        hard_stall_s=1800,
    ),
    "phase7": PhasePolicy(
        phase="phase7",
        artifacts=["consistency_report.json"],
        soft_stall_s=300,
        hard_stall_s=600,
    ),
    "phase8": PhasePolicy(
        phase="phase8",
        artifacts=["raw_assembly.mp4", ".edit_tmp/xfade_*.mp4", "storyboard_order_review.json"],
        soft_stall_s=600,
        hard_stall_s=1200,
    ),
    "phase9": PhasePolicy(
        phase="phase9",
        artifacts=["audio_processed.mp4", "subtitled.mp4", "polished.mp4"],
        soft_stall_s=600,
        hard_stall_s=1200,
    ),
    "phase9_5": PhasePolicy(
        phase="phase9_5",
        artifacts=["video_qa_report.json"],
        soft_stall_s=300,
        hard_stall_s=600,
    ),
}

#: Conservative fallback for phases without a registered policy.
DEFAULT_POLICY = PhasePolicy(phase="unknown", soft_stall_s=600, hard_stall_s=1800)


def get_policy(phase: str, overrides: dict | None = None) -> PhasePolicy:
    """Return the monitoring policy for ``phase``.

    ``overrides`` is an optional mapping of ``phase -> {soft_stall_s,
    hard_stall_s}`` taken from the run config so operators can tighten or
    relax thresholds without editing code. Unknown phases receive a defensive
    copy of :data:`DEFAULT_POLICY` so monitoring never silently disappears.
    """
    base = PHASE_POLICIES.get(phase)
    if base is None:
        base = PhasePolicy(
            phase=phase or DEFAULT_POLICY.phase,
            artifacts=list(DEFAULT_POLICY.artifacts),
            soft_stall_s=DEFAULT_POLICY.soft_stall_s,
            hard_stall_s=DEFAULT_POLICY.hard_stall_s,
        )
    else:
        base = PhasePolicy(
            phase=base.phase,
            artifacts=list(base.artifacts),
            soft_stall_s=base.soft_stall_s,
            hard_stall_s=base.hard_stall_s,
        )

    override = (overrides or {}).get(phase)
    if isinstance(override, dict):
        soft = override.get("soft_stall_s")
        hard = override.get("hard_stall_s")
        if isinstance(soft, int) and not isinstance(soft, bool) and soft > 0:
            base.soft_stall_s = soft
        if isinstance(hard, int) and not isinstance(hard, bool) and hard > 0:
            base.hard_stall_s = hard
        if base.hard_stall_s <= base.soft_stall_s:
            raise ValueError(
                f"monitor override for {phase} must keep hard_stall_s > soft_stall_s"
            )
    return base
