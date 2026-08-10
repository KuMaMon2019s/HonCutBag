"""Pipeline timing estimator based on historical run data."""

# Historical phase durations (seconds) from completed runs
HISTORICAL_DATA = {
    "phase1": {"avg": 200, "min": 120, "max": 310, "runs": 3},
    "phase2": {"avg": 63, "min": 58, "max": 68, "runs": 3},
    "phase3": {"avg": 460, "min": 180, "max": 706, "runs": 3},  # varies by char count
    "phase4": {"avg": 1, "min": 1, "max": 1, "runs": 3},
    "phase5": {"avg": 10, "min": 1, "max": 30, "runs": 0},
    "phase6": {"avg": 1400, "min": 1349, "max": 1482, "runs": 2},  # varies by shot count
    "phase7": {"avg": 0.5, "min": 0, "max": 1, "runs": 3},
    "phase8": {"avg": 14, "min": 13, "max": 14, "runs": 3},
    "phase9": {"avg": 15, "min": 2, "max": 28, "runs": 3},
}

# Per-unit costs for scaling
PER_CHARACTER_COST = 150  # seconds per character in Phase 3
PER_SHOT_COST = 140       # seconds per shot in Phase 6


def estimate_phase_duration(phase: str, num_characters: int = 3, num_shots: int = 10) -> float:
    """Estimate duration for a phase based on historical data and workload."""
    if phase == "phase3":
        return num_characters * PER_CHARACTER_COST
    elif phase == "phase6":
        return num_shots * PER_SHOT_COST
    else:
        data = HISTORICAL_DATA.get(phase, {})
        return data.get("avg", 10)


def estimate_total(num_characters: int = 3, num_shots: int = 10) -> dict:
    """Estimate total pipeline duration with per-phase breakdown."""
    phases = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9"]
    estimates = {}
    total = 0
    for p in phases:
        est = estimate_phase_duration(p, num_characters, num_shots)
        estimates[p] = est
        total += est
    return {"phases": estimates, "total": total, "total_human": _format_duration(total)}


def estimate_remaining(current_phase: str, elapsed_in_phase: float,
                       num_characters: int = 3, num_shots: int = 10) -> dict:
    """Estimate remaining time from current position."""
    phases = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9"]
    try:
        idx = phases.index(current_phase)
    except ValueError:
        return {"remaining_s": 0, "remaining_human": "unknown"}

    remaining = 0
    # Remaining time in current phase
    current_est = estimate_phase_duration(current_phase, num_characters, num_shots)
    remaining += max(0, current_est - elapsed_in_phase)

    # Full remaining phases
    for p in phases[idx + 1:]:
        remaining += estimate_phase_duration(p, num_characters, num_shots)

    return {"remaining_s": remaining, "remaining_human": _format_duration(remaining)}


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}分{s}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}小时{m}分"
