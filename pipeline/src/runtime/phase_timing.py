"""Console timing helpers shared by concrete phase runners."""

import time
from pathlib import Path


def now() -> float:
    return time.time()


def elapsed(start: float) -> float:
    return round(now() - start, 2)


def print_phase_banner(
    phase_num: int | float | str,
    total: int,
    name: str,
    dry_run: bool = False,
) -> None:
    tag = " [DRY-RUN]" if dry_run else ""
    print(f"\n{'=' * 60}")
    print(f"  [Phase {phase_num}/{total}] {name}{tag}")
    print(f"{'=' * 60}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_banner = print_phase_banner
_elapsed = elapsed
_ensure_dir = ensure_dir
_now = now


__all__ = [
    "_banner",
    "_elapsed",
    "_ensure_dir",
    "_now",
    "elapsed",
    "ensure_dir",
    "now",
    "print_phase_banner",
]
