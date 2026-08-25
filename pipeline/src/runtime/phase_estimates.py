"""Runtime-owned phase estimates derived from structured provider workload."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PHASES = (
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
    "phase9_5",
)

# Non-provider phase baselines remain historical. Image phases must override
# these values with their structured request counts before presenting an ETA.
HISTORICAL_SECONDS = {
    "phase1": 200.0,
    "phase2": 63.0,
    "phase3": 460.0,
    "phase4": 1.0,
    "phase5": 10.0,
    "phase6": 1400.0,
    "phase7": 0.5,
    "phase8": 14.0,
    "phase9": 15.0,
    "phase9_5": 15.0,
}
PER_VIDEO_SHOT_SECONDS = 140.0
DEFAULT_SEEDREAM_INTERVAL_SECONDS = 120.0
DEFAULT_PHASE5_CORRECTION_ATTEMPTS = 2
MAX_PHASE5_CORRECTION_ATTEMPTS = 3


@dataclass(frozen=True)
class PipelineWorkload:
    """JSON-safe request counts derived only from canonical Phase 1 artifacts."""

    schema: str
    character_count: int
    shot_count: int
    storyboard_beat_count: int
    character_reference_image_requests: int
    phase2_image_requests: int
    phase3_image_requests: int
    phase4_image_requests: int
    phase5_max_correction_image_requests: int
    phase5_correction_attempts: int

    def to_dict(self) -> dict:
        return asdict(self)


def seedream_request_interval_seconds() -> float:
    """Return the same configured request-start interval used by Seedream."""
    raw = os.environ.get(
        "SEEDREAM_MIN_INTERVAL",
        str(DEFAULT_SEEDREAM_INTERVAL_SECONDS),
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SEEDREAM_INTERVAL_SECONDS


def _phase5_correction_attempts(value: int | None) -> int:
    raw = (
        os.environ.get(
            "HONCUT_PHASE5_MAX_CORRECTIONS",
            str(DEFAULT_PHASE5_CORRECTION_ATTEMPTS),
        )
        if value is None
        else value
    )
    try:
        attempts = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 5 correction attempts must be an integer") from exc
    if not 0 <= attempts <= MAX_PHASE5_CORRECTION_ATTEMPTS:
        raise ValueError(
            "Phase 5 correction attempts must be between 0 and "
            f"{MAX_PHASE5_CORRECTION_ATTEMPTS}"
        )
    return attempts


def _dict_records(value: object, *, field: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} entries must be objects")
    return list(value)


def _character_id(character: dict, index: int) -> str:
    return str(character.get("id") or f"char_{index}").strip()


def _existing_character_reference_count(
    output_dir: Path | None,
    character_id: str,
) -> int:
    if output_dir is None:
        return 0
    character_dir = output_dir / "characters" / character_id
    return sum(
        (character_dir / f"{name}.png").is_file()
        for name in ("face_closeup", "full_body", "side", "back")
    )


def _character_references_ready(
    output_dir: Path | None,
    characters_by_id: dict[str, dict],
    character_ids: list[str],
) -> bool:
    if output_dir is None:
        return False
    return all(
        identity in characters_by_id
        and _existing_character_reference_count(output_dir, identity) >= 2
        for identity in character_ids
    )


def build_pipeline_workload(
    characters_data: dict,
    storyboard_data: dict,
    *,
    output_dir: str | Path | None = None,
    phase5_correction_attempts: int | None = None,
) -> PipelineWorkload:
    """Count baseline image submissions without interpreting narrative prose."""
    if not isinstance(characters_data, dict) or not isinstance(storyboard_data, dict):
        raise ValueError("structured workload inputs must be objects")
    characters = _dict_records(
        characters_data.get("characters"),
        field="CHARACTERS.characters",
    )
    shots = _dict_records(storyboard_data.get("shots"), field="STORYBOARD.shots")
    root = Path(output_dir) if output_dir is not None else None

    characters_by_id: dict[str, dict] = {}
    character_reference_requests = 0
    for index, character in enumerate(characters):
        identity = _character_id(character, index)
        if not identity or identity in characters_by_id:
            raise ValueError(f"empty or duplicate character id: {identity!r}")
        characters_by_id[identity] = character
        appearance = character.get("appearance")
        appearance = appearance if isinstance(appearance, dict) else {}

        existing_views = _existing_character_reference_count(root, identity)
        if existing_views < 4:
            character_reference_requests += 4

        identity_props = appearance.get("identity_props") or []
        if not isinstance(identity_props, list):
            raise ValueError(f"character {identity} appearance.identity_props must be an array")
        identity_detail = root / "characters" / identity / "identity_detail.png" if root else None
        if identity_props and not (identity_detail and identity_detail.is_file()):
            character_reference_requests += 1

        variants = appearance.get("variants") or []
        if not isinstance(variants, list):
            raise ValueError(f"character {identity} appearance.variants must be an array")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError(f"character {identity} variant must be an object")
            description = str(variant.get("description") or "").strip()
            if not description:
                continue
            state_name = str(variant.get("state_name") or "unknown")
            variant_path = (
                root / "characters" / identity / f"variant_{state_name}.png"
                if root
                else None
            )
            if not (variant_path and variant_path.is_file()):
                character_reference_requests += 1

    beat_count = 0
    has_deferred_character_shot = False
    for shot in shots:
        beats = _dict_records(
            shot.get("storyboard_beats"),
            field="STORYBOARD.shots[].storyboard_beats",
        )
        beat_count += len(beats)
        raw_character_ids = shot.get("character_ids") or shot.get("who") or []
        if isinstance(raw_character_ids, str):
            raw_character_ids = [raw_character_ids]
        if not isinstance(raw_character_ids, list):
            raise ValueError("STORYBOARD shot character_ids/who must be an array")
        character_ids = [
            str(value).strip() for value in raw_character_ids if str(value).strip()
        ]
        if character_ids and not _character_references_ready(
            root,
            characters_by_id,
            character_ids,
        ):
            has_deferred_character_shot = True

    storyboard_requests = beat_count + len(shots)
    attempts = _phase5_correction_attempts(phase5_correction_attempts)
    return PipelineWorkload(
        schema="honcut.pipeline-workload-estimate.v1",
        character_count=len(characters),
        shot_count=len(shots),
        storyboard_beat_count=beat_count,
        character_reference_image_requests=character_reference_requests,
        # Phase 2 defers the complete Pxx set as soon as any character-locked
        # shot lacks its canonical reference pack.
        phase2_image_requests=(0 if has_deferred_character_shot else storyboard_requests),
        # Phase 3 owns missing character packs and the canonical character-
        # locked Pxx refresh. Cache hits may make the real count lower.
        phase3_image_requests=character_reference_requests + storyboard_requests,
        phase4_image_requests=beat_count,
        # Each bounded correction can redraw every Pxx plus one Sxx board per
        # affected shot. It never submits video work.
        phase5_max_correction_image_requests=attempts * storyboard_requests,
        phase5_correction_attempts=attempts,
    )


def load_pipeline_workload(
    output_dir: str | Path,
    *,
    phase5_correction_attempts: int | None = None,
) -> PipelineWorkload | None:
    """Load a structured estimate only when both canonical artifacts exist."""
    root = Path(output_dir)
    characters_path = root / "CHARACTERS.json"
    storyboard_path = root / "STORYBOARD.json"
    if not characters_path.is_file() or not storyboard_path.is_file():
        return None
    characters = json.loads(characters_path.read_text(encoding="utf-8"))
    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    return build_pipeline_workload(
        characters,
        storyboard,
        output_dir=root,
        phase5_correction_attempts=phase5_correction_attempts,
    )


def estimate_phase_duration(
    phase: str,
    num_characters: int = 3,
    num_shots: int = 10,
    *,
    image_requests: int | None = None,
) -> float:
    """Estimate one phase, preferring provider request counts over prose size."""
    if image_requests is not None:
        if image_requests < 0:
            raise ValueError("image_requests must be non-negative")
        return image_requests * seedream_request_interval_seconds()
    if phase == "phase3":
        return num_characters * 150.0
    if phase == "phase6":
        return num_shots * PER_VIDEO_SHOT_SECONDS
    return HISTORICAL_SECONDS.get(phase, 10.0)


def estimate_total(
    num_characters: int = 3,
    num_shots: int = 10,
    *,
    phases: Iterable[str] | None = None,
    workload: PipelineWorkload | None = None,
) -> dict:
    """Estimate selected phases and expose the bounded Phase 5 redraw range."""
    selected = list(phases or PHASES)
    unknown = [phase for phase in selected if phase not in PHASES]
    if unknown:
        raise ValueError(f"unknown phases: {unknown}")
    if workload is not None:
        num_characters = workload.character_count
        num_shots = workload.shot_count

    image_requests_by_phase = (
        {
            "phase2": workload.phase2_image_requests,
            "phase3": workload.phase3_image_requests,
            "phase4": workload.phase4_image_requests,
        }
        if workload is not None
        else {}
    )
    estimates: dict[str, float] = {}
    for phase in selected:
        estimates[phase] = estimate_phase_duration(
            phase,
            num_characters,
            num_shots,
            image_requests=image_requests_by_phase.get(phase),
        )

    total = sum(estimates.values())
    correction_seconds = (
        workload.phase5_max_correction_image_requests
        * seedream_request_interval_seconds()
        if workload is not None and "phase5" in selected
        else 0.0
    )
    upper_total = total + correction_seconds
    return {
        "schema": "honcut.pipeline-duration-estimate.v2",
        "basis": (
            "structured_provider_workload"
            if workload is not None
            else "historical_provisional"
        ),
        "phases": estimates,
        "total": total,
        "upper_total": upper_total,
        "bounded": upper_total > total,
        "total_human": format_duration(total),
        "upper_total_human": format_duration(upper_total),
        "workload": workload.to_dict() if workload is not None else None,
    }


def remaining_phase_names(
    *,
    skip_phase: Iterable[int | float],
    completed_phases: Iterable[str],
) -> list[str]:
    """Resolve the exact uncompleted CLI-selected phases for ETA reporting."""
    skipped = {float(value) for value in skip_phase}
    completed = set(completed_phases)
    result = []
    for phase in PHASES:
        number = 9.5 if phase == "phase9_5" else float(phase.removeprefix("phase"))
        if number not in skipped and phase not in completed:
            result.append(phase)
    return result


def estimate_remaining(
    current_phase: str,
    elapsed_in_phase: float,
    num_characters: int = 3,
    num_shots: int = 10,
) -> dict:
    """Compatibility estimate from a current phase through delivery QA."""
    try:
        index = PHASES.index(current_phase)
    except ValueError:
        return {"remaining_s": 0, "remaining_human": "unknown"}
    remaining = max(
        0.0,
        estimate_phase_duration(current_phase, num_characters, num_shots)
        - elapsed_in_phase,
    )
    remaining += sum(
        estimate_phase_duration(phase, num_characters, num_shots)
        for phase in PHASES[index + 1 :]
    )
    return {
        "remaining_s": remaining,
        "remaining_human": format_duration(remaining),
    }


def format_duration(seconds: float) -> str:
    """Format seconds as a compact Chinese duration."""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}分{remainder}秒"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}小时{minutes}分"


__all__ = [
    "PipelineWorkload",
    "build_pipeline_workload",
    "estimate_phase_duration",
    "estimate_remaining",
    "estimate_total",
    "format_duration",
    "load_pipeline_workload",
    "remaining_phase_names",
    "seedream_request_interval_seconds",
]
