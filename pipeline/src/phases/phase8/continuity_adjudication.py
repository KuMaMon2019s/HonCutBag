"""Phase 8 adjudication for temporal rollback at native-extension seams.

Phase 6 can only make a provisional seam because it must finish generation before
the complete motion trajectory is available.  This module revisits each planned
internal boundary, looks one second beyond the planned replay budget, and emits
frame-accurate hard-trim decisions for Phase 6 to consume on its next pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from quality.continuity_bridge import detect_replayed_prefix
from runtime.continuity_chunks import load_continuity_plan
from runtime.continuity_provider import probe_continuity_frames

ADJUDICATION_KIND = "honcut.continuity_adjudication.v1"
SEAM_DECISIONS_KIND = "honcut.continuity_seam_decisions.v1"
TOPUP_REQUESTS_KIND = "honcut.continuity_topup_requests.v1"
OBJECT_TRAJECTORIES_KIND = "honcut.continuity_object_trajectories.v1"
TEMPORAL_REVIEW_KIND = "honcut.continuity_temporal_review.v1"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def decide_temporal_seam(
    candidates: list[dict[str, Any]],
    *,
    planned_overlap_frames: int,
    timeline_fps: int,
    max_extra_search_frames: int | None = None,
    object_trajectory_evidence: dict[str, Any] | None = None,
    human_approved: bool = False,
) -> dict[str, Any]:
    """Classify a late replay/rollback from an overlap-error trajectory.

    A rollback is not a single ugly frame.  It is the stronger signal that,
    after the planned cut point, the following prefix keeps converging toward
    the predecessor's tail.  Requiring both a sustained trend and a meaningful
    improvement avoids treating ordinary boundary noise as replay.
    """
    if timeline_fps <= 0:
        raise ValueError("timeline_fps must be positive")
    if planned_overlap_frames <= 0:
        return {
            "action": "keep",
            "rollback_detected": False,
            "reason": "the boundary has no planned replay budget",
        }
    extra_limit = max_extra_search_frames or timeline_fps
    upper = planned_overlap_frames + extra_limit
    window = [
        item
        for item in candidates
        if planned_overlap_frames
        <= round(float(item["seconds"]) * timeline_fps)
        <= upper
    ]
    if len(window) < 4:
        return {
            "action": "keep",
            "rollback_detected": False,
            "reason": "too few post-budget samples for a temporal decision",
            "sample_count": len(window),
        }

    baseline = min(
        window,
        key=lambda item: abs(
            round(float(item["seconds"]) * timeline_fps) - planned_overlap_frames
        ),
    )
    best = min(window, key=lambda item: float(item["frame_mae"]))
    baseline_mae = float(baseline["frame_mae"])
    best_mae = float(best["frame_mae"])
    improvement = baseline_mae - best_mae
    relative_improvement = improvement / max(baseline_mae, 1e-9)
    decreases = sum(
        float(following["frame_mae"]) < float(previous["frame_mae"])
        for previous, following in pairwise(window)
    )
    decreasing_ratio = decreases / max(1, len(window) - 1)
    best_index = window.index(best)
    best_position = best_index / max(1, len(window) - 1)
    selected_trim_frames = round(float(best["seconds"]) * timeline_fps)
    additional_trim_frames = selected_trim_frames - planned_overlap_frames

    rollback = bool(
        additional_trim_frames >= max(3, math.ceil(timeline_fps * 0.125))
        and improvement >= max(0.003, baseline_mae * 0.08)
        and decreasing_ratio >= 0.65
        and best_position >= 0.75
    )
    evidence = {
        "sample_count": len(window),
        "planned_overlap_frames": planned_overlap_frames,
        "baseline_frame_mae": round(baseline_mae, 6),
        "best_frame_mae": round(best_mae, 6),
        "absolute_improvement": round(improvement, 6),
        "relative_improvement": round(relative_improvement, 6),
        "decreasing_ratio": round(decreasing_ratio, 6),
        "best_position": round(best_position, 6),
    }
    object_evidence = object_trajectory_evidence or {}
    object_confidence = float(object_evidence.get("confidence") or 0.0)
    object_verdict = str(object_evidence.get("verdict") or "unavailable")
    object_supports = object_verdict == "rollback" and object_confidence >= 0.6
    object_contradicts = object_verdict in {"continuous", "forward"} and object_confidence >= 0.6
    if not rollback:
        if object_supports:
            tracked_trim = int(object_evidence.get("recommended_trim_frames") or 0)
            if object_evidence.get("repair_action") == "hard_trim" and tracked_trim > 0:
                selected = max(planned_overlap_frames, tracked_trim)
                return {
                    "action": "hard_trim",
                    "rollback_detected": True,
                    "reason": (
                        "subject tracking detects rollback despite a background-dominated "
                        "appearance trajectory"
                    ),
                    "trim_frames": selected,
                    "trim_seconds": round(selected / timeline_fps, 6),
                    "additional_trim_frames": selected - planned_overlap_frames,
                    "frame_policy": "do_not_interpolate",
                    "confidence": "object_trajectory_override",
                    "trim_source": "object_trajectory_catchup",
                    "evidence": evidence,
                    "object_trajectory_evidence": object_evidence,
                }
            return {
                "action": "human_review",
                "recommended_action": "regenerate",
                "rollback_detected": True,
                "reason": (
                    "subject tracking detects rollback but no safe catch-up frame exists"
                ),
                "confidence": "tracked_rollback_without_safe_cut",
                "evidence": evidence,
                "object_trajectory_evidence": object_evidence,
            }
        return {
            "action": "keep",
            "rollback_detected": False,
            "reason": "post-budget samples do not show sustained temporal rollback",
            "evidence": evidence,
        }
    proposed = {
        "rollback_detected": True,
        "reason": (
            "the following prefix keeps converging toward the previous tail "
            "after the planned cut point"
        ),
        "trim_frames": selected_trim_frames,
        "trim_seconds": round(selected_trim_frames / timeline_fps, 6),
        "additional_trim_frames": additional_trim_frames,
        "frame_policy": "do_not_interpolate",
        "evidence": evidence,
    }
    if human_approved:
        return {
            **proposed,
            "action": "hard_trim",
            "confidence": "human_confirmed",
            "corroboration": "exact-source human review",
        }
    if object_supports:
        tracked_trim = int(object_evidence.get("recommended_trim_frames") or 0)
        if object_evidence.get("repair_action") != "hard_trim" or tracked_trim <= 0:
            return {
                **proposed,
                "action": "human_review",
                "recommended_action": "regenerate",
                "confidence": "tracked_rollback_without_safe_cut",
                "corroboration": (
                    "object tracking confirms rollback but the subject never reaches "
                    "a safe continuation point"
                ),
                "object_trajectory_evidence": object_evidence,
            }
        proposed = {
            **proposed,
            "trim_frames": tracked_trim,
            "trim_seconds": round(tracked_trim / timeline_fps, 6),
            "additional_trim_frames": tracked_trim - planned_overlap_frames,
            "trim_source": "object_trajectory_catchup",
        }
        return {
            **proposed,
            "action": "hard_trim",
            "confidence": "high",
            "corroboration": "object trajectory agrees with appearance trajectory",
            "object_trajectory_evidence": object_evidence,
        }
    return {
        **proposed,
        "action": "human_review",
        "recommended_action": "hard_trim",
        "confidence": "conflicted" if object_contradicts else "appearance_only",
        "corroboration": (
            "object trajectory contradicts the appearance trajectory"
            if object_contradicts
            else "object trajectory evidence is unavailable or below confidence threshold"
        ),
        "object_trajectory_evidence": object_evidence or None,
    }


def _load_boundary_map(root: Path, filename: str, kind: str) -> dict[str, dict[str, Any]]:
    path = root / filename
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != kind:
        raise ValueError(f"unsupported continuity evidence in {path}")
    values = document.get("boundaries")
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a boundaries object")
    return {
        str(boundary_id): value
        for boundary_id, value in values.items()
        if isinstance(value, dict)
    }


def _object_evidence_needs_retry(evidence: dict[str, Any] | None) -> bool:
    """Retry missing/transient SAM3 evidence, not deterministic same-source misses."""
    if evidence is None:
        return True
    if str(evidence.get("verdict") or "") != "unavailable":
        return False
    return str(evidence.get("reason") or "").startswith(
        "SAM 3 trajectory analysis failed:"
    )


def adjudicate_continuity_seams(
    output_dir: str | Path,
    *,
    detector: Callable[..., dict[str, Any]] = detect_replayed_prefix,
    frame_probe: Callable[[Path, int], dict[str, Any]] = probe_continuity_frames,
    sam3_collector: Callable[..., dict[str, Any]] | None = None,
    sam3_base_url: str | None = None,
) -> dict[str, Any]:
    """Analyze internal and cross-shot extension seams and persist edit instructions."""
    root = Path(output_dir)
    plan_path = root / "CONTINUITY_PLAN.json"
    if not plan_path.is_file():
        return {"status": "skipped", "reason": "CONTINUITY_PLAN.json not found"}
    plan = load_continuity_plan(plan_path)
    planned_shots = [
        shot for shot in plan.shots if len(shot.chunks) > 1 or shot.extends_from_chunk_id
    ]
    if not planned_shots:
        return {"status": "skipped", "reason": "no native-extension continuity seams"}
    chunk_owners = {
        chunk.chunk_id: shot.shot_id
        for shot in plan.shots
        for chunk in shot.chunks
    }
    chunks_by_id = {
        chunk.chunk_id: chunk
        for shot in plan.shots
        for chunk in shot.chunks
    }

    object_trajectories = _load_boundary_map(
        root,
        "CONTINUITY_OBJECT_TRAJECTORIES.json",
        OBJECT_TRAJECTORIES_KIND,
    )
    temporal_reviews = _load_boundary_map(
        root,
        "CONTINUITY_TEMPORAL_REVIEW.json",
        TEMPORAL_REVIEW_KIND,
    )
    decisions: dict[str, dict[str, Any]] = {}
    shot_reports: list[dict[str, Any]] = []
    topup_requests: list[dict[str, Any]] = []
    analyzed_at = datetime.now(UTC).isoformat()
    sam3_url = (
        os.environ.get("HONCUT_SAM3_URL", "").strip()
        if sam3_base_url is None
        else sam3_base_url.strip()
    )
    sam3_attempted = False

    for shot in planned_shots:
        timing_path = root / "shots" / shot.shot_id / "CONTINUITY_TIMING.json"
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.is_file()
            else {}
        )
        timing_rows = {
            str(row.get("chunk_id")): row for row in timing.get("chunks", [])
        }
        boundaries: list[dict[str, Any]] = []
        applied_delta_frames = 0

        seam_pairs: list[tuple[Any, Any]] = []
        if shot.extends_from_chunk_id:
            predecessor = chunks_by_id.get(shot.extends_from_chunk_id)
            if predecessor is None:
                raise RuntimeError(
                    f"{shot.shot_id} references missing predecessor "
                    f"{shot.extends_from_chunk_id}"
                )
            seam_pairs.append((predecessor, shot.chunks[0]))
        seam_pairs.extend(pairwise(shot.chunks))

        for previous_chunk, following_chunk in seam_pairs:
            previous_shot_id = chunk_owners[previous_chunk.chunk_id]
            following_shot_id = chunk_owners[following_chunk.chunk_id]
            previous_path = (
                root
                / "shots"
                / previous_shot_id
                / "chunks"
                / f"{previous_chunk.chunk_id}.mp4"
            )
            following_path = (
                root
                / "shots"
                / following_shot_id
                / "chunks"
                / f"{following_chunk.chunk_id}.mp4"
            )
            boundary_id = f"{previous_chunk.chunk_id}__{following_chunk.chunk_id}"
            if not previous_path.is_file() or not following_path.is_file():
                boundaries.append(
                    {
                        "boundary_id": boundary_id,
                        "action": "unavailable",
                        "reason": "one or both provider chunks are missing",
                    }
                )
                continue

            planned_frames = int(following_chunk.expected_overlap_frames)
            following_probe = frame_probe(following_path, plan.timeline_fps)
            leave_frames = max(12, math.ceil(plan.timeline_fps * 0.5))
            searchable_frames = max(
                planned_frames,
                min(
                    planned_frames + plan.timeline_fps,
                    int(following_probe["frames"]) - leave_frames,
                ),
            )
            search_seconds = searchable_frames / plan.timeline_fps
            trajectory = detector(
                previous_path,
                following_path,
                search_seconds=search_seconds,
                sample_fps=max(4, min(12, plan.timeline_fps)),
            )
            source_fingerprint = {
                "previous_sha256": _sha256(previous_path),
                "following_sha256": _sha256(following_path),
            }
            object_evidence = object_trajectories.get(boundary_id)
            if (
                object_evidence is not None
                and object_evidence.get("source_fingerprint") != source_fingerprint
            ):
                object_evidence = None
            review = temporal_reviews.get(boundary_id) or {}
            exact_source_review = bool(
                review.get("source_fingerprint") == source_fingerprint
                and review.get("action") == "hard_trim"
                and int(review.get("approved_trim_frames") or 0) > 0
            )
            tracking_prompt = shot.anchors.tracking_prompt.strip()
            candidates = list(trajectory.get("candidates", []))
            preliminary = decide_temporal_seam(
                candidates,
                planned_overlap_frames=planned_frames,
                timeline_fps=plan.timeline_fps,
                max_extra_search_frames=max(0, searchable_frames - planned_frames),
                object_trajectory_evidence=object_evidence,
            )
            cross_shot_boundary = previous_shot_id != following_shot_id
            needs_object_corroboration = bool(
                preliminary.get("action") == "human_review"
                or (cross_shot_boundary and planned_frames > 0)
            )
            object_evidence_retriable = _object_evidence_needs_retry(object_evidence)
            if (
                needs_object_corroboration
                and object_evidence_retriable
                and not exact_source_review
                and sam3_url
                and tracking_prompt
            ):
                sam3_attempted = True
                try:
                    if sam3_collector is None:
                        from quality.object_trajectory import collect_sam3_trajectory

                        collector = collect_sam3_trajectory
                    else:
                        collector = sam3_collector
                    object_evidence = collector(
                        previous_path,
                        following_path,
                        boundary_id=boundary_id,
                        evidence_dir=root / "continuity_tracking",
                        prompt=tracking_prompt,
                        timeline_fps=plan.timeline_fps,
                        planned_overlap_frames=planned_frames,
                        following_frames=int(following_probe["frames"]),
                        screen_direction=shot.anchors.screen_direction,
                        camera_motion=shot.anchors.camera_motion,
                        base_url=sam3_url,
                    )
                except Exception as exc:
                    object_evidence = {
                        "verdict": "unavailable",
                        "confidence": 0.0,
                        "reason": f"SAM 3 trajectory analysis failed: {exc}",
                    }
                object_evidence = {
                    **object_evidence,
                    "source_fingerprint": source_fingerprint,
                }
                object_trajectories[boundary_id] = object_evidence
                preliminary = decide_temporal_seam(
                    candidates,
                    planned_overlap_frames=planned_frames,
                    timeline_fps=plan.timeline_fps,
                    max_extra_search_frames=max(0, searchable_frames - planned_frames),
                    object_trajectory_evidence=object_evidence,
                )
            human_approved = bool(
                review.get("action") == "hard_trim"
                and review.get("source_fingerprint") == source_fingerprint
                and int(review.get("approved_trim_frames") or 0)
                == int(preliminary.get("trim_frames") or 0)
            )
            human_selected_planned_trim = bool(
                review.get("action") == "hard_trim"
                and review.get("source_fingerprint") == source_fingerprint
                and int(review.get("approved_trim_frames") or 0) == planned_frames
                and preliminary.get("rollback_detected")
                and int(preliminary.get("trim_frames") or 0) > planned_frames
            )
            decision = (
                {
                    **preliminary,
                    "action": "hard_trim",
                    "confidence": "human_confirmed",
                    "corroboration": "exact-source, exact-frame human review",
                }
                if human_approved and preliminary.get("rollback_detected")
                else {
                    **preliminary,
                    "action": "hard_trim",
                    "rollback_detected": False,
                    "trim_frames": planned_frames,
                    "trim_seconds": round(planned_frames / plan.timeline_fps, 6),
                    "additional_trim_frames": 0,
                    "frame_policy": "do_not_interpolate",
                    "confidence": "human_confirmed_planned_overlap",
                    "corroboration": (
                        "exact-source human review rejected the appearance-only "
                        "additional trim"
                    ),
                    "reason": (
                        "remove only the intentional native-extension replay prefix; "
                        "the apparent late rollback is background-dominated"
                    ),
                    "appearance_only_proposal": preliminary,
                }
                if human_selected_planned_trim
                else preliminary
            )
            current_row = timing_rows.get(following_chunk.chunk_id, {})
            currently_trimmed = int(
                current_row.get("detected_overlap_frames", 0) or 0
            )
            if decision.get("action") == "keep" and planned_frames > currently_trimmed:
                decision = {
                    **decision,
                    "action": "hard_trim",
                    "rollback_detected": False,
                    "trim_frames": planned_frames,
                    "trim_seconds": round(planned_frames / plan.timeline_fps, 6),
                    "additional_trim_frames": 0,
                    "frame_policy": "do_not_interpolate",
                    "confidence": "planned_overlap",
                    "reason": "remove the intentionally generated native-extension replay prefix",
                }
            selected_trim = (
                int(decision.get("trim_frames", currently_trimmed))
                if decision.get("action") == "hard_trim"
                else currently_trimmed
            )
            newly_applied = max(0, selected_trim - currently_trimmed)
            applied_delta_frames += newly_applied
            boundary_report = {
                "boundary_id": boundary_id,
                "previous_chunk_id": previous_chunk.chunk_id,
                "following_chunk_id": following_chunk.chunk_id,
                "planned_overlap_frames": planned_frames,
                "currently_trimmed_frames": currently_trimmed,
                "newly_required_trim_frames": newly_applied,
                "source_fingerprint": source_fingerprint,
                **decision,
            }
            boundaries.append(boundary_report)
            if decision.get("action") == "hard_trim":
                decisions[boundary_id] = {
                    "action": "hard_trim",
                    "shot_id": shot.shot_id,
                    "boundary_kind": (
                        "cross_shot" if previous_shot_id != following_shot_id else "internal"
                    ),
                    "previous_shot_id": previous_shot_id,
                    "following_shot_id": following_shot_id,
                    "previous_chunk_id": previous_chunk.chunk_id,
                    "following_chunk_id": following_chunk.chunk_id,
                    "trim_frames": int(decision["trim_frames"]),
                    "trim_seconds": float(decision["trim_seconds"]),
                    "frame_policy": "do_not_interpolate",
                    "reason": decision["reason"],
                    "source_fingerprint": boundary_report["source_fingerprint"],
                    "decided_at": analyzed_at,
                }

        current_materialized = int(
            timing.get(
                "materialized_frames_before_closure",
                sum(int(row.get("effective_unique_frames", 0)) for row in timing_rows.values()),
            )
            or 0
        )
        projected_frames = max(0, current_materialized - applied_delta_frames)
        target_frames = int(
            shot.target_frames or round(shot.target_duration_s * plan.timeline_fps)
        )
        deficit_frames = max(0, target_frames - projected_frames)
        closure_limit = max(2, math.ceil(target_frames * 0.02))
        requires_topup = deficit_frames > closure_limit
        if requires_topup:
            topup_requests.append(
                {
                    "shot_id": shot.shot_id,
                    "reason": "Phase 8 temporal trim creates a material duration deficit",
                    "target_frames": target_frames,
                    "projected_frames": projected_frames,
                    "deficit_frames": deficit_frames,
                    "minimum_requested_unique_frames": deficit_frames,
                    "resume_phase": "phase6",
                }
            )
        shot_reports.append(
            {
                "shot_id": shot.shot_id,
                "target_frames": target_frames,
                "current_materialized_frames": current_materialized,
                "newly_required_trim_frames": applied_delta_frames,
                "projected_materialized_frames": projected_frames,
                "deficit_frames": deficit_frames,
                "bounded_duration_closure_limit_frames": closure_limit,
                "requires_continuation_topup": requires_topup,
                "boundaries": boundaries,
            }
        )

    decisions_document = {
        "kind": SEAM_DECISIONS_KIND,
        "timeline_fps": plan.timeline_fps,
        "decisions": decisions,
        "updated_at": analyzed_at,
    }
    topup_document = {
        "kind": TOPUP_REQUESTS_KIND,
        "status": "required" if topup_requests else "resolved",
        "requests": topup_requests,
        "updated_at": analyzed_at,
    }
    if object_trajectories or sam3_attempted:
        _atomic_write_json(
            root / "CONTINUITY_OBJECT_TRAJECTORIES.json",
            {
                "kind": OBJECT_TRAJECTORIES_KIND,
                "boundaries": object_trajectories,
                "updated_at": analyzed_at,
            },
        )
    report = {
        "kind": ADJUDICATION_KIND,
        "status": (
            "human_review_required"
            if any(
                boundary.get("action") == "human_review"
                for shot in shot_reports
                for boundary in shot["boundaries"]
            )
            else "topup_required" if topup_requests else "passed"
        ),
        "timeline_fps": plan.timeline_fps,
        "shot_count": len(shot_reports),
        "hard_trim_count": len(decisions),
        "requires_human_review": any(
            boundary.get("action") == "human_review"
            for shot in shot_reports
            for boundary in shot["boundaries"]
        ),
        "requires_phase6": bool(topup_requests),
        "shots": shot_reports,
        "outputs": [
            "CONTINUITY_ADJUDICATION.json",
            "CONTINUITY_SEAM_DECISIONS.json",
            "CONTINUITY_TOPUP_REQUESTS.json",
            *(
                ["CONTINUITY_OBJECT_TRAJECTORIES.json"]
                if object_trajectories or sam3_attempted
                else []
            ),
        ],
        "updated_at": analyzed_at,
    }
    _atomic_write_json(root / "CONTINUITY_SEAM_DECISIONS.json", decisions_document)
    _atomic_write_json(root / "CONTINUITY_TOPUP_REQUESTS.json", topup_document)
    _atomic_write_json(root / "CONTINUITY_ADJUDICATION.json", report)
    return report


__all__ = [
    "ADJUDICATION_KIND",
    "OBJECT_TRAJECTORIES_KIND",
    "SEAM_DECISIONS_KIND",
    "TEMPORAL_REVIEW_KIND",
    "TOPUP_REQUESTS_KIND",
    "adjudicate_continuity_seams",
    "decide_temporal_seam",
]
