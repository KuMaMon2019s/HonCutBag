"""Optional SAM 3 object-trajectory evidence for continuity seam adjudication."""

from __future__ import annotations

import base64
import io
import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

OBJECT_TRAJECTORIES_KIND = "honcut.continuity_object_trajectories.v1"


def decide_object_trajectory(
    frames: list[dict[str, Any]],
    *,
    seam_frame: int,
    planned_overlap_frames: int,
    screen_direction: str = "",
    camera_motion: str = "",
) -> dict[str, Any]:
    """Decide whether a tracked subject jumps backward at a chunk boundary."""
    usable = [
        frame
        for frame in frames
        if isinstance(frame.get("centroid"), (list, tuple))
        and len(frame["centroid"]) == 2
    ]
    before = [frame for frame in usable if int(frame["frame_idx"]) < seam_frame]
    after = [frame for frame in usable if int(frame["frame_idx"]) >= seam_frame]
    if len(before) < 3 or len(after) < 3:
        return {
            "verdict": "unavailable",
            "confidence": 0.0,
            "reason": "insufficient tracked subject frames across the boundary",
        }

    direction_key = screen_direction.strip().lower().replace("-", "_")
    explicit_directions = {
        "left_to_right": np.asarray([1.0, 0.0]),
        "right_to_left": np.asarray([-1.0, 0.0]),
        "top_to_bottom": np.asarray([0.0, 1.0]),
        "bottom_to_top": np.asarray([0.0, -1.0]),
    }
    direction = explicit_directions.get(direction_key)
    if direction is None:
        tail_points = np.asarray([frame["centroid"] for frame in before[-6:]], dtype=float)
        displacement = np.median(np.diff(tail_points, axis=0), axis=0)
        norm = float(np.linalg.norm(displacement))
        if norm < 0.002:
            return {
                "verdict": "unavailable",
                "confidence": 0.0,
                "reason": "tracked subject has no reliable pre-boundary direction",
            }
        direction = displacement / norm

    tail = np.median(
        np.asarray([frame["centroid"] for frame in before[-3:]], dtype=float),
        axis=0,
    )
    planned_diagnostic_frame = seam_frame + planned_overlap_frames
    planned = min(
        after,
        key=lambda frame: abs(int(frame["frame_idx"]) - planned_diagnostic_frame),
    )
    planned_position = np.asarray(planned["centroid"], dtype=float)
    signed_gap = float(np.dot(planned_position - tail, direction))
    rollback = signed_gap < -0.025
    scores = [float(frame.get("score") or 1.0) for frame in usable]
    coverage = len(usable) / max(1, int(usable[-1]["frame_idx"]) + 1)
    locked_camera = any(
        token in camera_motion.strip().lower()
        for token in ("locked", "static", "fixed", "固定")
    )
    # Subject centroid movement is not separable from camera movement without
    # a background homography.  Keep moving-camera evidence below the auto-cut
    # threshold until that compensation is available.
    camera_factor = 1.0 if locked_camera else 0.55
    confidence = min(
        0.99,
        max(0.0, float(np.mean(scores)) * min(1.0, coverage / 0.8) * camera_factor),
    )
    evidence: dict[str, Any] = {
        "verdict": "rollback" if rollback else "continuous",
        "confidence": round(confidence, 6),
        "planned_signed_position_gap": round(signed_gap, 6),
        "planned_overlap_frames": planned_overlap_frames,
        "tracked_frame_count": len(usable),
        "tracking_coverage": round(coverage, 6),
        "camera_compensation": (
            "not_required" if locked_camera else "required_before_automatic_cut"
        ),
    }
    if not rollback:
        evidence["reason"] = "subject position does not jump backward at the planned cut"
        return evidence

    catchup = next(
        (
            frame
            for frame in after
            if int(frame["frame_idx"]) >= planned_diagnostic_frame
            and float(
                np.dot(np.asarray(frame["centroid"], dtype=float) - tail, direction)
            )
            >= -0.01
        ),
        None,
    )
    if catchup is None:
        evidence.update(
            repair_action="regenerate",
            reason="the tracked subject never catches up to the previous tail position",
        )
        return evidence
    recommended_trim = int(catchup["frame_idx"]) - seam_frame
    evidence.update(
        repair_action="hard_trim",
        recommended_trim_frames=recommended_trim,
        reason="tracked subject catches up after replaying an earlier position",
    )
    return evidence


def _mask_centroid(encoded: str) -> tuple[float, float] | None:
    payload = encoded.split(",", 1)[-1]
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))).convert("L"))
    yy, xx = np.nonzero(mask > 127)
    if not len(xx):
        return None
    height, width = mask.shape
    return float(xx.mean() / width), float(yy.mean() / height)


class Sam3TrajectoryClient:
    """Small fail-closed client for Milimo's standalone SAM 3 service."""

    def __init__(self, base_url: str, *, timeout_s: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.trust_env = False

    def _post(self, endpoint: str, data: dict[str, Any], timeout: float | None = None) -> dict:
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            data=data,
            timeout=timeout or self.timeout_s,
        )
        response.raise_for_status()
        document = response.json()
        if not isinstance(document, dict) or document.get("error"):
            raise RuntimeError(f"SAM 3 returned an invalid response: {document}")
        return document

    def track(
        self,
        video_path: Path,
        *,
        prompt: str,
        prompt_frame: int,
    ) -> list[dict[str, Any]]:
        session_id = f"honcut-{uuid.uuid4().hex}"
        started = False
        try:
            self._post(
                "/track/start",
                {"video_path": str(video_path), "session_id": session_id},
                timeout=120,
            )
            started = True
            prompt_result = self._post(
                "/track/prompt",
                {
                    "session_id": session_id,
                    "frame_idx": str(prompt_frame),
                    "text": prompt,
                },
                timeout=120,
            )
            object_ids = [str(value) for value in prompt_result.get("object_ids", [])]
            propagated = self._post(
                "/track/propagate",
                {
                    "session_id": session_id,
                    "direction": "both",
                    "start_frame": "-1",
                    "max_frames": "-1",
                },
            )
            tracked_by_frame: dict[int, dict[str, Any]] = {}
            preferred_id = object_ids[0] if object_ids else None
            for frame in propagated.get("frames", []):
                frame_index = int(frame["frame_idx"])
                objects = frame.get("objects") or []
                if objects:
                    selected = next(
                        (
                            item
                            for item in objects
                            if str(item.get("object_id")) == preferred_id
                        ),
                        objects[0],
                    )
                    centroid = selected.get("centroid")
                    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                        tracked_by_frame[frame_index] = {
                            "frame_idx": frame_index,
                            "object_id": str(selected.get("object_id", "")),
                            "centroid": [round(float(centroid[0]), 6), round(float(centroid[1]), 6)],
                            "score": float(selected.get("score", 1.0)),
                        }
                        bbox = selected.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            tracked_by_frame[frame_index]["bbox"] = [
                                round(float(value), 6) for value in bbox
                            ]
                        if selected.get("area_ratio") is not None:
                            tracked_by_frame[frame_index]["area_ratio"] = float(
                                selected["area_ratio"]
                            )
                        continue
                # Compatibility with Milimo's original mask-heavy response.
                masks = frame.get("masks") or {}
                if not masks:
                    continue
                object_id = preferred_id if preferred_id in masks else next(iter(masks))
                centroid = _mask_centroid(str(masks[object_id]))
                if centroid is None:
                    continue
                tracked_by_frame[frame_index] = {
                    "frame_idx": frame_index,
                    "object_id": object_id,
                    "centroid": [round(centroid[0], 6), round(centroid[1], 6)],
                    "score": float((frame.get("scores") or {}).get(object_id, 1.0)),
                }
            return [tracked_by_frame[index] for index in sorted(tracked_by_frame)]
        finally:
            if started:
                try:
                    self._post("/track/stop", {"session_id": session_id}, timeout=30)
                except Exception:
                    pass


def build_tracking_clip(
    previous: Path,
    following: Path,
    output_path: Path,
    *,
    timeline_fps: int,
    previous_tail_frames: int,
    following_frames: int,
    analysis_fps: int | None = None,
) -> int:
    """Build a short analysis-only clip and return its seam frame index."""
    analysis_fps = analysis_fps or timeline_fps
    if timeline_fps <= 0 or analysis_fps <= 0:
        raise ValueError("timeline_fps and analysis_fps must be positive")
    previous_analysis_frames = max(
        1, math.ceil(previous_tail_frames * analysis_fps / timeline_fps)
    )
    following_analysis_frames = max(
        1, math.ceil(following_frames * analysis_fps / timeline_fps)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-sseof",
        f"-{previous_tail_frames / timeline_fps:.9f}",
        "-i",
        str(previous),
        "-i",
        str(following),
        "-filter_complex",
        (
            f"[0:v]fps={analysis_fps},trim=end_frame={previous_analysis_frames},"
            "setpts=PTS-STARTPTS[a];"
            f"[1:v]fps={analysis_fps},trim=end_frame={following_analysis_frames},"
            "setpts=PTS-STARTPTS[b];"
            "[a][b]concat=n=2:v=1:a=0[outv]"
        ),
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot build SAM 3 tracking clip: {detail[-1] if detail else 'unknown error'}"
        )
    os.replace(temporary, output_path)
    return previous_analysis_frames


def _decode_refinement_frames(
    path: Path,
    *,
    width: int = 320,
    height: int = 180,
) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    frame_bytes = width * height * 3
    usable = len(completed.stdout) - len(completed.stdout) % frame_bytes
    if completed.returncode != 0 or usable < frame_bytes:
        detail = completed.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"cannot decode refinement frames from {path}: "
            f"{detail[-1] if detail else 'no decoded video frames'}"
        )
    return np.frombuffer(completed.stdout[:usable], dtype=np.uint8).reshape(
        -1, height, width, 3
    )


def _tracked_bbox(frame: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = frame.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        left, top, right, bottom = (float(value) for value in bbox)
        if 0 <= left < right <= 1 and 0 <= top < bottom <= 1:
            return left, top, right, bottom
    return None


def _track_template_centroids(
    frames: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[list[list[float]], list[float]]:
    """Track one SAM-bounded patch over a short original-frame window."""
    if len(frames) == 0:
        return [], []
    height, width = frames.shape[1:3]
    left = max(0, math.floor(bbox[0] * width))
    top = max(0, math.floor(bbox[1] * height))
    right = min(width, math.ceil(bbox[2] * width))
    bottom = min(height, math.ceil(bbox[3] * height))
    box_width = right - left
    box_height = bottom - top
    if box_width < 4 or box_height < 4:
        return [], []

    gray = frames.astype(np.float32).mean(axis=3)
    template = gray[0, top:bottom, left:right]
    template_centered = template - float(template.mean())
    template_norm = float(np.linalg.norm(template_centered))
    if template_norm < 1e-6:
        return [], []

    centers = [[(left + right) / (2 * width), (top + bottom) / (2 * height)]]
    correlations = [1.0]
    radius = max(6, math.ceil(max(box_width, box_height) * 0.2))
    for frame in gray[1:]:
        search_left = max(0, left - radius)
        search_top = max(0, top - radius)
        search_right = min(width, left + box_width + radius)
        search_bottom = min(height, top + box_height + radius)
        search = frame[search_top:search_bottom, search_left:search_right]
        if search.shape[0] < box_height or search.shape[1] < box_width:
            break
        windows = np.lib.stride_tricks.sliding_window_view(
            search,
            (box_height, box_width),
        )
        window_means = windows.mean(axis=(-2, -1), keepdims=True)
        centered = windows - window_means
        denominator = np.linalg.norm(centered, axis=(-2, -1)) * template_norm
        numerator = np.sum(centered * template_centered, axis=(-2, -1))
        scores = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, -1.0),
            where=denominator > 1e-6,
        )
        row, column = np.unravel_index(int(np.argmax(scores)), scores.shape)
        top = search_top + int(row)
        left = search_left + int(column)
        centers.append(
            [
                (left + box_width / 2) / width,
                (top + box_height / 2) / height,
            ]
        )
        correlations.append(float(scores[row, column]))
    return centers, correlations


def _trajectory_direction(
    tracked: list[dict[str, Any]],
    *,
    seam_frame: int,
    screen_direction: str,
) -> np.ndarray | None:
    explicit = {
        "left_to_right": np.asarray([1.0, 0.0]),
        "right_to_left": np.asarray([-1.0, 0.0]),
        "top_to_bottom": np.asarray([0.0, 1.0]),
        "bottom_to_top": np.asarray([0.0, -1.0]),
    }
    key = screen_direction.strip().lower().replace("-", "_")
    if key in explicit:
        return explicit[key]
    before = [
        np.asarray(frame["centroid"], dtype=float)
        for frame in tracked
        if int(frame.get("frame_idx", -1)) < seam_frame
        and isinstance(frame.get("centroid"), (list, tuple))
        and len(frame["centroid"]) == 2
    ]
    if len(before) < 2:
        return None
    displacement = np.median(np.diff(np.asarray(before[-6:]), axis=0), axis=0)
    norm = float(np.linalg.norm(displacement))
    return displacement / norm if norm >= 0.002 else None


def refine_object_catchup_frame(
    previous: Path,
    following: Path,
    *,
    tracked: list[dict[str, Any]],
    seam_frame: int,
    coarse_trim_analysis_frames: int,
    analysis_fps: int,
    timeline_fps: int,
    planned_overlap_frames: int,
    following_frames: int,
    screen_direction: str,
) -> dict[str, Any]:
    """Refine a coarse SAM catch-up to the original timeline with template tracking."""
    if analysis_fps <= 0 or timeline_fps <= 0:
        raise ValueError("analysis and timeline fps must be positive")
    before = [frame for frame in tracked if int(frame.get("frame_idx", -1)) < seam_frame]
    anchor_index = seam_frame + max(0, coarse_trim_analysis_frames - 1)
    following_anchor = min(
        (frame for frame in tracked if int(frame.get("frame_idx", -1)) >= seam_frame),
        key=lambda frame: abs(int(frame["frame_idx"]) - anchor_index),
        default=None,
    )
    previous_anchor = before[-1] if before else None
    previous_bbox = _tracked_bbox(previous_anchor or {})
    following_bbox = _tracked_bbox(following_anchor or {})
    direction = _trajectory_direction(
        tracked,
        seam_frame=seam_frame,
        screen_direction=screen_direction,
    )
    if previous_bbox is None or following_bbox is None or direction is None:
        return {
            "status": "unavailable",
            "reason": "SAM trajectory lacks bounding boxes or a reliable direction",
        }

    previous_video = _decode_refinement_frames(previous)
    following_video = _decode_refinement_frames(following)
    stride = timeline_fps / analysis_fps
    previous_window = max(2, math.ceil(stride))
    previous_centers, previous_scores = _track_template_centroids(
        previous_video[-previous_window:],
        previous_bbox,
    )
    if len(previous_centers) != previous_window:
        return {"status": "unavailable", "reason": "previous-tail template tracking failed"}
    target = np.asarray(previous_centers[-1], dtype=float)

    anchor_frame = max(
        planned_overlap_frames,
        round(max(0, coarse_trim_analysis_frames - 1) * stride),
    )
    minimum_remaining = max(12, math.ceil(timeline_fps * 0.5))
    safe_last_trim = min(
        following_frames,
        len(following_video),
    ) - minimum_remaining
    if safe_last_trim < anchor_frame:
        return {
            "status": "no_safe_catchup",
            "reason": "coarse catch-up leaves fewer than 0.5 seconds of following material",
            "minimum_remaining_frames": minimum_remaining,
        }
    following_window = following_video[anchor_frame : safe_last_trim + 1]
    following_centers, following_scores = _track_template_centroids(
        following_window,
        following_bbox,
    )
    if len(following_centers) != len(following_window):
        return {"status": "unavailable", "reason": "following template tracking failed"}
    min_correlation = float(os.environ.get("HONCUT_SAM3_REFINE_MIN_CORRELATION", "0.55"))
    observed_scores = previous_scores[1:] + following_scores[1:]
    median_correlation = float(np.median(observed_scores)) if observed_scores else 1.0
    if median_correlation < min_correlation:
        return {
            "status": "unavailable",
            "reason": "local template correlation is below the refinement threshold",
            "median_correlation": round(median_correlation, 6),
        }

    catchup_frame = next(
        (
            anchor_frame + index
            for index, centroid in enumerate(following_centers)
            if float(np.dot(np.asarray(centroid, dtype=float) - target, direction)) >= -0.01
        ),
        None,
    )
    common = {
        "method": "sam_bbox_template_tracking",
        "window_start_frame": anchor_frame,
        "window_end_frame": safe_last_trim,
        "minimum_remaining_frames": minimum_remaining,
        "median_correlation": round(median_correlation, 6),
        "target_centroid": [round(float(value), 6) for value in target],
    }
    if catchup_frame is None:
        return {
            **common,
            "status": "no_safe_catchup",
            "reason": "the subject does not reach the exact previous-tail position before the safe trim limit",
        }
    return {
        **common,
        "status": "refined",
        "recommended_trim_frames": catchup_frame,
        "remaining_frames": min(following_frames, len(following_video)) - catchup_frame,
    }


def collect_sam3_trajectory(
    previous: Path,
    following: Path,
    *,
    boundary_id: str,
    evidence_dir: Path,
    prompt: str,
    timeline_fps: int,
    planned_overlap_frames: int,
    following_frames: int,
    screen_direction: str,
    camera_motion: str,
    base_url: str,
) -> dict[str, Any]:
    """Track the continuity subject and return a compact adjudication signal."""
    analysis_fps = max(
        1,
        min(timeline_fps, int(os.environ.get("HONCUT_SAM3_ANALYSIS_FPS", "3"))),
    )
    tail_frames = min(2 * timeline_fps, following_frames)
    clip_path = evidence_dir / boundary_id / "tracking_clip.mp4"
    seam_frame = build_tracking_clip(
        previous,
        following,
        clip_path,
        timeline_fps=timeline_fps,
        previous_tail_frames=tail_frames,
        following_frames=following_frames,
        analysis_fps=analysis_fps,
    )
    tracked = Sam3TrajectoryClient(base_url).track(
        clip_path,
        prompt=prompt,
        prompt_frame=max(0, seam_frame - 1),
    )
    analysis_overlap_frames = round(planned_overlap_frames * analysis_fps / timeline_fps)
    decision = decide_object_trajectory(
        tracked,
        seam_frame=seam_frame,
        planned_overlap_frames=analysis_overlap_frames,
        screen_direction=screen_direction,
        camera_motion=camera_motion,
    )
    if "recommended_trim_frames" in decision:
        analysis_trim = int(decision["recommended_trim_frames"])
        decision["recommended_trim_analysis_frames"] = analysis_trim
        coarse_timeline_trim = round(analysis_trim * timeline_fps / analysis_fps)
        decision["coarse_recommended_trim_frames"] = coarse_timeline_trim
        decision["recommended_trim_frames"] = coarse_timeline_trim
        try:
            refinement = refine_object_catchup_frame(
                previous,
                following,
                tracked=tracked,
                seam_frame=seam_frame,
                coarse_trim_analysis_frames=analysis_trim,
                analysis_fps=analysis_fps,
                timeline_fps=timeline_fps,
                planned_overlap_frames=planned_overlap_frames,
                following_frames=following_frames,
                screen_direction=screen_direction,
            )
        except Exception as exc:
            refinement = {
                "status": "unavailable",
                "reason": f"timeline refinement failed: {exc}",
            }
        decision["timeline_refinement"] = refinement
        if refinement.get("status") == "refined":
            decision["recommended_trim_frames"] = int(
                refinement["recommended_trim_frames"]
            )
            decision["reason"] = (
                "tracked subject catches up at an original-timeline refined frame"
            )
        elif refinement.get("status") == "no_safe_catchup":
            decision["repair_action"] = "regenerate"
            decision["reason"] = str(refinement["reason"])
    decision["planned_overlap_analysis_frames"] = analysis_overlap_frames
    decision["planned_overlap_frames"] = planned_overlap_frames
    return {
        **decision,
        "detector": "sam3_video_tracker",
        "tracking_prompt": prompt,
        "tracking_clip": str(clip_path),
        "analysis_fps": analysis_fps,
        "timeline_fps": timeline_fps,
        "seam_analysis_frame": seam_frame,
        "trajectory": tracked,
    }


__all__ = [
    "OBJECT_TRAJECTORIES_KIND",
    "Sam3TrajectoryClient",
    "build_tracking_clip",
    "collect_sam3_trajectory",
    "decide_object_trajectory",
    "refine_object_catchup_frame",
]
