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
        min(timeline_fps, int(os.environ.get("HONCUT_SAM3_ANALYSIS_FPS", "6"))),
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
        decision["recommended_trim_frames"] = round(
            analysis_trim * timeline_fps / analysis_fps
        )
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
]
