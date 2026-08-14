"""Tracking-only SAM 3 HTTP service for HonCut continuity diagnostics."""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException

from .policy import estimate_weight_bytes, resolve_checkpoint_path, resolve_runtime_policy

LOGGER = logging.getLogger("honcut.sam3")
REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_SAM3 = REPO_ROOT / "vendor" / "sam3"

app = FastAPI(title="HonCut SAM 3 continuity tracker", version="1.0")
_LOCK = threading.RLock()
_PREDICTOR: Any | None = None
_RUNTIME_REPORT: dict[str, Any] = {}


def _capabilities(torch: Any) -> tuple[bool, bool]:
    mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
    return mps, bool(torch.cuda.is_available())


def _checkpoint_path() -> Path:
    return resolve_checkpoint_path(
        REPO_ROOT,
        configured_checkpoint=os.environ.get("SAM3_CHECKPOINT", ""),
        asset_root=os.environ.get("SAM3_ASSET_ROOT", ""),
    )


def _policy_preview() -> dict[str, Any]:
    try:
        import torch

        mps, cuda = _capabilities(torch)
    except ImportError:
        mps = cuda = False
    try:
        policy = resolve_runtime_policy(
            requested_device=os.environ.get("SAM3_DEVICE", "auto"),
            requested_precision=os.environ.get("SAM3_PRECISION", "auto"),
            mps_available=mps,
            cuda_available=cuda,
            requested_cpu_threads=int(os.environ.get("SAM3_CPU_THREADS", "0")) or None,
        )
        return policy.as_dict()
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


def _count_parameters(model: Any) -> tuple[int, int]:
    import torch

    total = sum(parameter.numel() for parameter in model.parameters())
    linear = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.weight is not None
    )
    return int(total), int(linear)


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _apply_precision(predictor: Any, policy: Any, torch: Any) -> str | None:
    if policy.precision == "fp16":
        predictor.model.half()
        return None
    elif policy.precision == "fp32":
        predictor.model.float()
        return None
    elif policy.precision == "int8_dynamic":
        supported = set(torch.backends.quantized.supported_engines)
        if "qnnpack" not in supported:
            raise RuntimeError(
                "this PyTorch build has no QNNPACK engine for ARM CPU INT8; use SAM3_PRECISION=fp32"
            )
        # Current macOS wheels expose QNNPACK but leave the active engine as
        # "none". Selecting it is required before packing quantized weights.
        torch.backends.quantized.engine = "qnnpack"
        predictor.model.float()
        predictor.model = torch.ao.quantization.quantize_dynamic(
            predictor.model,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )
        return "qnnpack"
    return None


def _load_predictor() -> Any:
    global _PREDICTOR, _RUNTIME_REPORT
    with _LOCK:
        if _PREDICTOR is not None:
            return _PREDICTOR
        checkpoint = _checkpoint_path()
        if not checkpoint.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"SAM 3 checkpoint not found at {checkpoint}. Place sam3.pt there "
                    "or set SAM3_CHECKPOINT. HonCut does not download large weights implicitly."
                ),
            )
        if str(VENDORED_SAM3) not in sys.path:
            sys.path.insert(0, str(VENDORED_SAM3))

        try:
            import torch
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"SAM 3 runtime dependency missing: {exc}; run pipeline/scripts/setup_sam3.sh",
            ) from exc

        mps, cuda = _capabilities(torch)
        try:
            policy = resolve_runtime_policy(
                requested_device=os.environ.get("SAM3_DEVICE", "auto"),
                requested_precision=os.environ.get("SAM3_PRECISION", "auto"),
                mps_available=mps,
                cuda_available=cuda,
                requested_cpu_threads=int(os.environ.get("SAM3_CPU_THREADS", "0")) or None,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if policy.device == "cpu":
            torch.set_num_threads(policy.cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        started_at = time.monotonic()
        rss_before = _rss_bytes()
        LOGGER.info(
            "Loading SAM 3 video model on %s with %s precision",
            policy.device,
            policy.precision,
        )
        predictor = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint),
            apply_temporal_disambiguation=True,
            async_loading_frames=False,
            video_loader_type="cv2",
            device=policy.device,
        )
        total_parameters, linear_parameters = _count_parameters(predictor.model)
        try:
            quantization_backend = _apply_precision(predictor, policy, torch)
        except RuntimeError as exc:
            del predictor
            gc.collect()
            raise HTTPException(status_code=503, detail=f"SAM 3 quantization failed: {exc}") from exc
        _PREDICTOR = predictor
        rss_after = _rss_bytes()
        _RUNTIME_REPORT = {
            "loaded": True,
            "device": policy.device,
            "precision": policy.precision,
            "cpu_threads": policy.cpu_threads,
            "quantized_linear": policy.quantize_linear,
            "quantization_backend": quantization_backend,
            "checkpoint": str(checkpoint),
            "total_parameters": total_parameters,
            "linear_parameters": linear_parameters,
            "estimated_weight_bytes": estimate_weight_bytes(
                total_parameters=total_parameters,
                linear_parameters=linear_parameters,
                precision=policy.precision,
            ),
            "load_seconds": round(time.monotonic() - started_at, 3),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_load_delta_bytes": (
                rss_after - rss_before
                if rss_before is not None and rss_after is not None
                else None
            ),
        }
        return predictor


def _serialise_objects(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    object_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    probabilities = outputs.get("out_probs")
    if object_ids is None or masks is None:
        return []
    if hasattr(object_ids, "detach"):
        object_ids = object_ids.detach().cpu().numpy()
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    if probabilities is not None and hasattr(probabilities, "detach"):
        probabilities = probabilities.detach().cpu().numpy()

    objects: list[dict[str, Any]] = []
    for index, raw_object_id in enumerate(np.asarray(object_ids).reshape(-1)):
        mask = np.asarray(masks[index]).squeeze()
        yy, xx = np.nonzero(mask > 0)
        if not len(xx):
            continue
        height, width = mask.shape[-2:]
        score = 1.0 if probabilities is None else float(np.asarray(probabilities).reshape(-1)[index])
        objects.append(
            {
                "object_id": str(int(raw_object_id)),
                "centroid": [round(float(xx.mean() / width), 6), round(float(yy.mean() / height), 6)],
                "bbox": [
                    round(float(xx.min() / width), 6),
                    round(float(yy.min() / height), 6),
                    round(float((xx.max() + 1) / width), 6),
                    round(float((yy.max() + 1) / height), 6),
                ],
                "score": round(score, 6),
                "area_ratio": round(float(len(xx) / (height * width)), 6),
            }
        )
    return objects


@app.get("/health")
def health() -> dict[str, Any]:
    checkpoint = _checkpoint_path()
    return {
        "status": "ready" if _PREDICTOR is not None else "unloaded",
        "model_loaded": _PREDICTOR is not None,
        "checkpoint": str(checkpoint),
        "checkpoint_present": checkpoint.is_file(),
        "policy": _policy_preview(),
        "max_input_frames": int(os.environ.get("SAM3_MAX_FRAMES", "96")),
    }


@app.get("/runtime")
def runtime() -> dict[str, Any]:
    return _RUNTIME_REPORT or health()


@app.post("/track/start")
def track_start(video_path: str = Form(...), session_id: str | None = Form(None)) -> dict[str, Any]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {path}")
    with _LOCK:
        predictor = _load_predictor()
        # The service is intentionally single-session: abandoned frame caches
        # must not accumulate in 16 GB of unified memory.
        for existing_session in list(predictor._ALL_INFERENCE_STATES):
            predictor.close_session(existing_session)
        result = predictor.handle_request(
            {"type": "start_session", "resource_path": str(path), "session_id": session_id}
        )
        max_input_frames = max(1, int(os.environ.get("SAM3_MAX_FRAMES", "96")))
        if int(result.get("num_frames", 0)) > max_input_frames:
            predictor.close_session(result["session_id"])
            raise HTTPException(
                status_code=413,
                detail=(
                    f"tracking clip has {result['num_frames']} frames; "
                    f"SAM3_MAX_FRAMES is {max_input_frames}"
                ),
            )
        # SAM 3 stores decoded frames as FP16. CPU dynamic-INT8 keeps Conv
        # layers in FP32, so promote only the frame buffer to avoid dtype
        # mismatches while retaining quantized Linear weights.
        if _RUNTIME_REPORT.get("device") == "cpu":
            state = predictor._ALL_INFERENCE_STATES[result["session_id"]]["state"]
            state["input_batch"].img_batch = state["input_batch"].img_batch.float()
    return {"status": "started", **result}


@app.post("/track/prompt")
def track_prompt(
    session_id: str = Form(...),
    frame_idx: int = Form(0),
    text: str = Form(...),
) -> dict[str, Any]:
    with _LOCK:
        result = _load_predictor().handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": text,
            }
        )
    outputs = result.get("outputs") or {}
    objects = _serialise_objects(outputs)
    return {
        "status": "prompt_added",
        "frame_index": result.get("frame_index", frame_idx),
        "object_ids": [item["object_id"] for item in objects],
        "objects": objects,
    }


@app.post("/track/propagate")
def track_propagate(
    session_id: str = Form(...),
    direction: str = Form("both"),
    start_frame: int = Form(-1),
    max_frames: int = Form(-1),
) -> dict[str, Any]:
    if direction not in {"forward", "backward", "both"}:
        raise HTTPException(status_code=422, detail=f"invalid direction: {direction}")
    predictor = _load_predictor()
    by_frame: dict[int, dict[str, Any]] = {}
    with _LOCK:
        stream = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
                "start_frame_index": start_frame if start_frame >= 0 else None,
                "max_frame_num_to_track": max_frames if max_frames >= 0 else None,
            }
        )
        for result in stream:
            frame_index = int(result.get("frame_index", -1))
            objects = _serialise_objects(result.get("outputs") or {})
            if objects or frame_index not in by_frame:
                by_frame[frame_index] = {
                    "frame_idx": frame_index,
                    "num_objects": len(objects),
                    "objects": objects,
                }
    frames = [by_frame[index] for index in sorted(by_frame)]
    return {
        "status": "complete",
        "session_id": session_id,
        "direction": direction,
        "frame_count": len(frames),
        "frames": frames,
    }


@app.post("/track/stop")
def track_stop(session_id: str = Form(...)) -> dict[str, Any]:
    with _LOCK:
        _load_predictor().handle_request({"type": "close_session", "session_id": session_id})
    return {"status": "closed", "session_id": session_id}


@app.post("/unload")
def unload() -> dict[str, Any]:
    global _PREDICTOR, _RUNTIME_REPORT
    with _LOCK:
        predictor = _PREDICTOR
        if predictor is not None:
            for session_id in list(predictor._ALL_INFERENCE_STATES):
                predictor.close_session(session_id)
        _PREDICTOR = None
        _RUNTIME_REPORT = {}
        del predictor
        gc.collect()
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
    return {"status": "unloaded", "rss_bytes": _rss_bytes()}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("SAM3_LOG_LEVEL", "INFO"))
    uvicorn.run(
        "sam3_runtime.server:app",
        host=os.environ.get("SAM3_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAM3_PORT", "8001")),
        workers=1,
    )


if __name__ == "__main__":
    main()
