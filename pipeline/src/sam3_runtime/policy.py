"""Pure runtime-policy helpers; importing this module does not import PyTorch."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePolicy:
    """Resolved SAM 3 execution policy for one host."""

    device: str
    precision: str
    cpu_threads: int
    quantize_linear: bool
    bytes_per_float_parameter: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_checkpoint_path(
    repo_root: Path,
    *,
    configured_checkpoint: str = "",
    asset_root: str = "",
) -> Path:
    """Resolve a local checkpoint without copying or implicitly downloading it."""
    if configured_checkpoint.strip():
        return Path(configured_checkpoint).expanduser().resolve()

    local_checkpoint = repo_root / "pipeline" / "models" / "sam3" / "sam3.pt"
    shared_root = (
        Path(asset_root).expanduser().resolve()
        if asset_root.strip()
        else repo_root.parent / "sam3"
    )
    candidates = (
        local_checkpoint,
        shared_root / "sam3.pt",
        shared_root / "权重" / "sam3.pt",
        shared_root / "weights" / "sam3.pt",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), local_checkpoint)


def resolve_runtime_policy(
    *,
    requested_device: str = "auto",
    requested_precision: str = "auto",
    mps_available: bool = False,
    cuda_available: bool = False,
    cpu_count: int | None = None,
    requested_cpu_threads: int | None = None,
) -> RuntimePolicy:
    """Choose a conservative Apple-Silicon-friendly device and precision pair."""
    device = requested_device.strip().lower()
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported SAM3_DEVICE: {requested_device}")
    if device == "auto":
        device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if device == "mps" and not mps_available:
        raise ValueError("SAM3_DEVICE=mps was requested but MPS is unavailable")
    if device == "cuda" and not cuda_available:
        raise ValueError("SAM3_DEVICE=cuda was requested but CUDA is unavailable")

    precision = requested_precision.strip().lower()
    if precision not in {"auto", "fp32", "fp16", "int8_dynamic"}:
        raise ValueError(f"unsupported SAM3_PRECISION: {requested_precision}")
    if precision == "auto":
        precision = (
            "int8_dynamic"
            if device == "cpu"
            else "fp16"
            if device == "cuda"
            else "fp32"
        )
    if precision == "int8_dynamic" and device != "cpu":
        raise ValueError("dynamic INT8 quantization is supported only on CPU")
    if precision == "fp16" and device == "cpu":
        raise ValueError("FP16 CPU inference is unsupported; use fp32 or int8_dynamic")

    available_cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    default_threads = min(6, available_cpus)
    cpu_threads = requested_cpu_threads or default_threads
    cpu_threads = max(1, min(int(cpu_threads), available_cpus))
    return RuntimePolicy(
        device=device,
        precision=precision,
        cpu_threads=cpu_threads,
        quantize_linear=precision == "int8_dynamic",
        bytes_per_float_parameter=2 if precision == "fp16" else 4,
    )


def estimate_weight_bytes(
    *,
    total_parameters: int,
    linear_parameters: int,
    precision: str,
) -> int:
    """Estimate resident weight bytes, excluding activations and framework overhead."""
    total = max(0, int(total_parameters))
    linear = min(total, max(0, int(linear_parameters)))
    if precision == "fp32":
        return total * 4
    if precision == "fp16":
        return total * 2
    if precision == "int8_dynamic":
        # Dynamic quantization stores Linear weights as INT8 while keeping the
        # rest of the model in FP32. Bias/scale overhead is intentionally omitted.
        return linear + (total - linear) * 4
    raise ValueError(f"unsupported precision: {precision}")
