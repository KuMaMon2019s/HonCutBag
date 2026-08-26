"""Local OpenCLIP adapter for HonCut's fixed image-style vocabulary."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from utils.clip_interrogator_rank import rank_label_scores
from utils.visual_style_contract import CLIP_STYLE_SCHEMA, style_classification_labels

CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_MODEL_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
CLIP_MODEL_FILENAME = "ViT-B-32-openai.safetensors"
CLIP_MODEL_RECEIPT = "ViT-B-32-openai.json"


def clip_model_cache_dir() -> Path:
    override = str(os.environ.get("HONCUT_MODEL_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve() / "open_clip"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "HonCut" / "open_clip"
    cache_root = str(os.environ.get("XDG_CACHE_HOME") or "").strip()
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return root / "honcut" / "open_clip"


def clip_model_path() -> Path:
    return clip_model_cache_dir() / CLIP_MODEL_FILENAME


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_clip_model(path: Path | None = None) -> Path:
    candidate = Path(path) if path is not None else clip_model_path()
    if not candidate.is_file():
        raise RuntimeError(
            "HonCut CLIP style model is not installed; run "
            "`uv run python pipeline/scripts/install_clip_style_model.py`"
        )
    actual_sha = _sha256_file(candidate)
    if actual_sha != CLIP_MODEL_SHA256:
        raise RuntimeError(
            "HonCut CLIP style model hash mismatch: "
            f"expected {CLIP_MODEL_SHA256}, observed {actual_sha}"
        )
    return candidate


class ClipStyleClassifier:
    """Rank images against the controlled BaseStyle table without BLIP."""

    def __init__(self, *, model_path: Path | None = None, device: str | None = None):
        import open_clip
        import torch

        checkpoint = validate_clip_model(model_path)
        resolved_device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=str(checkpoint),
            device=resolved_device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        labels = style_classification_labels()
        label_names = list(labels)
        with torch.no_grad():
            tokens = tokenizer([labels[name] for name in label_names]).to(resolved_device)
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._device = resolved_device
        self._label_names = label_names
        self._text_features = text_features.detach().float().cpu().numpy()
        self._model_path = checkpoint

    def classify(self, path: Path) -> dict[str, Any]:
        image_path = Path(path)
        if not image_path.is_file():
            raise RuntimeError(f"CLIP style image is missing: {image_path}")
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            with self._torch.no_grad():
                tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                image_features = self._model.encode_image(tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            rankings = rank_label_scores(
                image_features.detach().float().cpu().numpy(),
                self._text_features,
                self._label_names,
                top_count=len(self._label_names),
            )
        except Exception as exc:
            raise RuntimeError(
                f"CLIP style classification failed for {image_path.name}: {exc}"
            ) from exc
        return {
            "schema": CLIP_STYLE_SCHEMA,
            "status": "done",
            "model": f"open_clip/{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}",
            "model_sha256": CLIP_MODEL_SHA256,
            "source_sha256": _sha256_file(image_path),
            "top_style": rankings[0]["base_style"],
            "rankings": rankings,
        }


def write_clip_model_receipt(path: Path | None = None) -> Path:
    checkpoint = validate_clip_model(path)
    receipt = {
        "schema": "honcut.local-clip-model.v1",
        "model": f"open_clip/{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}",
        "source": "pharmapsychotic/clip-interrogator custom LabelTable ranking",
        "source_commit": "bc07ce62c179d3aab3053a623d96a071101d11cb",
        "weights": checkpoint.name,
        "sha256": CLIP_MODEL_SHA256,
        "size_bytes": checkpoint.stat().st_size,
    }
    receipt_path = checkpoint.with_name(CLIP_MODEL_RECEIPT)
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt_path
