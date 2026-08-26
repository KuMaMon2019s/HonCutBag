"""Minimal custom-label ranking port from CLIP Interrogator.

Derived from ``pharmapsychotic/clip-interrogator`` commit
``bc07ce62c179d3aab3053a623d96a071101d11cb``. The upstream LabelTable also
downloads large generic artist/flavor tables and BLIP caption models; HonCut
only needs its normalized image-to-custom-label similarity ranking.

Upstream license: MIT. The full notice is kept in
``vendor/clip-interrogator/LICENSE``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("CLIP features must have non-zero norms")
    return rows / norms


def rank_label_scores(
    image_features: np.ndarray,
    text_features: np.ndarray,
    labels: Sequence[str],
    *,
    top_count: int = 3,
) -> list[dict[str, float | str]]:
    """Rank a fixed custom label table by normalized CLIP similarity."""
    label_values = [str(value) for value in labels]
    if not label_values:
        raise ValueError("CLIP label table must not be empty")
    image_rows = _normalized_rows(np.asarray(image_features))
    text_rows = _normalized_rows(np.asarray(text_features))
    if image_rows.shape[0] != 1:
        raise ValueError("CLIP ranking accepts exactly one image feature row")
    if text_rows.shape[0] != len(label_values):
        raise ValueError("CLIP text feature count does not match labels")
    if image_rows.shape[1] != text_rows.shape[1]:
        raise ValueError("CLIP image and text feature dimensions differ")
    count = max(1, min(int(top_count), len(label_values)))
    similarities = (image_rows @ text_rows.T)[0]
    indexes = np.argsort(-similarities, kind="stable")[:count]
    return [
        {
            "base_style": label_values[int(index)],
            "score": float(similarities[int(index)]),
        }
        for index in indexes
    ]
