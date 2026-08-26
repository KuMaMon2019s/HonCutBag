#!/usr/bin/env python3
# ruff: noqa: E402
"""Install the pinned local OpenCLIP style model outside the Git worktree."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.clip_style_classifier import (
    CLIP_MODEL_NAME,
    CLIP_MODEL_SHA256,
    CLIP_PRETRAINED,
    clip_model_cache_dir,
    clip_model_path,
    validate_clip_model,
    write_clip_model_receipt,
)


def _download_source(download_cache: Path) -> Path:
    import httpx
    import open_clip
    from huggingface_hub import close_session, set_client_factory

    set_client_factory(lambda: httpx.Client(trust_env=False, follow_redirects=True))
    close_session()
    open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device="cpu",
        cache_dir=str(download_cache),
    )
    candidates = [
        path.resolve()
        for path in download_cache.rglob("open_clip_model.safetensors")
        if path.is_file()
    ]
    for candidate in candidates:
        try:
            validate_clip_model(candidate)
        except RuntimeError:
            continue
        return candidate
    raise RuntimeError(
        "download completed without the pinned OpenCLIP weight "
        f"{CLIP_MODEL_SHA256}"
    )


def install(source: Path | None = None) -> Path:
    target = clip_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        validate_clip_model(target)
        write_clip_model_receipt(target)
        return target
    source_path = Path(source).expanduser().resolve() if source else _download_source(
        clip_model_cache_dir() / "downloads"
    )
    validate_clip_model(source_path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source_path, temporary)
    temporary.replace(target)
    validate_clip_model(target)
    write_clip_model_receipt(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="optional already-downloaded pinned safetensors file",
    )
    args = parser.parse_args()
    installed = install(args.source)
    print(f"Installed HonCut CLIP style model: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
