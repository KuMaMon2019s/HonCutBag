"""Build and upload Bridge image-contract v2.0 task directories."""

import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from tools.asset_packager import (
    CINEMATIC_FIRST_FRAME_SCHEMA,
    _assert_video_frame_provenance,
    _detect_shot_characters,
    collect_character_reference_assets,
    inject_omni_reference_instruction,
)
from utils.prompt_budget import enforce_prompt_budget


def _shot_meta(meta: Mapping, shot_id: str) -> dict:
    shots = meta.get("shots")
    if isinstance(shots, Mapping):
        value = shots.get(shot_id, {})
        return dict(value) if isinstance(value, Mapping) else {}
    if meta.get("shot_id") == shot_id or len(meta.get("shot_ids", [])) <= 1:
        return dict(meta)
    value = meta.get(shot_id, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _next_task_id(task_root: Path, now: datetime) -> str:
    prefix = now.strftime("%y-%m-%d")
    existing = []
    if task_root.exists():
        for path in task_root.glob(f"{prefix}_[0-9][0-9]"):
            try:
                existing.append(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                continue
    return f"{prefix}_{max(existing, default=0) + 1:02d}"


def _copy_image(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Task directory asset missing: {source}")
    from clients.tos_uploader import compress_image_bytes

    image_data = compress_image_bytes(source.read_bytes())
    if image_data.startswith(b"\xff\xd8\xff"):
        destination = destination.with_suffix(".jpg")
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(image_data)
    return destination


def build_task_dir(output_dir, shot_ids: Sequence[str], meta: Mapping) -> Path:
    """Assemble a local task tree matching contract v2.0 sections 2.1–2.4."""
    output_dir = Path(output_dir)
    shot_ids = list(shot_ids)
    now = datetime.now().astimezone()
    task_root = output_dir / ".task_dirs"
    task_id = _next_task_id(task_root, now)
    task_dir = task_root / task_id
    task_dir.mkdir(parents=True, exist_ok=False)

    for shot_id in shot_ids:
        current = _shot_meta(meta, shot_id)
        requested_strategy = current.get(
            "gen_strategy", current.get("strategy", "i2v")
        )
        if requested_strategy not in {"i2v", "flf2v", "phantom"}:
            requested_strategy = "i2v"
        strategy = requested_strategy
        shot_dir = task_dir / shot_id
        shot_dir.mkdir()

        frame_override = current.get("_storyboard_frame_path")
        first_source = (
            Path(str(frame_override))
            if frame_override and Path(str(frame_override)).is_absolute()
            else output_dir / str(frame_override)
            if frame_override
            else output_dir / "storyboard_images" / f"{shot_id}.png"
        )
        cinematic_frame = None
        if first_source.exists():
            _assert_video_frame_provenance(
                first_source,
                current.get("_storyboard_frame_kind")
                or CINEMATIC_FIRST_FRAME_SCHEMA,
            )
            first_destination = _copy_image(first_source, shot_dir / "分镜" / "分镜图.png")
            cinematic_frame = first_destination.relative_to(shot_dir).as_posix()
        first_frame = cinematic_frame if strategy == "flf2v" else None
        last_frame = None
        if strategy == "flf2v":
            end_source = output_dir / "storyboard_images" / f"{shot_id}_end.png"
            end_destination = _copy_image(end_source, shot_dir / "分镜" / "分镜尾图.png")
            last_frame = end_destination.relative_to(shot_dir).as_posix()

        references = []
        phantom_identity_assets = (
            collect_character_reference_assets(output_dir, current)
            if requested_strategy == "phantom"
            else []
        )
        expected_characters = (
            _detect_shot_characters(output_dir, current)
            if requested_strategy == "phantom"
            else []
        )
        if expected_characters and not phantom_identity_assets:
            raise FileNotFoundError(
                "Phantom character references missing for shot "
                f"{shot_id}; expected reference_board.png, legacy canonical "
                "views"
            )
        assets = []
        if strategy == "phantom" and cinematic_frame:
            assets.append({
                "path": first_source,
                "char_id": None,
                "role": "reference_image",
                "priority": "high",
                "bind_subject": False,
                "reference_kind": "cinematic_composition",
                "reference_description": (
                    f"{shot_id}成片质感第一帧，用于锁定本生成片段的构图、"
                    "角色站位、场景结构、项目美术风格、时间天气和光影；"
                    "该资产已经过无文字、无箭头、无分格的像素洁净检查"
                ),
            })
        if strategy == "phantom":
            assets.extend(phantom_identity_assets)
        for index, asset in enumerate(assets, start=1):
            source = asset["path"]
            if asset.get("reference_kind") == "cinematic_composition":
                relative = cinematic_frame
            elif source.name == "reference_board.png":
                filename = "四视图身份参考板.png"
            elif source.name == "face_closeup.png":
                filename = "大头照.png"
            elif source.name == "full_body.png":
                filename = "全身照.png"
            else:
                filename = "静态身份参考.png"
            if asset.get("reference_kind") != "cinematic_composition":
                destination = _copy_image(source, shot_dir / asset["char_id"] / filename)
                relative = destination.relative_to(shot_dir).as_posix()
            references.append({
                "path": relative,
                "label": f"图片{index}",
                "desc": asset["reference_description"],
            })

        prompt_dir = shot_dir / "提示词"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt = current.get("prompt", "")
        if references:
            # Preserve character ids so face/full-body/variant images of one
            # person bind to one subject, matching the ordinary content[] path.
            prompt = inject_omni_reference_instruction(prompt, assets)
        frame_instructions = []
        if first_frame:
            frame_instructions.append(
                f"{first_frame}是{shot_id}成片质感首帧，用于锁定构图、角色站位、"
                "场景结构、项目美术风格、时间天气和光影；不得使用 PREVIS 工作板"
            )
        if last_frame:
            frame_instructions.append(
                f"{last_frame}是{shot_id}分镜尾帧，用于锁定镜头结束时的"
                "动作、构图和光影"
            )
        if frame_instructions:
            prompt = f"成片首帧参考说明：{'；'.join(frame_instructions)}。{prompt}"
        enforce_prompt_budget(
            prompt,
            provider="bridge",
            model=str(meta.get("model") or "unknown"),
            purpose="video_generation",
        )
        (prompt_dir / "提示词.txt").write_text(prompt, encoding="utf-8")
        manifest = {
            "shot_id": shot_id,
            "strategy": strategy,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "references": references,
            "prompt_file": "提示词/提示词.txt",
            "duration": current.get("duration"),
            "width": current.get("width", 1280),
            "height": current.get("height", 720),
            "generate_audio": current.get("generate_audio", True),
        }
        (shot_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (shot_dir / "OutPut").mkdir()

    task_manifest = {
        "task_id": task_id,
        "created_at": now.isoformat(timespec="seconds"),
        "shots": shot_ids,
        "model": meta.get("model", "doubao-seedance-2.0-fast"),
        "resolution": meta.get("resolution", "480p"),
        "chain_mode": bool(meta.get("chain_mode", False)),
        "tos_prefix": f"tasks/{task_id}",
    }
    (task_dir / "task_manifest.json").write_text(
        json.dumps(task_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return task_dir


def upload_task_dir(task_dir, tos_prefix: str) -> str:
    """Recursively upload every file and return the v2.0 task id."""
    from clients import tos_uploader

    task_dir = Path(task_dir)
    task_id = task_dir.name
    prefix = tos_prefix.strip("/")
    if prefix.endswith(f"/{task_id}") or prefix == task_id:
        object_root = prefix
    else:
        object_root = f"{prefix}/{task_id}" if prefix else f"tasks/{task_id}"
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(task_dir).as_posix()
        object_key = f"{object_root}/{relative}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        uploaded = tos_uploader.upload_file(path.read_bytes(), object_key, content_type)
        if not uploaded:
            raise RuntimeError(f"Failed to upload task file: {object_key}")
    return task_id
