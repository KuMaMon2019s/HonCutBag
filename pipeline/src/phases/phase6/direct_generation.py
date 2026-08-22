"""Direct provider generation, privacy fallback, and durable task identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from phases.phase2.storyboard_assets import _shot_storyboard_reference
from prompt.shot_prompt_builder import build_batch_prompts
from quality.quality_gate import run_quality_check
from runtime.generation_fingerprint import (
    PHASE6_VIDEO_PROMPT_TEMPLATE_ID,
    PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION,
    GenerationFingerprint,
    build_generation_fingerprint,
)
from runtime.phase_timing import _banner, _elapsed, _now
from runtime.provider_policy import ProviderExecutionPolicy
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.vendor_adapter import VendorAdapter, VendorModel
from utils.character_body_contracts import character_visual_description
from utils.config import get_api_key
from utils.file_integrity import _file_sha256
from utils.timing_estimator import estimate_phase_duration


def _phase6_output_failure(
    shot_id: str,
    output_path: Path,
    receipt: dict[str, Any] | None,
    task: Any,
    *,
    validate_video: Callable[[Path], bool] | None = None,
) -> str | None:
    """Return why a Phase 6 file is not proven to belong to this execution."""
    if receipt is None:
        return "no successful current-input generation receipt"
    if task is None or task.status != "succeeded":
        return "generation ledger receipt is missing or not succeeded"
    if task.resource_id != shot_id:
        return "generation ledger receipt belongs to a different shot"
    if task.payload.get("input_fingerprint") != receipt.get("input_fingerprint"):
        return "generation ledger input fingerprint mismatch"
    if not output_path.is_file():
        return "output.mp4 missing after successful generation"
    if task.outcome.get("output_sha256") != _file_sha256(output_path):
        return "output.mp4 hash does not match generation ledger"
    if validate_video is None:
        from utils.video_validation import is_valid_video

        validate_video = is_valid_video
    if not validate_video(output_path):
        return "output.mp4 failed ffprobe validation"
    return None


def _apply_chain_relay(content_list, first_frame_b64, shot_id):
    """Replace the first frame with a relay frame unless content is reference-only."""
    if any(item.get("role") == "reference_image" for item in (content_list or [])):
        print(
            f"    [chain] {shot_id}: reference-only shot, skipping tail-frame relay",
            flush=True,
        )
        return content_list
    if any(
        item.get("type") == "text"
        and "[identity-lock: text-only; no reference media]" in item.get("text", "")
        for item in (content_list or [])
    ):
        print(
            f"    [chain] {shot_id}: identity-locked FLF2V shot, "
            "keeping its storyboard first frame",
            flush=True,
        )
        return content_list
    content_list = [
        item for item in (content_list or [])
        if item.get("role") != "first_frame"
    ]
    content_list.insert(1 if content_list else 0, {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{first_frame_b64}"},
        "role": "first_frame",
        "priority": "high",
    })
    return content_list


def _prompt_assets_for_shot(shot_meta: dict, characters_data: dict) -> list[dict]:
    """Return only character prompt assets explicitly bound to this shot."""
    requested = shot_meta.get("who") or shot_meta.get("characters") or []
    if not isinstance(requested, list):
        requested = [requested] if requested else []
    requested_keys = {str(value).casefold() for value in requested if value}
    for asset_id in shot_meta.get("associate_assets", []):
        if isinstance(asset_id, str) and asset_id.startswith("char:"):
            requested_keys.add(asset_id[5:].split(":", 1)[0].casefold())

    from utils.pixel_text_policy import strip_pixel_text_identity_markers

    selected = []
    for character in characters_data.get("characters", []):
        keys = {
            str(character.get("id", "")).casefold(),
            str(character.get("name", "")).casefold(),
            *{
                str(alias).casefold()
                for alias in character.get("aliases", [])
                if alias
            },
        }
        if requested_keys.intersection(keys):
            selected.append({
                "name": character.get("name", ""),
                "description": strip_pixel_text_identity_markers(
                    character_visual_description(character)
                ),
            })
    return selected


def _prepare_phase6_prompt(
    shot_id: str,
    shot_meta: dict,
    characters_data: dict,
    scene_consistency_data: dict,
    *,
    video_model: str,
    route_model: str,
) -> tuple[str, bool]:
    """Return the final transport prompt while retaining a reroutable base prompt."""
    from phases.phase6.video_generator import build_video_prompt, resolve_video_lighting
    from utils.pixel_text_policy import strip_pixel_text_identity_markers

    for field in ("subject_description", "character_visual_description"):
        if shot_meta.get(field):
            shot_meta[field] = strip_pixel_text_identity_markers(shot_meta[field])

    scene_contract = scene_consistency_data.get("shots", {}).get(shot_id, {})
    if scene_contract:
        shot_meta["lighting_description"] = resolve_video_lighting(
            scene_contract,
            scene_consistency_data.get("global_lighting")
            or shot_meta.get("lighting_description"),
        )
        shot_meta["style_anchor"] = (
            scene_contract.get("style_anchor")
            or scene_contract.get("style_suffix")
            or scene_consistency_data.get("global_style_lock")
            or shot_meta.get("style_anchor")
        )
        shot_meta["temporal_visual_contract"] = (
            scene_contract.get("temporal_visual_contract")
            or shot_meta.get("temporal_visual_contract")
        )

    prompt = str(shot_meta.get("prompt") or "")
    if scene_consistency_data:
        rendered = build_video_prompt(
            shot_meta,
            characters_data,
            scene_consistency_data,
            video_model,
        )
        if isinstance(rendered, dict):
            prompt = str(rendered.get("prompt") or "")
            shot_meta["negative_prompt"] = rendered.get("negative_prompt") or ""
        else:
            prompt = str(rendered or "")
    prompt = strip_pixel_text_identity_markers(prompt)
    shot_meta["prompt"] = prompt

    route_applied = False
    try:
        from prompt.prompt_router import route_prompt

        route_data = dict(shot_meta)
        route_data["prompt"] = prompt
        routed_prompt = route_prompt(
            model_name=route_model,
            mode="single_shot",
            shot_data=route_data,
            assets=_prompt_assets_for_shot(shot_meta, characters_data),
        )
        if routed_prompt:
            # The router may restate shot-level identity fields, so enforce the
            # pixel-text boundary once more on the final transport prompt.
            prompt = strip_pixel_text_identity_markers(routed_prompt)
            route_applied = True
    except Exception:
        # The complete Phase 6 prompt remains the compatibility fallback when
        # an optional model-specific formatter is unavailable.
        pass
    return prompt, route_applied


def _rejected_privacy_image_url(content: list[dict] | None, error: object) -> str | None:
    """Return the stable URL key for the provider-rejected content[N] image."""
    import re
    from urllib.parse import urlsplit

    match = re.search(r"content\[(\d+)\]", str(error))
    if not match or not content:
        return None
    index = int(match.group(1))
    if not 0 <= index < len(content):
        return None
    item = content[index]
    if item.get("type") != "image_url":
        return None
    url = item.get("image_url", {}).get("url")
    if not url:
        return None
    parsed = urlsplit(str(url))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _without_rejected_privacy_images(
    content: list[dict], rejected_urls: set[str]
) -> list[dict]:
    """Remove only previously rejected images while retaining safe references."""
    if not rejected_urls:
        return content
    from urllib.parse import urlsplit

    filtered = []
    for item in content:
        url = item.get("image_url", {}).get("url") if item.get("type") == "image_url" else None
        if url:
            parsed = urlsplit(str(url))
            stable_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if stable_url in rejected_urls:
                continue
        filtered.append(item)
    return filtered


def _privacy_fallback_strategy(gen_strategy: str) -> str:
    """Return a structurally valid route after an endpoint-frame rejection."""
    # FLF2V requires first_frame + last_frame as an inseparable pair. Removing
    # only one endpoint creates an invalid request, so fall back to Phantom's
    # identity references instead. Phantom/i2v can drop one rejected image.
    return "phantom" if gen_strategy == "flf2v" else gen_strategy


def _generation_fingerprint(
    *,
    output_dir: Path,
    shot_id: str,
    meta: dict,
    prompt: str,
    first_frame_b64: str | None,
    content: list[dict] | None,
    chain_source: tuple[str, Path] | None,
    provider_id: str,
    provider_version: str,
    model_id: str,
    model_version: str,
    generation_parameters: dict[str, Any],
) -> GenerationFingerprint:
    """Hash the immutable semantic and local-media inputs to one paid clip."""
    semantic_fields = (
        "who",
        "associate_assets",
        "gen_strategy",
        "generation_actions",
        "body_action_choreography",
        "body_action_contract",
        "generation_load",
        "source_action_unit_ids",
        "start_state",
        "end_state",
        "causal_link",
        "action_description",
        "camera_movement",
        "shot_size",
        "where",
        "time",
        "time_of_day",
        "time_window",
        "temporal_visual_contract",
        "lighting_description",
        "style_anchor",
    )
    local_assets = []
    for candidate in (
        output_dir / "storyboard_images" / f"{shot_id}.png",
        output_dir / "storyboard_images" / f"{shot_id}_end.png",
    ):
        if candidate.is_file():
            local_assets.append(
                {"path": str(candidate.relative_to(output_dir)), "sha256": _file_sha256(candidate)}
            )
    try:
        from tools.asset_packager import collect_character_reference_assets

        for asset in collect_character_reference_assets(output_dir, meta):
            candidate = Path(asset["path"])
            if candidate.is_file():
                local_assets.append(
                    {
                        "path": str(candidate.relative_to(output_dir)),
                        "sha256": _file_sha256(candidate),
                    }
                )
    except (OSError, KeyError, ValueError):
        pass
    if chain_source is not None and chain_source[1].is_file():
        local_assets.append(
            {
                "path": str(chain_source[1]),
                "sha256": _file_sha256(chain_source[1]),
            }
        )

    stable_content = []
    for item in content or []:
        stable_item = {
            key: item.get(key)
            for key in ("type", "role", "priority", "text")
            if item.get(key) is not None
        }
        stable_content.append(stable_item)

    run_manifest = output_dir / "RUN_MANIFEST.json"
    run_fingerprint = None
    if run_manifest.is_file():
        try:
            run_fingerprint = json.loads(run_manifest.read_text(encoding="utf-8")).get(
                "run_fingerprint"
            )
        except (OSError, json.JSONDecodeError):
            pass
    parameters = {
        "shot_id": shot_id,
        "meta": {field: meta.get(field) for field in semantic_fields},
        "first_frame_sha256": (
            hashlib.sha256(first_frame_b64.encode("ascii")).hexdigest()
            if first_frame_b64
            else None
        ),
        "content": stable_content,
        "run_fingerprint": run_fingerprint,
        **generation_parameters,
    }
    return build_generation_fingerprint(
        prompt_text=prompt,
        prompt_template_id=PHASE6_VIDEO_PROMPT_TEMPLATE_ID,
        prompt_template_version=PHASE6_VIDEO_PROMPT_TEMPLATE_VERSION,
        provider_id=provider_id,
        provider_version=provider_version,
        model_id=model_id,
        model_version=model_version,
        parameters=parameters,
        input_artifact_hashes={
            item["path"]: item["sha256"] for item in local_assets
        },
    )


def _generation_input_fingerprint(
    *,
    output_dir: Path,
    shot_id: str,
    meta: dict,
    prompt: str,
    first_frame_b64: str | None,
    content: list[dict] | None,
    chain_source: tuple[str, Path] | None,
) -> str:
    """Compatibility wrapper for callers that only need the digest."""
    return _generation_fingerprint(
        output_dir=output_dir,
        shot_id=shot_id,
        meta=meta,
        prompt=prompt,
        first_frame_b64=first_frame_b64,
        content=content,
        chain_source=chain_source,
        provider_id="compatibility",
        provider_version="1",
        model_id="unspecified",
        model_version="unspecified",
        generation_parameters={},
    ).value


def _run_phase6_fallback(output_dir: Path, chain_mode: bool = False) -> dict:
    """Generate Phase 6 video through direct ARK or the explicit local Bridge."""
    output_dir = Path(output_dir)

    shots_dir = output_dir / "shots"
    if not shots_dir.exists():
        return {"status": "skipped", "reason": "no shots directory"}

    from utils.config import get_video_route

    configured_provider = os.environ.get("VIDEO_PROVIDER", "seedance").lower()
    # ``bridge`` was historically overloaded as a provider name.  Keep it as
    # a compatibility alias for Seedance-over-Bridge while new configuration
    # uses VIDEO_PROVIDER=seedance + VIDEO_GENERATION_MODE=bridge.
    video_provider = (
        "seedance" if configured_provider in {"bridge", "ark"} else configured_provider
    )
    video_route = get_video_route(configured_provider)
    use_local = video_route in {"bridge", "local"}
    if use_local:
        try:
            from clients import local_video_client
        except ImportError:
            print("  ✗ Phase 6 前置检查失败: local_video_client 未找到", flush=True)
            return {"status": "error", "error": "local_video_client not found"}
        if not local_video_client.is_available(timeout=3.0):
            print("  ✗ Phase 6 前置检查失败: 本地视频 API 不可达", flush=True)
            return {"status": "error", "error": "local video API unreachable"}
        if video_route == "bridge":
            print(f"  → 路由: 通过 Bridge 使用 {video_provider} 模型", flush=True)
        else:
            print("  → 路由: 仅使用配置的本地视频 API", flush=True)
        if chain_mode and video_provider != "seedance":
            print("  [chain] 当前 provider 不是 seedance，Wan2.2 本地不支持接力；按普通模式执行", flush=True)
            chain_mode = False
    else:
        if video_provider != "seedance":
            return {
                "status": "error",
                "error": f"direct route is not implemented for provider {video_provider}",
            }
        print("  → 路由: Seedance 直连 ARK Agent Plan", flush=True)

    from runtime.artifact_manifest import ArtifactManifestStore
    from runtime.generation_tasks import GenerationTaskStore

    generation_tasks = GenerationTaskStore(output_dir / "runtime.db")
    artifact_store = ArtifactManifestStore.from_run_directory(
        output_dir,
        required=False,
    )

    # Load character reference images for consistency
    import base64 as _b64
    char_ref_map = {}   # {match_key_lower: base64_of_front_png}
    char_list = []      # [(char_id, char_name, b64)] for fallback
    chars_path = output_dir / "CHARACTERS.json"
    chars_data = {"characters": []}
    declared_character_ids = set()
    missing_character_fronts = set()
    if chars_path.exists():
        chars_data = json.loads(chars_path.read_text())
        for char in chars_data.get("characters", []):
            declared_character_ids.add(char["id"])
            # Try both directory structures: characters/{id}/ and characters/characters/{id}/
            reference_path = None
            for char_dir in (
                output_dir / "characters" / char["id"],
                output_dir / "characters" / "characters" / char["id"],
            ):
                candidates = [
                    char_dir / "face_closeup.png",
                    char_dir / "full_body.png",
                    *sorted(char_dir.glob("variant_*.png")),
                    char_dir / "front.png",  # legacy fallback
                ]
                reference_path = next((path for path in candidates if path.exists()), None)
                if reference_path is not None:
                    break
            if reference_path is not None:
                b64 = _b64.b64encode(reference_path.read_bytes()).decode()
                # Canonical id, display name, and declared aliases are the only
                # supported identity lookup keys.
                for match_key in (
                    char["name"],
                    char["id"],
                    char["id"].replace("_", ""),
                    *char.get("aliases", []),
                ):
                    if match_key:
                        char_ref_map[str(match_key).casefold()] = b64
                char_list.append((char["id"], char["name"], b64))
            else:
                missing_character_fronts.add(char["id"])
        if char_ref_map:
            print(f"  → 已加载 {len(char_list)} 个角色参考图")
        if missing_character_fronts:
            print(
                "  ⚠ Phase 6 前置检查: 缺少角色参考图 "
                "(face_closeup.png/full_body.png/variant_*.png): "
                + ", ".join(sorted(missing_character_fronts)),
                flush=True,
            )

    outputs = []
    successful_receipts: dict[str, dict[str, Any]] = {}
    # --- P1-C: Seed Locking（参考 HonCut asset_manifest seed）---
    # 同场景镜头使用相同 seed，确保背景一致性
    scene_seed_map = {}  # {where: seed}
    prev_shot_dir = None  # --- P1-D2: 上一镜头视频作为运动参考 ---
    scene_consistency_path = output_dir / "SCENE_CONSISTENCY.json"
    scene_consistency_data = (
        json.loads(scene_consistency_path.read_text(encoding="utf-8"))
        if scene_consistency_path.exists() else {}
    )

    # --- 并发配置 ---
    try:
        from utils.config import VIDEO_GEN_CONCURRENCY
        concurrency = VIDEO_GEN_CONCURRENCY
    except ImportError:
        concurrency = int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1"))
    provider_capacity = max(1, concurrency)
    provider_policy = ProviderExecutionPolicy.from_environment(
        "bridge" if use_local else "seedance"
    )
    provider_slots = None
    provider_leases = None
    if not use_local:
        from runtime.capacity import (
            CrossProcessSlotTable,
            SlotTable,
            default_capacity_lease_path,
        )

        provider_slots = SlotTable()
        provider_leases = CrossProcessSlotTable(default_capacity_lease_path())
        if not chain_mode:
            provider_capacity = provider_policy.capacity(provider_capacity)
    if chain_mode:
        concurrency = 1
        provider_capacity = 1
        print("  [chain] Seedance 尾帧接力已启用，强制按 shot_id 串行生成", flush=True)
    elif not use_local:
        concurrency = provider_capacity
    else:
        concurrency = max(1, concurrency)
    capacity_source = (
        "seedance video capacity" if not use_local else "VIDEO_GEN_CONCURRENCY"
    )
    print(
        f"  → 并发模式: {capacity_source}={concurrency} "
        f"({'串行' if concurrency == 1 else f'并行 workers={concurrency}'})"
    )

    shot_dirs = [d for d in sorted(shots_dir.iterdir()) if d.is_dir() and d.name.startswith("S")]
    # Every shot enters its durable execution boundary. Reuse is allowed only
    # by an exact succeeded ledger entry plus hash and ffprobe validation.
    pending_shot_dirs = list(shot_dirs)

    try:
        from utils.config import SEEDANCE_MODEL

        route_model_name = SEEDANCE_MODEL
    except ImportError:
        route_model_name = os.environ.get(
            "SEEDANCE_MODEL", "doubao-seedance-2.0-mini"
        )
    video_prompt_model = os.environ.get("VIDEO_MODEL", "seedance")

    task_dir_id = None
    if use_local and os.environ.get("HONCUT_TASK_DIR_MODE") == "1" and pending_shot_dirs:
        from tools import task_dir_exporter

        export_shots = {}
        for pending_dir in pending_shot_dirs:
            pending_meta_path = pending_dir / "SHOT_META.json"
            if not pending_meta_path.exists():
                raise FileNotFoundError(f"Missing shot metadata for task export: {pending_meta_path}")
            pending_meta = json.loads(pending_meta_path.read_text(encoding="utf-8"))
            pending_meta["_char_ids"] = sorted({
                asset_id[5:].split(":", 1)[0]
                for asset_id in pending_meta.get("associate_assets", [])
                if isinstance(asset_id, str) and asset_id.startswith("char:")
            })
            export_prompt, _route_applied = _prepare_phase6_prompt(
                pending_dir.name,
                pending_meta,
                chars_data,
                scene_consistency_data,
                video_model=video_prompt_model,
                route_model=route_model_name,
            )
            pending_meta["prompt"] = export_prompt
            export_shots[pending_dir.name] = pending_meta
        local_task_dir = task_dir_exporter.build_task_dir(
            output_dir,
            [directory.name for directory in pending_shot_dirs],
            {
                "shots": export_shots,
                "chain_mode": chain_mode,
                "model": os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2.0-mini"),
                "resolution": "720p",
            },
        )
        task_dir_id = task_dir_exporter.upload_task_dir(local_task_dir, "tasks")
        print(f"  [task_dir] uploaded tasks/{task_dir_id}", flush=True)
    
    def _process_shot(
        shot_dir: Path,
        chain_source: Optional[tuple[str, Path]] = None,
        chain_allowed: bool = True,
    ) -> Optional[dict]:
        """处理单个镜头的视频生成，返回 output.mp4 路径或 None"""
        meta_path = shot_dir / "SHOT_META.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        active_source = chain_source if chain_mode and chain_allowed else None
        meta["chain_source"] = active_source[0] if active_source else None
        meta["chain_active"] = bool(active_source)
        prompt, route_applied = _prepare_phase6_prompt(
            shot_dir.name,
            meta,
            chars_data,
            scene_consistency_data,
            video_model=video_prompt_model,
            route_model=route_model_name,
        )
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        gen_strategy = meta.get("gen_strategy", "i2v")
        if gen_strategy not in {"flf2v", "phantom", "i2v"}:
            gen_strategy = "i2v"
        route_reason = {
            "flf2v": "action shot",
            "phantom": "dialogue/emotion shot",
            "i2v": "scenery/ambient or default",
        }[gen_strategy]
        if video_route == "bridge" and video_provider == "seedance":
            bridge_model = "seedance"
        elif video_provider in {"wan", "wan22", "local"}:
            bridge_model = "wan22"
        else:
            bridge_model = {"flf2v": "flf2v", "phantom": "phantom", "i2v": "wan22"}[gen_strategy]
        print(f"    [route] {shot_dir.name} → {gen_strategy} ({route_reason})")
        associated_character_ids = {
            asset_id[5:].split(":", 1)[0]
            for asset_id in meta.get("associate_assets", [])
            if isinstance(asset_id, str) and asset_id.startswith("char:")
        }
        missing_for_shot = associated_character_ids & missing_character_fronts
        if missing_for_shot or (declared_character_ids and not char_list):
            missing_ids = missing_for_shot or missing_character_fronts
            print(
                f"    ✗ {shot_dir.name}: 缺少角色参考图 "
                "characters/*/{face_closeup.png,full_body.png,variant_*.png} "
                f"({', '.join(sorted(missing_ids))})，跳过镜头",
                flush=True,
            )
            return None
        if route_applied:
            print(f"    [M4] 提示词路由: {route_model_name} → single_shot")
        duration = meta.get("duration")  # 从 SHOT_META 读取；缺失时由模型 profile 选中间档
        if not prompt:
            return None

        # Find character reference for this shot
        # Strategy: match by name/id in prompt → fallback to protagonist
        first_frame_b64 = None
        prompt_lower = prompt.lower()

        # Canonical visual reference is the one-image-per-shot artifact.
        # Never inject storyboard.png: it is a multi-panel overview sheet.
        shot_image = _shot_storyboard_reference(output_dir, shot_dir.name)
        if shot_image is not None:
            first_frame_b64 = _b64.b64encode(shot_image.read_bytes()).decode()
            print(f"    [M2] 注入逐镜分镜图: {shot_image.name}")

        # --- P0-C: HonCut 资产ID绑定匹配（associateAssetsIds）---
        # --- P1-A4: 衍生参考图匹配（char:id:state → variant_state.png）---
        associate_assets = meta.get("associate_assets", [])
        if associate_assets and first_frame_b64 is None:
            for asset_id in associate_assets:
                if asset_id.startswith("char:"):
                    parts = asset_id[5:].split(":")
                    char_id = parts[0]
                    variant_state = parts[1] if len(parts) > 1 else None

                    if variant_state:
                        # 衍生参考图匹配（P1-A4）
                        variant_png = output_dir / "characters" / char_id / f"variant_{variant_state}.png"
                        if not variant_png.exists():
                            variant_png = output_dir / "characters" / "characters" / char_id / f"variant_{variant_state}.png"
                        if variant_png.exists():
                            first_frame_b64 = _b64.b64encode(variant_png.read_bytes()).decode()
                            print(f"    [P1-A] 衍生参考图匹配: {char_id}:{variant_state}")
                            break

                    # 基准参考图匹配（新资产优先，旧 front.png 仅兼容）
                    reference_path = None
                    for char_dir in (
                        output_dir / "characters" / char_id,
                        output_dir / "characters" / "characters" / char_id,
                    ):
                        candidates = [
                            char_dir / "face_closeup.png",
                            char_dir / "full_body.png",
                            *sorted(char_dir.glob("variant_*.png")),
                            char_dir / "front.png",
                        ]
                        reference_path = next(
                            (path for path in candidates if path.exists()), None
                        )
                        if reference_path is not None:
                            break
                    if reference_path is not None:
                        first_frame_b64 = _b64.b64encode(reference_path.read_bytes()).decode()
                        print(f"    [P0-C] 资产绑定匹配角色: {char_id}")
                        break

        # Strategy: match by name/id in prompt only when no canonical shot
        # frame was found. Global style text may mention a protagonist even for
        # explicit who=[] scenery shots; it must never replace Sxx.png.
        if first_frame_b64 is None:
            for char_name, b64 in char_ref_map.items():
                if char_name in prompt_lower:
                    first_frame_b64 = b64
                    print(f"    [ref] 注入角色参考: {char_name}")
                    break
        # Structured cast fallback. Never infer an identity from arbitrary
        # natural-language substrings or inject the first project character.
        shot_who = meta.get("who") or meta.get("characters") or []
        if not isinstance(shot_who, list):
            shot_who = [shot_who] if shot_who else []
        if first_frame_b64 is None:
            for cast_key in shot_who:
                cast_reference = char_ref_map.get(str(cast_key).casefold())
                if cast_reference:
                    first_frame_b64 = cast_reference
                    print(f"    [ref] 按结构化 who 注入角色参考: {cast_key}")
                    break

        # --- P0-A3: 场景参考图（逐镜图/角色参考缺失时使用）---
        if first_frame_b64 is None:
            shot_where = meta.get("where", "")
            if shot_where:
                scene_id = shot_where.replace(" ", "_").replace("/", "_")[:30]
                scene_ref = output_dir / "scenes" / scene_id / "reference.png"
                if scene_ref.exists() and scene_ref.stat().st_size > 1024:
                    first_frame_b64 = _b64.b64encode(scene_ref.read_bytes()).decode()
                    print(f"    [P0-A] 注入场景参考图: {scene_id}")

        if active_source:
            try:
                first_frame_b64 = _b64.b64encode(active_source[1].read_bytes()).decode()
                print(f"    [chain] {shot_dir.name}: 首帧接力自 {active_source[0]}", flush=True)
            except Exception as error:
                print(f"    [chain] {shot_dir.name}: 无法读取 {active_source[0]} 尾帧，回退独立首帧 — {error}", flush=True)
                active_source = None
                meta["chain_source"] = None
                meta["chain_active"] = False
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # --- P1-C2: 同场景同 seed ---
        shot_where = meta.get("where", "")
        shot_seed = None
        if shot_where:
            if shot_where not in scene_seed_map:
                import hashlib
                scene_seed_map[shot_where] = int(hashlib.md5(shot_where.encode()).hexdigest()[:8], 16) % 2147483647
            shot_seed = scene_seed_map[shot_where]

        # --- P1-D2: 上一镜头视频作为运动参考（可选，仅串行模式）---
        prev_video_ref = None
        if concurrency == 1 and prev_shot_dir is not None:
            prev_output = prev_shot_dir / "output.mp4"
            if prev_output.exists() and prev_output.stat().st_size > 10240:
                try:
                    prev_video_ref = _b64.b64encode(prev_output.read_bytes()).decode()
                    # 视频参考大小限制（>5MB 不传，避免超限）
                    if len(prev_video_ref) > 5 * 1024 * 1024 * 4 // 3:
                        prev_video_ref = None
                except Exception as error:
                    print(
                        f"  ⚠ {shot_dir.name}: 无法读取上一镜头运动参考 — {error}",
                        flush=True,
                    )

        max_policy_repairs = 3
        privacy_retries = 0
        policy_retries = 0
        privacy_rejected_urls: set[str] = set()
        privacy_retry_strategy = gen_strategy
        content_list = None
        # Retry budgets are failure-class specific. A temporary quota burst
        # must not consume the later privacy-fallback opportunity (or vice
        # versa), otherwise the first different error after backoff becomes a
        # false terminal failure.
        while True:
            try:
                out_path = str(shot_dir / "output.mp4")
                
                # --- 本地 API 路由 ---
                if use_local:
                    try:
                        print(f"  → {shot_dir.name}: 提交本地 API 视频生成...")
                        from clients import local_video_client
                        from tools import asset_packager
                        
                        # [LEGACY-KEEP v2.0] Build content[] for Windows Bridges not yet on task_dir.
                        shot_id = shot_dir.name  # e.g., "S01"
                        content_meta = dict(meta)
                        content_meta["prompt"] = prompt
                        content_meta["gen_strategy"] = gen_strategy
                        content_meta["_char_ids"] = sorted(associated_character_ids)
                        zip_path = None
                        base64_list = []
                        content_list = None
                        if task_dir_id is None:
                            content_list = asset_packager.build_content_for_shot(
                                output_dir=output_dir,
                                shot_id=shot_id,
                                shot_meta=content_meta,
                            )
                            if active_source and first_frame_b64:
                                content_list = _apply_chain_relay(
                                    content_list, first_frame_b64, shot_id
                                )

                            # [LEGACY-KEEP v2.0] zip/base64 fallback for old Bridges.
                            if not content_list or len(content_list) <= 1:
                                zip_path, base64_list = asset_packager.package_shot_assets(
                                    output_dir=output_dir,
                                    shot_id=shot_id,
                                    shot_meta=meta,
                                )
                                content_list = None

                        generate = (
                            local_video_client.generate_video_with_fallback
                            if bridge_model == "seedance"
                            else local_video_client.generate_video
                        )
                        from functools import partial

                        from runtime.bridge_execution import execute_bridge_video_task

                        from utils.video_geometry import resolve_video_geometry
                        from utils.video_validation import is_valid_video

                        aspect_ratio, video_width, video_height = resolve_video_geometry(meta)
                        generation_fingerprint = _generation_fingerprint(
                            output_dir=output_dir,
                            shot_id=shot_id,
                            meta=content_meta,
                            prompt=prompt,
                            first_frame_b64=first_frame_b64,
                            content=content_list,
                            chain_source=active_source,
                            provider_id="bridge",
                            provider_version="bridge-api-v1",
                            model_id=bridge_model,
                            model_version=bridge_model,
                            generation_parameters={
                                "duration": duration,
                                "seed": shot_seed if shot_seed is not None else -1,
                                "ratio": aspect_ratio,
                                "width": video_width,
                                "height": video_height,
                                "task_dir": task_dir_id,
                            },
                        )
                        bridge_generate = partial(
                            generate,
                            prompt=prompt,
                            output_path=out_path,
                            reference_image_base64=first_frame_b64,
                            seed=shot_seed if shot_seed is not None else -1,
                            duration=duration,
                            width=video_width,
                            height=video_height,
                            fps=24,
                            asset_zip_path=zip_path,
                            image_base64_list=base64_list,
                            content=content_list,
                            batch_id=output_dir.name,
                            model=bridge_model,
                            submit_timeout=int(
                                provider_policy.submit_timeout_seconds
                            ),
                            status_timeout=provider_policy.status_timeout_seconds,
                            poll_deadline=provider_policy.poll_deadline_seconds,
                            return_last_frame=chain_mode and chain_allowed,
                            task_dir=task_dir_id,
                        )
                        execution = execute_bridge_video_task(
                            generation_tasks,
                            run_id=str(output_dir.resolve()),
                            resource_id=shot_id,
                            payload={
                                "shot_id": shot_id,
                                "output_path": f"shots/{shot_id}/output.mp4",
                                "model": bridge_model,
                                "duration": duration,
                                "seed": shot_seed if shot_seed is not None else -1,
                                "task_dir": task_dir_id,
                                "ratio": aspect_ratio,
                                "width": video_width,
                                "height": video_height,
                                **generation_fingerprint.task_metadata(),
                            },
                            provider_endpoint=local_video_client.get_api_url(),
                            output_path=out_path,
                            generate=bridge_generate,
                            validate_output=is_valid_video,
                            artifact_store=artifact_store,
                        )
                        generation_result = execution.generation_result

                        if shot_seed is not None:
                            meta["seed"] = shot_seed
                            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                        print(f"    ✓ {shot_dir.name}: 视频已生成 (本地 API)")
                        if isinstance(generation_result, str):
                            generation_result = {
                                "output_path": generation_result,
                                "last_frame_path": None,
                                "actual_model": bridge_model,
                            }
                        generation_result["generation_task_id"] = execution.task_id
                        generation_result["input_fingerprint"] = (
                            generation_fingerprint.value
                        )
                        generation_result["relative_output"] = f"shots/{shot_dir.name}/output.mp4"
                        return generation_result
                    except Exception as local_err:
                        print(f"    ✗ {shot_dir.name}: 本地 API 失败 — {local_err}")
                        print("    ⚠ 不降级到 ARK（零成本测试模式），跳过此镜头")
                        return None

                print(f"  → {shot_dir.name}: 提交 ARK Agent Plan 视频生成...")
                from clients import seedance_client
                from tools import asset_packager

                shot_id = shot_dir.name
                content_meta = dict(meta)
                content_meta["prompt"] = prompt
                content_meta["gen_strategy"] = privacy_retry_strategy
                content_meta["_char_ids"] = sorted(associated_character_ids)
                content_list = asset_packager.build_content_for_shot(
                    output_dir=output_dir,
                    shot_id=shot_id,
                    shot_meta=content_meta,
                )
                content_list = _without_rejected_privacy_images(
                    content_list, privacy_rejected_urls
                )
                if active_source and first_frame_b64:
                    content_list = _apply_chain_relay(
                        content_list, first_frame_b64, shot_id
                    )

                api_key = get_api_key("ARK_AGENT")
                if not api_key:
                    raise RuntimeError("缺少 ARK_AGENT_API_KEY；检查 ARK_AGENT_API_KEY 或 Agent Plan 权限")
                try:
                    from utils.config import SEEDANCE_MODEL
                    direct_model = SEEDANCE_MODEL
                except ImportError:
                    direct_model = os.environ.get(
                        "SEEDANCE_MODEL", "doubao-seedance-2.0-mini"
                    )
                from functools import partial

                from runtime.seedance_execution import execute_seedance_video_task

                from utils.video_geometry import resolve_video_geometry
                from utils.video_validation import is_valid_video

                aspect_ratio, video_width, video_height = resolve_video_geometry(meta)
                generation_fingerprint = _generation_fingerprint(
                    output_dir=output_dir,
                    shot_id=shot_id,
                    meta=content_meta,
                    prompt=prompt,
                    first_frame_b64=first_frame_b64,
                    content=content_list,
                    chain_source=active_source,
                    provider_id="seedance",
                    provider_version="ark-agent-plan-v3",
                    model_id=direct_model,
                    model_version=direct_model,
                    generation_parameters={
                        "duration": duration or 12,
                        "seed": shot_seed,
                        "ratio": aspect_ratio,
                        "width": video_width,
                        "height": video_height,
                    },
                )
                execution = execute_seedance_video_task(
                    generation_tasks,
                    run_id=str(output_dir.resolve()),
                    resource_id=shot_id,
                    payload={
                        "shot_id": shot_id,
                        "output_path": f"shots/{shot_id}/output.mp4",
                        "model": direct_model,
                        "duration": duration or 12,
                        "seed": shot_seed,
                        "ratio": aspect_ratio,
                        "width": video_width,
                        "height": video_height,
                        **generation_fingerprint.task_metadata(),
                    },
                    provider_endpoint=seedance_client.BASE_URL,
                    output_path=out_path,
                    submit=partial(
                        provider_policy.execute_rate_limited,
                        partial(
                            seedance_client.submit_content,
                            content_list,
                            api_key=api_key,
                            model=direct_model,
                            duration=duration or 12,
                            ratio=aspect_ratio,
                            seed=shot_seed,
                            timeout=provider_policy.submit_timeout_seconds,
                        ),
                    ),
                    poll=provider_policy.bind_poll(
                        seedance_client.poll,
                        interval_seconds=15,
                        api_key=api_key,
                    ),
                    download=seedance_client.download,
                    validate_output=is_valid_video,
                    artifact_store=artifact_store,
                )
                task_id = execution.provider_job_id

                last_frame_path = shot_dir / "last_frame.jpg"
                if chain_mode and chain_allowed:
                    try:
                        subprocess.run(
                            [
                                "ffmpeg", "-sseof", "-0.1", "-i", out_path,
                                "-frames:v", "1", "-q:v", "1",
                                str(last_frame_path), "-y",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except Exception as frame_error:
                        print(
                            f"    ⚠ {shot_dir.name}: 尾帧提取失败 — {frame_error}",
                            flush=True,
                        )

                actual_duration = None
                try:
                    probe = subprocess.run(
                        [
                            "ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", out_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    actual_duration = float(probe.stdout.strip())
                except Exception:
                    pass

                meta.update({
                    "task_id": task_id,
                    "status": "completed",
                    "video_path": out_path,
                    "actual_model": direct_model,
                    "actual_duration": actual_duration,
                    "ratio": aspect_ratio,
                    "width": video_width,
                    "height": video_height,
                })
                if shot_seed is not None:
                    meta["seed"] = shot_seed
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                return {
                    "output_path": out_path,
                    "last_frame_path": (
                        str(last_frame_path) if last_frame_path.exists() else None
                    ),
                    "actual_model": direct_model,
                    "generation_task_id": execution.task_id,
                    "input_fingerprint": generation_fingerprint.value,
                    "relative_output": f"shots/{shot_dir.name}/output.mp4",
                }
            except Exception as e:
                err_str = str(e)
                # Transport quota retries are exhausted inside Runtime policy.
                if "QuotaExceeded" in err_str or "429" in err_str:
                    print(
                        f"    ✗ {shot_dir.name}: 配额超限，Runtime 策略已耗尽，跳过"
                    )
                    return None
                # Check auth only after classifying quota errors. Provider
                # request ids are arbitrary hexadecimal-ish strings and can
                # contain the substring "401" or "403" inside a genuine 429.
                if "401" in err_str or "403" in err_str:
                    raise RuntimeError(
                        f"{err_str}；检查 ARK_AGENT_API_KEY 或 Agent Plan 权限"
                    ) from e
                if "PrivacyInformation" in err_str and privacy_retries < max_policy_repairs:
                    privacy_retries += 1
                    fallback_strategy = _privacy_fallback_strategy(
                        privacy_retry_strategy
                    )
                    if fallback_strategy != privacy_retry_strategy:
                        privacy_retry_strategy = fallback_strategy
                        privacy_rejected_urls.clear()
                        print(
                            f"    ⚠ {shot_dir.name}: FLF2V 首尾帧有图片被隐私检测拒绝，"
                            "整组切换为 Phantom 身份参考模式后重试"
                        )
                        continue
                    rejected_url = _rejected_privacy_image_url(content_list, e)
                    if rejected_url:
                        privacy_rejected_urls.add(rejected_url)
                        print(
                            f"    ⚠ {shot_dir.name}: 参考图被隐私检测拒绝，"
                            "仅剔除被拒图片并保留其余安全参考后重试"
                        )
                        continue
                    print(
                        f"    ⚠ {shot_dir.name}: 无法定位被拒参考图，"
                        "停止自动重提以避免重复无效请求"
                    )
                if "PolicyViolation" in err_str and policy_retries < max_policy_repairs:
                    policy_retries += 1
                    print(f"    ⚠ {shot_dir.name}: 版权误报，重试 ({policy_retries}/{max_policy_repairs})...")
                    prompt = prompt.replace("Cinematic", "Original fictional")
                    prompt += ", original character design, non-copyrighted"
                    first_frame_b64 = None
                    continue
                print(f"    ✗ {shot_dir.name}: 异常 — {e}")
                return None

        return None

    def _process_shot_with_capacity(
        shot_dir: Path,
        chain_source: tuple[str, Path] | None = None,
        chain_allowed: bool = True,
    ) -> dict | None:
        if provider_slots is None or provider_leases is None:
            return _process_shot(shot_dir, chain_source, chain_allowed)
        with provider_slots.reserve(
            "seedance",
            "video",
            shot_dir.name,
            capacity=provider_capacity,
        ):
            lease_task_id = f"{output_dir.resolve()}:{shot_dir.name}"
            with provider_leases.reserve(
                "seedance",
                "video",
                lease_task_id,
                capacity=provider_capacity,
            ):
                return _process_shot(shot_dir, chain_source, chain_allowed)

    # --- 执行模式：串行或并发 ---
    if concurrency == 1:
        # 串行模式（默认，保持原有逻辑和状态更新）
        chain_source = None
        chain_allowed = True
        for shot_dir in pending_shot_dirs:
            result = _process_shot_with_capacity(
                shot_dir, chain_source, chain_allowed
            )
            if result:
                outputs.append(result["relative_output"])
                successful_receipts[shot_dir.name] = result
            if chain_mode:
                actual_model = result.get("actual_model") if result else None
                if actual_model == "wan22":
                    print(f"    [chain] {shot_dir.name}: 降级 Wan2.2，接力链中断，后续镜头回退独立首帧", flush=True)
                    chain_allowed = False
                    chain_source = None
                else:
                    last_frame_path = result.get("last_frame_path") if result else None
                    chain_source = (
                        (shot_dir.name, Path(last_frame_path))
                        if last_frame_path and Path(last_frame_path).exists()
                        else None
                    )
            prev_shot_dir = shot_dir
    else:
        # 并发模式（VIDEO_GEN_CONCURRENCY > 1）
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_shot_with_capacity, shot_dir): shot_dir
                for shot_dir in pending_shot_dirs
            }
            for future in as_completed(futures):
                shot_dir = futures[future]
                try:
                    result = future.result()
                    if result:
                        outputs.append(result["relative_output"])
                        successful_receipts[shot_dir.name] = result
                except Exception as e:
                    print(f"    ✗ {shot_dir.name}: 并发处理异常 — {e}")

    provider = "local_video_client" if use_local else "seedance_client"

    # A live file is not proof that this run produced it. Accept a shot only
    # when the current execution returned a task receipt whose immutable input,
    # ledger status, output hash, and decoded video all still match.
    errors = []
    missing_shots = []
    for sd in shot_dirs:
        out_mp4 = sd / "output.mp4"
        receipt = successful_receipts.get(sd.name)
        task = generation_tasks.get(str(receipt.get("generation_task_id"))) if receipt else None
        failure = _phase6_output_failure(sd.name, out_mp4, receipt, task)
        if failure:
            missing_shots.append(sd.name)
            errors.append({"shot": sd.name, "error": failure})
    if missing_shots:
        print(f"  ⚠ Phase 6 部分镜头无产出: {', '.join(missing_shots)}")

    return {
        "status": "error" if missing_shots or not outputs else "done",
        "outputs": outputs,
        "errors": errors,
        "missing_shots": missing_shots,
        "error": (
            "Phase 6 missing required shot outputs: " + ", ".join(missing_shots)
            if missing_shots
            else "Phase 6 produced no videos"
            if not outputs
            else None
        ),
        "provider": provider,
        "mode": "text_to_video",
    }
