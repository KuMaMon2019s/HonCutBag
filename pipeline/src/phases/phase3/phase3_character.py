"""Phase 3 character asset generation and QA."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from phases.phase2.storyboard_assets import _validate_storyboard_image_composition
from quality.quality_gate import run_quality_check
from runtime.phase_estimates import (
    build_pipeline_workload,
    estimate_phase_duration,
)
from runtime.phase_timing import _banner, _elapsed, _ensure_dir, _now
from runtime.retry_execution import _retry_with_policy
from utils.storyboard_geometry import _storyboard_canvas, _storyboard_image_size
from utils.style_slices import get_slice

PHASE3_DRY_RUN_RECEIPT_SCHEMA = "honcut.phase3-dry-run-receipt.v1"
PHASE3_DRY_RUN_RECEIPT_NAME = "phase3_dry_run_receipt.json"
PHASE3_DRY_RUN_SKIPPED_OPERATIONS = (
    "character_reference_image_generation",
    "character_reference_semantic_qa",
    "character_performance_board_generation",
    "character_locked_storyboard_refresh",
)


def _configured_character_registry(output_dir: Path, *, dry_run: bool):
    """Resolve the explicit project library without consulting ambient state."""
    if dry_run:
        return None, None
    manifest_path = output_dir / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved = manifest["resolved_config"]
        project_id = resolved["project_id"]
        configured = resolved.get("character_library_dir")
        run_id = manifest["run_fingerprint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("RUN_MANIFEST.json has invalid character-library identity") from error
    if not configured:
        return None, None
    library_root = Path(str(configured)).expanduser().resolve()
    run_root = output_dir.resolve()
    if library_root == run_root or library_root in run_root.parents or run_root in library_root.parents:
        raise RuntimeError(
            "character library must be outside and non-overlapping with the run directory"
        )
    from runtime.character_registry import CharacterRegistry

    return CharacterRegistry(library_root, project_id=str(project_id)), str(run_id)


def _write_character_registry_receipt(
    output_dir: Path,
    *,
    project_id: str,
    entries: list[dict],
) -> Path:
    """Persist a run-local audit of exact reuse and canonical promotion."""
    from runtime.character_registry import CHARACTER_REGISTRY_RECEIPT_SCHEMA

    payload = {
        "schema": CHARACTER_REGISTRY_RECEIPT_SCHEMA,
        "status": "completed",
        "project_id": project_id,
        "registry_provider_requests": 0,
        "characters": entries,
    }
    path = output_dir / "character_registry_receipt.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _hashed_artifact(root: Path, path: Path) -> dict | None:
    """Describe one existing run-local artifact without embedding its content."""
    if not path.is_file():
        return None
    from utils.file_integrity import file_sha256

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def _write_phase3_dry_run_receipt(
    output_dir: Path,
    characters: list[dict],
    required_reference_views: tuple[str, ...],
) -> Path:
    """Atomically record what Phase 3 validated and intentionally did not run."""
    source_artifacts = [
        artifact
        for name in ("CHARACTERS.json", "STORYBOARD.json", "visual-style.md")
        if (artifact := _hashed_artifact(output_dir, output_dir / name)) is not None
    ]
    character_cards = []
    character_ids = []
    for index, character in enumerate(characters):
        character_id = str(character.get("id") or f"char_{index}")
        character_ids.append(character_id)
        card = _hashed_artifact(
            output_dir,
            output_dir / "characters" / character_id / "character_card.json",
        )
        if card is not None:
            character_cards.append({"character_id": character_id, **card})

    payload = {
        "schema": PHASE3_DRY_RUN_RECEIPT_SCHEMA,
        "status": "completed",
        "dry_run": True,
        "character_ids": character_ids,
        "required_reference_views": list(required_reference_views),
        "source_artifacts": source_artifacts,
        "character_cards": character_cards,
        "skipped_operations": list(PHASE3_DRY_RUN_SKIPPED_OPERATIONS),
    }
    receipt_path = output_dir / PHASE3_DRY_RUN_RECEIPT_NAME
    temporary = receipt_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt_path


def detect_derive_assets(characters_data: dict) -> list:
    """Frozen test-facade compatibility symbol; production variants are retired.

    Runtime appearance changes belong to the Pxx continuity contract.  The
    compatibility facade still imports this name, so keep a side-effect-free
    boundary until that facade can be versioned independently.
    """
    del characters_data
    return []


def run_phase3(
    output_dir: Path,
    characters_data: dict,
    dry_run: bool,
    *,
    _acceptance_character_limit: int | None = None,
    _acceptance_max_new_image_requests: int | None = None,
    _acceptance_disable_provider_retries: bool = False,
    _acceptance_before_provider_request=None,
) -> dict:
    """Phase 3: generate canonical character identity and performance assets."""
    _banner(3, 9, "角色工厂 (Character Factory)", dry_run)
    start = _now()
    outputs = []
    output_dir = Path(output_dir)

    try:
        from phases.phase3.character_factory import (
            CharacterReferenceGenerationPaused,
            batch_generate,
        )
        from quality.character_reference_qa import (
            SEEDANCE_REFERENCE_VIEWS,
            CharacterReferenceQAError,
        )

        chars_dir = _ensure_dir(output_dir / "characters")
        characters_list = characters_data.get("characters", [])

        if (
            _acceptance_character_limit is not None
            and (
                isinstance(_acceptance_character_limit, bool)
                or not isinstance(_acceptance_character_limit, int)
                or _acceptance_character_limit < 1
            )
        ):
            raise ValueError(
                "_acceptance_character_limit must be a positive integer or None"
            )
        if (
            _acceptance_max_new_image_requests is not None
            and (
                isinstance(_acceptance_max_new_image_requests, bool)
                or not isinstance(_acceptance_max_new_image_requests, int)
                or _acceptance_max_new_image_requests < 1
            )
        ):
            raise ValueError(
                "_acceptance_max_new_image_requests must be a positive "
                "integer or None"
            )

        if not characters_list:
            print("  ⊘ 无角色数据，跳过")
            return {"status": "skipped", "reason": "no characters", "duration_s": _elapsed(start)}

        if not dry_run:
            from utils.canonical_visual_contracts import (
                load_canonical_visual_contract,
            )

            load_canonical_visual_contract(
                output_dir,
                characters_data=characters_data,
            )

        # Step 3.2: 生成基础角色四视图
        visual_style_path = output_dir / "visual-style.md"
        character_style = ""
        if visual_style_path.is_file():
            character_style = get_slice(
                visual_style_path.read_text(encoding="utf-8"), "character"
            )
        # 为每个角色准备静态身份描述。剧情动作、姿势、镜头互动和场景风格
        # 不得进入四视图，否则会把所有 canonical references 污染成剧照。
        from utils.character_body_contracts import (
            character_reference_identity_description,
        )

        char_dicts = []
        for i, c in enumerate(characters_list):
            char_dicts.append({
                "id": c.get("id", f"char_{i}"),
                "name": c.get("name", f"角色{i}"),
                "description": character_reference_identity_description(c),
                "appearance": c.get("appearance", {}),  # 传递完整 appearance dict
                "visual_identity_policy": c.get("visual_identity_policy"),
                "style": "\n\n".join(
                    part for part in (c.get("style", ""), character_style) if part
                ),
                "negative": ", ".join(filter(None, (
                    str(c.get("negative", "")).strip(),
                    str(c.get("negative_guardrails", "")).strip(),
                ))),
            })

        registry, source_run_id = _configured_character_registry(
            output_dir,
            dry_run=dry_run,
        )
        registry_entries: list[dict] = []
        reused_results: list[dict] = []
        generation_queue: list[dict] = []
        if registry is not None:
            for char_dict in char_dicts:
                approved = registry.find_exact(char_dict)
                if approved is None:
                    generation_queue.append(char_dict)
                    continue
                registry.import_into_run(approved, output_dir)
                reused_results.append({
                    "char_id": char_dict["id"],
                    "name": char_dict["name"],
                    "char_dir": str(output_dir / "characters" / char_dict["id"]),
                    "reused": True,
                    "version_id": approved.version_id,
                })
                registry_entries.append({
                    "character_id": char_dict["id"],
                    "action": "reused",
                    "version_id": approved.version_id,
                    "spec_fingerprint": approved.spec_fingerprint,
                    "asset_count": len(approved.assets),
                })
                print(
                    f"  ♻ {char_dict['name']}: reused canonical character "
                    f"{approved.version_id[:12]} (zero Provider requests)"
                )
        else:
            generation_queue = list(char_dicts)

        workload_storyboard_path = output_dir / "STORYBOARD.json"
        workload_storyboard = (
            json.loads(workload_storyboard_path.read_text(encoding="utf-8"))
            if workload_storyboard_path.is_file()
            else {"shots": []}
        )
        phase3_workload = build_pipeline_workload(
            characters_data,
            workload_storyboard,
            output_dir=output_dir,
        )
        _p3_est = estimate_phase_duration(
            "phase3",
            image_requests=phase3_workload.phase3_image_requests,
        )
        print(
            "  ⏱ Phase 3 开始 "
            f"(最多 {phase3_workload.phase3_image_requests} 次图片请求, "
            f"限流耗时上限 ~{int(_p3_est)}s；缓存命中时更短)"
        )
        print(
            f"  → batch_generate: {len(generation_queue)}/{len(char_dicts)} 个角色, "
            f"skip_images={dry_run}; registry_reused={len(reused_results)}"
        )

        # Use retry policy for each character generation
        results = list(reused_results)
        _p3_char_start = _now()
        acceptance_generation_queue = (
            generation_queue[:_acceptance_character_limit]
            if _acceptance_character_limit is not None
            else generation_queue
        )
        generated_character_ids: list[str] = []
        for i, char_dict in enumerate(acceptance_generation_queue):
            char_name = char_dict.get("name", f"角色{i}")
            print(
                f"    → [{i+1}/{len(acceptance_generation_queue)}] "
                f"{char_name}..."
            )
            _char_t0 = _now()

            def _gen_char(_character=char_dict):
                # Pass output_dir (not chars_dir) — generate_character appends /characters/ internally
                return batch_generate(
                    [_character],
                    str(output_dir),
                    skip_images=dry_run,
                    raise_on_error=True,
                    view_qa_max_retries=(
                        0 if _acceptance_disable_provider_retries else 2
                    ),
                    review_qa_max_retries=(
                        0 if _acceptance_disable_provider_retries else 2
                    ),
                    max_new_image_requests=(
                        _acceptance_max_new_image_requests
                    ),
                    before_provider_request=(
                        _acceptance_before_provider_request
                    ),
                )

            try:
                result = _retry_with_policy(
                    _gen_char,
                    max_attempts=(
                        1 if _acceptance_disable_provider_retries else 3
                    ),
                    backoff_factor=2.0,
                    non_retryable_exceptions=(
                        CharacterReferenceQAError,
                        CharacterReferenceGenerationPaused,
                    ),
                )
                results.extend(result or [])
                generated_character_ids.append(str(char_dict["id"]))
            except CharacterReferenceGenerationPaused as paused:
                if _acceptance_max_new_image_requests is None:
                    raise
                from clients.ark_multimodal_client import ArkMultimodalClient
                from quality.character_reference_qa import (
                    review_character_reference_pack,
                )

                review = review_character_reference_pack(
                    ArkMultimodalClient(),
                    {paused.view_name: paused.view_path},
                    paused.character_description,
                    paused.synthetic_styling,
                    before_provider_request=(
                        _acceptance_before_provider_request
                    ),
                )
                if review.get("passed") is not True or review.get(
                    "qa_verdict"
                ) not in {"pass", "acceptable_deviation"}:
                    raise RuntimeError(
                        "Phase 3 first identity image failed semantic QA"
                    )
                return {
                    "status": "acceptance_gate_passed",
                    "gate": "first_character_identity_image",
                    "character_id": paused.character_id,
                    "view_name": paused.view_name,
                    "view_path": paused.view_path.relative_to(
                        output_dir
                    ).as_posix(),
                    "view_sha256": _hashed_artifact(
                        output_dir,
                        paused.view_path,
                    )["sha256"],
                    "image_provider_request_count": 1,
                    "qa_provider_request_count": 1,
                    "qa_observation_id": review.get(
                        "qa_observation_id"
                    ),
                    "qa_decision_id": review.get("qa_decision_id"),
                    "qa_verdict": review.get("qa_verdict"),
                    "provider_retry_policy": "disabled",
                    "duration_s": _elapsed(start),
                }
            except Exception as e:
                print(f"    ✗ {char_name} 生成失败: {e}")
                if _acceptance_disable_provider_retries:
                    raise
                results.append(None)
            _char_elapsed = round(_now() - _char_t0, 1)
            _char_cumulative = round(_now() - _p3_char_start, 1)
            print(f"  ⏱ {char_name} 完成 (耗时 {_char_elapsed}s, 累计 {_char_cumulative}s / 预估 {int(_p3_est)}s)")

        if _acceptance_character_limit is not None:
            if len(generated_character_ids) != _acceptance_character_limit:
                raise RuntimeError(
                    "Phase 3 acceptance gate did not generate the exact configured "
                    "number of fresh character packs"
                )
            from quality.character_reference_qa import (
                validate_character_reference_qa_receipt,
            )

            result_by_id = {
                str(item.get("char_id") or ""): item
                for item in results
                if isinstance(item, dict)
            }
            character_by_id = {
                str(item.get("id") or ""): item
                for item in characters_list
                if isinstance(item, dict)
            }
            for character_id in generated_character_ids:
                generated = result_by_id.get(character_id)
                character = character_by_id.get(character_id)
                if generated is None or character is None:
                    raise RuntimeError(
                        f"Phase 3 acceptance gate lost result for {character_id}"
                    )
                view_paths = {
                    name: Path(str(path))
                    for name, path in (generated.get("views") or {}).items()
                    if isinstance(path, str) and path
                }
                card_path = Path(str(generated.get("card") or ""))
                try:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"Phase 3 acceptance card is invalid for {character_id}"
                    ) from error
                appearance = character.get("appearance")
                synthetic_styling = (
                    appearance.get("synthetic_styling")
                    if isinstance(appearance, dict)
                    else None
                )
                report_path = (
                    output_dir
                    / "characters"
                    / character_id
                    / "character_reference_qa.json"
                )
                if not validate_character_reference_qa_receipt(
                    report_path,
                    view_paths,
                    synthetic_styling=(
                        synthetic_styling
                        if isinstance(synthetic_styling, dict)
                        else None
                    ),
                    generation_contract=card.get(
                        "reference_generation_contract"
                    ),
                ):
                    raise RuntimeError(
                        f"Phase 3 acceptance QA receipt is invalid for {character_id}"
                    )
            return {
                "status": "acceptance_gate_passed",
                "gate": "first_fresh_character_identity_pack",
                "generated_character_ids": generated_character_ids,
                "new_character_limit": _acceptance_character_limit,
                "provider_retry_policy": (
                    "disabled"
                    if _acceptance_disable_provider_retries
                    else "production_default"
                ),
                "duration_s": _elapsed(start),
            }

        # 统计输出
        for r in (results or []):
            if isinstance(r, dict):
                name = r.get("name", r.get("id", "unknown"))
                outputs.append(f"characters/{name}/")
            elif isinstance(r, str):
                outputs.append(r)

        if not outputs:
            # fallback: 扫描目录
            for d in chars_dir.iterdir():
                if d.is_dir():
                    outputs.append(f"characters/{d.name}/")

        print(f"  ✓ Phase 3 完成: {len(outputs)} 角色卡")

        if dry_run:
            receipt_path = _write_phase3_dry_run_receipt(
                output_dir,
                characters_list,
                SEEDANCE_REFERENCE_VIEWS,
            )
            outputs.append(receipt_path.name)
            print(
                "  ⊘ dry-run: 跳过生产四视图质检与角色锁定 Pxx 刷新；"
                f"凭证写入 {receipt_path.name}"
            )
            return {
                "status": "done",
                "dry_run": True,
                "dry_run_receipt": receipt_path.name,
                "duration_s": _elapsed(start),
                "outputs": outputs,
            }

        # Quality gate: Phase 3 (CRITICAL — blocks pipeline if character images missing)
        qg_report = run_quality_check("phase3", output_dir)
        if not qg_report.passed:
            return {
                "status": "error",
                "error": (
                    f"Phase 3 质检未通过: {qg_report.grade} — 角色四视图缺失、"
                    "语义视角错误或审核凭证已过期，不能继续"
                ),
                "quality_report": qg_report,
                "duration_s": _elapsed(start),
            }

        registry_summary = None
        if registry is not None:
            from runtime.artifact_manifest import ArtifactManifestStore, file_sha256
            from runtime.character_registry import (
                CharacterRegistryError,
                character_has_unapproved_variants,
            )

            reused_ids = {
                entry["character_id"]
                for entry in registry_entries
                if entry.get("action") == "reused"
            }
            for char_dict in char_dicts:
                if char_dict["id"] in reused_ids:
                    continue
                if character_has_unapproved_variants(char_dict):
                    registry_entries.append({
                        "character_id": char_dict["id"],
                        "action": "not_promoted",
                        "reason": "state variants are outside the v1 approval contract",
                    })
                    continue
                try:
                    approved = registry.promote_from_run(
                        output_dir,
                        char_dict,
                        quality_grade=qg_report.grade,
                        source_run_id=source_run_id,
                    )
                except CharacterRegistryError:
                    # A configured library is a correctness boundary. Do not
                    # hide an approval conflict and continue with an ambiguous
                    # canonical identity.
                    raise
                registry_entries.append({
                    "character_id": char_dict["id"],
                    "action": "promoted",
                    "version_id": approved.version_id,
                    "spec_fingerprint": approved.spec_fingerprint,
                    "asset_count": len(approved.assets),
                })
            receipt_path = _write_character_registry_receipt(
                output_dir,
                project_id=registry.project_id,
                entries=registry_entries,
            )
            outputs.append(receipt_path.name)
            receipt_sha256 = file_sha256(receipt_path)
            artifact_store = ArtifactManifestStore.from_run_directory(
                output_dir,
                required=False,
            )
            if artifact_store is not None:
                artifact_store.register_file(
                    receipt_path,
                    artifact_type="character_registry_receipt",
                    producer_node="phase3.character_registry",
                    expected_sha256=receipt_sha256,
                    semantic_fingerprint=receipt_sha256,
                )
            registry_summary = {
                "receipt": receipt_path.name,
                "reused": sum(
                    entry.get("action") == "reused" for entry in registry_entries
                ),
                "generated": len(generation_queue),
                "promoted": sum(
                    entry.get("action") == "promoted" for entry in registry_entries
                ),
                "not_promoted": sum(
                    entry.get("action") == "not_promoted"
                    for entry in registry_entries
                ),
                "registry_provider_requests": 0,
            }

        performance_boards: list[dict] = []

        # Phase 3 owns the first point at which character reference packs are
        # guaranteed to exist. Regenerate the canonical Pxx chain here so the
        # continuity runtime and ordinary Sxx path consume the same identity-
        # locked visual truth.
        storyboard_path = output_dir / "STORYBOARD.json"
        if storyboard_path.is_file():
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            expected_shots = len(storyboard.get("shots", []))
            from phases.phase2.shot_storyboards import (
                generate_shot_storyboards,
                validate_shot_storyboard_artifacts,
            )

            video_width, video_height, aspect_ratio = _storyboard_canvas(storyboard)
            director_reference = storyboard.get("director_storyboard") or {}
            refreshed = generate_shot_storyboards(
                output_dir,
                storyboard,
                characters_list,
                size=_storyboard_image_size(
                    video_width=video_width,
                    video_height=video_height,
                ),
                director_storyboard_path=(
                    director_reference.get("image")
                    if isinstance(director_reference, dict)
                    else None
                ),
                aspect_ratio=aspect_ratio,
            )
            if refreshed.get("total_boards") != expected_shots:
                return {
                    "status": "error",
                    "error": (
                        "Phase 3 could not refresh all character-locked Pxx boards: "
                        f"{refreshed.get('total_boards', 0)}/{expected_shots}"
                    ),
                    "duration_s": _elapsed(start),
                }
            artifact_errors = validate_shot_storyboard_artifacts(
                output_dir,
                storyboard,
            )
            if artifact_errors:
                return {
                    "status": "error",
                    "error": "Phase 3 character-locked Pxx validation failed",
                    "artifact_errors": artifact_errors,
                    "duration_s": _elapsed(start),
                }
            storyboard_path.write_text(
                json.dumps(storyboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            composition_report = _validate_storyboard_image_composition(
                output_dir, storyboard
            )
            if not composition_report["valid"]:
                return {
                    "status": "error",
                    "error": "Storyboard composition validation failed after Phase 3",
                    "composition_report": composition_report,
                    "duration_s": _elapsed(start),
                }

            # Performance boards are run-local story assets. Build them only
            # after canonical Pxx action lineage and static v6 identities have
            # both passed their owners; never promote them into the registry.
            from phases.phase3.performance_reference_board import (
                attach_performance_guides_to_storyboard,
                build_character_performance_plan,
                generate_performance_reference_boards,
            )

            performance_characters = [
                character
                for character in characters_list
                if build_character_performance_plan(storyboard, character) is not None
            ]
            if performance_characters:
                from clients.ark_multimodal_client import ArkMultimodalClient
                from clients.seedream_client import SeedreamClient

                performance_boards = generate_performance_reference_boards(
                    output_dir,
                    storyboard,
                    performance_characters,
                    image_client=SeedreamClient(),
                    review_client=ArkMultimodalClient(),
                    allow_provider_corrections=(
                        not _acceptance_disable_provider_retries
                    ),
                )
                outputs.extend(
                    str(board["board"])
                    for board in performance_boards
                    if board.get("board")
                )
            attach_performance_guides_to_storyboard(
                storyboard,
                performance_boards,
            )
            storyboard_path.write_text(
                json.dumps(storyboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        result = {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs or ["characters/"],
        }
        if registry_summary is not None:
            result["character_registry"] = registry_summary
        result["performance_boards"] = {
            "count": len(performance_boards),
            "provider_requests": sum(
                int(board.get("provider_requests") or 0)
                for board in performance_boards
            ),
            "characters": [
                board.get("character_id") for board in performance_boards
            ],
        }
        return result

    except ImportError as e:
        print(f"  ⚠ Phase 3 import 失败: {e}")
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
