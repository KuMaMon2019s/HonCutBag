"""Phase 1 screenwriting, style resolution, and durable intermediate artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Optional

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
from phases.phase1.phase1_director import (
    DirectorPlanningError,
    run_phase1_director,
)
from prompt.shot_prompt_builder import build_batch_prompts
from prompt.speech_pacing import annotate_shot_pacing
from prompt.prompt_sanitizer import sanitize_quality_prompt
from prompt.three_part_prompt import build_three_part_prompt
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from tools.asset_binder import bind_assets
from utils.ark_llm import call_llm_stream, configure_heartbeat_callback
from utils.media_profiles import _project_video_spec
from utils.progress_reporter import ProgressReporter
from utils.storyboard_geometry import _storyboard_canvas, _storyboard_image_size
from utils.timing_estimator import estimate_phase_duration


STYLE_SUMMARY_WALL_TIMEOUT = 180.0
STYLE_SUMMARY_IDLE_TIMEOUT = 75.0


def _plan_director_for_events(
    events: list[dict[str, Any]],
    output_dir: Path,
    dry_run: bool,
    director_runner,
) -> dict[str, Any]:
    result = director_runner(events, output_dir, dry_run)
    status = result.get("status")
    if status != "done" and not (dry_run and status == "skipped"):
        detail = (
            result.get("error")
            or result.get("reason")
            or "missing success evidence"
        )
        raise DirectorPlanningError(
            f"director planning returned {status}: {detail}"
        )
    return result


def _integrate_storyboard_prompts(storyboard: dict, characters: list[dict]) -> dict:
    """Normalize generated prompts through the shared Phase 1 contracts."""
    assets = [
        {"id": character.get("id", index), "name": character.get("name", ""), "type": "角色"}
        for index, character in enumerate(characters, 1)
        if character.get("name")
    ]
    for shot in storyboard.get("shots", []):
        visual = shot.get("prompt") or shot.get("visual") or shot.get("what") or "scene"
        lighting = shot.get("lighting_key") or "natural cinematic lighting"
        style = storyboard.get("style") or "cinematic"
        prompt = build_three_part_prompt(str(visual), str(lighting), str(style))
        referenced = shot.get("associate_assets") or shot.get("who") or []
        referenced_names = {str(value) for value in referenced}
        shot_assets = [
            asset for asset in assets
            if str(asset["id"]) in referenced_names or asset["name"] in referenced_names
        ]
        shot["prompt"] = sanitize_quality_prompt(bind_assets(prompt, shot_assets))
    return storyboard


def _attach_director_storyboard(
    output_dir: Path,
    storyboard: dict,
    characters: Optional[list[dict]] = None,
    *,
    client=None,
    dry_run: bool = False,
) -> dict:
    """Generate and register the mandatory Phase 1 director overview artifact."""
    from phases.phase1.director_storyboard import generate_director_storyboard

    video_width, video_height, aspect_ratio = _storyboard_canvas(storyboard)
    storyboard.setdefault("aspect_ratio", aspect_ratio)
    manifest = generate_director_storyboard(
        output_dir,
        storyboard,
        characters,
        client=client,
        dry_run=dry_run,
        size=_storyboard_image_size(
            video_width=video_width,
            video_height=video_height,
        ),
    )
    storyboard["director_storyboard"] = {
        "image": manifest["image"],
        "manifest": "director_storyboard.json",
        "prompt": manifest["prompt"],
        "status": manifest["status"],
        "provider": manifest["provider"],
        "model": manifest["model"],
        "panel_count": len(manifest["panels"]),
        "panel_schema": (
            manifest.get("panel_extraction", {}).get("schema")
            if isinstance(manifest.get("panel_extraction"), dict)
            else None
        ),
        "panel_dir": "director_panels",
        "preliminary_groups": list(dict.fromkeys(
            panel["group_id"] for panel in manifest["panels"]
        )),
    }
    return manifest


def _extract_visual_style_text(script_text: str) -> Optional[str]:
    """Extract a declared art-style paragraph without interpreting the script."""
    match = re.search(
        r"(?im)^\s*(?:美术风格|Art\s+style)\s*[：:]\s*(.+(?:\n(?!\s*(?:角色设定|剧情|人物设定|Characters?|Plot|Story)\s*[：:]).+)*)",
        script_text,
    )
    if not match:
        return None
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return " ".join(lines).strip() or None


def _summarize_visual_style_with_llm(script_text: str) -> Optional[str]:
    """Best-effort style summary; deliberately isolated so tests can mock it."""
    api_key = os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ 风格总结不可用，降级使用 default_visual_style")
        return None
    try:
        return call_llm_stream(
            messages=[{
                "role": "user",
                "content": "用一句话总结以下剧本的美术风格，只输出风格描述：\n" + script_text,
            }],
            max_tokens=1024,
            wall_timeout=STYLE_SUMMARY_WALL_TIMEOUT,
            read_timeout=STYLE_SUMMARY_IDLE_TIMEOUT,
            idle_timeout=STYLE_SUMMARY_IDLE_TIMEOUT,
        ).strip() or None
    except Exception as exc:
        print(f"  ⚠ 风格总结失败，降级使用 default_visual_style: {exc}")
        return None


PHASE1_CHECKPOINT_SCHEMA_VERSION = 3


def _phase1_input_hash(items: list) -> str:
    serialized = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_write_phase1_json(
    path: Path,
    payload: dict,
    *,
    collection_key: str,
    input_hash: str,
) -> None:
    """Persist a completed Phase 1 substage without exposing partial JSON."""
    stored = dict(payload)
    stored["_checkpoint"] = {
        "schema_version": PHASE1_CHECKPOINT_SCHEMA_VERSION,
        "collection_key": collection_key,
        "input_hash": input_hash,
        "item_count": len(stored.get(collection_key, [])),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Persist one canonical artifact without exposing a partial JSON file."""

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _load_phase1_checkpoint(
    path: Path,
    collection_key: str,
    *,
    input_hash: str,
) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get(collection_key), list):
        return None
    metadata = payload.get("_checkpoint")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("schema_version") != PHASE1_CHECKPOINT_SCHEMA_VERSION:
        return None
    if metadata.get("collection_key") != collection_key:
        return None
    if metadata.get("input_hash") != input_hash:
        return None
    if metadata.get("item_count") != len(payload[collection_key]):
        return None
    return payload


def _write_project_visual_style(output_dir: Path, style_text: str) -> Path:
    """Write the minimal frontmatter accepted by parse_visual_style."""
    import yaml
    payload = {
        "name": "Script-derived project style",
        "version": "1.0",
        "style_prompt_short": style_text,
        "style_prompt_full": style_text,
    }
    style_path = Path(output_dir) / "visual-style.md"
    style_path.write_text(
        "---\n" + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000) + "---\n",
        encoding="utf-8",
    )
    return style_path


def _continuity_mode_from_text(text: str) -> str | None:
    """Extract only explicit single-take direction from the source brief."""
    normalized = re.sub(r"\s+", " ", str(text or "")).casefold()
    markers = (
        "一镜到底",
        "单镜到底",
        "one take",
        "one-take",
        "single continuous shot",
        "continuous oner",
    )
    return "one_take" if any(marker in normalized for marker in markers) else None


def run_phase1_screenwriter(
    text: str,
    output_dir: Path,
    duration: int,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    project_video_spec: dict[str, Any] | None = None,
    *,
    _director_runner=None,
) -> dict:
    """Phase 1: parse → events → director intent → adaptation → storyboard."""
    _banner(1, 9, "编剧引擎 (Screenwriter)", dry_run)
    start = _now()
    _p2_est = estimate_phase_duration("phase1")
    print(f"  ⏱ Phase 1 开始 (预估 ~{int(_p2_est)}s)")
    output_dir = Path(output_dir)
    director_runner = _director_runner or run_phase1_director
    director: dict[str, Any] | None = None

    try:
        from prompt.text_parser import parse_text
    except ImportError as e:
        return {"status": "error", "error": f"Phase 1 import failed: {e}", "duration_s": _elapsed(start)}

    outputs = []
    if reporter:
        reporter.start_heartbeat("phase1")
        configure_heartbeat_callback(
            lambda: reporter.step(
                "phase1", "LLM 流式响应", progress_pct=reporter._progress_pct
            )
        )
    try:
        # Step 2.1: text_parser → segments list
        print("  → text_parser: 解析文本结构...")
        if reporter:
            reporter.step("phase1", "解析文本结构", progress_pct=10)
        parsed = parse_text(text)
        segments = parsed.get("segments", [])
        print(f"    ✓ 解析出 {len(segments)} 个段落")
        if reporter:
            reporter.step("phase1", f"解析出 {len(segments)} 个段落", progress_pct=20)

        # dry-run derives its capacity receipt and structural fixtures from
        # the actual source text. It never calls an LLM, image model, or Provider.
        if dry_run:
            from phases.phase1.dry_run_capacity import (
                build_dry_run_capacity_preflight,
                write_dry_run_receipt,
            )

            print("  ⊘ dry-run 模式，执行真实源文本容量预检（零远程请求）...")
            if reporter:
                reporter.step("phase1", "dry-run: 源文本容量预检", progress_pct=30)
            preflight = build_dry_run_capacity_preflight(
                text,
                segments,
                duration=max(15, int(duration or 15)),
                shot_duration=shot_duration,
            )
            receipt = preflight["receipt"]
            receipt_path = output_dir / "phase1_dry_run_receipt.json"
            write_dry_run_receipt(receipt_path, receipt)
            outputs = ["phase1_dry_run_receipt.json"]
            capacity_plan = receipt["capacity_plan"]
            if receipt["status"] != "passed":
                return {
                    "status": "error",
                    "error": (
                        "dry-run source capacity preflight failed: "
                        f"{capacity_plan['action_capacity_status']}"
                    ),
                    "duration_s": _elapsed(start),
                    "outputs": outputs,
                    "dry_run_receipt": receipt_path.name,
                    "capacity_plan": capacity_plan,
                }

            source_events = preflight["events"]
            director = _plan_director_for_events(
                source_events,
                output_dir,
                True,
                director_runner,
            )
            if reporter:
                reporter.step(
                    "phase1",
                    f"dry-run: 源文本事件 {len(source_events)} 个",
                    progress_pct=50,
                )
            characters = {
                "characters": [],
                "dry_run": True,
                "source_derived": True,
            }
            resolved_video_spec = project_video_spec or _project_video_spec("1080p")
            requested_duration = max(15, int(duration or 15))
            shot_count = max(1, int(capacity_plan["primary_shots"]))
            base_duration, remainder = divmod(requested_duration, shot_count)
            shots = []
            for index in range(shot_count):
                event_start = math.floor(index * len(source_events) / shot_count)
                event_end = math.floor((index + 1) * len(source_events) / shot_count)
                assigned_events = source_events[event_start:event_end]
                if not assigned_events and source_events:
                    assigned_events = [
                        source_events[min(event_start, len(source_events) - 1)]
                    ]
                source_slice = "；".join(
                    str(event.get("what") or "").strip()
                    for event in assigned_events
                    if str(event.get("what") or "").strip()
                ) or text.strip() or "source-derived dry-run scene"
                micro_actions = [
                    action
                    for event in assigned_events
                    for action in event.get("micro_actions", [])
                ]
                generation_action_units = [
                    {
                        "unit_id": f"DRYRUN_GAU{unit_index:03d}",
                        "kind": (
                            "simultaneous"
                            if event.get("generation_motion_mode") == "composite"
                            else "sequential"
                        ),
                        "actions": list(event.get("micro_actions", [])),
                        "ledger_indexes": [unit_index - 1],
                    }
                    for unit_index, event in enumerate(assigned_events, 1)
                    if event.get("micro_actions")
                ]
                shot_id = index + 1
                shots.append(
                    {
                        "id": shot_id,
                        "name": f"source-derived dry-run shot {shot_id}",
                        "prompt": source_slice,
                        "caption": "",
                        "duration": base_duration + (1 if index < remainder else 0),
                        "suggested_duration": base_duration + (
                            1 if index < remainder else 0
                        ),
                        "aspect_ratio": resolved_video_spec["aspect_ratio"],
                        "width": resolved_video_spec["width"],
                        "height": resolved_video_spec["height"],
                        "scene": source_slice,
                        "where": "dry-run source",
                        "action": source_slice,
                        "action_description": source_slice,
                        "what": source_slice,
                        "visual": source_slice,
                        "camera": "source-derived structural preflight",
                        "emotion": "",
                        "who": [],
                        "source_events": [
                            event["event_id"] for event in assigned_events
                        ],
                        "source_action_unit_ids": [
                            event["action_unit_id"] for event in assigned_events
                        ],
                        "micro_actions": micro_actions,
                        "generation_actions": micro_actions,
                        "generation_action_units": generation_action_units,
                        "shot_size": ("wide" if index % 2 == 0 else "medium"),
                        "camera_movement": (
                            "static" if index % 2 == 0 else "dolly_in"
                        ),
                        "lighting_key": "natural",
                        "shot_intent": (
                            "establishing" if index == 0 else "action"
                        ),
                        "hero_moment": index == shot_count - 1,
                        "texture_keywords": ["source-derived", "dry-run"],
                        "dialogue": None,
                        "gen_strategy": "t2v",
                    }
                )
            storyboard = {
                "shots": shots,
                "events": source_events,
                "target_duration": requested_duration,
                "delivery_target_duration": requested_duration,
                "material_duration": requested_duration,
                "total_duration": requested_duration,
                "capacity_plan": capacity_plan,
                "style": "source-derived dry-run structural fixture",
                "aspect_ratio": resolved_video_spec["aspect_ratio"],
                "width": resolved_video_spec["width"],
                "height": resolved_video_spec["height"],
                "dry_run_receipt": receipt_path.name,
            }
            from phases.phase1.storyboard_beats import plan_storyboard_beats

            plan_storyboard_beats(storyboard)
            _integrate_storyboard_prompts(storyboard, characters["characters"])
            _attach_director_storyboard(
                output_dir,
                storyboard,
                characters["characters"],
                dry_run=True,
            )

            if reporter:
                reporter.step(
                    "phase1",
                    f"dry-run: 生成 {len(shots)} 个源文本结构分镜",
                    progress_pct=80,
                )
            storyboard_path = output_dir / "STORYBOARD.json"
            characters_path = output_dir / "CHARACTERS.json"
            events_path = output_dir / "events.json"
            storyboard_path.write_text(
                json.dumps(storyboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            characters_path.write_text(
                json.dumps(characters, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            events_path.write_text(
                json.dumps(
                    {
                        "events": source_events,
                        "dry_run_receipt": receipt_path.name,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            outputs.extend(
                [
                    "STORYBOARD.json",
                    "CHARACTERS.json",
                    "events.json",
                    "director_storyboard_prompt.txt",
                    "director_storyboard.json",
                ]
            )
            print(f"  ✓ Phase 1 完成 (dry-run): {outputs}")
            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "dry_run_receipt": receipt_path.name,
                "capacity_plan": capacity_plan,
                "director": director,
                "_storyboard": storyboard,
                "_characters": characters,
            }


        # 正常模式：调用 API
        try:
            from phases.phase1.character_discoverer import (
                CHARACTER_CONTEXT_SCHEMA_VERSION,
                _is_human_character,
                discover_characters,
            )
            from phases.phase1.adaptation_engine import adapt_events
            from phases.phase1.storyboard_generator import generate_storyboard
            from prompt.event_extractor import EVENT_FLOW_SCHEMA_VERSION, extract_events
        except ImportError as e:
            return {"status": "error", "error": f"Phase 1 import failed: {e}", "duration_s": _elapsed(start)}

        # Step 2.2: event_extractor → events list
        print("  → event_extractor: 提取事件...")
        if reporter:
            reporter.step("phase1", "提取事件", progress_pct=30)
        events_checkpoint = output_dir / "phase1_events.json"
        nonempty_segments = [
            segment for segment in segments if str(segment.get("content", "")).strip()
        ]
        continuity_mode = _continuity_mode_from_text(text)
        events_input_hash = _phase1_input_hash([
            {
                "event_flow_schema_version": EVENT_FLOW_SCHEMA_VERSION,
                "continuity_mode": continuity_mode,
            },
            *nonempty_segments,
        ])
        expected_segment_ids = [segment.get("id", 0) for segment in nonempty_segments]
        events_result = _load_phase1_checkpoint(
            events_checkpoint,
            "events",
            input_hash=events_input_hash,
        )
        if events_result is not None:
            complete_coverage = (
                events_result.get("source_segments_hash") == events_input_hash
                and events_result.get("source_segment_count") == len(nonempty_segments)
                and events_result.get("covered_segment_ids") == expected_segment_ids
                and events_result.get("total_events") == len(events_result["events"])
            )
            if not complete_coverage:
                events_result = None
        if events_result is not None:
            print("    ↻ 复用 phase1_events.json，跳过事件提取")
        else:
            events_result = dict(extract_events(
                segments,
                checkpoint_dir=output_dir,
                continuity_mode=continuity_mode,
            ))
            events_result["schema_version"] = EVENT_FLOW_SCHEMA_VERSION
            events_result["continuity_mode"] = continuity_mode
            events_result["source_segments_hash"] = events_input_hash
            events_result.setdefault("source_segment_count", len(nonempty_segments))
            events_result.setdefault("covered_segment_ids", expected_segment_ids)
            events_result.setdefault("total_events", len(events_result.get("events", [])))
            _atomic_write_phase1_json(
                events_checkpoint,
                events_result,
                collection_key="events",
                input_hash=events_input_hash,
            )
        events = events_result.get("events", [])
        print(f"    ✓ 提取 {len(events)} 个事件")
        if reporter:
            reporter.step("phase1", f"提取 {len(events)} 个事件", progress_pct=40)

        print("  → director_planner: 规划 sequence 导演意图...")
        if reporter:
            reporter.step("phase1", "规划 sequence 导演意图", progress_pct=45)
        director = _plan_director_for_events(
            events,
            output_dir,
            False,
            director_runner,
        )

        # Step 2.3: character_discoverer → characters dict
        print("  → character_discoverer: 发现角色...")
        if reporter:
            reporter.step("phase1", "发现角色", progress_pct=50)
        characters_checkpoint = output_dir / "phase1_characters.json"
        from utils.privacy_visual_policy import (
            NO_REAL_PERSON_POLICY,
            apply_no_real_person_character_policy,
            is_no_real_person_enabled,
        )

        characters_input_hash = _phase1_input_hash([
            {
                "character_context_schema": CHARACTER_CONTEXT_SCHEMA_VERSION,
                "events": events,
                "no_real_person": is_no_real_person_enabled(),
                "no_real_person_policy": (
                    NO_REAL_PERSON_POLICY if is_no_real_person_enabled() else None
                ),
            }
        ])
        characters_result = _load_phase1_checkpoint(
            characters_checkpoint,
            "characters",
            input_hash=characters_input_hash,
        )
        if characters_result is not None:
            characters = characters_result["characters"]
            valid_characters = (
                characters_result.get("source_text_hash") == characters_input_hash
                and characters_result.get("total_characters") == len(characters)
                and all(
                    isinstance(character, dict)
                    and bool(str(character.get("name", "")).strip())
                    and _is_human_character(str(character.get("name", "")).strip())
                    for character in characters
                )
            )
            if not valid_characters:
                characters_result = None
        if characters_result is not None:
            print("    ↻ 复用 phase1_characters.json，跳过角色发现")
        else:
            characters_result = dict(discover_characters(events))
            characters_result["source_text_hash"] = characters_input_hash
            characters_result.setdefault(
                "total_characters", len(characters_result.get("characters", []))
            )
            _atomic_write_phase1_json(
                characters_checkpoint,
                characters_result,
                collection_key="characters",
                input_hash=characters_input_hash,
            )
        if (
            is_no_real_person_enabled()
            and characters_result.get("visual_identity_policy")
            != NO_REAL_PERSON_POLICY
        ):
            characters_result = apply_no_real_person_character_policy(
                characters_result
            )
            characters_result["source_text_hash"] = characters_input_hash
            characters_result.setdefault(
                "total_characters", len(characters_result.get("characters", []))
            )
            _atomic_write_phase1_json(
                characters_checkpoint,
                characters_result,
                collection_key="characters",
                input_hash=characters_input_hash,
            )
        characters_list = characters_result.get("characters", [])
        print(f"    ✓ 发现 {len(characters_list)} 个角色")
        if reporter:
            reporter.step("phase1", f"发现 {len(characters_list)} 个角色", progress_pct=60)

        # Text labels are presentation data. Bind a copy to stable asset IDs so
        # checkpoint hashes remain based on the original understanding output,
        # while all downstream phases consume one canonical identity contract.
        from utils.semantic_contracts import bind_story_semantics

        events = copy.deepcopy(events)
        characters_list = copy.deepcopy(characters_list)
        semantic_ledger = bind_story_semantics(events, characters_list)
        semantic_ledger["source_events_hash"] = events_input_hash
        semantic_ledger["canonical_characters_hash"] = _phase1_input_hash(
            characters_list
        )
        semantic_ledger_path = output_dir / "SEMANTIC_LEDGER.json"
        _atomic_write_json_artifact(semantic_ledger_path, semantic_ledger)
        characters_result = dict(characters_result)
        characters_result["characters"] = characters_list
        characters_result["semantic_ledger"] = semantic_ledger_path.name

        # Step 2.4: adaptation_engine → adapted shots list
        print("  → adaptation_engine: 影视化改编...")
        if reporter:
            reporter.step("phase1", "影视化改编", progress_pct=70)
        adapted = adapt_events(
            events,
            characters_list,
            target_duration=duration,
            shot_duration=shot_duration,
            source_text=text,
            output_dir=output_dir,
            director_plan=director["plan"],
        )
        adapted_shots = adapted.get("shots", [])
        print(f"    ✓ 改编完成，{len(adapted_shots)} 个镜头")
        if reporter:
            reporter.step("phase1", f"改编完成，{len(adapted_shots)} 个镜头", progress_pct=80)

        # Phase 1 storyboard_generator step → storyboard dict
        style_source = "剧本提取"
        visual_style_text = _extract_visual_style_text(text)
        if not visual_style_text:
            style_source = "LLM 总结"
            visual_style_text = _summarize_visual_style_with_llm(text)
        visual_style_path = None
        if visual_style_text:
            visual_style_path = _write_project_visual_style(output_dir, visual_style_text)
            print(f"  ✓ 项目风格: visual-style.md（{style_source}）")
        print("  → storyboard_generator: 生成分镜...")
        if reporter:
            reporter.step("phase1", "生成分镜", progress_pct=90)
        storyboard = generate_storyboard(
            adapted_shots,
            characters_list,
            visual_style_path=str(visual_style_path) if visual_style_path else None,
            visual_style_text=visual_style_text,
            config=project_video_spec or _project_video_spec("1080p"),
        )
        storyboard["delivery_target_duration"] = duration
        storyboard["material_duration"] = adapted.get(
            "material_duration",
            storyboard.get("target_duration"),
        )
        if adapted.get("capacity_plan"):
            storyboard["capacity_plan"] = adapted["capacity_plan"]
            storyboard["generated_duration_ratio_reference"] = adapted[
                "capacity_plan"
            ].get(
                "generated_duration_ratio_reference"
            )
        if continuity_mode:
            storyboard["continuity_mode"] = continuity_mode
        storyboard["semantic_understanding"] = {
            "schema": semantic_ledger["schema"],
            "ledger": semantic_ledger_path.name,
            "source_events_hash": events_input_hash,
        }
        from phases.phase1.storyboard_beats import plan_storyboard_beats

        plan_storyboard_beats(storyboard)
        material_budget = storyboard.get("material_budget") or {}
        print(
            "  ✓ 时长三账本: 故事时钟 "
            f"{float(material_budget.get('story_clock_duration_s') or 0):g}s "
            "；内容 Provider 请求 "
            f"{float(material_budget.get('content_provider_request_duration_s') or 0):g}s "
            "+ 桥接 Provider 请求 "
            f"{float(material_budget.get('bridge_provider_request_duration_s') or 0):g}s "
            "= 总 Provider 请求 "
            f"{float(material_budget.get('total_provider_request_duration_s') or 0):g}s；"
            "桥接在时间线替换等长边界把手",
            flush=True,
        )
        _integrate_storyboard_prompts(storyboard, characters_list)
        annotate_shot_pacing(storyboard.get("shots", []))
        print("  → Seedream: 生成单张手绘导演故事板总览...")
        director_storyboard = _attach_director_storyboard(
            output_dir,
            storyboard,
            characters_list,
        )
        print(
            f"  ✓ 导演故事板总览: director_storyboard.png "
            f"({len(director_storyboard['panels'])} 格)"
        )

        # 写出文件
        storyboard_path = output_dir / "STORYBOARD.json"
        characters_path = output_dir / "CHARACTERS.json"

        storyboard_path.write_text(json.dumps(storyboard, ensure_ascii=False, indent=2))
        characters_path.write_text(json.dumps(characters_result, ensure_ascii=False, indent=2))

        outputs = [
            "STORYBOARD.json", "CHARACTERS.json", "SEMANTIC_LEDGER.json",
            "director_storyboard.png", "director_storyboard_prompt.txt",
            "director_storyboard.json",
        ]
        print(f"  ✓ Phase 1 完成: {outputs}")

        # Quality gate: Phase 1
        qg_report = run_quality_check("phase1", output_dir, {
            "events": storyboard.get("events", []),
            "shots": storyboard.get("shots", []),
        })
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 1 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start), "outputs": outputs}

        # --- M5: 监督层审核（增量，失败不影响后续）---
        try:
            from quality.quality_gate import run_storyboard_review
            review = run_storyboard_review(
                storyboard_data=storyboard,
                script_text=text,
                characters=characters_result.get("characters", []),
            )
            result_data = {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "_storyboard": storyboard,
                "_characters": characters_result,
                "storyboard_review": review,
                "director": director,
            }
            if review.get("grade") == "D":
                print(f"  ⚠ [M5] 分镜审核 D 级，建议重做（但不阻断管线）")
            return result_data
        except Exception as e:
            print(f"  ⚠ [M5] 分镜审核跳过: {e}")

        phase1_result = {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "_storyboard": storyboard,
            "_characters": characters_result,
            "director": director,
        }
        return phase1_result

    except DirectorPlanningError:
        raise
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start), "outputs": outputs}
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()
