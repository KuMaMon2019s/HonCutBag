"""Phase 1 screenwriting, style resolution, and durable intermediate artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Optional

from phases.phase1.adaptation_engine import AVG_SHOT_DURATION
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
) -> dict:
    """Phase 1: text_parser → event_extractor → character_discoverer → adaptation_engine → storyboard_generator"""
    _banner(1, 9, "编剧引擎 (Screenwriter)", dry_run)
    start = _now()
    _p2_est = estimate_phase_duration("phase1")
    print(f"  ⏱ Phase 1 开始 (预估 ~{int(_p2_est)}s)")
    output_dir = Path(output_dir)

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

        # dry-run 模式：生成模拟数据，不调用 API
        if dry_run:
            print("  ⊘ dry-run 模式，生成模拟数据（跳过 API 调用）...")
            if reporter:
                reporter.step("phase1", "dry-run: 生成模拟事件", progress_pct=30)

            # 模拟事件数据
            mock_events = [
                {
                    "id": 1,
                    "who": ["主角"],
                    "where": "场景A",
                    "what": "发现关键线索",
                    "emotion": "紧张",
                    "visual": "主角在昏暗的房间中发现了一张神秘的地图",
                    "time": "夜晚",
                    "action_type": "discovery"
                },
                {
                    "id": 2,
                    "who": ["主角", "配角"],
                    "where": "场景B",
                    "what": "展开行动",
                    "emotion": "激动",
                    "visual": "两人在阳光下讨论计划，充满希望",
                    "time": "白天",
                    "action_type": "action"
                },
                {
                    "id": 3,
                    "who": ["主角"],
                    "where": "场景C",
                    "what": "面临挑战",
                    "emotion": "坚定",
                    "visual": "主角独自站在山顶，眺望远方",
                    "time": "黄昏",
                    "action_type": "resolution"
                }
            ]

            if reporter:
                reporter.step("phase1", f"dry-run: 提取 {len(mock_events)} 个事件", progress_pct=45)

            # 模拟角色数据
            mock_characters = {
                "characters": [
                    {
                        "id": "protagonist",
                        "name": "主角",
                        "aliases": ["他", "主人公"],
                        "role": "protagonist",
                        "appearance": {
                            "gender": "male",
                            "age_range": "25-35",
                            "height": "中等身高",
                            "build": "athletic",
                            "hair": "黑色短发",
                            "face": "坚毅的面容",
                            "clothing": "休闲装",
                            "distinguishing": "无明显特征",
                            "summary": "25-35岁男性，黑色短发，身材健壮，面容坚毅"
                        },
                        "personality": {
                            "traits": ["勇敢", "坚定", "善良"],
                            "speech_style": "简洁有力",
                            "motivation": "寻找真相"
                        },
                        "style": "写实风格, 35mm film, 自然光",
                        "negative": "卡通, 3D渲染, 过度饱和",
                        "size": "2K",
                        "first_appearance": 1,
                        "appearance_count": 3
                    },
                    {
                        "id": "supporting",
                        "name": "配角",
                        "aliases": ["朋友"],
                        "role": "supporting",
                        "appearance": {
                            "gender": "female",
                            "age_range": "20-30",
                            "height": "中等身高",
                            "build": "slim",
                            "hair": "棕色长发",
                            "face": "温和的面容",
                            "clothing": "职业装",
                            "distinguishing": "戴眼镜",
                            "summary": "20-30岁女性，棕色长发，身材纤细，戴眼镜，面容温和"
                        },
                        "personality": {
                            "traits": ["聪明", "细心", "支持"],
                            "speech_style": "理性分析",
                            "motivation": "帮助主角"
                        },
                        "style": "写实风格, 35mm film, 自然光",
                        "negative": "卡通, 3D渲染, 过度饱和",
                        "size": "2K",
                        "first_appearance": 2,
                        "appearance_count": 1
                    }
                ]
            }

            if reporter:
                reporter.step("phase1", f"dry-run: 发现 {len(mock_characters['characters'])} 个角色", progress_pct=60)

            # 模拟分镜数据
            resolved_video_spec = project_video_spec or _project_video_spec("1080p")
            mock_storyboard = {
                "shots": [
                    {
                        "id": 1,
                        "prompt": "A young man discovers a mysterious map in a dimly lit room, cinematic lighting, 35mm film, natural light, tense atmosphere",
                        "caption": "发现神秘地图",
                        "duration": 5,
                        "aspect_ratio": resolved_video_spec["aspect_ratio"],
                        "width": resolved_video_spec["width"],
                        "height": resolved_video_spec["height"],
                        "scene": "昏暗的房间",
                        "action": "发现地图",
                        "camera": "中景",
                        "emotion": "紧张"
                    },
                    {
                        "id": 2,
                        "prompt": "Two people discussing plans under bright sunlight, hopeful atmosphere, cinematic composition, natural lighting",
                        "caption": "讨论计划",
                        "duration": 5,
                        "aspect_ratio": resolved_video_spec["aspect_ratio"],
                        "width": resolved_video_spec["width"],
                        "height": resolved_video_spec["height"],
                        "scene": "阳光明媚的户外",
                        "action": "讨论计划",
                        "camera": "双人镜头",
                        "emotion": "激动"
                    },
                    {
                        "id": 3,
                        "prompt": "A determined man standing alone on a mountain top at sunset, looking into the distance, epic cinematic shot, golden hour lighting",
                        "caption": "眺望远方",
                        "duration": 5,
                        "aspect_ratio": resolved_video_spec["aspect_ratio"],
                        "width": resolved_video_spec["width"],
                        "height": resolved_video_spec["height"],
                        "scene": "山顶",
                        "action": "眺望",
                        "camera": "远景",
                        "emotion": "坚定"
                    }
                ],
                "total_duration": 15,
                "style": "写实电影风格",
                "aspect_ratio": resolved_video_spec["aspect_ratio"],
                "width": resolved_video_spec["width"],
                "height": resolved_video_spec["height"],
            }
            # Dry-run artifacts must obey the same 15-30s primary contract as
            # paid runs; otherwise later phases validate a fixture that can
            # never exist in production.
            requested_duration = max(15, int(duration or 15))
            mock_shot_count = max(1, math.ceil(requested_duration / 30))
            templates = list(mock_storyboard["shots"])
            base_duration, remainder = divmod(requested_duration, mock_shot_count)
            mock_storyboard["shots"] = []
            for index in range(mock_shot_count):
                shot = dict(templates[index % len(templates)])
                shot["id"] = index + 1
                shot["duration"] = base_duration + (1 if index < remainder else 0)
                mock_storyboard["shots"].append(shot)
            mock_storyboard["total_duration"] = requested_duration
            from phases.phase1.storyboard_beats import plan_storyboard_beats
            from utils.privacy_visual_policy import (
                apply_no_real_person_character_policy,
                is_no_real_person_enabled,
            )

            if is_no_real_person_enabled():
                mock_characters = apply_no_real_person_character_policy(mock_characters)

            plan_storyboard_beats(mock_storyboard)
            _integrate_storyboard_prompts(mock_storyboard, mock_characters["characters"])
            _attach_director_storyboard(
                output_dir,
                mock_storyboard,
                mock_characters["characters"],
                dry_run=True,
            )

            if reporter:
                reporter.step("phase1", f"dry-run: 生成 {len(mock_storyboard['shots'])} 个分镜", progress_pct=80)

            # 写出文件
            storyboard_path = output_dir / "STORYBOARD.json"
            characters_path = output_dir / "CHARACTERS.json"
            events_path = output_dir / "events.json"

            storyboard_path.write_text(json.dumps(mock_storyboard, ensure_ascii=False, indent=2))
            characters_path.write_text(json.dumps(mock_characters, ensure_ascii=False, indent=2))
            events_path.write_text(json.dumps({"events": mock_events}, ensure_ascii=False, indent=2))

            outputs = [
                "STORYBOARD.json", "CHARACTERS.json", "events.json",
                "director_storyboard_prompt.txt", "director_storyboard.json",
            ]
            print(f"  ✓ Phase 1 完成 (dry-run): {outputs}")

            return {
                "status": "done",
                "duration_s": _elapsed(start),
                "outputs": outputs,
                "_storyboard": mock_storyboard,
                "_characters": mock_characters,
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
        from phases.phase1.storyboard_beats import plan_storyboard_beats

        plan_storyboard_beats(storyboard)
        material_budget = storyboard.get("material_budget") or {}
        print(
            "  ✓ 素材双账本: 故事时钟 "
            f"{float(material_budget.get('story_clock_duration_s') or 0):g}s "
            "+ 桥接生成 "
            f"{float(material_budget.get('bridge_generation_duration_s') or 0):g}s "
            "= 总生成 "
            f"{float(material_budget.get('total_generated_duration_s') or 0):g}s；"
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
            "STORYBOARD.json", "CHARACTERS.json",
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
        }
        return phase1_result

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start), "outputs": outputs}
    finally:
        configure_heartbeat_callback(None)
        if reporter:
            reporter.stop_heartbeat()
