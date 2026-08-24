"""Independent LLM supervision of a storyboard after deterministic QA."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI

from utils.config import ARK_BASE_URL, DEFAULT_TEXT_MODEL, get_api_key
from utils.ark_llm import call_llm_stream


MAX_TOKENS = 4096

_SUPERVISION_TOP_LEVEL_FIELDS = (
    "title",
    "target_duration",
    "delivery_target_duration",
    "material_duration",
    "total_shots",
    "storyboard_beat_count",
    "aspect_ratio",
)
_SUPERVISION_SHOT_FIELDS = (
    "id",
    "shot_order",
    "name",
    "duration",
    "where",
    "who",
    "character_ids",
    "what",
    "visual",
    "description",
    "action_description",
    "micro_actions",
    "start_state",
    "end_state",
    "emotion",
    "hero_moment",
    "shot_intent",
    "shot_size",
    "camera_angle",
    "camera_movement",
    "lens_mm",
    "lighting_key",
    "lighting_description",
    "time",
    "time_of_day",
    "time_window",
    "transition_to_next",
    "source_sequence_ids",
    "source_excerpt",
    "dialogue",
)
_SUPERVISION_BEAT_FIELDS = (
    "beat_id",
    "position",
    "duration_s",
    "effective_story_duration_s",
    "action",
    "micro_actions",
    "start_state",
    "end_state",
    "hero_moment",
    "shot_intent",
    "shot_size",
    "camera_angle",
    "camera_movement",
    "lens_mm",
    "lighting_key",
    "source_action_unit_ids",
)
_SUPERVISION_DIRECTOR_FIELDS = (
    "sequence_id",
    "scene_goal",
    "emotion_arc",
    "visual_focus",
    "spatial_intent",
    "transition_intent",
)
_SUPERVISION_TEMPORAL_FIELDS = (
    "period",
    "label",
    "source_time",
    "source_kind",
    "local_clock_window",
    "visible_light_requirements",
    "forbidden_visual_cues",
    "continuity",
)

SYSTEM_PROMPT = """You are an independent film producer reviewing a completed storyboard.
Review it with fresh eyes. Check narrative and spatial continuity, character
identity and appearance consistency across shots, conformance to the supplied
visual style, pacing against the target duration, and dialogue plausibility.
Do not rewrite or invent story material. Return only one JSON object with:
grade (A, B, C, or D), issues (an array of objects containing shot_order,
category, severity, and description), verdict (pass, warn, or block), and a
concise summary. Use block only for problems serious enough to make generation
wasteful; otherwise use warn or pass."""


class SupervisionBlockedError(RuntimeError):
    """Raised when blocking supervision rejects a storyboard."""


def _get_llm_client(config: dict[str, Any]) -> OpenAI:
    api_key = config.get("api_key") or get_api_key("ARK_AGENT_API_KEY")
    if not api_key:
        raise RuntimeError("ARK_AGENT_API_KEY is required for supervision")
    timeout = float(
        config.get("request_timeout")
        or os.environ.get("HONCUT_SUPERVISION_REQUEST_TIMEOUT_S", "60")
    )
    return OpenAI(
        api_key=api_key,
        base_url=config.get("base_url") or ARK_BASE_URL,
        timeout=timeout,
        max_retries=0,
        http_client=DefaultHttpxClient(timeout=timeout, trust_env=False),
    )


def _call_llm(prompt: str, config: dict[str, Any]) -> str:
    client = _get_llm_client(config)
    return call_llm_stream(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=config.get("model") or DEFAULT_TEXT_MODEL,
        max_tokens=int(config.get("max_tokens", MAX_TOKENS)),
        wall_timeout=float(
            config.get("supervision_wall_timeout")
            or os.environ.get("HONCUT_SUPERVISION_WALL_TIMEOUT_S", "180")
        ),
        idle_timeout=float(
            config.get("supervision_idle_timeout")
            or os.environ.get("HONCUT_SUPERVISION_IDLE_TIMEOUT_S", "60")
        ),
        _client=client,
    )


def _selected_fields(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        field: value[field]
        for field in fields
        if field in value and value[field] not in (None, "", [], {})
    }


def _supervision_storyboard_contract(storyboard: dict[str, Any]) -> dict[str, Any]:
    """Project authored semantics without Provider prompts or artifact history."""
    contract = {
        "schema": "honcut.supervision-storyboard-projection.v1",
        **_selected_fields(storyboard, _SUPERVISION_TOP_LEVEL_FIELDS),
        "shots": [],
    }
    for raw_shot in storyboard.get("shots") or []:
        if not isinstance(raw_shot, dict):
            continue
        shot = _selected_fields(raw_shot, _SUPERVISION_SHOT_FIELDS)
        director_intent = raw_shot.get("director_intent")
        if isinstance(director_intent, dict):
            selected_director = _selected_fields(
                director_intent,
                _SUPERVISION_DIRECTOR_FIELDS,
            )
            if selected_director:
                shot["director_intent"] = selected_director
        temporal_contract = raw_shot.get("temporal_visual_contract")
        if isinstance(temporal_contract, dict):
            selected_temporal = _selected_fields(
                temporal_contract,
                _SUPERVISION_TEMPORAL_FIELDS,
            )
            if selected_temporal:
                shot["temporal_visual_contract"] = selected_temporal
        beats = [
            _selected_fields(beat, _SUPERVISION_BEAT_FIELDS)
            for beat in (raw_shot.get("storyboard_beats") or [])
            if isinstance(beat, dict)
        ]
        if beats:
            shot["storyboard_beats"] = beats
        contract["shots"].append(shot)
    return contract


def _review_prompt(storyboard: dict, visual_style: str) -> str:
    contract = _supervision_storyboard_contract(storyboard)
    return (
        "Review the following storyboard independently. The supplied style is "
        "a constraint, not story material. Evaluate total shot duration against "
        "any target duration recorded in the storyboard.\n\n"
        f"VISUAL STYLE:\n{visual_style or '(not specified)'}\n\n"
        "STORYBOARD SEMANTIC CONTRACT:\n"
        f"{json.dumps(contract, ensure_ascii=False, sort_keys=True)}"
    )


def _fallback_report(reason: str) -> dict[str, Any]:
    return {
        "grade": "B",
        "issues": [],
        "verdict": "warn",
        "summary": f"Supervision response could not be parsed; manual review advised. {reason}",
    }


def _parse_response(response: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        grade = str(value.get("grade", "")).upper()
        verdict = str(value.get("verdict", "")).lower()
        issues = value.get("issues")
        summary = value.get("summary")
        if grade in {"A", "B", "C", "D"} and verdict in {"pass", "warn", "block"} and isinstance(issues, list) and isinstance(summary, str):
            return {"grade": grade, "issues": issues, "verdict": verdict, "summary": summary}
    return _fallback_report("Invalid JSON or schema.")


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _blocking_message(report: dict[str, Any]) -> str:
    descriptions = []
    for issue in report.get("issues", []):
        if isinstance(issue, dict):
            shot = issue.get("shot_order", "?")
            descriptions.append(f"shot {shot}: {issue.get('description', 'unspecified issue')}")
    details = "; ".join(descriptions) or report.get("summary", "unspecified issue")
    return f"Supervision blocked video generation: {details}"


def run_supervision(
    storyboard: dict,
    visual_style: str,
    output_dir: Path,
    config: dict,
) -> dict:
    """Review *storyboard*, persist the report, and enforce optional blocking."""
    if not config.get("supervision", True):
        return {"status": "skipped", "reason": "supervision disabled"}

    try:
        response = _call_llm(_review_prompt(storyboard, visual_style), config)
        report = _parse_response(response)
    except Exception as exc:
        report = _fallback_report(f"{type(exc).__name__}: {exc}")

    _write_atomic(Path(output_dir) / "supervision_report.json", report)
    if config.get("supervision_blocking", False) and report["verdict"] == "block":
        raise SupervisionBlockedError(_blocking_message(report))
    return report
