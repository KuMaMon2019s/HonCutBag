"""Independent LLM supervision of a storyboard after deterministic QA."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils.config import ARK_BASE_URL, DEFAULT_TEXT_MODEL, get_api_key


MAX_TOKENS = 4096

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
    return OpenAI(
        api_key=api_key,
        base_url=config.get("base_url") or ARK_BASE_URL,
    )


def _call_llm(prompt: str, config: dict[str, Any]) -> str:
    client = _get_llm_client(config)
    stream = client.chat.completions.create(
        model=config.get("model") or DEFAULT_TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=int(config.get("max_tokens", MAX_TOKENS)),
    )
    chunks: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    return "".join(chunks)


def _review_prompt(storyboard: dict, visual_style: str) -> str:
    return (
        "Review the following storyboard independently. The supplied style is "
        "a constraint, not story material. Evaluate total shot duration against "
        "any target duration recorded in the storyboard.\n\n"
        f"VISUAL STYLE:\n{visual_style or '(not specified)'}\n\n"
        "STORYBOARD JSON:\n"
        f"{json.dumps(storyboard, ensure_ascii=False, sort_keys=True)}"
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
