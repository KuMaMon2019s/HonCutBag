"""Volcengine SeedASR client for Phase 8 subtitle transcription."""

from __future__ import annotations

import base64
import os
import time
import uuid
import wave
from pathlib import Path

import requests


ASR_BASE_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel"
ASR_RESOURCE_ID = "volc.seedasr.sauc.duration"
SUBMIT_URL = f"{ASR_BASE_URL}/submit"
QUERY_URL = f"{ASR_BASE_URL}/query"


def _wav_duration_ms(audio_path: Path) -> int:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            return round(wav_file.getnframes() * 1000 / wav_file.getframerate())
    except (wave.Error, EOFError):
        return 0


def _mock_transcription(audio_path: Path) -> dict:
    duration_ms = _wav_duration_ms(audio_path) or 1000
    words = ("Hon", "Cut")
    midpoint = duration_ms // 2
    return {
        "text": "HonCut",
        "segments": [
            {"word": words[0], "start_ms": 0, "end_ms": midpoint},
            {"word": words[1], "start_ms": midpoint, "end_ms": duration_ms},
        ],
        "duration_ms": duration_ms,
    }


def _headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": ASR_RESOURCE_ID,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }


def _check_response(response: requests.Response, operation: str) -> None:
    response.raise_for_status()
    status_code = response.headers.get("X-Api-Status-Code")
    if status_code and status_code != "20000000":
        message = response.headers.get("X-Api-Message", "unknown ASR error")
        raise RuntimeError(f"SeedASR {operation} failed ({status_code}): {message}")


def _normalise_transcription(payload: dict) -> dict:
    result = payload.get("result") or {}
    utterances = result.get("utterances") or []
    segments = []
    for utterance in utterances:
        for word in utterance.get("words") or []:
            text = word.get("text") or word.get("word") or ""
            if text:
                segments.append({
                    "word": text,
                    "start_ms": int(word.get("start_time", word.get("start_ms", 0))),
                    "end_ms": int(word.get("end_time", word.get("end_ms", 0))),
                })
    duration_ms = int((payload.get("audio_info") or {}).get("duration") or 0)
    return {
        "text": result.get("text") or "".join(item["word"] for item in segments),
        "segments": segments,
        "duration_ms": duration_ms,
    }


def transcribe_audio(audio_path: str) -> dict:
    """Submit a local WAV to SeedASR, poll, and return word-level timestamps.

    ``HONCUT_ASR_MOCK=1`` avoids HTTP and returns deterministic output. Real
    request failures deliberately propagate; Phase 8 must never pretend an ASR
    outage is a silent shot.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASR audio file does not exist: {path}")
    if os.getenv("HONCUT_ASR_MOCK") == "1":
        return _mock_transcription(path)

    api_key = os.getenv("ARK_AGENT_API_KEY") or os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("SeedASR requires ARK_AGENT_API_KEY or ARK_API_KEY")

    request_id = str(uuid.uuid4())
    encoded_audio = base64.b64encode(path.read_bytes()).decode("ascii")
    body = {
        "user": {"uid": request_id},
        "audio": {"data": encoded_audio, "format": path.suffix.lstrip(".") or "wav"},
        "request": {"model_name": "bigmodel", "enable_itn": True},
    }
    response = requests.post(
        SUBMIT_URL, headers=_headers(api_key, request_id), json=body, timeout=60
    )
    _check_response(response, "submit")

    log_id = response.headers.get("X-Tt-Logid", "")
    poll_headers = _headers(api_key, request_id)
    if log_id:
        poll_headers["X-Tt-Logid"] = log_id
    deadline = time.monotonic() + float(os.getenv("HONCUT_ASR_TIMEOUT_S", "300"))
    while time.monotonic() < deadline:
        query = requests.post(QUERY_URL, headers=poll_headers, json={}, timeout=30)
        query.raise_for_status()
        status_code = query.headers.get("X-Api-Status-Code", "")
        if status_code == "20000000":
            return _normalise_transcription(query.json())
        if status_code not in {"20000001", "20000002"}:
            _check_response(query, "query")
        time.sleep(float(os.getenv("HONCUT_ASR_POLL_INTERVAL_S", "1")))
    raise TimeoutError(f"SeedASR query timed out for request {request_id}")
