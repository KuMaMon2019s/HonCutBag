"""Volcengine SeedASR WebSocket client for Phase 8 transcription."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
import uuid
import wave
from pathlib import Path
from typing import Any

import websockets


ASR_WS_URL = "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_nostream"
ASR_RESOURCE_ID = "volc.seedasr.sauc.duration"

_FULL_CLIENT_REQUEST = 0x1
_AUDIO_ONLY_REQUEST = 0x2
_FULL_SERVER_RESPONSE = 0x9
_SERVER_ACK = 0xB
_SERVER_ERROR = 0xF
_POSITIVE_SEQUENCE = 0x1
_NEGATIVE_SEQUENCE = 0x3
_JSON = 0x1
_GZIP = 0x1


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


def _protocol_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    # Protocol version 1, four-byte header, followed by the message-specific body.
    return bytes((0x11, (message_type << 4) | flags, (serialization << 4) | compression, 0))


def _request_frame(message_type: int, sequence: int, payload: bytes, *, final: bool = False) -> bytes:
    flags = _NEGATIVE_SEQUENCE if final else _POSITIVE_SEQUENCE
    wire_payload = gzip.compress(payload)
    serialization = _JSON if message_type == _FULL_CLIENT_REQUEST else 0
    return (
        _protocol_header(message_type, flags, serialization, _GZIP)
        + struct.pack(">iI", -sequence if final else sequence, len(wire_payload))
        + wire_payload
    )


def _decode_response(frame: bytes) -> tuple[int, int | None, dict[str, Any]]:
    if not isinstance(frame, bytes) or len(frame) < 4:
        raise RuntimeError("SeedASR returned an invalid WebSocket frame")
    header_size = (frame[0] & 0x0F) * 4
    message_type = frame[1] >> 4
    flags = frame[1] & 0x0F
    compression = frame[2] & 0x0F
    offset = header_size
    sequence = None

    if message_type == _SERVER_ERROR:
        if len(frame) < offset + 8:
            raise RuntimeError("SeedASR returned a truncated error frame")
        code, size = struct.unpack_from(">iI", frame, offset)
        payload = frame[offset + 8 : offset + 8 + size]
        if compression == _GZIP and payload:
            payload = gzip.decompress(payload)
        message = payload.decode("utf-8", errors="replace")
        try:
            detail = json.loads(message)
            message = detail.get("message") or detail.get("error") or message
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"SeedASR recognition failed ({code}): {message}")

    if message_type == _SERVER_ACK:
        if flags & 0x1:
            sequence = struct.unpack_from(">i", frame, offset)[0]
            offset += 4
        # ACK frames can contain an optional payload; it isn't the final result.
        return message_type, sequence, {}

    if message_type != _FULL_SERVER_RESPONSE:
        raise RuntimeError(f"SeedASR returned unsupported message type 0x{message_type:x}")
    if flags & 0x1:
        sequence = struct.unpack_from(">i", frame, offset)[0]
        offset += 4
    if len(frame) < offset + 4:
        raise RuntimeError("SeedASR returned a truncated result frame")
    size = struct.unpack_from(">I", frame, offset)[0]
    payload = frame[offset + 4 : offset + 4 + size]
    if len(payload) != size:
        raise RuntimeError("SeedASR returned an incomplete result payload")
    if compression == _GZIP and payload:
        payload = gzip.decompress(payload)
    try:
        return message_type, sequence, json.loads(payload.decode("utf-8")) if payload else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SeedASR returned a malformed JSON result") from exc


def _normalise_transcription(payload: dict, fallback_duration_ms: int) -> dict:
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
    audio_info = payload.get("audio_info") or result.get("audio_info") or {}
    duration_ms = int(audio_info.get("duration") or fallback_duration_ms)
    return {
        "text": result.get("text") or "".join(item["word"] for item in segments),
        "segments": segments,
        "duration_ms": duration_ms,
    }


async def _transcribe_websocket(path: Path, api_key: str, timeout_s: float) -> dict:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            if wav_file.getcomptype() != "NONE":
                raise ValueError("SeedASR requires an uncompressed PCM WAV file")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"SeedASR could not read WAV audio: {path}") from exc
    # Send the complete WAV file (RIFF header + PCM) — the server's WAV decoder
    # rejects bare PCM chunks even when format is declared as "wav".
    wav_bytes = path.read_bytes()

    request = {
        "user": {"uid": str(uuid.uuid4())},
        "audio": {
            "format": "wav", "codec": "raw", "rate": sample_rate,
            "bits": sample_width * 8, "channel": channels,
        },
        "request": {
            "model_name": "bigmodel", "enable_itn": True, "enable_punc": True,
            "enable_ddc": True, "show_utterances": True, "result_type": "full",
        },
    }
    request_id = str(uuid.uuid4())
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": ASR_RESOURCE_ID,
        "X-Api-Request-Id": request_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1",
    }
    # The reference client sends roughly 200 ms of PCM in each audio frame.
    chunk_size = max(1, sample_rate * channels * sample_width // 5)
    try:
        async with asyncio.timeout(timeout_s):
            async with websockets.connect(
                ASR_WS_URL, additional_headers=headers, open_timeout=min(timeout_s, 30),
                max_size=64 * 1024 * 1024,
            ) as websocket:
                await websocket.send(_request_frame(_FULL_CLIENT_REQUEST, 1, json.dumps(request).encode()))
                sequence = 2
                chunks = [wav_bytes[i : i + chunk_size] for i in range(0, len(wav_bytes), chunk_size)] or [b""]
                for index, chunk in enumerate(chunks):
                    await websocket.send(_request_frame(
                        _AUDIO_ONLY_REQUEST, sequence, chunk, final=index == len(chunks) - 1
                    ))
                    sequence += 1

                latest_result: dict[str, Any] | None = None
                while True:
                    try:
                        message_type, response_sequence, payload = _decode_response(await websocket.recv())
                    except websockets.exceptions.ConnectionClosedOK:
                        # The server closes the connection normally after the
                        # final audio frame — treat this as end-of-recognition.
                        break
                    if message_type == _FULL_SERVER_RESPONSE:
                        latest_result = payload
                        if response_sequence is not None and response_sequence < 0:
                            break
                if latest_result is None:
                    raise RuntimeError("SeedASR closed without a recognition result")
                return _normalise_transcription(latest_result, _wav_duration_ms(path))
    except TimeoutError as exc:
        raise TimeoutError(f"SeedASR WebSocket timed out after {timeout_s:g}s ({request_id})") from exc
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        raise RuntimeError(f"SeedASR WebSocket connection failed ({request_id}): {exc}") from exc


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe a local WAV and return text plus word-level timestamps."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASR audio file does not exist: {path}")
    if os.getenv("HONCUT_ASR_MOCK") == "1":
        return _mock_transcription(path)

    api_key = os.getenv("ARK_AGENT_API_KEY") or os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("SeedASR requires ARK_AGENT_API_KEY or ARK_API_KEY")
    timeout_s = float(os.getenv("HONCUT_ASR_TIMEOUT_S", "300"))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_transcribe_websocket(path, api_key, timeout_s))
    raise RuntimeError("transcribe_audio cannot run inside an active asyncio event loop")
