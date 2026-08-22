"""Subtitle rendering and ASR-to-caption timeline conversion."""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Optional


def _probe_shot_duration(shots_dir: Path, shot_id: int) -> float:
    """Probe the real duration of a shot video via ffprobe.

    Falls back to 2.0s if the file is missing or ffprobe fails.
    """
    shot_video = shots_dir / f"S{shot_id:02d}" / "output.mp4"
    if not shot_video.exists():
        return 2.0
    try:
        import subprocess as _sp
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(shot_video),
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return 2.0


def _write_srt(segments: list, srt_path: str) -> None:
    """Write segments to an SRT subtitle file as fallback."""
    import os
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    lines = []
    for idx, seg in enumerate(segments, 1):
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)
        text = seg.get("text", "")
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}")
        lines.append(text)
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_subtitle_text(text: str) -> str:
    """去除中英文标点并规范空白。

    中文分词由 ASR word segment 边界提供；本函数不猜测中文词界，
    因此没有 word 级数据时只去标点，不强行切字。
    """
    import unicodedata

    if not isinstance(text, str):
        return ""
    without_punctuation = "".join(
        character for character in text
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _merge_shot_transcripts(
    sb_shots: list,
    durations_ms: list[int],
    shot_transcripts: list[dict],
    edit_timeline: Optional[dict] = None,
) -> dict:
    """Offset per-shot ASR words and create caption-burn segments.

    In a scripted-dialogue scene, the authored dialogue contract is the source
    of truth: ASR may validate audible speech but cannot replace the line or
    invent dialogue on a shot explicitly authored without one. In a fully
    unscripted scene, ASR remains authoritative.
    """
    merged_words = []
    caption_segments = []
    cumulative_ms = 0
    shot_entries = []
    timeline_by_shot = {
        str(item.get("shot_id")): item
        for item in (edit_timeline or {}).get("shots", [])
        if isinstance(item, dict) and item.get("shot_id")
    }
    scripted_scene = any(
        clean_subtitle_text(
            (shot.get("dialogue") or {}).get("line", "")
            if isinstance(shot.get("dialogue"), dict)
            else str(shot.get("dialogue") or "")
        )
        for shot in sb_shots
    )
    for index, (shot, duration_ms, transcription) in enumerate(
        zip(sb_shots, durations_ms, shot_transcripts), 1
    ):
        shot_id = str(shot.get("shot_id") or f"S{index:02d}")
        timeline_item = timeline_by_shot.get(shot_id)
        if timeline_item:
            shot_start_ms = round(float(timeline_item.get("output_start_s", 0.0)) * 1000)
            shot_duration_ms = round(float(timeline_item.get("output_duration_s", 0.0)) * 1000)
            source_in_ms = round(float(timeline_item.get("source_in_s", 0.0)) * 1000)
            speed = float(timeline_item.get("speed", 1.0) or 1.0)
        else:
            shot_start_ms = cumulative_ms
            shot_duration_ms = duration_ms
            source_in_ms = 0
            speed = 1.0
        if duration_ms <= 0 or transcription.get("skipped"):
            # Shot missing output.mp4 — skip caption generation entirely
            shot_entries.append({
                "shot_id": shot_id,
                "text": "",
                "source": "skipped",
                "start_ms": shot_start_ms,
                "end_ms": shot_start_ms,
                "segments": [],
            })
            continue
        local_words = transcription.get("segments") or []
        asr_text = clean_subtitle_text(transcription.get("text") or "")
        dialogue = shot.get("dialogue")
        dialogue_line = (
            dialogue.get("line", "") if isinstance(dialogue, dict) else str(dialogue or "")
        )
        scripted_text = clean_subtitle_text(dialogue_line)
        if scripted_text:
            text = scripted_text
            source = "dialogue_script"
            words = [{
                "word": text,
                "start_ms": round(shot_start_ms + shot_duration_ms * 0.2),
                "end_ms": round(shot_start_ms + shot_duration_ms * 0.8),
                "source": source,
            }]
        elif scripted_scene:
            # Do not let ASR hallucinations introduce dialogue into a shot that
            # the authored scripted scene explicitly leaves silent.
            words = []
            text = ""
            source = "none"
        elif local_words or asr_text:
            words = []
            for item in local_words:
                cleaned_word = clean_subtitle_text(item.get("word") or item.get("text") or "")
                if not cleaned_word:
                    continue
                source_start_ms = int(item["start_ms"])
                source_end_ms = int(item["end_ms"])
                # Words outside the retained source window must not leak into
                # the edited timeline after head/tail trims.
                source_out_ms = source_in_ms + round(shot_duration_ms * speed)
                if source_end_ms <= source_in_ms or source_start_ms >= source_out_ms:
                    continue
                mapped_start_ms = shot_start_ms + max(
                    0, round((source_start_ms - source_in_ms) / speed)
                )
                mapped_end_ms = min(
                    shot_start_ms + shot_duration_ms,
                    shot_start_ms + max(
                        0, round((source_end_ms - source_in_ms) / speed)
                    ),
                )
                if mapped_end_ms <= mapped_start_ms:
                    continue
                words.append({
                    "word": cleaned_word,
                    "start_ms": mapped_start_ms,
                    "end_ms": mapped_end_ms,
                    "source": "asr",
                })
            text = " ".join(item["word"] for item in words) if words else asr_text
            if not words and text:
                words = [{
                    "word": text,
                    "start_ms": shot_start_ms,
                    "end_ms": shot_start_ms + shot_duration_ms,
                    "source": "asr",
                }]
            source = "asr"
        else:
            words = []
            text = ""
            source = "none"

        merged_words.extend(words)
        shot_entries.append({
            "shot_id": shot_id,
            "text": text,
            "source": source,
            "start_ms": shot_start_ms,
            "end_ms": shot_start_ms + shot_duration_ms,
            "segments": words,
        })
        if words:
            caption_segments.append({
                "text": text,
                "start": words[0]["start_ms"] / 1000,
                "end": words[-1]["end_ms"] / 1000,
                "source": source,
                "words": [{
                    "word": item["word"],
                    "start": item["start_ms"] / 1000,
                    "end": item["end_ms"] / 1000,
                    "source": item["source"],
                } for item in words],
            })
        cumulative_ms += duration_ms
    output_duration_ms = round(
        float((edit_timeline or {}).get("duration_s", 0.0)) * 1000
    ) or cumulative_ms
    return {
        "text": "".join(entry["text"] for entry in shot_entries if entry["text"]),
        "duration_ms": output_duration_ms,
        "segments": merged_words,
        "shots": shot_entries,
        "caption_segments": caption_segments,
    }


def _caption_segments_from_final_asr(transcription: dict) -> list[dict]:
    """Build cue-preserving captions from the audible final mix."""
    captions = []
    utterances = transcription.get("utterances") or []
    if not utterances and (transcription.get("segments") or transcription.get("text")):
        utterances = [{
            "text": transcription.get("text", ""),
            "words": transcription.get("segments") or [],
        }]
    for utterance in utterances:
        cleaned_words = []
        for item in utterance.get("words") or []:
            cleaned = clean_subtitle_text(item.get("word") or item.get("text") or "")
            start_ms = int(item.get("start_ms", item.get("start_time", -1)))
            end_ms = int(item.get("end_ms", item.get("end_time", -1)))
            if not cleaned or start_ms < 0 or end_ms <= start_ms:
                continue
            cleaned_words.append({
                "word": cleaned,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
                "source": "final_mix_asr",
            })
        word_groups = []
        for word in cleaned_words:
            if word_groups and word["start"] - word_groups[-1][-1]["end"] >= 0.25:
                word_groups.append([])
            if not word_groups:
                word_groups.append([])
            word_groups[-1].append(word)
        if not word_groups:
            text = clean_subtitle_text(utterance.get("text") or "")
            start_ms = int(utterance.get("start_ms", utterance.get("start_time", -1)))
            end_ms = int(utterance.get("end_ms", utterance.get("end_time", -1)))
            if not text or start_ms < 0 or end_ms <= start_ms:
                continue
            word_groups = [[{
                "word": text,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
                "source": "final_mix_asr",
            }]]
        for group in word_groups:
            # Generated impacts and metallic transients can be decoded as a
            # run of implausibly short syllables. Without confidence scores,
            # reject only the narrow high-speed pattern that repeatedly caused
            # unstable text across ASR passes; normal short interjections stay.
            mean_word_duration = sum(
                item["end"] - item["start"] for item in group
            ) / len(group)
            if len(group) >= 2 and mean_word_duration < 0.1:
                continue
            captions.append({
                "text": "".join(item["word"] for item in group),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "source": "final_mix_asr",
                "words": group,
            })
    return captions
