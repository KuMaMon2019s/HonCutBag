"""HonCut edit decisions builder and executor.

Builds structured edit_decisions from shot videos, then executes
frame-accurate assembly with normalization, transitions, and audio handling.

Key capabilities:
1. Frame-accurate trimming (blackdetect + static frame removal)
2. Segment normalization (resolution/fps/codec/pix_fmt/sar)
3. Silent audio track injection (anullsrc for clips without audio)
4. Per-cut speed adjustment (setpts + atempo chain)
5. xfade transition chain (dissolve/fadeblack/cut)
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# ─── Probing ─────────────────────────────────────────────────────────────────

def probe_video(video_path: str) -> dict:
    """Probe a video file for duration, resolution, fps, audio presence."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        format_duration = float(data.get("format", {}).get("duration", 0))
        duration = format_duration
        video_duration = audio_duration = None
        width, height, fps = 1920, 1080, 30.0
        has_audio = has_video = False

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                has_video = True
                try:
                    video_duration = float(stream.get("duration"))
                except (TypeError, ValueError):
                    pass
                width = int(stream.get("width", 1920))
                height = int(stream.get("height", 1080))
                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / max(float(den), 1)
                else:
                    fps = float(fps_str)
            elif stream.get("codec_type") == "audio":
                has_audio = True
                try:
                    audio_duration = float(stream.get("duration"))
                except (TypeError, ValueError):
                    pass

        # The visual timeline is authoritative for video artifacts. Using the
        # container's longest-stream duration lets padded audio conceal a
        # truncated video stream from both assembly offsets and duration gates.
        if has_video and video_duration is not None:
            duration = video_duration

        return {
            "duration": round(duration, 3),
            "video_duration": None if video_duration is None else round(video_duration, 3),
            "audio_duration": None if audio_duration is None else round(audio_duration, 3),
            "format_duration": round(format_duration, 3),
            "width": width, "height": height,
            "fps": round(fps, 2),
            "has_audio": has_audio, "has_video": has_video,
        }
    except Exception as e:
        print(f"  [probe] Error: {e}")
        return {"duration": 0, "width": 1920, "height": 1080,
                "fps": 30, "has_audio": False, "has_video": False}


# ─── Black Frame Detection ───────────────────────────────────────────────────

def detect_black_frames(video_path: str, threshold: float = 0.1) -> dict:
    """Detect leading/trailing black or static frames to trim.

    Returns: {"trim_start": seconds, "trim_end": seconds}
    """
    info = probe_video(video_path)
    duration = info["duration"]
    if duration <= 0:
        return {"trim_start": 0, "trim_end": 0}

    trim_start = trim_end = 0.0

    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"blackdetect=d=0.2:pix_th={threshold}",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        blacks = re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", result.stderr)

        for start_s, end_s in blacks:
            sf, ef = float(start_s), float(end_s)
            if sf < 0.3:
                trim_start = max(trim_start, ef)
            if ef > duration - 0.3:
                trim_end = max(trim_end, duration - sf)
    except Exception:
        pass

    # Trim static first/last 0.1s (AI videos often have static frames)
    trim_start = max(trim_start, 0.1)
    trim_end = max(trim_end, 0.1)

    # Don't trim more than 20%
    max_trim = duration * 0.2
    trim_start = min(trim_start, max_trim)
    trim_end = min(trim_end, max_trim)

    return {"trim_start": round(trim_start, 2), "trim_end": round(trim_end, 2)}


# ─── Build Edit Decisions ────────────────────────────────────────────────────

def build_edit_decisions(
    shots_dir: str,
    target_width: int = 1920,
    target_height: int = 1080,
    transition_decisions: Optional[list] = None,
    quality_report: Optional[dict] = None,
    shot_order: Optional[list[str]] = None,
    target_duration: Optional[float] = None,
    transition_duration: float = 0.5,
) -> dict:
    """Build reviewed edit decisions from shot videos.

    For each shot: probe → merge automatic boundary trims → build cut entry.
    Shots still marked ``reshoot`` are rejected instead of silently assembled.
    """
    shots_dir = Path(shots_dir)
    available = {
        directory.name: directory
        for directory in shots_dir.iterdir()
        if directory.is_dir() and directory.name.startswith("S")
    }
    shot_dirs = (
        [available[name] for name in shot_order or [] if name in available]
        if shot_order
        else [available[name] for name in sorted(available)]
    )
    quality_shots = (quality_report or {}).get("shots", {})

    cuts = []
    for shot_dir in shot_dirs:
        video_path = shot_dir / "output.mp4"
        if not video_path.exists():
            continue

        info = probe_video(str(video_path))
        if info["duration"] <= 0:
            continue

        quality = quality_shots.get(shot_dir.name, {})
        if quality.get("action") == "reshoot":
            raise ValueError(
                f"{shot_dir.name} still requires reshoot: "
                f"{'; '.join(quality.get('reasons', []))}"
            )
        trims = detect_black_frames(str(video_path))
        in_s = max(float(trims["trim_start"]), float(quality.get("trim_start_s", 0.0)))
        quality_end = float(quality.get("trim_end_s", info["duration"]) or info["duration"])
        out_s = min(info["duration"] - float(trims["trim_end"]), quality_end)
        if out_s <= in_s:
            raise ValueError(
                f"{shot_dir.name} quality trims are invalid: {in_s:.3f}s-{out_s:.3f}s "
                f"for {info['duration']:.3f}s clip"
            )

        cut = {
            "source": str(video_path),
            "shot_id": shot_dir.name,
            "in_seconds": round(in_s, 2),
            "out_seconds": round(out_s, 2),
            "speed": 1.0,
            "has_audio": info["has_audio"],
            "original_duration": info["duration"],
            "trimmed": in_s > 0.15 or out_s < info["duration"] - 0.15,
            "quality_action": quality.get("action", "keep"),
            "quality_reasons": quality.get("reasons", []),
        }
        cuts.append(cut)

        trim_info = f" (裁切 {in_s:.1f}s-{out_s:.1f}s)" if cut["trimmed"] else ""
        audio_icon = "🔇" if not info["has_audio"] else "🔊"
        print(f"    • {shot_dir.name}: {info['duration']:.1f}s{trim_info} {audio_icon}")

    # Build transitions
    transitions = []
    if transition_decisions:
        for i, td in enumerate(transition_decisions):
            if i < len(cuts) - 1:
                transitions.append({
                    "index": i,
                    "type": td["decision"],
                    "duration": transition_duration,
                })

    # Crossfades overlap neighboring clips. Compensate with one bounded speed
    # factor so the assembled timeline stays close to the requested duration.
    if target_duration and cuts:
        source_duration = sum(cut["out_seconds"] - cut["in_seconds"] for cut in cuts)
        overlap = sum(
            0.01 if item["type"] == "cut" else float(item["duration"])
            for item in transitions
        )
        projected = source_duration - overlap
        if projected < float(target_duration):
            speed = source_duration / (float(target_duration) + overlap)
            if 0.85 <= speed < 1.0:
                for cut in cuts:
                    cut["speed"] = round(speed, 6)

    return {
        "cuts": cuts,
        "transitions": transitions,
        "metadata": {
            "compose_target": {"width": target_width, "height": target_height, "fit": "pad"},
            "target_fps": 30,
            "target_duration": target_duration,
            "quality_reviewed": bool(quality_report),
        },
    }


# ─── Execute Edit Decisions ──────────────────────────────────────────────────

def execute_edit_decisions(edit_decisions: dict, output_path: str) -> dict:
    """Execute: trim → normalize → transition → concat.

    Returns: {"success": bool, "output": str, "duration": float, "segments": int}
    """
    cuts = edit_decisions.get("cuts", [])
    transitions = edit_decisions.get("transitions", [])
    meta = edit_decisions.get("metadata", {})
    target = meta.get("compose_target", {})
    tw = target.get("width", 1920)
    th = target.get("height", 1080)
    tfps = meta.get("target_fps", 30)

    if not cuts:
        return {"success": False, "error": "No cuts"}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".edit_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files: list[Path] = []

    try:
        # Step 1: Trim + normalize each segment
        segments = []
        for i, cut in enumerate(cuts):
            src = Path(cut["source"])
            if not src.exists():
                print(f"  ⚠ Segment not found: {src}")
                continue

            seg = temp_dir / f"seg_{i:04d}.mp4"
            in_s = cut["in_seconds"]
            dur = cut["out_seconds"] - in_s
            speed = cut.get("speed", 1.0)
            has_audio = cut.get("has_audio", False)

            # --- P2-5c: cover fit mode（裁切填充，无黑边）---
            fit_mode = edit_decisions.get("metadata", {}).get("compose_target", {}).get("fit", "pad")
            if fit_mode == "cover":
                vf = [
                    f"scale={tw}:{th}:force_original_aspect_ratio=increase",
                    f"crop={tw}:{th}",
                    "setsar=1", f"fps={tfps}",
                ]
            else:
                # pad 模式（默认，带黑边）
                vf = [
                    f"scale={tw}:{th}:force_original_aspect_ratio=decrease",
                    f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "setsar=1", f"fps={tfps}",
                ]
            af = []
            if speed != 1.0:
                vf.append(f"setpts={1.0/speed}*PTS")
                af.append(f"atempo={speed}")

            if has_audio:
                cmd = ["ffmpeg", "-y", "-ss", str(in_s), "-t", str(dur), "-i", str(src),
                       "-filter:v", ",".join(vf)]
                if af:
                    cmd += ["-filter:a", ",".join(af)]
                cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-r", str(tfps),
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                        str(seg)]
            else:
                # Inject silent audio via lavfi
                cmd = ["ffmpeg", "-y", "-ss", str(in_s), "-t", str(dur),
                       "-i", str(src),
                       "-f", "lavfi", "-t", str(dur),
                       "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                       "-filter:v", ",".join(vf)]
                if af:
                    cmd += ["-filter:a", ",".join(af)]
                cmd += ["-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-r", str(tfps),
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                        str(seg)]

            try:
                subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                segments.append(str(seg))
                temp_files.append(seg)
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Segment {i} failed: {str(e.stderr)[:200] if e.stderr else e}")
                shutil.copy2(str(src), str(seg))
                segments.append(str(seg))
                temp_files.append(seg)

        if not segments:
            return {"success": False, "error": "No segments produced"}

        # Step 2: Concat with transitions
        if len(segments) == 1:
            shutil.copy2(segments[0], str(output_path))
        elif transitions and any(t["type"] != "cut" for t in transitions):
            _xfade_chain(segments, transitions, output_path, temp_dir, temp_files)
        else:
            _concat_copy(segments, output_path, temp_dir, temp_files)

        info = probe_video(str(output_path))
        return {"success": True, "output": str(output_path),
                "duration": info["duration"], "segments": len(segments)}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        for f in temp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── P2-B: Boundary Frame Consistency Check ──────────────────────────────────

def check_boundary_consistency(video_a: Path, video_b: Path) -> dict:
    """检查两个相邻视频的边界帧一致性（参考 HonCut frame sampler）。

    Returns:
        {"consistent": bool, "issues": [...], "recommended_transition": str}
    """
    issues = []

    def _probe(path):
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_streams", "-show_format", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return json.loads(result.stdout)
        except Exception:
            return None

    info_a = _probe(video_a)
    info_b = _probe(video_b)

    if not info_a or not info_b:
        return {"consistent": True, "issues": [], "recommended_transition": "dissolve"}

    # 检查分辨率一致性
    def _get_resolution(info):
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                return s.get("width", 0), s.get("height", 0)
        return (0, 0)

    res_a = _get_resolution(info_a)
    res_b = _get_resolution(info_b)
    if res_a != res_b:
        issues.append(f"分辨率不一致: {res_a} vs {res_b}")

    # 检查 fps 一致性
    def _get_fps(info):
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                fps_str = s.get("r_frame_rate", "30/1")
                try:
                    num, den = fps_str.split("/")
                    return float(num) / float(den)
                except Exception:
                    return 30.0
        return 30.0

    fps_a = _get_fps(info_a)
    fps_b = _get_fps(info_b)
    if abs(fps_a - fps_b) > 1.0:
        issues.append(f"FPS 不一致: {fps_a:.1f} vs {fps_b:.1f}")

    # 推荐转场
    if issues:
        recommended = "dissolve"  # 不一致时用 dissolve 过渡
    else:
        recommended = "cut"  # 一致时可以直接 cut

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "recommended_transition": recommended,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _concat_copy(segments: list, output_path: Path,
                 temp_dir: Path, temp_files: list):
    """Simple concat with stream copy (all hard cuts)."""
    concat_list = temp_dir / "concat_list.txt"
    temp_files.append(concat_list)
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{Path(seg).resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(concat_list), "-c", "copy", str(output_path)]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)


def _xfade_chain(segments: list, transitions: list, output_path: Path,
                 temp_dir: Path, temp_files: list):
    """Apply xfade transitions as bounded pairwise renders.

    FFmpeg 9 can truncate a long, nested xfade filter graph even when every
    input has normalized timestamps. Rendering one transition at a time keeps
    each graph simple and makes every intermediate duration independently
    probeable.
    """
    trans_map = {t["index"]: t for t in transitions}
    current = Path(segments[0])
    try:
        for i, next_segment in enumerate(segments[1:]):
            following = Path(next_segment)
            current_duration = probe_video(str(current))["duration"]
            if current_duration <= 0:
                raise RuntimeError(f"invalid xfade intermediate duration: {current}")

            t = trans_map.get(i, {"type": "cut", "duration": 0.5})
            ttype = t["type"]
            tdur = t.get("duration", 0.5)

            # --- P2-B: 边界帧一致性检查（参考 HonCut AI Clip Chaining）---
            try:
                boundary = check_boundary_consistency(current, following)
                if not boundary["consistent"]:
                    # 不一致时强制使用 dissolve，不用 cut
                    if ttype == "cut":
                        ttype = "dissolve"
                        print(f"    [P2-B] 边界不一致({boundary['issues'][0]})，cut→dissolve")
            except Exception:
                pass  # 检查失败不影响现有流程

            if ttype == "cut":
                tdur = 0.0
                filter_complex = (
                    "[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];"
                    "[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];"
                    "[v0][v1]concat=n=2:v=1:a=0[v];"
                    "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a0];"
                    "[1:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a1];"
                    "[a0][a1]concat=n=2:v=0:a=1[a]"
                )
            else:
                xname = {
                "dissolve": "fade", "fade": "fadeblack", "cut": "fade"
                }.get(ttype, "fade")
                offset = round(max(0, current_duration - tdur), 3)
                filter_complex = (
                    "[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];"
                    "[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];"
                    f"[v0][v1]xfade=transition={xname}:duration={tdur}:offset={offset}[v];"
                    "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a0];"
                    "[1:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a1];"
                    f"[a0][a1]acrossfade=d={tdur}[a]"
                )
            intermediate = temp_dir / f"xfade_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", str(current), "-i", str(following),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                str(intermediate),
            ]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            following_duration = probe_video(str(following))["duration"]
            rendered_duration = probe_video(str(intermediate))["duration"]
            expected_duration = current_duration + following_duration - tdur
            if abs(rendered_duration - expected_duration) > 0.25:
                raise RuntimeError(
                    "xfade intermediate video duration mismatch: "
                    f"expected={expected_duration:.3f}s actual={rendered_duration:.3f}s "
                    f"at transition {i}"
                )
            temp_files.append(intermediate)
            current = intermediate
        shutil.copy2(current, output_path)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        stderr = getattr(e, "stderr", None)
        print(f"  ⚠ xfade failed, fallback concat: {str(stderr)[:200] if stderr else e}")
        _concat_copy(segments, output_path, temp_dir, temp_files)
