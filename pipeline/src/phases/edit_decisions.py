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

        duration = float(data.get("format", {}).get("duration", 0))
        width, height, fps = 1920, 1080, 30.0
        has_audio = has_video = False

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                has_video = True
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

        return {
            "duration": round(duration, 3),
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
) -> dict:
    """Build edit_decisions from shot videos.

    For each shot: probe → detect black frames → build cut entry.
    """
    shots_dir = Path(shots_dir)
    shot_dirs = sorted(d for d in shots_dir.iterdir()
                       if d.is_dir() and d.name.startswith("S"))

    cuts = []
    for shot_dir in shot_dirs:
        video_path = shot_dir / "output.mp4"
        if not video_path.exists():
            continue

        info = probe_video(str(video_path))
        if info["duration"] <= 0:
            continue

        trims = detect_black_frames(str(video_path))
        in_s = trims["trim_start"]
        out_s = info["duration"] - trims["trim_end"]
        if out_s <= in_s:
            in_s, out_s = 0, info["duration"]

        cut = {
            "source": str(video_path),
            "shot_id": shot_dir.name,
            "in_seconds": round(in_s, 2),
            "out_seconds": round(out_s, 2),
            "speed": 1.0,
            "has_audio": info["has_audio"],
            "original_duration": info["duration"],
            "trimmed": trims["trim_start"] > 0.15 or trims["trim_end"] > 0.15,
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
                    "duration": 0.5,
                })

    return {
        "cuts": cuts,
        "transitions": transitions,
        "metadata": {
            "compose_target": {"width": target_width, "height": target_height, "fit": "pad"},
            "target_fps": 30,
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
    """Apply xfade transitions between segments."""
    n = len(segments)
    durations = [probe_video(s)["duration"] for s in segments]
    trans_map = {t["index"]: t for t in transitions}

    input_args = []
    for seg in segments:
        input_args += ["-i", seg]

    vf, af = [], []
    cum_offset = 0.0

    for i in range(n - 1):
        t = trans_map.get(i, {"type": "cut", "duration": 0.5})
        ttype = t["type"]
        tdur = t.get("duration", 0.5)

        # --- P2-B: 边界帧一致性检查（参考 HonCut AI Clip Chaining）---
        try:
            boundary = check_boundary_consistency(Path(segments[i]), Path(segments[i + 1]))
            if not boundary["consistent"]:
                # 不一致时强制使用 dissolve，不用 cut
                if ttype == "cut":
                    ttype = "dissolve"
                    print(f"    [P2-B] 边界不一致({boundary['issues'][0]})，cut→dissolve")
        except Exception:
            pass  # 检查失败不影响现有流程

        xname = {"dissolve": "fade", "fade": "fadeblack", "cut": "fade"}.get(ttype, "fade")
        if ttype == "cut":
            tdur = 0.01

        offset = round(max(0, cum_offset + durations[i] - tdur), 3)

        if i == 0:
            vf.append(f"[0:v][1:v]xfade=transition={xname}:duration={tdur}:offset={offset}[v{i}]")
            af.append(f"[0:a][1:a]acrossfade=d={tdur}[a{i}]")
        else:
            vf.append(f"[v{i-1}][{i+1}:v]xfade=transition={xname}:duration={tdur}:offset={offset}[v{i}]")
            af.append(f"[a{i-1}][{i+1}:a]acrossfade=d={tdur}[a{i}]")
        cum_offset = offset

    last = n - 2
    fc = ";".join(vf + af)
    cmd = ["ffmpeg", "-y", *input_args,
           "-filter_complex", fc,
           "-map", f"[v{last}]", "-map", f"[a{last}]",
           "-c:v", "libx264", "-crf", "23", "-preset", "medium",
           "-c:a", "aac", "-b:a", "192k",
           str(output_path)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ xfade failed, fallback concat: {str(e.stderr)[:200] if e.stderr else e}")
        _concat_copy(segments, output_path, temp_dir, temp_files)
