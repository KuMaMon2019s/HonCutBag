#!/usr/bin/env python3
"""
rhythm_editor.py — Phase 8 后处理节奏模块

对视频做节奏调整：
  - 变速（高潮加速 / 抒情减速）
  - 卡点（按 BGM 节拍切）
  - 转场精修（按情绪选 dissolve / wipe / fade / cut）

输入：visual_final.mp4 + STORYBOARD.json + 可选 BGM
输出：polished.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# OpenMontage 工具可选导入（graceful fallback 到 ffmpeg）
# ---------------------------------------------------------------------------
OM_TOOLS_DIR = "/Users/soda/projects/OpenMontage/tools"
OM_AVAILABLE = False

if os.path.isdir(OM_TOOLS_DIR):
    sys.path.insert(0, OM_TOOLS_DIR)
    try:
        from analysis.audio_energy import AudioEnergy
        from analysis.scene_detect import SceneDetect
        OM_AVAILABLE = True
    except Exception:
        OM_AVAILABLE = False


# ---------------------------------------------------------------------------
# 情绪 → 速度 / 转场 映射
# ---------------------------------------------------------------------------

# 情绪关键词 → 速度因子
_EMOTION_SPEED: dict[str, float] = {
    # 紧张/冲突/追逐 → 1.2x
    "紧张": 1.2, "冲突": 1.2, "追逐": 1.2, "tense": 1.2,
    "conflict": 1.2, "chase": 1.2, "action": 1.2, "intense": 1.2,
    # 温暖/感动/回忆 → 0.85x
    "温暖": 0.85, "感动": 0.85, "回忆": 0.85, "warm": 0.85,
    "touching": 0.85, "memory": 0.85, "nostalgia": 0.85,
    # 悲伤/离别 → 0.8x
    "悲伤": 0.8, "离别": 0.8, "sad": 0.8, "farewell": 0.8,
    "sorrow": 0.8, "melancholy": 0.8, "grief": 0.8,
    # 惊喜/发现 → 1.1x
    "惊喜": 1.1, "发现": 1.1, "surprise": 1.1, "discovery": 1.1,
    "wonder": 1.1, "excitement": 1.1,
    # 平静/日常 → 1.0x
    "平静": 1.0, "日常": 1.0, "calm": 1.0, "daily": 1.0,
    "peaceful": 1.0, "neutral": 1.0, "normal": 1.0,
}

# 情绪关键词 → 转场类型
_EMOTION_TRANSITION: dict[str, str] = {
    "紧张": "cut", "冲突": "cut", "追逐": "cut",
    "tense": "cut", "conflict": "cut", "chase": "cut", "action": "cut",
    "温暖": "dissolve", "感动": "dissolve", "回忆": "dissolve",
    "warm": "dissolve", "touching": "dissolve", "memory": "dissolve",
    "悲伤": "fade", "离别": "fade", "sad": "fade", "farewell": "fade",
    "sorrow": "fade", "melancholy": "fade",
    "惊喜": "wipe", "发现": "wipe", "surprise": "wipe", "discovery": "wipe",
    "wonder": "wipe",
    "平静": "dissolve", "日常": "dissolve", "calm": "dissolve",
    "daily": "dissolve", "peaceful": "dissolve", "neutral": "dissolve",
}


def emotion_to_speed(emotion: str) -> float:
    """情绪 → 速度因子映射。

    在 emotion 字符串中查找关键词（不区分大小写），返回对应速度。
    未匹配时返回 1.0（正常速度）。
    """
    emotion_lower = emotion.lower()
    for keyword, speed in _EMOTION_SPEED.items():
        if keyword in emotion_lower:
            return speed
    return 1.0


def emotion_to_transition(emotion: str) -> str:
    """情绪 → 转场类型映射。

    返回: "cut" | "dissolve" | "fade" | "wipe"
    """
    emotion_lower = emotion.lower()
    for keyword, trans in _EMOTION_TRANSITION.items():
        if keyword in emotion_lower:
            return trans
    return "dissolve"  # 默认


# ---------------------------------------------------------------------------
# 辅助工具函数
# ---------------------------------------------------------------------------

def _run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """运行 shell 命令，失败时抛异常。"""
    print(f"  [ffmpeg] {desc or ' '.join(cmd[:4])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-500:]}")
    return result


def _probe_duration(path: str) -> float:
    """用 ffprobe 获取视频时长（秒）。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _probe_fps(path: str) -> float:
    """用 ffprobe 获取视频帧率。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frac = result.stdout.strip()
    if "/" in frac:
        num, den = frac.split("/")
        return float(num) / float(den)
    return float(frac)


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def analyze_beats(bgm_path: str) -> list[float]:
    """检测 BGM 节拍时间点（秒）。

    优先用 OM audio_energy 分析能量峰值，fallback 到 ffmpeg silencedetect。
    """
    if OM_AVAILABLE:
        try:
            tool = AudioEnergy()
            result = tool.run(input_path=bgm_path)
            # OM AudioEnergy 返回 peaks 列表（时间戳秒）
            if isinstance(result, dict) and "peaks" in result:
                return sorted(result["peaks"])
            if isinstance(result, list):
                return sorted(result)
        except Exception as e:
            print(f"  [warn] OM audio_energy failed: {e}, fallback to ffmpeg")

    # Fallback: 用 ffmpeg silencedetect 找静音段后的能量突增点
    # 简化方案：用 aeval 检测能量超过阈值的帧
    cmd = [
        "ffmpeg", "-i", bgm_path,
        "-af", "silencedetect=noise=-30dB:d=0.3,ametadata=print:file=-",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    beats: list[float] = []
    for line in result.stderr.split("\n"):
        if "lavfi.silence_end" in line:
            try:
                t = float(line.split("silence_end")[-1].split()[0].strip())
                beats.append(t)
            except (ValueError, IndexError):
                pass
    return sorted(beats) if beats else [0.0]


def detect_scene_cuts(video_path: str) -> list[float]:
    """检测视频场景切换点（秒）。

    优先用 OM scene_detect，fallback 到 ffmpeg select filter。
    """
    if OM_AVAILABLE:
        try:
            tool = SceneDetect()
            result = tool.execute({"input_path": video_path})
            if isinstance(result, dict) and "cuts" in result:
                return sorted(result["cuts"])
            if isinstance(result, list):
                return sorted(result)
        except Exception as e:
            print(f"  [warn] OM scene_detect failed: {e}, fallback to ffmpeg")

    # Fallback: ffmpeg scene detection via select filter
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    cuts: list[float] = []
    for line in result.stderr.split("\n"):
        if "showinfo" in line and "pts_time:" in line:
            try:
                pts = line.split("pts_time:")[1].split()[0]
                cuts.append(float(pts))
            except (ValueError, IndexError):
                pass
    return sorted(cuts)


def apply_speed_ramp(
    video_path: str,
    speed_map: dict[int, float],
    scene_cuts: list[float],
    output_path: str = "speed_ramped.mp4",
) -> str:
    """按 shot 分段变速。

    speed_map: {segment_index: speed_factor}
    scene_cuts: 场景切换时间点列表（用于确定分段边界）
    """
    if not speed_map or not scene_cuts:
        # 无变速需求，直接拷贝
        if video_path != output_path:
            _run(["cp", video_path, output_path], "copy (no speed change)")
        return output_path

    duration = _probe_duration(video_path)

    # 构建分段边界: [0, cut1, cut2, ..., duration]
    boundaries = [0.0] + scene_cuts + [duration]

    # 检查是否所有段都是 1.0x（无需变速）
    all_normal = all(speed_map.get(i, 1.0) == 1.0 for i in range(len(boundaries) - 1))
    if all_normal:
        if video_path != output_path:
            _run(["cp", video_path, output_path], "copy (all 1.0x)")
        return output_path

    # 用 ffmpeg 复杂滤镜分段变速
    # 策略：split → 每段 trim + setpts + atempo → concat
    n_segments = len(boundaries) - 1

    # 构建 filter_complex
    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i in range(n_segments):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_dur = end - start
        speed = speed_map.get(i, 1.0)

        # 视频: trim → setpts
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
            f"setpts={1.0/speed}*PTS[v{i}]"
        )
        # 音频: atrim → atempo (atempo 范围 0.5~2.0)
        atempo_val = speed
        # atempo 只支持 0.5~2.0，超出需要链式
        if atempo_val < 0.5:
            atempo_chain = f"atempo=0.5,atempo={atempo_val/0.5:.4f}"
        elif atempo_val > 2.0:
            atempo_chain = f"atempo=2.0,atempo={atempo_val/2.0:.4f}"
        else:
            atempo_chain = f"atempo={atempo_val:.4f}"

        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,"
            f"{atempo_chain}[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    # concat 所有段
    concat_str = "".join(concat_inputs)
    filter_parts.append(
        f"{concat_str}concat=n={n_segments}:v=1:a=1[outv][outa]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    _run(cmd, f"speed ramp ({n_segments} segments)")
    return output_path


def refine_transitions(
    video_path: str,
    transitions: list[dict],
    scene_cuts: list[float],
    output_path: str = "transitions_refined.mp4",
) -> str:
    """在场景切换点叠加转场效果。

    transitions: [{"at": 5.0, "type": "dissolve", "duration": 0.5}, ...]
    简化方案：输出 transition_edl.json 供后续处理，同时尝试用 ffmpeg xfade。
    """
    if not transitions:
        if video_path != output_path:
            _run(["cp", video_path, output_path], "copy (no transitions)")
        return output_path

    # 写 EDL 供手动调整或后续管线使用
    edl_path = output_path.rsplit(".", 1)[0] + "_transition_edl.json"
    with open(edl_path, "w") as f:
        json.dump({"transitions": transitions, "scene_cuts": scene_cuts}, f, indent=2)
    print(f"  [edl] wrote {edl_path}")

    # 尝试用 ffmpeg xfade 实现转场
    # xfade 需要两个输入段，对单文件需要切割再拼接
    # 简化：只处理有明确转场类型的点，用 overlay + fade 模拟
    if len(transitions) == 0:
        return output_path

    # 对于单文件转场精修，使用 zoompan + fade 等滤镜叠加
    # 这里采用务实方案：输出 EDL + 标记，不做复杂重切
    # （完整实现需要 video_stitch 配合，留给管线上层调用）
    print(f"  [info] {len(transitions)} transitions marked in EDL")
    print(f"  [info] for full transition rendering, use video_stitch with EDL")

    # 如果只有 1 个转场点，可以尝试简单 xfade
    if len(transitions) == 1 and len(scene_cuts) >= 1:
        t = transitions[0]
        at_sec = t["at"]
        t_type = t.get("type", "dissolve")
        t_dur = t.get("duration", 0.5)

        # xfade 映射
        xfade_map = {
            "dissolve": "dissolve",
            "wipe": "slideleft",
            "fade": "fade",
            "cut": None,  # cut 不需要 xfade
        }
        xfade_type = xfade_map.get(t_type)
        if xfade_type is None:
            # cut = 硬切，不需要额外处理
            if video_path != output_path:
                _run(["cp", video_path, output_path], "copy (cut transition)")
            return output_path

        duration = _probe_duration(video_path)
        offset = max(0, at_sec - t_dur)

        # 用 xfade 对单文件做自交叉（简化版）
        # 实际上 xfade 需要两段输入，这里用 fade 滤镜模拟
        if t_type == "fade":
            # 在切换点加一个短暂 fade
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fade=t=in:st={at_sec}:d={t_dur},"
                       f"fade=t=out:st={at_sec + t_dur}:d={t_dur}",
                "-af", f"afade=t=in:st={at_sec}:d={t_dur},"
                       f"afade=t=out:st={at_sec + t_dur}:d={t_dur}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", output_path,
            ]
            try:
                _run(cmd, f"fade transition at {at_sec}s")
                return output_path
            except RuntimeError:
                pass  # fallback to copy

    # 默认：拷贝原文件 + EDL
    if video_path != output_path:
        _run(["cp", video_path, output_path], "copy (transitions in EDL)")
    return output_path


def beat_sync_adjust(
    scene_cuts: list[float],
    beats: list[float],
    tolerance: float = 0.3,
    max_shift: float = 1.0,
) -> list[float]:
    """将场景切换点对齐到最近的节拍点。

    tolerance: 容差（秒），偏差在此范围内才调整
    max_shift: 最大偏移（秒），超过则不强制对齐
    """
    if not beats or not scene_cuts:
        return scene_cuts

    adjusted: list[float] = []
    for cut in scene_cuts:
        # 找最近的节拍
        best_beat = min(beats, key=lambda b: abs(b - cut))
        delta = best_beat - cut
        if abs(delta) <= tolerance and abs(delta) <= max_shift:
            adjusted.append(best_beat)
        else:
            adjusted.append(cut)
    return sorted(adjusted)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def edit_rhythm(
    video_path: str,
    storyboard_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    enable_speed_ramp: bool = True,
    enable_beat_sync: bool = True,
    enable_transition_refine: bool = True,
    output_path: str = "polished.mp4",
) -> str:
    """主入口：节奏编辑。

    流程：
      1. 分析节奏（BGM 节拍检测 / 场景切割点）
      2. 变速调整（按情绪映射）
      3. 转场精修（按情绪选转场类型）
      4. 输出 polished.mp4
    """
    print(f"\n{'='*60}")
    print(f"Phase 8 — Rhythm Editor")
    print(f"{'='*60}")
    print(f"  input:    {video_path}")
    print(f"  output:   {output_path}")
    print(f"  options:  speed={enable_speed_ramp} beat={enable_beat_sync} "
          f"transition={enable_transition_refine}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 读取 storyboard
    shots: list[dict] = []
    if storyboard_path and os.path.isfile(storyboard_path):
        with open(storyboard_path) as f:
            sb = json.load(f)
        # 支持多种格式
        if isinstance(sb, list):
            shots = sb
        elif isinstance(sb, dict):
            shots = sb.get("shots", sb.get("segments", []))
        print(f"  storyboard: {len(shots)} shots loaded")

    # ---- Step 1: 分析节奏 ----
    print(f"\n[1/3] Analyzing rhythm...")

    # 场景切割检测
    scene_cuts = detect_scene_cuts(video_path)
    print(f"  scene cuts: {len(scene_cuts)} detected")

    # BGM 节拍检测
    beats: list[float] = []
    if bgm_path and os.path.isfile(bgm_path):
        beats = analyze_beats(bgm_path)
        print(f"  bgm beats: {len(beats)} detected")

    # 卡点对齐
    if enable_beat_sync and beats and scene_cuts:
        scene_cuts = beat_sync_adjust(scene_cuts, beats)
        print(f"  beat-synced cuts: {len(scene_cuts)}")

    # ---- Step 2: 变速调整 ----
    current_path = video_path
    if enable_speed_ramp and shots:
        print(f"\n[2/3] Applying speed ramp...")
        speed_map: dict[int, float] = {}
        for i, shot in enumerate(shots):
            emotion = shot.get("emotion", "")
            speed = emotion_to_speed(emotion)
            if speed != 1.0:
                speed_map[i] = speed

        if speed_map:
            ramped_path = output_path.rsplit(".", 1)[0] + "_ramped.mp4"
            apply_speed_ramp(video_path, speed_map, scene_cuts, ramped_path)
            current_path = ramped_path
            # 变速后 scene_cuts 时间点会变化，需要重新检测或按比例调整
            # 简化：重新检测
            scene_cuts = detect_scene_cuts(current_path)
        else:
            print(f"  no speed changes needed (all 1.0x)")
    else:
        print(f"\n[2/3] Speed ramp skipped")

    # ---- Step 3: 转场精修 ----
    if enable_transition_refine and shots:
        print(f"\n[3/3] Refining transitions...")
        transitions: list[dict] = []
        for i, shot in enumerate(shots):
            emotion = shot.get("emotion", "")
            trans_type = shot.get("transition_to_next", "")
            if not trans_type:
                trans_type = emotion_to_transition(emotion)

            # 转场位置 = 下一个 scene_cut
            if i < len(scene_cuts):
                at_sec = scene_cuts[i]
            elif i < len(shots) - 1:
                # 估算位置
                duration = _probe_duration(current_path)
                at_sec = duration * (i + 1) / len(shots)
            else:
                continue

            # 转场时长按情绪
            dur_map = {"cut": 0.0, "dissolve": 0.8, "fade": 1.0, "wipe": 0.6}
            t_dur = dur_map.get(trans_type, 0.5)

            transitions.append({
                "at": round(at_sec, 3),
                "type": trans_type,
                "duration": t_dur,
                "shot_index": i,
            })

        if transitions:
            refined_path = output_path
            refine_transitions(current_path, transitions, scene_cuts, refined_path)
            current_path = refined_path
        else:
            print(f"  no transitions to refine")
    else:
        print(f"\n[3/3] Transition refine skipped")

    # ---- 最终输出 ----
    if current_path != output_path:
        _run(["cp", current_path, output_path], "final copy")

    # 清理中间文件
    ramped = output_path.rsplit(".", 1)[0] + "_ramped.mp4"
    if os.path.isfile(ramped) and ramped != output_path:
        os.remove(ramped)

    print(f"\n{'='*60}")
    print(f"✅ Rhythm edit complete → {output_path}")
    print(f"{'='*60}\n")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 Rhythm Editor — 视频节奏调整（变速/卡点/转场精修）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rhythm_editor.py --input visual_final.mp4 --storyboard STORYBOARD.json --output polished.mp4
  python rhythm_editor.py --input visual_final.mp4 --bgm music.mp3 --output polished.mp4
  python rhythm_editor.py --input visual_final.mp4 --no-speed --no-beat --output polished.mp4
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="输入视频路径 (visual_final.mp4)")
    parser.add_argument("--storyboard", "-s", default=None, help="STORYBOARD.json 路径")
    parser.add_argument("--bgm", "-b", default=None, help="BGM 音频文件路径（用于卡点分析）")
    parser.add_argument("--output", "-o", default="polished.mp4", help="输出视频路径")
    parser.add_argument("--no-speed", action="store_true", help="禁用变速调整")
    parser.add_argument("--no-beat", action="store_true", help="禁用卡点对齐")
    parser.add_argument("--no-transition", action="store_true", help="禁用转场精修")

    args = parser.parse_args()

    edit_rhythm(
        video_path=args.input,
        storyboard_path=args.storyboard,
        bgm_path=args.bgm,
        enable_speed_ramp=not args.no_speed,
        enable_beat_sync=not args.no_beat,
        enable_transition_refine=not args.no_transition,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
