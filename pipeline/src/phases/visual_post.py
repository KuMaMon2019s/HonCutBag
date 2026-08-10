#!/usr/bin/env python3
"""visual_post.py — Phase 8 视觉后处理模块

对 audio_mixed.mp4 做视觉后处理：
  1. 画质增强（降噪 + 锐化 + 可选超分）
  2. 画幅适配（16:9 → 9:16 竖屏 / 1:1 方形）
  3. 片头（标题卡，2-3 秒，淡入）
  4. 片尾（演员表/制作信息，3-5 秒，淡出）

优先使用 HonCut 工具，不可用时 graceful fallback 到 ffmpeg。

Usage:
    python visual_post.py --input audio_mixed.mp4 --title "雪狼传说" --output visual_final.mp4
    python visual_post.py --input audio_mixed.mp4 --ratio 9:16 --no-intro --output vertical.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 第三方兼容工具路径
# ---------------------------------------------------------------------------
OM_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "video_tools" / "tools"
OM_AVAILABLE = OM_TOOLS_DIR.exists()

if OM_AVAILABLE:
    _om_parent = OM_TOOLS_DIR.parent  # vendor package root
    if str(_om_parent) not in sys.path:
        sys.path.insert(0, str(_om_parent))

# ---------------------------------------------------------------------------
# 工具检测
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _try_import_om_upscale():
    """尝试导入 OM Upscale 工具"""
    if not OM_AVAILABLE:
        return None
    try:
        from vendor.video_tools.tools.enhancement.upscale import Upscale
        return Upscale
    except (ImportError, Exception):
        return None


def _try_import_om_reframe():
    """尝试导入 OM AutoReframe 工具"""
    if not OM_AVAILABLE:
        return None
    try:
        from vendor.video_tools.tools.video.auto_reframe import AutoReframe
        return AutoReframe
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# 辅助：获取视频信息
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> dict:
    """用 ffprobe 获取视频宽高、时长等信息"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return {
                    "width": int(s.get("width", 0)),
                    "height": int(s.get("height", 0)),
                    "duration": float(data.get("format", {}).get("duration", 0)),
                    "codec": s.get("codec_name", "unknown"),
                    "fps": s.get("avg_frame_rate") or s.get("r_frame_rate") or "24",
                }
    except Exception:
        pass
    return {"width": 0, "height": 0, "duration": 0, "codec": "unknown", "fps": "24"}


def _run_cmd(cmd: list[str], desc: str = "") -> bool:
    """运行命令，返回是否成功"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  ⚠️  {desc} 失败: {r.stderr[:300]}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"  ⚠️  {desc} 异常: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# 1. 画质增强
# ---------------------------------------------------------------------------

def enhance_video(
    video_path: str,
    output_path: str = "enhanced.mp4",
    denoise: bool = True,
    sharpen: bool = True,
    upscale: bool = False,
) -> str:
    """画质增强：降噪 + 锐化 + 可选超分

    优先用 OM enhancement/ 工具，fallback 到 ffmpeg 滤镜。
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")

    print(f"[enhance] 输入: {video_path}")

    # --- 尝试 OM Upscale（仅当 upscale=True）---
    if upscale:
        UpscaleCls = _try_import_om_upscale()
        if UpscaleCls is not None:
            try:
                print("[enhance] 使用 OM Upscale (Real-ESRGAN)...")
                tool = UpscaleCls()
                result = tool.run(
                    input_path=video_path,
                    output_path=output_path,
                    model_name="RealESRGAN_x4plus",
                )
                if result and result.get("status") == "success":
                    print(f"[enhance] ✅ OM Upscale 完成")
                    return output_path
            except Exception as e:
                print(f"[enhance] OM Upscale 失败，fallback 到 ffmpeg: {e}")

    # --- FFmpeg fallback: hqdn3d 降噪 + unsharp 锐化 ---
    if not _ffmpeg_available():
        print("[enhance] ⚠️ ffmpeg 不可用，跳过增强，直接复制")
        shutil.copy2(video_path, output_path)
        return output_path

    filters = []
    if denoise:
        # hqdn3d: luma_spatial=3, chroma_spatial=2, luma_tmp=4
        filters.append("hqdn3d=3:2:4:2")
    if sharpen:
        # unsharp: 5x5 luma matrix, amount=0.8, 3x3 chroma, amount=0.4
        filters.append("unsharp=5:5:0.8:3:3:0.4")

    if not filters:
        print("[enhance] 无需滤镜，直接复制")
        shutil.copy2(video_path, output_path)
        return output_path

    vf = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    print(f"[enhance] ffmpeg 滤镜: {vf}")
    if _run_cmd(cmd, "ffmpeg enhance"):
        print(f"[enhance] ✅ 增强完成: {output_path}")
    else:
        print("[enhance] ⚠️ 增强失败，使用原始文件")
        shutil.copy2(video_path, output_path)

    return output_path


# ---------------------------------------------------------------------------
# 2. 画幅适配
# ---------------------------------------------------------------------------

def reframe_video(
    video_path: str,
    target_ratio: str,
    output_path: str = "reframed.mp4",
) -> str:
    """画幅适配：16:9 → 9:16 / 1:1 等

    优先用 OM auto_reframe，fallback 到 ffmpeg crop/scale + 背景模糊填充。
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")

    # 解析目标比例
    ratio_map = {
        "16:9": (16, 9),
        "9:16": (9, 16),
        "1:1": (1, 1),
        "4:5": (4, 5),
        "21:9": (21, 9),
    }
    if target_ratio not in ratio_map:
        raise ValueError(f"不支持的比例: {target_ratio}，支持: {list(ratio_map.keys())}")

    tw, th = ratio_map[target_ratio]
    info = _probe_video(video_path)
    src_w, src_h = info["width"], info["height"]

    print(f"[reframe] {src_w}x{src_h} → {target_ratio} ({tw}:{th})")

    # 如果源已经是目标比例，直接复制
    if src_w > 0 and src_h > 0:
        src_ratio = src_w / src_h
        target_ratio_f = tw / th
        if abs(src_ratio - target_ratio_f) < 0.02:
            print("[reframe] 源已是目标比例，跳过")
            shutil.copy2(video_path, output_path)
            return output_path

    # --- 尝试 OM AutoReframe ---
    AutoReframeCls = _try_import_om_reframe()
    if AutoReframeCls is not None:
        try:
            print("[reframe] 使用 OM AutoReframe...")
            tool = AutoReframeCls()
            # 映射比例到 preset
            preset_map = {
                "9:16": "portrait",
                "1:1": "square",
                "16:9": "landscape",
                "21:9": "cinematic",
                "4:5": "vertical_4_5",
            }
            result = tool.run(
                input_path=video_path,
                output_path=output_path,
                target_aspect=preset_map.get(target_ratio, "portrait"),
            )
            if result and result.get("status") == "success":
                print(f"[reframe] ✅ OM AutoReframe 完成")
                return output_path
        except Exception as e:
            print(f"[reframe] OM AutoReframe 失败，fallback: {e}")

    # --- FFmpeg fallback: 背景模糊填充 ---
    if not _ffmpeg_available():
        print("[reframe] ⚠️ ffmpeg 不可用，跳过画幅适配")
        shutil.copy2(video_path, output_path)
        return output_path

    # 策略：缩放视频使其适应目标框，背景用模糊填充
    # 输出分辨率基于源高度保持质量
    if src_h > 0:
        out_h = src_h
        out_w = int(out_h * tw / th)
        # 确保偶数
        out_w = out_w + (out_w % 2)
        out_h = out_h + (out_h % 2)
    else:
        out_w, out_h = 1080, 1920 if target_ratio == "9:16" else 1080

    # 复杂滤镜：
    # [0] 分裂为两路
    # 路1: 模糊放大作为背景
    # 路2: 缩放适配作为前景
    # 叠加
    filter_complex = (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},gblur=sigma=20[bg];"
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    print(f"[reframe] ffmpeg 输出: {out_w}x{out_h}")
    if _run_cmd(cmd, "ffmpeg reframe"):
        print(f"[reframe] ✅ 画幅适配完成: {output_path}")
    else:
        print("[reframe] ⚠️ 画幅适配失败，使用原始文件")
        shutil.copy2(video_path, output_path)

    return output_path


# ---------------------------------------------------------------------------
# 3. 片头
# ---------------------------------------------------------------------------

def add_intro(
    video_path: str,
    title: str = "Untitled",
    logo_path: Optional[str] = None,
    duration: float = 2.5,
    output_path: str = "with_intro.mp4",
) -> str:
    """添加片头标题卡：纯色/渐变背景 + 文字叠加 + 淡入

    用 ffmpeg 生成标题卡视频片段，然后与原视频拼接。
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")

    if not title:
        print("[intro] 无标题，跳过片头")
        shutil.copy2(video_path, output_path)
        return output_path

    if not _ffmpeg_available():
        print("[intro] ⚠️ ffmpeg 不可用，跳过片头")
        shutil.copy2(video_path, output_path)
        return output_path

    info = _probe_video(video_path)
    w = info["width"] or 1920
    h = info["height"] or 1080

    print(f"[intro] 生成片头: \"{title}\" ({duration}s, {w}x{h})")

    # 创建临时目录
    tmpdir = tempfile.mkdtemp(prefix="visual_intro_")
    try:
        intro_path = os.path.join(tmpdir, "intro.mp4")

        # 生成标题卡视频：深色背景 + 居中白色文字 + 1s 淡入
        # 使用 drawtext + fade
        fontsize = max(36, h // 15)
        # 转义标题中的特殊字符（ffmpeg drawtext 需要）
        safe_title = title.replace("'", "\u2019").replace(":", "\\:").replace("%", "%%")

        filter_complex = (
            f"color=c=0x1a1a2e:s={w}x{h}:d={duration}:r=30[bg];"
            f"[bg]drawtext=text='{safe_title}'"
            f":fontcolor=white:fontsize={fontsize}"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":font=Sans"
            f",fade=t=in:st=0:d=1"
            f",fade=t=out:st={duration - 0.5}:d=0.5[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x1a1a2e:s={w}x{h}:d={duration}:r=30",
            "-vf", (
                f"drawtext=text='{safe_title}'"
                f":fontcolor=white:fontsize={fontsize}"
                f":x=(w-text_w)/2:y=(h-text_h)/2"
                f":font=Sans,"
                f"fade=t=in:st=0:d=1,"
                f"fade=t=out:st={duration - 0.5}:d=0.5"
            ),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            intro_path,
        ]

        if not _run_cmd(cmd, "生成片头"):
            print("[intro] ⚠️ 片头生成失败，跳过")
            shutil.copy2(video_path, output_path)
            return output_path

        # 拼接：intro + 原视频
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            f.write(f"file '{intro_path}'\n")
            f.write(f"file '{video_path}'\n")

        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path,
        ]
        if _run_cmd(cmd_concat, "拼接片头"):
            print(f"[intro] ✅ 片头添加完成: {output_path}")
        else:
            print("[intro] ⚠️ 拼接失败，使用原始文件")
            shutil.copy2(video_path, output_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return output_path


# ---------------------------------------------------------------------------
# 4. 片尾
# ---------------------------------------------------------------------------

def add_outro(
    video_path: str,
    credits_text: str = "Made with AI\nPowered by HonCut",
    duration: float = 4.0,
    output_path: str = "with_outro.mp4",
) -> str:
    """添加片尾：黑底 + 滚动/静态文字 + 淡出"""
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")

    if not _ffmpeg_available():
        print("[outro] ⚠️ ffmpeg 不可用，跳过片尾")
        shutil.copy2(video_path, output_path)
        return output_path

    info = _probe_video(video_path)
    w = info["width"] or 1920
    h = info["height"] or 1080

    print(f"[outro] 生成片尾 ({duration}s, {w}x{h})")

    tmpdir = tempfile.mkdtemp(prefix="visual_outro_")
    try:
        outro_path = os.path.join(tmpdir, "outro.mp4")

        # 处理多行 credits
        safe_credits = credits_text.replace("'", "\u2019").replace(":", "\\:").replace("%", "%%")
        # 将换行转为 ffmpeg drawtext 的多行
        safe_credits = safe_credits.replace("\\n", "\n")

        fontsize = max(28, h // 22)

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d={duration}:r=30",
            "-vf", (
                f"drawtext=text='{safe_credits}'"
                f":fontcolor=0xCCCCCC:fontsize={fontsize}"
                f":x=(w-text_w)/2:y=(h-text_h)/2"
                f":fontfile=/System/Library/Fonts/PingFang.ttc"
                f":line_spacing=12,"
                f"fade=t=in:st=0:d=0.8,"
                f"fade=t=out:st={duration - 1.5}:d=1.5"
            ),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            outro_path,
        ]

        if not _run_cmd(cmd, "生成片尾"):
            print("[outro] ⚠️ 片尾生成失败，跳过")
            shutil.copy2(video_path, output_path)
            return output_path

        # Normalize both inputs before concatenation.  The concat demuxer with
        # stream-copy preserves incompatible/non-zero timestamps and can make
        # the delivery encode duplicate frames.  A generated silent track also
        # guarantees that the outro participates in an A/V concat.
        source_duration = max(float(info.get("duration") or 0), 0.001)
        source_fps = str(info.get("fps") or "24")
        probe_audio = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        has_source_audio = probe_audio.returncode == 0 and bool(probe_audio.stdout.strip())
        source_audio_index = 0 if has_source_audio else 3
        audio_inputs = [
            "-f", "lavfi", "-t", str(duration),
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
        if not has_source_audio:
            audio_inputs += [
                "-f", "lavfi", "-t", str(source_duration),
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
        filter_graph = (
            f"[0:v]scale={w}:{h},fps={source_fps},format=yuv420p,settb=AVTB,"
            "setpts=PTS-STARTPTS[v0];"
            f"[1:v]scale={w}:{h},fps={source_fps},format=yuv420p,settb=AVTB,"
            "setpts=PTS-STARTPTS[v1];"
            f"[{source_audio_index}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a0];"
            "[2:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            "asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
        )
        cmd_concat = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", outro_path,
            *audio_inputs,
            "-filter_complex", filter_graph,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            output_path,
        ]
        if _run_cmd(cmd_concat, "拼接片尾"):
            print(f"[outro] ✅ 片尾添加完成: {output_path}")
        else:
            print("[outro] ⚠️ 拼接失败，使用原始文件")
            shutil.copy2(video_path, output_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return output_path


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def process_visual(
    video_path: str,
    title: Optional[str] = None,
    logo_path: Optional[str] = None,
    target_ratio: Optional[str] = None,
    enable_enhance: bool = True,
    enable_intro: bool = True,
    enable_outro: bool = True,
    output_path: str = "visual_final.mp4",
) -> str:
    """主入口：视觉后处理

    Args:
        video_path: 输入视频路径（audio_mixed.mp4）
        title: 片头标题（None 则不添加片头）
        logo_path: Logo 图片路径（可选，暂未使用）
        target_ratio: 目标画幅 "16:9"/"9:16"/"1:1"（None 则不改变）
        enable_enhance: 是否启用画质增强
        enable_intro: 是否添加片头
        enable_outro: 是否添加片尾
        output_path: 最终输出路径

    Returns:
        输出文件路径
    """
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")

    print("=" * 60)
    print("🎬 Phase 8: 视觉后处理")
    print("=" * 60)
    print(f"  输入: {video_path}")
    print(f"  输出: {output_path}")
    print(f"  标题: {title or '(无)'}")
    print(f"  画幅: {target_ratio or '(不变)'}")
    print(f"  增强: {'✅' if enable_enhance else '❌'}")
    print(f"  片头: {'✅' if enable_intro and title else '❌'}")
    print(f"  片尾: {'✅' if enable_outro else '❌'}")
    print()

    # 使用临时文件链式处理
    tmpdir = tempfile.mkdtemp(prefix="visual_post_")
    current = video_path

    try:
        # Step 1: 画质增强
        if enable_enhance:
            print("─" * 40)
            enhanced = os.path.join(tmpdir, "enhanced.mp4")
            current = enhance_video(current, enhanced)
            print()

        # Step 2: 画幅适配
        if target_ratio:
            print("─" * 40)
            reframed = os.path.join(tmpdir, "reframed.mp4")
            current = reframe_video(current, target_ratio, reframed)
            print()

        # Step 3: 片头
        if enable_intro and title:
            print("─" * 40)
            with_intro = os.path.join(tmpdir, "with_intro.mp4")
            current = add_intro(current, title, logo_path, output_path=with_intro)
            print()

        # Step 4: 片尾
        if enable_outro:
            print("─" * 40)
            credits = f"{title or 'Project'}\n\nMade with AI\nPowered by HonCut"
            with_outro = os.path.join(tmpdir, "with_outro.mp4")
            current = add_outro(current, credits, output_path=with_outro)
            print()

        # 最终输出
        if current != output_path:
            shutil.copy2(current, output_path)

        print("=" * 60)
        print(f"🎬 视觉后处理完成: {output_path}")
        print("=" * 60)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 视觉后处理：画质增强 + 画幅适配 + 片头片尾",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visual_post.py --input audio_mixed.mp4 --title "雪狼传说" --output visual_final.mp4
  python visual_post.py --input audio_mixed.mp4 --ratio 9:16 --no-intro --output vertical.mp4
  python visual_post.py --input audio_mixed.mp4 --no-enhance --no-outro --output simple.mp4
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="输入视频路径 (audio_mixed.mp4)")
    parser.add_argument("--output", "-o", default="visual_final.mp4", help="输出视频路径 (default: visual_final.mp4)")
    parser.add_argument("--title", "-t", default=None, help="片头标题文字")
    parser.add_argument("--logo", default=None, help="Logo 图片路径（可选）")
    parser.add_argument("--ratio", "-r", default=None, choices=["16:9", "9:16", "1:1", "4:5", "21:9"],
                        help="目标画幅 (default: 不变)")
    parser.add_argument("--credits", default=None, help="片尾文字 (default: 自动生成)")
    parser.add_argument("--intro-duration", type=float, default=2.5, help="片头时长秒 (default: 2.5)")
    parser.add_argument("--outro-duration", type=float, default=4.0, help="片尾时长秒 (default: 4.0)")
    parser.add_argument("--no-enhance", action="store_true", help="跳过画质增强")
    parser.add_argument("--no-intro", action="store_true", help="跳过片头")
    parser.add_argument("--no-outro", action="store_true", help="跳过片尾")
    parser.add_argument("--upscale", action="store_true", help="启用超分（需 OM/GPU）")

    args = parser.parse_args()

    process_visual(
        video_path=args.input,
        title=args.title,
        logo_path=args.logo,
        target_ratio=args.ratio,
        enable_enhance=not args.no_enhance,
        enable_intro=not args.no_intro,
        enable_outro=not args.no_outro,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
