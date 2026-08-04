"""Remotion caption burn tool — **runtime-specific (Remotion-only)**.

Renders animated word-by-word captions onto a talking-head video using
the Remotion CaptionOverlay component. Falls back to FFmpeg subtitle
burning if Remotion is not available.

The tool:
1. Converts word-level transcript segments to Remotion WordCaption JSON
2. Writes a props file for the TalkingHead composition
3. Renders via ``npx remotion render``
4. Returns the captioned video path

Fallback: if Remotion is unavailable, burns subtitles at the bottom of
the frame using FFmpeg's ``subtitles`` filter with bold styling.

## Runtime scope

This tool is **Remotion-specific** and deliberately has no HyperFrames
counterpart in Phase 1. Word-level caption burn parity on the HyperFrames
runtime is explicitly deferred work (see ``skills/core/hyperframes.md`` →
"What stays Remotion-only in Phase 1").

If a brief requires word-level/karaoke captions, lock
``render_runtime="remotion"`` at proposal even if the rest of the
composition would otherwise be a good fit for HyperFrames. Do NOT attempt
to bolt this tool onto a HyperFrames workspace — the TalkingHead
composition ID and ``WordCaption`` prop shape it emits are tied to the
React scene stack in ``remotion-composer/``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolTier,
)


class RemotionCaptionBurn(BaseTool):
    name = "remotion_caption_burn"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "subtitle"
    provider = "remotion"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "cmd:ffprobe"]
    install_instructions = (
        "Remotion (optional, preferred): npm install in remotion-composer/\n"
        "FFmpeg (required for fallback): https://ffmpeg.org/download.html"
    )
    agent_skills = ["remotion-best-practices", "ffmpeg"]

    capabilities = [
        "burn_remotion_captions",
        "burn_ffmpeg_captions_fallback",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path", "output_path"],
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to the input video (enhanced talking-head footage).",
            },
            "output_path": {
                "type": "string",
                "description": "Path for the output video with captions burned in.",
            },
            "segments": {
                "type": "array",
                "description": (
                    "Word-level transcript segments from transcriber tool. "
                    "Each segment has 'words' array with {word, start, end}."
                ),
            },
            "srt_path": {
                "type": "string",
                "description": (
                    "Path to an SRT file. Used as an alternative to segments. "
                    "If both provided, segments take priority."
                ),
            },
            "words_per_page": {
                "type": "integer",
                "default": 4,
                "description": "Words shown at once in the caption overlay.",
            },
            "font_size": {
                "type": "integer",
                "default": 52,
                "description": "Font size for captions.",
            },
            "highlight_color": {
                "type": "string",
                "default": "#22D3EE",
                "description": "Highlight color for the active word (hex).",
            },
            "corrections": {
                "type": "object",
                "description": (
                    "Dictionary of word corrections for common misrecognitions. "
                    "Keys are the wrong word (case-insensitive), values are the "
                    "correct replacement. Example: {\"cloud\": \"Claude\"}."
                ),
            },
            "overlays": {
                "type": "array",
                "description": (
                    "Array of overlay objects to render on top of the video. "
                    "Each overlay has: type (text_card, stat_card, callout, "
                    "comparison, bar_chart, line_chart, pie_chart, kpi_grid, "
                    "hero_title, section_title, stat_reveal), in_seconds, "
                    "out_seconds, position (lower_third, upper_third, "
                    "left_panel, right_panel, full_overlay), and component-"
                    "specific props (text, stat, chartData, etc.). "
                    "See asset_manifest overlays from the asset-director."
                ),
            },
            "force_ffmpeg": {
                "type": "boolean",
                "default": False,
                "description": "Force FFmpeg fallback even if Remotion is available.",
            },
        },
    }

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=500)
    idempotency_key_fields = ["input_path", "segments", "srt_path"]
    side_effects = ["writes captioned video to output_path"]
    user_visible_verification = [
        "Play the output video and verify captions appear at the bottom of the frame",
        "Check that the active word is highlighted in the specified color",
        "Verify face is not occluded by caption text",
    ]

    # ------------------------------------------------------------------ #
    #  Chrome path resolution
    # ------------------------------------------------------------------ #

    def _resolve_chrome_path(self) -> str:
        """Resolve Chrome executable path from env or system defaults.
        
        Priority:
        1. CHROME_PATH env var (set in .env)
        2. System Chrome on macOS
        3. System Chrome on Linux
        4. Let Remotion auto-detect (fallback)
        """
        import os
        
        # 1. Check CHROME_PATH env var
        chrome_path = os.environ.get("CHROME_PATH")
        if chrome_path and Path(chrome_path).exists():
            return chrome_path
        
        # 2. macOS system Chrome
        mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if Path(mac_chrome).exists():
            return mac_chrome
        
        # 3. Linux system Chrome
        linux_chrome = "/usr/bin/google-chrome"
        if Path(linux_chrome).exists():
            return linux_chrome
        
        # 4. Fallback: let Remotion auto-detect (returns empty string)
        return ""

    # ------------------------------------------------------------------ #
    #  Remotion detection
    # ------------------------------------------------------------------ #

    def _find_remotion_root(self) -> Path | None:
        """Find the remotion-composer directory relative to the repo."""
        candidates = [
            Path.cwd() / "remotion-composer",
            Path(__file__).resolve().parent.parent.parent / "remotion-composer",
        ]
        for p in candidates:
            if (
                p.is_dir()
                and (p / "package.json").exists()
                and (p / "node_modules").is_dir()
            ):
                return p
        return None

    def _remotion_available(self) -> bool:
        return (
            shutil.which("npx") is not None
            and self._find_remotion_root() is not None
        )

    # ------------------------------------------------------------------ #
    #  Word caption conversion
    # ------------------------------------------------------------------ #

    def _segments_to_word_captions(
        self, segments: list[dict], corrections: dict[str, str] | None = None
    ) -> list[dict]:
        """Convert transcriber segments to [{word, startMs, endMs}, ...]."""
        captions: list[dict] = []
        corr = {k.lower(): v for k, v in (corrections or {}).items()}

        for segment_index, seg in enumerate(segments):
            words = seg.get("words", [])
            if words:
                for w in words:
                    raw = w["word"].strip()
                    fixed = corr.get(raw.lower().strip(".,!?;:"), raw)
                    # Preserve trailing punctuation from original
                    trailing = ""
                    if raw and raw[-1] in ".,!?;:":
                        trailing = raw[-1]
                    if fixed != raw and not fixed.endswith(trailing):
                        fixed = fixed + trailing
                    captions.append({
                        "word": fixed,
                        "startMs": int(w["start"] * 1000),
                        "endMs": int(w["end"] * 1000),
                        "cueId": segment_index,
                    })
            elif "text" in seg:
                text_words = seg["text"].strip().split()
                dur = seg["end"] - seg["start"]
                per_word = dur / max(len(text_words), 1)
                for i, tw in enumerate(text_words):
                    fixed = corr.get(tw.lower().strip(".,!?;:"), tw)
                    captions.append({
                        "word": fixed,
                        "startMs": int((seg["start"] + i * per_word) * 1000),
                        "endMs": int((seg["start"] + (i + 1) * per_word) * 1000),
                        "cueId": segment_index,
                    })
        return captions

    def _srt_to_word_captions(
        self, srt_path: str, corrections: dict[str, str] | None = None
    ) -> list[dict]:
        """Parse SRT file into word captions."""
        content = Path(srt_path).read_text(encoding="utf-8")
        blocks = re.split(r"\n\n+", content.strip())
        corr = {k.lower(): v for k, v in (corrections or {}).items()}
        captions: list[dict] = []

        for cue_index, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            m = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                lines[1],
            )
            if not m:
                continue
            start_ms = (
                int(m.group(1)) * 3600000
                + int(m.group(2)) * 60000
                + int(m.group(3)) * 1000
                + int(m.group(4))
            )
            end_ms = (
                int(m.group(5)) * 3600000
                + int(m.group(6)) * 60000
                + int(m.group(7)) * 1000
                + int(m.group(8))
            )
            text = " ".join(lines[2:]).strip()
            words = text.split()
            per_word = (end_ms - start_ms) / max(len(words), 1)
            for i, w in enumerate(words):
                fixed = corr.get(w.lower().strip(".,!?;:"), w)
                captions.append({
                    "word": fixed,
                    "startMs": int(start_ms + i * per_word),
                    "endMs": int(start_ms + (i + 1) * per_word),
                    "cueId": cue_index,
                })
        return captions

    # ------------------------------------------------------------------ #
    #  Remotion render
    # ------------------------------------------------------------------ #

    def _render_remotion(
        self,
        input_path: str,
        output_path: str,
        captions: list[dict],
        words_per_page: int,
        font_size: int,
        highlight_color: str,
        overlays: list[dict] | None = None,
    ) -> ToolResult:
        root = self._find_remotion_root()
        if root is None:
            return ToolResult(success=False, error="Remotion root not found")

        # Get video duration in frames
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            input_path,
        ]
        dur_result = self.run_command(dur_cmd)
        dur_out = dur_result.stdout
        duration_s = float(dur_out.strip().split("\n")[0])
        total_frames = math.ceil(duration_s * 30)

        # Detect video dimensions
        dim_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            input_path,
        ]
        dim_result = self.run_command(dim_cmd)
        dim_parts = dim_result.stdout.strip().split("x")
        width = int(dim_parts[0])
        height = int(dim_parts[1])

        # Copy video to Remotion public folder
        pub_dir = root / "public" / "talking-head"
        pub_dir.mkdir(parents=True, exist_ok=True)
        video_filename = Path(input_path).name
        dest_video = pub_dir / video_filename
        shutil.copy2(input_path, dest_video)

        # Build props JSON
        props = {
            "videoSrc": f"public/talking-head/{video_filename}",
            "captions": captions,
            "overlays": overlays or [],
            "wordsPerPage": words_per_page,
            "fontSize": font_size,
            "highlightColor": highlight_color,
        }
        props_dir = root / "public" / "demo-props"
        props_dir.mkdir(parents=True, exist_ok=True)
        props_file = props_dir / f"caption-burn-{Path(input_path).stem}.json"
        props_file.write_text(json.dumps(props, indent=2), encoding="utf-8")

        # Render (use npx.cmd on Windows for subprocess compatibility)
        import sys
        npx_bin = "npx.cmd" if sys.platform == "win32" else "npx"
        render_cmd = [
            npx_bin, "remotion", "render",
            "TalkingHead",
            f"--props={props_file.relative_to(root)}",
            f"--width={width}", f"--height={height}", "--fps=30",
            f"--frames=0-{total_frames - 1}",
            "--codec=h264", "--crf=18",
            f"--output={str(Path(output_path).resolve())}",
            f"--browser-executable={self._resolve_chrome_path()}",
        ]
        self.run_command(render_cmd, cwd=str(root))

        if not Path(output_path).exists():
            return ToolResult(success=False, error="Remotion render produced no output")

        return ToolResult(
            success=True,
            data={
                "method": "remotion",
                "output": output_path,
                "duration_seconds": round(duration_s, 2),
                "total_frames": total_frames,
                "caption_count": len(captions),
                "overlay_count": len(overlays or []),
                "words_per_page": words_per_page,
            },
            artifacts=[output_path],
        )

    # ------------------------------------------------------------------ #
    #  FFmpeg fallback
    # ------------------------------------------------------------------ #

    def _render_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        captions: list[dict],
    ) -> ToolResult:
        """Fall back to FFmpeg subtitle burning at bottom of frame."""
        # Generate temporary SRT from word captions
        tmp_srt = Path(output_path).parent / f"_tmp_captions_{int(time.time())}.srt"
        tmp_srt.parent.mkdir(parents=True, exist_ok=True)

        srt_lines = []
        idx = 1
        # Group into pages of ~4 words
        page_size = 4
        for i in range(0, len(captions), page_size):
            page = captions[i : i + page_size]
            text = " ".join(c["word"] for c in page)
            start = page[0]["startMs"]
            end = page[-1]["endMs"]
            srt_lines.append(str(idx))
            srt_lines.append(
                f"{self._ms_to_srt(start)} --> {self._ms_to_srt(end)}"
            )
            srt_lines.append(text)
            srt_lines.append("")
            idx += 1

        tmp_srt.write_text("\n".join(srt_lines), encoding="utf-8")

        try:
            filters = self._ffmpeg_filters()
            if "subtitles" in filters:
                self._render_with_subtitles(input_path, output_path, tmp_srt)
                method = "ffmpeg_subtitles"
            elif "drawtext" in filters:
                self._render_with_drawtext(input_path, output_path, captions)
                method = "ffmpeg_drawtext"
            elif "overlay" in filters:
                # Minimal FFmpeg bottles can omit both libass and libfreetype.
                # Render glyphs with Pillow and let FFmpeg perform only the
                # timed compositing, which is available in those builds.
                self._render_with_image_overlays(input_path, output_path, captions)
                method = "ffmpeg_image_overlay"
            else:
                return ToolResult(
                    success=False,
                    error="FFmpeg has none of the subtitles, drawtext, or overlay filters",
                )
        finally:
            try:
                tmp_srt.unlink()
            except OSError:
                pass

        if not Path(output_path).exists():
            return ToolResult(success=False, error="FFmpeg subtitle burn produced no output")

        return ToolResult(
            success=True,
            data={
                "method": method,
                "output": output_path,
                "caption_count": len(captions),
                "note": "Used FFmpeg fallback. Install Remotion for animated captions.",
            },
            artifacts=[output_path],
        )

    @staticmethod
    def _ffmpeg_filters() -> set[str]:
        """Return filter names supported by the active FFmpeg binary."""
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        names = set()
        for line in result.stdout.splitlines():
            columns = line.split()
            if (
                len(columns) >= 2
                and 2 <= len(columns[0]) <= 3
                and columns[1] != "="
                and all(char in ".|" or "A" <= char <= "Z" for char in columns[0])
            ):
                names.add(columns[1])
        return names

    def _render_with_subtitles(
        self, input_path: str, output_path: str, srt_path: Path,
    ) -> None:
        # Quotes here belong to FFmpeg's filtergraph grammar. The command is
        # passed as argv (without a shell), so shell quoting must not be added.
        escaped = str(srt_path).replace("\\", "/")
        escaped = escaped.replace("'", r"\'").replace(":", r"\:")
        video_filter = (
            f"subtitles=filename='{escaped}':"
            "force_style='FontName=serif,FontSize=52,"
            "PrimaryColour=&H004242c9,Outline=0,"
            "Shadow=1,Alignment=2,MarginV=80'"
        )
        self.run_command(self._ffmpeg_video_command(
            input_path, output_path, ["-vf", video_filter]
        ))

    def _render_with_drawtext(
        self, input_path: str, output_path: str, captions: list[dict],
    ) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="caption_drawtext_"))
        try:
            filters = []
            for index, (text, start_s, end_s) in enumerate(self._caption_pages(captions)):
                text_path = temp_dir / f"cue_{index:04d}.txt"
                text_path.write_text(text, encoding="utf-8")
                escaped_path = str(text_path).replace("\\", "/")
                escaped_path = escaped_path.replace("'", r"\'").replace(":", r"\:")
                filters.append(
                    "drawtext="
                    f"textfile='{escaped_path}':fontsize=52:fontcolor=#c94242:"
                    "borderw=2:bordercolor=black:"
                    "x=(w-text_w)/2:y=h-text_h-80:"
                    f"enable='between(t,{start_s:.3f},{end_s:.3f})'"
                )
            self.run_command(self._ffmpeg_video_command(
                input_path, output_path, ["-vf", ",".join(filters)]
            ))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _render_with_image_overlays(
        self, input_path: str, output_path: str, captions: list[dict],
    ) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required when FFmpeg lacks subtitles and drawtext"
            ) from exc

        temp_dir = Path(tempfile.mkdtemp(prefix="caption_overlay_"))
        try:
            font = self._caption_font(ImageFont, 52)
            pages = self._caption_pages(captions)
            png_paths = []
            for index, (caption_text, _, _) in enumerate(pages):
                scratch = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                draw = ImageDraw.Draw(scratch)
                box = draw.textbbox((0, 0), caption_text, font=font, stroke_width=2)
                width = max(2, box[2] - box[0] + 24)
                height = max(2, box[3] - box[1] + 20)
                image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                draw.text(
                    (12 - box[0], 10 - box[1]), caption_text, font=font,
                    fill=(201, 66, 66, 255), stroke_width=2,
                    stroke_fill=(0, 0, 0, 255),
                )
                png_path = temp_dir / f"cue_{index:04d}.png"
                image.save(png_path)
                png_paths.append(png_path)

            cmd = ["ffmpeg", "-y", "-i", input_path]
            for png_path in png_paths:
                cmd.extend(["-i", str(png_path)])

            chains = []
            previous = "[0:v]"
            for index, (_, start_s, end_s) in enumerate(pages, start=1):
                output = f"[captioned{index}]"
                chains.append(
                    f"{previous}[{index}:v]overlay="
                    f"x=(W-w)/2:y=H-h-80:enable='between(t,{start_s:.3f},{end_s:.3f})'"
                    f"{output}"
                )
                previous = output

            cmd.extend([
                "-filter_complex", ";".join(chains),
                "-map", previous, "-map", "0:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "copy", output_path,
            ])
            self.run_command(cmd)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _caption_font(image_font: Any, size: int) -> Any:
        candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return image_font.truetype(candidate, size=size)
        return image_font.load_default()

    @staticmethod
    def _caption_pages(captions: list[dict]) -> list[tuple[str, float, float]]:
        pages = []
        cue_groups: list[list[dict]] = []
        for caption in captions:
            if not cue_groups or (
                caption.get("cueId") != cue_groups[-1][-1].get("cueId")
            ):
                cue_groups.append([])
            cue_groups[-1].append(caption)
        for cue in cue_groups:
            for index in range(0, len(cue), 4):
                page = cue[index:index + 4]
                pages.append((
                    " ".join(item["word"] for item in page),
                    page[0]["startMs"] / 1000.0,
                    page[-1]["endMs"] / 1000.0,
                ))
        return pages

    @staticmethod
    def _ffmpeg_video_command(
        input_path: str, output_path: str, filter_args: list[str],
    ) -> list[str]:
        return [
            "ffmpeg", "-y", "-i", input_path, *filter_args,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", output_path,
        ]

    @staticmethod
    def _ms_to_srt(ms: int) -> str:
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        rem = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{rem:03d}"

    # ------------------------------------------------------------------ #
    #  Main execute
    # ------------------------------------------------------------------ #

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = inputs["input_path"]
        output_path = inputs["output_path"]
        corrections = inputs.get("corrections")
        force_ffmpeg = inputs.get("force_ffmpeg", False)
        words_per_page = inputs.get("words_per_page", 4)
        font_size = inputs.get("font_size", 52)
        highlight_color = inputs.get("highlight_color", "#22D3EE")

        if not Path(input_path).exists():
            return ToolResult(success=False, error=f"Input video not found: {input_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        # Build word captions from segments or SRT
        segments = inputs.get("segments")
        srt_path = inputs.get("srt_path")

        if segments:
            captions = self._segments_to_word_captions(segments, corrections)
        elif srt_path:
            captions = self._srt_to_word_captions(srt_path, corrections)
        else:
            return ToolResult(
                success=False,
                error="Provide either 'segments' (from transcriber) or 'srt_path'.",
            )

        if not captions:
            return ToolResult(success=False, error="No caption words extracted.")

        overlays = inputs.get("overlays")

        # Choose render method
        if not force_ffmpeg and self._remotion_available():
            result = self._render_remotion(
                input_path, output_path, captions,
                words_per_page, font_size, highlight_color,
                overlays=overlays,
            )
        else:
            result = self._render_ffmpeg(input_path, output_path, captions)

        result.duration_seconds = round(time.time() - start, 2)
        return result
