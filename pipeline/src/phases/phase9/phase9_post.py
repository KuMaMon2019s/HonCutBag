"""Phase 9 post-production entry point."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

from phases.phase9.captions import (
    _caption_segments_from_final_asr,
    _merge_shot_transcripts,
    _probe_shot_duration,
    _write_srt,
)
from phases.phase9.delivery_encoding import (
    _final_encode_duration_gate,
    _final_encode_filters,
    _validated_reviewed_delivery_contract,
)
from phases.phase9.score_and_mix import (
    _detect_bgm,
    _phase9_real_audio_mix_request,
    _phase9_real_audio_tracks,
    _prepare_continuous_bgm,
)
from quality.quality_gate import run_quality_check
from runtime.phase_timing import _banner, _elapsed, _now
from utils.media_probe import _assert_duration_conserved, _probe_av_durations
from utils.media_profiles import _get_profile_dict
from utils.timing_estimator import estimate_phase_duration


def run_phase9(output_dir: Path, dry_run: bool, color_grade: Optional[str] = None,
               upscale: Optional[int] = None, media_profile: str = "1080p",
               target_duration: Optional[float] = None) -> dict:
    """Phase 9: audio_pipeline + visual_post + [color_grade] + [upscale] + rhythm_editor → polished.mp4

    Audio processing (enhanced with OM AudioMixer capabilities):
    - Loudness normalization (loudnorm filter, target -14 LUFS)
    - Background music ducking (sidechaincompress when BGM detected)
    - Fade in/out (1s fade-in, 2s fade-out)
    - Falls back to basic FFmpeg processing if enhanced pipeline fails
    - Uses lib.media_profiles (OM) for final output encoding parameters

    Args:
        color_grade: Optional color profile name (cinematic_warm, cinematic_cool, moody_dark,
                     bright_clean, vintage_film, high_contrast, neutral)
        upscale: Optional target height in pixels (e.g. 720 for 720p output)
        media_profile: encoding profile name (default: "1080p")
    """
    _banner(9, 9, "后期处理 (Post-Production)", dry_run)
    start = _now()
    _p8_est = estimate_phase_duration("phase9")
    print(f"  ⏱ Phase 9 开始 (预估 ~{int(_p8_est)}s)")
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过后期处理")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    raw_video = output_dir / "raw_assembly.mp4"
    if not raw_video.exists():
        return {"status": "skipped", "reason": "no raw_assembly.mp4", "duration_s": _elapsed(start)}

    outputs = []
    storyboard_path = output_dir / "STORYBOARD.json"
    sb_path_str = str(storyboard_path) if storyboard_path.exists() else None
    storyboard_data = None

    # --- P0-D3: Check whether the audio track is genuinely audible ──────────
    # Previously this only checked "has audio stream" — but local Wan2.2 videos
    # have an anullsrc-injected silent track from edit_decisions normalisation.
    # Now we run volumedetect: mean_volume < -60 dB → treat as silent → run
    # ambient fallback so the final video is never silent.
    has_real_audio = False
    try:
        from tools.audio_pipeline import is_silent_audio
        import subprocess as _sp
        # First check: does an audio stream exist at all?
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                     "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                     str(raw_video)]
        probe_result = _sp.run(probe_cmd, capture_output=True, text=True, timeout=10)
        has_stream = bool(probe_result.stdout.strip())
        if has_stream:
            # Second check: is it actually audible?
            has_real_audio = not is_silent_audio(str(raw_video))
    except Exception:
        pass

    if has_real_audio:
        print("  → [P0-D3] 视频已有真实音轨（Seedance generate_audio），跳过环境音合成")

    # Step 9.1: transcribe every source shot before subtitle rendering. Keep
    # this outside Phase 9's broad post-processing guard so extraction/API
    # failures propagate instead of being reported as an ordinary soft failure.
    transcript_data = None
    if sb_path_str:
        from clients.asr_client import transcribe_audio
        from tools.audio_pipeline import extract_audio_track, is_silent_audio

        storyboard_data = json.loads(storyboard_path.read_text(encoding="utf-8"))
        sb_shots = storyboard_data.get("shots", [])
        shots_dir = output_dir / "shots"
        asr_receipts_dir = output_dir / "asr_transcripts"
        asr_receipts_dir.mkdir(parents=True, exist_ok=True)
        durations_ms = []
        shot_transcripts = []
        print("  → asr_transcription: 逐镜提取音轨并转写...")
        for index, _shot in enumerate(sb_shots, 1):
            shot_dir = shots_dir / f"S{index:02d}"
            shot_video = shot_dir / "output.mp4"
            wav_path = shot_dir / "audio.wav"
            if not shot_video.is_file():
                print(f"    ⚠ S{index:02d}: output.mp4 缺失，跳过 ASR（该镜未进入成片）")
                durations_ms.append(0)
                shot_transcripts.append({"text": "", "segments": [], "skipped": True})
                continue
            durations_ms.append(round(_probe_shot_duration(shots_dir, index) * 1000))
            shot_id = _shot.get("shot_id") or _shot.get("id") or f"S{index:02d}"
            if is_silent_audio(str(shot_video)):
                print(f"    ⊘ S{index:02d}: 无可听音轨，跳过 ASR")
                transcription = {
                    "text": "",
                    "segments": [],
                    "skipped": True,
                    "reason": "no_audible_audio",
                }
                shot_transcripts.append(transcription)
                receipt = {
                    "shot_id": str(shot_id),
                    "audio_path": None,
                    "duration_ms": durations_ms[-1],
                    "transcription": transcription,
                }
                (asr_receipts_dir / f"S{index:02d}.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                continue
            extract_audio_track(str(shot_video), str(wav_path))
            transcription = transcribe_audio(str(wav_path))
            shot_transcripts.append(transcription)
            receipt = {
                "shot_id": str(shot_id),
                "audio_path": str(wav_path),
                "duration_ms": durations_ms[-1],
                "transcription": transcription,
            }
            (asr_receipts_dir / f"S{index:02d}.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        timeline_path = output_dir / "edit_timeline.json"
        edit_timeline = None
        if timeline_path.is_file():
            edit_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        transcript_data = _merge_shot_transcripts(
            sb_shots,
            durations_ms,
            shot_transcripts,
            edit_timeline=edit_timeline,
        )
        transcript_data["asr_summary"] = {
            "shots_considered": len(shot_transcripts),
            "shots_submitted": sum(not item.get("skipped") for item in shot_transcripts),
            "shots_skipped_no_audio": sum(
                item.get("reason") == "no_audible_audio" for item in shot_transcripts
            ),
            "shots_with_text": sum(bool(item.get("text") or item.get("segments"))
                                   for item in shot_transcripts),
            "raw_word_segments": sum(len(item.get("segments") or [])
                                     for item in shot_transcripts),
            "caption_segments": len(transcript_data["caption_segments"]),
            "receipts_dir": "asr_transcripts",
        }
        transcript_path = output_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs.append("transcript.json")
        outputs.append("asr_transcripts/")
        summary = transcript_data["asr_summary"]
        print(
            "    ✓ ASR 完成: "
            f"{summary['shots_with_text']}/{summary['shots_considered']} 镜有语音, "
            f"{summary['raw_word_segments']} 个原始词段; "
            f"生成 {summary['caption_segments']} 条字幕"
        )

    try:
        from phases.phase9.visual_post import process_visual
        from phases.phase9.rhythm_editor import edit_rhythm

        # Track step statuses for quality gate integrity
        step_status = {}
        if transcript_data is not None:
            step_status["asr_transcription"] = "done"

        # Step 9.1: Audio processing via OM AudioMixer
        bgm_path = _detect_bgm(output_dir, storyboard_path)
        if (
            not bgm_path
            and storyboard_data
            and storyboard_data.get("audio", {}).get("enabled", False)
        ):
            try:
                from phases.phase9.audio_mixer import AudioMixer as Phase9MaterialMixer

                mood = (
                    storyboard_data.get("audio", {}).get("mood")
                    or storyboard_data.get("metadata", {}).get("mood")
                )
                selected_bgm = Phase9MaterialMixer().select_bgm(mood, target_duration)
                bgm_path = selected_bgm.path if selected_bgm else None
            except Exception as exc:
                print(f"    ⚠ 全局配乐选择不可用: {exc}")
        if bgm_path:
            try:
                bgm_path = _prepare_continuous_bgm(
                    bgm_path,
                    float(
                        target_duration
                        or _probe_av_durations(raw_video)["video"]
                        or 0.0
                    ),
                    output_dir / "audio_layer" / "continuous_bgm.m4a",
                )
                outputs.append("audio_layer/continuous_bgm.m4a")
                print("    ✓ 全局配乐已跨全片延展，并对循环点做等功率交叉淡化")
            except Exception as exc:
                print(f"    ⚠ 全局配乐延展失败，使用原始曲目: {exc}")
        if has_real_audio:
            audio_out = output_dir / "audio_processed.mp4"
            from vendor.video_tools.tools.audio.audio_mixer import AudioMixer

            base_audio = output_dir / "audio_layer" / "source_audio.m4a"
            base_audio.parent.mkdir(parents=True, exist_ok=True)
            extract_base = [
                "ffmpeg", "-y", "-i", str(raw_video), "-vn",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                str(base_audio),
            ]
            import subprocess as _sp
            extracted = _sp.run(extract_base, capture_output=True, text=True)
            base_track = base_audio if extracted.returncode == 0 and base_audio.is_file() else raw_video
            tracks, skipped_tts = _phase9_real_audio_tracks(
                output_dir, storyboard_data, transcript_data, base_track, bgm_path
            )
            overlay_count = sum(track.get("role") == "speech" for track in tracks)
            mixer = AudioMixer()
            mix_result = mixer.execute(_phase9_real_audio_mix_request(tracks, audio_out))
            audio_success = bool(mix_result.success)
            if audio_success:
                remux_tmp = output_dir / "audio_remux_tmp.mp4"
                remux_cmd = [
                    "ffmpeg", "-y", "-i", str(raw_video), "-i", str(audio_out),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-af", "apad", "-shortest", str(remux_tmp),
                ]
                import subprocess as _sp
                _sp.run(remux_cmd, capture_output=True, check=True)
                import shutil
                shutil.move(str(remux_tmp), str(audio_out))
                outputs.append("audio_processed.mp4")
                step_status["audio_pipeline"] = "done"
                if overlay_count:
                    print(
                        f"  ✓ [P0-D3] base track preserved + {overlay_count} TTS overlays "
                        f"({skipped_tts} skipped: dialogue already in source audio)"
                    )
                else:
                    print("  ✓ [P0-D3] base track only, loudnorm applied")
            else:
                print(f"  ⚠ [P0-D3] real-audio processing failed: {mix_result.error}")
                step_status["audio_pipeline"] = "failed"
                import shutil
                shutil.copy2(str(raw_video), audio_out)
        else:
            print("  → audio_pipeline: 音频处理 (AudioMixer: loudnorm + ducking)...")
            audio_out = output_dir / "audio_processed.mp4"

            # Detect background music for ducking
            if bgm_path:
                print(f"    ✓ BGM detected: {Path(bgm_path).name}")
            else:
                print(f"    ⊘ No BGM detected (skipping ducking)")

            audio_success = False
            try:
                from vendor.video_tools.tools.audio.audio_mixer import AudioMixer
                mixer = AudioMixer()

                # Prepare tracks
                tracks = [{"path": str(raw_video), "role": "speech"}]
                if bgm_path:
                    tracks.append({"path": bgm_path, "role": "music", "volume": 0.2})

                mix_result = mixer.execute({
                    "operation": "full_mix" if bgm_path else "mix",
                    "tracks": tracks,
                    "ducking": {"enabled": True, "music_volume_during_speech": 0.15} if bgm_path else None,
                    "normalize": True,
                    "loudnorm_target": -14,  # YouTube/TikTok standard
                    "output_path": str(audio_out),
                })

                if mix_result.success:
                    outputs.append("audio_processed.mp4")
                    audio_success = True
                    print(f"  ✓ Audio processing complete")
                    step_status["audio_pipeline"] = "done"

                    # AudioMixer outputs audio-only (-vn); remux processed audio
                    # back into the original video stream so downstream steps
                    # (visual_post, rhythm_editor, final_encode) still have video.
                    remux_tmp = output_dir / "audio_remux_tmp.mp4"
                    remux_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(raw_video),       # original video with video stream
                        "-i", str(audio_out),        # processed audio-only file
                        "-map", "0:v",              # take video from original
                        "-map", "1:a",              # take audio from processed
                        "-c:v", "copy",             # don't re-encode video
                        "-c:a", "aac",
                        "-af", "apad", "-shortest",
                        str(remux_tmp),
                    ]
                    import subprocess as _sp
                    try:
                        _sp.run(remux_cmd, capture_output=True, check=True)
                        import shutil
                        shutil.move(str(remux_tmp), str(audio_out))
                        print(f"  ✓ Audio remuxed into video stream")
                    except Exception as remux_err:
                        print(f"  ⚠ Audio remux failed: {remux_err}, using original video")
                        import shutil
                        if remux_tmp.exists():
                            remux_tmp.unlink()
                        shutil.copy2(str(raw_video), str(audio_out))
                else:
                    print(f"  ⚠ AudioMixer failed: {mix_result.error}")
                    step_status["audio_pipeline"] = "failed"
                    # Fallback: just copy video
                    import shutil
                    shutil.copy2(str(raw_video), audio_out)
            except ImportError as e:
                print(f"  ⚠ AudioMixer unavailable: {e}")
                step_status["audio_pipeline"] = "failed"
                # Fallback: just copy video
                import shutil
                shutil.copy2(str(raw_video), audio_out)

            # ── Ambient fallback: if AudioMixer path produced a silent track ──
            # AudioMixer may not be available or may fail, leaving audio_out as
            # a copy of the silent raw_video.  Detect and inject generated ambience.
            try:
                from tools.audio_pipeline import is_silent_audio, generate_ambient_audio
                if audio_out.exists() and is_silent_audio(str(audio_out)):
                    print("  → [ambient-fallback] AudioMixer output still silent, generating ambient audio...")
                    from phases.phase8.edit_decisions import probe_video
                    vid_info = probe_video(str(raw_video))
                    ambient_dur = vid_info.get("duration", 12.0)
                    # Pick scene hint from storyboard if available
                    scene_hint = "generic"
                    if storyboard_data:
                        scene_desc = str(storyboard_data.get("metadata", {}).get("scene", "")).lower()
                        if "forest" in scene_desc or "林" in scene_desc:
                            scene_hint = "forest"
                        elif "city" in scene_desc or "城" in scene_desc:
                            scene_hint = "city"
                    ambient_tmp = output_dir / ".ambient_fallback.m4a"
                    if generate_ambient_audio(ambient_dur, str(ambient_tmp), scene_hint=scene_hint, target_db=-10.0):
                        # Mix ambient audio into the video
                        ambient_out = output_dir / ".ambient_remux.mp4"
                        import subprocess as _sp
                        mix_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(audio_out),
                            "-i", str(ambient_tmp),
                            "-filter_complex",
                            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                            "-map", "0:v", "-map", "[aout]",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-shortest",
                            str(ambient_out),
                        ]
                        try:
                            _sp.run(mix_cmd, capture_output=True, check=True, timeout=60)
                            import shutil
                            shutil.move(str(ambient_out), str(audio_out))
                            print(f"  ✓ [ambient-fallback] Ambient audio mixed in ({scene_hint}, {ambient_dur:.1f}s)")
                            step_status["audio_pipeline"] = "done"
                            if "audio_processed.mp4" not in outputs:
                                outputs.append("audio_processed.mp4")
                        except Exception as mix_err:
                            print(f"  ⚠ [ambient-fallback] Mix failed: {mix_err}")
                        finally:
                            if ambient_tmp.exists():
                                ambient_tmp.unlink()
                            if ambient_out.exists():
                                ambient_out.unlink()
                    else:
                        print("  ⚠ [ambient-fallback] Ambient generation failed")
            except ImportError:
                pass  # audio_pipeline not available, skip fallback

        audio_out = str(audio_out)

        # Subtitle timing must reflect the audible final mix, including TTS
        # overlays and generated speech that was absent from storyboard fields.
        if sb_path_str:
            final_mix_wav = output_dir / "asr_transcripts" / "final_mix.wav"
            final_mix_wav.parent.mkdir(parents=True, exist_ok=True)
            extract_audio_track(str(audio_out), str(final_mix_wav))
            final_mix_transcription = transcribe_audio(str(final_mix_wav))
            final_mix_receipt = {
                "audio_path": str(final_mix_wav),
                "transcription": final_mix_transcription,
            }
            (final_mix_wav.parent / "final_mix.json").write_text(
                json.dumps(final_mix_receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            final_mix_captions = _caption_segments_from_final_asr(final_mix_transcription)
            transcript_data["shot_caption_segments"] = transcript_data["caption_segments"]
            transcript_data["caption_segments"] = final_mix_captions
            transcript_data["final_mix_transcription"] = final_mix_transcription
            transcript_data["asr_summary"]["final_mix_caption_segments"] = len(
                final_mix_captions
            )
            (output_dir / "transcript.json").write_text(
                json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"  ✓ final_mix_asr: 最终混音识别出 {len(final_mix_captions)} 条对白字幕"
            )

        # Step 9.2: visual_post
        print("  → visual_post: 视觉后期...")
        visual_out = str(output_dir / "visual_processed.mp4")
        process_visual(
            video_path=audio_out,
            output_path=visual_out,
            enable_outro=False,
        )
        outputs.append("visual_processed.mp4")

        current_video = visual_out

        # Step 9.2.1: Subtitle burn (optional, from OM RemotionCaptionBurn)
        if sb_path_str:
            print("  → subtitle_burn: 字幕烧录 (RemotionCaptionBurn)...")
            subtitled_out = str(output_dir / "subtitled.mp4")
            try:
                from vendor.video_tools.tools.video.remotion_caption_burn import RemotionCaptionBurn
                caption_burner = RemotionCaptionBurn()

                segments = transcript_data["caption_segments"] if transcript_data else []

                # Also generate SRT as fallback
                srt_path = str(output_dir / "subtitles.srt")
                _write_srt(segments, srt_path)

                if segments:
                    burn_result = caption_burner.execute({
                        "input_path": str(current_video),
                        "output_path": str(subtitled_out),
                        "segments": segments,
                        "srt_path": srt_path,
                        "font_size": 48,
                        "font_color": "#FFFFFF",
                        "outline_color": "#000000",
                        "outline_width": 3,
                        "margin_bottom": 60,
                        "fade_in_ms": 180,
                        "fade_out_ms": 220,
                        "force_ffmpeg": True,
                    })

                    if burn_result.success:
                        current_video = subtitled_out
                        outputs.append("subtitled.mp4")
                        outputs.append("subtitles.srt")
                        print(f"    ✓ 字幕烧录完成: {len(segments)} 条字幕")
                        step_status["subtitle_burn"] = "done"
                    else:
                        print(f"    ⚠ 字幕烧录失败: {burn_result.error}")
                        step_status["subtitle_burn"] = "failed"
                else:
                    print(f"    ⊘ No subtitle data available, skipping subtitle burn")
                    step_status["subtitle_burn"] = "not_required"
            except ImportError as e:
                print(f"    ⚠ RemotionCaptionBurn unavailable: {e}")
                step_status["subtitle_burn"] = "failed"
            except Exception as e:
                print(f"    ⚠ 字幕烧录异常: {e}")
                step_status["subtitle_burn"] = "failed"

        # Step 9.2.5: Color grade (optional, from OM ColorGrade)
        if color_grade:
            print(f"  → color_grade: 应用调色 ({color_grade})...")
            graded_out = str(output_dir / "color_graded.mp4")
            try:
                from vendor.video_tools.tools.enhancement.color_grade import ColorGrade
                grader = ColorGrade()
                grade_result = grader.execute({
                    "input_path": str(current_video),
                    "output_path": str(graded_out),
                    "profile": color_grade,
                    "intensity": 1.0,
                })
                if grade_result.success:
                    current_video = graded_out
                    outputs.append("color_graded.mp4")
                    print(f"    ✓ 调色完成: {color_grade}")
                else:
                    print(f"    ⚠ 调色失败: {grade_result.error}")
            except ImportError as e:
                print(f"    ⚠ ColorGrade unavailable: {e}")

        # Step 9.2.6: Upscale (optional, from OM Upscale — lanczos)
        if upscale:
            print(f"  → upscale: 超分到 {upscale}p (lanczos)...")
            upscaled_out = str(output_dir / "upscaled.mp4")
            try:
                from vendor.video_tools.tools.enhancement.upscale import Upscale
                upscaler = Upscale()
                upscale_result = upscaler.execute({
                    "input_path": str(current_video),
                    "output_path": str(upscaled_out),
                    "target_height": upscale,
                })
                if upscale_result.success:
                    current_video = upscaled_out
                    outputs.append("upscaled.mp4")
                    print(f"    ✓ 超分完成: {upscale}p")
                else:
                    print(f"    ⚠ 超分失败: {upscale_result.error}")
            except ImportError as e:
                print(f"    ⚠ Upscale unavailable: {e}")

        # Step 9.3: rhythm_editor → polished.mp4
        print("  → rhythm_editor: 节奏编辑...")
        final_out = str(output_dir / "polished.mp4")
        # A failed rerun must not inherit a receipt from an older artifact.
        (output_dir / "delivery_timeline.json").unlink(missing_ok=True)
        try:
            edit_rhythm(
                video_path=current_video,
                storyboard_path=sb_path_str,
                timeline_path=str(output_dir / "edit_timeline.json"),
                output_path=final_out,
            )
            outputs.append("polished.mp4")
            step_status["rhythm_editor"] = "done"
        except Exception as e:
            print(f"  ⚠ rhythm_editor failed: {e}")
            step_status["rhythm_editor"] = "failed"
            # Fallback: just copy video
            import shutil
            shutil.copy2(current_video, final_out)
            outputs.append("polished.mp4")

        # Step 9.4: Final encoding with media profile
        print(f"  → final_encode: 使用 {media_profile} 配置重新编码...")
        final_encoded = str(output_dir / "polished_final.mp4")
        profile = _get_profile_dict(media_profile)
        encode_input_durations = _probe_av_durations(Path(final_out))
        delivery_contract = _validated_reviewed_delivery_contract(
            output_dir,
            Path(final_out),
            encode_input_durations,
            fps=float(profile["fps"]),
        )

        video_filters, audio_filters = _final_encode_filters(profile)

        cmd = [
            "ffmpeg", "-y",
            "-i", final_out,
            "-vf", video_filters,
            "-af", audio_filters,
            "-c:v", profile["codec"],
            "-crf", str(profile["crf"]),
            "-preset", "medium",
            "-c:a", profile["audio_codec"],
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", profile["pixel_format"],
            final_encoded,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            step_status["final_encode"] = "failed"
            raise RuntimeError(f"Final encoding failed: {result.stderr[-1000:]}")

        encoded_durations = _probe_av_durations(Path(final_encoded))
        final_duration_gate = _final_encode_duration_gate(
            encode_input_durations,
            encoded_durations,
            delivery_contract=delivery_contract,
            requested_duration=target_duration,
            fps=float(profile["fps"]),
        )
        duration_tolerance_s = final_duration_gate["tolerance_s"]["video"]
        audio_duration_tolerance_s = final_duration_gate["tolerance_s"]["audio"]
        _assert_duration_conserved(
            encode_input_durations,
            encoded_durations,
            tolerance_s=duration_tolerance_s,
            audio_tolerance_s=audio_duration_tolerance_s,
        )
        if not final_duration_gate["passed"]:
            raise RuntimeError(
                "Final duration gate rejected the encoded candidate before promotion: "
                f"{final_duration_gate}"
            )

        # Only promote the encoded artifact after its independent A/V duration
        # assertions pass.  The delivery gate deliberately probes polished.mp4.
        import shutil
        shutil.move(final_encoded, final_out)
        polished_durations = _probe_av_durations(Path(final_out))
        final_duration_gate = _final_encode_duration_gate(
            encode_input_durations,
            polished_durations,
            delivery_contract=delivery_contract,
            requested_duration=target_duration,
            fps=float(profile["fps"]),
        )
        (output_dir / "final_duration_gate.json").write_text(
            json.dumps(final_duration_gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _assert_duration_conserved(
            encode_input_durations,
            polished_durations,
            tolerance_s=duration_tolerance_s,
            audio_tolerance_s=audio_duration_tolerance_s,
        )
        if not final_duration_gate["passed"]:
            raise RuntimeError(
                "Final duration gate failed to conserve the reviewed encode input: "
                f"expected={encode_input_durations}, actual={polished_durations}"
            )
        outputs.extend([
            f"polished.mp4 (encoded with {media_profile})",
            "final_duration_gate.json",
        ])
        print(f"    ✓ 最终编码完成: {profile['width']}x{profile['height']} @ {profile['fps']}fps")
        step_status["final_encode"] = "done"
        step_status["final_duration_gate"] = "done"

        # Final character-animation QA runs against the delivered video and
        # persists its complete structured result for later inspection.
        character_qa_result = None
        try:
            from quality.character_qa import CharacterAnimationQA

            character_video = output_dir / "polished.mp4"
            characters_json = output_dir / "CHARACTERS.json"
            qa_started = time.time()
            print(
                f"  → [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"CharacterAnimationQA.check: video={character_video}, "
                f"characters={characters_json}"
            )
            character_tool_result = CharacterAnimationQA().execute({
                "operation": "full_qa",
                "video_path": str(character_video),
                "characters_json_path": str(characters_json),
                "output_dir": str(output_dir / "character_qa_samples"),
            })
            if character_tool_result.success:
                qa_data = character_tool_result.data or {}
                verdict = qa_data.get("verdict", "unknown")
                grade = {"pass": "A", "revise": "C", "fail": "D"}.get(verdict, "N/A")
                character_qa_result = {
                    "status": "success",
                    "grade": grade,
                    "duration_seconds": character_tool_result.duration_seconds,
                    **qa_data,
                }
                (output_dir / "character_qa_report.json").write_text(
                    json.dumps(character_qa_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                outputs.append("character_qa_report.json")
                step_status["character_qa"] = "done" if verdict == "pass" else "failed"
                print(
                    f"  ✓ Character QA 完成: elapsed={time.time() - qa_started:.1f}s, "
                    f"grade={grade}, verdict={verdict}, issues={len(qa_data.get('issues', []))}"
                )
                print(f"    CharacterAnimationQA result: {json.dumps(qa_data, ensure_ascii=False, default=str)}")
            else:
                character_qa_result = {"status": "failed", "error": character_tool_result.error}
                step_status["character_qa"] = "failed"
                print(
                    f"  ⚠ Character QA 失败: elapsed={time.time() - qa_started:.1f}s, "
                    f"error={character_tool_result.error}"
                )
        except Exception as e:
            character_qa_result = {"status": "skipped", "reason": str(e)}
            step_status["character_qa"] = "skipped"
            print(f"  ⚠ Character QA 不可用: {e}")

        print(f"  ✓ Phase 9 完成: polished.mp4")

        # Quality gate: Phase 9
        qg_report = run_quality_check("phase9", output_dir, step_status=step_status)
        if not qg_report.passed:
            return {"status": "error", "error": f"Phase 9 质检未通过: {qg_report.grade}", "quality_report": qg_report, "duration_s": _elapsed(start)}

        return {
            "status": "done",
            "duration_s": _elapsed(start),
            "outputs": outputs,
            "color_grade": color_grade,
            "upscale": upscale,
            "audio_enhanced": audio_success,
            "bgm_detected": bgm_path is not None,
            "media_profile": media_profile,
            "step_status": step_status,
            "character_qa": character_qa_result,
        }
    except ImportError as e:
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "duration_s": _elapsed(start)}
