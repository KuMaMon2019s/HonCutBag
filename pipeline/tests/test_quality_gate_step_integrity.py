#!/usr/bin/env python3
"""
回归测试：质检诚信机制

验证：当关键步骤失败或跳过时，质检等级必须降级，不能给 A。
这是为了防止 Phase 6-8 多项失败（字幕烧录、片尾ffmpeg、rhythm仅EDL、音频跳过、crossfade未渲染）
但最终质检仍给 A 的假阳性问题。

测试场景：
1. 字幕烧录失败 → grade 必须 < A
2. 音频处理失败 → grade 必须 < A
3. 所有步骤成功 → grade = A
4. 非关键步骤跳过 → 不影响 A 级
"""

import tempfile
import subprocess
from pathlib import Path
from quality_gate import run_quality_check


def _create_test_video(path: Path, has_audio: bool = False):
    """创建一个真实的测试视频文件（10秒彩色噪声视频，确保 > 500KB）"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=10:size=1280x720:rate=30",
    ]
    if has_audio:
        cmd.extend([
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-c:a", "aac", "-shortest",
        ])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(path),
    ])
    subprocess.run(cmd, capture_output=True, check=True, timeout=30)


def test_subtitle_burn_failure_downgrades_grade():
    """字幕烧录失败时，质检等级必须降级，不能给 A"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 模拟字幕烧录失败
        step_status = {
            "audio_pipeline": "done",
            "subtitle_burn": "failed",  # 关键步骤失败
            "rhythm_editor": "done",
            "final_encode": "done",
        }
        
        report = run_quality_check("phase8", output_dir, step_status=step_status)
        
        # 断言：grade 不能是 A
        assert report.grade != "A", f"字幕烧录失败时 grade 不能是 A，实际是 {report.grade}"
        assert report.grade in ["B", "C", "D"], f"grade 应该是 B/C/D，实际是 {report.grade}"
        assert not report.passed, "字幕烧录失败时 passed 应该是 False"
        
        # 验证 step_summary 包含失败信息
        assert "subtitle_burn" in report.step_summary
        assert report.step_summary["subtitle_burn"] == "failed"
        
        print(f"✓ 测试通过：字幕烧录失败 → grade={report.grade}, passed={report.passed}")


def test_audio_pipeline_failure_downgrades_grade():
    """音频处理失败时，质检等级必须降级"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频（带音频）
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 模拟音频处理失败
        step_status = {
            "audio_pipeline": "failed",  # 关键步骤失败
            "subtitle_burn": "done",
            "rhythm_editor": "done",
            "final_encode": "done",
        }
        
        report = run_quality_check("phase8", output_dir, step_status=step_status)
        
        assert report.grade != "A", f"音频处理失败时 grade 不能是 A，实际是 {report.grade}"
        assert not report.passed, "音频处理失败时 passed 应该是 False"
        
        print(f"✓ 测试通过：音频处理失败 → grade={report.grade}, passed={report.passed}")


def test_all_steps_success_gets_grade_a():
    """所有关键步骤成功时，质检等级应该是 A"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频（带音频）
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 所有步骤成功
        step_status = {
            "audio_pipeline": "done",
            "subtitle_burn": "done",
            "rhythm_editor": "done",
            "final_encode": "done",
        }
        
        report = run_quality_check("phase8", output_dir, step_status=step_status)
        
        assert report.grade == "A", f"所有步骤成功时 grade 应该是 A，实际是 {report.grade}"
        assert report.passed, "所有步骤成功时 passed 应该是 True"
        
        print(f"✓ 测试通过：所有步骤成功 → grade={report.grade}, passed={report.passed}")


def test_critical_step_skipped_downgrades_grade():
    """关键步骤跳过时，质检等级必须降级"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频（带音频）
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 关键步骤跳过（比如字幕烧录没有数据）
        step_status = {
            "audio_pipeline": "done",
            "subtitle_burn": "skipped",  # 关键步骤跳过
            "rhythm_editor": "done",
            "final_encode": "done",
        }
        
        report = run_quality_check("phase8", output_dir, step_status=step_status)
        
        # 字幕烧录是 critical_steps 之一，所以跳过也会降级
        assert report.grade != "A", f"关键步骤跳过时 grade 不能是 A，实际是 {report.grade}"
        
        print(f"✓ 测试通过：关键步骤跳过 → grade={report.grade}（降级）")


def test_multiple_critical_failures():
    """多个关键步骤失败时，质检等级应该更低"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频（带音频）
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 多个关键步骤失败
        step_status = {
            "audio_pipeline": "failed",
            "subtitle_burn": "failed",
            "rhythm_editor": "failed",
            "final_encode": "done",
        }
        
        report = run_quality_check("phase8", output_dir, step_status=step_status)
        
        # 3 个关键步骤失败，应该是 D 级
        assert report.grade == "D", f"3 个关键步骤失败时 grade 应该是 D，实际是 {report.grade}"
        assert not report.passed
        
        print(f"✓ 测试通过：3 个关键步骤失败 → grade={report.grade}")


def test_backward_compatibility_without_step_status():
    """向后兼容：不传 step_status 时，行为与之前一致"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建真实的测试视频（带音频）
        polished = output_dir / "polished.mp4"
        _create_test_video(polished, has_audio=True)
        
        # 不传 step_status
        report = run_quality_check("phase8", output_dir)
        
        # 应该基于文件检查给 A（polished.mp4 存在且 > 500KB，有视频/音频流）
        assert report.grade == "A", f"不传 step_status 时应该基于文件检查，实际 grade={report.grade}"
        assert report.passed
        
        print(f"✓ 测试通过：向后兼容（不传 step_status）→ grade={report.grade}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("质检诚信机制回归测试")
    print("=" * 70 + "\n")
    
    tests = [
        ("字幕烧录失败降级", test_subtitle_burn_failure_downgrades_grade),
        ("音频处理失败降级", test_audio_pipeline_failure_downgrades_grade),
        ("所有步骤成功得 A", test_all_steps_success_gets_grade_a),
        ("关键步骤跳过降级", test_critical_step_skipped_downgrades_grade),
        ("多个关键步骤失败", test_multiple_critical_failures),
        ("向后兼容", test_backward_compatibility_without_step_status),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            print(f"\n测试: {test_name}")
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ 测试失败: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
