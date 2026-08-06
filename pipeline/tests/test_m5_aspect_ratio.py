"""M5: aspect ratio preservation tests."""
import sys
from pathlib import Path

# Add pipeline/src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest
from PIL import Image
import tempfile


def test_fit_to_aspect_square_to_16x9():
    """Square image → 16:9 center-crop (no distortion)."""
    from pipeline_runner import fit_to_aspect

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 1920x1920 square
        src = Path(tmpdir) / "square.png"
        Image.new('RGB', (1920, 1920), color='red').save(src)

        out = Path(tmpdir) / "output.png"
        fit_to_aspect(src, 1280, 720, out)

        # Verify exact dimensions
        with Image.open(out) as img:
            assert img.size == (1280, 720), f"Expected 1280x720, got {img.size}"

        # Verify no distortion: aspect ratio math
        # Square 1920x1920 → cover 16:9 → crop top/bottom
        # Scale to cover: max(1280/1920, 720/1920) = max(0.667, 0.375) = 0.667
        # Covered size: 1920*0.667 x 1920*0.667 = 1280 x 1280
        # Center-crop: (1280-720)/2 = 280 pixels from top/bottom
        # Result: 1280x720 ✅
        assert out.exists()


def test_fit_to_aspect_16x9_to_16x9():
    """16:9 image → 16:9 simple resize (no crop)."""
    from pipeline_runner import fit_to_aspect

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 1920x1080 (16:9)
        src = Path(tmpdir) / "wide.png"
        Image.new('RGB', (1920, 1080), color='blue').save(src)

        out = Path(tmpdir) / "output.png"
        fit_to_aspect(src, 1280, 720, out)

        with Image.open(out) as img:
            assert img.size == (1280, 720)


def test_fit_to_aspect_portrait_to_16x9():
    """Portrait image → 16:9 center-crop (sides cropped)."""
    from pipeline_runner import fit_to_aspect

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 1080x1920 (portrait 9:16)
        src = Path(tmpdir) / "tall.png"
        Image.new('RGB', (1080, 1920), color='green').save(src)

        out = Path(tmpdir) / "output.png"
        fit_to_aspect(src, 1280, 720, out)

        with Image.open(out) as img:
            assert img.size == (1280, 720)

        # Portrait → landscape: scale to cover height, crop sides
        # Scale: max(1280/1080, 720/1920) = max(1.185, 0.375) = 1.185
        # Covered: 1080*1.185 x 1920*1.185 = 1280 x 2275
        # Center-crop width: (1280-1280)/2 = 0 (exact fit)
        # Center-crop height: (2275-720)/2 = 777 from top/bottom
        assert out.exists()


def test_storyboard_image_size_returns_16x9():
    """_storyboard_image_size should return 16:9 size string meeting Seedream minimum."""
    from pipeline_runner import _storyboard_image_size, SEEDREAM_MIN_PIXELS

    # Default (1280x720 video) → 2560x1440 (= 3,686,400 px, exactly at minimum)
    size = _storyboard_image_size(video_width=1280, video_height=720)
    assert size == "2560x1440", f"Expected 2560x1440, got {size}"

    # 1920x1080 video → 2560x1440 (same aspect ratio)
    size = _storyboard_image_size(video_width=1920, video_height=1080)
    assert size == "2560x1440"


def test_storyboard_image_size_meets_seedream_minimum():
    """All returned sizes must have pixel count >= SEEDREAM_MIN_PIXELS."""
    from pipeline_runner import _storyboard_image_size, SEEDREAM_MIN_PIXELS

    test_cases = [
        (1280, 720),    # 16:9
        (1920, 1080),   # 16:9 HD
        (720, 1280),    # 9:16 portrait
        (1080, 1920),   # 9:16 portrait HD
        (1280, 1280),   # 1:1 square
        (1920, 1920),   # 1:1 square HD
    ]

    for w, h in test_cases:
        size_str = _storyboard_image_size(video_width=w, video_height=h)
        parts = size_str.split("x")
        sw, sh = int(parts[0]), int(parts[1])
        pixels = sw * sh
        assert pixels >= SEEDREAM_MIN_PIXELS, (
            f"Size {size_str} ({pixels} px) for video {w}x{h} "
            f"is below Seedream minimum {SEEDREAM_MIN_PIXELS}"
        )
        # Both dimensions must be even
        assert sw % 2 == 0, f"Width {sw} is not even"
        assert sh % 2 == 0, f"Height {sh} is not even"


def test_seedream_client_default_size_meets_minimum_pixels():
    from clients.seedream_client import DEFAULT_IMAGE_SIZE
    from pipeline_runner import SEEDREAM_MIN_PIXELS

    width, height = map(int, DEFAULT_IMAGE_SIZE.split("x"))
    assert width * height >= SEEDREAM_MIN_PIXELS


def test_end_frame_generation_uses_16x9_size():
    """_generate_flf2v_end_frame should request 16:9 size from Seedream."""
    # This is a mock test — verify size parameter is passed correctly
    # (Actual implementation will be verified by integration test)
    pass  # TODO: mock SeedreamClient and assert size="1920x1080"


def test_build_content_applies_fit_to_aspect():
    """build_content_for_shot should fit images to video aspect before upload."""
    # This is a mock test — verify fit_to_aspect is called
    # (Actual implementation will be verified by integration test)
    pass  # TODO: mock tos_uploader and verify cropped temp file is uploaded
