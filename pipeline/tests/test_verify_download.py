#!/usr/bin/env python
"""Test _verify_download function with a fake mp4 file."""
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clients.local_video_client import _verify_download

def test_verify_download_mismatch():
    """Test that _verify_download raises RuntimeError on mismatch."""
    # Create a fake 1KB mp4 file (not a real video, just garbage)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"\x00" * 1024)
        fake_path = f.name

    try:
        # This should fail because ffprobe can't parse the fake file
        print("Testing _verify_download with fake mp4...")
        try:
            result = _verify_download(
                fake_path,
                expected_duration=5.0,
                expected_width=1280,
                expected_height=720,
            )
            print(f"✗ FAIL: Expected RuntimeError but got result: {result}")
            return False
        except RuntimeError as e:
            print(f"✓ PASS: Got expected RuntimeError: {e}")
            return True
    finally:
        # Cleanup
        try:
            os.remove(fake_path)
        except:
            pass

if __name__ == "__main__":
    success = test_verify_download_mismatch()
    sys.exit(0 if success else 1)
