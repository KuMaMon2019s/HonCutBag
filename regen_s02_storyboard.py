"""Regenerate S02 storyboard image with pure-mechanical characters.

The old S02.png was generated before silver_tech was regenerated as a pure
mechanical character, so it contained a realistic human face which triggered
Seedance PrivacyInformation detection in Phase 5 (content[1]).
"""
import sys
import os

sys.path.insert(0, "pipeline/src")
os.environ.setdefault("VIDEO_PROVIDER", "seedance")

from pathlib import Path
from clients.seedream_client import SeedreamClient

output_dir = Path("workspaces/2026-08-08_02/output")
shot_image_path = output_dir / "storyboard_images" / "S02.png"
ref_image_path = output_dir / "characters" / "black_merc" / "face_closeup.png"

shot_prompt = (
    "Cyberpunk underground mech black-market repair station. A black heavy "
    "mechanical synthetic mercenary (full armor plating, single red optical "
    "sensor eye, '07' marking on shoulder) walks into the repair station and "
    "places a data chip on the workbench. Beside stands a silver-white "
    "mechanical technician with a smooth metal faceplate, blue optical sensor "
    "strip and six repair mechanical arms. Dim industrial lighting, holographic "
    "projection, corrugated metal walls, scattered gears and parts. Cinematic "
    "composition, 16:9, realistic sci-fi style. "
    "Virtual avatar declaration: all characters are AI-generated fictional "
    "fully-mechanical cybernetic beings, not real people, no human skin or "
    "facial features."
)

client = SeedreamClient()
client.image_to_image(
    prompt=shot_prompt,
    ref_image=str(ref_image_path),
    output_path=str(shot_image_path),
    size="2560x1440",
)
print("S02 storyboard regenerated:", shot_image_path)
print("size:", shot_image_path.stat().st_size)
