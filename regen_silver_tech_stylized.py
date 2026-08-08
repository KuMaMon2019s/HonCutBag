"""Regenerate silver_tech reference images as pure mechanical body.

The script defines her as: slender silver mechanical body, head covered by a
smooth metal faceplate, six precision repair mechanical arms on the back.
Goal: remove photorealistic human face/hair cues that trigger Seedance
PrivacyInformation detection.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline", "src"))

from clients.seedream_client import SeedreamClient

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "workspaces", "2026-08-08_02", "output", "characters", "silver_tech",
)

FICTIONAL = (
    "This is a fully fictional AI-generated synthetic android character, "
    "a virtual avatar, NOT a real person and NOT a human: "
)

IDENTITY = (
    "a slender elegant female mechanical technician android, entirely silver-white "
    "mechanical body with polished titanium alloy panels, a smooth featureless metal "
    "faceplate covering the head with NO human skin and NO human hair, subtle glowing "
    "pale-blue optical sensor lines on the faceplate, six folded precision repair "
    "mechanical arms deployed from the back, fine articulated joints and delicate "
    "mechanical fingers, silver-gray fitted workwear plating over the chassis"
)

STYLE = (
    "cyberpunk sci-fi concept art, cinematic rim lighting, cool neon reflections, "
    "high detail mechanical surface rendering, 8K"
)

SOURCE_RULES = (
    "front-facing or three-quarter view, solid dark neutral background, "
    "no strong shadows, character occupies most of the frame"
)

FACE_PROMPT = (
    f"{FICTIONAL}{IDENTITY}, head-and-shoulders close-up, "
    f"the smooth metal faceplate fills 70 percent of the frame, {STYLE}, {SOURCE_RULES}"
)

BODY_PROMPT = (
    f"{FICTIONAL}{IDENTITY}, full-body standing pose, all six back-mounted repair "
    f"arms visible, complete chassis and footwear from head to toe, {STYLE}, {SOURCE_RULES}"
)


def main() -> None:
    client = SeedreamClient()
    face_path = os.path.join(OUT_DIR, "face_closeup.png")
    body_path = os.path.join(OUT_DIR, "full_body.png")

    os.makedirs(os.path.join(OUT_DIR, "original_backup"), exist_ok=True)
    for name in ("face_closeup.png", "full_body.png"):
        src = os.path.join(OUT_DIR, name)
        bak = os.path.join(OUT_DIR, "original_backup", name)
        if os.path.exists(src) and not os.path.exists(bak):
            os.replace(src, bak)

    print("[1/2] Generating stylized face close-up ...", flush=True)
    client.text_to_image(prompt=FACE_PROMPT, output_path=face_path, size="1920x1920")
    print("  ->", face_path, os.path.getsize(face_path), "bytes", flush=True)

    print("[2/2] Generating stylized full body ...", flush=True)
    client.text_to_image(prompt=BODY_PROMPT, output_path=body_path, size="1920x1920")
    print("  ->", body_path, os.path.getsize(body_path), "bytes", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
