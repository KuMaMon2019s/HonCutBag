"""Regenerate S02's black_merc reference images with stylized/mechanical look.

Goal: bypass Seedance PrivacyInformation detection by removing photorealistic
human face cues (real hair, skin) and emphasizing pure synthetic mechanical head.
Identity is anchored to character 07 (black-gray titanium armor, right red
monocular optic, chest blue energy core, left shoulder "07").
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline", "src"))

from clients.seedream_client import SeedreamClient

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "workspaces", "2026-08-08_02", "output", "characters", "black_merc",
)

FICTIONAL = (
    "This is a fully fictional AI-generated synthetic android character, "
    "a virtual avatar, NOT a real person and NOT a human: "
)

IDENTITY = (
    "a tall heavy-duty mechanical synthetic mercenary, matte black-gray titanium alloy "
    "armor plates with visible panel seams and rivets, a completely mechanical head "
    "with NO human skin and NO human hair, angular metallic faceplate, a single glowing "
    "red monocular visual optic module on the right side of the head, a circular glowing "
    "blue energy core embedded in the chest, the designation number 07 engraved on the "
    "left shoulder plate, folded mechanical structure on the back, hydraulic joints and "
    "energy conduits visible between armor plates"
)

STYLE = (
    "cyberpunk sci-fi concept art, cinematic rim lighting, neon reflections, "
    "high detail mechanical surface rendering, 8K"
)

SOURCE_RULES = (
    "front-facing or three-quarter view, solid dark neutral background, "
    "no strong shadows, character occupies most of the frame"
)

FACE_PROMPT = (
    f"{FICTIONAL}{IDENTITY}, head-and-shoulders close-up, "
    f"the mechanical head fills 70 percent of the frame, {STYLE}, {SOURCE_RULES}"
)

BODY_PROMPT = (
    f"{FICTIONAL}{IDENTITY}, full-body standing pose, complete armor and "
    f"heavy mechanical boots visible head to toe, {STYLE}, {SOURCE_RULES}"
)


def main() -> None:
    client = SeedreamClient()
    face_path = os.path.join(OUT_DIR, "face_closeup.png")
    body_path = os.path.join(OUT_DIR, "full_body.png")

    print("[1/2] Generating stylized face close-up ...", flush=True)
    client.text_to_image(prompt=FACE_PROMPT, output_path=face_path, size="1920x1920")
    print("  ->", face_path, os.path.getsize(face_path), "bytes", flush=True)

    print("[2/2] Generating stylized full body ...", flush=True)
    client.text_to_image(prompt=BODY_PROMPT, output_path=body_path, size="1920x1920")
    print("  ->", body_path, os.path.getsize(body_path), "bytes", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
