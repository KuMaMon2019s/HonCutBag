"""Shot frame extraction + doubao-embedding-vision embedding + Qdrant storage.

Extracts first/last frames from each shot video, embeds them via
Volcano Ark doubao-embedding-vision API, and stores vectors in Qdrant
for visual similarity-based transition decisions.
"""

import os
import base64
import subprocess
from pathlib import Path
from typing import Optional

import requests

# ─── Config ──────────────────────────────────────────────────────────────────

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
EMBEDDING_MODEL = "doubao-embedding-vision"
QDRANT_URL = "http://127.0.0.1:6335"
COLLECTION_NAME = "shot_frames"


def _get_api_key() -> str:
    return os.environ.get("ARK_AGENT_API_KEY", "")


# ─── Frame Extraction ────────────────────────────────────────────────────────

def extract_keyframes(video_path: str, output_dir: Optional[str] = None) -> dict:
    """Extract first and last frames from a video using ffmpeg.

    Returns: {"first": path, "last": path}
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {}

    if output_dir is None:
        output_dir = str(video_path.parent / "keyframes")
    os.makedirs(output_dir, exist_ok=True)

    stem = video_path.stem
    first_path = os.path.join(output_dir, f"{stem}_first.png")
    last_path = os.path.join(output_dir, f"{stem}_last.png")

    # Get video duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = 6.0

    # Extract first frame (at 0.1s)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0.1", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", first_path],
        capture_output=True, timeout=15,
    )

    # Extract last frame
    last_ts = max(0.1, duration - 0.2)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(last_ts), "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", last_path],
        capture_output=True, timeout=15,
    )

    frames = {}
    if os.path.exists(first_path) and os.path.getsize(first_path) > 0:
        frames["first"] = first_path
    if os.path.exists(last_path) and os.path.getsize(last_path) > 0:
        frames["last"] = last_path
    return frames


# ─── Embedding ───────────────────────────────────────────────────────────────

def embed_image(image_path: str) -> Optional[list]:
    """Embed an image using doubao-embedding-vision API."""
    api_key = _get_api_key()
    if not api_key:
        return None

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}],
    }

    try:
        resp = requests.post(f"{ARK_BASE_URL}/embeddings", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"  [embed] API error {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        return data.get("data", [{}])[0].get("embedding")
    except Exception as e:
        print(f"  [embed] Error: {e}")
        return None


# ─── Qdrant Storage ──────────────────────────────────────────────────────────

def _ensure_collection(dim: int = 1024) -> bool:
    """Ensure Qdrant collection exists."""
    try:
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", timeout=5)
        if resp.status_code == 200:
            return True
        requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
            json={"vectors": {"size": dim, "distance": "Cosine"}},
            timeout=10,
        )
        return True
    except Exception as e:
        print(f"  [qdrant] Collection setup failed: {e}")
        return False


def store_embedding(shot_id: str, frame_type: str, vector: list,
                    payload: Optional[dict] = None) -> bool:
    """Store an embedding vector in Qdrant."""
    if not _ensure_collection(len(vector)):
        return False

    point_id = abs(hash(f"{shot_id}_{frame_type}")) % (2**63)
    point_payload = {"shot_id": shot_id, "frame_type": frame_type, **(payload or {})}

    try:
        requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": [{"id": point_id, "vector": vector, "payload": point_payload}]},
            params={"wait": "true"},
            timeout=10,
        )
        return True
    except Exception as e:
        print(f"  [qdrant] Store failed: {e}")
        return False


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def embed_all_shots(shots_dir: str, run_id: str = "") -> dict:
    """Extract keyframes and embed all shots.

    Returns: {shot_id: {"first": vector, "last": vector}}
    """
    shots_dir = Path(shots_dir)
    embeddings = {}
    shot_dirs = sorted(d for d in shots_dir.iterdir() if d.is_dir() and d.name.startswith("S"))

    print(f"  → 抽帧+向量化: {len(shot_dirs)} 个镜头")

    for shot_dir in shot_dirs:
        shot_id = shot_dir.name
        video_path = shot_dir / "output.mp4"
        if not video_path.exists():
            continue

        frames = extract_keyframes(str(video_path))
        if not frames:
            continue

        shot_embeddings = {}
        for frame_type, frame_path in frames.items():
            vector = embed_image(frame_path)
            if vector:
                shot_embeddings[frame_type] = vector
                store_embedding(shot_id, frame_type, vector, {"run_id": run_id})

        if shot_embeddings:
            dim = len(next(iter(shot_embeddings.values())))
            embeddings[shot_id] = shot_embeddings
            print(f"    ✓ {shot_id}: {list(shot_embeddings.keys())} ({dim}维)")

    print(f"  → 向量化完成: {len(embeddings)}/{len(shot_dirs)} 个镜头")
    return embeddings


def compute_transition_similarity(embeddings: dict) -> dict:
    """Compute visual similarity between adjacent shots (last frame N vs first frame N+1).

    Returns: {"S01->S02": cosine_similarity, ...}
    """
    import numpy as np

    shot_ids = sorted(embeddings.keys())
    similarities = {}

    for i in range(len(shot_ids) - 1):
        curr_last = embeddings[shot_ids[i]].get("last")
        next_first = embeddings[shot_ids[i + 1]].get("first")

        if curr_last and next_first:
            a = np.array(curr_last, dtype=np.float32)
            b = np.array(next_first, dtype=np.float32)
            a = a / (np.linalg.norm(a) + 1e-8)
            b = b / (np.linalg.norm(b) + 1e-8)
            cosine = float(np.dot(a, b))
            similarities[f"{shot_ids[i]}->{shot_ids[i+1]}"] = round(cosine, 4)

    return similarities
