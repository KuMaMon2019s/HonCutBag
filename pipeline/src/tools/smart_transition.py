"""Smart transition decision engine — three-layer voting.

Layer 1: Semantic (emotion + narrative structure) — weight 0.4
Layer 2: Visual (frame embedding cosine similarity) — weight 0.4
Layer 3: Rhythm (shot duration + pacing) — weight 0.2

Final decision = weighted vote → cut / dissolve / fade
"""

from typing import Optional

# ─── Layer 1: Semantic (Emotion + Narrative) ─────────────────────────────────

EMOTION_TRANSITION_MAP = {
    # Gentle emotions → dissolve
    "温柔": "dissolve", "深情": "dissolve", "心动": "dissolve",
    "欣喜": "dissolve", "喜悦": "dissolve", "暧昧": "dissolve",
    "羞涩": "dissolve", "温暖": "dissolve", "感动": "dissolve",
    "回忆": "dissolve", "平静": "dissolve", "日常": "dissolve",
    # Intense emotions → cut
    "紧张": "cut", "愤怒": "cut", "惊讶": "cut", "震惊": "cut",
    "慌乱": "cut", "压迫": "cut", "冲突": "cut", "追逐": "cut",
    # Sad/transitional emotions → fade
    "悲伤": "fade", "失落": "fade", "离别": "fade", "隐忍": "fade",
    "克制": "fade", "孤独": "fade", "惆怅": "fade",
}


def semantic_transition(emotion: str, scene_changed: bool = False) -> str:
    """Layer 1: Decide transition based on emotion and scene change."""
    if scene_changed:
        return "fade"
    if not emotion:
        return "dissolve"
    if emotion in EMOTION_TRANSITION_MAP:
        return EMOTION_TRANSITION_MAP[emotion]
    for key, value in EMOTION_TRANSITION_MAP.items():
        if key in emotion:
            return value
    return "dissolve"


# ─── Layer 2: Visual (Cosine Similarity) ─────────────────────────────────────

def visual_transition(cosine_similarity: float) -> str:
    """Layer 2: Decide transition based on visual similarity.

    >0.85 → cut (visually continuous)
    0.5-0.85 → dissolve (some change, smooth blend)
    <0.5 → fade (big jump, ease with black)
    """
    if cosine_similarity > 0.85:
        return "cut"
    elif cosine_similarity >= 0.5:
        return "dissolve"
    else:
        return "fade"


# ─── Layer 3: Rhythm (Duration + Pacing) ─────────────────────────────────────

def rhythm_transition(curr_duration: float, next_duration: float) -> str:
    """Layer 3: Decide transition based on shot durations.

    Both short (<3s) → cut (fast pacing)
    Either long (>8s) → dissolve (slow pacing)
    """
    if curr_duration < 3.0 and next_duration < 3.0:
        return "cut"
    elif curr_duration > 8.0 or next_duration > 8.0:
        return "dissolve"
    return "dissolve"


# ─── Three-Layer Voting ──────────────────────────────────────────────────────

WEIGHTS = {"semantic": 0.4, "visual": 0.4, "rhythm": 0.2}


def decide_transition(
    emotion: str = "",
    scene_changed: bool = False,
    cosine_similarity: Optional[float] = None,
    curr_duration: float = 6.0,
    next_duration: float = 6.0,
) -> dict:
    """Three-layer weighted voting for transition decision.

    Returns:
        {"decision": "cut"|"dissolve"|"fade", "layers": {...}, "scores": {...}}
    """
    sem_choice = semantic_transition(emotion, scene_changed)

    if cosine_similarity is not None:
        vis_choice = visual_transition(cosine_similarity)
    else:
        vis_choice = "dissolve"
        cosine_similarity = -1

    rhy_choice = rhythm_transition(curr_duration, next_duration)

    scores = {"cut": 0.0, "dissolve": 0.0, "fade": 0.0}
    scores[sem_choice] += WEIGHTS["semantic"]
    scores[vis_choice] += WEIGHTS["visual"]
    scores[rhy_choice] += WEIGHTS["rhythm"]

    decision = max(scores, key=scores.get)

    return {
        "decision": decision,
        "layers": {
            "semantic": {"choice": sem_choice, "weight": WEIGHTS["semantic"], "emotion": emotion},
            "visual": {"choice": vis_choice, "weight": WEIGHTS["visual"], "similarity": cosine_similarity},
            "rhythm": {"choice": rhy_choice, "weight": WEIGHTS["rhythm"]},
        },
        "scores": {k: round(v, 2) for k, v in scores.items()},
    }


# ─── Batch Decision ──────────────────────────────────────────────────────────

def decide_all_transitions(
    shot_metas: list,
    similarities: Optional[dict] = None,
) -> list:
    """Decide transitions for all adjacent shot pairs.

    Args:
        shot_metas: List of shot metadata dicts (emotion, duration, where)
        similarities: {"S01->S02": cosine, ...} from compute_transition_similarity()

    Returns:
        List of transition decisions, one per adjacent pair
    """
    if similarities is None:
        similarities = {}

    decisions = []
    for i in range(len(shot_metas) - 1):
        curr = shot_metas[i]
        next_shot = shot_metas[i + 1]

        curr_id = f"S{i+1:02d}"
        next_id = f"S{i+2:02d}"
        pair_key = f"{curr_id}->{next_id}"

        emotion = curr.get("emotion", "")
        scene_changed = curr.get("where", "") != next_shot.get("where", "")
        cosine = similarities.get(pair_key)
        curr_dur = curr.get("suggested_duration", curr.get("duration", 6.0))
        next_dur = next_shot.get("suggested_duration", next_shot.get("duration", 6.0))

        result = decide_transition(
            emotion=emotion,
            scene_changed=scene_changed,
            cosine_similarity=cosine,
            curr_duration=curr_dur,
            next_duration=next_dur,
        )
        result["pair"] = pair_key
        decisions.append(result)

    return decisions
