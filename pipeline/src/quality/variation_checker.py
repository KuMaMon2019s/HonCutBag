"""Eight structural scene-variation checks adapted from OpenMontage."""

from collections import Counter
from typing import Any

GENERIC_PHRASES = {"a person", "a beautiful", "modern", "futuristic", "cutting-edge", "sleek design", "innovative", "state-of-the-art", "next-generation", "revolutionary", "a professional", "dynamic", "vibrant", "stunning", "breathtaking", "amazing", "incredible", "powerful", "seamless", "elegant solution"}


def check_scene_variation(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    if not scenes:
        return {"score": 5.0, "verdict": "fail", "violations": ["No scenes to check"], "suggestions": []}
    violations, suggestions = [], []
    sizes = [s.get("shot_language", {}).get("shot_size", "unspecified") for s in scenes]
    if len(scenes) >= 4:
        size, count = Counter(sizes).most_common(1)[0]
        if count / len(scenes) > .5:
            violations.append(f"Shot size '{size}' used in {count}/{len(scenes)} scenes ({count/len(scenes):.0%}). Vary shot sizes for visual interest.")
            suggestions.append("Mix wide establishing shots with close-ups for visual rhythm.")
    longest = current = 1
    for previous, size in zip(sizes, sizes[1:]):
        current = current + 1 if size == previous and size != "unspecified" else 1
        longest = max(longest, current)
    if longest >= 3: violations.append(f"{longest} consecutive same-size shots. Vary shot sizes between scenes for editorial rhythm.")
    movements = [s.get("shot_language", {}).get("camera_movement", "unspecified") for s in scenes]
    static = sum(m in ("static", "unspecified") for m in movements)
    if len(scenes) >= 4 and static / len(scenes) > .6:
        violations.append(f"{static}/{len(scenes)} scenes are static or unspecified movement. Add intentional camera movement to at least 40% of scenes.")
        suggestions.append("Consider dolly_in for emphasis, tracking for energy, or crane for scale.")
    lighting = {s.get("shot_language", {}).get("lighting_key") for s in scenes if s.get("shot_language", {}).get("lighting_key")}
    if len(scenes) >= 4 and len(lighting) <= 1: violations.append(f"Only {len(lighting)} unique lighting setup(s) across {len(scenes)} scenes. Vary lighting to create mood shifts.")
    heroes = [(i, s) for i, s in enumerate(scenes) if s.get("hero_moment")]
    if len(scenes) >= 4 and not heroes:
        violations.append("No hero_moment flagged. Every video should have at least one visual peak.")
        suggestions.append("Mark the most impactful scene as hero_moment=true.")
    for index, hero in heroes:
        hero_size = hero.get("shot_language", {}).get("shot_size")
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(scenes) and hero_size and hero_size == scenes[neighbor].get("shot_language", {}).get("shot_size"):
                violations.append(f"Hero scene '{hero.get('id')}' has same shot size as neighbor. Hero moments should be visually distinct from surrounding scenes.")
    generic = sum(any(p in s.get("description", "").lower() for p in GENERIC_PHRASES) for s in scenes)
    if generic >= len(scenes) * .3:
        violations.append(f"{generic}/{len(scenes)} scenes use generic language. Replace vague descriptions with specific visual details.")
        suggestions.append("Replace generic adjectives with concrete subjects, locations, light, and texture.")
    textured = sum(bool(s.get("texture_keywords")) for s in scenes)
    if len(scenes) >= 4 and textured < len(scenes) * .3: violations.append(f"Only {textured}/{len(scenes)} scenes have texture_keywords. Add texture descriptors to visual scenes for richer generation prompts.")
    intented = sum(bool(s.get("shot_intent")) for s in scenes)
    if len(scenes) >= 4 and intented < len(scenes) * .5: violations.append(f"Only {intented}/{len(scenes)} scenes have shot_intent. Every scene should explain WHY it exists in the video.")
    score = min(5.0, len(violations) * .6)
    verdict = "strong" if score < 2 else "acceptable" if score < 3 else "revise" if score < 4 else "fail"
    return {"score": round(score, 1), "verdict": verdict, "violations": violations, "suggestions": suggestions}
