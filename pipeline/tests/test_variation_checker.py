import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from variation_checker import check_scene_variation

def test_empty_plan_fails():
    assert check_scene_variation([])["verdict"] == "fail"

def test_repeated_static_plan_triggers_multiple_checks():
    scenes = [{"description": "a beautiful modern place", "shot_language": {"shot_size": "wide", "camera_movement": "static"}} for _ in range(4)]
    report = check_scene_variation(scenes)
    assert len(report["violations"]) >= 6
    assert report["score"] > 0

def test_varied_complete_plan_is_strong():
    scenes = [{"id": str(i), "description": f"specific {i}", "shot_intent": "reveal", "texture_keywords": ["grain"], "hero_moment": i == 2, "shot_language": {"shot_size": size, "camera_movement": "dolly", "lighting_key": f"light{i}"}} for i, size in enumerate(("wide", "medium", "close", "extreme"))]
    assert check_scene_variation(scenes)["verdict"] == "strong"
