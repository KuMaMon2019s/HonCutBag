import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from quality.slideshow_risk import score_slideshow_risk


def test_empty_plan_fails():
    report = score_slideshow_risk([])
    assert report["verdict"] == "fail" and report["average"] == 5.0


def test_all_six_dimensions_are_reported():
    report = score_slideshow_risk([{"shot_intent": "reveal", "shot_language": {"camera_movement": "dolly"}}])
    assert len(report["dimensions"]) == 6
    assert "unsupported_cinematic_claims" in report["dimensions"]


def test_cinematic_claim_without_structure_is_penalized():
    scenes = [{"type": "text_card", "description": "same", "shot_language": {"shot_size": "wide"}}] * 4
    report = score_slideshow_risk(scenes, renderer_family="cinematic", render_runtime="ffmpeg")
    assert report["dimensions"]["unsupported_cinematic_claims"]["score"] > 0
    assert report["render_runtime"] == "ffmpeg"
