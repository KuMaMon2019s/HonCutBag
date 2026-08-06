import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from quality.composition_validator import validate_composition
def test_no_cuts_is_invalid(tmp_path): assert not validate_composition({},tmp_path)["valid"]
def test_missing_asset_is_reported(tmp_path): assert "Missing asset" in validate_composition({"cuts":[{"in_seconds":0,"out_seconds":1,"source":"x.mp4"}]},tmp_path)["errors"][0]
def test_narration_overshoot_is_error(tmp_path):
    (tmp_path/"a.wav").touch(); report=validate_composition({"cuts":[{"in_seconds":0,"out_seconds":2}],"audio":{"narration":{"src":"a.wav"}}},tmp_path,lambda _:4)
    assert not report["valid"] and "exceeds video" in report["errors"][0]
