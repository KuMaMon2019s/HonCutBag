import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from prompt_sanitizer import find_quality_downgrades,sanitize_quality_prompt
def test_replaces_grain_safely(): assert sanitize_quality_prompt("film grain portrait") == "subtle cinematic texture portrait"
def test_deletes_focus_downgrades(): assert "失焦" not in sanitize_quality_prompt("人物失焦，柔焦")
def test_detector_reports_before_sanitize(): assert set(find_quality_downgrades("film grain and blurry background")) == {"film grain","blurry background"}
