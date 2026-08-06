import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from prompt.three_part_prompt import build_three_part_prompt,validate_three_part_prompt
def test_sections_are_in_required_order(): assert validate_three_part_prompt(build_three_part_prompt("人物奔跑","侧逆光","写实")) == []
def test_quality_and_prohibitions_are_appended():
    value=build_three_part_prompt("画面","光","风格"); assert "高清画质" in value and "水印" in value
def test_visual_is_required():
    with pytest.raises(ValueError): build_three_part_prompt("","light","style")
