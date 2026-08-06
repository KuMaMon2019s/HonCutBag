import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from shot_prompt_builder import build_batch_prompts, build_shot_prompt
def test_five_layers_are_ordered():
    p=build_shot_prompt({"description":"hero","texture_keywords":["grain"],"shot_language":{"lens_mm":35,"depth_of_field":"shallow","shot_size":"wide","camera_movement":"dolly_in","lighting_key":"golden_hour"}},{"mood":"epic"})
    assert p.index("35mm") < p.index("wide shot") < p.index("hero") < p.index("golden") < p.index("Style")
def test_unknown_enum_is_preserved(): assert "custom_move" in build_shot_prompt({"description":"x","shot_language":{"camera_movement":"custom_move"}})
def test_batch_skips_transitions(): assert len(build_batch_prompts([{"type":"transition"},{"id":"S1","description":"x"}])) == 1
