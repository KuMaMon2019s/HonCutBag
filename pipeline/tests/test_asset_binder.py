import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from asset_binder import bind_assets,build_asset_bindings,validate_bindings
def test_assets_number_in_input_order(): assert build_asset_bindings([{"id":"b"},{"id":"a"}]) == {"b":"@图1","a":"@图2"}
def test_skipped_storyboard_gets_no_number(): assert len(build_asset_bindings([], [{"id":"S1","shouldGenerateImage":False}])) == 0
def test_names_are_replaced_and_valid():
    value=bind_assets("沈辞看城楼",[{"id":"A","name":"沈辞","type":"角色"},{"id":"B","name":"城楼","type":"场景"}])
    assert "@图1看@图2" in value and validate_bindings(value,2)
