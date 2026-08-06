import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from tools.vendor_adapter import VendorAdapter,VendorModel,merge_models
class Demo(VendorAdapter):
    def text_request(self,config,model): return model.model_name+":"+config["prompt"]
def test_routes_request_by_model_type(): assert Demo([VendorModel("GPT","gpt","text")]).request("gpt",{"prompt":"hi"}) == "gpt:hi"
def test_missing_config_is_actionable():
    with pytest.raises(ValueError,match="api_key"): Demo([]).validate_config()
def test_custom_model_overrides_builtin():
    merged=merge_models([VendorModel("old","m","text")],[VendorModel("new","m","text")]); assert merged[0].name == "new"
