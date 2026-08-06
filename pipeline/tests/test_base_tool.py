import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]/"src"))
from base_tool import BaseTool, ToolResult, ToolTier
class Echo(BaseTool):
    name="echo"; capabilities=["echo"]
    def execute(self, inputs): return ToolResult(True, inputs)
def test_contract_metadata(): assert Echo().get_info()["tier"] == ToolTier.CORE.value
def test_result_contract(): assert Echo().execute({"x": 1}).data == {"x": 1}
def test_dry_run_is_declarative(): assert Echo().dry_run({})["would_execute"] is True
