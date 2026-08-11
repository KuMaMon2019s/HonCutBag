import ast
import inspect
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import pipeline_core
from quality.quality_gate import run_storyboard_review


def test_storyboard_review_accepts_none_string_fields():
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": [],
                "where": "房间",
                "dialogue": None,
                "what": None,
                "visual": None,
            }
        ]
    }

    result = run_storyboard_review(storyboard, script_text="", characters=[])

    assert isinstance(result, dict)
    assert "grade" in result


def test_storyboard_review_accepts_structured_dialogue():
    storyboard = {
        "shots": [{
            "shot_id": "S01",
            "who": ["凛"],
            "where": "庭院",
            "dialogue": {"speaker": "凛", "line": "「这是一句结构化对白」"},
            "what": "凛开口",
            "visual": "凛站在庭院中开口说话，镜头保持稳定",
        }]
    }

    result = run_storyboard_review(
        storyboard,
        script_text="这是一句结构化对白",
        characters=[{"id": "lin", "name": "凛"}],
    )

    assert isinstance(result, dict)
    assert "grade" in result


def test_phase1_screenwriter_m5_fallback_returns_result():
    tree = ast.parse(inspect.getsource(pipeline_core.run_phase1_screenwriter))

    fallback_has_return = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "phase1_result" for target in node.targets)
        and index + 1 < len(parent.body)
        and isinstance(parent.body[index + 1], ast.Return)
        and isinstance(parent.body[index + 1].value, ast.Name)
        and parent.body[index + 1].value.id == "phase1_result"
        for parent in ast.walk(tree)
        if hasattr(parent, "body") and isinstance(parent.body, list)
        for index, node in enumerate(parent.body)
    )

    assert fallback_has_return
