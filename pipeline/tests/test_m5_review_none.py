import ast
import inspect
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import pipeline_core
from phases.phase1 import phase1_screenwriter
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


def test_storyboard_review_accepts_qualified_declared_alias():
    result = run_storyboard_review(
        {
            "shots": [{
                "shot_id": "S01",
                "who": ["身穿深灰色战术服的Agent", "敌方保安"],
                "where": "旋转走廊",
                "what": "双方开始搏斗",
                "visual": "双方在旋转走廊内抓住扶手进行清晰的近身搏斗",
            }],
        },
        script_text="双方开始搏斗",
        characters=[
            {"id": "agent", "name": "特工", "aliases": ["Agent"]},
            {"id": "security_guard", "name": "敌方保安", "aliases": ["保安"]},
        ],
    )

    assert result["severe"] == 0
    assert not any("[R1]" in issue for issue in result["issues"])


def test_storyboard_review_rejects_partial_latin_alias_collision():
    result = run_storyboard_review(
        {
            "shots": [{
                "shot_id": "S01",
                "who": ["Agent007"],
                "where": "机库",
                "what": "陌生人进入机库",
                "visual": "陌生人从机库入口走到中央控制台前",
            }],
        },
        script_text="陌生人进入机库",
        characters=[{"id": "agent", "name": "特工", "aliases": ["Agent"]}],
    )

    assert result["severe"] == 1
    assert any("[R1]" in issue for issue in result["issues"])


def test_phase1_screenwriter_m5_fallback_returns_result():
    tree = ast.parse(inspect.getsource(phase1_screenwriter.run_phase1_screenwriter))

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
