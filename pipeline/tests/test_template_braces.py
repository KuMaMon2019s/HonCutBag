import ast
import inspect
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases import adaptation_engine
from phases.adaptation_engine import USER_PROMPT_TEMPLATE


def test_user_prompt_template_formats_literal_dialogue_json():
    prompt = USER_PROMPT_TEMPLATE.format(
        target_duration=60,
        shot_duration=12,
        max_shots=5,
        events_json="[]",
        characters_summary="test",
    )

    assert '"speaker"' in prompt


def test_all_formatted_uppercase_templates_accept_minimal_parameters():
    source = inspect.getsource(adaptation_engine)
    tree = ast.parse(source)
    format_parameters = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "format":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue

        constant_name = node.func.value.id
        if not constant_name.isupper():
            continue
        format_parameters.setdefault(constant_name, set()).update(
            keyword.arg for keyword in node.keywords if keyword.arg is not None
        )

    uppercase_constants = {
        name: value
        for name, value in vars(adaptation_engine).items()
        if name.isupper()
    }
    for constant_name, parameter_names in format_parameters.items():
        template = uppercase_constants[constant_name]
        template.format(**dict.fromkeys(parameter_names, ""))
