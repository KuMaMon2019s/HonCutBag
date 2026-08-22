"""Independent storyboard supervision at the Phase 5/6 boundary."""

from pathlib import Path


def run_storyboard_supervision(storyboard: dict, output_dir: Path) -> dict:
    from quality.supervision_agent import run_supervision
    from utils.pipeline_config import load_config

    config = load_config()
    style_path = output_dir / "visual-style.md"
    visual_style = (
        style_path.read_text(encoding="utf-8")
        if style_path.is_file()
        else str(storyboard.get("style", ""))
    )
    return run_supervision(storyboard, visual_style, output_dir, config)


_run_storyboard_supervision = run_storyboard_supervision


__all__ = ["_run_storyboard_supervision", "run_storyboard_supervision"]
