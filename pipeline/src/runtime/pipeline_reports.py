"""Atomic pipeline report serialization."""

import json
from pathlib import Path


def write_pipeline_report(report: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    report_path = output_dir / "pipeline_report.json"
    clean = json.loads(json.dumps(report, default=str))
    if clean.get("status") == "completed":
        clean.pop("error", None)
    report_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))
    print(f"\n  📄 报告已写入: {report_path}")


_write_report = write_pipeline_report


__all__ = ["_write_report", "write_pipeline_report"]
