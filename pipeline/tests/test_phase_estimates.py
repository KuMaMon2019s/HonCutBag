from __future__ import annotations

from runtime.phase_estimates import (
    build_pipeline_workload,
    estimate_phase_duration,
    estimate_total,
    remaining_phase_names,
)


def _structured_inputs() -> tuple[dict, dict]:
    characters = {
        "characters": [
            {
                "id": "hero",
                "appearance": {
                    "identity_props": [{"id": "chip"}],
                    "variants": [
                        {"state_name": "wet", "description": "rain soaked"},
                        {"state_name": "torn", "description": "torn coat"},
                        {"state_name": "ignored", "description": ""},
                    ],
                },
            }
        ]
    }
    storyboard = {
        "shots": [
            {
                "id": 1,
                "character_ids": ["hero"],
                "storyboard_beats": [
                    {"beat_id": "S01_P01"},
                    {"beat_id": "S01_P02"},
                ],
            }
        ]
    }
    return characters, storyboard


def test_structured_workload_counts_provider_image_requests(tmp_path):
    characters, storyboard = _structured_inputs()

    workload = build_pipeline_workload(
        characters,
        storyboard,
        output_dir=tmp_path,
        phase5_correction_attempts=2,
    )

    assert workload.character_count == 1
    assert workload.shot_count == 1
    assert workload.storyboard_beat_count == 2
    assert workload.character_reference_image_requests == 7
    assert workload.phase2_image_requests == 0
    assert workload.phase3_image_requests == 10
    assert workload.phase4_image_requests == 1
    assert workload.phase5_max_correction_image_requests == 6


def test_image_phase_estimate_uses_effective_seedream_interval(monkeypatch):
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "120")

    assert estimate_phase_duration("phase3", image_requests=37) == 4440
    assert estimate_phase_duration("phase4", image_requests=10) == 1200


def test_total_estimate_exposes_bounded_phase5_correction_range(monkeypatch, tmp_path):
    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "120")
    characters, storyboard = _structured_inputs()
    workload = build_pipeline_workload(characters, storyboard, output_dir=tmp_path)

    estimate = estimate_total(
        phases=["phase2", "phase3", "phase4", "phase5"],
        workload=workload,
    )

    assert estimate["phases"] == {
        "phase2": 0,
        "phase3": 1200,
        "phase4": 120,
        "phase5": 10,
    }
    assert estimate["total"] == 1330
    assert estimate["upper_total"] == 2050
    assert estimate["bounded"] is True
    assert estimate["basis"] == "structured_provider_workload"


def test_remaining_phase_names_honors_cli_selection_and_checkpoints():
    assert remaining_phase_names(
        skip_phase=[1, 6, 7, 8, 9, 9.5],
        completed_phases={"phase1", "phase2"},
    ) == ["phase3", "phase4", "phase5"]
