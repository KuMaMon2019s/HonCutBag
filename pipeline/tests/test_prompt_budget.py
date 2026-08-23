"""Regression tests for provider/model prompt budgets and contract deduplication."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients.video_client import VideoClient
from clients.ark_multimodal_client import ArkMultimodalClient
from phases.pipeline_core import _prepare_phase6_prompt
from tools.task_dir_exporter import build_task_dir
from utils.prompt_budget import (
    DuplicatePromptContractError,
    PromptBudgetExceededError,
    enforce_prompt_budget,
    prompt_section_lengths,
    resolve_prompt_budget,
)


def test_provider_model_budget_logs_sections_and_warns_at_soft_limit(monkeypatch, caplog):
    monkeypatch.setenv("HONCUT_SEEDANCE_PROMPT_SOFT_CHARS", "40")
    monkeypatch.setenv("HONCUT_SEEDANCE_PROMPT_HARD_CHARS", "100")
    prompt = "开场说明。\n[honcut-video-generation-contract-v2]\n" + "x" * 45
    caplog.set_level(logging.INFO)

    report = enforce_prompt_budget(
        prompt,
        provider="seedance",
        model="doubao-seedance-2.0-mini",
        purpose="video_generation",
    )

    assert report.over_soft_limit is True
    assert report.budget.model == "doubao-seedance-2.0-mini"
    assert report.section_chars["[honcut-video-generation-contract-v2]"] > 45
    assert "prompt soft limit reached" in caplog.text


def test_prompt_budget_fails_closed_before_hard_limit_submission(monkeypatch):
    monkeypatch.setenv("HONCUT_ARK_TEXT_PROMPT_SOFT_CHARS", "20")
    monkeypatch.setenv("HONCUT_ARK_TEXT_PROMPT_HARD_CHARS", "30")

    with pytest.raises(PromptBudgetExceededError, match="hard limit exceeded"):
        enforce_prompt_budget(
            "x" * 30,
            provider="ark",
            model="doubao-seed-1-6",
            purpose="text_llm",
        )


def test_duplicate_singleton_prompt_contract_fails_closed():
    marker = "[honcut-video-generation-contract-v2]"
    with pytest.raises(DuplicatePromptContractError, match="duplicate singleton"):
        enforce_prompt_budget(
            f"{marker}\nfirst\n{marker}\nsecond",
            provider="seedance",
            model="doubao-seedance-2.0-mini",
            purpose="video_generation",
        )


def test_prompt_section_lengths_sum_to_total_and_multimodal_has_own_budget():
    prompt = "intro。\n[section-a]\naaa\n[section-b]\nbbbb"
    sections = prompt_section_lengths(prompt)
    budget = resolve_prompt_budget(
        provider="ark",
        model="doubao-seed-1-6-vision",
        purpose="multimodal_review",
    )

    assert sum(sections.values()) == len(prompt)
    assert budget.soft_chars == 30_000
    assert budget.hard_chars == 45_000


def test_exact_provider_model_budget_can_override_route_default(monkeypatch):
    prefix = (
        "HONCUT_PROMPT_SEEDANCE_DOUBAO_SEEDANCE_2_0_MINI_VIDEO_GENERATION"
    )
    monkeypatch.setenv(f"{prefix}_SOFT_CHARS", "7000")
    monkeypatch.setenv(f"{prefix}_HARD_CHARS", "9000")

    budget = resolve_prompt_budget(
        provider="seedance",
        model="doubao-seedance-2.0-mini",
        purpose="video_generation",
    )

    assert (budget.soft_chars, budget.hard_chars) == (7000, 9000)


def test_video_transport_budget_blocks_before_direct_generator(monkeypatch):
    monkeypatch.setenv("VIDEO_PROVIDER_KLING", "direct")
    monkeypatch.setenv("HONCUT_GENERIC_PROMPT_SOFT_CHARS", "20")
    monkeypatch.setenv("HONCUT_GENERIC_PROMPT_HARD_CHARS", "30")
    called = False

    def direct_generator(**_kwargs):
        nonlocal called
        called = True
        return "unexpected"

    client = VideoClient("kling", direct_generator=direct_generator)
    with pytest.raises(PromptBudgetExceededError):
        client.generate("x" * 30, model="kling-v3")

    assert called is False


def test_multimodal_budget_counts_image_labels_before_reading_media(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_ARK_MULTIMODAL_PROMPT_SOFT_CHARS", "35")
    monkeypatch.setenv("HONCUT_ARK_MULTIMODAL_PROMPT_HARD_CHARS", "40")
    client = ArkMultimodalClient(
        client=object(),
        model="doubao-seed-vision",
    )
    missing_image = tmp_path / "this_is_long_evidence_name.png"

    with pytest.raises(PromptBudgetExceededError):
        client.review([missing_image], "x" * 25)


def test_task_directory_checks_the_exact_written_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_SEEDANCE_PROMPT_SOFT_CHARS", "50")
    monkeypatch.setenv("HONCUT_SEEDANCE_PROMPT_HARD_CHARS", "60")

    with pytest.raises(PromptBudgetExceededError):
        build_task_dir(
            tmp_path,
            ["S01"],
            {
                "model": "doubao-seedance-2.0-mini",
                "shots": {"S01": {"prompt": "x" * 60, "gen_strategy": "i2v"}},
            },
        )


def test_phase6_task_and_direct_routes_share_one_final_prompt_preparation():
    shot = {
        "shot_id": "S01",
        "duration": 5,
        "where": "露天训练场",
        "time": "日间",
        "who": [],
        "visual": "空场旗帜轻摆",
        "generation_actions": ["旗帜随风轻摆"],
    }

    prompt, route_applied = _prepare_phase6_prompt(
        "S01",
        shot,
        {"characters": []},
        {"shots": {"S01": {"scene_description": "露天训练场"}}},
        video_model="seedance",
        route_model="doubao-seedance-2.0-mini",
    )

    assert route_applied is True
    base_prompt = shot["prompt"]
    assert base_prompt in prompt
    rerouted, reroute_applied = _prepare_phase6_prompt(
        "S01",
        shot,
        {"characters": []},
        {"shots": {"S01": {"scene_description": "露天训练场"}}},
        video_model="seedance",
        route_model="doubao-seedance-2.0-mini",
    )
    assert reroute_applied is True
    assert rerouted == prompt
    assert shot["prompt"] == base_prompt
    assert prompt.count("[honcut-video-generation-contract-v2]") == 1
    assert prompt == base_prompt
    assert "时空连续性硬约束" in prompt
    assert "本地场景钟点锁定为 10:00–16:00（日间）" in prompt
