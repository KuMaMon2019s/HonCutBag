import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from provider_scoring import ProviderScore, rank_providers, score_provider
def test_seven_weights_sum_to_one(): assert ProviderScore("x","y",1,1,1,1,1,1,1).weighted_score == 1
def test_capability_fit_rewards_match(): assert score_provider({"name":"x","capabilities":["video"]},{"capabilities":["video"]}).task_fit == 1
def test_ranking_is_descending():
    ranked=rank_providers([{"name":"low","quality":0},{"name":"high","quality":1}],{})
    assert ranked[0].tool_name == "high"
