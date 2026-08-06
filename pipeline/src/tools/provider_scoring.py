"""HonCut seven-dimension provider ranking."""
from dataclasses import asdict, dataclass
from typing import Any
@dataclass
class ProviderScore:
    tool_name:str; provider:str; task_fit:float=0; output_quality:float=0; control:float=0; reliability:float=0; cost_efficiency:float=0; latency:float=0; continuity:float=0
    @property
    def weighted_score(self): return self.task_fit*.30+self.output_quality*.20+self.control*.15+self.reliability*.15+self.cost_efficiency*.10+self.latency*.05+self.continuity*.05
    def to_dict(self): value=asdict(self); value["weighted_score"]=self.weighted_score; return value
    def explain(self): return f"{self.tool_name} ({self.provider}): {self.weighted_score:.2f}"
def score_provider(provider: dict[str, Any], context: dict[str, Any]) -> ProviderScore:
    supported=set(provider.get("capabilities", [])); needed=set(context.get("capabilities", []))
    fit=len(supported&needed)/len(needed) if needed else .5
    cost=float(provider.get("cost",0)); budget=context.get("budget")
    efficiency=1.0 if cost<=0 else max(0.0, 1-cost/budget) if budget else max(.1,1-cost)
    return ProviderScore(provider.get("name","unknown"), provider.get("provider","unknown"), fit, float(provider.get("quality",.5)), float(provider.get("control",.5)), float(provider.get("reliability",.5)), efficiency, float(provider.get("latency_score",.5)), 1.0 if provider.get("name")==context.get("locked_provider") else .5)
def rank_providers(providers:list[dict[str,Any]], context:dict[str,Any])->list[ProviderScore]: return sorted((score_provider(p,context) for p in providers), key=lambda p:p.weighted_score, reverse=True)
