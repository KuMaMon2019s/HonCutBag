"""Three-engine compose routing with immutable runtime selection."""
from dataclasses import dataclass
from typing import Any
RUNTIMES=("remotion","hyperframes","ffmpeg")
@dataclass(frozen=True)
class ComposeRoute: runtime:str; reason:str
def route_composition(composition:dict[str,Any], available:set[str]|None=None, locked_runtime:str|None=None)->ComposeRoute:
    available=available or set(RUNTIMES)
    if locked_runtime:
        if locked_runtime not in RUNTIMES: raise ValueError(f"Unknown render runtime: {locked_runtime}")
        if locked_runtime not in available: raise RuntimeError(f"Locked runtime unavailable: {locked_runtime}")
        return ComposeRoute(locked_runtime,"runtime lock")
    cuts=composition.get("cuts",[])
    desired="remotion" if any(c.get("type") in {"text_card","chart","kpi_grid"} for c in cuts) else "hyperframes" if composition.get("html_entry") else "ffmpeg"
    if desired in available: return ComposeRoute(desired,"composition features")
    fallback=next((r for r in reversed(RUNTIMES) if r in available),None)
    if not fallback: raise RuntimeError("No compose runtime available")
    return ComposeRoute(fallback,f"{desired} unavailable")
def lock_runtime(composition:dict[str,Any], **kwargs)->dict[str,Any]:
    route=route_composition(composition,**kwargs); return {**composition,"render_runtime":route.runtime,"render_runtime_reason":route.reason}
