"""Pre-render composition checks for HonCut."""
from pathlib import Path
from typing import Any,Callable
def validate_composition(composition:dict[str,Any],assets_root:str|Path,probe_duration:Callable[[Path],float|None]|None=None)->dict[str,Any]:
    root=Path(assets_root); errors=[]; warnings=[]; info=[]; cuts=composition.get("cuts",[])
    if not cuts: return {"valid":False,"errors":["No cuts defined in composition"],"warnings":warnings,"info":info}
    duration=max(float(c.get("out_seconds",0)) for c in cuts); info.append(f"Video duration: {duration}s ({len(cuts)} cuts)")
    for i,cut in enumerate(sorted(cuts,key=lambda c:c.get("in_seconds",0))):
        if float(cut.get("out_seconds",0))<=float(cut.get("in_seconds",0)): errors.append(f"Cut '{cut.get('id',i)}' has invalid duration")
        for field in ("source","backgroundImage"):
            if cut.get(field) and not (root/cut[field]).exists(): errors.append(f"Missing asset: {cut[field]}")
    audio=composition.get("audio",{})
    for kind in ("narration","music"):
        source=audio.get(kind,{}).get("src","")
        if source and not (root/source).exists(): errors.append(f"Missing {kind} audio: {source}")
        elif source and probe_duration:
            measured=probe_duration(root/source)
            if measured is not None and kind=="narration" and measured-duration>1: errors.append(f"Narration ({measured:.1f}s) exceeds video ({duration}s)")
            if measured is not None and kind=="music" and measured<duration: warnings.append(f"Music ({measured:.1f}s) is shorter than video ({duration}s)")
    if not any(audio.get(k,{}).get("src") for k in ("narration","music")): warnings.append("No audio configured (no narration or music)")
    return {"valid":not errors,"errors":errors,"warnings":warnings,"info":info,"error_count":len(errors),"warning_count":len(warnings)}
