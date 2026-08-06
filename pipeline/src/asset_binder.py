"""Toonflow-compatible @图N asset binding."""
import re
from typing import Any
def build_asset_bindings(assets:list[dict[str,Any]],shots:list[dict[str,Any]]|None=None)->dict[str,str]:
    bindings={str(a["id"]):f"@图{i}" for i,a in enumerate(assets,1)}; index=len(bindings)+1
    for shot in shots or []:
        if shot.get("shouldGenerateImage",True): bindings[f"storyboard:{shot.get('id',index)}"]=f"@图{index}"; index+=1
    return bindings
def bind_assets(prompt:str,assets:list[dict[str,Any]])->str:
    bindings=build_asset_bindings(assets); output=prompt
    for asset in sorted(assets,key=lambda a:len(str(a.get("name",""))),reverse=True): output=re.sub(re.escape(str(asset.get("name",""))),bindings[str(asset["id"])],output)
    prefix=" ".join(f"{bindings[str(a['id'])]} 为{a.get('name','')}{a.get('type','')}" for a in assets)
    return f"{prefix}, {output}" if prefix else output
def validate_bindings(prompt:str,reference_count:int)->bool: return all(1<=int(n)<=reference_count for n in re.findall(r"@图(\d+)",prompt))
