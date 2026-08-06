"""Unified Toonflow-style vendor/model adapter contract."""
from abc import ABC
from dataclasses import dataclass,field
from typing import Any
@dataclass(frozen=True)
class VendorModel: name:str; model_name:str; type:str; modes:tuple[str,...]=(); capabilities:dict[str,Any]=field(default_factory=dict)
class VendorAdapter(ABC):
    id="vendor"; version="1.0"; name="Vendor"
    def __init__(self,models:list[VendorModel],config:dict[str,str]|None=None): self.models=models; self.config=config or {}
    def get_model(self,model_name:str,model_type:str|None=None)->VendorModel:
        for model in self.models:
            if model.model_name==model_name and (model_type is None or model.type==model_type): return model
        raise KeyError(f"Unknown {model_type or ''} model: {model_name}")
    def validate_config(self,required:tuple[str,...]=("api_key",))->None:
        missing=[key for key in required if not self.config.get(key)]
        if missing: raise ValueError(f"Missing vendor configuration: {', '.join(missing)}")
    def request(self,model_name:str,config:dict[str,Any])->Any:
        model=self.get_model(model_name); method=getattr(self,f"{model.type}_request",None)
        if method is None: raise NotImplementedError(f"{self.id} does not implement {model.type}_request")
        return method(config,model)
def merge_models(built_in:list[VendorModel],custom:list[VendorModel])->list[VendorModel]:
    merged={m.model_name:m for m in built_in}; merged.update({m.model_name:m for m in custom}); return list(merged.values())
