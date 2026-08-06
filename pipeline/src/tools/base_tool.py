"""Declarative tool contract adapted from OpenMontage."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

class ToolTier(str, Enum): CORE="core"; OPTIONAL="optional"; EXPERIMENTAL="experimental"
class ToolStatus(str, Enum): AVAILABLE="available"; UNAVAILABLE="unavailable"; DEGRADED="degraded"
class ToolRuntime(str, Enum): LOCAL="local"; API="api"; BROWSER="browser"
class ExecutionMode(str, Enum): SYNC="sync"; ASYNC="async"
class Determinism(str, Enum): DETERMINISTIC="deterministic"; SEEDED="seeded"; NONDETERMINISTIC="nondeterministic"
@dataclass(frozen=True)
class ResourceProfile: cpu_cores: int=1; ram_mb: int=256; vram_mb: int=0; disk_mb: int=0; network_required: bool=False
@dataclass
class ToolResult: success: bool; data: Any=None; error: str|None=None; duration_seconds: float=0.0; metadata: dict[str, Any]=field(default_factory=dict)

class BaseTool(ABC):
    name="unnamed"; version="0.1.0"; tier=ToolTier.CORE; runtime=ToolRuntime.LOCAL; execution_mode=ExecutionMode.SYNC; determinism=Determinism.DETERMINISTIC
    dependencies: list[str]=[]; capabilities: list[str]=[]; input_schema: dict[str, Any]={}; resource_profile=ResourceProfile()
    def get_status(self) -> ToolStatus: return ToolStatus.AVAILABLE
    def get_info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "tier": self.tier.value, "runtime": self.runtime.value, "status": self.get_status().value, "capabilities": self.capabilities, "input_schema": self.input_schema, "resource_profile": asdict(self.resource_profile)}
    def estimate_cost(self, inputs: dict[str, Any]) -> float: return 0.0
    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]: return {"tool": self.name, "status": self.get_status().value, "estimated_cost_usd": self.estimate_cost(inputs), "would_execute": True}
    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> ToolResult: raise NotImplementedError
