"""Declarative HonCut tool contract."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import os
import shutil
from typing import Any


class DependencyError(Exception):
    """Raised when a tool's declared dependency is unavailable."""

class ToolTier(str, Enum):
    CORE="core"; OPTIONAL="optional"; EXPERIMENTAL="experimental"
    VOICE="voice"; ENHANCE="enhance"; GENERATE="generate"; SOURCE="source"; ANALYZE="analyze"; PUBLISH="publish"
class ToolStability(str, Enum): EXPERIMENTAL="experimental"; BETA="beta"; PRODUCTION="production"
class ToolStatus(str, Enum): AVAILABLE="available"; UNAVAILABLE="unavailable"; DEGRADED="degraded"
class ToolRuntime(str, Enum): LOCAL="local"; LOCAL_GPU="local_gpu"; API="api"; HYBRID="hybrid"; BROWSER="browser"
class ExecutionMode(str, Enum): SYNC="sync"; ASYNC="async"
class Determinism(str, Enum): DETERMINISTIC="deterministic"; SEEDED="seeded"; STOCHASTIC="stochastic"; NONDETERMINISTIC="nondeterministic"
class ResumeSupport(str, Enum): NONE="none"; FROM_START="from_start"; FROM_CHECKPOINT="from_checkpoint"
@dataclass(frozen=True)
class ResourceProfile: cpu_cores: int=1; ram_mb: int=256; vram_mb: int=0; disk_mb: int=0; network_required: bool=False
@dataclass
class RetryPolicy: max_retries: int=0; backoff_seconds: float=1.0; retryable_errors: list[str]=field(default_factory=list)
@dataclass
class ToolResult:
    success: bool
    data: Any=None
    error: str|None=None
    duration_seconds: float=0.0
    metadata: dict[str, Any]=field(default_factory=dict)
    artifacts: list[str]=field(default_factory=list)
    cost_usd: float=0.0
    seed: int|None=None
    model: str|None=None

class BaseTool(ABC):
    name="unnamed"; version="0.1.0"; tier=ToolTier.CORE; runtime=ToolRuntime.LOCAL; execution_mode=ExecutionMode.SYNC; determinism=Determinism.DETERMINISTIC
    dependencies: list[str]=[]; capabilities: list[str]=[]; input_schema: dict[str, Any]={}; resource_profile=ResourceProfile()
    def check_dependencies(self) -> None:
        for dependency in self.dependencies:
            if dependency.startswith(("cmd:", "binary:")):
                command = dependency.split(":", 1)[1]
                if shutil.which(command) is None:
                    raise DependencyError(f"Command {command!r} not found")
            elif dependency.startswith("env:"):
                variable = dependency[4:]
                if not os.environ.get(variable):
                    raise DependencyError(f"Environment variable {variable!r} not set")
            elif dependency.startswith("python:"):
                module = dependency[7:]
                try:
                    __import__(module)
                except ImportError as exc:
                    raise DependencyError(f"Python module {module!r} not installed") from exc

    def get_status(self) -> ToolStatus:
        try:
            self.check_dependencies()
        except DependencyError:
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE
    def get_info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "tier": self.tier.value, "runtime": self.runtime.value, "status": self.get_status().value, "capabilities": self.capabilities, "input_schema": self.input_schema, "resource_profile": asdict(self.resource_profile)}
    def estimate_cost(self, inputs: dict[str, Any]) -> float: return 0.0
    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]: return {"tool": self.name, "status": self.get_status().value, "estimated_cost_usd": self.estimate_cost(inputs), "would_execute": True}
    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> ToolResult: raise NotImplementedError
