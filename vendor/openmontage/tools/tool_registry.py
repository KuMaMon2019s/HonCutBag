"""Tool registry with status, stability, and support-envelope reporting.

Vendored from OpenMontage's ``tools.tool_registry`` and adjusted so discovery
uses HonCut's ``vendor.openmontage.tools`` package layout.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any, Optional

from .base_tool import BaseTool, ToolStability, ToolStatus, ToolTier


_UNICODE_DASH_REPLACEMENTS = {
    "\u2014": "--", "\u2013": "-", "\u2212": "-", "\u2018": "'",
    "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
}


def _scrub_unicode_dashes(value: Any) -> Any:
    if isinstance(value, str):
        for needle, replacement in _UNICODE_DASH_REPLACEMENTS.items():
            value = value.replace(needle, replacement)
        return value
    if isinstance(value, list):
        return [_scrub_unicode_dashes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_unicode_dashes(item) for item in value)
    if isinstance(value, dict):
        return {key: _scrub_unicode_dashes(item) for key, item in value.items()}
    return value


class ToolRegistry:
    """Central registry of all vendored OpenMontage tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._discovered_packages: set[str] = set()

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")
        self._tools[tool.name] = tool

    def clear(self) -> None:
        self._tools.clear()
        self._discovered_packages.clear()

    def register_module(self, module: ModuleType) -> list[str]:
        registered = []
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is BaseTool or not issubclass(cls, BaseTool):
                continue
            if cls.__module__ != module.__name__ or inspect.isabstract(cls):
                continue
            tool = cls()
            self.register(tool)
            registered.append(tool.name)
        return registered

    @staticmethod
    def _load_dotenv() -> None:
        from pathlib import Path
        import os
        import re

        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.is_file():
            return
        with env_path.open(encoding="utf-8", errors="ignore") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value[:1] in ("'", '"'):
                    quote = value[0]
                    end = value.find(quote, 1)
                    value = value[1:end] if end != -1 else value[1:]
                else:
                    match = re.search(r"(^|\s)#", value)
                    if match:
                        value = value[:match.start()]
                    value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value

    def discover(self, package_name: str = "vendor.openmontage.tools") -> list[str]:
        self._load_dotenv()
        package = importlib.import_module(package_name)
        package_paths = getattr(package, "__path__", None)
        if package_paths is None:
            return self.register_module(package)
        discovered = []
        for module_info in pkgutil.walk_packages(package_paths, f"{package.__name__}."):
            if module_info.name.endswith((".base_tool", ".tool_registry")):
                continue
            module = importlib.import_module(module_info.name)
            discovered.extend(self.register_module(module))
        self._discovered_packages.add(package_name)
        return discovered

    def ensure_discovered(self, package_name: str = "vendor.openmontage.tools") -> None:
        if package_name not in self._discovered_packages:
            self.discover(package_name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_all(self) -> list[str]:
        return list(self._tools)

    def get_by_tier(self, tier: ToolTier) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if tool.tier == tier]

    def get_by_capability(self, capability: str) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if tool.capability == capability]

    def get_by_provider(self, provider: str) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if tool.provider == provider]

    def get_by_status(self, status: ToolStatus) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if tool.get_status() == status]

    def get_available(self) -> list[BaseTool]:
        return self.get_by_status(ToolStatus.AVAILABLE)

    def get_unavailable(self) -> list[BaseTool]:
        return self.get_by_status(ToolStatus.UNAVAILABLE)

    def get_by_stability(self, stability: ToolStability) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if tool.stability == stability]

    def find_by_capability(self, capability: str) -> list[BaseTool]:
        return [tool for tool in self._tools.values() if capability in tool.capabilities]

    def find_fallback(self, tool_name: str) -> Optional[BaseTool]:
        tool = self.get(tool_name)
        if tool is None:
            return None
        candidates = list(tool.fallback_tools or [])
        if tool.fallback and tool.fallback not in candidates:
            candidates.append(tool.fallback)
        for name in candidates:
            fallback = self.get(name)
            if fallback and fallback.get_status() == ToolStatus.AVAILABLE:
                return fallback
        return None

    def support_envelope(self) -> dict[str, Any]:
        self.ensure_discovered()
        return {name: tool.get_info() for name, tool in self._tools.items()}

    def capability_catalog(self) -> dict[str, list[dict[str, Any]]]:
        return self._grouped_catalog("capability")

    def provider_catalog(self) -> dict[str, list[dict[str, Any]]]:
        return self._grouped_catalog("provider")

    def _grouped_catalog(self, attribute: str) -> dict[str, list[dict[str, Any]]]:
        self.ensure_discovered()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for tool in self._tools.values():
            grouped.setdefault(getattr(tool, attribute), []).append(tool.get_info())
        for items in grouped.values():
            items.sort(key=lambda item: (item["provider"], item["name"]))
        return dict(sorted(grouped.items()))

    def tier_summary(self) -> dict[str, dict[str, int]]:
        summary = {}
        for tier in ToolTier:
            tools = self.get_by_tier(tier)
            if tools:
                counts = {"available": 0, "unavailable": 0, "degraded": 0}
                for tool in tools:
                    status = tool.get_status().value
                    counts[status] = counts.get(status, 0) + 1
                summary[tier.value] = counts
        return summary

    def provider_menu(self) -> dict[str, dict[str, Any]]:
        self.ensure_discovered()
        menu: dict[str, dict[str, Any]] = {}
        for tool in self._tools.values():
            if tool.provider == "selector":
                continue
            bucket = menu.setdefault(tool.capability, {
                "available": [], "unavailable": [], "total": 0, "configured": 0,
            })
            info = tool.get_info()
            status = tool.get_status()
            entry = {
                "name": tool.name, "provider": tool.provider,
                "runtime": tool.runtime.value, "best_for": tool.best_for,
                "dependencies": info.get("dependencies", []),
                "install_instructions": tool.install_instructions,
                "status": status.value,
            }
            target = "available" if status == ToolStatus.AVAILABLE else "unavailable"
            bucket[target].append(entry)
            bucket["total"] += 1
            if target == "available":
                bucket["configured"] += 1
        return dict(sorted(menu.items()))

    def provider_menu_summary(self) -> dict[str, Any]:
        menu = self.provider_menu()
        capabilities = []
        for capability, bucket in menu.items():
            available = {entry["provider"] for entry in bucket["available"]}
            unavailable = {entry["provider"] for entry in bucket["unavailable"]} - available
            capabilities.append({
                "capability": capability,
                "configured": bucket["configured"], "total": bucket["total"],
                "available_providers": sorted(available),
                "unavailable_providers": sorted(unavailable),
            })
        return _scrub_unicode_dashes({
            "composition_runtimes": {}, "capabilities": capabilities,
            "setup_offers": [], "runtime_warnings": [],
        })

    def gpu_required_tools(self) -> list[str]:
        return [tool.name for tool in self._tools.values() if tool.resource_profile.vram_mb > 0]

    def network_required_tools(self) -> list[str]:
        return [tool.name for tool in self._tools.values() if tool.resource_profile.network_required]


registry = ToolRegistry()
