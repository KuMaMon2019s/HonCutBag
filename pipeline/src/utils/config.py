#!/usr/bin/env python3
"""
HonCut 配置管理 - 集中管理所有 API keys、tokens 和工具常量

使用方式：
    from utils.config import get_api_key, ARK_BASE_URL
    
    api_key = get_api_key("ARK_AGENT")
    base_url = ARK_BASE_URL
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from dotenv import dotenv_values, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BRIDGE_API_URL = "http://127.0.0.1:9100"
VIDEO_ROUTE_VALUES = {"bridge", "direct", "local"}
_PROVIDER_DEFAULT_VIDEO_ROUTES = {
    "BRIDGE": "bridge",
    "LOCAL": "local",
    "WAN": "bridge",
    "WAN22": "bridge",
}


class PathConfig(BaseSettings):
    """Filesystem roots, overridable with ``HONCUT_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="HONCUT_")

    projects_dir: Path = Field(default=Path.home() / "projects")
    repo_root: Path = Field(default=Path.home() / "projects" / "honcut")


class ExternalAPIEndpoints:
    """Default provider endpoints and their environment-variable overrides."""

    DEFAULTS: ClassVar[dict[str, str]] = {
        "DASHSCOPE_TTS": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        "DOUBAO_TTS": "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional",
        "FREESOUND": "https://freesound.org/apiv2",
        "GOOGLE_TTS": "https://texttospeech.googleapis.com",
        "PIXABAY_MUSIC": "https://pixabay.com",
        "SUNO": "https://api.sunoapi.org/api/v1",
        "GOOGLE_IMAGEN_VERTEX": "https://{location}-aiplatform.googleapis.com/v1",
        "GOOGLE_IMAGEN_STUDIO": "https://generativelanguage.googleapis.com/v1beta",
        "FAL_FLUX": "https://fal.run/fal-ai/flux/dev",
        "PEXELS_IMAGE": "https://api.pexels.com/v1/search",
        "PIXABAY_IMAGE": "https://pixabay.com/api/",
        "KLING": "https://api-singapore.klingai.com",
    }
    ENV_VARS: ClassVar[dict[str, str]] = {"KLING": "KLING_API_BASE_URL"}

    @classmethod
    def get(cls, service: str) -> str:
        """Return a configured endpoint, falling back to the provider default."""
        try:
            default = cls.DEFAULTS[service]
        except KeyError as exc:
            raise ValueError(f"Unknown external API endpoint: {service}") from exc
        env_var = cls.ENV_VARS.get(service, f"{service}_API_URL")
        return os.environ.get(env_var, default).rstrip("/")


def get_external_api_url(service: str) -> str:
    """Return an external provider endpoint from environment-backed config."""
    return ExternalAPIEndpoints.get(service)


class VideoModel(str, Enum):
    """Video model routes supported by the Bridge v3.2 contract."""

    WAN22 = "wan22"
    PHANTOM = "phantom"
    FLF2V = "flf2v"
    SEEDANCE = "seedance"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


def configure_ark_agent_environment(env_file: Path = ENV_FILE) -> str:
    """Pin HonCut to the project Agent Plan key and discard the Coding key.

    A long-lived launcher can retain an obsolete ``ARK_AGENT_API_KEY`` even
    after the project ``.env`` changes.  The project value therefore wins for
    this one credential.  Deployments without a project value may still supply
    ``ARK_AGENT_API_KEY`` directly.  ``ARK_API_KEY`` is never a HonCut runtime
    fallback.

    The return value is a safe source label and never contains credential data.
    """
    project_values = dotenv_values(env_file) if env_file.is_file() else {}
    project_agent_key = project_values.get("ARK_AGENT_API_KEY")
    if isinstance(project_agent_key, str) and project_agent_key.strip():
        os.environ["ARK_AGENT_API_KEY"] = project_agent_key.strip()
        source = "project_env"
    elif os.environ.get("ARK_AGENT_API_KEY"):
        source = "process_env"
    else:
        source = "missing"
    os.environ.pop("ARK_API_KEY", None)
    return source


# Load before pipeline_runner starts child processes so they inherit the same
# repository-level configuration.  Other explicitly exported variables retain
# priority; the Ark credential follows the narrower policy above.
load_dotenv(ENV_FILE, override=False)
ARK_AGENT_CREDENTIAL_SOURCE = configure_ark_agent_environment()

# 火山系服务（ARK/TOS/TTS/ASR）是国内服务，走本地代理会多一跳且不稳：
# 2026-08-09 R5/R7 实测 http_proxy=127.0.0.1:7897 时事件提取 8/19 段超时、
# adaptation_engine 240s ReadTimeout（traceback 里 http_proxy.py 实锤）。
# 追加 NO_PROXY 保证直连；已有值则合并，不覆盖用户显式配置。
_NO_PROXY_VOLC_DOMAINS = ".volces.com,.bytedance.com"
for _np_key in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_np_key, "")
    if _existing:
        if "volces.com" not in _existing:
            os.environ[_np_key] = _existing + "," + _NO_PROXY_VOLC_DOMAINS
    else:
        os.environ[_np_key] = _NO_PROXY_VOLC_DOMAINS.lstrip(".")


def get_video_route(provider: str) -> str:
    """Return the configured video route for *provider*.

    Provider-specific configuration wins over the global setting.  ``direct``
    is the production default so an unconfigured run uses the online provider;
    Bridge and local execution remain explicit opt-in routes.
    """
    provider_name = provider.strip().upper()
    if not provider_name:
        raise ValueError("video provider must not be empty")
    route = os.environ.get(f"VIDEO_PROVIDER_{provider_name}") or os.environ.get(
        "VIDEO_GENERATION_MODE"
    )
    if route is None:
        route = _PROVIDER_DEFAULT_VIDEO_ROUTES.get(provider_name, "direct")
    route = route.strip().lower()
    if route not in VIDEO_ROUTE_VALUES:
        raise ValueError(
            f"Invalid video route {route!r} for {provider}; expected bridge, direct, or local"
        )
    return route


def get_bridge_api_url() -> str:
    """Return the Bridge base URL without a trailing slash."""
    return os.environ.get("BRIDGE_API_URL", DEFAULT_BRIDGE_API_URL).rstrip("/")

# ═══════════════════════════════════════════════════════════════════════════════
# API Keys 配置
# ═══════════════════════════════════════════════════════════════════════════════

class APIKeys:
    """API Key 管理器"""
    
    # 火山方舟 (Volcano Ark)
    ARK_AGENT: str = "ARK_AGENT_API_KEY"  # Agent Plan (主要)
    # External-tool compatibility only; HonCut strips this variable at runtime.
    ARK_CODING: str = "ARK_API_KEY"
    
    # OpenAI 兼容
    OPENAI: str = "OPENAI_API_KEY"        # 备选
    
    # 音频服务
    ELEVENLABS: str = "ELEVENLABS_API_KEY"
    DOUBAO_SPEECH: str = "DOUBAO_SPEECH_API_KEY"
    
    # 图像/视频生成
    FAL: str = "FAL_KEY"                  # fal.ai (OM 工具用)
    FAL_AI: str = "FAL_AI_API_KEY"        # 备选
    
    @classmethod
    def get(cls, key_name: str, fallback: Optional[str] = None) -> Optional[str]:
        """获取 API key
        
        Args:
            key_name: API key 名称 (如 "ARK_AGENT" 或 "ARK_AGENT_API_KEY")
            fallback: 备选 key 名称 (如 "OPENAI")
            
        Returns:
            API key 值，如果不存在返回 None
        """
        # 1. 尝试作为类属性名查找 (如 "ARK_AGENT" -> "ARK_AGENT_API_KEY")
        env_var = getattr(cls, key_name, None)
        if env_var and isinstance(env_var, str):
            value = os.environ.get(env_var)
            if value:
                return value
        
        # 2. 尝试直接作为环境变量名查找 (如 "ARK_AGENT_API_KEY")
        value = os.environ.get(key_name)
        if value:
            return value
        
        # 3. 尝试备选
        if fallback:
            return cls.get(fallback)
        
        return None
    
    @classmethod
    def get_or_raise(cls, key_name: str, fallback: Optional[str] = None) -> str:
        """获取 API key，如果不存在抛出异常
        
        Args:
            key_name: API key 名称
            fallback: 备选 key 名称
            
        Returns:
            API key 值
            
        Raises:
            ValueError: 如果 API key 不存在
        """
        value = cls.get(key_name, fallback)
        if not value:
            env_var = getattr(cls, key_name, key_name)
            raise ValueError(f"缺少必需的 API key: {env_var}")
        return value


# ═══════════════════════════════════════════════════════════════════════════════
# Base URLs 配置
# ═══════════════════════════════════════════════════════════════════════════════

class BaseURLs:
    """API Base URL 配置"""
    
    # 火山方舟
    ARK_AGENT_PLAN: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    ARK_CODING_PLAN: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    ARK_PAY_AS_YOU_GO: str = "https://ark.cn-beijing.volces.com/api/v3"  # 不使用
    
    # OpenAI
    OPENAI: str = "https://api.openai.com/v1"
    
    # fal.ai (OM 工具)
    FAL: str = "https://fal.ai/api/v1"


# ═══════════════════════════════════════════════════════════════════════════════
# 模型配置
# ═══════════════════════════════════════════════════════════════════════════════

class Models:
    """AI 模型配置"""
    
    # 火山方舟 - 文本生成
    ARK_TEXT_LITE: str = "doubao-seed-evolving"      # 标准文本模型（1M 上下文）
    ARK_TEXT_PRO: str = "doubao-seed-2.0-pro"        # 进阶级
    ARK_TEXT_TURBO: str = "doubao-seed-evolving"     # 兼容旧快速路由名

    # 火山方舟 - 多模态理解。不要复用文本默认模型；Phase 5/8 会传图片。
    ARK_MULTIMODAL: str = "doubao-seed-2.0-lite"
    
    # 火山方舟 - 图像生成
    ARK_IMAGE: str = "doubao-seedream-5.0-lite"
    
    # 火山方舟 - 视频生成
    ARK_VIDEO: str = "doubao-seedance-2.0-mini"
    
    # 火山方舟 - 语音合成 (TTS 2.0)
    ARK_TTS: str = "seed-tts-2.0"
    
    # 火山方舟 - 语音识别 (ASR 2.0)
    ARK_ASR: str = "volc.seedasr.sauc.duration"
    
    # 火山方舟 - Embedding
    ARK_EMBEDDING: str = "doubao-embedding-vision",
    
    # OpenAI
    OPENAI_TEXT: str = "gpt-4"
    OPENAI_IMAGE: str = "dall-e-3"


# ═══════════════════════════════════════════════════════════════════════════════
# 工具路径配置
# ═══════════════════════════════════════════════════════════════════════════════

class ToolPaths:
    """外部工具路径配置"""
    
    # 第三方兼容工具目录
    OM_TOOLS_DIR: Path = Path(__file__).resolve().parents[3] / "vendor" / "video_tools"
    
    # HonCut 项目目录
    HONCUT_DIR: Path = Path(__file__).resolve().parents[2]
    
    # 提示词模板目录
    PROMPTS_DIR: Path = HONCUT_DIR / "prompts"
    
    # 输出目录
    OUTPUT_DIR: Path = HONCUT_DIR / "output"
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════

def get_api_key(service: str, fallback: Optional[str] = None) -> Optional[str]:
    """获取 API key 的便捷函数
    
    Args:
        service: 服务名称 (如 "ARK_AGENT", "OPENAI")
        fallback: 备选服务名称
        
    Returns:
        API key 值
    """
    load_dotenv(ENV_FILE, override=False)
    os.environ.pop("ARK_API_KEY", None)
    return APIKeys.get(service, fallback)


def get_api_key_or_raise(service: str, fallback: Optional[str] = None) -> str:
    """获取 API key，不存在则抛出异常
    
    Args:
        service: 服务名称
        fallback: 备选服务名称
        
    Returns:
        API key 值
        
    Raises:
        ValueError: 如果 API key 不存在
    """
    return APIKeys.get_or_raise(service, fallback)


def get_base_url(service: str) -> str:
    """获取 Base URL
    
    Args:
        service: 服务名称 (如 "ARK_AGENT_PLAN", "OPENAI")
        
    Returns:
        Base URL
    """
    return getattr(BaseURLs, service, "")


def get_model(service: str) -> str:
    """获取模型名称
    
    Args:
        service: 服务名称 (如 "ARK_TEXT_LITE", "OPENAI_TEXT")
        
    Returns:
        模型名称
    """
    return getattr(Models, service, "")


# ═══════════════════════════════════════════════════════════════════════════════
# 配置验证
# ═══════════════════════════════════════════════════════════════════════════════

def validate_config(required_keys: Optional[list] = None) -> Dict[str, Any]:
    """验证配置是否完整
    
    Args:
        required_keys: 必需的 API key 列表 (如 ["ARK_AGENT", "ARK_IMAGE"])
        
    Returns:
        验证结果字典
    """
    if required_keys is None:
        required_keys = ["ARK_AGENT"]
    
    results = {
        "valid": True,
        "missing": [],
        "available": []
    }
    
    for key_name in required_keys:
        value = APIKeys.get(key_name)
        if value:
            results["available"].append(key_name)
        else:
            results["missing"].append(key_name)
            results["valid"] = False
    
    return results


def print_config():
    """打印当前配置状态（用于调试）"""
    print("=" * 60)
    print("HonCut 配置状态")
    print("=" * 60)
    
    # API Keys
    print("\nAPI Keys:")
    for attr in dir(APIKeys):
        if not attr.startswith("_") and attr.isupper():
            env_var = getattr(APIKeys, attr)
            value = os.environ.get(env_var)
            status = "✓" if value else "✗"
            print(f"  {status} {attr}: {env_var}")
    
    # Base URLs
    print("\nBase URLs:")
    for attr in dir(BaseURLs):
        if not attr.startswith("_") and attr.isupper():
            url = getattr(BaseURLs, attr)
            print(f"  • {attr}: {url}")
    
    # Models
    print("\nModels:")
    for attr in dir(Models):
        if not attr.startswith("_") and attr.isupper():
            model = getattr(Models, attr)
            print(f"  • {attr}: {model}")
    
    # Tool Paths
    print("\nTool Paths:")
    for attr in dir(ToolPaths):
        if not attr.startswith("_") and attr.isupper():
            path = getattr(ToolPaths, attr)
            exists = "✓" if Path(str(path)).exists() else "✗"
            print(f"  {exists} {attr}: {path}")
    
    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 向后兼容的常量（保持与现有代码兼容）
# ═══════════════════════════════════════════════════════════════════════════════

# API Keys
ARK_AGENT_API_KEY_ENV = APIKeys.ARK_AGENT
ARK_API_KEY_ENV = APIKeys.ARK_CODING
OPENAI_API_KEY_ENV = APIKeys.OPENAI

# Base URLs
ARK_BASE_URL = BaseURLs.ARK_AGENT_PLAN
ARK_CODING_BASE_URL = BaseURLs.ARK_CODING_PLAN

# Models
DEFAULT_TEXT_MODEL = Models.ARK_TEXT_LITE
DEFAULT_MULTIMODAL_MODEL = Models.ARK_MULTIMODAL
DEFAULT_IMAGE_MODEL = Models.ARK_IMAGE
DEFAULT_VIDEO_MODEL = Models.ARK_VIDEO

# ═══════════════════════════════════════════════════════════════════════════════
# 本地视频 API 配置（HonCutBag ComfyUI 后端）
# ═══════════════════════════════════════════════════════════════════════════════

# 本地视频生成 API URL（Windows ComfyUI 机器）
LOCAL_VIDEO_API_URL = os.environ.get("LOCAL_VIDEO_API_URL", DEFAULT_BRIDGE_API_URL)

# 是否优先使用本地视频 API（True = 优先本地，False = 仅用 ARK）
USE_LOCAL_VIDEO_API = os.environ.get("USE_LOCAL_VIDEO_API", "true").lower() in ("true", "1", "yes")

# Phase 9 audio-material layer. Values remain environment-overridable so CI and
# local projects can use isolated music libraries and provider credentials.
AUDIO_CONFIG = {
    "music_dir": os.environ.get("HONCUT_MUSIC_DIR", "~/.honcut/music/"),
    "tts_api_key": os.environ.get("ARK_AGENT_API_KEY"),
    "ducking_threshold": float(os.environ.get("DUCKING_THRESHOLD", "-20.0")),
    "ducking_ratio": float(os.environ.get("DUCKING_RATIO", "4.0")),
}

# Seedance 模型 ID（Agent Plan 支持的模型）
# 可选: doubao-seedance-2.0, doubao-seedance-2.0-fast, doubao-seedance-2.0-mini
SEEDANCE_MODEL = os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2.0-mini")

# Video provider comparison matrix (from ai-video-gen skill)
VIDEO_PROVIDER_MATRIX = {
    "seedance": {
        "best_for": ["cinematic", "dialogue", "lip-sync", "multi-shot"],
        "cost_tier": "high",
        "audio_sync": True,
        "lip_sync": True,
        "max_duration_s": 20,
        "notes": "Preferred default for cinematic work. Native audio sync.",
    },
    "veo": {
        "best_for": ["landscape", "photoreal"],
        "cost_tier": "medium",
        "audio_sync": False,
        "lip_sync": False,
        "max_duration_s": 8,
        "notes": "Good for landscape-only scenes. No dialogue support.",
    },
    "kling": {
        "best_for": ["anime", "stylized"],
        "cost_tier": "medium",
        "audio_sync": False,
        "lip_sync": False,
        "max_duration_s": 10,
        "notes": "Best for anime/stylized content.",
    },
    "sora": {
        "best_for": ["creative", "abstract"],
        "cost_tier": "high",
        "audio_sync": False,
        "lip_sync": False,
        "max_duration_s": 20,
        "notes": "Creative/abstract content. Slow generation.",
    },
}

# 视频生成并发数（1=串行，>1=ThreadPoolExecutor 并发）
# 本地 Bridge 单 GPU 串行即可；切在线模型时可调大
VIDEO_GEN_CONCURRENCY = int(os.environ.get("VIDEO_GEN_CONCURRENCY", "1"))

# Paths
OM_TOOLS_DIR = ToolPaths.OM_TOOLS_DIR


if __name__ == "__main__":
    # 运行配置验证
    print_config()
    
    # 验证必需的配置
    result = validate_config(["ARK_AGENT"])
    if result["valid"]:
        print("\n✓ 配置验证通过")
    else:
        print(f"\n✗ 缺少必需的 API key: {result['missing']}")
