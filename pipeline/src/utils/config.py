#!/usr/bin/env python3
"""
HonCut 配置管理 - 集中管理所有 API keys、tokens 和工具常量

使用方式：
    from utils.config import get_api_key, ARK_BASE_URL
    
    api_key = get_api_key("ARK_AGENT")
    base_url = ARK_BASE_URL
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def _load_env_file():
    """从项目根目录加载 .env 文件（手动解析，不依赖 python-dotenv）
    
    搜索策略（按优先级）：
    1. 当前工作目录 (cwd)
    2. config.py 所在包的父目录（pipeline/）
    3. 再上一级（项目根目录）
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",      # pipeline/
        Path(__file__).parent.parent.parent / ".env", # project root
    ]
    env_file = None
    for candidate in candidates:
        if candidate.exists():
            env_file = candidate
            break
    if env_file is None:
        return
    
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                # 解析 KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # 移除引号
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    # 只设置未存在的环境变量
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


# 启动时自动加载 .env
_load_env_file()

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    # 从项目根目录加载 .env
    _project_root = Path(__file__).parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    # python-dotenv 未安装，跳过
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# API Keys 配置
# ═══════════════════════════════════════════════════════════════════════════════

class APIKeys:
    """API Key 管理器"""
    
    # 火山方舟 (Volcano Ark)
    ARK_AGENT: str = "ARK_AGENT_API_KEY"  # Agent Plan (主要)
    ARK_CODING: str = "ARK_API_KEY"       # Coding Plan (Hermes 用)
    
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
    ARK_TEXT_LITE: str = "doubao-seed-2.0-lite"      # 标准级
    ARK_TEXT_PRO: str = "doubao-seed-2.0-pro"        # 进阶级
    ARK_TEXT_TURBO: str = "doubao-seed-2.1-turbo"    # 快速
    
    # 火山方舟 - 图像生成
    ARK_IMAGE: str = "doubao-seedream-5.0-lite"
    
    # 火山方舟 - 视频生成
    ARK_VIDEO: str = "doubao-seedance-2.0-mini"
    
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
    OM_TOOLS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "vendor" / "openmontage"
    
    # HonCut 项目目录
    HONCUT_DIR: Path = Path(__file__).parent.parent
    
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
DEFAULT_IMAGE_MODEL = Models.ARK_IMAGE
DEFAULT_VIDEO_MODEL = Models.ARK_VIDEO

# ═══════════════════════════════════════════════════════════════════════════════
# 本地视频 API 配置（HonCutBag ComfyUI 后端）
# ═══════════════════════════════════════════════════════════════════════════════

# 本地视频生成 API URL（Windows ComfyUI 机器）
LOCAL_VIDEO_API_URL = os.environ.get("LOCAL_VIDEO_API_URL", "http://192.168.31.221:9100")

# 是否优先使用本地视频 API（True = 优先本地，False = 仅用 ARK）
USE_LOCAL_VIDEO_API = os.environ.get("USE_LOCAL_VIDEO_API", "true").lower() in ("true", "1", "yes")

# Seedance 模型 ID（Agent Plan 支持的模型）
# 可选: doubao-seedance-2.0, doubao-seedance-2.0-fast, doubao-seedance-2.0-mini
SEEDANCE_MODEL = os.environ.get("SEEDANCE_MODEL", "doubao-seedance-2.0-mini")

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
