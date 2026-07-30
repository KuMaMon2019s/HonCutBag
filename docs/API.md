# HonCut API 文档

## 核心模块

### pipeline_runner.py

主管线执行器，串联所有阶段。

#### 主要函数

```python
def run_pipeline(
    input_file: str,
    duration: int = 60,
    output_dir: str = "data/output",
    dry_run: bool = False,
    auto_approve: bool = False,
    skip_phase: List[int] = None,
    transition: str = "crossfade",
    media_profile: str = "1080p"
) -> dict
```

**参数**:
- `input_file`: 输入剧本文件路径
- `duration`: 目标视频时长（秒）
- `output_dir`: 输出目录
- `dry_run`: 仅验证流程，不生成视频
- `auto_approve`: 自动批准人工审核节点
- `skip_phase`: 跳过的阶段列表
- `transition`: 转场模式（crossfade/fade/cut）
- `media_profile`: 编码配置

**返回**: 执行报告字典

---

### character_factory.py

角色工厂，生成角色三视图。

#### 主要函数

```python
def generate_character_views(
    character: dict,
    output_dir: str
) -> dict
```

**参数**:
- `character`: 角色数据（来自 CHARACTERS.json）
- `output_dir`: 输出目录

**返回**: 包含三视图路径的字典

---

### seedance_client.py

Seedance API 客户端，用于视频生成。

#### 主要函数

```python
def submit_video_generation(
    prompt: str,
    image_url: str = None,
    duration: int = 5,
    **kwargs
) -> dict
```

**参数**:
- `prompt`: 视频生成提示词
- `image_url`: 参考图片 URL（可选）
- `duration`: 视频时长
- `**kwargs`: 其他参数

**返回**: 任务提交结果

```python
def poll_video_status(
    task_id: str,
    timeout: int = 300
) -> dict
```

**参数**:
- `task_id`: 任务 ID
- `timeout`: 超时时间（秒）

**返回**: 视频生成状态和结果

---

### seedream_client.py

Seedream API 客户端，用于图片生成。

#### 主要函数

```python
def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> dict
```

**参数**:
- `prompt`: 图片生成提示词
- `width`: 图片宽度
- `height`: 图片高度
- `**kwargs`: 其他参数

**返回**: 图片生成结果

---

### consistency_guard.py

一致性守卫，检查角色一致性。

#### 主要函数

```python
def run_consistency_check(
    output_dir: str,
    threshold: int = 70
) -> dict
```

**参数**:
- `output_dir`: 输出目录
- `threshold`: 一致性阈值（0-100）

**返回**: 一致性检查报告

---

### storyboard_generator.py

分镜生成器，生成 STORYBOARD.json。

#### 主要函数

```python
def generate_storyboard(
    script_text: str,
    duration: int = 60
) -> dict
```

**参数**:
- `script_text`: 剧本文本
- `duration`: 目标时长

**返回**: 分镜数据

---

## 数据结构

### CHARACTERS.json

```json
{
  "characters": [
    {
      "id": "char_001",
      "name": "主角",
      "description": "角色描述",
      "appearance": {
        "hair": "黑色短发",
        "clothing": "休闲装",
        "face": "坚毅面容",
        "build": "中等身材",
        "gender": "男",
        "age_range": "25-35"
      },
      "reference_images": {
        "front": "path/to/front.png",
        "side": "path/to/side.png",
        "back": "path/to/back.png"
      }
    }
  ]
}
```

### STORYBOARD.json

```json
{
  "shots": [
    {
      "id": "S01",
      "prompt": "镜头描述",
      "duration": 5,
      "characters": ["char_001"],
      "scene": "场景描述",
      "camera": "镜头运动"
    }
  ]
}
```

### consistency_report.json

```json
{
  "overall_score": 85,
  "passed": true,
  "details": {
    "char_001": {
      "score": 90,
      "issues": []
    }
  }
}
```

---

## 配置选项

### config.yaml

```yaml
pipeline:
  default_duration: 60
  default_transition: crossfade
  default_media_profile: 1080p

api:
  ark_base_url: https://api.volcengine.com
  timeout: 300

consistency:
  threshold: 70
  max_retries: 3

generation:
  max_concurrent: 3
  retry_delay: 5
```

---

## 错误处理

### 常见错误码

- `API_KEY_MISSING`: API 密钥未配置
- `API_TIMEOUT`: API 调用超时
- `GENERATION_FAILED`: 视频/图片生成失败
- `CONSISTENCY_LOW`: 一致性检查未通过
- `PHASE_FAILED`: 阶段执行失败

### 错误处理示例

```python
try:
    result = run_pipeline(...)
except APIKeyError as e:
    print(f"API 密钥错误: {e}")
except TimeoutError as e:
    print(f"API 调用超时: {e}")
except PipelineError as e:
    print(f"管线执行错误: {e}")
```

---

## 扩展开发

### 添加新的阶段

1. 在 `pipeline/src/` 创建新模块
2. 实现 `execute()` 函数
3. 在 `pipeline_runner.py` 注册阶段
4. 添加测试到 `pipeline/tests/`

### 自定义转场效果

1. 修改 `phase7_assembly.py`
2. 实现新的转场函数
3. 在配置文件中注册

### 集成新的 API

1. 创建新的 client 模块
2. 实现标准接口（submit/poll）
3. 在配置文件中添加 API 配置
