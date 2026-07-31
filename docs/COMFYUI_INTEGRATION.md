# ComfyUI 本地视频生成集成方案（待实施）

> 状态: 📋 方案阶段
> 日期: 2026-07-31
> 目标: Phase 5 视频生成支持本地 ComfyUI 后端，零费用试错

## 策略

```
测试阶段: VIDEO_BACKEND=comfyui（本地 RTX 3070 Ti，零费用，随便试错）
生产阶段: VIDEO_BACKEND=ark（ARK Seedance 2.0，高质量成品）
```

图片生成 + LLM 推理始终用 ARK（已调通，不折腾）。

## 上海 PC 配置

- CPU: i9 12代
- GPU: RTX 3070 Ti 8GB VRAM
- RAM: 32GB
- 可用模型: Wan 2.1 1.3B（480p，~6-8GB VRAM）

## 架构

```
Mac（HonCut 管线调度）
  ├── Phase 1-4: ARK LLM + ARK Seedream（不变）
  ├── Phase 5: 
  │   ├── VIDEO_BACKEND=ark → seedance_client.py（现有）
  │   └── VIDEO_BACKEND=comfyui → comfyui_client.py（新增）
  │       └── HTTP → 上海PC:8188（ComfyUI API）
  └── Phase 6-8: 本地 ffmpeg（不变）
```

## 需要做的事

### 1. 上海 PC 部署 ComfyUI
- 安装 ComfyUI + ComfyUI-VideoHelperSuite
- 下载 Wan 2.1 1.3B 模型（~6GB）
- 启动: `python main.py --listen 0.0.0.0 --port 8188`
- 配置工作流: text_to_video（480p, 5s）

### 2. HonCut 新增 comfyui_client.py
```python
# pipeline/src/comfyui_client.py
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://192.168.x.x:8188")

def submit(prompt, duration=5, width=832, height=480, ...):
    """提交 ComfyUI 视频生成任务"""
    workflow = load_workflow("wan21_t2v.json")
    workflow["prompt"]["text"] = prompt
    # ... 填充参数
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    return resp.json()["prompt_id"]

def poll(task_id, timeout=600):
    """轮询 ComfyUI 任务状态"""
    # GET /history/{task_id}
    
def download(task_id, output_path):
    """下载生成的视频"""
    # GET /view?filename=xxx&type=output
```

### 3. config.py 加 backend 路由
```python
VIDEO_BACKEND = os.environ.get("VIDEO_BACKEND", "ark")  # ark | comfyui
```

### 4. pipeline_runner.py Phase 5 路由
```python
if VIDEO_BACKEND == "comfyui":
    from comfyui_client import submit, poll, download
else:
    from seedance_client import submit, poll, download
```

### 5. .env 配置
```
# 测试模式（本地视频）
VIDEO_BACKEND=comfyui
COMFYUI_URL=http://192.168.x.x:8188

# 生产模式（ARK 视频）
VIDEO_BACKEND=ark
```

## 注意事项

- ComfyUI API 是异步的（提交→轮询→下载），与 seedance_client 的 submit/poll/download 模式一致
- 480p 视频质量较低，但足够验证管线流程和镜头衔接
- 本地生成速度慢（3070Ti 约 2-5 分钟/5s 视频），但不花钱
- 调通后切 `VIDEO_BACKEND=ark` 即可用 Seedance 出高质量成品
