# HonCut 快速开始指南

## 环境要求

- Python 3.11+
- Conda 或 venv
- FFmpeg
- Docker（可选，用于服务部署）

## 快速安装

### 1. 克隆项目

```bash
cd /Users/soda/projects/honcut
```

### 2. 创建虚拟环境

使用 Conda（推荐）：
```bash
conda create -n honcut python=3.11 -y
conda activate honcut
```

或使用 venv：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
make install
```

这会安装所有 Python 依赖并配置项目。

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必要的 API 密钥：
- `ARK_AGENT_API_KEY`: 火山方舟 API 密钥（必需）
- `ARK_BASE_URL`: 火山方舟 API 地址（默认已配置）

## 运行管线

### 基本用法

```bash
make run INPUT_FILE=your_script.txt
```

### 直接调用

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --duration 60 \
  --output-dir data/output \
  --auto-approve
```

### 参数说明

- `--input`: 输入剧本文件路径
- `--duration`: 目标视频时长（秒），默认 60
- `--output-dir`: 输出目录，默认 `data/output`
- `--auto-approve`: 自动批准人工审核节点（CI/测试用）
- `--dry-run`: 仅验证流程，不生成视频
- `--transition`: 转场模式（crossfade/fade/cut）
- `--media-profile`: 编码配置（1080p/480p 等）

## 管线阶段

HonCut 管线包含 8 个阶段：

1. **Phase 1**: 初始化
2. **Phase 2**: 编剧引擎（文本解析 + 分镜生成）
3. **Phase 2.5**: 故事板图片生成
4. **Phase 3**: 角色工厂（三视图生成）
5. **Phase 4**: 编排器（镜头调度）
6. **Phase 5**: 视频生成（Seedance API）
7. **Phase 6**: 一致性守卫（质检）
8. **Phase 7**: 组装引擎（视频拼接）
9. **Phase 8**: 后期处理（字幕 + 转场）

## 测试

运行所有测试：
```bash
make test
```

运行单个测试：
```bash
pytest pipeline/tests/test_consistency_guard.py -v
```

## 常见问题

### Q: 如何跳过某个阶段？

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --skip-phase 5 6 7 8
```

### Q: 如何从检查点恢复？

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --resume
```

### Q: 视频生成失败怎么办？

1. 检查 `ARK_AGENT_API_KEY` 是否正确配置
2. 查看 `data/output/pipeline_report.json` 了解失败阶段
3. 使用 `--dry-run` 验证流程
4. 查看日志文件了解详细错误

### Q: 如何查看进度？

管线会实时输出进度信息，包括：
- 当前阶段
- 处理进度
- 生成的文件列表

## 输出文件

成功运行后，`data/output/` 目录包含：

- `polished.mp4`: 最终视频
- `STORYBOARD.json`: 分镜数据
- `CHARACTERS.json`: 角色数据
- `consistency_report.json`: 一致性检查报告
- `pipeline_report.json`: 管线执行报告
- `shots/`: 各镜头的中间文件

## 下一步

- 阅读 [API 文档](API.md) 了解详细接口
- 查看 [使用示例](EXAMPLES.md) 了解实际用法
- 参考 [管线架构](PIPELINE.md) 了解设计细节

## 需要帮助？

如果遇到问题：
1. 检查日志文件
2. 运行 `make test` 验证环境
3. 查看 `docs/MIGRATION.md` 了解项目结构
