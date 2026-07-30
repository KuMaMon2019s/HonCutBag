# HonCut - AI 视频生成管线

HonCut 是一个端到端的 AI 视频生成管线，从文本输入到最终视频输出。

## 架构

```
文本输入 → 编剧引擎 → 角色工厂 → 编排器 → 视频生成 → 组装引擎 → 后期处理 → 视频输出
```

## 快速开始

### 1. 安装依赖

```bash
# 使用 conda（推荐）
conda env create -f environment.yml
conda activate honcut

# 或使用 pip
pip install -r pipeline/requirements.txt
```

### 2. 启动 Docker 服务

```bash
make docker-up
```

这将启动：
- **Qdrant** (端口 6333) - 向量数据库
- **MinIO** (端口 9000/9001) - 对象存储
- **n8n** (端口 5678) - 工作流自动化

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入 API keys
```

### 4. 运行管线

```bash
# 使用 make（自动创建日期工作空间）
make run INPUT_FILE=path/to/script.txt

# 或直接运行（指定输出目录）
python pipeline/src/pipeline_runner.py \
  --input path/to/script.txt \
  --output-dir workspaces/$(date +%Y-%m-%d)_01
```

## 工作空间结构

工作空间（日期+序号）保存在 `workspaces/` 目录下（软链接到 `/Users/soda/knowledge-base`）：

```
workspaces/
├── 2026-07-30_01/          # 日期_序号
│   ├── input/              # 输入文件
│   ├── output/             # 输出视频和中间文件
│   ├── shots/              # 各镜头数据
│   └── checkpoint.json     # 检查点（用于 --resume）
├── 2026-07-31_01/
└── ...
```

**命名规则**：`YYYY-MM-DD_XX`，其中 `XX` 是当天序号（01, 02, ...）

## 项目结构

```
honcut/
├── pipeline/           # Python 视频管线
│   ├── src/           # 源代码
│   ├── tests/         # 测试
│   └── requirements.txt
├── docker/            # Docker 配置
│   └── docker-compose.yml
├── data/              # 数据目录
│   ├── input/         # 输入文件
│   └── output/        # 输出文件
├── docs/              # 文档
├── scripts/           # 辅助脚本
├── Makefile           # 常用命令
├── pyproject.toml     # Python 项目配置
└── environment.yml    # Conda 环境配置
```

## 管线阶段

### Phase 1: 编剧引擎
- 文本解析
- 事件提取
- 角色发现
- 分镜生成

### Phase 2: 角色工厂
- 角色三视图生成
- 角色描述生成

### Phase 3: 编排器
- 镜头调度
- 时间线规划

### Phase 4: 视频生成
- Seedance API 调用
- 视频片段生成

### Phase 5: 组装引擎
- 视频片段拼接
- 转场效果

### Phase 6: 后期处理
- 调色
- 音频处理
- 字幕烧录

## 开发

### 代码格式化与检查

项目使用 **Black** 和 **Ruff** 进行代码格式化和检查：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化代码
black pipeline/src/ pipeline/tests/

# 检查代码
ruff check pipeline/src/ pipeline/tests/
```

### Pre-commit Hooks

项目配置了 pre-commit hooks，在提交时自动运行代码检查和测试：

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 git hooks
pre-commit install

# 手动运行所有 hooks
pre-commit run --all-files
```

Pre-commit hooks 包括：
- 代码格式化检查 (black)
- 代码质量检查 (ruff)
- 基础文件检查 (trailing whitespace, end-of-file, yaml/toml 语法等)
- 运行测试 (pytest)

跳过 hooks（不推荐）：
```bash
git commit --no-verify
```

### 运行测试

```bash
# 运行测试
make test

# 或使用 pytest
pytest pipeline/tests/ -v --cov=pipeline/src

# 清理构建产物
make clean
```

## 依赖

- Python 3.11+
- Docker & Docker Compose
- FFmpeg
- 火山方舟 API (Seedance, Seedream)

## 许可证

MIT
