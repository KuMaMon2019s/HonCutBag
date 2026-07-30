# HonCut 项目迁移文档

## 迁移概述

**迁移日期**: 2026-07-30  
**源位置**: `/Users/soda/knowledge-base/2026-07-28_01/`  
**目标位置**: `/Users/soda/projects/honcut/`

## 迁移原因

1. **项目结构混乱**: 代码散落在多个日期目录中（2026-07-27_05, 2026-07-27_06, 2026-07-28_01, 2026-07-30_01）
2. **版本不一致**: 不同目录包含不同版本的代码
3. **缺乏标准化**: 没有统一的项目配置和依赖管理
4. **环境依赖**: 缺少虚拟环境配置，导致依赖冲突

## 新项目结构

```
/Users/soda/projects/honcut/
├── pipeline/              # 核心管线代码
│   ├── src/              # Python 源代码 (23个文件)
│   │   ├── pipeline_runner.py
│   │   ├── character_factory.py
│   │   ├── seedance_client.py
│   │   └── ...
│   └── requirements.txt  # Python 依赖
├── docker/               # Docker 配置
│   └── docker-compose.yml
├── data/                 # 数据目录
│   ├── input/           # 输入文件
│   └── output/          # 输出文件
├── docs/                 # 文档
├── scripts/              # 辅助脚本
├── .gitignore           # Git 忽略规则
├── .env.example         # 环境变量示例
├── Makefile             # 常用命令
├── pyproject.toml       # Python 项目配置
├── environment.yml      # Conda 环境配置
└── README.md            # 项目说明
```

## 旧目录处理

### 已归档
- `/Users/soda/knowledge-base/archived/2026-07-27_05/` - 旧版本代码
- `/Users/soda/knowledge-base/archived/2026-07-27_06/` - 旧版本代码

### 已删除
- `/Users/soda/knowledge-base/2026-07-30_01/` - 重复的副本

### 保留
- `/Users/soda/knowledge-base/2026-07-28_01/` - 作为参考（最新版本）

## 新项目使用方法

### 1. 激活虚拟环境

```bash
conda activate honcut
```

### 2. 安装依赖

```bash
cd /Users/soda/projects/honcut
make install
```

### 3. 运行管线

```bash
# 使用 make
make run INPUT_FILE=path/to/script.txt

# 或直接运行
python pipeline/src/pipeline_runner.py --input path/to/script.txt --output-dir data/output
```

### 4. 启动 Docker 服务

```bash
make docker-up
```

## 关键改进

1. **统一项目结构**: 所有代码集中在 `/Users/soda/projects/honcut/`
2. **标准化配置**: 使用 `pyproject.toml` 和 `environment.yml`
3. **依赖管理**: `requirements.txt` 明确列出所有依赖
4. **环境隔离**: Conda 虚拟环境避免依赖冲突
5. **Docker 集成**: `docker-compose.yml` 管理服务
6. **Makefile**: 简化常用操作

## 验证清单

- [x] 23个 Python 文件已迁移
- [x] 配置文件已创建（pyproject.toml, environment.yml）
- [x] Docker 配置已创建
- [x] Git 仓库已初始化
- [x] `make install` 测试通过
- [x] `pipeline_runner.py --help` 测试通过
- [x] 旧目录已归档/删除

## 回滚方案

如果需要回滚，可以从以下位置恢复：
- `/Users/soda/knowledge-base/2026-07-28_01/` - 完整备份
- `/Users/soda/knowledge-base/archived/` - 旧版本代码

## 后续工作

1. 更新所有文档指向新项目路径
2. 配置 CI/CD 流程
3. 添加自动化测试
4. 完善 Docker 镜像构建

---

**迁移完成时间**: 2026-07-30 22:15  
**迁移状态**: ✅ 成功
