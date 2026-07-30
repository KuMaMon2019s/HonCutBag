# HonCut 使用示例

## 示例 1: 基础视频生成

最简单的用法，从剧本生成 60 秒视频。

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/basic_example \
  --auto-approve
```

**输出**:
- `data/output/basic_example/polished.mp4` - 最终视频
- `data/output/basic_example/STORYBOARD.json` - 分镜数据
- `data/output/basic_example/CHARACTERS.json` - 角色数据

---

## 示例 2: Dry Run 测试

仅验证流程，不实际生成视频（快速测试）。

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 30 \
  --output-dir data/output/dry_run_test \
  --dry-run
```

**用途**: 快速检查剧本格式、配置是否正确，不消耗 API 额度。

---

## 示例 3: 跳过特定阶段

如果某些阶段已经完成，可以跳过。

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/skip_example \
  --skip-phase 2 3 4 \
  --auto-approve
```

**用途**: 重新生成视频（Phase 5+），跳过编剧和角色生成。

---

## 示例 4: 从检查点恢复

如果管线中断，可以从检查点恢复。

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/resume_example \
  --resume
```

**用途**: 恢复中断的管线执行，避免重复已完成的工作。

---

## 示例 5: 自定义转场和编码

指定转场效果和编码配置。

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_encode \
  --transition crossfade \
  --media-profile 1080p \
  --auto-approve
```

**可用转场**:
- `crossfade` - 交叉淡入淡出（默认）
- `fade` - 淡入淡出
- `cut` - 硬切

**可用编码配置**:
- `1080p` - 1920x1080 @ 30fps
- `480p` - 854x480 @ 30fps
- `720p` - 1280x720 @ 30fps
- `cinematic` - 2048x858 @ 24fps
- `youtube_shorts` - 1080x1920 @ 30fps（竖屏）

---

## 示例 6: Python API 调用

在 Python 代码中调用管线。

```python
from pipeline.src.pipeline_runner import run_pipeline

result = run_pipeline(
    input_file="scripts/sample_story.txt",
    duration=60,
    output_dir="data/output/python_api",
    dry_run=False,
    auto_approve=True,
    transition="crossfade",
    media_profile="1080p"
)

print(f"执行状态: {result['status']}")
print(f"总耗时: {result['total_duration']:.2f}s")
print(f"输出目录: {result['output_dir']}")
```

---

## 示例 7: 批量处理多个剧本

使用脚本批量处理多个剧本。

```bash
#!/bin/bash
# batch_process.sh

for story in scripts/stories/*.txt; do
    output_name=$(basename "$story" .txt)
    echo "处理: $story"
    
    python pipeline/src/pipeline_runner.py \
        --input "$story" \
        --duration 60 \
        --output-dir "data/output/batch/$output_name" \
        --auto-approve
    
    echo "完成: $output_name"
done
```

**用法**:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

---

## 示例 8: 检查一致性报告

生成视频后检查角色一致性。

```python
from pipeline.src.consistency_guard import run_consistency_check

report = run_consistency_check(
    output_dir="data/output/basic_example",
    threshold=70
)

print(f"整体一致性: {report['overall_score']:.2f}")
print(f"是否通过: {report['passed']}")

for char_id, details in report['details'].items():
    print(f"  {char_id}: {details['score']:.2f}")
    if details['issues']:
        print(f"    问题: {details['issues']}")
```

---

## 示例 9: 自定义提示词生成

手动修改分镜提示词后重新生成。

```bash
# 1. 先生成分镜
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_prompt \
  --skip-phase 5 6 7 8 \
  --auto-approve

# 2. 编辑 STORYBOARD.json
# 修改 prompts 字段，添加自定义描述

# 3. 从 Phase 5 继续
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_prompt \
  --resume \
  --auto-approve
```

---

## 示例 10: 监控管线进度

实时监控管线执行进度。

```bash
# 在另一个终端监控
tail -f data/output/monitor_example/pipeline.log

# 查看当前阶段
watch -n 1 'cat data/output/monitor_example/progress.json | jq .current_phase'
```

---

## 示例 11: 使用 Makefile 命令

使用预定义的 Makefile 命令简化操作。

```bash
# 安装依赖
make install

# 运行测试
make test

# 运行管线
make run INPUT_FILE=scripts/sample_story.txt

# 清理输出
make clean

# 启动 Docker 服务
make docker-up

# 停止 Docker 服务
make docker-down
```

---

## 示例 12: 调试模式

启用详细日志进行调试。

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 运行管线
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/debug_example \
  --auto-approve 2>&1 | tee debug.log
```

---

## 示例 13: 生成不同风格视频

使用不同的媒体配置生成不同风格的视频。

```bash
# 电影风格
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --output-dir data/output/cinematic \
  --media-profile cinematic \
  --transition fade \
  --auto-approve

# YouTube Shorts（竖屏）
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --output-dir data/output/shorts \
  --media-profile youtube_shorts \
  --transition crossfade \
  --auto-approve
```

---

## 示例 14: 检查管线报告

查看管线执行报告。

```python
import json

with open("data/output/basic_example/pipeline_report.json") as f:
    report = json.load(f)

print("=== 管线执行报告 ===")
print(f"状态: {report['status']}")
print(f"总耗时: {report['total_duration']:.2f}s")
print(f"阶段数: {len(report['phases'])}")

for phase_name, phase_data in report['phases'].items():
    print(f"\n{phase_name}:")
    print(f"  状态: {phase_data['status']}")
    print(f"  耗时: {phase_data['duration']:.2f}s")
    if phase_data.get('error'):
        print(f"  错误: {phase_data['error']}")
```

---

## 示例 15: 环境变量配置

使用环境变量配置 API 和其他设置。

```bash
# 配置 API 密钥
export ARK_AGENT_API_KEY=your_api_key_here

# 配置 API 地址
export ARK_BASE_URL=https://api.volcengine.com

# 配置日志级别
export LOG_LEVEL=INFO

# 运行管线
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/env_config \
  --auto-approve
```

---

## 常见问题

### Q: 如何查看 API 调用详情？

设置 `LOG_LEVEL=DEBUG` 并查看日志文件。

### Q: 如何中断正在运行的管线？

按 `Ctrl+C`，管线会保存检查点，下次可以用 `--resume` 恢复。

### Q: 如何重新生成某个镜头？

删除对应的 `shots/SXX/` 目录，然后使用 `--resume` 重新运行。

### Q: 如何修改角色外观？

编辑 `CHARACTERS.json` 中的角色描述，然后使用 `--resume` 重新生成。

### Q: 如何导出中间结果？

所有中间文件都在 `output_dir/` 目录下，包括：
- `STORYBOARD.json` - 分镜数据
- `CHARACTERS.json` - 角色数据
- `shots/` - 各镜头的视频和图片

---

## 下一步

- 阅读 [QUICKSTART.md](QUICKSTART.md) 了解快速开始
- 阅读 [API.md](API.md) 了解详细接口
- 阅读 [PIPELINE.md](PIPELINE.md) 了解管线架构
