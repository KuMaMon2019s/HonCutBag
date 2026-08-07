# OpenMontage 搬运工具审计报告

审计日期：2026-08-07  
审计范围：`vendor/openmontage/` 中任务书指定的 10 个 `lib` 模块、8 个叶子工具模块，以及所有工具共同依赖的 `tools/base_tool.py`，合计 19 个 Python 模块。

## 结论

- 19/19 模块均可在仓库根目录通过 `importlib.import_module()` 独立导入。
- 8/8 具体工具类均可实例化，但只有 5 个在当前环境中能正确返回 `available`；`upscale` 和 `seedance_video` 返回 `unavailable`，`image_selector.get_status()` 直接抛出 `ModuleNotFoundError`。
- `image_selector` 是已验证的 P0：仓库中不存在其硬依赖 `tools.tool_registry`，且它还使用了不适用于 vendor 包布局的 `lib.scoring` 绝对导入。
- `seedance_video` 是已验证的 P0（若 HonCut 生产链路预期调用此 OM 工具）：它实现的是 fal.ai API，只检查 `FAL_KEY`/`FAL_AI_API_KEY`；仅配置 HonCut 主 Key `ARK_AGENT_API_KEY` 时状态为不可用。不能只把环境变量名替换成 ARK Key，因为 fal.ai 与火山方舟的认证方式、端点和请求协议不同。
- `upscale` 是 P1：当前环境缺少 `realesrgan`、`basicsr`、`cv2` 和 `gfpgan`，项目依赖清单也未声明这些运行时依赖；同时引用了包外的 `tools.video._shared`。
- `remotion_caption_burn` 会误报可用：`get_status()` 只检查 `ffmpeg`/`ffprobe`，没有检查实际渲染所需的 `npx`、`remotion-composer` 和其 `node_modules`。本仓库当前未找到 `remotion-composer`。

## 审计方法与边界

1. 从仓库根目录使用当前 Python 解释器，对每个模块启动独立子进程执行导入。
2. 实例化 8 个具体工具类，在清除 `FAL_KEY`、`FAL_AI_API_KEY` 和 `ARK_AGENT_API_KEY` 后调用 `get_status()`。
3. 对工具执行无副作用的最小调用：不存在的输入路径、状态查询或 `dry_run`；未发起网络/API 请求，也未生成媒体文件。
4. 静态检查模块的顶层与延迟导入、`dependencies` 声明、根目录 `requirements.txt`、`pyproject.toml` 以及仓库内被引用模块是否存在。
5. 检查当前环境中的 Python 包和命令。结果会随机器变化，报告同时区分“当前环境已安装”和“项目依赖已声明”。

任务书的目录树实际列出 10 个 `lib` 加 8 个叶子工具；其“9 个 tools”计数包含 `tools/base_tool.py`，因此本报告将该公共模块作为第 19 个模块审计。

## 工具状态总览

| # | 模块/工具 | 可导入 | 依赖完整 | API Key 匹配 | 状态 | 备注 |
|---:|---|:---:|:---:|:---:|---|---|
| 1 | `lib.delivery_promise` | ✅ | ✅ | N/A | ✅ 可用 | `classify_from_brief("cinematic", {})` 通过 |
| 2 | `lib.events` | ✅ | ✅ | N/A | ✅ 可用 | 仅标准库和相对导入 `lib.paths` |
| 3 | `lib.media_profiles` | ✅ | ✅ | N/A | ✅ 可用 | `get_profile("youtube_landscape")` 通过 |
| 4 | `lib.paths` | ✅ | ✅ | N/A | ✅ 可用 | 仅标准库；支持可选 `OPENMONTAGE_PROJECTS_DIR` |
| 5 | `lib.scoring` | ✅ | ✅ | N/A | ✅ 可用 | 纯 Python；最小规范化调用通过 |
| 6 | `lib.shot_prompt_builder` | ✅ | ✅ | N/A | ✅ 可用 | 纯 Python；最小 prompt 构建通过 |
| 7 | `lib.slideshow_risk` | ✅ | ✅ | N/A | ✅ 可用 | 纯 Python；空场景评分通过 |
| 8 | `lib.source_media_review` | ✅ | ⚠️ | N/A | ⚠️ 部分可用 | 图片路径延迟导入 Pillow，但 Pillow 未在项目依赖中声明；默认 registry 路径不存在，不过代码会捕获异常并尝试本地探测 |
| 9 | `lib.variation_checker` | ✅ | ✅ | N/A | ✅ 可用 | 纯 Python；空场景检查通过 |
| 10 | `lib.verify_scene_pacing` | ✅ | ✅ | N/A | ✅ 可用 | 纯 Python；空步骤 trace 通过 |
| 11 | `tools.base_tool` | ✅ | ⚠️ | N/A | ⚠️ 部分可用 | 核心依赖检查可用；遥测延迟导入 `lib.events`，在标准 vendor 包布局中路径错误且异常被静默吞掉 |
| 12 | `tools.audio.audio_mixer` | ✅ | ✅ | N/A | ✅ 可用 | 当前 `ffmpeg` 可用；不存在输入的最小调用正确返回失败结果。文档提及的 pydub 是可选依赖且当前未安装 |
| 13 | `tools.enhancement.color_grade` | ✅ | ✅ | N/A | ✅ 可用 | 当前 `ffmpeg` 可用；不存在输入的最小调用正确返回失败结果 |
| 14 | `tools.enhancement.upscale` | ✅ | ❌ | N/A | ❌ 不可用 | `get_status()` 为 `unavailable`；实际还需要 `cv2`、`basicsr`，face enhance 需要 `gfpgan`，并引用包外 `_shared` |
| 15 | `tools.graphics.image_selector` | ✅ | ❌ | N/A | ❌ 不可用（P0） | 可实例化，但 `get_status()` 因缺少 `tools.tool_registry` 抛异常；执行路径还会因 `lib.scoring` 绝对导入失败 |
| 16 | `tools.video.auto_reframe` | ✅ | ⚠️ | N/A | ⚠️ 基础功能可用 | 当前 `ffmpeg` 可用；内部尝试导入不存在的 `tools.analysis.face_tracker`，异常被吞掉，故自动人脸跟踪会静默降级 |
| 17 | `tools.video.remotion_caption_burn` | ✅ | ❌ | N/A | ⚠️ 状态误报 | 状态为 `available`，但只验证 FFmpeg；当前有 `npx`，仓库没有所需 `remotion-composer/node_modules`。Pillow fallback 已安装但未声明 |
| 18 | `tools.video.seedance_video` | ✅ | ❌ | ❌ | ❌ 不可用（P0） | 清空 fal Key 后状态为 `unavailable`；只设置 ARK Key 也不会可用；`requests` 已声明，但本地上传/输出探测依赖包外 `_shared` |
| 19 | `tools.video.video_stitch` | ✅ | ⚠️ | N/A | ⚠️ 基础功能可用 | 当前 `ffmpeg`/`ffprobe` 可用，`dry_run` 通过；指定 profile 时错误导入 `lib.media_profiles`，异常被捕获后会静默忽略 profile |

## 导入、实例化与最小调用实测

### 导入

所有 19 个模块的独立子进程返回码均为 0，无标准错误输出。该结果只证明顶层导入成功；延迟导入问题由后续状态与调用测试发现。

### 工具状态

| 工具类 | 实例化 | `get_status()`（无 API Key） |
|---|:---:|---|
| `AudioMixer` | ✅ | `available` |
| `ColorGrade` | ✅ | `available` |
| `Upscale` | ✅ | `unavailable` |
| `ImageSelector` | ✅ | 抛出 `ModuleNotFoundError: No module named 'tools'` |
| `AutoReframe` | ✅ | `available` |
| `RemotionCaptionBurn` | ✅ | `available`（误报，见依赖审计） |
| `SeedanceVideo` | ✅ | `unavailable` |
| `VideoStitch` | ✅ | `available` |

### 无副作用调用

- `AudioMixer`、`ColorGrade`、`Upscale`、`AutoReframe` 对不存在输入均返回结构化失败结果，没有写文件。
- `SeedanceVideo.execute()` 在缺少 fal Key 时返回 `FAL_KEY not set`，未发出请求。
- `VideoStitch` 的空 clips `dry_run` 成功。
- `ImageSelector.execute({"operation": "rank", ...})` 在 provider 排名路径抛出 `ModuleNotFoundError: No module named 'lib'`。
- `RemotionCaptionBurn` 的最小无效输入在读取完整输入 schema 前即要求 `output_path`；未进行真实渲染。其 composer 缺失已通过文件系统检查确认。

## 缺失或不完整依赖清单

| 依赖 | 分类 | 当前环境 | 项目声明/仓库状态 | 影响 |
|---|---|---|---|---|
| `tools.tool_registry` | HonCut 项目内/需搬运或适配 | ❌ | 仓库不存在 | `ImageSelector.get_status()`、provider 发现和执行全部失败；P0 |
| `lib.scoring` / `lib.events` / `lib.media_profiles` | 包内路径错误 | ❌ 顶层 `lib` 包 | 实际文件位于 `vendor.openmontage.lib` | ImageSelector 执行失败；BaseTool 遥测丢失；VideoStitch profile 被静默忽略 |
| `tools.video._shared` | HonCut 项目内/路径耦合 | ⚠️ | 存在于 `pipeline/src/tools/video/_shared.py`，不在 vendor 包内；从仓库根直接运行时不可导入 | Seedance 本地图片上传、输出探测以及 Upscale device 选择失败 |
| `tools.analysis.face_tracker` | HonCut 项目内/路径错误 | ❌ | 仓库没有该路径；相近实现是 `pipeline/src/tools/face_tracker.py` | AutoReframe 人脸跟踪静默降级 |
| `realesrgan` | 第三方 Python | ❌ | 未在 requirements/pyproject 声明 | Upscale 不可用 |
| `basicsr` | 第三方 Python | ❌ | 未声明 | Upscale 模型构建失败 |
| `opencv-python`（`cv2`） | 第三方 Python | ❌ | 未声明，也未列入工具 `dependencies` | 图片/视频 upscale 处理失败 |
| `gfpgan` | 第三方 Python，可选功能 | ❌ | 未声明 | `face_enhance=True` 失败 |
| `torch` | 第三方 Python | ✅ | 未在 requirements/pyproject 声明 | 当前机器偶然可用，但新环境不可复现 |
| Pillow（`PIL`） | 第三方 Python | ✅ | 未在 requirements/pyproject 声明 | SourceMediaReview 图片探测和 Remotion Pillow fallback 在新环境可能失败 |
| `requests` | 第三方 Python | ✅ | 根 requirements 与 pyproject 均已声明 | Seedance HTTP 主流程依赖完整 |
| `ffmpeg` / `ffprobe` | 系统命令 | ✅ | requirements 注释有安装说明 | AudioMixer、ColorGrade、AutoReframe、Remotion、VideoStitch 基础依赖满足 |
| `npx` / Node.js | 系统命令 | ✅ | 仅 requirements 注释提到 Node | Remotion CLI 前置命令存在 |
| `remotion-composer` + `node_modules` | HonCut 项目资产/Node 依赖 | ❌ | 仓库未找到 | Remotion 主渲染路径不可用，但状态误报 |
| `pydub` | 第三方 Python，可选 | ❌ | 未声明；AudioMixer 明确可 FFmpeg-only 降级 | 不阻塞基础 AudioMixer |

## API Key 审计

只有 `tools.video.seedance_video` 在本次 19 模块范围内直接检查 API Key：

- `_get_api_key()` 仅返回 `FAL_KEY` 或 `FAL_AI_API_KEY`。
- `get_status()` 只以 fal Key 是否存在判断可用性。
- 安装说明、错误信息和 HTTP 实现均明确指向 fal.ai。
- HonCut 主链路广泛使用 `ARK_AGENT_API_KEY`；仅设置该 Key 时，此 OM 工具仍为 `unavailable`。

这不是简单的环境变量别名问题。`ARK_AGENT_API_KEY` 是火山方舟凭证，不能安全地作为 fal.ai Bearer token 使用。建议在以下方案中二选一：

1. **推荐：改造为 HonCut/ARK 实现。** 复用现有 Seedance ARK client、请求端点、任务轮询和下载逻辑，让状态检查与调用统一使用 `ARK_AGENT_API_KEY`。
2. **保留 fal.ai provider。** 将工具明确命名/标记为 fal.ai Seedance provider，继续要求 `FAL_KEY`，并让 HonCut 的 provider 路由在无 fal Key 时选择 ARK Seedance 实现。此方案下报告中的“API Key 不匹配”仍成立，但属于 provider 配置不匹配，而不是 Key 名拼写错误。

不要采用“仅把 `FAL_KEY` 改成 `ARK_AGENT_API_KEY`”的修复，这会把 ARK 凭证发送给错误的服务且调用仍无法成功。

## 修复建议

### P0：修复 `ImageSelector` 的 registry 与包导入

1. 从对应 OpenMontage 版本搬运 `tool_registry.py` 及其发现机制，或为 HonCut 实现一个满足 `ensure_discovered()`、`get_by_capability()`、`get()` 的最小 registry adapter。
2. 将 `tools.tool_registry`、`lib.scoring` 改为 vendor 包内相对导入，或通过明确的 adapter 注入 registry；不要依赖调用方碰巧把 `pipeline/src` 加入 `sys.path`。
3. 增加测试：顶层 import、实例化、`get_status()`、无 provider、单 provider、`rank` 和正常选择路径。

### P0：统一 Seedance provider 与认证架构

1. 决定该类代表 ARK Seedance 还是 fal.ai Seedance，并在类名、provider 元数据和文档中明确。
2. 若接入 HonCut 主链路，复用 ARK client 并检查 `ARK_AGENT_API_KEY`；同步替换 fal.ai endpoint、payload、上传和轮询逻辑。
3. 若保留 fal.ai，补充 provider 路由和明确的可选 `FAL_KEY` 配置，不要把 ARK Key 当作 fal Key。
4. 把 `requests` 和共享上传/探测 helper 纳入工具依赖契约，并为 text/image/reference 三种 operation 添加离线 mock 测试。

### P1：补全 `Upscale` 可复现依赖

1. 在可选依赖组（建议 `upscale` extra）声明兼容版本的 `torch`、`realesrgan`、`basicsr`、`opencv-python`；`face_enhance` extra 再加入 `gfpgan`。
2. 把 `cv2`、`basicsr`、可选 `gfpgan` 纳入状态检查，使 `get_status()` 与真实执行条件一致。
3. 搬运/封装 `get_torch_device`，改用 vendor 相对导入，避免依赖 `pipeline/src` 的顶层 `tools` 包。
4. 添加 CPU/MPS/CUDA 的最小图片测试及视频帧拆装测试。

### P1：让 Remotion 状态反映真实可用性

1. 在依赖检查中加入 `npx`、composer 根目录、`package.json`、`node_modules` 以及 Chrome/可用 fallback 的检查。
2. 将 `remotion-composer` 及 lockfile 搬入仓库或把路径做成显式配置，并提供确定性的安装步骤。
3. 在 Python 依赖中声明 Pillow（若保留 Pillow fallback）。
4. 增加 `get_status()` 不得在 composer 缺失时返回 available 的回归测试。

### P1：消除其余包路径耦合和静默降级

1. `BaseTool` 使用 `vendor.openmontage.lib.events` 相对导入，并记录遥测初始化失败，而非无提示吞掉。
2. `VideoStitch` 使用 vendor 相对 `media_profiles` 导入；profile 不可解析时返回明确错误，不要静默采用默认值。
3. `AutoReframe` 修正 FaceTracker 的真实模块路径或注入 tracker；把“跟踪不可用”写入结果 metadata/status。
4. `SourceMediaReview` 接受显式 registry adapter；在没有 registry 时仅启用清晰标记的 FFprobe/Pillow fallback。

### P2：整理依赖声明

1. 将 Pillow 加入项目依赖或对应 optional extra。
2. 保留 FFmpeg、FFprobe、Node/Remotion 的系统依赖安装说明，并在启动前执行统一 preflight。
3. 建立 CI 审计矩阵：最小依赖环境验证导入，完整媒体环境验证状态与小样本调用，mock 环境验证 API 工具。

## 验收项

- [x] 19 个模块均执行独立导入测试
- [x] 所有模块的顶层与延迟依赖均检查
- [x] 8 个具体工具类均实例化并检查状态
- [x] `FAL_KEY` / `FAL_AI_API_KEY` 与 `ARK_AGENT_API_KEY` 逻辑已审计
- [x] 已执行不写文件、不调用外部 API 的最小功能验证
- [x] 已列出缺失依赖、风险等级和具体修复步骤
- [x] 未修改 vendor 或 pipeline 源码

