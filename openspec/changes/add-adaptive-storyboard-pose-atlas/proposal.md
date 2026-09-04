## Why

真实 7 秒 Seedance 实验表明，模型可以在约 5～6 秒内完成一组高幅度连续动作，并在剩余约 1 秒保持稳定格挡；继续强制每一秒都有新动作反而会制造不必要的节奏失败。HonCut 需要把 Gxx 从“逐格独立动作”明确为可扩展的姿态采样，并按分镜时长、语义动作组、媒体预算和单一连续运镜确定性选择单页或分页图集。

## What Changes

- 引入持续时间感知的动作时间合同，将 `initial_anchor`、`story_action` 和可接受的 `terminal_hold` 分离；普通终态默认验证动作语义，不要求精确关节复刻。
- 将 Gxx 定义为 pose samples，并把一个或多个 pose samples 绑定到有序 action groups；4～15 秒分镜按 Provider 能力配置确定采样容量，而不是写死剧情或固定九格动作数。
- 在媒体预算允许且动作密度较高时使用分页九宫格，在预算只容纳一张导航图时使用单张 18/27/36 格 atlas；两种布局都由 Phase 2 本地确定性渲染。
- 扩展 canonical camera-motion contract 的受控参数和时长可行性验证；Adaptation 仍是运镜选择 owner，Phase 2 只沿同一连续摄影机路径投影各姿态视角。
- 版本化升级 pose、guide、shot-storyboard、continuity 和任务 fingerprint 合同；旧资产只在完整血缘可验证时零请求迁移。
- 补充单元、Graph/顺序一致性、零 Provider 恢复、媒体预算和版本迁移测试，并记录真实实验只作为校准证据、不作为生产验收通过。

### Scope

- Phase 2 动作分组、姿态采样、图集布局、身份中立 renderer 和收据。
- Adaptation 已有 camera-motion contract 的能力参数与可行性校验，不改变镜头选择 ownership。
- Phase 4 continuity schema、Phase 6 Prompt/媒体消费、Provider 能力配置及相应测试/文档。

### Non-goals

- 不修改 Graph 拓扑、Provider transport、重试/补拍策略或 QA 阈值。
- 不让 Phase 2 重新选择运镜、改写剧情或增加 action unit。
- 不保证模型逐格机械复现密集 atlas，也不把一次实验剧情写入生产代码。
- 不修改 `pipeline/src/phases/pipeline_core.py`，本轮不调用任何付费 Provider。

## Capabilities

### New Capabilities

- `adaptive-storyboard-pose-atlas`: 定义持续时间感知的动作组、姿态采样、终态稳定窗口、单页/分页身份中立图集、运镜可行性和下游验证合同。

### Modified Capabilities

- 无。现有 `storyboard-guide-pose-semantics` change 尚未归档为主规格；本 change 在其已实现的 v4 导航图行为上增加新能力，并保留其全部不变量。

## Impact

- **Pipeline**: Phase 2、Phase 4 continuity、Phase 6 精确消费与 Runtime video capability profile；Graph 与顺序执行器继续调用相同 owner。
- **API / frontend / backend**: 不适用；无外部 HTTP API 或前端合同变化。
- **Database**: 不改变 `runtime.db` schema；任务 payload/fingerprint 增加 JSON-safe 字段。
- **Configuration**: Provider 能力 profile 增加动作采样、语义动作组和终态稳定窗口参数；无新环境变量。
- **Architecture**: 不改变 owner 或依赖方向，但属于跨 Phase 的版本化公共合同变化，因此同一 change 更新 `docs/HONCUT_ARCHITECTURE.md`。
- **Tests / migration**: 更新 Phase 2/4/6 合同测试、媒体预算测试、Graph/顺序一致性、十轮零请求恢复和旧版迁移/future-schema fail-closed 覆盖。
