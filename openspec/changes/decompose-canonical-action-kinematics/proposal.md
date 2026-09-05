## Why

HonCut 目前只把复杂人体动作压缩成 technique、limbs、footwork、torso 等自由文本或单一姿态结果，无法稳定表达左右肢体、腰与头的协同、人物朝向变化以及翻转/旋转过程。因此下游骨架、动作板和视频控制即使收到正确动作名称，也容易退化成小幅单姿态插值、缺少组合招式且节奏迟缓。

## What Changes

- 在来源层按 `micro_action_index` 编译运动学事实，并在 generation action unit 与 Pxx 已确定后生成严格、无重叠的生产投影，避免在 GAU 尚不存在时提前绑定错误 ID。
- 每个动作保存一个或多个 `actor_tracks`；每条 performer 轨道分别包含左右手臂/手、左右腿/脚、腰/躯干、头部和根节点的有序动作通道，禁止把同步攻防或抓控双方压成一个骨架。
- 将演员相对于 Pxx 起始姿态的朝向变化与摄影机相对视图分离；正面、背面、左/右侧面和三分之四是可验证的视图投影，不伪造不存在的绝对世界朝向。
- 为通道阶段保存量化、归一化的关节位移、旋转、根位移和幅度等级，使下游无需再次解释动作 token 即可产生大幅度骨架变化。
- 将翻转与旋转建模为有轴线、方向、角度/圈数、支撑状态和起止朝向的受控空间变换；未在来源或既有身体力学事实中出现的变换不得被补造。
- 保留原 action-unit、source-action、performer、target 和时序血缘；运动学子通道只细化现有动作，不增加剧情动作、不改变 Pxx 容量计数。
- 保持现有 `BodyActionUnderstanding` 模型输出字段不变；运动学由代码根据已经验证的身体力学与来源血缘编译，不增加 LLM 请求或长结构化响应负担。
- Phase 2 姿态采样、Phase 3 动作参考板和 Phase 6 动作提示词只消费同一已验证合同，禁止各自再次从动作名称推断身体运动。
- 对已知旧合同提供包含它的 Event/Storyboard Artifact 级严格零 Provider 并排迁移；无法从完整动作血缘与身体力学字段无歧义编译的父资产及其下游转为 audit-only/stale。
- 升级相关 schema、fingerprint、收据和恢复校验，并增加通用的左右肢体、朝向、翻转、旋转、组合招式及快节奏回归测试。
- **Non-goals**：不改变故事事件、动作数量、时长布局、角色身份、摄影机 owner、Provider 重试/预算策略；不把剧本专属动作表写入生产代码；不把实验 motion blueprint 作为普通 Phase 6 `reference_video`；本 change 不发起真实付费请求。

## Capabilities

### New Capabilities

- `canonical-action-kinematics`: 定义 canonical 动作的身体通道、朝向与空间变换合同，以及下游一致消费、迁移和可验证恢复行为。

### Modified Capabilities

无。相关 storyboard guide、pose atlas 与 motion blueprint 规格目前仍位于尚未归档的 change 中；本 change 以新的 `canonical-action-kinematics` 能力定义其共同上游合同和消费约束，不伪造不存在的 main-spec 修改目标。

## Impact

- **Pipeline / architecture**：Phase 1 仍拥有来源动作与身体力学事实；现有 body-action owner 增加来源层编译和 GAU/Pxx 生产投影。Phase 2、Phase 3、Phase 6 与隔离验收工具仅消费其版本化投影。Graph 拓扑、Lifecycle、普通 Phase 6 媒体类型和 `pipeline_core.py` 不变。
- **Schemas / persistence**：保持 Provider 理解 DTO 不变；升级 canonical body-action、Storyboard/Pxx、pose/atlas、动作板、Phase 6 brief、任务 fingerprint 和父 Artifact 迁移收据。未知未来版本继续 fail closed。
- **Provider / configuration**：不新增 Provider、模型、配置开关、重试或费用范围；回归和迁移均为零请求。
- **API / database / frontend-backend**：不适用；不修改外部 API、数据库 schema 或前端。
- **Tests / docs**：更新 Phase 1 动作合同、Phase 2 姿态语义、Phase 3 动作参考、Phase 6 Prompt/指纹、Graph/顺序恢复、旧版迁移测试及 `docs/HONCUT_ARCHITECTURE.md`。
