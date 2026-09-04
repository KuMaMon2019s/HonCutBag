## Why

Phase 2 的身份中立 `storyboard_guides` 当前只根据格子文本改变箭头方向，火柴人的关节与重心始终是同一站姿。这会让 Phase 6 收到与 Gxx 动作描述不一致的剧情导航证据，削弱动作顺序、攻防关系和空间演绎的约束。

## What Changes

- 为每个 Gxx 增加来源绑定的结构化姿态语义：动作单元、身体动作拍、执行者/目标、阶段、方向、接触与终态。
- 将每个 Pxx 获得的 Gxx 按 canonical 动作顺序确定性分配；不得根据自由文本哈希随机选择姿态。
- 当且仅当 P01 的首个 canonical 动作是纯戒备姿态、后面存在动态动作且 Phase 4 成片首帧已经建立该姿态时，将其压缩为一个 `t=0` 零时长初始锚点；首帧后必须立即进入下一动态动作，不能让准备姿态消耗成片叙事时间。
- 将身份中立 renderer 升级为可绘制不同关节、躯干、重心和多人交互的确定性骨架 renderer；红色动作箭头必须与姿态运动方向一致，蓝色箭头仍只表达运镜。
- 导航图继续只使用灰色中性骨架和抽象关系标识，不复制审核九宫格像素，不包含脸、发型、服装纹理、道具外观或剧情文字。
- 版本化升级导航图语义合同、renderer、收据和哈希；旧版只有在完整 canonical 动作血缘可验证时才能零 Provider 重绘，否则标记 audit-only 并要求从 Phase 2 重建。
- 增加关节级、血缘级和下游媒体合同回归，确保不同动作产生不同姿态、相同输入恢复时像素与哈希稳定。

Non-goals:

- 不让 Phase 2 重新编剧情、调用 LLM/VLM/图像 Provider 或推断新的角色身份。
- 不改变 Phase 4 首帧、Phase 3 动作参考板、Phase 6 媒体优先级、Graph 拓扑、Provider 重试或任务账本。
- 不删除、改写或重排 Phase 1 的来源动作；单独存在、没有后续动态动作的戒备仍是正常有时长动作。
- 不修复 run-17 的 Phase 3 `inconsistent source action-unit lineage`；该失败属于独立 owner，run-17 继续 audit-only。
- 不向 `pipeline_core.py` 增加任何生产逻辑。

## Capabilities

### New Capabilities

- `storyboard-guide-pose-semantics`: 定义 Gxx 动作血缘、身份中立姿态渲染、版本迁移及下游消费的可验证行为。

### Modified Capabilities

无。

## Impact

- Pipeline：修改 Phase 2 的九宫格语义绑定、本地导航图 renderer、Artifact 校验和确定性迁移；Phase 4/5/6 只适配新的版本化字段并保持现有职责。
- API：不涉及外部 HTTP API；内部 storyboard guide schema/renderer 版本会升级。
- Database：不涉及数据库 schema 或任务账本。
- Frontend/backend：不适用；HonCut 当前为本地 pipeline。
- Configuration：不新增生产开关或环境变量。
- Architecture：不改变 owner 或依赖方向；属于跨文件的行为与内部合同变更，不是 Architecture Change。架构文档只需同步版本化合同说明。
- Provider/cost：生产修复本身为本地零调用 renderer；回归阶段禁止真实 Provider 请求，后续真实门仍需单独授权。
