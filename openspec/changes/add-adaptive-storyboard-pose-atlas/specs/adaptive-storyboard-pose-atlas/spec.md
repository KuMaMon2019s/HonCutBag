## Purpose

让 HonCut 根据 4～15 秒分镜的实际动作密度、终态可读性、连续运镜和媒体预算，确定性生成可恢复的身份中立姿态图集，而不把姿态采样误当作必须逐格机械执行的独立剧情动作。

## ADDED Requirements

### Requirement: 分镜必须持久化动作时间合同
每个 Pxx MUST 将 `initial_anchor`、`story_action` 和 `terminal_hold` 分别建模。纯初始锚点不得占故事动作时长；动态动作 MUST 获得明确完成窗口；普通终态 MUST 允许有限的语义稳定时间，而不能因没有新增动作自动判定失败。

#### Scenario: 七秒动作在末段前完成
- **WHEN** 7 秒分镜的动态动作在持久化完成窗口内结束，并在剩余允许窗口保持与 canonical end state 一致的格挡或防御姿态
- **THEN** 系统将结果视为节奏合同内完成，不因末段没有新动作而阻断

#### Scenario: 初始锚点不偷占动作预算
- **WHEN** P01 的首帧已经建立纯戒备姿态且后续存在动态动作
- **THEN** 唯一初始锚点保持零故事时间，动态动作从首帧后立即开始

#### Scenario: 过长或错误终态
- **WHEN** 动作在允许完成窗口之前过早停止并形成过长停顿，或末段姿态与 canonical end state 的语义类别冲突
- **THEN** 系统报告节奏或终态偏差，且不得把错误姿态解释为合格稳定时间

### Requirement: Gxx 必须区分姿态采样与语义动作组
图集中的每个 Gxx MUST 是有序 pose sample；一个语义 action group MAY 由多个连续 pose samples 表达。每个 action group MUST 绑定当前 Pxx 的 canonical generation/source action lineage，且图集不得新增、重排或删除动作。

#### Scenario: 一个复杂动作使用多个采样
- **WHEN** 一个动作包含蓄力、峰值、接触和落位等连续阶段
- **THEN** 系统可以把多个连续 Gxx 绑定到同一 action group，并保存单调进度和唯一动作血缘

#### Scenario: 动作组跨越 Pxx
- **WHEN** 图集分组尝试把另一 Pxx 的动作或未来动作放入当前 Pxx
- **THEN** 系统在媒体生成前 fail closed

### Requirement: 图集容量必须由时长和 Provider 能力确定
系统 SHALL 使用版本化 Provider capability profile，根据有效动态时长确定 pose sample 和 reliable action-group 上限。4～15 秒范围内的规划 MUST 确定、可审计且与剧情词汇无关；超过可靠动作容量时 MUST 拆分 Pxx，而不是通过增加格子伪装可执行。

#### Scenario: 不同时长选择不同容量
- **WHEN** 相同动作密度分别规划为 4、7、10 和 15 秒 Pxx
- **THEN** 系统按同一能力配置确定相应的姿态采样与动作组容量，重复执行结果和哈希完全相同

#### Scenario: 超过可靠动作组上限
- **WHEN** 当前 Pxx 的 canonical 动作组数量超过该模型和时长的可靠上限
- **THEN** 上游布局必须拆分 Pxx，Phase 2 不得压缩、删除或重新编写动作以强行装入图集

### Requirement: 图集包装必须服从媒体预算与语义边界
Phase 2 SHALL 从同一个 canonical pose payload 零调用生成 `single_atlas` 与适用的 `paged_atlas` 候选；Phase 6 在冻结最终权威媒体后 MUST 确定性选择且只提交其中一种。动作密度高且媒体预算允许时 SHOULD 优先分页九宫格；只有一个导航图媒体位时 MAY 使用单张 18/27/36 格 atlas。候选之间不得改变动作、采样顺序、终态或摄影机连续性。

#### Scenario: 七秒高密度动作且有两个媒体位
- **WHEN** 7 秒连续动作超过单页高保真阈值且 Phase 6 最终媒体预算至少保留两个导航图位置
- **THEN** 系统输出两张有序九宫格，并把全部 pose samples 完整、无重叠地分配到页面

#### Scenario: 只剩一个导航图媒体位
- **WHEN** 身份板、首帧或前序视频、动作参考和连续性锚点使图像预算只剩一个位置
- **THEN** 系统输出一张符合 Provider 尺寸限制的密集 atlas，保持相同 action groups 和 pose-sample 顺序

#### Scenario: 必需媒体仍超限
- **WHEN** 任一合法包装策略仍使最终图片超过 Provider 上限
- **THEN** 系统在提交前 fail closed，禁止静默删除身份、动作或导航权威媒体

### Requirement: 运镜必须形成单一连续且时长可行的合同
每个连续 Pxx MUST 只使用一个由 Adaptation 选择的 primary camera technique。合同 MUST 保存机位高度、移动方向与速度、焦段变化、摇摄/俯仰角速度、分段和停顿；所有 Gxx 视角 MUST 是同一摄影机路径的连续投影，不得由 Phase 2 独立选择新运镜。

#### Scenario: 摇摄时长不足
- **WHEN** 运镜要求以 10 度每秒完成 90 度水平摇摄但 Pxx 只有 7 秒
- **THEN** 合同在付费前被判定不可执行并返回 Adaptation，Phase 2 不得静默改成 70 度或提高速度

#### Scenario: 分段式轨道运镜
- **WHEN** 运镜包含三段轨道移动和段间摄影机停顿
- **THEN** 合同把移动与停顿计入摄影机时长，而人物动作可在摄影机停顿期间继续，除非 canonical 动作合同明确要求人物停顿

#### Scenario: 多视角动作采样
- **WHEN** 连续摄影机路径从正面移动到侧面或三分之四背面
- **THEN** 对应 Gxx 的人体关节投影、遮挡和朝向随路径变化，而不只是蓝色箭头变化

### Requirement: 终态验证必须区分语义和精确模式
默认 `semantic_hold` MUST 验证 end-state 动作家族、支撑状态和主要交互关系，并允许非关键关节角度偏差。只有显式 `exact_pose` 才可要求专用终态参考图；该参考图 MUST 计入最终媒体预算和 fingerprint。

#### Scenario: 语义格挡可接受
- **WHEN** canonical end state 是格挡，生成结果保持稳定横向防御但与导航图的精确高位手臂角度不同
- **THEN** `semantic_hold` 可接受该偏差，不得要求额外参考图或自动补拍

#### Scenario: 精确终态占用媒体位
- **WHEN** canonical 合同显式要求 `exact_pose`
- **THEN** Phase 6 使用经验证的终态局部参考并在提交前计入九图预算；缺失或超限时 fail closed

### Requirement: 自适应图集必须版本化且可恢复
pose、guide、shot-storyboard、continuity、camera 和任务合同的当前版本、所有页面/采样/动作组、时间窗口、媒体职责、Prompt 与内容哈希 MUST 进入收据和 generation fingerprint。相同输入恢复 MUST 零 Provider 且输出稳定；未知未来版本 MUST fail closed。

#### Scenario: 十次恢复稳定
- **WHEN** 已生成的自适应图集和 continuity plan 在相同输入下恢复十次
- **THEN** 页面、Gxx/action-group 顺序、时间合同、图像哈希、任务 ID 和 fingerprint 均不变，Provider 请求数为零

#### Scenario: 旧版可验证迁移
- **WHEN** 单九宫格旧合同具有完整 canonical action lineage、Gxx 分配、源板、renderer 和内容哈希
- **THEN** 系统可零 Provider 生成并排的新版本合同和迁移收据，不覆盖旧文件

#### Scenario: 损坏或未来版本
- **WHEN** 旧合同血缘或哈希不完整，或 schema 高于当前支持版本
- **THEN** 旧资产只能 audit-only，并要求从对应 owner 重建或 fail closed
