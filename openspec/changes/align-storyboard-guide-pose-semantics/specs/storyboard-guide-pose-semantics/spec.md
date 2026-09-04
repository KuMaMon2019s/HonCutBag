## Purpose

确保 Phase 2 身份中立剧情导航图中的骨架姿态真实表达各 Gxx 已绑定的动作、阶段与交互关系，同时保持零 Provider、无角色身份像素和可恢复的确定性产物。

## ADDED Requirements

### Requirement: Gxx 必须绑定 canonical 动作语义
每个进入剧情导航图的 Gxx MUST 持久化其有序动作绑定，包括所属 Sxx/Pxx、动作阶段、canonical generation action unit、source action unit、执行者、目标以及可用的身体动作拍。导航图不得仅根据自由文本或文本哈希决定姿态。

#### Scenario: 多动作按来源顺序分配
- **WHEN** 一个 Pxx 获得多个 Gxx 且包含多个 canonical 动作单元
- **THEN** 系统按 canonical 顺序将动作单元确定性分配给 Gxx，并在收据中保存完整有序血缘

#### Scenario: 动作血缘不完整
- **WHEN** Gxx 无法唯一绑定到当前 Pxx 的 canonical 动作单元或其来源索引冲突
- **THEN** 系统在生成或消费导航图前 fail closed，且不得根据动作描述猜测新血缘

### Requirement: 骨架姿态必须表达当前格动作
身份中立 renderer SHALL 根据受控的动作与身体力学字段绘制关节、躯干、重心、朝向和接触关系。存在可执行动作变化时，相邻 Gxx 的姿态指纹 MUST 随动作或阶段变化；箭头变化不得替代骨架姿态变化。

#### Scenario: 起始、执行和终态
- **WHEN** 三个连续 Gxx 分别绑定同一动作的 start、action_progress 和 end 阶段
- **THEN** 三格以可辨识且来源一致的准备姿态、执行姿态和终止姿态表现动作进程

#### Scenario: 不同身体动作
- **WHEN** Gxx 分别绑定移动、闪避、格挡、踢击、抓控或持用道具等不同动作
- **THEN** 对应骨架的关节、躯干或重心至少一项发生与动作相符的结构变化，并生成不同姿态指纹

#### Scenario: 否定的接触描述不得覆盖正向动作
- **WHEN** technique、footwork、torso、weight shift、end pose 或 canonical action 明确描述闪避、挥击等动作，而 contact 说明“无格挡”“未击中”“without blocking”或同类否定事实
- **THEN** 分类器必须采用正向动作证据，持久化被排除的否定匹配，且不得把否定词误判成格挡或攻击

#### Scenario: 身体动作拍只绑定匹配的动作单元
- **WHEN** 一个 Pxx 含多个动作单元，而身体动作拍只通过 source micro-action index 对应其中一项
- **THEN** 该动作拍的技术、步法、躯干、重心和接触字段只能影响匹配的动作单元，不得漂移到同一 Pxx 的其他动作

#### Scenario: 多格展开形成连续身体进度
- **WHEN** 同一动作单元被展开到多个 Gxx
- **THEN** 每格必须持久化单调且互异的动作进度，并使根位移、躯干、重心、步幅或关节产生达到确定性最小位移的连续变化

#### Scenario: 连续动作不得回到中立姿态重置
- **WHEN** 一个 Pxx 的相邻 Gxx 从一个 canonical action unit 进入下一个 action unit
- **THEN** 后一动作必须从前一动作的最终关节与累计根位置开始，并持久化前一 action unit 的 transition origin，不得在动作边界瞬间重置为默认站姿

#### Scenario: 多人交互
- **WHEN** 当前动作具有多个执行者或明确目标与接触关系
- **THEN** 导航图使用相应数量的中性骨架表达角色位置、相对朝向和接触关系，不把多个角色压缩成一个站立骨架

#### Scenario: 非身体事件
- **WHEN** 当前 Gxx 只描述环境、车辆、光线或其他无人物身体动作的事件
- **THEN** 系统使用中性空间/对象占位和关系标识，不伪造人物动作

### Requirement: 姿态、动作箭头与运镜箭头必须一致
红色箭头 MUST 表达当前动作绑定中的主体或物体运动方向，并与骨架朝向、重心和动作阶段一致；蓝色箭头 MUST 只表达 canonical 运镜，不得由动作文本哈希随机生成。

#### Scenario: 相反方向动作
- **WHEN** 两个 Gxx 的 canonical 动作方向相反
- **THEN** 骨架朝向/重心和红色动作箭头同时反映相反方向，而蓝色运镜箭头仅随运镜合同变化

### Requirement: 导航图必须保持身份中立
导航图 MUST 只使用固定灰色中性骨架、Gxx 序号、红色动作箭头、蓝色运镜箭头和抽象空间/视线/交互标识。它 MUST NOT 复制审核九宫格像素，也不得包含可识别的人脸、发型、服装纹理、道具外观或剧情文字。

#### Scenario: 身份像素隔离
- **WHEN** 系统从审核九宫格和结构化合同派生导航图
- **THEN** 收据记录 `source_pixel_usage=none`，且像素级检查找不到源九宫格人物或道具外观的复用

### Requirement: 产物必须版本化、可验证并可恢复
导航图 schema、renderer、语义 payload、姿态指纹、源审核板哈希和父动作血缘 MUST 进入 Artifact 与收据校验。相同输入重复生成或恢复 MUST 得到相同语义 payload、姿态指纹、图像哈希和零 Provider 请求。

#### Scenario: 冷启动与恢复稳定
- **WHEN** 同一 Pxx 合同被冷启动生成一次并恢复十次
- **THEN** 每次的 Gxx 顺序、姿态指纹、语义哈希和图像哈希完全一致，Provider 请求数为零

#### Scenario: 旧版可验证重绘
- **WHEN** 旧导航图的源九宫格、Gxx→Pxx 分配、canonical 动作血缘和所有哈希完整可验证
- **THEN** 系统可在不覆盖旧文件且不调用 Provider 的前提下生成当前版本导航图和独立迁移收据

#### Scenario: 旧版不可验证或未来版本
- **WHEN** 旧导航图缺失动作血缘、哈希损坏、职责冲突或版本高于当前支持版本
- **THEN** 系统将旧资产隔离为 audit-only，并要求从 Phase 2 重建或 fail closed

### Requirement: 下游只能消费已验证的当前动作导航图
Phase 4、Phase 5 和 Phase 6 SHALL 只接受与当前 Pxx、canonical hash、Gxx 顺序和当前 renderer 版本完全匹配的导航图。导航图不得冒充首帧、角色身份板或动作参考板，跨 Sxx bridge 不得消费导航图。

#### Scenario: Phase 6 当前 Pxx 消费
- **WHEN** Phase 6 构建一个 Pxx 的最终媒体清单
- **THEN** 它只注入该 Pxx 的已验证导航图，保持既有媒体职责与编号顺序，并将导航图版本、Gxx、姿态指纹和哈希纳入任务 fingerprint

#### Scenario: 版本或 Pxx 漂移
- **WHEN** 导航图属于另一 Pxx、包含错误 Gxx、使用旧 renderer 或 canonical hash 不匹配
- **THEN** 下游在任何视频提交前拒绝该资产
