## Purpose

为 HonCut 的 canonical 人体动作建立可恢复、可审计且与剧情词汇无关的运动学事实合同，使身体各部位、人物朝向、翻转与旋转能被所有视觉阶段一致地编译成幅度明确、节奏紧凑的组合动作。

## ADDED Requirements

### Requirement: 运动学必须先绑定来源动作，再投影到最终生产动作单元
Phase 1 body-action owner MUST 先为每个 `micro_action_index` 编译版本化的来源层运动学事实。在 generation action units 与 Pxx 最终确定后，同一 owner MUST 生成生产层 `kinematics_projection`。每个投影 MUST 保留 performer、source event、全部 `source_micro_action_indexes`、generation action unit、时间片和 Pxx 血缘；同一 Pxx 内的来源索引覆盖 MUST 完整、无重叠，且不得引用其他 Pxx。运动学细化 MUST NOT 增加、删除、重复、重排或跨 Pxx 借用剧情动作。

#### Scenario: 复杂动作被细化而不扩写剧情
- **WHEN** 一个 canonical action 表达一次攻击、闪避、格挡、翻转或旋转
- **THEN** 系统为同一个 generation action unit 写入多个有序身体通道阶段，且 action-group 数量与来源血缘保持不变

#### Scenario: 最终 GAU 形成后才绑定生产投影
- **WHEN** Adaptation 或 Storyboard 将一个或多个来源 micro-actions 归入最终 GAU/Pxx
- **THEN** 生产投影精确列出该 GAU 的 `source_micro_action_indexes`，其并集等于 Pxx 的来源动作集合，交集为空，且任何早期来源层记录都不得伪装成最终 GAU

#### Scenario: 运动学子阶段不得冒充新动作
- **WHEN** 一个招式被拆成蓄力、发力、峰值、接触、随动和落位
- **THEN** 这些阶段全部绑定原 action unit，容量与故事时钟仍只计算一个 canonical 动作

#### Scenario: 同步多演员动作保持独立轨道
- **WHEN** 一个 GAU 同时描述攻击者与防御者、抓控双方或其他多个 performer
- **THEN** 投影包含按稳定 performer ID 排序的 `actor_tracks`，每个演员独立保存自己的动作通道和来源血缘，不得合并成一个人体骨架

### Requirement: 人体必须按完整双侧身体通道表达
每条 `actor_track` MUST 分别表达左臂、左手、右臂、右手、左腿、左脚、右腿、右脚、腰/躯干、头部和身体根节点。每个通道 MUST 声明有序阶段中的语义状态、量化且归一化的平移/旋转或关节目标、幅度等级以及支撑/接触职责。没有来源或既有身体力学证据的自由度 MUST 使用确定性的 `inherit`、`stabilize`、`support` 或 `balance` 渲染状态，不得因 `unspecified` 而缺失几何，也不得猜造成剧情事实。

#### Scenario: 左右肢体协同招式
- **WHEN** canonical 身体力学声明左手控制、右腿发力、腰部旋转且左脚支撑
- **THEN** 四个相关通道分别保存自己的轨迹和时序，其他通道保存明确的平衡或未指定状态，渲染结果不得把左右侧互换

#### Scenario: 腿与脚不可合并
- **WHEN** 一个动作同时包含抬腿和脚掌蹬地、点地、勾脚或落脚
- **THEN** 腿部关节链与脚部朝向/接地必须是两个可独立验证的通道

#### Scenario: 头与腰参与组合动作
- **WHEN** 动作包含观察目标、闪避、转身或全身发力
- **THEN** 头部朝向与腰/躯干旋转分别持久化，并与根节点、支撑腿和主要攻击/防御肢体保持时间一致

#### Scenario: 语义 token 必须具有可渲染数值几何
- **WHEN** 一个通道被标记为主动、支撑、平衡或继承
- **THEN** 下游可直接读取固定精度的归一化关节/根节点几何进行渲染，而不需要重新解释 technique、自然语言或动作模板

#### Scenario: 全身大幅动作满足通用幅度下限
- **WHEN** canonical 身体力学要求攻击、闪避、转体、翻转或显著位移
- **THEN** 根节点、腰/躯干和至少一个主要肢体链的量化变化达到由动作类别决定的通用最小幅度；普通静止或短暂戒备不被错误放大

### Requirement: 人物朝向必须使用 Pxx 局部锚点并独立于摄影机运动
每条演员轨道 MUST 以 Pxx 起始姿态为局部 yaw `0` 锚点，保存演员相对该锚点的连续旋转。正面、背面、左侧面、右侧面及三分之四只作为可验证的摄影机相对视图投影；只有 canonical camera contract 足以计算时才写入，否则使用受控未知值。合同 MUST NOT 伪造不存在的绝对世界坐标朝向，Actor orientation 也 MUST NOT 由蓝色运镜箭头、画面镜像或当前渲染视角替代。

#### Scenario: 正面转到侧面再到背面
- **WHEN** canonical 动作要求人物连续转身
- **THEN** 有序关键阶段分别记录正面、侧面和背面关系，且相邻阶段的旋转方向连续

#### Scenario: 摄影机绕行但人物不转身
- **WHEN** canonical 相机路径从人物正面移动到背面，而演员相对于 Pxx 起始锚点的 yaw 不变
- **THEN** 系统只改变相机相对视图，不伪造演员腰、脚或根节点旋转

#### Scenario: 左右侧面不可含糊
- **WHEN** 来源或现有 body mechanics 明确左侧或右侧
- **THEN** 合同保存精确侧向；无法证明侧别时使用受控未知值，不得随机选择或在恢复时漂移

### Requirement: 翻转与旋转必须使用受控空间变换
翻转与旋转 MUST 分别保存变换类型、轴线、方向、角度或圈数、相对起止朝向、根节点轨迹、支撑释放/恢复和落地状态。角度或圈数只有在来源或既有 canonical 事实明确时才能作为精确事实；否则 MUST 使用由已知变换类别决定的受控区间或 `unspecified`，不得猜造精确数值。只有来源动作或已验证 canonical 身体力学明确要求对应变换时才能生成；普通转身、躲闪、运镜或画面镜像 MUST NOT 被提升为翻转。

#### Scenario: 前翻和后翻
- **WHEN** canonical 动作明确要求前翻或后翻
- **THEN** 合同使用 pitch 轴与对应方向，包含起跳、倒置、越顶和落地阶段，并保持左右肢体与支撑关系

#### Scenario: 水平旋转
- **WHEN** canonical 动作明确要求转体或旋转攻击
- **THEN** 合同使用 yaw 轴、稳定方向和角度/圈数，腰、头、根节点与支撑脚按阶段协调运动

#### Scenario: 运镜不得伪造人物旋转
- **WHEN** 只有摄影机旋摇而 canonical 演员动作没有翻转或旋转
- **THEN** 人物变换合同保持 `none`，Phase 2 和 Phase 6 不得加入翻身或转体动作

### Requirement: 运动学阶段必须形成连续且可执行的组合招式
同一 Pxx 内的 canonical action groups MUST 通过前一组终态到后一组起态的连续身体状态连接。动态阶段 MUST 具有可辨识的蓄力、发力峰值与随动/落位；不得在动作之间回到默认中立或戒备姿态，也不得用长时间保持姿势替代剩余组合动作。

#### Scenario: 多招连续执行
- **WHEN** 一个 Pxx 含按序排列的多个 canonical action groups
- **THEN** 后一组从前一组的根位置、支撑脚、腰部朝向、头部目标和肢体终态继续，所有动作按 canonical 顺序连续执行

#### Scenario: 大幅度动作
- **WHEN** canonical technique 与身体力学要求全身位移、攻击、闪避、翻转或旋转
- **THEN** 运动学合同至少在根节点、腰/躯干和一个主要肢体链上产生非零的大幅度阶段变化，而不能只移动末端手脚或箭头

#### Scenario: 快节奏不删除身体过程
- **WHEN** 多个动作需要在较短 Pxx 时长内完成
- **THEN** 系统压缩各阶段持续时间但保留关键身体通道、峰值、接触和落位，不以省略招式或延长静态准备姿态实现节奏

### Requirement: 下游必须消费同一已验证运动学事实
Phase 2 姿态导航、Phase 3 动作参考、Phase 6 动作 Prompt/任务 fingerprint 和零费用 motion blueprint MUST 消费同一合同版本与哈希。下游 MUST NOT 再从自由动作文本、技术名称、箭头方向或固定 technique 模板独立推断身体通道。

#### Scenario: 导航图真实改变姿态
- **WHEN** 相邻 Gxx 绑定不同运动学阶段
- **THEN** 实际人体关节、根位移、腰/头朝向或视图投影产生可测差异，不能只有红/蓝箭头变化

#### Scenario: 动作参考与视频指令一致
- **WHEN** Phase 3 和 Phase 6 消费同一 Pxx
- **THEN** 动作参考的通道/朝向与 Phase 6 有序动作投影一致，Prompt 不得添加合同之外的翻转、旋转或肢体动作

#### Scenario: 验收蓝图不得维护平行动作库
- **WHEN** motion blueprint 编译当前 Pxx
- **THEN** 它从生产运动学合同获得关键阶段，不得用独立 technique registry 产生与生产不同的身体动作

#### Scenario: 验收蓝图不得进入普通视频请求
- **WHEN** Phase 6 组装普通生产视频请求的媒体清单
- **THEN** motion blueprint 仍只属于隔离能力门，不得作为 `reference_video` 或其他媒体角色进入请求，除非其独立 OpenSpec 能力门未来通过并完成单独激活变更

### Requirement: 运动学编译不得扩大模型输出合同
本能力 MUST 保持现有 `BodyActionUnderstanding` Provider DTO、JSON schema 和模型请求数量不变。所有来源层运动学及最终 GAU/Pxx 投影 MUST 由纯代码根据已验证的身体力学字段、来源索引和生产分组编译；不得要求 Event Extractor 返回逐关节时间线或新增一次 LLM 修补调用。

#### Scenario: 现有模型 Observation 足以编译
- **WHEN** `BodyActionUnderstanding` 已提供 technique、side、limbs、footwork、torso、weight_shift、direction、contact 与 end_pose
- **THEN** 代码在零新增 Provider 请求的情况下生成运动学合同，并保持 Provider DTO schema 字节语义不变

### Requirement: 运动学结构校验必须严格但不建立概率性语义门禁
Schema、ID、hash、lineage、左右通道枚举、阶段顺序、时间边界和跨 Pxx 归属错误 MUST fail closed。动作美感、自然度或未明确自由度的概率性判断 MUST 作为诊断或 `unspecified` 保存，不得触发 Provider 重试、自动改写或补造动作。

#### Scenario: 确定性结构损坏
- **WHEN** 通道引用未知 actor/action、左右枚举非法、阶段时间逆序、hash 不匹配或动作来自另一 Pxx
- **THEN** 系统在图片、上传或视频付费边界前阻断

#### Scenario: 未明确的次要关节
- **WHEN** 来源与现有 canonical body mechanics 没有说明某个次要关节的精确角度
- **THEN** 系统保存未指定或平衡职责并继续，不以 LLM 置信度不足否定整个动作

### Requirement: 旧动作资产只能严格零请求迁移或隔离
已知旧合同只有在 canonical action、performer、Pxx、source hashes、body mechanics 与全部父血缘可验证时，才可在本地编译当前运动学 sidecar 并挂接到包含该合同的 Event/Storyboard Artifact；原父资产 MUST 保持不变并转为 audit-only。由旧父资产派生的下游资产 MUST 标记 stale/audit-only。语义不足、证据损坏或未来版本 MUST fail closed，并要求从原 owner 重建。

#### Scenario: 完整旧身体力学可迁移
- **WHEN** 旧 action 的 limbs、footwork、torso、weight shift、direction、contact、end pose 和来源血缘全部有效
- **THEN** 系统零 Provider 编译当前通道 sidecar、写绑定父 Artifact 的独立迁移收据并保持原资产不变

#### Scenario: 旧动作无法证明翻转或侧别
- **WHEN** 旧自由文本不足以无歧义证明翻转、旋转方向或左右侧
- **THEN** 迁移结果不得猜造这些事实；若当前合同的必要结构无法满足，则将旧资产标记 audit-only 并停止生产复用

#### Scenario: 恢复稳定
- **WHEN** 同一运动学合同及其下游姿态产物连续恢复十次
- **THEN** 合同 hash、姿态指纹、媒体 hash、任务 fingerprint 与 action-group 数量完全一致，Provider 请求数为零
