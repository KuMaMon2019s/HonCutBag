## Context

现有 Phase 2 已能把 Pxx canonical action units 编译为固定九格的身份中立 pose contract，并能把 P01 的纯戒备压缩为零时长初始锚点。真实 7 秒实验提供了两个可复用结论：两页九宫格具有更好的动作/视角辨识度；单张 18 格 atlas 节省媒体位，但更适合作为运动包络而不是逐格执行表。实验还显示动态动作约在 5～6 秒完成、最后约 1 秒稳定在防御姿态是可接受结果。

Serena 定位的最小调用链为 Phase 2 `compile_pose_contracts` → `_narrative_grid_contract` / guide renderer → Phase 4 `GenerationChunk` → Runtime continuity provider → Phase 6 `build_video_prompt`。Graphify 的反向影响分析另外确认 Phase 2 纠偏/迁移、Phase 3 character-lock 刷新、Phase 5 重画、Graph/顺序入口及相关 acceptance tests 都会消费同一 owner。运镜事实仍由 Adaptation 和现有 camera-motion utilities 拥有；视频时长与媒体上限继续由 Runtime capability profile 拥有。

## Goals / Non-Goals

**Goals:**

- 将末段稳定姿态变成显式时间合同，而非需要用更多动作填满的“浪费”。
- 分离语义动作组与姿态采样，使高幅度复杂动作可用多个姿态连续表达。
- 根据时长、动作组数量和媒体余量确定性选择单页或分页图集。
- 使各格视角来自一个可验证的连续运镜路径。
- 保持所有新字段 JSON-safe、可 fingerprint、可迁移和零请求恢复。

**Non-Goals:**

- 不承诺生成模型精确复刻每个 Gxx。
- 不新增自动补拍、重试、QA 强门或 Provider 请求。
- 不改变 Adaptation、Phase 2、Phase 6、Runtime 的 ownership。
- 不把实测角色、剧情、动作名称或某次输出作为生产常量。

## Decisions

### 1. 使用 `action_group` 与 `pose_sample` 两级结构

Phase 2 先按当前 Pxx 的 canonical generation/source action units 形成有序 action groups，再为每组分配一个或多个 pose samples。组内样本通过单调 `pose_progress`、累计根位移和前组 transition origin 连续变化。

选择该结构是因为 Gxx 是动作可视化采样，不等同于 Provider 必须逐一执行的剧情事件。备选方案“每格一个动作”会在密集图集上夸大模型能力并破坏来源血缘；“只保留自由文本运动包络”则失去可验证性。

### 2. 时间合同采用完成窗口和有限终态稳定

能力配置提供 `terminal_hold_ratio=0.15`、最短 0.8 秒、最长 1.5 秒作为初始校准值；终态允许窗口按分镜时长推导，动态预算等于有效时长减去目标稳定时间。值属于版本化 Provider profile，不散落在 Phase Prompt 中，也不作为像素级绝对时码。

默认 `semantic_hold` 只锁动作家族、支撑与主要交互；`exact_pose` 才请求专用终态像素。这样接受可读的横向格挡，同时保留少数必须精准落位的镜头。备选方案“动作必须结束在最后一帧”对概率模型过度苛刻；完全不设终态合同则可能接受错误动作或过长静止。

### 3. 容量属于 Provider capability profile

扩展 `VideoModelCapabilities`，保存按持续时间计算 pose-sample 密度、reliable action-group 密度、单页高保真动作组阈值、支持的 atlas page sizes 和终态窗口。初始校准目标为：4 秒约 9 samples/4～6 groups，7 秒约 18 samples/7～10 groups，9～11 秒约 27 samples/10～15 groups，12～15 秒约 36 samples/14～20 groups。

这些是确定性规划上限而非 QA 对模型逐格复现的承诺。若 canonical action groups 超限，Phase 1 的布局/Adaptation owner 必须拆 Pxx；Phase 2 只报告不兼容，不重编动作。

### 4. 媒体余量决定单页或分页包装

Phase 2 在自己的生命周期位置无法知道 Phase 4 最终尾帧锚点，因此它从同一 canonical pose payload 零调用生成单页和适用的分页候选，而不是提前猜测最终媒体余量。包装器先冻结角色身份板、P01 首帧或 P02+ 前序视频、当前动作图、必要尾帧和 exact terminal reference，再由 Phase 6 从候选中选择一种：动作组超过单页高保真阈值且至少有两个空位时选 `paged_atlas`；只有一个空位时选 `single_atlas`。页面固定使用最多 6000px 边长、合法宽高比和确定性行列布局。

Phase 2 生成所有候选页面语义与像素；Phase 6 只选择并按 manifest 顺序消费一个候选，不能重新绘图或改变分组。备选方案“永远单张大图”节省媒体但降低语义保真；“永远分页”会在多角色镜头中挤爆九图上限；“Phase 6 临时调用 Phase 2 renderer”会破坏依赖方向和恢复语义。

### 5. 运镜目录扩展但选择权不下沉

现有 camera-motion contract 扩展为受控技术参数：机位高度、轨道位移/速度、焦段起止、pan/tilt 角度与速度、分段数和停顿。支持推拉、固定机位变焦、分段轨道、水平摇摄和高低机位旋摇等通用类别。

Adaptation 根据镜头时长选择并解析完整合同；可行性验证使用位移、角度、速度和停顿计算最小时长。Phase 2 只把同一合同的连续采样投影为各格 view transform，不能修改参数。无法执行时返回 Adaptation，而不是静默降速、减角或换运镜。

### 6. 版本化合同沿既有路径传播

升级 pose contract、guide、shot storyboard manifest 和 continuity schema；`GenerationChunk` 保存页面列表、页面/格顺序、action groups、pose samples、完成窗口、terminal mode 和 camera contract hash。Phase 6 Prompt 仅描述动作应在窗口内自然完成并稳定保持终态，不输出逐格秒点；任务 fingerprint 纳入全部新字段与媒体顺序。

旧单页九宫格只有在其动作血缘、分配、renderer 和哈希完整时才能并排迁移；未知未来版本和不完整合同保持 fail closed。Graph 与顺序执行器通过共同 Phase owner 自动获得同一行为，不增加节点逻辑。

### 7. 演员别名只在 Phase 2 边界做确定性投影

Phase 2 从当前 Pxx 的 canonical instance ID 出发，只吸收角色资产中的显示名/已登记 aliases/source mentions，以及 shot/beat `participant_refs` 中明确绑定该 instance 的 mention。别名映射冲突直接阻断；没有身份血缘的相似文本不参与归并。pose compiler 将匹配到的 performer 规范为 canonical actor role，使普通九宫格与 adaptive atlas 共用同一映射，而不把身份推断下沉到 renderer 或 Phase 6。

像素回归必须在排除标签、动作箭头和运镜箭头的演员区域比较实际栅格，证明关节/根位移变化确实进入图像；仅比较 JSON fingerprint 不能满足该门禁。

## Risks / Trade-offs

- [单张高密度 atlas 仍可能被模型忽略部分姿态] → 将其定义为运动包络；媒体允许时优先分页，并以 action-group 语义而非逐格像素复现验收。
- [动作容量经验值可能随模型版本变化] → 所有值归版本化 capability profile；后续真实验收只调整 profile，不修改 Phase 算法或写剧情特例。
- [分页消耗图片预算] → 包装前先冻结全部权威媒体，预算不足时确定性回退单页；仍超限则付费前阻断。
- [终态稳定可能退回错误的早期姿态] → `terminal_hold` 必须绑定 canonical end-state action group，Prompt 禁止回放初始锚点；exact 模式可使用专用终态参考。
- [复杂运镜与高密度动作相互争夺模型注意力] → 每个 Pxx 仅一个 primary technique，时长不可行或真实硬切时由 Adaptation 拆 Pxx。
- [跨合同升级影响恢复] → 并排迁移、保留旧资产 audit-only、未知版本 fail closed，并用十轮 Provider-deny 恢复验证。

## Migration Plan

1. 先加入能力配置、纯时间/容量规划 DTO 与单元测试，不改变现有 v4 产物。
2. 升级 Phase 2 pose/action-group 和单页/分页 renderer，写新版本 sidecar；旧资产不覆盖。
3. 升级 continuity 和 Phase 6 消费/fingerprint，保持 bridge 与其他媒体职责不变。
4. 更新旧版迁移、架构文档和 acceptance 预检；运行完整零请求回归后才允许生成 paid-admission 预检。
5. 回滚时撤销新版本消费者；新资产保留审计，旧 v4 生产路径由上一提交继续读取。

## Open Questions

- 不同 Seedance 模型/分辨率的 action-group 密度需要未来独立的真实样本校准；本 change 只固化当前 480p Seedance 2.x 的初始 profile 和安全扩展接口。
