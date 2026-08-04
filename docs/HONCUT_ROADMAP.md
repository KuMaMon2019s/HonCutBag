# HonCut 改造路线图

融合 Toonflow + OpenMontage 精华 | 2026-07-31
原则: 只增不删，只扩不改 — 在现有 8 Phase 基础上叠加能力，不重构不重写

## 一、现有架构（不动）

| Phase | 名称 | 模块链 |
|-------|------|--------|
| Phase 2 | 编剧引擎 | text_parser → event_extractor → character_discoverer → adaptation_engine → storyboard_generator |
| Phase 2.5 | 故事板 | seedream_client → storyboard.png |
| Phase 3 | 角色工厂 | character_factory → 2×2 grid → crop → front/side/back.png |
| Phase 4 | 编排器 | orchestrator → shots/S01-S10/SHOT_META.json |
| Phase 5 | 视频生成 | seedance_client (TOS + reference_image + 429退避) |
| Phase 6 | 一致性守卫 | consistency_guard + quality_gate |
| Phase 7 | 组装引擎 | edit_decisions (trim + normalize + smart_transition + xfade) |
| Phase 8 | 后期处理 | audio_pipeline → visual_post → rhythm_editor → final_encode |

已有模块（25个，约9600行）全部保留，不做任何删改。

## 二、改造总览（6 个增量模块）

| # | 增量模块 | 学谁 | 插入位置 | 方式 |
|---|---------|------|---------|------|
| M1 | 导演规划层 | Toonflow 阶段1 | Phase 2 之前新增 Phase 1 | 新增 director_planner.py |
| M2 | 分镜图序列 | Toonflow 阶段6 | Phase 2.5 扩展 | 扩展 seedream_client 调用 |
| M3 | 片段间过渡桥梁 | Toonflow 分镜表 | Phase 2 adaptation_engine 扩展 | 扩展 LLM prompt |
| M4 | 模型路由 | Toonflow 视频提示词 | Phase 5 之前新增路由层 | 新增 prompt_router.py |
| M5 | 监督层审核 | Toonflow 监督层 | Phase 6 扩展 | 扩展 quality_gate.py |
| M6 | 产物链 + Checkpoint | OpenMontage | 全流程叠加 | 新增 artifact_chain.py |

## 三、M1 导演规划层（学 Toonflow 阶段1）

目标：在 Phase 2 之前新增 Phase 1，产出结构化的导演规划，让下游分镜有情绪依据和一致性锚点。

新增文件：pipeline/src/director_planner.py → 输出 director_plan.json

| 字段 | 说明 |
|------|------|
| scenes scene_id | 场次编号 Sc1, Sc2 |
| scenes scene_name | 场景名（地点+概况） |
| scenes dialogue_count | 台词条数 |
| scenes dialogue_words | 台词总字数 |
| scenes emotion_intensity | 情绪浓度 0-10 |
| scenes emotion_arc | 情绪弧线 X到Y |
| notes emotional_peak | 关键情感砸点 |
| notes consistency_anchors | 视觉一致性锚点 |
| notes spatial | 空间与距离 |
| notes ambient_sound | 环境音提示 |
| notes pitfall | 易错提示 |
| scene_transitions | 场间过渡（4种桥梁） |

实现方式：调用 LLM（doubao-seed-2.0-lite），prompt 学 Toonflow 导演规划。只做四件事：拆分场、台词统计、情绪分析、过渡设计。不规划光影、色调、配乐。输出纯结构化 JSON，不写创作叙述。

与现有代码的关系：不改 pipeline_runner.py 的 Phase 2 逻辑。在 run_pipeline() 的 Phase 2 之前插入 run_phase1() 调用。director_plan.json 作为额外输入传给 Phase 2 的 adaptation_engine。

### 场间过渡 4 种桥梁（学 Toonflow）

| 桥梁 | 触发条件 | 做法 |
|------|---------|------|
| 动作桥梁 | 同一组人物连续动作 | 前段结尾=动作起始态，后段首镜=进行时或完成时 |
| 情绪接力 | 对话或冲突情绪延续 | 前段结尾用反应镜或微表情铺垫，后段承接强化 |
| 空间视线 | 场景切换或视线转移 | 空镜+视线引导+声音延续 |
| 台词黏合 | 台词或音效需画面回应 | 前段末尾声音延续到后段首镜 |

## 四、M2 分镜图序列（学 Toonflow 阶段6）

目标：Phase 2.5 从生成 1 张 storyboard.png 扩展为每个镜头一张分镜图。

改动方式：不改 seedream_client.py，在 pipeline_runner.py 的 run_phase2_5() 中扩展调用。

输出结构：
- storyboard.png（保留，总览）
- storyboard_images/S01.png 到 S10.png（新增，每镜头一张）

与 Phase 5 的联动：Phase 5 生成视频时，把对应镜头的分镜图作为构图参考（reference_image），优先级低于角色参考图。

## 五、M3 片段间过渡桥梁（学 Toonflow 分镜表）

目标：在 adaptation_engine 的 LLM prompt 中增加片段间过渡设计规则。

改动方式：不改 adaptation_engine.py 的代码逻辑，只扩展 LLM prompt 内容。

追加的过渡规则（学 Toonflow 原文）：
1. 动作桥梁：前段结尾=动作起始态，后段首镜=进行时或完成时
2. 情绪接力：前段结尾用反应镜或微表情铺垫，后段承接强化
3. 空间视线：场景切换时用空镜+视线引导+声音延续
4. 台词黏合：前段末尾声音延续到后段首镜

铁律优先级（学 Toonflow 原文）：台词零删改 > 出场人物完整 > 只描述动作状态 > 长台词拆镜

与现有代码的关系：现有 adaptation_engine.py 已有承接上镜规则（line 70），只需扩展 prompt 内容。不改代码逻辑，只改 prompt 字符串。

## 六、M4 模型路由（学 Toonflow 4 种提示词模式）

目标：Phase 5 之前新增提示词路由层，按模型名自动匹配提示词格式。

新增文件：pipeline/src/prompt_router.py

4 种模式（学 Toonflow 原文）：

| 模式 | 触发条件 | 提示词格式 |
|------|---------|-----------|
| Seedance 2.0 多分镜 | 模型=seedance-2-0 + 多分镜 | 中文结构化12维编码 + 图N引用 + 毫秒时长 |
| Seedance 2.0 单镜头 | 模型=seedance-2-0 + 单镜头 | reference_image + 英文 prompt |
| 通用首尾帧 | 其他模型 + 首尾帧 | Visual/Motion/Camera/Audio/Narrative 五维度 |
| Wan 2.6 | 模型=wan2.6 | 叙事式英文（风格到主体到光线到镜头） |

与现有代码的关系：不改 seedance_client.py。在 pipeline_runner.py 的 _run_phase5_fallback() 中，构建 prompt 之前调用 prompt_router.route_prompt()。现有的 style_prefix 和 reference_image 注入逻辑保留。

## 七、M5 监督层审核（学 Toonflow 监督层）

目标：扩展 quality_gate.py，增加 Toonflow 的 4 条红线审核和 A/B/C/D 评分。

改动方式：不改现有 quality_gate.py 的 run_quality_check()，新增函数 run_storyboard_review()。

4 条红线（违反即严重）：

| 红线 | 内容 |
|------|------|
| R1 资产引用合法 | 角色和场景必须在 characters 中存在 |
| R2 剧本忠实 | 台词一字不差，不遗漏不新增 |
| R3 具象可感 | 禁止抽象笼统词，声音具体到声源 |
| R4 父子资产正确 | 衍生状态用衍生 ID，不主和衍生同存 |

评分标准：

| 评分 | 严重问题 | 中等问题 |
|------|---------|---------|
| A 可直接使用 | 0 | ≤2 |
| B 小修后可用 | 0 | ≤5 |
| C 需较大修改 | 1-2 | 不限 |
| D 建议重做 | ≥3 | 不限 |

与现有代码的关系：现有 run_quality_check() 检查文件存在性（Phase 级别），新增 run_storyboard_review() 检查内容质量（分镜级别）。两个函数互补，不冲突。在 run_phase2() 末尾调用 run_storyboard_review()。

## 八、M6 产物链 + Checkpoint（学 OpenMontage）

目标：每阶段产出结构化 JSON 产物，支持从任意阶段恢复。

新增文件：pipeline/src/artifact_chain.py

产物链定义：

| Phase | 产出 | 依赖 |
|-------|------|------|
| phase1 | director_plan.json | 无 |
| phase2 | events.json + characters.json + storyboard.json | director_plan.json |
| phase2_5 | storyboard_images/ | storyboard.json |
| phase3 | characters/ | characters.json |
| phase4 | shots/ | storyboard.json |
| phase5 | shots/S*/output.mp4 | shots/ |
| phase6 | quality_report.json | shots/S*/output.mp4 |
| phase7 | edit_decisions.json + raw_assembly.mp4 | shots/S*/output.mp4 |
| phase8 | polished.mp4 + render_report.json | raw_assembly.mp4 |

Checkpoint 增强：每阶段完成后写 checkpoint_phaseN.json。支持 --resume-from phase5 从任意阶段恢复。现有 LangGraph checkpoint 保留，新增文件级 checkpoint 互补。

## 九、实施顺序

| 批次 | 模块 | 依赖 | 复杂度 |
|------|------|------|--------|
| 第1批 | M2 分镜图序列 + M3 过渡桥梁 | 无 | 低（扩展现有调用和prompt） |
| 第2批 | M1 导演规划 + M5 监督层 | 无 | 中（新增文件和扩展函数） |
| 第3批 | M4 模型路由 + M6 产物链 | M2 | 高（新增路由和产物链） |

## 铁律

1. 不删除任何现有代码
2. 不重写任何现有函数
3. 只新增文件或在现有函数末尾追加逻辑
4. 新增模块通过 try/except 优雅降级，失败不影响现有流程
5. 每批完成后全量重跑验证，确认不破坏现有功能

## 十、参考来源

| 来源 | 文件 | 学到什么 |
|------|------|---------|
| Toonflow 导演规划 | production_execution_director_plan.md | 分场、情绪、过渡、注意事项 |
| Toonflow 分镜表 | production_execution_storyboard_table.md | 铁律、拆镜、过渡桥梁 |
| Toonflow 监督层 | production_agent_supervision.md | 4条红线、A-D评分、审核维度 |
| Toonflow 视频提示词 | fixDB.ts (videoPromptGeneration) | 4种模式路由、Seedance 2.0格式 |
| Toonflow 分镜图生成 | batchGenerateImage.ts | 每镜头一张、并发批量 |
| OpenMontage cinematic | cinematic.yaml | 8阶段产物链、checkpoint、审批门 |
| OpenMontage VideoCompose | video_compose.py | edit_decisions、帧精确、归一化、多引擎 |
| OpenMontage pipeline_loader | pipeline_loader.py | 阶段排序、子阶段、条件激活 |
