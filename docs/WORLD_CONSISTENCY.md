# 世界观 + 场景统一 + 角色一致性 改造方案（待实施）

> 状态: 📋 方案阶段
> 日期: 2026-07-31
> 来源: Toonflow + OpenMontage 源码级调研

## 穿帮根因分析

HonCut 当前穿帮的根本原因：**依赖文字描述保持一致性，而非参考图锚定 + 显式ID绑定**。

两家验证的核心思想：**AI 视频模型看参考图比读文字描述可靠 100 倍。**

---

## 一、Toonflow 的一致性体系

### 数据流
```
剧本 → 资产提取(role/scene/tool 三类, 每个带 prompt+参考图)
     → 衍生资产(换装/场景时间变体, 独立参考图)
     → 导演规划(一致性锚点: 跨场沿用的视觉元素)
     → 分镜表(每镜 associateAssetsIds 显式绑定资产ID)
     → 分镜面板(@图N 引用资产参考图)
     → 视频提示词([References] @图1:角色 @图3:场景)
     → 视频生成(模型直接看到参考图)
```

### 关键机制
1. **资产三类分法**: role/scene/tool，每类有唯一ID + 参考图 + prompt
2. **衍生资产**: 换装=L0底模→L5配饰叠加层级（面容不变只换服装层）；场景时间变体（日→夜）独立参考图
3. **@图N 引用**: 视频提示词中 `@图1:沈辞参考图 @图3:城楼参考图`
4. **associateAssetsIds**: 每个分镜条目显式声明引用哪些资产ID
5. **人物空间位置基准表**: 分镜前预分析角色位置/朝向，跨镜锁定
6. **承接上镜段**: 同场内非首组必须写上镜结束瞬间的定格状态

### 踩坑经验
- 衍生资产范围严格控制：角色只提取变身状态，场景只提取时间变体，道具不衍生
- 光影/色调不进分镜 prompt，由场景参考图自动承载
- @图N 按输入顺序分配，不按类型归组
- 分镜图是视频首帧，遵循「首帧识别原则」
- 台词零删改，100%逐字搬运

---

## 二、OpenMontage 的一致性体系

### 核心机制
1. **Style Playbook**: 项目级视觉语言锁定（mood/palette/texture/lighting）
2. **Scene Plan Schema**: 每场景必须定义 shot_language（枚举值，机器可验证）+ texture_keywords + required_assets
3. **Character Design Schema**: 每角色定义 silhouette_notes + required_views(front/3-4/side/back) + props + constraints
4. **Asset Manifest**: 每个资产记录 id/type/path/scene_id/prompt/**seed**
5. **5层 shot_prompt_builder**: 同场景共享 Layer 3-5（Subject/Lighting/Style），只有 Layer 1-2（Camera/Movement）随镜头变

### Identity Anchor 铁律（最值钱的经验）
> "Repeat identity verbatim across every shot. 'the same character' / pronouns / 'Aang again' **do not work**. Repeat the 3-6 disambiguating visual attributes verbatim in every shot block."

示例：
```
Shot 2: Aang — bald, blue arrow tattoo, orange robes — plants his staff...
Shot 3: Rack focus from Aang's glowing arrow tattoo...
```
每个 shot 都重复「bald, blue arrow tattoo, orange robes」，不省略不用代词。

### 场景统一 4 招
1. **5层构建**: 同场景共享 Layer 3-5
2. **Seed Locking**: 同场景多镜头使用相同 seed
3. **全局 Color Grade**: 拼接后统一调色
4. **AI Clip Chaining**: 重叠提示 + frame_sampler 边界帧比对 + crossfade/fade-through-black

### 穿帮检测
- 无自动穿帮检测器
- 靠 Pre/Post Self-Review（5-aspect checklist）
- frame_sampler 拼接边界帧视觉检查
- Human Approval Gates

---

## 三、HonCut 改造计划

### 🔴 P0（解决最痛的穿帮）

#### P0-A: 场景参考图生成 + 每镜引用
- Phase 2.5 扩展：除了角色三视图，每个场景(where)也生成一张基准参考图
- 存储: output_dir/scenes/{scene_id}/reference.png
- Phase 5 视频生成时，同场景镜头引用该场景参考图
- 学 Toonflow: 场景时间变体（日/夜/黄昏）作为衍生资产独立生成

#### P0-B: Identity Anchor 逐字复述
- 改 storyboard_generator.py 的 _build_shot_prompt()
- 每个镜头的 prompt 必须包含角色的 3-6 个视觉特征（从 CHARACTERS.json 的 appearance 提取）
- 禁止使用代词（he/she/该角色/同上）
- 格式: "林夏 — 黑色长直发及肩, 白色修身衬衫, 深蓝西装裤 — 站在便利店门口..."
- 学 OpenMontage: "Repeat identity verbatim, pronouns don't work"

#### P0-C: 分镜绑定资产ID
- adaptation_engine 输出的每个 shot 增加 associate_assets 字段
- 格式: ["char:lin_xia", "scene:convenience_store"]
- Phase 5 根据 associate_assets 自动匹配参考图
- 学 Toonflow: associateAssetsIds 显式绑定

### 🟡 P1（进一步提升一致性）

#### P1-A: 衍生资产（换装/状态变化）
- character_discoverer 扩展：检测角色状态变化（淋湿/换装/受伤）
- 每个状态变化生成独立的衍生参考图
- 分镜中通过 associate_assets 引用衍生资产ID
- 学 Toonflow: L0底模→L5配饰叠加，面容不变

#### P1-B: 同场景共享视觉参数
- storyboard_generator 扩展：同 where 的镜头共享光影/色调/质感描述
- 学 OpenMontage: 5层构建，同场景共享 Layer 3-5
- 场景级 style_suffix 自动追加到同场景所有镜头

#### P1-C: Seed Locking
- seedance_client 扩展：同场景镜头使用相同 seed 参数
- 记录在 SHOT_META.json 中
- 学 OpenMontage: asset_manifest 记录 seed

### 🔵 P2（锦上添花）

#### P2-A: 空间位置基准表
- 导演规划(M1)扩展：输出角色空间位置基准（左/右/中 + 朝向）
- 分镜 prompt 中引用位置基准
- 学 Toonflow: 人物空间位置预分析

#### P2-B: 拼接边界帧检查
- Phase 7 扩展：xfade 前用 frame_sampler 提取边界帧做视觉比对
- 小不一致 → crossfade，大不一致 → fade-through-black
- 学 OpenMontage: AI Clip Chaining

---

## 四、与现有模块的关系

| 改造 | 改动文件 | 方式 |
|------|----------|------|
| P0-A 场景参考图 | pipeline_runner.py (Phase 2.5), seedream_client.py | 扩展现有调用 |
| P0-B Identity Anchor | storyboard_generator.py | 改 _build_shot_prompt() |
| P0-C 资产ID绑定 | adaptation_engine.py (prompt), pipeline_runner.py (Phase 5) | 扩展 prompt + 匹配逻辑 |
| P1-A 衍生资产 | character_discoverer.py, character_factory.py | 扩展检测 + 生成 |
| P1-B 共享视觉参数 | storyboard_generator.py | 扩展 prompt 构建 |
| P1-C Seed Locking | seedance_client.py, pipeline_runner.py | 扩展参数 |
| P2-A 空间位置 | director_planner.py | 扩展输出 |
| P2-B 边界帧检查 | edit_decisions.py | 扩展 Phase 7 |
