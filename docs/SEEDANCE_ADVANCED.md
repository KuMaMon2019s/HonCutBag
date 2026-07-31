# Seedance 2.0 进阶能力集成方案（待实施）

> 状态: 📋 方案阶段 — 等 Toonflow/OpenMontage 踩坑经验研究完成后再实施
> 日期: 2026-07-31
> 来源: 火山方舟文档 https://console.volcengine.com/ark/region:cn-beijing/docs/82379/2291680

## 原则

- **Agent Plan** Model ID: `doubao-seedance-2-0-mini`（无日期后缀！按量计费才用带日期的）
- 不需要安装方舟 SDK，现有 requests 直接调 REST API 即可
- 所有新参数都是 JSON body 顶层字段

---

## 能力 1: return_last_frame（🔴 最高优先）

### 是什么
Seedance 2.0 支持在视频生成完成后返回尾帧图片。

### 为什么重要
把 Shot N 的尾帧当 Shot N+1 的**首帧参考**（first_frame），实现像素级镜头衔接。
比 Toonflow 的"承接上镜"文字描述强 10 倍——直接用画面锚定消灭跳跃感。

### API 参数
```json
{
  "return_last_frame": "true"
}
```

### 改动范围
1. `seedance_client.py` submit() — 加 `return_last_frame: str = "false"` 参数
2. `seedance_client.py` poll() — 解析返回的尾帧 URL/base64
3. `pipeline_runner.py` Phase 5 循环 — Shot N 尾帧 → Shot N+1 的 first_frame_base64
4. 链式传递：第一个镜头用角色参考图，后续镜头用上一镜尾帧

### 风险
- 尾帧可能包含 AI 伪影，传递给下一镜会放大
- 需要确认 Agent Plan 是否支持此参数
- 首尾帧模式 vs 纯首帧模式的 API 差异

---

## 能力 2: generate_audio（🟡 中优先）

### 是什么
Seedance 直接生成带环境音/音效的视频。

### API 参数
```json
{
  "generate_audio": "true"
}
```

### 现状
seedance_client.py line 80 已有 `"generate_audio": False`（写死 False）

### 改动范围
1. submit() 加 `generate_audio: str = "false"` 参数
2. Phase 5 调用时可选开启
3. Phase 8 音频管线适配（有原生音频时跳过环境音合成）

### 风险
- 生成的音频质量是否够用
- 与 Phase 8 的 BGM/配音如何混合
- 可能增加生成时间和 token 消耗

---

## 能力 3: 多模态组合参考（🟡 中优先）

### 是什么
同时传入图片+视频+音频作为参考输入。

### 应用场景
- 角色三视图 + 上一镜头视频 → 更强的角色一致性
- 风格参考视频 + 角色图 → 风格+角色双锚定

### 改动范围
1. submit() content 数组支持多个 role 元素
2. Phase 5 构建 content 时组合多种参考

### 风险
- 多参考可能互相冲突
- API 对参考数量/大小可能有限制

---

## 能力 4: 视频延长（🔵 低优先）

### 是什么
对已生成的视频进行延长，不用重新生成。

### 应用场景
- 镜头时长不够时延长，省 token
- 质检发现时长不足时自动补救

### 风险
- 延长部分的一致性
- 是否支持 Agent Plan

---

## 能力 5: 视频编辑（🔵 低优先）

### 是什么
对已生成视频的局部进行修改。

### 应用场景
- 不满意的镜头局部修改，不用整段重新生成

---

## 研究结果（2026-07-31 Toonflow + OpenMontage 源码级调研）

### return_last_frame — ⚠️ 两家都没用！

| 项目 | 状态 | 替代方案 |
|------|------|----------|
| Toonflow | ❌ 未使用 | prompt 级"承接上镜"机制 |
| OpenMontage | ❌ 未使用 | 有 end_image_url 输入参数，但无尾帧回传 |

**结论：两家都选择了 prompt 级衔接而非 API 级尾帧回传。**
可能原因：API 尾帧回传在实际中效果不佳（画面跳变、伪影放大）。
**建议：暂缓 return_last_frame，优先强化现有 prompt 级衔接 + 角色参考图锚定。**

### generate_audio — ✅ 两家都用！

| 项目 | 实现 | 踩坑经验 |
|------|------|----------|
| Toonflow | model.audio 配置（optional/true/false） | 无明显坑 |
| OpenMontage | **默认 True** | "sync audio is the moat. Strip in compose if unused." |

**OpenMontage 核心经验：默认开启，后期不需要再剥离，而不是关闭生成。**
音频生成失败会导致整个视频生成失败（无降级）。

### 多模态组合参考 — ✅ 两家都用！

| 项目 | 限制 | 踩坑经验 |
|------|------|----------|
| Toonflow | mode 数组控制数量 | 图片需压缩（zipImage 3*1024*104），大图直传会超限 |
| OpenMontage | **图片≤9, 视频≤3, 音频≤3** | 严格上限验证，超出直接报错 |

**OpenMontage 角色一致性经验（SKILL.md）：**
- Identity-anchor phrases: "the same character", "consistent across all shots", "maintain exact appearance from reference image"
- 减少面部漂移的关键是在 prompt 中反复强调身份锚定

### 视频延长/编辑 — ❌ 两家都没用

### Toonflow 额外踩坑

1. `sequential_image_generation` 必须显式设为 `"disabled"`（仅 seedream 5.0-lite/4.5/4.0 支持）
2. 不同供应商参数格式差异大（volcengine 用 content[]+role，atlascloud 用扁平参数）
3. 参考图需要压缩后上传，大图直传超限

### OpenMontage 额外踩坑

1. generate_audio 无降级——音频生成失败 = 整个视频失败
2. 参考图走 TOS 上传避免隐私检测（与 HonCut 现有方案一致）
3. 跨镜头 continuity 完全依赖 prompt 层 identity 重复，无代码级状态管理

---

## 修订后的实施优先级

| 优先级 | 能力 | 理由 |
|--------|------|------|
| 🔴 P0 | generate_audio 默认开启 | 两家都验证过，OpenMontage 说"这是护城河" |
| 🔴 P0 | Identity-anchor phrases 强化 | OpenMontage 验证的角色一致性关键技巧 |
| 🟡 P1 | 多模态组合参考（图片≤9） | 两家都用，有明确限制和踩坑经验 |
| 🟡 P1 | 参考图压缩（学 Toonflow zipImage） | 防止大图超限 |
| 🔵 P2 | return_last_frame | 两家都没用于镜头衔接，可能用于**视频时间延长**（尾帧接续生成下一段），等验证 |
| 🔵 P2 | 视频延长/编辑 | 两家都没用，等 API 更成熟 |
