# OM Skills → HonCut Phase 自动启用映射

> 来源: OpenMontage/.agents/skills/ 83 个 skill 中筛选
> 分级: Tier 1 直接可用 | Tier 2 可搬运适配 | Tier 3 参考学习

## 自动启用规则

Pipeline 启动时根据 Phase 自动加载对应 Skill：

| Phase | 自动加载的 Skill | Tier |
|-------|-----------------|------|
| Phase 2 (编剧引擎) | seedance-2-0 (运镜模板), visual-style (风格系统) | T1 |
| Phase 2.5 (故事板) | flux-best-practices (图片 prompt) | T3 |
| Phase 5 (视频生成) | seedance-2-0 (identity-lock), ai-video-gen (路由) | T1/T3 |
| Phase 6 (质检) | character-animation-qa, video-understand | T2 |
| Phase 7 (组装) | video-edit (FFmpeg) | T1 |
| Phase 8 (后期) | doubao-tts (字幕), music (BGM), sound-effects (音效) | T1/T2 |

## Tier 1: 直接可用 (4 个)

| Task ID | Skill | 目标 Phase | 解决的问题 |
|---------|-------|-----------|-----------|
| t_7edbcf0e | seedance-2-0 | Phase 2/5 | 发型穿帮 + 运镜割裂 |
| t_1dd8a1c4 | doubao-tts | Phase 8 | 字幕是旁白 |
| t_f9e61d0a | visual-style | Phase 2 | 视觉风格一致性 |
| t_7815a275 | video-edit | Phase 7/8 | FFmpeg 标准化 |

## Tier 2: 可搬运适配 (4 个)

| Task ID | Skill | 目标 Phase | 用途 |
|---------|-------|-----------|------|
| t_5e8a3f51 | music (ElevenLabs) | Phase 8 | BGM 自动生成 |
| t_d95961c3 | sound-effects | Phase 8 | 音效自动生成 |
| t_1abb3231 | video-understand | Phase 6 | 本地视频分析 |
| t_d8c31352 | character-animation-qa | Phase 6 | 动画 QA 检查 |

## Tier 3: 参考学习 (3 个)

| Task ID | Skill | 参考用途 |
|---------|-------|---------|
| t_ba19671d | flux-best-practices | 优化 Phase 2.5 图片 prompt |
| t_b14db36d | ai-video-gen | 优化 Phase 5 路由选择 |
| t_09558684 | video-toolkit | 参考完整视频制作流程 |

## 实施顺序

1. **P0 (立即)**: Tier1-1 seedance-2-0 → 解决发型穿帮 + 运镜割裂
2. **P0 (立即)**: Tier1-2 doubao-tts → 解决字幕问题
3. **P1 (本周)**: Tier1-3 visual-style → 视觉一致性
4. **P2 (下周)**: Tier2 music/sound-effects → 音频层
5. **P2 (下周)**: Tier1-4 video-edit → FFmpeg 标准化
6. **P3 (后续)**: Tier2 video-understand/character-qa → 质检增强
7. **P3 (后续)**: Tier3 参考学习 → 架构优化
