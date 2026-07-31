# 参考源码索引（Toonflow + OpenMontage）

> Qwen 实施时直接按此索引读源码，不用重新搜索
> 日期: 2026-07-31
> 注意: Toonflow 实际路径是 Toonflow-app（不是 Toonflow）

---

## Toonflow（/Users/soda/projects/Toonflow-app/）

| # | 功能 | 文件 | 行号 | 关键内容 |
|---|------|------|------|----------|
| 1 | 资产提取（role/scene/tool） | `src/lib/fixDB.ts` | L145 | universal_agent skill，三类资产提取规则 |
| 1b | 资产提取初始化 | `src/lib/initDB.ts` | L358, L738-739 | 同内容初始化 + path/name 注册 |
| 2 | 衍生资产规则 | `data/skills/production_execution_derive_assets.md` | L19-L100 | 角色仅变身(L82)、场景仅时间变体(L83)、道具不衍生(L84) |
| 2b | L0-L5 叠加层级 | `data/skills/art_skills/3D_chinese_traditional/art_prompt/art_character_derivative.md` | L22-L32 | L0底模→L5配饰，面容不变原则 L11-L18 |
| 3 | 导演规划一致性锚点 | `data/skills/production_execution_director_plan.md` | L91, L120 | 视觉一致性锚点定义 + 输出字段格式 |
| 4 | associateAssetsIds | `data/skills/production_skills/storyboard_table_techniques.md` | L181-L184 | 资产ID绑定规则（角色出现即引用、场景必选） |
| 4b | associateAssetsIds schema | `src/agents/productionAgent/tools.ts` | L29, L250-L271 | zod schema + Output 解析 |
| 5 | @图N 引用系统 | `data/skills/production_skills/storyboard_prompt_techniques.md` | L255-L292 | @图N 按 associateAssetsIds 顺序编号 |
| 5b | @图N 编号规则+示例 | `data/modelPrompt/video/universalMulti-parameterMode.md` | L89-L165 | 完整编号规则和示例 |
| 5c | volcengine 参考图构建 | `data/vendor/volcengine.ts` | L534-L571 | 多模态参考模式 content 数组构建 |
| 6 | 视频提示词4种路由 | `src/lib/fixDB.ts` | L147-L148 | videoPromptGeneration skill data |
| 6b | Seedance 2.0 多分镜格式 | `data/modelPrompt/video/seedance2Multi-parameterMode.md` | L1-L262 | 完整格式定义 |
| 7 | 分镜图批量生成 | `src/routes/production/storyboard/batchGenerateImage.ts` | L18, L26, L129-L139 | concurrentCount=5, Promise.all 并发 |
| 7b | 参考图输入 | 同上 | L103 | referenceList: getAssetsImageBase64() |
| 8 | 监督层4条红线 | `data/skills/production_agent_supervision.md` | L84-L112 | R1(L89)/R2(L96)/R3(L102)/R4(L108) |
| 8b | A-D 评分标准 | 同上 | L63-L70 | A=0严重≤2中等, D=≥3严重 |
| 9 | 空间位置基准表 | `data/skills/production_execution_storyboard_panel.md` | L95-L101 | 全局基准表（角色→位置+朝向） |
| 10 | generate_audio | `data/vendor/volcengine.ts` | L583-L589 | model.audio 配置决定 generate_audio |
| 11 | 多模态组合参考 | `data/vendor/volcengine.ts` | L534-L571 | imageReference:N/videoReference:N/audioReference:N |
| 12 | 参考图压缩 | `src/utils/vm.ts` | L65-L74, L76-L80 | zipImage(sharp降quality) + zipImageResolution |
| 13 | 角色参考图生成 | `data/skills/art_skills/3D_chinese_traditional/art_prompt/art_character.md` | L1-L196 | 基础形象约束 |
| 14 | 铁律 | `data/skills/production_execution_storyboard_table.md` | L12-L36 | 优先级链(L14)、≤15秒(L18)、>20字拆镜(L20)、台词零删改(L22) |

---

## OpenMontage（/Users/soda/projects/OpenMontage/）

| # | 功能 | 文件 | 行号 | 关键内容 |
|---|------|------|------|----------|
| 1 | cinematic.yaml 管线 | `pipeline_defs/cinematic.yaml` | L59-L274 | 8阶段 + produces/requires/review_focus |
| 2 | scene_plan schema | `schemas/artifacts/scene_plan.schema.json` | L31-L100 | shot_language(L31), texture_keywords(L92), required_assets(L97) |
| 3 | character_design schema | `schemas/artifacts/character_design.schema.json` | L32-L37 | silhouette_notes(L32), required_views(L35), props(L36), constraints(L37) |
| 4 | asset_manifest schema | `schemas/artifacts/asset_manifest.schema.json` | L14-L25 | scene_id(L23), seed(L25) |
| 5 | shot_prompt_builder 5层 | `lib/shot_prompt_builder.py` | L82-L143 | L1:Camera(L99) L2:Movement(L110) L3:Subject(L119) L4:Lighting(L127) L5:Style(L136) |
| 5b | 批量 prompt | 同上 | L146-L166 | build_batch_prompts |
| 6 | Identity Anchor 铁律 | `skills/creative/prompting/seedance-prompting.md` | L47 | "pronouns don't work, repeat 3-6 attributes verbatim" |
| 6b | Identity phrases 列表 | `.agents/skills/seedance-2-0/SKILL.md` | L218-L231 | identity-anchor phrases + anti-drift fallback |
| 7 | seedance_video.py | `tools/video/seedance_video.py` | L107-L145, L209-L249 | generate_audio(L107/L209), ref_images(L124/L224), ref_videos(L134/L234) |
| 8 | Seedance prompting 指南 | `skills/creative/prompting/seedance-prompting.md` | L27,L42-43,L47,L68 | identity(L27), reference(L42), verbatim(L47), multi-shot(L68) |
| 9 | video_compose.py | `tools/video/video_compose.py` | L390-L700, L1292-L1493 | _compose FFmpeg(L390), _render 路由(L1292) |
| 10 | checkpoint 机制 | `lib/checkpoint.py` | L178-L228, L263-L298, L336+ | init_project(L178), history归档(L263), write_checkpoint(L336) |
| 11 | pipeline_loader.py | `lib/pipeline_loader.py` | L79-L86, L98-L150 | conditions(L79), sub_stages(L98), get_stage_order(L125) |
| 12 | color_grade | `tools/enhancement/color_grade.py` | L24-L42, L181-L210 | PROFILES(L24), cinematic_warm(L26), cinematic_cool(L34), _build_filter(L181) |
| 13 | frame_sampler | `tools/analysis/frame_sampler.py` | L231-L280 | 边界帧提取: 场景首帧+长场景中点帧 |
| 14 | Seedance 2.0 SKILL.md | `.agents/skills/seedance-2-0/SKILL.md` | L53,L73,L206-L231 | reference_to_video(L53), identity_lock标签(L209), anchor phrases(L218) |
| 15 | audio_mixer | `tools/audio/audio_mixer.py` | L198-L214, L332-L437 | loudnorm(L198, -16LUFS), duck(L332) |

---

## HonCut 改造 → 参考源码对照

| HonCut 改造项 | 抄 Toonflow | 抄 OpenMontage |
|---|---|---|
| P0-A 场景参考图 | #1 资产提取 + #5 @图N | #2 scene_plan schema |
| P0-B Identity Anchor | #5 @图N 引用 | #6 Identity 铁律 + #6b phrases |
| P0-C 资产ID绑定 | #4 associateAssetsIds | #4 asset_manifest |
| P0 generate_audio | #10 volcengine.ts | #7 seedance_video.py L107 |
| P1-A 衍生资产 | #2 衍生规则 + #2b L0-L5 | #3 character_design props |
| P1-B 共享视觉参数 | #3 一致性锚点 | #5 shot_prompt_builder 5层 |
| P1-C Seed Locking | — | #4 asset_manifest seed(L25) |
| P1 多模态参考 | #11 多模态组合 | #7 ref_images/videos/audios |
| P1 参考图压缩 | #12 zipImage | — |
| P2-A 空间位置 | #9 位置基准表 | — |
| P2-B 边界帧检查 | — | #13 frame_sampler |
| M4 模型路由 | #6 4种路由 + #6b 格式 | — |
| M5 监督层 | #8 4条红线 + #8b 评分 | — |
| M3 铁律 | #14 铁律 L12-L36 | — |
| M2 分镜图并发 | #7 batchGenerateImage | — |
