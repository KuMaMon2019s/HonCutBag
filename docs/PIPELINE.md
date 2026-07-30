# Honcut AI 视频管线 — PIPELINE 总纲

> **版本**: v1.0.0  
> **更新日期**: 2026-07-28  
> **状态**: 架构定义阶段

---

## 1. 数据流总览（ASCII）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Honcut AI 视频管线 · 9 Phase 架构                      │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │   任意文本输入     │
                          │ (一句话/大纲/长篇) │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 1: 基础设施 (已就绪)   │
                    │  n8n · KB · ARK · Qdrant     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 2: 事件图谱引擎 (编剧) │
                    │  text → 结构化叙事            │
                    └──────┬───────────────┬──────┘
                           │               │
              STORYBOARD.json      CHARACTERS.json
                           │               │
                           │    ┌──────────▼──────────┐
                           │    │  Phase 3: 角色资产    │
                           │    │  三视图 + 角色卡      │
                           │    └──────────┬──────────┘
                           │               │
                           │    characters/*/front|side|back.png
                           │    character_card.json / angle_map.json
                           │               │
                    ┌──────▼───────────────▼──────┐
                    │  Phase 4: 智能路由            │
                    │  镜头 → 工具决策              │
                    └──────────────┬──────────────┘
                                   │
                    shots/S*/SHOT_META.json
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 5: 视频生成            │
                    │  Seedance 异步生成            │
                    └──────────────┬──────────────┘
                                   │
                    shots/S*/output.mp4 + frames/
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 6: 质量检查            │
                    │  一致性守卫                   │
                    └──┬─────────────────────┬────┘
                       │                     │
                  ✅ ≥0.7              🔴 <0.7
                       │                     │
                       │              ┌──────▼──────┐
                       │              │ 回 Phase 5   │
                       │              │ re-gen       │
                       │              └─────────────┘
                       │
                    ┌──▼───────────────────────────┐
                    │  Phase 7: 粗剪                │
                    │  拼接毛坯                     │
                    └──────────────┬───────────────┘
                                   │
                          stitched.mp4
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 8: 后处理 (精剪)       │
                    │  音频/画质/节奏/转场          │
                    └──────────────┬──────────────┘
                                   │
                          polished.mp4
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 9: 全流程集成          │
                    │  Go 后端 + n8n 触发 + E2E    │
                    └─────────────────────────────┘
```

---

## 2. Phase 依赖关系

```
Phase 1 (基础设施)
  └── 被所有 Phase 依赖（运行时环境）

Phase 2 (事件图谱) ──依赖──▶ Phase 1
  ├── 输出 STORYBOARD.json ──▶ Phase 4
  └── 输出 CHARACTERS.json ──▶ Phase 3

Phase 3 (角色资产) ──依赖──▶ Phase 2 (CHARACTERS.json)
  └── 输出角色资产 ──▶ Phase 4

Phase 4 (智能路由) ──依赖──▶ Phase 2 + Phase 3
  └── 输出 SHOT_META.json ──▶ Phase 5

Phase 5 (视频生成) ──依赖──▶ Phase 4
  ├── 输出 output.mp4 ──▶ Phase 6, Phase 7
  └── 输出 frames/ ──▶ Phase 6

Phase 6 (质量检查) ──依赖──▶ Phase 5
  ├── ✅ 通过 ──▶ Phase 7
  └── 🔴 不通过 ──▶ Phase 5 (闭环重试)

Phase 7 (粗剪) ──依赖──▶ Phase 5 (所有 output.mp4)
  └── 输出 stitched.mp4 ──▶ Phase 8

Phase 8 (后处理) ──依赖──▶ Phase 7
  └── 输出 polished.mp4 ──▶ Phase 9 / 交付

Phase 9 (全流程集成) ──依赖──▶ Phase 1~8 全部
  └── 端到端编排
```

---

## 3. Phase 详细定义

### Phase 1: 基础设施 ✅

> **一句话**: 搭建运行环境，确保所有外部服务和密钥就绪。

| 项目 | 说明 |
|------|------|
| **输入** | 环境变量 `ARK_API_KEY`、Docker 镜像、knowledge-base 目录 |
| **输出** | 可用服务：n8n (localhost:5678)、Qdrant (localhost:6333)、知识库向量索引 |
| **工具/脚本** | `docker compose up`、`kb_scan`、`kb_search` |
| **状态** | ✅ 已完成 |

**组件清单**:
- n8n Docker（工作流触发器）
- knowledge-base 目录结构
- ARK_API_KEY（火山方舟 API）
- Qdrant 向量数据库

---

### Phase 2: 事件图谱引擎（编剧）❌

> **一句话**: 将任意文本解析为结构化事件图谱和角色列表。

| 项目 | 说明 |
|------|------|
| **输入** | 任意文本（一句话 / 一段描述 / 大纲 / 长篇均可，不特指小说） |
| **输出** | `STORYBOARD.json` + `CHARACTERS.json` |
| **工具链** | `text_parser.py` → `event_extractor.py` → `character_discoverer.py` → `adaptation_engine.py` → `storyboard_generator.py` |
| **状态** | ❌ 未实现 |

**工具链说明**:

| 工具 | 职责 |
|------|------|
| `text_parser.py` | 自动判断输入规模：短文直接提事件，长文按段落/章节拆分 |
| `event_extractor.py` | 从文本块中提取事件（who/what/where/when/why） |
| `character_discoverer.py` | 从事件中发现角色并提取属性 |
| `adaptation_engine.py` | 将原始事件适配为可视觉化的镜头语言 |
| `storyboard_generator.py` | 生成最终分镜脚本 |

**关键约束**:
- 输入不限格式，text_parser 负责自适应
- 输出必须严格符合下方 JSON Schema

---

### Phase 3: 角色资产 ✅（需增强）

> **一句话**: 为每个角色生成三视图和角色卡。

| 项目 | 说明 |
|------|------|
| **输入** | `CHARACTERS.json`（来自 Phase 2） |
| **输出** | `characters/{name}/front.png` · `side.png` · `back.png` · `character_card.json` · `angle_map.json` |
| **工具** | `character_factory.py`（调用 Seedream 5.0-lite `/images/generations`） |
| **状态** | ✅ 已完成，需增强 |

**增强方向**:
- 参考 ComfyUI 三视图工作流的 prompt 模板和负面提示词
- 将 IPAdapter / ControlNet 思路转为 Seedream API 参数（如 reference_image、control_strength）
- 统一角色一致性（同一角色不同镜头的外貌锚定）

**输出目录结构**:
```
characters/
├── {角色名}/
│   ├── front.png          # 正面视图
│   ├── side.png           # 侧面视图
│   ├── back.png           # 背面视图
│   ├── character_card.json # 角色属性卡
│   └── angle_map.json     # 角度映射表
```

---

### Phase 4: 智能路由 ✅

> **一句话**: 为每个镜头选择最优生成工具和参数。

| 项目 | 说明 |
|------|------|
| **输入** | `STORYBOARD.json` + 角色三视图路径 |
| **输出** | `shots/S{N}/SHOT_META.json`（含路由决策） |
| **工具** | `orchestrator.py` + `tool_router.py` |
| **状态** | ✅ 已完成 |

**路由决策维度**:
- 镜头类型（全景/中景/特写/运动）
- 角色数量（单人/多人/无人）
- 动作复杂度（静态/动态/交互）
- 情绪基调（决定风格参数）

**输出目录结构**:
```
shots/
├── S1/
│   └── SHOT_META.json
├── S2/
│   └── SHOT_META.json
└── ...
```

---

### Phase 5: 视频生成 ✅

> **一句话**: 调用 Seedance 异步生成每个镜头的视频片段。

| 项目 | 说明 |
|------|------|
| **输入** | `shots/S{N}/SHOT_META.json` |
| **输出** | `shots/S{N}/output.mp4` + `shots/S{N}/frames/` |
| **工具** | `seedance_client.py`（Agent Plan 异步 submit/poll） |
| **状态** | ✅ 已完成 |

**API 约束**:
- Endpoint: `/api/plan/v3/`（Agent Plan）
- 模型: `doubao-seedance-2.0-fast`
- 参数全部顶层，`watermark: false`
- 异步模式：submit → poll → 下载

---

### Phase 6: 质量检查 ✅（需增强）

> **一句话**: 检查角色一致性和画面质量，不合格则回退重生成。

| 项目 | 说明 |
|------|------|
| **输入** | `shots/S{N}/frames/`（抽帧） |
| **输出** | `consistency_report.json` |
| **工具** | `consistency_guard.py`（embedding 比对） |
| **状态** | ✅ 已完成，需增强 |

**增强方向**:
- 用 OM `frame_sampler` 替代手写 ffmpeg 抽帧
- 增加 `composition_validator`（构图检查）
- 增加 `face_tracker`（人脸追踪一致性）

**闭环机制**:
- 一致性分数 ≥ 0.7 → ✅ 通过，进入 Phase 7
- 一致性分数 < 0.7 → 🔴 不通过，回退 Phase 5 re-gen（最多重试 3 次）

---

### Phase 7: 粗剪 ✅（需增强）

> **一句话**: 将所有镜头拼接为毛坯视频。

| 项目 | 说明 |
|------|------|
| **输入** | 所有 `shots/S{N}/output.mp4` |
| **输出** | `stitched.mp4`（毛坯） |
| **工具** | `assembly_engine.py`（OM `video_stitch` + `remotion_caption_burn`） |
| **状态** | ✅ 已完成，需增强 |

**增强方向**:
- OM `video_trimmer`：去除每个镜头的废片（开头/结尾黑帧、模糊帧）
- OM `silence_cutter`：去除静默段落（如有音频轨道）

---

### Phase 8: 后处理（精剪）❌

> **一句话**: 对毛坯视频进行音频、画质、节奏、转场的精细加工，输出成品。

| 项目 | 说明 |
|------|------|
| **输入** | `stitched.mp4` |
| **输出** | `polished.mp4`（成品） |
| **工具** | 全部来自 OpenMontage（见下表） |
| **状态** | ❌ 未实现 |

**工具清单（全部来自 OM）**:

| 类别 | 工具 | 职责 |
|------|------|------|
| **音频** | `audio_mixer` | 多轨混音（BGM + 音效 + 对白） |
| | `music_gen` | AI 生成背景音乐 |
| | `doubao_tts` | 豆包 TTS 生成旁白/对白 |
| | `audio_enhance` | 音频降噪/增强 |
| **画质** | `enhancement/` | 超分辨率 / 降噪 |
| **画幅** | `auto_reframe` | 自动裁切适配不同比例（16:9 / 9:16 / 1:1） |
| **图文** | `graphics/` | 片头 / 片尾 / 字幕卡 |
| **节奏** | 变速 / 卡点 | 根据音乐节奏调整镜头速度 |
| **转场** | 转场精修 | 按情绪选择转场类型（切/溶解/擦除/缩放） |

---

### Phase 9: 全流程集成 ✅

> **一句话**: 用 Go 后端串联所有 Phase，支持 n8n 触发和端到端验证。

| 项目 | 说明 |
|------|------|
| **输入** | 用户请求（文本 + 参数） |
| **输出** | `polished.mp4`（最终交付物） |
| **工具** | Go 后端 + n8n webhook 触发 + E2E 测试 |
| **状态** | ✅ 已完成 |

**集成架构**:
```
用户请求 → n8n webhook → Go 后端 → Phase 2~8 顺序执行 → 交付 polished.mp4
```

---

## 4. 关键 JSON 接口规范

### 4.1 CHARACTERS.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CHARACTERS",
  "type": "object",
  "required": ["version", "characters"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "source_text_hash": { "type": "string", "description": "输入文本的 SHA-256，用于溯源" },
    "characters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name", "appearance", "role"],
        "properties": {
          "id": { "type": "string", "description": "角色唯一标识，如 char_001" },
          "name": { "type": "string", "description": "角色名称" },
          "role": { "type": "string", "enum": ["protagonist", "antagonist", "supporting", "extra"], "description": "角色定位" },
          "appearance": {
            "type": "object",
            "required": ["gender", "age_range", "summary"],
            "properties": {
              "gender": { "type": "string", "enum": ["male", "female", "nonbinary", "unknown"] },
              "age_range": { "type": "string", "description": "如 '20-30'" },
              "height": { "type": "string", "description": "如 '175cm'" },
              "build": { "type": "string", "description": "体型，如 'slim', 'athletic', 'heavy'" },
              "hair": { "type": "string", "description": "发型发色，如 '黑色短发'" },
              "face": { "type": "string", "description": "面部特征，如 '圆脸、戴眼镜'" },
              "clothing": { "type": "string", "description": "典型穿着，如 '白色衬衫+牛仔裤'" },
              "distinguishing": { "type": "string", "description": "显著标记，如 '左脸颊疤痕'" },
              "summary": { "type": "string", "description": "一句话外貌总结，用于 prompt 生成" }
            }
          },
          "personality": {
            "type": "object",
            "properties": {
              "traits": { "type": "array", "items": { "type": "string" } },
              "speech_style": { "type": "string" },
              "motivation": { "type": "string" }
            }
          },
          "relationships": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "target_id": { "type": "string" },
                "type": { "type": "string", "description": "如 'friend', 'rival', 'mentor'" },
                "description": { "type": "string" }
              }
            }
          },
          "asset_path": { "type": "string", "description": "角色资产目录，如 'characters/char_001/'" }
        }
      }
    }
  }
}
```

---

### 4.2 STORYBOARD.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "STORYBOARD",
  "type": "object",
  "required": ["version", "title", "shots"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "title": { "type": "string", "description": "作品标题" },
    "genre": { "type": "string", "description": "类型，如 'sci-fi', 'romance', 'thriller'" },
    "tone": { "type": "string", "description": "整体基调，如 'dark', 'uplifting', 'suspenseful'" },
    "total_duration_target": { "type": "number", "description": "目标总时长（秒）" },
    "synopsis": { "type": "string", "description": "故事梗概（3-5 句）" },
    "shots": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["shot_id", "sequence", "description", "characters"],
        "properties": {
          "shot_id": { "type": "string", "description": "镜头 ID，如 'S1', 'S2'" },
          "sequence": { "type": "integer", "description": "镜头序号（从 1 开始）" },
          "description": { "type": "string", "description": "镜头内容描述（自然语言）" },
          "scene": { "type": "string", "description": "场景/地点，如 '咖啡馆内'、'雨夜街道'" },
          "time_of_day": { "type": "string", "enum": ["dawn", "morning", "noon", "afternoon", "dusk", "night"] },
          "camera": {
            "type": "object",
            "properties": {
              "shot_type": { "type": "string", "enum": ["wide", "medium", "close-up", "extreme-close-up", "over-shoulder", "pov", "aerial"] },
              "angle": { "type": "string", "enum": ["eye-level", "low-angle", "high-angle", "dutch", "birds-eye"] },
              "movement": { "type": "string", "enum": ["static", "pan-left", "pan-right", "tilt-up", "tilt-down", "dolly-in", "dolly-out", "tracking", "crane", "handheld"] },
              "lens": { "type": "string", "description": "如 '50mm', 'wide-angle', 'telephoto'" }
            }
          },
          "characters": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "character_id": { "type": "string" },
                "action": { "type": "string", "description": "该角色在此镜头中的动作" },
                "emotion": { "type": "string", "description": "情绪状态" },
                "position": { "type": "string", "description": "画面中的位置，如 'left', 'center', 'right'" }
              }
            }
          },
          "dialogue": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "character_id": { "type": "string" },
                "line": { "type": "string" },
                "delivery": { "type": "string", "description": "语气，如 'whisper', 'shout', 'calm'" }
              }
            }
          },
          "mood": { "type": "string", "description": "镜头情绪，如 'tense', 'peaceful', 'joyful'" },
          "duration_hint": { "type": "number", "description": "建议时长（秒）" },
          "transition_to_next": { "type": "string", "enum": ["cut", "dissolve", "fade", "wipe", "match-cut"], "description": "到下一镜头的转场" },
          "notes": { "type": "string", "description": "导演备注/特殊要求" }
        }
      }
    }
  }
}
```

---

### 4.3 SHOT_META.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SHOT_META",
  "type": "object",
  "required": ["shot_id", "route", "prompt", "parameters"],
  "properties": {
    "shot_id": { "type": "string", "description": "镜头 ID，如 'S1'" },
    "source_storyboard": { "type": "string", "description": "来源 STORYBOARD.json 的 shot_id" },
    "route": {
      "type": "object",
      "required": ["tool", "model"],
      "properties": {
        "tool": { "type": "string", "description": "使用的工具，如 'seedance_client'" },
        "model": { "type": "string", "description": "模型名，如 'doubao-seedance-2.0-fast'" },
        "reason": { "type": "string", "description": "路由决策理由" },
        "fallback": { "type": "string", "description": "备选工具/模型" }
      }
    },
    "prompt": {
      "type": "object",
      "required": ["text"],
      "properties": {
        "text": { "type": "string", "description": "最终发送给生成模型的 prompt" },
        "negative_prompt": { "type": "string", "description": "负面提示词" },
        "style_prefix": { "type": "string", "description": "风格前缀，如 'cinematic, 4K'" }
      }
    },
    "parameters": {
      "type": "object",
      "properties": {
        "duration": { "type": "number", "description": "视频时长（秒）" },
        "resolution": { "type": "string", "description": "如 '1280x720', '720x1280'" },
        "fps": { "type": "integer", "description": "帧率" },
        "aspect_ratio": { "type": "string", "enum": ["16:9", "9:16", "1:1", "4:3"] },
        "seed": { "type": "integer", "description": "随机种子（可复现）" },
        "cfg_scale": { "type": "number", "description": "引导强度" },
        "watermark": { "type": "boolean", "const": false }
      }
    },
    "reference_images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "character_id": { "type": "string" },
          "image_path": { "type": "string", "description": "参考图路径，如 'characters/char_001/front.png'" },
          "role": { "type": "string", "enum": ["face_ref", "pose_ref", "style_ref", "full_body_ref"] },
          "weight": { "type": "number", "description": "参考权重 0.0-1.0" }
        }
      }
    },
    "status": { "type": "string", "enum": ["pending", "generating", "completed", "failed", "retrying"] },
    "retry_count": { "type": "integer", "default": 0 },
    "output_path": { "type": "string", "description": "生成结果路径" },
    "task_id": { "type": "string", "description": "Agent Plan 异步任务 ID" }
  }
}
```

---

### 4.4 consistency_report.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CONSISTENCY_REPORT",
  "type": "object",
  "required": ["shot_id", "overall_score", "checks", "verdict"],
  "properties": {
    "shot_id": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "overall_score": { "type": "number", "minimum": 0, "maximum": 1, "description": "综合一致性分数" },
    "verdict": { "type": "string", "enum": ["pass", "fail"], "description": "pass: ≥0.7, fail: <0.7" },
    "checks": {
      "type": "object",
      "properties": {
        "character_consistency": {
          "type": "object",
          "properties": {
            "score": { "type": "number", "minimum": 0, "maximum": 1 },
            "method": { "type": "string", "description": "如 'embedding_cosine_similarity'" },
            "details": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "character_id": { "type": "string" },
                  "reference_image": { "type": "string" },
                  "frame_scores": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "frame_index": { "type": "integer" },
                        "score": { "type": "number" },
                        "face_detected": { "type": "boolean" }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "composition": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "method": { "type": "string", "description": "如 'rule_of_thirds_analysis'" },
            "issues": { "type": "array", "items": { "type": "string" } }
          }
        },
        "temporal_coherence": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "method": { "type": "string", "description": "如 'frame_to_frame_embedding_delta'" },
            "max_delta": { "type": "number" },
            "avg_delta": { "type": "number" }
          }
        },
        "face_tracking": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "faces_tracked": { "type": "integer" },
            "tracking_loss_frames": { "type": "array", "items": { "type": "integer" } }
          }
        }
      }
    },
    "retry_recommendation": {
      "type": "object",
      "properties": {
        "should_retry": { "type": "boolean" },
        "retry_reason": { "type": "string" },
        "suggested_parameter_changes": {
          "type": "object",
          "description": "建议调整的生成参数"
        }
      }
    }
  }
}
```

---

## 5. 状态汇总

| Phase | 名称 | 状态 | 备注 |
|-------|------|------|------|
| 1 | 基础设施 | ✅ 已完成 | — |
| 2 | 事件图谱引擎 | ❌ 未实现 | 核心编剧模块 |
| 3 | 角色资产 | ✅ 需增强 | ComfyUI prompt 迁移 |
| 4 | 智能路由 | ✅ 已完成 | — |
| 5 | 视频生成 | ✅ 已完成 | — |
| 6 | 质量检查 | ✅ 需增强 | OM frame_sampler 替代 |
| 7 | 粗剪 | ✅ 需增强 | OM video_trimmer + silence_cutter |
| 8 | 后处理 | ❌ 未实现 | OM 全套后处理工具 |
| 9 | 全流程集成 | ✅ 已完成 | — |

**实现进度**: 7/9 Phase 可用（其中 3 个需增强），2 个未实现（Phase 2, Phase 8）。

---

## 6. 约束与约定

### 6.1 API 约束
- **Agent Plan Endpoint**: `/api/plan/v3/`
- **Seedream**: `doubao-seedream-5.0-lite`，同步接口 `/images/generations`
- **Seedance**: `doubao-seedance-2.0-fast`，异步接口 `/contents/generations/tasks`
- **Seedance 参数**: 全部顶层，`watermark: false`

### 6.2 硬件约束
- M4 Mac 16GB，不跑本地模型
- ComfyUI 只参考工作流/prompt，不做本地推理

### 6.3 架构约定
- 输入是"任意文本"，不特指小说/章节
- OCC（OpenChatCut）已归档，OM（OpenMontage）是核心引擎
- 所有 Phase 输出写入工作目录，便于回溯和调试

### 6.4 目录结构约定
```
{project_root}/
├── characters/          # Phase 3 输出
├── shots/               # Phase 4/5/6 输出
│   ├── S1/
│   │   ├── SHOT_META.json
│   │   ├── output.mp4
│   │   └── frames/
│   └── ...
├── STORYBOARD.json      # Phase 2 输出
├── CHARACTERS.json      # Phase 2 输出
├── consistency_report.json  # Phase 6 输出
├── stitched.mp4         # Phase 7 输出
└── polished.mp4         # Phase 8 输出（最终交付）
```
