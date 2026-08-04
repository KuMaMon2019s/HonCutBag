# HonCut I2V Pipeline Test Issues Report

**Test Date:** 2026-08-02  
**Input:** `pipeline/input/i2v_nature.txt` (landscape-only script, no characters)  
**Output Directory:** `pipeline/output/i2v_nature_test`  
**Pipeline Status:** PARTIAL (67.14s total)

---

## Issue #1: Phase 1 API Authentication Error (401)

**Phase:** Phase 1 - 导演规划 (Director Planner)  
**Status:** ❌ Failed  
**Duration:** 0.53s

### Error Message
```
Error code: 401 - {'error': {'code': 'AuthenticationError', 'message': 'The API key or AK/SK in the request is missing or invalid. Request id: 02178568485889816cf7c6382c9de3ded6c40bd16bc9a114c7d0c', 'param': '', 'type': 'Unauthorized'}}
```

### Root Cause Analysis
The `ARK_AGENT_API_KEY` in `pipeline/.env` is invalid or expired. Phase 1 attempts to call the Volcengine API but fails authentication.

### Impact
- Phase 1 is marked as "degraded skip" (降级跳过)
- Pipeline continues to Phase 2+ without director planning data
- Not a critical blocker since Phase 1 is optional (M1 incremental module)

### Fix Required
- Verify the ARK_AGENT_API_KEY is valid
- Check if the API key has expired or been revoked
- Regenerate the API key from Volcengine console if needed

---

## Issue #2: Phase 5 Seedance API Model Incompatibility (404)

**Phase:** Phase 5 - 视频生成 (Seedance)  
**Status:** ❌ Failed (all 10 shots)  
**Duration:** 4.88s

### Error Message
```
Seedance API 404: {"error":{"code":"UnsupportedModel","message":"The requested model does not support the agent plan feature. Please refer to the documentation at https://www.volcengine.com/docs/82379/2366394 to select a compatible model. Request id: 02178568492108086f296e4c58ddf650991f54d763d292a9eb3d6","param":"","type":""}}
```

### Root Cause Analysis
The current implementation uses model `doubao-seedance-2.0-mini` (hardcoded in `seedance_client.py` line 22), but this model **does not support the Agent Plan feature**.

The API key `ARK_AGENT_API_KEY` is specifically for Agent Plan (按量计费), which has a different set of compatible models compared to the standard API.

**Key Finding:** The error message explicitly states:
> "The requested model does not support the agent plan feature"

This means:
1. The API key is valid (not a 401 error)
2. The model `doubao-seedance-2.0-mini` is incompatible with Agent Plan
3. Need to use a model that supports Agent Plan

### Impact
- All 10 video generation attempts failed
- No video clips were generated
- Phase 7 (Assembly) failed due to missing video files
- Pipeline status: PARTIAL

### Fix Required
**Option A: Use Agent Plan-compatible model**
- Check Volcengine documentation for Agent Plan-compatible Seedance models
- Update `seedance_client.py` to use the correct model name
- Possible candidates (need verification):
  - `doubao-seedance-1-0` (older version, may support Agent Plan)
  - Other Seedance variants

**Option B: Use standard API instead of Agent Plan**
- Switch from `ARK_AGENT_API_KEY` to `ARK_API_KEY` (standard API)
- Update `config.py` and `seedance_client.py` to use standard endpoint
- Standard API may have different pricing/billing model

**Option C: Implement model fallback logic**
- Try Agent Plan-compatible model first
- Fall back to standard API if Agent Plan fails
- Add retry logic with different models

### Verification Needed
- [ ] Check Volcengine docs for Agent Plan-compatible models
- [ ] Test with alternative model names
- [ ] Verify API key permissions and billing plan

---

## Issue #3: Quality Gate Failed (Character Consistency)

**Phase:** Phase 6 - 一致性守卫  
**Status:** ⚠️ Quality gate failed  
**Score:** 0 < 70

### Root Cause Analysis
The quality gate checks character consistency across video clips, but:
1. No video clips were generated (Phase 5 failed)
2. Character consistency score = 0 (no data to evaluate)
3. Quality gate recommends rollback to Phase 5

### Impact
- Pipeline suggests retrying Phase 5
- Not a code bug - expected behavior when Phase 5 fails
- Quality gate logic is working correctly

### Fix Required
- Fix Issue #2 first (Seedance API model compatibility)
- Once videos are generated, quality gate should pass

---

## Issue #4: Phase 7 Assembly Failed (No Video Clips)

**Phase:** Phase 7 - 组装引擎 (Assembly)  
**Status:** ❌ Failed  
**Error:** "No video clips found"

### Root Cause Analysis
Phase 7 attempts to assemble video clips into a final video, but:
1. Phase 5 failed to generate any video clips
2. No `.mp4` files in `shots/S*/` directories
3. Assembly engine has nothing to work with

### Impact
- No `raw_assembly.mp4` generated
- Phase 8 (Post-Production) skipped
- No final video output

### Fix Required
- Fix Issue #2 first (Seedance API model compatibility)
- Once videos are generated, assembly should succeed

---

## Issue #5: OM Tools Unavailable (lib.scoring module missing)

**Phase:** Phase 2.5 - 故事板图片生成  
**Status:** ⚠️ Degraded (fallback to Seedream API)  
**Duration:** 61.58s

### Error Message
```
⚠ OM image_selector 不可用: No module named 'lib.scoring'
```

### Root Cause Analysis
The OpenMontage (OM) tools require `lib.scoring` module, which is not available in the current environment. The pipeline correctly falls back to the Seedream API.

### Impact
- Phase 2.5 still succeeded using Seedream API
- Generated `storyboard.png` (4MB, quality grade A)
- Not a critical issue - fallback mechanism working as designed

### Fix Required
- Optional: Install missing `lib.scoring` module
- Or: Continue using Seedream API fallback (working solution)

---

## Issue #6: Phase 2 Skipped (User-Specified)

**Phase:** Phase 2 - 编剧引擎  
**Status:** ⊘ Skipped (user-specified with `--skip-phase 2`)

### Root Cause Analysis
The test was run with `--skip-phase 2` flag to reuse existing `STORYBOARD.json` and `CHARACTERS.json` from a previous run.

### Impact
- Pipeline used pre-generated storyboard data (10 shots)
- Not an issue - intentional skip for testing purposes

### Note
The existing `STORYBOARD.json` has 10 shots but lacks detailed descriptions (all shot descriptions are empty), which may contribute to quality issues in later phases.

---

## Summary

### Critical Blockers
1. **Seedance API Model Incompatibility** (Issue #2) - Must fix to generate videos
2. **API Key Authentication** (Issue #1) - Should verify/refresh API key

### Working Components
- ✅ Phase 2.5: Storyboard image generation (Seedream API)
- ✅ Phase 3: Character factory (correctly handles empty character list)
- ✅ Phase 4: Orchestrator (creates shot directories)
- ✅ Phase 6: Quality gate (correctly identifies missing videos)
- ✅ Fallback mechanisms (OM → API fallbacks working)

### Pipeline Flow
```
Phase 1 (❌ 401 Auth) → Phase 2 (⊘ Skipped) → Phase 2.5 (✅ Done) → 
Phase 3 (⊘ Skipped) → Phase 4 (✅ Done) → Phase 5 (❌ 404 Model) → 
Phase 6 (⚠️ Quality Failed) → Phase 7 (❌ No Videos) → Phase 8 (⊘ Skipped)
```

### Next Steps
1. **Priority 1:** Fix Seedance model compatibility (Issue #2)
   - Research Agent Plan-compatible models
   - Update `seedance_client.py` with correct model name
   - Test with single shot before full pipeline
   
2. **Priority 2:** Verify API key (Issue #1)
   - Check if ARK_AGENT_API_KEY is still valid
   - Regenerate if necessary
   
3. **Priority 3:** Re-run full pipeline
   - After fixes, run complete pipeline without `--skip-phase 2`
   - Verify all phases complete successfully
   - Generate final video output

---

## Files Modified
- None (no code changes made during this test)

## Files Generated
- `STORYBOARD.json` (10 shots, pre-existing)
- `CHARACTERS.json` (0 characters, landscape-only)
- `storyboard.png` (4MB, quality A)
- `pipeline_report.json` (detailed phase results)
- `checkpoint.db` (LangGraph state)
- `shots/S01/` through `shots/S10/` (empty directories)

## Recommendations
1. Add model compatibility check before submitting API requests
2. Implement better error messages for Agent Plan vs Standard API confusion
3. Add pre-flight validation for API keys and model availability
4. Consider adding a "dry-run with mock videos" mode for testing pipeline flow without API costs

---

## 2026-08-03 更新

### Issue #7: Phase 5 本地 API 轮询超时

**Phase:** Phase 5 - 视频生成  
**Status:** ❌ Failed (timeout after 600s)  
**Duration:** 2788.82s (46 minutes)

#### Error Message
```
[local_poll 1/60] status=running progress=0%
...
[local_poll 60/60] status=running progress=0%
TimeoutError: Local video task timed out after 600s
```

#### Root Cause Analysis
1. 本地 API 路由正确配置，任务成功提交到 Bridge
2. Bridge 任务 ID: `5eddbd3e`，实际进度 25%（仍在运行）
3. 管线轮询 60 次 × 10 秒 = 600 秒后超时退出
4. 超时后降级到 ARK API，但 ARK 配额超限（429）

#### Fix Required
- 增加轮询超时时间（从 600s 增加到 1800s）
- 优化视频生成参数（减少帧数或分辨率）

---

### Issue #8: local_video_client.py NoneType 错误

**Phase:** Phase 5 - 视频生成  
**Status:** ✅ Fixed  
**Duration:** N/A

#### Error Message
```
'NoneType' object has no attribute 'lower'
```

#### Root Cause Analysis
`local_video_client.py` 的 `poll()` 函数在处理错误响应时，调用 `error_msg.lower()` 但 `error_msg` 可能为 `None`。

#### Fix Applied
```python
# Before:
if "error" in data:
    error_msg = data["error"]
    if "not found" in error_msg.lower():  # ← NoneType error

# After:
if "error" in data and data["error"] is not None:
    error_msg = data["error"]
    if "not found" in error_msg.lower():  # ← Safe
```

#### Verification
- ✅ 代码已修复
- ✅ 语法检查通过

---

### Issue #9: ARK API 配额超限

**Phase:** Phase 5 - 视频生成  
**Status:** ❌ Failed (429)  
**Duration:** N/A

#### Error Message
```
⚠ S01: 配额超限(429)，等待 60s 后重试 (2/3)...
⚠ S01: 配额超限(429)，等待 120s 后重试 (3/3)...
```

#### Root Cause Analysis
本地 API 超时后，管线降级到 ARK Agent Plan API，但 API 配额已超限。

#### Fix Required
- 检查 ARK_AGENT_API_KEY 配额
- 购买更多配额或等待配额重置
- 优先使用本地 API，避免依赖 ARK

---

## 2026-08-03 代码修改

### 新增文件
- `pipeline/src/local_video_client.py` - 本地视频 API 客户端（对接 Windows ComfyUI Bridge）

### 修改文件
- `pipeline/src/config.py` - 新增本地 API 配置
- `pipeline/src/director_planner.py` - 修复 ARK_BASE_URL
- `pipeline/src/pipeline_runner.py` - 新增本地 API 路由逻辑
- `pipeline/src/quality_gate.py` - 新增 landscape-only fallback
- `pipeline/src/seedance_client.py` - 模型路由透传

---

## 当前状态（2026-08-03）

### 已修复
- ✅ 本地 API 路由配置
- ✅ NoneType 错误处理
- ✅ ARK_BASE_URL 修复

### 待修复
- ❌ Phase 5 轮询超时（需要增加到 1800s）
- ❌ ARK API 配额超限（需要购买配额）
- ❌ ComfyUI 服务不稳定（Windows 端需要重启）

### 下一步
1. 修复轮询超时问题
2. 重启 Windows ComfyUI 服务
3. 重新运行管线验证 Phase 5 → Phase 7

---

# ═══════════════════════════════════════════════════════════
# 2026-08-04 Codex CLI 攻坚档案（Handoff for OpenAI Codex CLI）
# ═══════════════════════════════════════════════════════════

## Issue #10【P0 核心阻塞】Seedream 图片生成 429 限流导致角色图/故事板全空，Phase 5 无参考图白跑

**涉及 Phase:** Phase 2.5 故事板图 / Phase 3 角色三视图 / M2 分镜图
**状态:** ❌ 反复复现（v3 / v4 / v5 三轮全中）
**影响:** 角色图 0 张 → Phase 5 视频生成没有任何角色参考图 → 白跑 GPU

### 症状
- `characters/{lin_xiao,chen_yang}/` 目录被创建，但 **0 张 PNG**（v3 只有 JSON 卡片，v5 连 JSON 都没有）
- `storyboard.png` 未生成
- `pipeline_run.log` **0 字节**（`conda run | tee` 缓冲，实时日志不可见，只能靠 checkpoint.db / 文件落盘推断）
- 管线却照样推进到 Phase 5 并向 Bridge 提交视频任务（无参考图）

### 实测关键事实（2026-08-04）
1. **单张手动调用 Seedream 能成功**：冷却 2 分钟后，单独 POST `/images/generations`（size=1920x1920）一次就返回 200 + 图片 URL。说明 API 本身可用、key 有效、非配额耗尽。
2. **400 与 429 是两回事**：
   - 用 size=1024x1024 → 400 `InvalidParameter: image size must be at least 3686400 pixels`（1920×1920 才达标）
   - 管线内短时间连发 → 429 `Too Many Requests`
3. **结论：问题是"并发/突发请求触发限流"，不是 API 故障或 key 失效。**

### 根因分析
管线在 Phase 2.5/3 阶段**密集连发** Seedream 请求，触发限流：
- Phase 2.5：1 张 storyboard（1920x1920）
- Phase 3：每个角色 1 张 character_sheet（1920x1920，2x2 四视图）→ `batch_generate` / LangGraph `Send("generate_character")` 并行 fan-out
- M2：每镜头 1 张分镜图 → `ThreadPoolExecutor(max_workers=3)` 并发（pipeline_runner.py:1033-1035）
- 总计约 **8+ 张**图在几分钟内打出，且每张都是 1920x1920 大图，限流窗口扛不住。

**现有重试为何救不了：**
- `character_factory.py:532` 指数退避 `[5,15,45]`（共 65s，3 次）
- `pipeline_runner.py:273 _retry_with_policy`：`backoff_factor=2.0` → 间隔仅 1s/2s/4s（**间隔太短，等于没退避**）
- 重试耗尽后 Phase 3 抛异常 → 质检红线"角色图片缺失"拦截，但 Phase 5 仍可能在部分路径继续

### 涉及文件与行号（给 Codex 定位）
| 文件 | 位置 | 说明 |
|------|------|------|
| `pipeline/src/seedream_client.py` | L24-28, L48+ | 端点 `/api/plan/v3/images/generations`，model `doubao-seedream-5.0-lite`，size 默认 1920x1920 |
| `pipeline/src/character_factory.py` | L528-575 | sheet 生成 + 429 重试（5/15/45s），`crop_character_sheet` 切四视图 |
| `pipeline/src/pipeline_runner.py` | L273-292 | `_retry_with_policy`（backoff 间隔太短） |
| `pipeline/src/pipeline_runner.py` | L315-339 | `task_call_seedream` / `task_generate_character`（LangGraph @task） |
| `pipeline/src/pipeline_runner.py` | L395-403 | Phase 3 `Send("generate_character")` 并行 fan-out |
| `pipeline/src/pipeline_runner.py` | L1033-1035 | M2 分镜图 `ThreadPoolExecutor(max_workers=3)` |
| `pipeline/src/pipeline_runner.py` | L1076-1145 | Phase 2.5 storyboard + Seedream fallback |

### 建议修复方向（供 Codex 评估，按优先级）
**方向 A｜全局限流器（推荐，治本）**
- 给 Seedream 调用加进程级令牌桶/信号量：同一时刻最多 1 个在途图片请求，请求间强制最小间隔（如 3-5s）
- 所有调用点（Phase 2.5 / Phase 3 / M2）共享同一个限流器，而非各自重试

**方向 B｜串行化图片生成**
- Phase 3 角色图改为逐个串行生成（去掉 Send fan-out 的并发）
- M2 分镜图 `max_workers=3` → `max_workers=1`
- 简单稳妥，代价是总耗时变长（但图片生成不是瓶颈大头）

**方向 C｜加大退避 + 抖动**
- `_retry_with_policy` backoff 从 1s/2s/4s 改为 30s/60s/120s + 随机抖动（jitter）
- character_factory 的 [5,15,45] 同步加大到 [30,60,120]
- 治标，若限流窗口长仍可能不够，但改动最小

**方向 D｜失败快速熔断**
- Phase 2.5/3 图片生成失败达到阈值时，立即终止管线（别让 Phase 5 无参考图白跑 GPU）
- 在 Phase 4/5 前加前置检查：`characters/*/front.png` 不存在则 abort

### 验收标准
- [ ] 单独跑 Phase 2.5+3：storyboard.png 生成成功、每个角色 ≥4 张视图 PNG（front/side/back/closeup）
- [ ] 全程日志可见（解决 `conda run | tee` 缓冲，建议 `python -u` 或直接重定向）
- [ ] 复现限流时能自动退避重试成功，而非抛异常
- [ ] Phase 5 提交时 content[] 里能看到角色三视图 URL（证明参考图真正注入）

---

## Issue #11【P1 环境】Bridge 孤儿任务堆积 + /generate_zip 端点不存在

**状态:** ⚠️ 需 Windows 端配合
- v4/v5 被手动 kill 后，Bridge 留下孤儿 running 任务（如 `7a4c6983 / f6dc1fa7 / e7f33124 / 8a4620c3` 卡 75%），无清理 API，会浪费 GPU
- `/generate_zip` `/upload` 均 404 —— **zip 打包方案作废**，已改用 Kumamon 的 content[] 契约
- content[] 契约（权威）：`POST /generate` body `{"content":[{type:text},{type:image_url,image_url:{url},role,priority}]}`，图片必须**公网 https URL**（base64 data URL 不支持），最多 9 张，Bridge 按 priority 挑 3 张，返回 `{task_id, images_used, warnings}`

### 已完成的多图注入实现（commit 02be36b）
- `asset_packager.build_content_for_shot()`：收集三视图+分镜图+故事板 → 上传 TOS 拿签名 URL → 组装 content[]（role: first_frame/reference_image，priority: high/medium）
- TOS 凭证已在 `.env`（TOS_ACCESS_KEY 等），实测 front.png 上传成功、公网 HTTP 200
- `local_video_client.submit()/generate_video()` 已支持 `content` 参数

---

## Issue #12【P2 可观测性】pipeline_run.log 始终 0 字节

**根因:** `conda run -n honcut python ... | tee log` 的 stdout 被缓冲，且子进程输出未实时 flush
**影响:** 无法实时看 Phase 进度/报错，只能靠文件落盘和 checkpoint.db 推断
**建议:** 改用 `python -u`（unbuffered）+ `PYTHONUNBUFFERED=1`，或不走 tee 直接 `> log 2>&1`，或在关键节点写 `events.jsonl`

---

## 2026-08-04 提交记录
| commit | 内容 |
|--------|------|
| `02be36b` | 429 retry + content[] contract with TOS upload for Phase 3/5 |
| `f3a3c01` | multi-image asset packaging with zip fallback to base64（zip 方案已作废） |
| `ddc9c75` | download verification + bgm_path unbound + Phase 5 silent shot loss |
| `353d25b` | Bridge running/100 quirk + configurable concurrency + zero ARK fallback |
| `b357f89` | progress-based timeout in poll() |
| `4c5dac1` | 智能分镜逻辑（按剧本长度自动算镜头数） |

## 环境快照（2026-08-04）
- Mac 端管线：`/Users/soda/projects/honcut/pipeline/src`，conda env `honcut`
- Bridge：`http://192.168.31.221:9100`（Windows ComfyUI Wan2.2 TI2V-5B，8GB VRAM），curl 必须加 `--noproxy '*'`
- ARK：Agent Plan，endpoint `/api/plan/v3`，模型 `doubao-seedream-5.0-lite`（图片）
- TOS：bucket `hermess`，endpoint `tos-cn-beijing.volces.com`
- 红线：视频生成只走本地 API，绝不降级 ARK 视频模型（成本控制）
