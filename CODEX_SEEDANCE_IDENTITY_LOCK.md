# Codex 任务书：Seedance Identity-Lock + 运镜模板

## 任务目标

将 OpenMontage seedance-2-0 skill 中的 identity-lock 和运镜模板整理成 HonCut 可用的代码，修改 `pipeline/src/phases/storyboard_generator.py` 的 `_build_shot_prompt()` 函数。

## 背景问题

V5 测试视频存在三个问题：
1. **发型穿帮** — 角色在不同镜头中发型不一致
2. **运镜割裂** — 所有镜头都是 static camera，没有运镜变化
3. **字幕是旁白** — 没有角色对白（这个任务不处理，另一个任务处理）

## 来源参考

- OpenMontage skill: `/Users/soda/projects/OpenMontage/.agents/skills/seedance-2-0/SKILL.md`
- HonCut 目标文件: `pipeline/src/phases/storyboard_generator.py`
- 角色数据: `CHARACTERS.json` (包含 appearance.hair, appearance.face, appearance.clothing 等结构化字段)

---

## 修改 1: Identity-Lock 模板

### 当前代码问题

`_build_shot_prompt()` 函数（约 line 256-330）当前使用 `appearance.summary`（自然语言句子）作为 identity anchor，效果不好。

### 需要改成

从 CHARACTERS.json 读取结构化外观字段，逐字拼接成英文视觉特征列表，并添加 identity-lock 短语。

### 参考模板（来自 seedance-2-0 SKILL.md）

```
[reference_image: character_ref.png]
[identity_lock]
The same character — {hair}, {face}, {clothing} — consistent across all shots, no drift or deformation. Do not alter clothing category or primary color.
```

### Identity-anchor 短语（必须全部叠加）

```python
IDENTITY_LOCK_PHRASES = [
    "the same character",
    "consistent across all shots",
    "maintain exact appearance from reference image",
    "no deformation, no drift, no face morph",
    "Do not alter clothing category or primary color",
]
```

### 实现要求

1. 从 `characters` 参数中提取角色的 `appearance` 字典
2. 读取 `hair`, `face`, `clothing`, `distinguishing` 字段
3. 将中文字段翻译成英文（或直接用中文，但 Seedance 对英文 prompt 效果更好）
4. 拼接成 identity-lock 块
5. 在所有有角色的 shot prompt 中逐字重复

### 示例输出

对于角色"林晓"（CHARACTERS.json 中 hair="黑色长直发及肩", clothing="浅米色棉麻短袖衬衫+浅蓝色A字牛仔半身裙+白色低帮帆布鞋+米色编织斜挎小包"）：

```
[identity_lock]
The same character — black long straight hair to shoulders, oval face with willow-leaf eyebrows and gentle almond eyes, light beige cotton linen short-sleeve shirt with light blue A-line denim half skirt and white low-top canvas sneakers and beige woven crossbody bag — consistent across all shots, no drift or deformation. Do not alter clothing category or primary color.
```

---

## 修改 2: 运镜声明模板

### 当前代码问题

所有镜头的 `camera_movement` 默认是 `"static"`，没有运镜变化。

### 需要改成

根据 `shot_intent` 和 `shot_size` 自动选择合适的运镜方式，并确保相邻镜头不重复。

### 运镜模板（来自 seedance-2-0 SKILL.md）

#### 运镜声明 Opener（加在 prompt 开头）

```python
CAMERA_OPENERS = {
    "establishing": "Wide establishing shot, slow cinematic push-in, cinematic lighting, photorealistic, 35mm film quality.",
    "close_up": "Medium close-up, subtle handheld motion, shallow depth of field, photorealistic, 35mm film.",
    "action": "Dynamic tracking shot, cinematic lighting, photorealistic, 35mm film quality, motion blur on fast actions.",
    "reaction": "One continuous shot, natural head movement, photorealistic, 35mm film grain, no cuts, no zoom.",
    "transition": "Slow pan across scene, cinematic lighting, photorealistic, volumetric haze, 35mm film.",
    "atmosphere": "Wide aerial shot, slow drift, golden hour lighting, photorealistic, 35mm film quality.",
}
```

#### 运镜否定短语（显式声明不要什么）

```python
CAMERA_NEGATIONS = {
    "static": "no camera movement, locked tripod, no pan, no tilt, no zoom",
    "slow_pan": "no zoom, no cuts, smooth slow pan only",
    "tracking": "no zoom, no cuts, smooth tracking only",
    "handheld": "no zoom, no cuts, natural handheld movement",
}
```

#### 运镜选择逻辑

```python
# 根据 shot_intent 选择运镜
INTENT_TO_CAMERA = {
    "establishing": "slow_pan",
    "transition": "slow_pan",
    "reveal": "tracking",
    "emotional": "static",  # 情感镜头用静态
    "action": "tracking",
    "atmosphere": "slow_pan",
}

# 确保相邻镜头不重复
def select_camera_movement(shot, prev_shot):
    intent = shot.get("shot_intent", "establishing")
    camera = INTENT_TO_CAMERA.get(intent, "slow_pan")
    
    # 如果和上一个镜头相同，切换到备选
    if prev_shot and prev_shot.get("camera_movement") == camera:
        alternatives = ["slow_pan", "tracking", "static"]
        alternatives.remove(camera)
        camera = alternatives[0]
    
    return camera
```

---

## 修改 3: 8-Part Prompt 模板

### 参考模板（来自 seedance-2-0 SKILL.md）

```
[Shot / framing] + [Camera movement] +
[Subject description — physical detail that must persist across shots] +
[Action beat 1] → [optional cut] → [Action beat 2] +
[Setting / environment] + [Lighting / palette] +
[Style / grade / era] + [Audio — ambient, diegetic, music, dialogue]
```

### HonCut 实现

修改 `_build_shot_prompt()` 返回的 prompt 格式，按 8-part 结构组织：

```python
def _build_shot_prompt(shot, characters, scene_style_map):
    # Part 1: Shot/framing
    shot_size = shot.get("shot_size", "medium")
    framing = SHOT_SIZE_MAP.get(shot_size, "Medium shot")
    
    # Part 2: Camera movement
    camera = select_camera_movement(shot, prev_shot)
    camera_desc = CAMERA_OPENERS.get(shot.get("shot_intent", "establishing"), "")
    
    # Part 3: Subject with identity-lock
    identity_block = build_identity_lock(shot.get("who", []), characters)
    
    # Part 4: Action
    action = shot.get("what", "")
    
    # Part 5: Setting
    setting = shot.get("where", "")
    
    # Part 6: Lighting
    lighting = shot.get("lighting_key", "natural")
    
    # Part 7: Style
    style = "Photorealistic, cinematic, 35mm film quality, no 3D, no cartoon, no VFX aesthetic."
    
    # Part 8: Audio (if generate_audio enabled)
    audio = "Ambient natural sound, no music."
    
    # 组合
    prompt = f"{camera_desc} {identity_block} {action}. {setting}. {lighting} lighting. {style} {audio}"
    
    return prompt
```

---

## 验收标准

1. **Identity-Lock**: 每个有角色的 shot prompt 包含 `[identity_lock]` 块，逐字重复角色外观特征
2. **运镜变化**: 相邻镜头的 camera_movement 不重复，根据 shot_intent 自动选择
3. **8-Part 结构**: prompt 按 8-part 模板组织
4. **测试通过**: `pytest pipeline/tests/ -v` 全部通过
5. **回归测试**: 用 westlake_3shots_mini.txt 跑一次全链路，检查 prompt 输出

---

## 注意事项

- 不要修改 Phase 5 视频生成逻辑，只改 Phase 2 的 prompt 构建
- 保持向后兼容：如果 characters 为空，不添加 identity-lock
- 运镜选择要有 fallback：如果 shot_intent 不在映射表中，默认用 "slow_pan"
- 翻译问题：先用中文，后续可以加翻译层

---

## 参考文件

- OM Skill: `/Users/soda/projects/OpenMontage/.agents/skills/seedance-2-0/SKILL.md`
- 当前实现: `pipeline/src/phases/storyboard_generator.py` line 256-330
- 角色数据示例: `pipeline/output/westlake_seedance_v5/CHARACTERS.json`
- 分镜数据示例: `pipeline/output/westlake_seedance_v5/STORYBOARD.json`
