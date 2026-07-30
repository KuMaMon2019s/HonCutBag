# 角色描述约束文件（Character Description Schema）

> 本文件定义 LLM 提取角色信息时的**最小字段集**，确保不遗漏关键视觉细节。
> 所有颜色描述**必须包含具体色值**（十六进制如 `#FF0000` 或 RGB 如 `rgb(255,0,0)`）。

---

## Schema 定义

```yaml
角色描述 Schema:
  # ===== 基本信息（必填）=====
  name: string          # 角色名称
  gender: string        # 性别（男/女/中性）
  age_range: string     # 年龄段（少年/青年/中年/老年）

  # ===== 外貌（必填）=====
  appearance:
    hair:
      color: string     # 发色（必须包含具体色值描述，如"火红色 #C00000"）
      style: string     # 发型（如"凌乱长发，发丝拂面"）
      length: string    # 长度（短发/中长发/长发）

    face:
      shape: string     # 脸型（如"瓜子脸"、"方脸"）
      eyes: string      # 眼睛描述（形状、颜色、神态，如"狭长凤眼，琥珀色瞳孔"）
      skin: string      # 肤色（如"清透瓷白 #F5E6D3"）
      expression: string # 表情/神态（如"疯批感，柔和感情的眼神"）

    body:
      height: string    # 身高（如"178cm"）
      build: string     # 体型（如"修长挺拔"）

  # ===== 服装（必填）=====
  clothing:
    top: string         # 上装（如"黑色带金属装饰的夹克"）
    bottom: string      # 下装（如"黑色机车裤"）
    accessories: string # 配饰（如"金属项链、手环"）
    color_palette:      # 配色板（必须包含十六进制色值）
      primary: string   # 主色（如"#000000 黑色"）
      secondary: string # 辅色（如"#C00000 火红色"）
      accent: string    # 点缀色（如"#CCCCCC 银灰色"）

  # ===== 特征与风格（可选）=====
  features:
    distinctive: string # 标志性特征（如"极致妖孽的容貌"）
    mood: string        # 气质/氛围（如"阴郁感，潇洒"）
    style: string       # 风格标签（如"真人写实，现代风格"）
```

---

## 字段必填/可选标注

| 字段路径 | 必填 | 说明 |
|---------|------|------|
| `name` | ✅ 必填 | 角色名称，不可为空 |
| `gender` | ✅ 必填 | 男/女/中性 |
| `age_range` | ✅ 必填 | 少年/青年/中年/老年 |
| `appearance.hair.color` | ✅ 必填 | 必须含色值 |
| `appearance.hair.style` | ✅ 必填 | |
| `appearance.hair.length` | ✅ 必填 | |
| `appearance.face.shape` | ✅ 必填 | |
| `appearance.face.eyes` | ✅ 必填 | |
| `appearance.face.skin` | ✅ 必填 | 必须含色值 |
| `appearance.face.expression` | ⚪ 可选 | 无明确描述时留空 |
| `appearance.body.height` | ✅ 必填 | |
| `appearance.body.build` | ✅ 必填 | |
| `clothing.top` | ✅ 必填 | |
| `clothing.bottom` | ✅ 必填 | |
| `clothing.accessories` | ⚪ 可选 | 无配饰时留空 |
| `clothing.color_palette.primary` | ✅ 必填 | 必须含十六进制色值 |
| `clothing.color_palette.secondary` | ✅ 必填 | 必须含十六进制色值 |
| `clothing.color_palette.accent` | ⚪ 可选 | 无点缀色时留空 |
| `features.distinctive` | ⚪ 可选 | |
| `features.mood` | ⚪ 可选 | |
| `features.style` | ⚪ 可选 | |

---

## 示例：红发少年

```yaml
name: 焰（Homura）
gender: 男
age_range: 少年

appearance:
  hair:
    color: "火红色 #C00000"
    style: "凌乱长发，发丝拂面，额前碎发"
    length: 长发

  face:
    shape: "瘦削瓜子脸"
    eyes: "狭长凤眼，琥珀色瞳孔 #D4A017，眼神锐利"
    skin: "清透瓷白 #F5E6D3"
    expression: "疯批感，柔和感情的眼神"

  body:
    height: "178cm"
    build: "修长挺拔"

clothing:
  top: "黑色带金属装饰的夹克，内搭深灰色高领衫"
  bottom: "黑色机车裤，膝盖处有磨损设计"
  accessories: "银色金属项链、黑色皮革手环"
  color_palette:
    primary: "#000000 黑色"
    secondary: "#C00000 火红色"
    accent: "#CCCCCC 银灰色"

features:
  distinctive: "极致妖孽的容貌，左眼下方有一颗泪痣"
  mood: "阴郁感，潇洒"
  style: "真人写实，现代风格"
```

---

## 约束规则

1. **颜色必须含色值**：所有涉及颜色的字段（发色、肤色、服装配色）必须附带十六进制色值（如 `#FF0000`）或 RGB 值，确保视觉还原一致性。
2. **必填字段不可省略**：标注为 ✅ 必填的字段在提取时必须填充，若原文未提及则标注为"未提及"。
3. **描述要具体**：避免模糊描述（如"好看"），使用可量化的视觉描述（如"狭长凤眼"、"178cm"）。
4. **风格标签统一**：`features.style` 使用固定标签体系，如"真人写实"、"动漫风格"、"赛博朋克"等。
