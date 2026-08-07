---
name: "HonCut Cinematic Default"
version: "1.0"
tags:
  - cinematic
  - warm-tones
  - film-grain

style_prompt_short: >
  Cinematic warm-toned visual style with film grain texture,
  golden hour lighting, and shallow depth of field.

style_prompt_full: >
  Cinematic film style with warm color palette (#F5DEB3 wheat backgrounds,
  #2C1810 deep brown shadows, #FF8C42 amber highlights). Shallow depth of field
  with bokeh. Natural film grain texture. Golden hour lighting with soft
  directional key light from camera-left. Colors should feel organic and
  inviting — avoid oversaturated neon tones. Composition follows rule of thirds.
  Motion should be smooth and deliberate — no jarring cuts within shots.
  Overall mood: nostalgic, intimate, cinematic.

colors:
  primary:
    - name: "Wheat"
      hex: "#F5DEB3"
      role: "warm background tones, skin highlights"
    - name: "Deep Brown"
      hex: "#2C1810"
      role: "shadows, grounding elements"
  accent:
    - name: "Amber"
      hex: "#FF8C42"
      role: "key highlights, rim lighting, emphasis"
  neutral:
    - name: "Slate"
      hex: "#4A5568"
      role: "supporting elements, text overlays"

typography:
  display:
    family: "Inter"
    weight: "bold"
    style: "sentence case, tight tracking for titles"
  body:
    family: "Inter"
    weight: "regular"
    style: "natural reading size"
  caption:
    family: "Inter"
    weight: "medium"
    style: "small, for subtitles and labels"

layout:
  grid: "Rule of thirds, 16:9 cinematic frame"
  alignment: "Centered subject with leading lines"
  aspect_ratio: "16:9"

motion:
  transitions:
    - "crossfade"
    - "slow push-in"
  animation_style: "Smooth, deliberate camera movement. No sudden pans."
  pacing: "Measured, contemplative rhythm"

mood:
  keywords:
    - "cinematic"
    - "nostalgic"
    - "intimate"
    - "warm"
  era: "contemporary"
  avoid:
    - "neon colors"
    - "oversaturated tones"
    - "harsh fluorescent lighting"
    - "digital/CGI look"
---
