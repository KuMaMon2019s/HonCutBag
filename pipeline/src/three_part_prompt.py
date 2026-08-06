"""Toonflow three-part prompt: visual content, lighting, then concise style."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ThreePartPrompt:
    visual:str; lighting:str; style:str
    def render(self)->str: return f"【画面】{self.visual}。\n\n【光影】{self.lighting}。\n\n【风格】{self.style}。"
def build_three_part_prompt(visual:str,lighting:str,style:str,quality_lock:str="高清画质，主体清晰",prohibitions:str="禁止画外字幕、水印、UI 文字")->str:
    if not visual.strip(): raise ValueError("visual content is required")
    style_text="，".join(filter(None,(style.strip(),quality_lock.strip(),prohibitions.strip())))
    return ThreePartPrompt(visual.strip(),lighting.strip() or "自然光影",style_text).render()
def validate_three_part_prompt(prompt:str)->list[str]:
    labels=["【画面】","【光影】","【风格】"]; errors=[]
    if any(label not in prompt for label in labels): errors.append("missing section")
    elif not (prompt.index(labels[0])<prompt.index(labels[1])<prompt.index(labels[2])): errors.append("sections out of order")
    return errors
