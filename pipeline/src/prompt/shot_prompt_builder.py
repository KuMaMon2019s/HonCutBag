"""HonCut five-layer cinematography prompt builder."""
from typing import Any
SHOT={"wide":"wide shot capturing full scene","medium":"medium shot from waist up","close_up":"close-up focusing on face or detail","establishing":"establishing shot setting the location"}
MOVE={"dolly_in":"slow dolly in toward subject","tracking_left":"tracking shot moving left alongside subject","handheld":"handheld camera with natural movement","orbital":"orbital camera circling subject"}
LIGHT={"high_key":"bright high-key lighting, minimal shadows","low_key":"dramatic low-key lighting with deep shadows","golden_hour":"warm golden hour sunlight","neon":"neon-lit with vibrant color spill"}
DOF={"shallow":"shallow depth of field with bokeh","medium":"medium depth of field","deep":"deep focus with everything sharp"}
def build_shot_prompt(scene:dict[str,Any],style_context:dict[str,Any]|None=None)->str:
    sl=scene.get("shot_language",{}); layers=[]
    camera=[f"{sl['lens_mm']}mm lens" if sl.get("lens_mm") else "",DOF.get(sl.get("depth_of_field"),"")]
    if any(camera): layers.append(", ".join(filter(None,camera)))
    movement=[SHOT.get(sl.get("shot_size"),sl.get("shot_size","")),MOVE.get(sl.get("camera_movement"),sl.get("camera_movement","") if sl.get("camera_movement")!="static" else "")]
    if any(movement): layers.append(", ".join(filter(None,movement)))
    layers.append(". ".join(filter(None,[scene.get("description",""),", ".join(scene.get("texture_keywords",[]))])))
    lighting=[LIGHT.get(sl.get("lighting_key"),sl.get("lighting_key","")),sl.get("color_temperature","")]
    if any(lighting): layers.append(", ".join(filter(None,lighting)))
    if style_context and (style_context.get("visual_language",{}).get("aesthetic") or style_context.get("mood")): layers.append("Style: "+(style_context.get("visual_language",{}).get("aesthetic") or style_context["mood"]))
    return ". ".join(filter(None,layers))
def build_batch_prompts(scenes,style_context=None): return [{"scene_id":s.get("id","unknown"),"prompt":build_shot_prompt(s,style_context),"hero_moment":s.get("hero_moment",False)} for s in scenes if s.get("type")!="transition"]
