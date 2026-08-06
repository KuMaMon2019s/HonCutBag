"""Remove universal image-quality downgrade phrases from HonCut prompts."""
import re
REPLACEMENTS={"film grain":"subtle cinematic texture","胶片颗粒":"轻微电影质感","imperfect focus":"","失焦":"","edges not perfectly sharp":"","slight natural deviation":"","not completely stable":"","blurry background":"background bokeh, subject in sharp focus","柔焦":"","朦胧感":""}
def sanitize_quality_prompt(prompt:str)->str:
    value=prompt
    for phrase,replacement in REPLACEMENTS.items(): value=re.sub(re.escape(phrase),replacement,value,flags=re.IGNORECASE)
    value=re.sub(r"\s*,\s*,+",", ",value); return re.sub(r"\s{2,}"," ",value).strip(" ,")
def find_quality_downgrades(prompt:str)->list[str]: return [phrase for phrase in REPLACEMENTS if re.search(re.escape(phrase),prompt,re.I)]
