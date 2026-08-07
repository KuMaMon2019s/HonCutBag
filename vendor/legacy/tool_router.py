#!/usr/bin/env python3
"""
Phase 4.2 - Tool Router
Dynamically select appropriate tools based on shot characteristics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


class ToolRouter:
    """Route shots to appropriate generation tools based on metadata"""
    
    def __init__(self):
        # Tool definitions with capabilities
        self.tools = {
            'image_to_video': {
                'name': 'Seedream + Seedance',
                'description': 'Generate character-consistent video from reference image',
                'capabilities': ['character_lock', 'consistency'],
                'use_when': ['character reference available', 'need character consistency'],
                'api': 'Seedance img2vid',
                'endpoint': 'POST /api/plan/v3/contents/generations/tasks'
            },
            'text_to_video': {
                'name': 'Seedance',
                'description': 'Generate video from text prompt',
                'capabilities': ['motion', 'action', 'atmosphere'],
                'use_when': ['no character lock', 'action/motion focus', 'atmospheric shots'],
                'api': 'Seedance txt2vid',
                'endpoint': 'POST /api/plan/v3/contents/generations/tasks'
            },
            'occ_mcp': {
                'name': 'OCC MCP Tools',
                'description': 'Precise timeline editing with transitions',
                'capabilities': ['precise_cuts', 'transitions', 'timeline'],
                'use_when': ['need precise cuts', 'dissolve transitions', 'timeline control'],
                'api': 'OCC MCP',
                'endpoint': 'http://localhost:5173/api/external-mcp/mcp'
            },
            'om_video_stitch': {
                'name': 'OM video_stitch',
                'description': 'Combine multiple video clips',
                'capabilities': ['stitch', 'crossfade', 'concat'],
                'use_when': ['multiple clips to combine', 'need crossfade', 'assembly'],
                'api': 'Python import',
                'endpoint': str(Path(__file__).resolve().parent.parent / "video_tools" / "tools" / "video" / "video_stitch.py")
            },
            'om_caption_burn': {
                'name': 'OM remotion_caption_burn',
                'description': 'Burn subtitles onto video',
                'capabilities': ['caption', 'subtitle', 'text_overlay'],
                'use_when': ['need captions', 'subtitle overlay'],
                'api': 'Python import',
                'endpoint': str(Path(__file__).resolve().parent.parent / "video_tools" / "tools" / "video" / "remotion_caption_burn.py")
            }
        }
    
    def analyze_shot(self, shot_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze shot and recommend tools based on metadata.
        
        Supports two schemas:
        - StoryboardEngine: characters, mood, motion_type, features
        - Orchestrator: route, ref_type, prompt, caption
        """
        recommendations = []
        
        # Extract shot characteristics from either schema
        characters = shot_meta.get('characters', [])
        ref_type = shot_meta.get('ref_type', '')
        route = shot_meta.get('route', '')
        prompt = shot_meta.get('prompt', '').lower()
        motion_type = shot_meta.get('motion_type', '').lower()
        features = shot_meta.get('features', '').lower()
        name = shot_meta.get('name', '').lower()
        
        # Rule 1: Character lock → image_to_video (Seedream + Seedance)
        has_character_lock = (
            (characters and len(characters) > 0) or
            ref_type == 'character'
        )
        if has_character_lock:
            char_names = characters if characters else [ref_type]
            recommendations.append({
                'tool': 'image_to_video',
                'reason': f"Character lock: {', '.join(char_names)} — use reference image for consistency",
                'priority': 'high',
                'confidence': 0.9
            })
        
        # Rule 2: Action/motion focus without character lock → text_to_video
        is_action = any(kw in motion_type or kw in prompt or kw in name 
                       for kw in ['动作', '运动', 'action', 'motion', '大场面', 'burst', 'rush', 'lunges'])
        is_scene_locked = ref_type in ('scene', 'prop')
        
        if is_action and is_scene_locked and not has_character_lock:
            recommendations.append({
                'tool': 'text_to_video',
                'reason': f"Action/motion shot with scene lock — Seedance txt2vid for dynamic motion",
                'priority': 'high',
                'confidence': 0.85
            })
        elif not has_character_lock and not is_action:
            # Atmospheric/ambient shot
            recommendations.append({
                'tool': 'text_to_video',
                'reason': "Atmospheric shot — Seedance txt2vid",
                'priority': 'medium',
                'confidence': 0.7
            })
        
        # Rule 3: Precise cuts needed → OCC MCP
        needs_precise_cuts = any(kw in features or kw in prompt 
                                for kw in ['精确', '剪辑', 'precise', 'cut', 'no cuts', 'timeline'])
        # Shots that explicitly say "No cuts" benefit from OCC timeline control
        if 'no cuts' in prompt or needs_precise_cuts:
            recommendations.append({
                'tool': 'occ_mcp',
                'reason': "Precise timeline control — OCC MCP for cut management",
                'priority': 'medium',
                'confidence': 0.8
            })
        
        # Rule 4: Multiple clips to combine → OM video_stitch
        needs_stitch = any(kw in features or kw in prompt 
                          for kw in ['拼接', '多段', 'combine', 'stitch', 'multiple clips'])
        if needs_stitch:
            recommendations.append({
                'tool': 'om_video_stitch',
                'reason': "Multiple clips need combination — OM video_stitch with crossfade",
                'priority': 'medium',
                'confidence': 0.75
            })
        
        # Rule 5: Always add caption burn if caption exists
        caption = shot_meta.get('caption', '')
        if caption:
            recommendations.append({
                'tool': 'om_caption_burn',
                'reason': f"Caption overlay: {caption}",
                'priority': 'low',
                'confidence': 0.95,
                'post_process': True
            })
        
        # Default fallback: if no high-priority recommendation, use text_to_video
        if not any(r['priority'] == 'high' for r in recommendations):
            recommendations.insert(0, {
                'tool': 'text_to_video',
                'reason': "Default: no specific requirements",
                'priority': 'medium',
                'confidence': 0.6
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        shot_id = shot_meta.get('id') or shot_meta.get('shot_id', 'unknown')
        
        return {
            'shot_id': shot_id,
            'primary_tool': recommendations[0]['tool'] if recommendations else 'text_to_video',
            'recommendations': recommendations,
            'tool_details': {
                rec['tool']: {
                    'name': self.tools[rec['tool']]['name'],
                    'api': self.tools[rec['tool']]['api'],
                    'endpoint': self.tools[rec['tool']]['endpoint']
                }
                for rec in recommendations
            }
        }
    
    def _check_character_reference(self, shot_meta: Dict[str, Any]) -> bool:
        """Check if character reference images exist"""
        # In real implementation, check if character images exist
        # For now, assume they exist if characters are specified
        return len(shot_meta.get('characters', [])) > 0
    
    def route_shot(self, shot_dir: Path):
        """Route a single shot based on its SHOT_META.json"""
        meta_path = shot_dir / "SHOT_META.json"
        
        if not meta_path.exists():
            print(f"Warning: {meta_path} not found")
            return None
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            shot_meta = json.load(f)
        
        # Analyze and recommend tools
        routing = self.analyze_shot(shot_meta)
        
        # Update SHOT_META.json with routing info
        shot_meta['tool_routing'] = routing
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(shot_meta, f, ensure_ascii=False, indent=2)
        
        shot_id = shot_meta.get('id') or shot_meta.get('shot_id', 'unknown')
        print(f"Routed {shot_id}: {routing['primary_tool']}")
        return routing
    
    def route_all_shots(self, shots_dir: Path):
        """Route all shots in the directory"""
        shot_dirs = sorted([d for d in shots_dir.iterdir() if d.is_dir()])
        
        routings = []
        for shot_dir in shot_dirs:
            routing = self.route_shot(shot_dir)
            if routing:
                routings.append(routing)
        
        return routings


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tool_router.py <shots_directory>")
        sys.exit(1)
    
    shots_dir = Path(sys.argv[1])
    
    if not shots_dir.exists():
        print(f"Error: {shots_dir} does not exist")
        sys.exit(1)
    
    router = ToolRouter()
    routings = router.route_all_shots(shots_dir)
    
    print(f"\nRouted {len(routings)} shots:")
    for routing in routings:
        print(f"  {routing['shot_id']}: {routing['primary_tool']}")
