"""Pipeline phase entry points."""

from importlib import import_module

from .phase1_director import run_phase1
from .phase2_screenwriter import run_phase2
run_phase2_storyboard = import_module(".phase2" + "_5_storyboard", __name__).run_phase2_storyboard
from .phase3_character import run_phase3
from .phase4_orchestrator import run_phase4
from .storyboard_qa_gate import run_storyboard_qa_gate
from .phase5_video_gen import run_phase5
from .phase6_consistency import run_phase6
from .phase7_assembly import run_phase7
from .phase8_post import run_phase8
