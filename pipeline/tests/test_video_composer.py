import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from video_composer import lock_runtime, route_composition
def test_slide_grammar_routes_remotion(): assert route_composition({"cuts":[{"type":"chart"}]}).runtime == "remotion"
def test_html_routes_hyperframes(): assert route_composition({"html_entry":"index.html"}).runtime == "hyperframes"
def test_runtime_lock_cannot_silently_fallback():
    with pytest.raises(RuntimeError): lock_runtime({},available={"ffmpeg"},locked_runtime="remotion")
