#!/usr/bin/env python3
"""
Test script to verify quality_gate import works in pipeline context
"""
import sys
from pathlib import Path

# Simulate pipeline_runner.py context
pipeline_src = Path(__file__).parent / "src"
sys.path.insert(0, str(pipeline_src))

print("Testing quality_gate import in pipeline context...")

try:
    # Test the import as it appears in pipeline_runner.py
    from quality.quality_gate import run_quality_check
    
    print("✅ Import successful")
    
    # Test that the function is callable
    assert callable(run_quality_check), "run_quality_check is not callable"
    print("✅ Function is callable")
    
    # Test with a mock directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_quality_check("phase3", tmpdir)
        print(f"✅ Function executes successfully (grade={result.grade})")
        
    print("\n✅ All import tests passed!")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
