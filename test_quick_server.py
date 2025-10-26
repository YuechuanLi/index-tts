#!/usr/bin/env python3
"""
Quick test to diagnose streaming server startup issues
This bypasses FastAPI and loads models directly to see where it's stuck
"""

import sys
import time
import torch

print("=" * 60)
print("IndexTTS Model Loading Diagnostic")
print("=" * 60)
print()

# Test 1: Basic imports
print("[1/8] Testing basic imports...")
start = time.time()
try:
    from indextts.infer_v2 import IndexTTS2
    print(f"✓ Imports OK ({time.time()-start:.1f}s)")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check GPU
print("\n[2/8] Checking GPU...")
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("✗ No GPU available")

# Test 3: Check model files
print("\n[3/8] Checking model files...")
import os
files_to_check = [
    "checkpoints/config.yaml",
    "checkpoints/gpt.pth",
    "checkpoints/s2mel.pth",
]
for f in files_to_check:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1e6
        print(f"✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {f} NOT FOUND")

# Test 4-8: Load models with detailed timing
print("\n[4/8] Loading IndexTTS2 (this will show detailed progress)...")
print("-" * 60)

start_total = time.time()
try:
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=True,
        use_cuda_kernel=True,
        use_deepspeed=False
    )
    elapsed = time.time() - start_total
    print("-" * 60)
    print(f"✓ All models loaded successfully in {elapsed:.1f}s")

    # Test inference
    print("\n[5/8] Testing basic inference...")
    print("  (Skipped - models loaded OK)")

except Exception as e:
    elapsed = time.time() - start_total
    print("-" * 60)
    print(f"\n✗ Model loading failed after {elapsed:.1f}s")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ DIAGNOSTIC COMPLETE - All models loaded successfully!")
print(f"   Total time: {time.time()-start_total:.1f}s")
print("=" * 60)
print()
print("The streaming server should work. If it's slow, this is normal")
print("on first run. Subsequent runs will be faster with cached models.")
