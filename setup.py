#!/usr/bin/env python3
"""
setup.py - Environment Bootstrap for GTA 6 Stream Shorts Clipper
Verifies FFmpeg, installs Python dependencies, and downloads the YuNet face detection model.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BIN_DIR = ROOT / "bin"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def run_cmd(cmd):
    return subprocess.run(cmd, check=True)

def check_python_packages():
    print("📦 Checking Python dependencies...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    print("   ✓ Python packages installed.")

def check_binaries():
    print("🔍 Checking system binaries...")
    ffmpeg_ok = shutil.which("ffmpeg") or (BIN_DIR / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")).exists()
    if not ffmpeg_ok:
        print("   ⚠️ ffmpeg was not found on your PATH or in bin/.")
        print("      Windows: winget install Gyan.FFmpeg or download from https://ffmpeg.org")
        print("      macOS  : brew install ffmpeg")
        print("      Linux  : sudo apt install ffmpeg")
    else:
        print("   ✓ ffmpeg detected.")

def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if target.exists() and target.stat().st_size > 100_000:
        print(f"   ✓ YuNet Facecam model already present ({target.stat().st_size / 1e6:.2f} MB).")
        return

    # Check local machine cache if available
    local_cache = Path("c:/AI OS/execution/models/face_detection_yunet_2023mar.onnx")
    if local_cache.exists():
        print("   ✓ Copying YuNet model from local cache...")
        shutil.copy(local_cache, target)
        return

    print("📥 Downloading YuNet face detection model...")
    try:
        urllib.request.urlretrieve(YUNET_URL, target)
        print("   ✓ YuNet model downloaded.")
    except Exception as e:
        print(f"   ⚠️ Could not automatically download model: {e}")
        print(f"      Download manually from: {YUNET_URL}")
        print(f"      and save to: {target}")

def main():
    print("=" * 60)
    print("🚀 GTA 6 Stream Shorts Clipper - Setup")
    print("=" * 60)
    check_python_packages()
    check_binaries()
    download_models()
    print("\n🎉 Setup complete! You are ready to clip.")
    print("   Run: python clipper.py --help\n")

if __name__ == "__main__":
    main()
