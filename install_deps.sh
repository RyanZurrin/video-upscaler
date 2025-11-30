#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
# adjust torch index for your CUDA version; below is for CUDA 12.4 (RTX 2060 supports many)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python-headless diffusers transformers accelerate safetensors tqdm ffmpeg-python pillow numpy scikit-image pyqt6
echo "Done. Make sure ffmpeg is on PATH and ncnn-realesrgan binary is available if using NCNN."
