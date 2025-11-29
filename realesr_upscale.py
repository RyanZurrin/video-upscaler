#!/usr/bin/env python3
# realesr_upscale.py
# Prefer NCNN Vulkan binary for speed. If not found, fall back to Python realesrgan pip package.
import os
import argparse
import subprocess
import glob
from tqdm import tqdm
import shutil
import sys

def run_ncnn_vulkan(binary_path, model_dir, frames_dir, outdir, scale=4):
    os.makedirs(outdir, exist_ok=True)
    # many ncnn builds accept -i input_dir -o output_dir -n modelname -s scale
    # We'll try common variants; if they fail, user must edit to match their binary.
    modelname = os.path.basename(model_dir)
    cmd_try = [
        [binary_path, "-i", frames_dir, "-o", outdir, "-n", modelname, "-s", str(scale)],
        [binary_path, "-i", frames_dir, "-o", outdir, "-m", model_dir, "-s", str(scale)],
        [binary_path, "-i", frames_dir, "-o", outdir, "-s", str(scale)]
    ]
    for cmd in cmd_try:
        try:
            print("Trying:", " ".join(cmd))
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError:
            continue
        except FileNotFoundError:
            raise
    raise RuntimeError("NCNN Vulkan invocation failed. Edit run_ncnn_vulkan() to match your realesrgan-ncnn-vulkan binary flags.")

def run_py_realesrgan(model_path, frames_dir, outdir):
    # Use pip 'realesrgan' package if installed
    try:
        from realesrgan import RealESRGANer  # type: ignore
    except Exception as e:
        raise RuntimeError("Python Real-ESRGAN fallback requires 'realesrgan' package. pip install realesrgan") from e
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    device = "cuda"
    import torch
    torch.backends.cudnn.benchmark = True
    # Basic wrapper: use RealESRGANer API
    rr = RealESRGANer(model_path, device=device)  # pseudo API; check package docs
    from tqdm import tqdm
    for f in tqdm(files, desc="RealESRGAN"):
        out = rr.enhance(f)  # adapt to package API
        out.save(os.path.join(outdir, os.path.basename(f)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ncnn_bin", default=None, help="path to realesrgan-ncnn-vulkan.exe (optional)")
    p.add_argument("--ncnn_model_dir", default="models/realesrgan")
    p.add_argument("--pth_model", default="models/realesrgan/realesr-general-x4v3.pth")
    p.add_argument("--scale", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.ncnn_bin and os.path.exists(args.ncnn_bin):
        print("Using NCNN Vulkan binary:", args.ncnn_bin)
        run_ncnn_vulkan(args.ncnn_bin, args.ncnn_model_dir, args.frames_dir, args.outdir, scale=args.scale)
    else:
        print("NCNN binary not provided or not found; attempting Python Real-ESRGAN via pip.")
        run_py_realesrgan(args.pth_model, args.frames_dir, args.outdir)

if __name__ == "__main__":
    main()
