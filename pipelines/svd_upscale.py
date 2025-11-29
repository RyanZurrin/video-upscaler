#!/usr/bin/env python3
# svd_upscale.py
# Diffusion-based upscaling using diffusers img2img / img2vid pipeline.
# This is slow. Use low steps for testing.

import os
import argparse
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
import cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    p.add_argument("--steps", type=int, default=18)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device=="cuda" else torch.float32
    print("Loading model", args.model)
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    # memory helpers
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    for f in tqdm(files, desc="Diffusion SR"):
        im = Image.open(f).convert("RGB")
        # The exact call signature depends on the model. We try a simple .__call__ style
        out = pipe(image=np.array(im), num_inference_steps=args.steps).frames[0]
        out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.outdir, os.path.basename(f)), out_bgr)
    print("SVD done ->", args.outdir)

if __name__ == "__main__":
    main()
