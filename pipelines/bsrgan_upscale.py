#!/usr/bin/env python3
# bsrgan_upscale.py
# This wrapper loads BSRGAN .pth and runs frame-by-frame inference.
# It uses a minimal loader expecting the model class from cszn/BSRGAN converted to PyTorch .pth format.
# If you prefer, replace loader with the official repo class and weights.

import os
import argparse
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

# NOTE: the BSRGAN architecture may not be pip-installable.
# We implement a small loader using the generic BSRGAN wrapper available in community packages.
# Try to import a helper package first.

try:
    # community package that exposes a simple wrapper
    from bsrgan import BSRGAN  # type: ignore
    HAVE_BSRGAN_PKG = True
except Exception:
    HAVE_BSRGAN_PKG = False

def load_model_pth(model_path, device):
    if HAVE_BSRGAN_PKG:
        model = BSRGAN(model_path).to(device).eval()
        return model
    # fallback: try loading a state dict and let user provide custom loader
    state = torch.load(model_path, map_location="cpu")
    # If fallback, we cannot construct network automatically here.
    raise RuntimeError("No BSRGAN loader found. Install 'bsrgan' package or use the official repo. pip install bsrgan")

def process_frame_with_model(model, frame_bgr, device):
    # frame_bgr: numpy uint8 BGR
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model(x)  # model should return tensor in range 0-1
    out = out.squeeze().permute(1,2,0).cpu().numpy()
    out = (out*255.0).clip(0,255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_bgr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="models/bsrgan/BSRGAN.pth")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not files:
        print("No frames found in", args.frames_dir); return

    device = args.device if args.device != "cpu" else "cpu"
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        else:
            device = "cpu"

    print("Loading BSRGAN model from", args.model)
    model = load_model_pth(args.model, device)

    pbar = tqdm(total=len(files), desc="BSRGAN")
    for f in files:
        frame = cv2.imread(f)
        out = process_frame_with_model(model, frame, device)
        cv2.imwrite(os.path.join(args.outdir, os.path.basename(f)), out)
        pbar.update(1)
    pbar.close()
    print("BSRGAN done ->", args.outdir)

if __name__ == "__main__":
    main()
