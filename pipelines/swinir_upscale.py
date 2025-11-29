#!/usr/bin/env python3
# swinir_upscale.py
# Requires: you have SwinIR network code available as models/swinir_network.py or pip package.

import os
import argparse
from tqdm import tqdm
import torch
import glob
import cv2
import numpy as np
from PIL import Image

# Attempt to import a packaged SwinIR or local loader
try:
    from swinir import SwinIR  # community pip wrapper
    HAVE_SWIN_PKG = True
except Exception:
    HAVE_SWIN_PKG = False

def load_model(model_path, device="cuda"):
    if HAVE_SWIN_PKG:
        model = SwinIR(model_path).to(device).eval()
        return model
    # else try to import local implementation
    import importlib.util
    spec = importlib.util.spec_from_file_location("swinir_network", os.path.join("models", "swinir_network.py"))
    if spec is None:
        raise RuntimeError("SwinIR loader not found. Put models/swinir_network.py from SwinIR repo or pip install swinir.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SwinIRClass = getattr(mod, "SwinIR")
    model = SwinIRClass(upscale=4, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6,6,6,6,6,6], embed_dim=180,
                   num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler="pixelshuffle",
                   resi_connection="3conv")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model

def process_batch(model, imgs, device="cuda"):
    xb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 for im in imgs])
    xb = torch.from_numpy(xb).permute(0,3,1,2).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model(xb)
    out_np = out.permute(0,2,3,1).cpu().numpy()
    outs = []
    for o in out_np:
        o = (o*255.0).clip(0,255).astype(np.uint8)
        o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        outs.append(o)
    return outs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="models/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not files:
        print("No frames found"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device=device)
    i = 0
    pbar = tqdm(total=len(files), desc="SwinIR")
    while i < len(files):
        batch_files = files[i:i+args.batch]
        imgs = [cv2.imread(f) for f in batch_files]
        outs = process_batch(model, imgs, device=device)
        for fpath, out in zip(batch_files, outs):
            cv2.imwrite(os.path.join(args.outdir, os.path.basename(fpath)), out)
        i += args.batch
        pbar.update(len(batch_files))
    pbar.close()
    print("SwinIR done ->", args.outdir)

if __name__ == "__main__":
    main()
